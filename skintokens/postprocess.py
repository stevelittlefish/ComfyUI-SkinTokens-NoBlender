"""Optional voxel-skin post-process (Phase 6, ``use_postprocess``).

Upstream ``demo.py`` offers a heuristic that refines the predicted skin with a
geodesic voxel-heat solve (``src/data/vertex_group.voxel_skin``, after
``asset.voxel(...)``): it multiplies the model's skin by a voxel-graph weight and
renormalizes, sharpening weights around limb boundaries. It is **opt-in / default
off** and purely a refinement.

Two upstream dependencies are replaced to keep the pack Blender/open3d-free:
  * ``voxel_skin`` — copied verbatim (already pure numpy/scipy).
  * ``asset.voxel`` — upstream voxelizes with open3d; we voxelize the surface with
    a pure-numpy sampler (``_voxelize_surface``) that returns world-space voxel
    centres. voxel_skin is then called with those centres + the voxel size (the
    meaningful world-space interpretation of the upstream grid).

Speed is not a priority (see spec); correctness/ecosystem fit are.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import cKDTree

from .vendor.rig_package.info.asset import Asset


def _voxelize_surface(
    vertices: np.ndarray, faces: np.ndarray, resolution: int = 196,
    voxel_size: Optional[float] = None, samples_per_face: int = 4,
) -> Tuple[np.ndarray, float]:
    """Voxelize the mesh *surface* to world-space voxel centres (open3d-free).

    Densely samples each triangle (barycentric), snaps samples to a regular grid
    of side ``voxel_size`` and returns the unique occupied cells' centres. Matches
    ``Asset.voxel`` in spirit (surface occupancy, not solid fill), which is all
    ``voxel_skin`` uses the grid for.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if voxel_size is None:
        extent = (vertices.max(axis=0) - vertices.min(axis=0)).max()
        voxel_size = float(extent) / resolution
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    # Barycentric samples per triangle (include the 3 corners for coverage).
    rng = np.random.default_rng(0)
    pts = [v0, v1, v2]
    for _ in range(samples_per_face):
        u = rng.random((faces.shape[0], 1))
        w = rng.random((faces.shape[0], 1))
        over = (u + w) > 1.0
        u[over] = 1.0 - u[over]
        w[over] = 1.0 - w[over]
        pts.append(v0 + u * (v1 - v0) + w * (v2 - v0))
    sample = np.vstack(pts)

    origin = vertices.min(axis=0)
    grid_idx = np.floor((sample - origin) / voxel_size).astype(np.int64)
    grid_idx = np.unique(grid_idx, axis=0)
    centres = origin + (grid_idx + 0.5) * voxel_size
    return centres.astype(np.float64), float(voxel_size)


def voxel_skin(
    grid: int,
    grid_coords: np.ndarray,  # (M, 3)
    joints: np.ndarray,       # (J, 3)
    vertices: np.ndarray,     # (N, 3)
    faces: np.ndarray,        # (F, 3)
    alpha: float = 0.5,
    link_dis: float = 0.00001,
    grid_query: int = 27,
    vertex_query: int = 27,
    grid_weight: float = 3.0,
    voxel_size: Optional[float] = None,
    mode: str = "square",
    parents: Optional[np.ndarray] = None,
):
    """Geodesic voxel-heat skin weights. Copied verbatim from upstream
    ``src/data/vertex_group.voxel_skin`` (pure numpy/scipy). Returns (N, J)."""
    # modified from https://dl.acm.org/doi/pdf/10.1145/2485895.2485919
    assert mode in ["square", "exp"]
    J = joints.shape[0]
    M = grid_coords.shape[0]
    N = vertices.shape[0]

    if voxel_size is None:
        _range = 2 / grid * 1.74
    else:
        _range = voxel_size * 1.74

    grid_tree = cKDTree(grid_coords)
    vertex_tree = cKDTree(vertices)
    if parents is not None:
        son = defaultdict(list)
        for i, p in enumerate(parents):
            if i == -1:
                continue
            son[p].append(i)
        divide_joints = []
        joints_map = []
        for u in range(len(parents)):
            if len(son[u]) != 1:
                divide_joints.append(joints[u])
                joints_map.append(u)
            else:
                pu = joints[u]
                pv = joints[son[u][0]]
                seg = 10
                for i in range(seg + 1):
                    p = (pu * i + pv * (seg - i)) / seg
                    divide_joints.append(p)
                    joints_map.append(u)
        divide_joints = np.stack(divide_joints)
        joints_map = np.array(joints_map)
    else:
        divide_joints = joints
        joints_map = np.arange(joints.shape[0])
    joint_tree = cKDTree(divide_joints)

    combined_vertices = np.concatenate([vertices, grid_coords], axis=0)

    dist, idx = grid_tree.query(grid_coords, min(grid_query, M))
    dist = dist[:, 1:]
    idx = idx[:, 1:]
    mask = (0 < dist) & (dist < _range)
    source_grid2grid = np.repeat(np.arange(M), idx.shape[1])[mask.ravel()] + N
    to_grid2grid = idx[mask] + N
    weight_grid2grid = dist[mask] * grid_weight

    dist, idx = vertex_tree.query(vertices, min(4, N))
    dist = dist[:, 1:]
    idx = idx[:, 1:]
    mask = (0 < dist) & (dist < link_dis)
    source_close = np.repeat(np.arange(N), idx.shape[1])[mask.ravel()]
    to_close = idx[mask]
    weight_close = dist[mask]

    dist, idx = vertex_tree.query(grid_coords, min(vertex_query, N))
    mask = (0 < dist) & (dist < _range)
    source_grid2vertex = np.repeat(np.arange(M), idx.shape[1])[mask.ravel()] + N
    to_grid2vertex = idx[mask]
    weight_grid2vertex = dist[mask]

    combined_tree = cKDTree(combined_vertices)
    _, joint_indices = combined_tree.query(divide_joints)

    source_vertex2vertex = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]], axis=0)
    to_vertex2vertex = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]], axis=0)
    weight_vertex2vertex = np.sqrt(
        ((vertices[source_vertex2vertex] - vertices[to_vertex2vertex]) ** 2).sum(axis=-1)
    )
    graph = csr_matrix(
        (
            np.concatenate([weight_close, weight_vertex2vertex, weight_grid2grid, weight_grid2vertex]),
            (
                np.concatenate([source_close, source_vertex2vertex, source_grid2grid, source_grid2vertex], axis=0),
                np.concatenate([to_close, to_vertex2vertex, to_grid2grid, to_grid2vertex], axis=0),
            ),
        ),
        shape=(N + M, N + M),
    )

    dist_matrix = shortest_path(graph, method="D", directed=False, indices=joint_indices)

    dis_vertex2bone = dist_matrix[:, :N]
    unreachable = np.isinf(dis_vertex2bone).all(axis=0)
    k = min(J, 3)
    dist, idx = joint_tree.query(vertices[unreachable], k)
    if k == 1:
        idx = idx.reshape(-1, 1)
        dist = dist.reshape(-1, 1)

    unreachable_indices = np.where(unreachable)[0]
    row_indices = idx
    col_indices = np.repeat(unreachable_indices, k).reshape(-1, k)
    dis_vertex2bone[row_indices, col_indices] = dist

    finite_vals = dis_vertex2bone[np.isfinite(dis_vertex2bone)]
    max_dis = np.max(finite_vals)
    dis_vertex2bone = np.nan_to_num(dis_vertex2bone, nan=max_dis, posinf=max_dis, neginf=max_dis)
    dis_vertex2bone = np.maximum(dis_vertex2bone, 1e-6)

    dis_vertex2joint = np.full((joints.shape[0], vertices.shape[0]), max_dis)
    for i in range(len(dis_vertex2bone)):
        dis_vertex2joint[joints_map[i]] = np.minimum(dis_vertex2bone[i], dis_vertex2joint[joints_map[i]])

    if mode == "exp":
        skin = np.exp(-dis_vertex2joint / max_dis * 20.0)
    elif mode == "square":
        skin = (1.0 / ((1 - alpha) * dis_vertex2joint + alpha * dis_vertex2joint ** 2)) ** 2
    else:
        assert False, f"invalid mode: {mode}"
    skin = skin / skin.sum(axis=0)
    skin = skin.transpose()
    return skin


def apply_voxel_postprocess(asset: Asset, resolution: int = 196, mode: str = "square") -> Asset:
    """Refine ``asset.skin`` in place with the voxel-heat weights (upstream demo).

    ``asset.skin`` (N, J) is multiplied by the voxel_skin weights and renormalized.
    Mirrors ``demo.py``'s ``use_postprocess`` branch. Returns the same asset.
    """
    if asset.vertices is None or asset.faces is None:
        raise ValueError("asset needs vertices and faces for voxel post-process")
    if asset.joints is None or asset.skin is None:
        raise ValueError("asset needs joints and dense skin (rig it first)")

    centres, vsize = _voxelize_surface(asset.vertices, asset.faces, resolution=resolution)
    weights = voxel_skin(
        grid=0,
        grid_coords=centres,
        joints=np.asarray(asset.joints, dtype=np.float64),
        vertices=np.asarray(asset.vertices, dtype=np.float64),
        faces=np.asarray(asset.faces, dtype=np.int64),
        mode=mode,
        voxel_size=vsize,
    )
    asset.skin = asset.skin * weights
    asset.normalize_skin()
    return asset
