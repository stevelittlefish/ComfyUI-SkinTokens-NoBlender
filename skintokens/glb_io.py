"""Pure-Python glb/gltf import (replaces upstream's Blender-based load).

Phase 2 scope (see spec/02, spec/TODO): import a mesh from a glb/gltf/obj file
into the same ``Asset`` mesh fields that ``BpyParser.load`` produced, using
``trimesh`` instead of ``bpy``. Skinned *export* (the hard part) is Phase 3;
armature *import* (``use_skeleton``) is Phase 6.

Fields reproduced (mesh-only; no armature):
  vertices, faces, vertex_normals, face_normals, vertex_bias, face_bias,
  mesh_names. Normals are computed by trimesh from the merged mesh — matching
  upstream, which also recomputed them rather than reading them from the file.

AXIS CONVENTION (deferred to Phase 3): Blender's glTF importer converts glTF's
Y-up to Blender's Z-up, so upstream Assets are Z-up. trimesh keeps the file's
native glTF Y-up. We load in the file's native space here and DO NOT rotate;
the up-axis constant is calibrated in Phase 3 (spec/03) so import and export
agree. If a mesh rigs "sideways", this is the first suspect.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import trimesh

from .vendor.rig_package.info.asset import Asset

PathLike = Union[str, Path]
SUPPORTED_EXT = {".glb", ".gltf", ".obj", ".ply", ".stl"}

# glTF component types
_FLOAT = 5126
_UNSIGNED_SHORT = 5123
_UNSIGNED_INT = 5125
# glTF bufferView targets
_ARRAY_BUFFER = 34962
_ELEMENT_ARRAY_BUFFER = 34963

GROUP_PER_VERTEX = 4  # glTF max influences per vertex (upstream group_per_vertex=4)


@dataclass
class LoadedMesh:
    """Merged mesh arrays, matching the fields BpyParser.load produced."""

    vertices: np.ndarray        # (N, 3) float32
    faces: np.ndarray           # (F, 3) int64
    vertex_normals: np.ndarray  # (N, 3) float32
    face_normals: np.ndarray    # (F, 3) float32
    vertex_bias: np.ndarray     # (P,) cumulative vertex counts per part
    face_bias: np.ndarray       # (P,) cumulative face counts per part
    mesh_names: List[str]       # (P,) part names, in load order

    @property
    def num_parts(self) -> int:
        return len(self.mesh_names)


def _iter_parts(loaded) -> List[tuple]:
    """Yield (name, Trimesh) parts in a stable order, with transforms applied.

    A glb with several meshes loads as a ``trimesh.Scene``; a single mesh loads
    as a ``trimesh.Trimesh``. For a scene we walk the graph so each geometry is
    placed by its node transform (as Blender would), and concatenate in graph
    order — reproducing upstream's "iterate mesh objects and stack" behavior.
    """
    if isinstance(loaded, trimesh.Trimesh):
        return [("mesh", loaded)]

    if isinstance(loaded, trimesh.Scene):
        parts = []
        for node_name in loaded.graph.nodes_geometry:
            transform, geom_name = loaded.graph[node_name]
            geom = loaded.geometry[geom_name]
            if not isinstance(geom, trimesh.Trimesh):
                continue  # skip non-mesh geometry (points, paths)
            part = geom.copy()
            part.apply_transform(transform)
            parts.append((geom_name or node_name, part))
        if not parts:
            raise ValueError("no mesh geometry found in scene")
        return parts

    raise TypeError(f"unsupported trimesh load result: {type(loaded)}")


def load_mesh(path: PathLike) -> LoadedMesh:
    """Load a mesh file into merged arrays matching upstream's mesh fields.

    Concatenates all mesh parts (offsetting faces by the running vertex count)
    and recomputes normals from the merged mesh with trimesh — as upstream did.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"file does not exist: {path}")
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXT:
        raise ValueError(f"unsupported type: {ext} (supported: {sorted(SUPPORTED_EXT)})")

    # process=False / maintain_order=True: do not merge/reorder vertices, so the
    # arrays stay aligned with the file (upstream used the same trimesh flags).
    loaded = trimesh.load(path, process=False, maintain_order=True)
    parts = _iter_parts(loaded)

    vertices_list, faces_list, mesh_names, vertex_bias, face_bias = [], [], [], [], []
    cur_v, cur_f = 0, 0
    for name, part in parts:
        v = np.asarray(part.vertices, dtype=np.float32)
        f = np.asarray(part.faces, dtype=np.int64)
        vertices_list.append(v)
        faces_list.append(f + cur_v)  # offset faces into the merged vertex array
        mesh_names.append(str(name))
        cur_v += v.shape[0]
        cur_f += f.shape[0]
        vertex_bias.append(cur_v)
        face_bias.append(cur_f)

    vertices = np.vstack(vertices_list).astype(np.float32)
    faces = np.vstack(faces_list).astype(np.int64)

    merged = trimesh.Trimesh(vertices=vertices, faces=faces, process=False, maintain_order=True)
    vertex_normals = np.asarray(merged.vertex_normals, dtype=np.float32)
    face_normals = np.asarray(merged.face_normals, dtype=np.float32)

    return LoadedMesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=vertex_normals,
        face_normals=face_normals,
        vertex_bias=np.array(vertex_bias, dtype=np.int64),
        face_bias=np.array(face_bias, dtype=np.int64),
        mesh_names=mesh_names,
    )


def load_asset(path: PathLike, cls: str = "articulation") -> Asset:
    """Load a mesh file into an unrigged ``Asset`` ready for inference.

    Populates the same mesh fields ``BpyParser.load`` produced, so the downstream
    transform + model code is unchanged. ``skin``/armature fields are left unset
    (armature import is Phase 6).
    """
    m = load_mesh(path)
    return Asset(
        vertices=m.vertices,
        faces=m.faces,
        vertex_normals=m.vertex_normals,
        face_normals=m.face_normals,
        vertex_bias=m.vertex_bias,
        face_bias=m.face_bias,
        mesh_names=m.mesh_names,
        cls=cls,
        path=str(path),
    )


# ---------------------------------------------------------------------------
# Export (Phase 3) — the critical correctness path. See spec/03 and the
# authoritative reference skin-tokens.cpp/src/glb.cpp (save_skinned_animation_glb_file)
# and binding.cpp (top-4 packing). Convention: joints are plain points with
# IDENTITY rotation; node local translation = joint - parent_joint; the inverse
# bind matrix is a pure translate(-joint_world). Thus world @ IBM == identity at
# bind, and the mesh deforms correctly when a bone is posed.
# ---------------------------------------------------------------------------


def pack_top4(dense_skin: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce dense per-vertex weights (N, J) to top-4 influences per vertex.

    Matches binding.cpp: for each vertex, keep the 4 largest weights (tie-break
    by lower joint index), pad with joint 0 / weight 0 if fewer are non-zero,
    force weight[0]=1 if all zero, then normalize the 4 to sum to 1.0.

    Returns ``(joints, weights)`` with shapes (N, 4): joints uint16, weights float32.
    """
    dense = np.asarray(dense_skin, dtype=np.float64)
    if dense.ndim != 2:
        raise ValueError(f"dense_skin must be (N, J), got {dense.shape}")
    n, j = dense.shape
    k = min(GROUP_PER_VERTEX, j)

    # Rank by (-weight, joint_index) so ties prefer the lower joint index.
    # argsort is ascending; sort on weight descending via negation, stable so the
    # existing (ascending) joint order breaks ties toward lower indices.
    order = np.argsort(-dense, axis=1, kind="stable")[:, :k]  # (N, k)
    top_j = order.astype(np.int64)
    top_w = np.take_along_axis(dense, order, axis=1)  # (N, k)
    top_w = np.clip(top_w, 0.0, None)

    joints = np.zeros((n, GROUP_PER_VERTEX), dtype=np.uint16)
    weights = np.zeros((n, GROUP_PER_VERTEX), dtype=np.float64)
    joints[:, :k] = top_j.astype(np.uint16)
    weights[:, :k] = top_w

    sums = weights.sum(axis=1)
    empty = sums <= 1e-12
    # Vertices with no positive influence: assign fully to joint 0.
    joints[empty, 0] = 0
    weights[empty, :] = 0.0
    weights[empty, 0] = 1.0
    sums = weights.sum(axis=1)
    weights = weights / sums[:, None]

    return joints, weights.astype(np.float32)


def _accessor(
    gltf, blob: bytearray, array: np.ndarray, component_type: int, type_str: str,
    target: Optional[int] = None, with_minmax: bool = False,
):
    """Append ``array`` to the binary blob and register a bufferView + accessor.

    Returns the new accessor index. ``array`` must already be the right dtype
    (float32 / uint16 / uint32) and shape ((count,) or (count, comps)).
    """
    from pygltflib import Accessor, BufferView

    data = np.ascontiguousarray(array).tobytes()
    byte_offset = len(blob)
    blob.extend(data)
    while len(blob) % 4 != 0:  # 4-byte align the next view
        blob.append(0)

    bv = BufferView(buffer=0, byteOffset=byte_offset, byteLength=len(data))
    if target is not None:
        bv.target = target
    gltf.bufferViews.append(bv)
    bv_index = len(gltf.bufferViews) - 1

    count = array.shape[0]
    acc = Accessor(
        bufferView=bv_index,
        componentType=component_type,
        count=count,
        type=type_str,
    )
    if with_minmax:
        acc.min = array.min(axis=0).tolist()
        acc.max = array.max(axis=0).tolist()
    gltf.accessors.append(acc)
    return len(gltf.accessors) - 1


def _skeleton_from_gltf(g) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Extract (joints_world, parents, joint_node_indices) from a skinned glTF.

    ``joints_world`` is (J, 3) rest-pose world positions of each joint (in glTF
    joint order), ``parents`` is (J,) with -1 for the skeleton root, and
    ``joint_node_indices`` maps joint index -> glTF node index (for renaming).
    Uses the first skin. Node transforms are composed down the hierarchy.
    """
    if not g.skins:
        raise ValueError("glTF has no skin (not a rigged mesh)")
    joint_nodes = list(g.skins[0].joints)

    parent_of = {}
    for i, node in enumerate(g.nodes):
        for c in (node.children or []):
            parent_of[c] = i

    def local_matrix(node) -> np.ndarray:
        if node.matrix:
            # glTF matrices are column-major; reshape+T to row-major.
            return np.array(node.matrix, dtype=np.float64).reshape(4, 4).T
        m = np.eye(4, dtype=np.float64)
        if node.rotation:
            x, y, z, w = node.rotation
            m[:3, :3] = [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        m[:3, 3] = node.translation or [0.0, 0.0, 0.0]
        return m

    _cache = {}

    def world_matrix(i: int) -> np.ndarray:
        if i in _cache:
            return _cache[i]
        m = local_matrix(g.nodes[i])
        if i in parent_of:
            m = world_matrix(parent_of[i]) @ m
        _cache[i] = m
        return m

    idx = {n: k for k, n in enumerate(joint_nodes)}
    J = len(joint_nodes)
    joints = np.zeros((J, 3), dtype=np.float64)
    parents = np.full(J, -1, dtype=np.int64)
    for k, n in enumerate(joint_nodes):
        joints[k] = world_matrix(n)[:3, 3]
        pn = parent_of.get(n)
        parents[k] = idx.get(pn, -1) if pn is not None else -1
    return joints, parents, joint_nodes


def relabel_glb(
    in_path: PathLike,
    out_path: PathLike,
    convention: str = "mixamo",
    with_fingers: bool = True,
) -> dict:
    """Rename the humanoid joint nodes of a rigged glb in place; write to out_path.

    Loads the skinned glb, recognizes the humanoid core from its skeleton
    (:func:`relabel.label_humanoid`), renames only the matched joint *node names*
    (indices/geometry/skin untouched — JOINTS_0 references joints by index, so the
    skin stays attached), and saves. Returns ``{node_index: new_name}``.
    """
    from pygltflib import GLTF2

    from .relabel import label_humanoid

    g = GLTF2().load_binary(str(in_path))
    joints, parents, joint_nodes = _skeleton_from_gltf(g)
    mapping = label_humanoid(joints, parents, convention=convention, with_fingers=with_fingers)

    applied = {}
    for joint_index, name in mapping.items():
        node_index = joint_nodes[joint_index]
        g.nodes[node_index].name = name
        applied[node_index] = name

    g.save_binary(str(out_path))
    return applied


def import_skeleton(path: PathLike) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Read an existing armature from a rigged glb: (joints, parents, joint_names).

    ``joints`` is (J, 3) rest-pose world joint positions, ``parents`` (J,) with -1
    for the root, ``joint_names`` the glTF joint node names (falling back to
    ``bone_{i}``). Used by the ``use_skeleton`` skin-only path (Phase 6).
    """
    from pygltflib import GLTF2

    g = GLTF2().load_binary(str(path))
    joints, parents, joint_nodes = _skeleton_from_gltf(g)
    names = [g.nodes[n].name or f"bone_{k}" for k, n in enumerate(joint_nodes)]
    return joints.astype(np.float32), parents.astype(np.int64), names


def load_asset_with_skeleton(path: PathLike, cls: str = "articulation") -> Asset:
    """Load a rigged glb into an ``Asset`` carrying both mesh and armature.

    Combines :func:`load_asset` (mesh fields) with :func:`import_skeleton`
    (joints/parents/joint_names), setting ``matrix_local`` from the joints so the
    tokenizer can encode the existing skeleton for the skin-only path.
    """
    asset = load_asset(path, cls=cls)
    joints, parents, names = import_skeleton(path)
    J = joints.shape[0]
    matrix_local = np.tile(np.eye(4, dtype=np.float32), (J, 1, 1))
    matrix_local[:, :3, 3] = joints
    asset.matrix_local = matrix_local
    asset.parents = parents
    asset.joint_names = names
    return asset


def export_glb(asset: Asset, path: PathLike) -> None:
    """Export a rigged ``Asset`` to a skinned glb (one-frame rest pose).

    Requires ``asset`` to have vertices, faces, joints, parents, and dense
    ``skin`` (N, J). Vertex normals are computed if absent. Follows glb.cpp:
    translation-only joint nodes, translate(-joint) inverse bind matrices, top-4
    JOINTS_0/WEIGHTS_0. The file opens in any glTF viewer as a static skinned
    mesh and deforms correctly when bones are posed.
    """
    from pygltflib import (
        GLTF2, Attributes, Buffer, Mesh, Node, Primitive, Scene, Skin,
    )

    if asset.vertices is None or asset.faces is None:
        raise ValueError("asset needs vertices and faces")
    if asset.joints is None or asset.parents is None:
        raise ValueError("asset needs joints and parents (rig it first)")
    if asset.skin is None:
        raise ValueError("asset needs dense skin weights")

    vertices = np.asarray(asset.vertices, dtype=np.float32)
    faces = np.asarray(asset.faces, dtype=np.uint32)
    joints_xyz = np.asarray(asset.joints, dtype=np.float32)
    parents = np.asarray(asset.parents, dtype=np.int64)
    J = joints_xyz.shape[0]

    if asset.vertex_normals is not None:
        normals = np.asarray(asset.vertex_normals, dtype=np.float32)
    else:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False, maintain_order=True)
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32)

    joint_names = asset.joint_names or [f"bone_{i}" for i in range(J)]

    joints_0, weights_0 = pack_top4(asset.skin)
    if joints_0.shape[0] != vertices.shape[0]:
        raise ValueError(
            f"skin rows ({joints_0.shape[0]}) != vertex count ({vertices.shape[0]})"
        )

    # Inverse bind matrices: pure translate(-joint_world), column-major flat 16.
    ibm = np.tile(np.eye(4, dtype=np.float32).flatten(), (J, 1))  # (J, 16)
    ibm[:, 12] = -joints_xyz[:, 0]
    ibm[:, 13] = -joints_xyz[:, 1]
    ibm[:, 14] = -joints_xyz[:, 2]

    gltf = GLTF2()
    gltf.asset.generator = "ComfyUI-SkinTokens-NoBlender"
    blob = bytearray()

    a_pos = _accessor(gltf, blob, vertices, _FLOAT, "VEC3", _ARRAY_BUFFER, with_minmax=True)
    a_nrm = _accessor(gltf, blob, normals, _FLOAT, "VEC3", _ARRAY_BUFFER)
    a_jnt = _accessor(gltf, blob, joints_0, _UNSIGNED_SHORT, "VEC4", _ARRAY_BUFFER)
    a_wgt = _accessor(gltf, blob, weights_0, _FLOAT, "VEC4", _ARRAY_BUFFER)
    a_idx = _accessor(gltf, blob, faces.reshape(-1), _UNSIGNED_INT, "SCALAR", _ELEMENT_ARRAY_BUFFER)
    a_ibm = _accessor(gltf, blob, ibm, _FLOAT, "MAT4")

    # Joint nodes: local translation = joint - parent_joint (absolute for root).
    for i in range(J):
        p = int(parents[i])
        local = joints_xyz[i] - joints_xyz[p] if p >= 0 else joints_xyz[i]
        children = [c for c in range(J) if int(parents[c]) == i]
        node = Node(name=str(joint_names[i]), translation=local.tolist())
        if children:
            node.children = children
        gltf.nodes.append(node)

    # Mesh node (references the skin + mesh).
    mesh_node_index = len(gltf.nodes)
    gltf.nodes.append(Node(name="SkinnedMesh", mesh=0, skin=0))

    gltf.meshes.append(Mesh(primitives=[Primitive(
        attributes=Attributes(POSITION=a_pos, NORMAL=a_nrm, JOINTS_0=a_jnt, WEIGHTS_0=a_wgt),
        indices=a_idx,
        mode=4,  # TRIANGLES
    )]))

    root_joint = next((i for i in range(J) if int(parents[i]) == -1), 0)
    gltf.skins.append(Skin(
        inverseBindMatrices=a_ibm,
        skeleton=root_joint,
        joints=list(range(J)),
    ))

    gltf.scenes.append(Scene(nodes=[root_joint, mesh_node_index]))
    gltf.scene = 0

    while len(blob) % 4 != 0:
        blob.append(0)
    gltf.buffers.append(Buffer(byteLength=len(blob)))
    gltf.set_binary_blob(bytes(blob))

    gltf.save_binary(str(path))
