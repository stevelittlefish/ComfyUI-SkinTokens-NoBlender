"""Texture-preserving rig transfer (Phase 6, ``use_transfer``).

The model rigs a *normalized, surface-sampled* proxy of the input mesh. When the
user wants to keep the ORIGINAL glb's materials/textures/UVs/scale, we transfer
the generated skeleton + skin onto that original mesh instead of re-exporting the
proxy with a flat default material.

Port of upstream ``rig_package/parser/bpy.transfer_rigging`` (which used Blender)
to pure Python:

  1. Estimate a similarity transform ``T`` aligning the source (rigged proxy)
     vertices to the target (original) vertices — Umeyama when the point counts
     match, otherwise a PCA-axis alignment (``estimate_similarity_transform``).
  2. Map the source joints through ``T`` into target space.
  3. Assign each target vertex the skin weights of its nearest source vertex
     (nearest-neighbour in the aligned source cloud).
  4. Write the skeleton + skin into the *original* glb, preserving its materials,
     textures, UVs and node graph.

Rather than rebuild the glTF (which would lose PBR channels), we edit the target
glTF in place: bake each skinned mesh into world space, reparent it under an
identity scene root, and *append* JOINTS_0/WEIGHTS_0 + joint nodes + a skin.
Everything the target already had (materials/images/samplers/TEXCOORDs) is left
untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from .glb_io import _FLOAT, _UNSIGNED_INT, _UNSIGNED_SHORT, _accessor, pack_top4
from .vendor.rig_package.info.asset import Asset

PathLike = Union[str, Path]

# glTF componentType -> (numpy dtype, component count is per-accessor-type).
_COMPONENT_DTYPE = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
_TYPE_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


# ---------------------------------------------------------------------------
# Similarity transform (ports of the upstream numpy helpers in bpy.py)
# ---------------------------------------------------------------------------


def _umeyama_similarity(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Least-squares similarity (scale*R + t) for point sets of equal size."""
    assert src.shape == tgt.shape
    n = src.shape[0]
    src_mean = src.mean(axis=0)
    tgt_mean = tgt.mean(axis=0)
    src_c = src - src_mean
    tgt_c = tgt - tgt_mean

    C = (src_c.T @ tgt_c) / n
    U, S, Vt = np.linalg.svd(C)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    var_src = (src_c ** 2).sum() / n
    scale = S.sum() / var_src
    t = tgt_mean - scale * R @ src_mean
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = t
    return T


def _pca_similarity(src: np.ndarray, tgt: np.ndarray, max_points: int = 4096) -> np.ndarray:
    """Align principal axes when the point counts differ (no correspondence)."""
    if src.shape[0] > max_points:
        src = src[np.random.choice(src.shape[0], max_points, replace=False)]
    if tgt.shape[0] > max_points:
        tgt = tgt[np.random.choice(tgt.shape[0], max_points, replace=False)]
    src_mean = src.mean(axis=0)
    tgt_mean = tgt.mean(axis=0)
    src_c = src - src_mean
    tgt_c = tgt - tgt_mean
    U_src, _, _ = np.linalg.svd(src_c.T @ src_c)
    U_tgt, _, _ = np.linalg.svd(tgt_c.T @ tgt_c)
    R = U_tgt @ U_src.T
    if np.linalg.det(R) < 0:
        U_tgt[:, -1] *= -1
        R = U_tgt @ U_src.T
    scale = np.sqrt((tgt_c ** 2).sum() / (src_c ** 2).sum())
    t = tgt_mean - scale * R @ src_mean
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = t
    return T


def estimate_similarity_transform(
    src: np.ndarray, tgt: np.ndarray, max_points: int = 4096
) -> np.ndarray:
    """(4, 4) similarity transform mapping ``src`` points onto ``tgt``.

    Umeyama (exact correspondence) when the counts match, else PCA-axis alignment.
    """
    src = np.asarray(src, dtype=np.float64)
    tgt = np.asarray(tgt, dtype=np.float64)
    if src.shape[0] == tgt.shape[0]:
        return _umeyama_similarity(src, tgt)
    return _pca_similarity(src, tgt, max_points)


# ---------------------------------------------------------------------------
# glTF accessor decoding + node world transforms
# ---------------------------------------------------------------------------


def _read_accessor(gltf, blob: bytes, accessor_index: int) -> np.ndarray:
    """Decode accessor ``accessor_index`` to a (count, ncomp) numpy array.

    Handles the packed and interleaved (``byteStride``) layouts that mesh
    attributes use. Sparse accessors are not expected for POSITION/NORMAL on the
    meshes we transfer onto; they raise a clear error.
    """
    acc = gltf.accessors[accessor_index]
    if acc.sparse is not None:
        raise NotImplementedError("sparse accessors are not supported for rig transfer")
    dtype = _COMPONENT_DTYPE[acc.componentType]
    ncomp = _TYPE_NCOMP[acc.type]
    comp_bytes = np.dtype(dtype).itemsize
    bv = gltf.bufferViews[acc.bufferView]
    base = (bv.byteOffset or 0) + (acc.byteOffset or 0)
    stride = bv.byteStride or (ncomp * comp_bytes)

    out = np.empty((acc.count, ncomp), dtype=dtype)
    row = ncomp * comp_bytes
    for i in range(acc.count):
        start = base + i * stride
        out[i] = np.frombuffer(blob, dtype=dtype, count=ncomp, offset=start)
    return out


def _node_parents(gltf) -> dict:
    parent_of = {}
    for i, node in enumerate(gltf.nodes):
        for c in (node.children or []):
            parent_of[c] = i
    return parent_of


def _local_matrix(node) -> np.ndarray:
    if node.matrix:
        return np.array(node.matrix, dtype=np.float64).reshape(4, 4).T
    m = np.eye(4, dtype=np.float64)
    if node.rotation:
        x, y, z, w = node.rotation
        m[:3, :3] = [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    if node.scale:
        m[:3, :3] = m[:3, :3] @ np.diag(node.scale)
    m[:3, 3] = node.translation or [0.0, 0.0, 0.0]
    return m


def _world_matrix(gltf, parent_of: dict, i: int, cache: dict) -> np.ndarray:
    if i in cache:
        return cache[i]
    m = _local_matrix(gltf.nodes[i])
    if i in parent_of:
        m = _world_matrix(gltf, parent_of, parent_of[i], cache) @ m
    cache[i] = m
    return m


def _identity_node(node) -> None:
    node.matrix = None
    node.translation = None
    node.rotation = None
    node.scale = None


# ---------------------------------------------------------------------------
# Transfer
# ---------------------------------------------------------------------------


def _skin_for_targets(
    source_vertices: np.ndarray,
    source_skin: np.ndarray,
    target_vertices: np.ndarray,
) -> np.ndarray:
    """Dense (M, J) skin for target vertices via nearest source vertex."""
    from scipy.spatial import cKDTree

    tree = cKDTree(source_vertices)
    _, idx = tree.query(target_vertices, k=1)
    return source_skin[idx]


def transfer_rigging(
    source_asset: Asset,
    target_path: PathLike,
    export_path: PathLike,
    max_points: int = 4096,
) -> None:
    """Transfer ``source_asset``'s rig onto the original glb at ``target_path``.

    Writes a skinned glb to ``export_path`` that keeps the target's geometry,
    materials, textures, UVs and scale, with the generated skeleton + (nearest-
    neighbour) skin attached. ``source_asset`` must be rigged (vertices, joints,
    parents, dense ``skin``).
    """
    from pygltflib import GLTF2, Node, Skin

    if source_asset.vertices is None or source_asset.joints is None:
        raise ValueError("source_asset needs vertices and joints (rig it first)")
    if source_asset.parents is None or source_asset.skin is None:
        raise ValueError("source_asset needs parents and dense skin")

    source_vertices = np.asarray(source_asset.vertices, dtype=np.float64)
    source_joints = np.asarray(source_asset.joints, dtype=np.float64)
    source_skin = np.asarray(source_asset.skin, dtype=np.float64)
    parents = np.asarray(source_asset.parents, dtype=np.int64)
    J = source_joints.shape[0]

    ext = Path(target_path).suffix.lower()
    if ext not in (".glb", ".gltf"):
        raise ValueError(f"transfer target must be a glb/gltf, got {ext}")
    g = GLTF2().load_binary(str(target_path))
    blob = bytearray(g.binary_blob() or b"")

    parent_of = _node_parents(g)
    cache: dict = {}

    # Collect every skinned-mesh primitive's world-space vertices/normals.
    parts: List[dict] = []
    for node_i, node in enumerate(g.nodes):
        if node.mesh is None:
            continue
        M = _world_matrix(g, parent_of, node_i, cache)
        R = M[:3, :3]
        # normal transform = inverse-transpose of the upper 3x3
        try:
            N = np.linalg.inv(R).T
        except np.linalg.LinAlgError:
            N = R
        for prim in g.meshes[node.mesh].primitives:
            pos = _read_accessor(g, blob, prim.attributes.POSITION).astype(np.float64)
            world = (R @ pos.T).T + M[:3, 3]
            nrm = None
            if prim.attributes.NORMAL is not None:
                ln = _read_accessor(g, blob, prim.attributes.NORMAL).astype(np.float64)
                wn = (N @ ln.T).T
                norm = np.linalg.norm(wn, axis=1, keepdims=True)
                norm[norm == 0] = 1.0
                nrm = (wn / norm).astype(np.float32)
            parts.append({"node": node_i, "prim": prim, "world": world, "normal": nrm})
        _identity_node(node)

    if not parts:
        raise ValueError("target glb has no mesh to transfer the rig onto")

    target_vertices = np.vstack([p["world"] for p in parts])

    # Align source -> target, move joints into target space.
    T = estimate_similarity_transform(source_vertices, target_vertices, max_points)
    src_v_h = np.concatenate([source_vertices, np.ones((len(source_vertices), 1))], axis=1)
    aligned_source = (T @ src_v_h.T).T[:, :3]
    src_j_h = np.concatenate([source_joints, np.ones((J, 1))], axis=1)
    target_joints = (T @ src_j_h.T).T[:, :3].astype(np.float32)

    # Skin every target vertex from its nearest (aligned) source vertex.
    dense = _skin_for_targets(aligned_source, source_skin, target_vertices)

    # Inverse bind matrices: translate(-joint), column-major flat 16.
    ibm = np.tile(np.eye(4, dtype=np.float32).flatten(), (J, 1))
    ibm[:, 12] = -target_joints[:, 0]
    ibm[:, 13] = -target_joints[:, 1]
    ibm[:, 14] = -target_joints[:, 2]

    while len(blob) % 4 != 0:
        blob.append(0)
    a_ibm = _accessor(g, blob, ibm, _FLOAT, "MAT4")

    # Per-part: bake world positions/normals + JOINTS_0/WEIGHTS_0 into new accessors.
    offset = 0
    for part in parts:
        prim = part["prim"]
        world = part["world"].astype(np.float32)
        n = world.shape[0]
        joints_0, weights_0 = pack_top4(dense[offset:offset + n])
        offset += n

        a_pos = _accessor(g, blob, world, _FLOAT, "VEC3", 34962, with_minmax=True)
        prim.attributes.POSITION = a_pos
        if part["normal"] is not None:
            a_nrm = _accessor(g, blob, part["normal"], _FLOAT, "VEC3", 34962)
            prim.attributes.NORMAL = a_nrm
        a_jnt = _accessor(g, blob, joints_0, _UNSIGNED_SHORT, "VEC4", 34962)
        a_wgt = _accessor(g, blob, weights_0, _FLOAT, "VEC4", 34962)
        prim.attributes.JOINTS_0 = a_jnt
        prim.attributes.WEIGHTS_0 = a_wgt

    # Append the joint node hierarchy (translation-only, in target space).
    joint_names = source_asset.joint_names or [f"bone_{i}" for i in range(J)]
    joint_node_base = len(g.nodes)
    for i in range(J):
        p = int(parents[i])
        local = target_joints[i] - target_joints[p] if p >= 0 else target_joints[i]
        children = [joint_node_base + c for c in range(J) if int(parents[c]) == i]
        node = Node(name=str(joint_names[i]), translation=local.tolist())
        if children:
            node.children = children
        g.nodes.append(node)
    root_joint = joint_node_base + next((i for i in range(J) if int(parents[i]) == -1), 0)

    skin_index = len(g.skins)
    g.skins.append(Skin(
        inverseBindMatrices=a_ibm,
        skeleton=root_joint,
        joints=[joint_node_base + i for i in range(J)],
    ))

    # Attach the skin to each (now identity-transform) mesh node and lift it to
    # the scene root so no ancestor transform is re-applied to a skinned mesh.
    mesh_nodes = {part["node"] for part in parts}
    for node_i in mesh_nodes:
        g.nodes[node_i].skin = skin_index
    for node in g.nodes:
        if node.children:
            node.children = [c for c in node.children if c not in mesh_nodes]
            if not node.children:
                node.children = None

    scene = g.scenes[g.scene or 0]
    scene_nodes = list(scene.nodes or [])
    for node_i in mesh_nodes:
        if node_i not in scene_nodes:
            scene_nodes.append(node_i)
    scene_nodes.append(root_joint)
    scene.nodes = scene_nodes

    while len(blob) % 4 != 0:
        blob.append(0)
    g.buffers[0].byteLength = len(blob)
    g.set_binary_blob(bytes(blob))
    g.save_binary(str(export_path))
