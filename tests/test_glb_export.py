"""Phase 3 local tests: skinned glb export (Gate C, the critical correctness path).

No GPU, no Blender. We build a synthetic rigged Asset (known joints/parents/skin),
export it, re-import with pygltflib/trimesh, and check every Gate-C property:
skin structure present, bind-pose identity, weight sanity, and — the one that
matters — the POSING GATE: rotating a bone deforms the mesh locally and
plausibly (implemented with a numpy reference linear-blend-skinning).
"""

import numpy as np
import pytest
from pygltflib import GLTF2

from skintokens import glb_io
from skintokens.vendor.rig_package.info.asset import Asset


# --- helpers ---------------------------------------------------------------


def _two_bone_rig():
    """A vertical 2-bone chain skinning an 8-vertex column of two stacked boxes.

    Joint 0 at y=0 (root), joint 1 at y=1. Lower verts -> joint 0, upper -> joint 1.
    """
    import trimesh

    lower = trimesh.creation.box(extents=(1, 1, 1))
    lower.apply_translation([0, 0.5, 0])
    upper = trimesh.creation.box(extents=(1, 1, 1))
    upper.apply_translation([0, 1.5, 0])
    mesh = trimesh.util.concatenate([lower, upper])
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    joints = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    parents = np.array([-1, 0], dtype=np.int64)

    # Dense skin: weight by which half the vertex is in (hard assignment).
    skin = np.zeros((verts.shape[0], 2), dtype=np.float32)
    skin[verts[:, 1] < 1.0, 0] = 1.0
    skin[verts[:, 1] >= 1.0, 1] = 1.0

    asset = Asset(vertices=verts, faces=faces, parents=parents, skin=skin)
    # matrix_local with identity rotation + joint translation (as from_data builds).
    ml = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    ml[:, :3, 3] = joints
    asset.matrix_local = ml
    asset.build_normals()
    return asset, joints, parents


def _read_back(path):
    g = GLTF2().load_binary(str(path))
    blob = g.binary_blob()

    def acc(i):
        a = g.accessors[i]
        bv = g.bufferViews[a.bufferView]
        dtype = {5126: np.float32, 5123: np.uint16, 5125: np.uint32}[a.componentType]
        comps = {"SCALAR": 1, "VEC3": 3, "VEC4": 4, "MAT4": 16}[a.type]
        off = (bv.byteOffset or 0) + (a.byteOffset or 0)
        n = a.count * comps
        arr = np.frombuffer(blob, dtype=dtype, count=n, offset=off)
        return arr.reshape(a.count, comps) if comps > 1 else arr

    return g, acc


def _node_world_translations(g):
    """World translation of each node by walking parent->child from the scene."""
    parent_of = {}
    for i, node in enumerate(g.nodes):
        for c in (node.children or []):
            parent_of[c] = i
    world = {}

    def resolve(i):
        if i in world:
            return world[i]
        t = np.array(g.nodes[i].translation or [0, 0, 0], dtype=np.float64)
        world[i] = t if i not in parent_of else resolve(parent_of[i]) + t
        return world[i]

    return {i: resolve(i) for i in range(len(g.nodes))}


# --- Gate C: structure -----------------------------------------------------


def test_export_writes_skin_structure(tmp_path):
    asset, _, _ = _two_bone_rig()
    out = tmp_path / "rig.glb"
    glb_io.export_glb(asset, out)

    g, acc = _read_back(out)
    assert len(g.skins) == 1
    skin = g.skins[0]
    assert skin.inverseBindMatrices is not None
    assert skin.joints == [0, 1]
    prim = g.meshes[0].primitives[0]
    assert prim.attributes.JOINTS_0 is not None
    assert prim.attributes.WEIGHTS_0 is not None
    # the mesh node references the skin
    mesh_nodes = [n for n in g.nodes if n.mesh is not None]
    assert len(mesh_nodes) == 1 and mesh_nodes[0].skin == 0


def test_reimports_with_trimesh(tmp_path):
    asset, _, _ = _two_bone_rig()
    out = tmp_path / "rig.glb"
    glb_io.export_glb(asset, out)
    import trimesh

    loaded = trimesh.load(out, process=False)  # must not raise
    assert loaded is not None


# --- Gate C: bind-pose identity -------------------------------------------


def test_bind_pose_identity(tmp_path):
    asset, joints, _ = _two_bone_rig()
    out = tmp_path / "rig.glb"
    glb_io.export_glb(asset, out)

    g, acc = _read_back(out)
    ibm = acc(g.skins[0].inverseBindMatrices)  # (J, 16) column-major
    world = _node_world_translations(g)

    for jidx, node_idx in enumerate(g.skins[0].joints):
        # world translation must match the joint's rest position
        np.testing.assert_allclose(world[node_idx], joints[jidx], atol=1e-5)
        # worldMatrix @ IBM == identity (translation-only => world_t + ibm_t == 0)
        m = ibm[jidx].reshape(4, 4).T  # stored column-major -> row-major
        ibm_t = m[:3, 3]
        np.testing.assert_allclose(world[node_idx] + ibm_t, np.zeros(3), atol=1e-5)


# --- Gate C: weight sanity -------------------------------------------------


def test_weight_sanity(tmp_path):
    asset, _, _ = _two_bone_rig()
    out = tmp_path / "rig.glb"
    glb_io.export_glb(asset, out)

    g, acc = _read_back(out)
    w = acc(g.meshes[0].primitives[0].attributes.WEIGHTS_0).astype(np.float64)
    j = acc(g.meshes[0].primitives[0].attributes.JOINTS_0)
    assert (w >= 0).all()
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-5)
    assert j.max() < len(g.skins[0].joints)


def test_pack_top4_selection_and_norm():
    # 5 joints; vertex 0 favors joints 3,1; vertex 1 all-zero -> joint 0.
    dense = np.array(
        [[0.1, 0.4, 0.0, 0.5, 0.0], [0, 0, 0, 0, 0]], dtype=np.float32
    )
    joints, weights = glb_io.pack_top4(dense)
    assert joints.shape == (2, 4) and weights.shape == (2, 4)
    # top influences of vertex 0 are joints 3 and 1 (descending weight)
    assert joints[0, 0] == 3 and joints[0, 1] == 1
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-6)
    # empty vertex collapses to joint 0 with weight 1
    assert joints[1, 0] == 0 and abs(weights[1, 0] - 1.0) < 1e-6


# --- Gate C: THE POSING GATE ----------------------------------------------


def _lbs_deform(g, acc, joint_local_rotations):
    """Reference linear-blend skinning: pose bones, return deformed vertices.

    ``joint_local_rotations``: {joint_index: 3x3 rotation applied in the joint's
    local frame}. Returns (N, 3) deformed positions. This is the check a glTF
    engine performs — if our IBM/node transforms are wrong, verts fly off.
    """
    prim = g.meshes[0].primitives[0]
    pos = acc(prim.attributes.POSITION).astype(np.float64)
    jnt = acc(prim.attributes.JOINTS_0).astype(np.int64)
    wgt = acc(prim.attributes.WEIGHTS_0).astype(np.float64)
    ibm = acc(g.skins[0].inverseBindMatrices).astype(np.float64)  # (J,16) col-major

    parent_of = {}
    for i, node in enumerate(g.nodes):
        for c in (node.children or []):
            parent_of[c] = i

    joints = g.skins[0].joints

    def local_matrix(node_idx):
        t = np.array(g.nodes[node_idx].translation or [0, 0, 0], dtype=np.float64)
        m = np.eye(4)
        m[:3, 3] = t
        jpos = joints.index(node_idx) if node_idx in joints else None
        if jpos is not None and jpos in joint_local_rotations:
            r = np.eye(4)
            r[:3, :3] = joint_local_rotations[jpos]
            m = m @ r  # rotate in the joint's local frame (about its own origin)
        return m

    world_cache = {}

    def world_matrix(node_idx):
        if node_idx in world_cache:
            return world_cache[node_idx]
        m = local_matrix(node_idx)
        if node_idx in parent_of:
            m = world_matrix(parent_of[node_idx]) @ m
        world_cache[node_idx] = m
        return m

    # skinning matrix per joint = world(joint) @ ibm(joint)
    skin_mats = []
    for jpos, node_idx in enumerate(joints):
        w = world_matrix(node_idx)
        b = ibm[jpos].reshape(4, 4).T  # col-major -> row-major
        skin_mats.append(w @ b)
    skin_mats = np.array(skin_mats)  # (J,4,4)

    out = np.zeros_like(pos)
    homog = np.hstack([pos, np.ones((pos.shape[0], 1))])
    for slot in range(4):
        js = jnt[:, slot]
        ws = wgt[:, slot]
        mats = skin_mats[js]  # (N,4,4)
        transformed = np.einsum("nij,nj->ni", mats, homog)[:, :3]
        out += ws[:, None] * transformed
    return out


def test_posing_gate(tmp_path):
    asset, joints, _ = _two_bone_rig()
    out = tmp_path / "rig.glb"
    glb_io.export_glb(asset, out)
    g, acc = _read_back(out)

    pos = acc(g.meshes[0].primitives[0].attributes.POSITION).astype(np.float64)

    # 1. Rest pose (no rotation) reproduces the original vertices exactly.
    rest = _lbs_deform(g, acc, {})
    np.testing.assert_allclose(rest, pos, atol=1e-4)

    # 2. Rotate the upper bone (joint 1) 90° about Z. Upper verts move; lower stay.
    theta = np.pi / 2
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta), np.cos(theta), 0],
                   [0, 0, 1]], dtype=np.float64)
    deformed = _lbs_deform(g, acc, {1: Rz})

    lower = pos[:, 1] < 1.0
    upper = ~lower

    # lower half (bound to root) must not move
    np.testing.assert_allclose(deformed[lower], pos[lower], atol=1e-4)
    # upper half must move (bone bent) ...
    assert np.linalg.norm(deformed[upper] - pos[upper], axis=1).max() > 0.3
    # ... but stay attached: no vertex collapses to origin or flies away
    assert np.linalg.norm(deformed[upper], axis=1).min() > 1e-3
    # displacement is bounded (local deformation, not a whole-mesh explosion)
    assert np.linalg.norm(deformed - pos, axis=1).max() < 5.0
    # the upper bone pivots about joint 1 (y=1): points stay at fixed radius from it
    pivot = joints[1]
    r_before = np.linalg.norm(pos[upper] - pivot, axis=1)
    r_after = np.linalg.norm(deformed[upper] - pivot, axis=1)
    np.testing.assert_allclose(r_before, r_after, atol=1e-4)


def test_export_requires_rig(tmp_path):
    import trimesh

    m = trimesh.creation.box()
    asset = Asset(
        vertices=np.asarray(m.vertices, dtype=np.float32),
        faces=np.asarray(m.faces, dtype=np.int64),
    )
    with pytest.raises(ValueError):
        glb_io.export_glb(asset, tmp_path / "x.glb")
