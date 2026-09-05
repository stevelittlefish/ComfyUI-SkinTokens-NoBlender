"""Phase 6 local tests: texture-preserving rig transfer (``use_transfer``).

No GPU / no Blender. We build a synthetic rigged source Asset and transfer it onto
a real textured target glb (the committed ``dummy.glb``), then check that:
  * the output is a valid skinned glb (skin + JOINTS_0/WEIGHTS_0 present),
  * the target's materials/textures/UVs survive untouched,
  * the geometry is preserved (baked world positions == original),
  * bind pose is identity (world @ IBM == I) and weights sum to 1.
Also covers the similarity-transform ports directly.
"""

import numpy as np
import pytest
from pygltflib import GLTF2

from skintokens import transfer
from skintokens.glb_io import _skeleton_from_gltf, load_mesh
from skintokens.transfer import (
    _read_accessor,
    estimate_similarity_transform,
    transfer_rigging,
)
from skintokens.vendor.rig_package.info.asset import Asset

DUMMY = "tests/fixtures/meshes/dummy.glb"


# --- similarity transform --------------------------------------------------


def test_umeyama_recovers_known_similarity():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(200, 3))
    theta = 0.7
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1.0]])
    scale, t = 2.5, np.array([1.0, -2.0, 3.0])
    tgt = (scale * (R @ src.T).T) + t
    T = estimate_similarity_transform(src, tgt)  # equal counts -> Umeyama
    src_h = np.concatenate([src, np.ones((len(src), 1))], axis=1)
    recovered = (T @ src_h.T).T[:, :3]
    assert np.allclose(recovered, tgt, atol=1e-6)


def test_pca_similarity_aligns_extent_when_counts_differ():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(300, 3))
    tgt = (3.0 * src) + np.array([5.0, 0.0, 0.0])
    tgt = tgt[:250]  # different count -> PCA branch
    T = estimate_similarity_transform(src, tgt)
    src_h = np.concatenate([src, np.ones((len(src), 1))], axis=1)
    aligned = (T @ src_h.T).T[:, :3]
    # centroids and overall scale should match closely.
    assert np.allclose(aligned.mean(0), tgt.mean(0), atol=0.3)
    assert abs(aligned.std() - tgt.std()) < 0.3


# --- transfer onto a real textured glb -------------------------------------


def _source_from_target(step=4):
    """A synthetic rigged source: a decimated copy of the target with a 3-bone rig."""
    m = load_mesh(DUMMY)
    V = m.vertices.astype(np.float64)
    sv = V[np.arange(0, len(V), step)]
    lo, hi = V.min(0), V.max(0)
    cx, cz = (lo[0] + hi[0]) / 2, (lo[2] + hi[2]) / 2
    joints = np.array([[cx, (lo[1] + hi[1]) / 2, cz],
                       [cx, hi[1] * 0.8, cz],
                       [cx, hi[1], cz]])
    parents = np.array([-1, 0, 1])
    d = np.linalg.norm(sv[:, None, :] - joints[None], axis=2)
    skin = np.zeros((len(sv), 3))
    skin[np.arange(len(sv)), d.argmin(1)] = 1.0
    return Asset.from_data(
        vertices=sv.astype(np.float32), joints=joints.astype(np.float32),
        parents=parents, skin=skin, joint_names=["hips", "spine", "head"],
    ), V


def test_transfer_preserves_materials_and_adds_skin(tmp_path):
    src_asset, orig_world = _source_from_target()
    before = GLTF2().load_binary(DUMMY)
    out = tmp_path / "transferred.glb"
    transfer_rigging(src_asset, DUMMY, str(out))

    g = GLTF2().load_binary(str(out))
    blob = g.binary_blob()

    # materials / textures / images survive.
    assert len(g.materials) == len(before.materials) >= 1
    assert len(g.images) == len(before.images) >= 1
    assert len(g.skins) == 1

    prim = g.meshes[0].primitives[0]
    assert prim.attributes.JOINTS_0 is not None
    assert prim.attributes.WEIGHTS_0 is not None
    assert prim.attributes.TEXCOORD_0 is not None  # UVs preserved

    # geometry preserved (dummy has an identity node transform).
    pos = _read_accessor(g, blob, prim.attributes.POSITION)
    assert np.allclose(pos, orig_world, atol=1e-5)

    # weights normalized.
    w = _read_accessor(g, blob, prim.attributes.WEIGHTS_0)
    assert np.allclose(w.sum(1), 1.0, atol=1e-5)

    # bind pose identity: joint world (translation-only) @ IBM == I.
    joints, _, joint_nodes = _skeleton_from_gltf(g)
    ibm = _read_accessor(g, blob, g.skins[0].inverseBindMatrices)
    ibm = ibm.reshape(-1, 4, 4).transpose(0, 2, 1)  # column-major -> row-major
    for k in range(len(joint_nodes)):
        W = np.eye(4)
        W[:3, 3] = joints[k]
        assert np.allclose(W @ ibm[k], np.eye(4), atol=1e-4)


def test_transfer_reimportable_by_trimesh(tmp_path):
    import trimesh

    src_asset, _ = _source_from_target()
    out = tmp_path / "t.glb"
    transfer_rigging(src_asset, DUMMY, str(out))
    scene = trimesh.load(str(out))
    assert isinstance(scene, trimesh.Scene)
    # the textured geometry is still there.
    assert sum(len(g.vertices) for g in scene.geometry.values()) > 0


def test_transfer_rejects_unrigged_source(tmp_path):
    m = load_mesh(DUMMY)
    asset = Asset(vertices=m.vertices, faces=m.faces)
    with pytest.raises(ValueError):
        transfer_rigging(asset, DUMMY, str(tmp_path / "x.glb"))
