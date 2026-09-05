"""Phase 6 local test: armature import for the ``use_skeleton`` skin-only path.

No GPU. We export a synthetic rigged Asset to a skinned glb, then re-import its
armature with :func:`import_skeleton` / :func:`load_asset_with_skeleton` and check
the joints/parents/names round-trip. (The full skin-only *inference* is model /
server-side, deferred like Gate B.)
"""

import numpy as np

from skintokens.glb_io import (
    export_glb,
    import_skeleton,
    load_asset_with_skeleton,
)
from skintokens.vendor.rig_package.info.asset import Asset


def _rigged_box_asset():
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
    skin = np.zeros((verts.shape[0], 2), dtype=np.float32)
    skin[verts[:, 1] < 1.0, 0] = 1.0
    skin[verts[:, 1] >= 1.0, 1] = 1.0
    return Asset.from_data(
        vertices=verts, faces=faces, joints=joints, parents=parents,
        skin=skin, joint_names=["hips", "spine"],
    )


def test_import_skeleton_round_trip(tmp_path):
    path = tmp_path / "rigged.glb"
    export_glb(_rigged_box_asset(), str(path))

    joints, parents, names = import_skeleton(str(path))
    assert joints.shape == (2, 3)
    assert list(parents) == [-1, 0]
    assert names == ["hips", "spine"]
    assert np.allclose(joints[0], [0, 0, 0], atol=1e-5)
    assert np.allclose(joints[1], [0, 1, 0], atol=1e-5)


def test_load_asset_with_skeleton_populates_armature(tmp_path):
    path = tmp_path / "rigged.glb"
    export_glb(_rigged_box_asset(), str(path))

    asset = load_asset_with_skeleton(str(path))
    assert asset.vertices is not None and asset.faces is not None
    assert asset.parents is not None and asset.joints is not None
    assert asset.joints.shape == (2, 3)
    assert asset.joint_names == ["hips", "spine"]
    # matrix_local translations match the joints (so the tokenizer sees them).
    assert np.allclose(asset.matrix_local[:, :3, 3], asset.joints, atol=1e-6)
