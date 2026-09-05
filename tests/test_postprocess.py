"""Phase 6 local tests: optional voxel-skin post-process (``use_postprocess``).

No GPU. We voxelize a synthetic mesh surface and run the voxel-heat refinement on
a known 3-bone cylinder, checking the output is a valid (normalized, finite) skin
and that the refinement keeps weight concentrated along the bone axis rather than
collapsing or exploding.
"""

import numpy as np
import pytest
import trimesh

from skintokens.postprocess import _voxelize_surface, apply_voxel_postprocess, voxel_skin
from skintokens.vendor.rig_package.info.asset import Asset


def _cylinder_rig(sections=24):
    m = trimesh.creation.cylinder(radius=0.2, height=2.0, sections=sections)
    V = np.asarray(m.vertices, dtype=np.float64)
    F = np.asarray(m.faces, dtype=np.int64)
    joints = np.array([[0, 0, -0.8], [0, 0, 0.0], [0, 0, 0.8]], dtype=np.float64)
    parents = np.array([-1, 0, 1])
    d = np.linalg.norm(V[:, None, :] - joints[None], axis=2)
    skin = np.zeros((len(V), 3))
    skin[np.arange(len(V)), d.argmin(1)] = 1.0
    return V, F, joints, parents, skin


def test_voxelize_surface_covers_bounds():
    V, F, *_ = _cylinder_rig()
    centres, vsize = _voxelize_surface(V, F, resolution=40)
    assert vsize > 0
    assert centres.shape[0] > 0
    # voxel centres stay within (an eps of) the mesh bounds.
    assert centres.min(0).min() >= V.min(0).min() - vsize
    assert centres.max(0).max() <= V.max(0).max() + vsize


def test_voxel_skin_returns_normalized_finite():
    V, F, joints, _, _ = _cylinder_rig()
    centres, vsize = _voxelize_surface(V, F, resolution=40)
    skin = voxel_skin(
        grid=0, grid_coords=centres, joints=joints, vertices=V, faces=F,
        mode="square", voxel_size=vsize,
    )
    assert skin.shape == (len(V), 3)
    assert np.isfinite(skin).all()
    assert np.allclose(skin.sum(1), 1.0, atol=1e-6)


def test_apply_voxel_postprocess_in_place():
    V, F, joints, parents, skin = _cylinder_rig()
    asset = Asset.from_data(
        vertices=V.astype(np.float32), faces=F,
        joints=joints.astype(np.float32), parents=parents, skin=skin.copy(),
    )
    out = apply_voxel_postprocess(asset, resolution=40)
    assert out is asset
    assert asset.skin.shape == (len(V), 3)
    assert np.isfinite(asset.skin).all()
    assert np.allclose(asset.skin.sum(1), 1.0, atol=1e-5)
    # top-end vertices should still favour the top joint after refinement.
    top = V[:, 2] > 0.7
    assert asset.skin[top].argmax(1).mean() > 1.0


def test_apply_voxel_postprocess_requires_rig():
    V, F, *_ = _cylinder_rig()
    asset = Asset(vertices=V.astype(np.float32), faces=F)
    with pytest.raises(ValueError):
        apply_voxel_postprocess(asset)
