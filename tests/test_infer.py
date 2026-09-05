"""Phase 1 local tests: the pure-CPU parts of the inference pipeline.

Covered here (no GPU, no weights):
- ``build_asset``: mesh arrays -> upstream Asset with normals.
- the pre-model transform (surface sampling) via ``prepare_asset``.
- ``rig_mesh`` input guards.

The full ``rig_mesh`` run needs the ~14 GB model on a GPU and is Gate B
(server-side); it is exercised by ``test_rig_mesh_end_to_end`` only when
SKINTOKENS_RUN_MODEL is set.
"""

import os

import numpy as np
import pytest
import torch
import trimesh

from skintokens import infer
from skintokens.model_loader import SkinTokensModel
from skintokens.vendor.data.sampler import SamplerMix
from skintokens.vendor.data.transform import Transform


def _box():
    m = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
    return np.asarray(m.vertices, dtype=np.float32), np.asarray(m.faces, dtype=np.int64)


def test_build_asset_populates_mesh_fields():
    verts, faces = _box()
    asset = infer.build_asset(verts, faces)

    assert asset.vertices.shape == verts.shape
    assert asset.faces.shape == faces.shape
    assert asset.vertex_normals.shape == verts.shape
    assert asset.face_normals.shape == faces.shape
    assert asset.mesh_names == ["mesh"]
    assert asset.cls == "articulation"
    assert asset.N == verts.shape[0]
    assert asset.F == faces.shape[0]


def test_build_asset_uses_supplied_normals():
    verts, faces = _box()
    normals = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (verts.shape[0], 1))
    asset = infer.build_asset(verts, faces, normals=normals)
    np.testing.assert_allclose(asset.vertex_normals, normals)


@pytest.mark.parametrize(
    "verts, faces",
    [
        (np.zeros((4, 2), dtype=np.float32), np.zeros((2, 3), dtype=np.int64)),
        (np.zeros((4, 3), dtype=np.float32), np.zeros((2, 4), dtype=np.int64)),
    ],
)
def test_build_asset_rejects_bad_shapes(verts, faces):
    with pytest.raises(ValueError):
        infer.build_asset(verts, faces)


def _cpu_bundle(transform):
    # A bundle with no model — enough to test the CPU-only transform path.
    return SkinTokensModel(
        model=None,  # type: ignore[arg-type]
        tokenizer=None,  # type: ignore[arg-type]
        transform=transform,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_prepare_asset_samples_surface():
    verts, faces = _box()
    asset = infer.build_asset(verts, faces)
    n = 256
    transform = Transform(sampler=SamplerMix(num_samples=n, num_vertex_samples=0))

    infer.prepare_asset(_cpu_bundle(transform), asset)

    assert asset.sampled_vertices.shape == (n, 3)
    assert asset.sampled_normals.shape == (n, 3)


def test_rig_mesh_rejects_use_skeleton():
    verts, faces = _box()
    transform = Transform(sampler=SamplerMix(num_samples=64, num_vertex_samples=0))
    with pytest.raises(NotImplementedError):
        infer.rig_mesh(_cpu_bundle(transform), verts, faces, use_skeleton=True)


@pytest.mark.server
@pytest.mark.skipif(
    not os.environ.get("SKINTOKENS_RUN_MODEL"),
    reason="needs the ~14 GB model + GPU (Gate B, server-side)",
)
def test_rig_mesh_end_to_end():
    from skintokens.model_loader import load_model

    verts, faces = _box()
    bundle = load_model(device=os.environ.get("SKINTOKENS_DEVICE", "cuda"))
    rigged = infer.rig_mesh(bundle, verts, faces)
    assert rigged.parents is not None and rigged.J > 0
    assert rigged.skin is not None and rigged.skin.shape[0] == verts.shape[0]
