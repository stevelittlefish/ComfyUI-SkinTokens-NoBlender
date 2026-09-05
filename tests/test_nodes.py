"""Phase 5 local tests: ComfyUI node wiring (no GPU, no ComfyUI, no torch).

These cover the parts of the node pack that run without ComfyUI/torch/the model:
node registration + INPUT_TYPES contract, the pure-Python ``relabel_glb`` path
(via a rigged glb built from the committed skeleton fixtures), the standalone
``SkinTokensRelabel`` node, and the VRAM-wrapper's no-ComfyUI fallback + size
estimator. The full Rig pipeline on the GPU is a ``server``-marked test here
(``test_rig_node_end_to_end``); ComfyUI's own VRAM lifecycle is Gate F, inside ComfyUI.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from pygltflib import GLTF2

import nodes
from skintokens import glb_io
from skintokens.vendor.rig_package.info.asset import Asset

_FIXTURES = Path(__file__).parent / "fixtures" / "rigs"


# --- helpers ---------------------------------------------------------------


def _rigged_glb_from_fixture(name, path):
    """Build a minimal rigged glb from a committed skeleton fixture.

    Mesh = one vertex per joint (placed at the joint), a few dummy triangles so
    the primitive is valid, identity skin (vertex i -> joint i). Enough to make a
    real skinned glb that ``relabel_glb`` can read a skeleton from.
    """
    d = json.loads((_FIXTURES / f"{name}.json").read_text())
    joints = np.array(d["joints"], dtype=np.float32)
    parents = np.array(d["parents"], dtype=np.int64)
    J = joints.shape[0]

    verts = joints.copy()
    faces = np.array([[i, (i + 1) % J, (i + 2) % J] for i in range(J)], dtype=np.int64)
    skin = np.eye(J, dtype=np.float32)  # vertex i fully weighted to joint i

    asset = Asset(vertices=verts, faces=faces, parents=parents, skin=skin)
    ml = np.tile(np.eye(4, dtype=np.float32), (J, 1, 1))
    ml[:, :3, 3] = joints
    asset.matrix_local = ml
    asset.build_normals()
    glb_io.export_glb(asset, path)
    return path


# --- registration / metadata ----------------------------------------------


def test_node_mappings_present():
    assert set(nodes.NODE_CLASS_MAPPINGS) == {
        "SkinTokensLoader", "SkinTokensRig", "SkinTokensRelabel"
    }
    for key, cls in nodes.NODE_CLASS_MAPPINGS.items():
        assert key in nodes.NODE_DISPLAY_NAME_MAPPINGS
        assert hasattr(cls, "INPUT_TYPES")
        assert isinstance(cls.INPUT_TYPES(), dict)
        assert cls.RETURN_TYPES  # non-empty tuple
        assert isinstance(cls.FUNCTION, str) and hasattr(cls, cls.FUNCTION)
        assert cls.CATEGORY == "SkinTokens"


def test_rig_input_contract():
    it = nodes.SkinTokensRig.INPUT_TYPES()["required"]
    assert it["model"][0] == "SKINTOKENS_MODEL"
    # mesh socket accepts native MESH or any FILE_3D_* (comma-joined type string).
    accepted = it["mesh"][0].split(",")
    assert "MESH" in accepted and "FILE_3D_GLB" in accepted
    assert it["convention"][0] == nodes.CONVENTIONS
    assert nodes.SkinTokensRig.RETURN_TYPES == ("FILE_3D_GLB",)


def test_relabel_input_output_contract():
    it = nodes.SkinTokensRelabel.INPUT_TYPES()["required"]
    accepted = it["glb"][0].split(",")
    assert "FILE_3D_GLB" in accepted and "MESH" not in accepted  # rigged file only
    assert nodes.SkinTokensRelabel.RETURN_TYPES == ("FILE_3D_GLB",)


def test_loader_outputs_model_type():
    assert nodes.SkinTokensLoader.RETURN_TYPES == ("SKINTOKENS_MODEL",)


def test_loader_is_a_model_dropdown():
    it = nodes.SkinTokensLoader.INPUT_TYPES()["required"]
    # A ComfyUI combo: first element is the list of options.
    assert isinstance(it["model"][0], list) and it["model"][0]
    assert it["model"][1]["default"] in it["model"][0]
    # options map to real HF checkpoint paths.
    for name in it["model"][0]:
        assert nodes.MODELS[name].endswith(".ckpt")
    # the confusing old widgets are gone.
    assert "download" not in it and "models_dir" not in it


def test_convention_keys_map_to_engine():
    assert nodes._CONVENTION_KEY["Mixamo"] == "mixamo"
    assert nodes._CONVENTION_KEY["UE5"] == "ue5"
    assert set(nodes.CONVENTIONS) == set(nodes._CONVENTION_KEY)


# --- relabel_glb (pure Python engine path) ---------------------------------


@pytest.mark.parametrize("name", ["knight", "peasant", "robot_industrial", "sci-fi-dude"])
def test_relabel_glb_renames_joint_nodes(name, tmp_path):
    src = _rigged_glb_from_fixture(name, str(tmp_path / f"{name}.glb"))
    out = str(tmp_path / f"{name}_relabeled.glb")

    applied = glb_io.relabel_glb(src, out, convention="mixamo")
    assert applied  # something was renamed

    g = GLTF2().load_binary(out)
    names = {g.nodes[i].name for i in range(len(g.nodes))}
    # the core body bones must now carry mixamorig names.
    for role in ("Hips", "Spine", "Neck", "Head", "LeftHand", "RightHand",
                 "LeftUpLeg", "RightUpLeg"):
        assert f"mixamorig:{role}" in names, f"{name} missing {role}"


def test_relabel_glb_ue5_and_index_stability(tmp_path):
    src = _rigged_glb_from_fixture("knight", str(tmp_path / "k.glb"))
    before = GLTF2().load_binary(src)
    joints_before = list(before.skins[0].joints)

    out = str(tmp_path / "k_ue5.glb")
    glb_io.relabel_glb(src, out, convention="ue5")
    after = GLTF2().load_binary(out)

    # skin joint indices (and thus JOINTS_0 references) are unchanged.
    assert list(after.skins[0].joints) == joints_before
    names = {n.name for n in after.nodes}
    assert "pelvis" in names and "hand_l" in names and "head" in names


def test_relabel_glb_on_unrigged_raises(tmp_path):
    import trimesh

    p = str(tmp_path / "plain.glb")
    trimesh.creation.box().export(p)
    with pytest.raises(ValueError):
        glb_io.relabel_glb(p, str(tmp_path / "out.glb"))


# --- native-type bridge (comfy_types) ---------------------------------------


class _FakeFile3D:
    """Minimal stand-in for ComfyUI's File3D (disk-backed) for tests."""

    def __init__(self, path):
        self._path = path
        self.format = "glb"

    def get_source(self):
        return self._path

    def get_bytes(self):
        return Path(self._path).read_bytes()

    def save_to(self, dest):
        import shutil

        shutil.copy2(self._path, dest)
        return dest


class _FakeMesh:
    """Minimal stand-in for ComfyUI's MESH (batched numpy tensors)."""

    def __init__(self, vertices, faces, normals=None):
        # batch dim (B, N, 3) like the real type.
        self.vertices = vertices[None]
        self.faces = faces[None]
        self.normals = normals[None] if normals is not None else None


def test_comfy_mesh_to_arrays_takes_batch_item0():
    from skintokens import comfy_types

    v = np.arange(12, dtype=np.float32).reshape(4, 3)
    f = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    n = np.ones((4, 3), dtype=np.float32)
    verts, faces, normals = comfy_types.comfy_mesh_to_arrays(_FakeMesh(v, f, n))
    assert verts.shape == (4, 3) and faces.shape == (2, 3)
    assert np.allclose(verts, v) and normals.shape == (4, 3)


def test_is_comfy_mesh_vs_file3d(tmp_path):
    from skintokens import comfy_types

    src = _rigged_glb_from_fixture("knight", str(tmp_path / "k.glb"))
    assert comfy_types.is_file3d(_FakeFile3D(src))
    assert not comfy_types.is_comfy_mesh(_FakeFile3D(src))
    assert comfy_types.is_comfy_mesh(
        _FakeMesh(np.zeros((1, 3), np.float32), np.zeros((1, 3), np.int64))
    )


def test_file3d_to_path_disk_backed(tmp_path):
    from skintokens import comfy_types

    src = _rigged_glb_from_fixture("knight", str(tmp_path / "k.glb"))
    assert comfy_types.file3d_to_path(_FakeFile3D(src)) == src


def test_make_file3d_fallback_returns_path_without_comfy():
    from skintokens import comfy_types

    # comfy_api is not importable here -> falls back to the path string.
    assert comfy_types.make_file3d("/tmp/x.glb", "glb") == "/tmp/x.glb"


# --- standalone relabel node ------------------------------------------------


def test_relabel_node_end_to_end(tmp_path, monkeypatch):
    src = _rigged_glb_from_fixture("peasant", str(tmp_path / "peasant.glb"))
    monkeypatch.setattr(nodes, "_output_dir", lambda: str(tmp_path))

    node = nodes.SkinTokensRelabel()
    # comfy_api absent -> make_file3d returns the output path string.
    (out_path,) = node.relabel(
        glb=_FakeFile3D(src), convention="Mixamo", filename_prefix="out"
    )

    assert Path(out_path).parent == tmp_path
    names = {n.name for n in GLTF2().load_binary(out_path).nodes}
    assert "mixamorig:Hips" in names


def test_relabel_node_rejects_non_file3d(tmp_path, monkeypatch):
    monkeypatch.setattr(nodes, "_output_dir", lambda: str(tmp_path))
    with pytest.raises(TypeError):
        nodes.SkinTokensRelabel().relabel(glb="/some/path.glb")  # bare string, not File3D


# --- server: full node pipeline on the GPU (Phase 5 glue end to end) ---------


@pytest.mark.server
@pytest.mark.skipif(
    not os.environ.get("SKINTOKENS_RUN_MODEL"),
    reason="needs the ~14 GB model + GPU (server-side)",
)
def test_rig_node_end_to_end(tmp_path, monkeypatch):
    """SkinTokensRig via a File3D input -> rigged skinned FILE_3D_GLB, on the GPU.

    Exercises the Phase 5 node glue (comfy_types bridge + wrapper.prepare no-op +
    relabel + export) around real inference, without needing ComfyUI. ComfyUI's
    own VRAM lifecycle (Gate F) is a separate check inside ComfyUI.
    """
    from skintokens.comfy_model import SkinTokensModelWrapper
    from skintokens.model_loader import load_model

    glb = Path(__file__).parent / "fixtures" / "meshes" / "dummy.glb"

    models_dir = os.environ.get("SKINTOKENS_MODELS_DIR") or None
    bundle = load_model(
        device=os.environ.get("SKINTOKENS_DEVICE", "cuda"), models_dir=models_dir
    )
    model = SkinTokensModelWrapper(bundle, patcher=None)  # no ComfyUI -> prepare() no-op
    monkeypatch.setattr(nodes, "_output_dir", lambda: str(tmp_path))

    # comfy_api absent here -> make_file3d falls back to the output path string.
    out = nodes.SkinTokensRig().rig(
        model=model, mesh=_FakeFile3D(str(glb)), relabel=True, convention="Mixamo"
    )[0]
    out_path = out if isinstance(out, str) else out.get_source()

    g = GLTF2().load_binary(out_path)
    assert len(g.skins) == 1 and len(g.skins[0].joints) > 0
    prim = g.meshes[0].primitives[0]
    assert prim.attributes.JOINTS_0 is not None
    assert prim.attributes.WEIGHTS_0 is not None
    # dummy.glb is a humanoid -> the relabeler should have named the core bones.
    names = {g.nodes[j].name for j in g.skins[0].joints}
    assert any(str(n).startswith("mixamorig:") for n in names), names


# --- VRAM wrapper (no-ComfyUI fallback + size estimator) --------------------


def test_wrap_for_comfy_without_comfy_returns_passthrough():
    from skintokens.comfy_model import wrap_for_comfy

    sentinel = object()

    class FakeBundle:
        model = sentinel

    w = wrap_for_comfy(FakeBundle())  # comfy not importable here
    assert w.patcher is None
    assert w.model is sentinel
    assert w.prepare() is not None  # no-op path returns the bundle


def test_estimate_model_size():
    from skintokens.comfy_model import estimate_model_size

    class _T:
        def __init__(self, n, es):
            self._n, self._es = n, es

        def numel(self):
            return self._n

        def element_size(self):
            return self._es

    class _Model:
        def parameters(self):
            return [_T(1000, 4), _T(500, 4)]

        def buffers(self):
            return [_T(10, 2)]

    assert estimate_model_size(_Model()) == 1000 * 4 + 500 * 4 + 10 * 2
