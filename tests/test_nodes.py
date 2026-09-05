"""Phase 5 local tests: ComfyUI node wiring (no GPU, no ComfyUI, no torch).

These cover the parts of the node pack that run without ComfyUI/torch/the model:
node registration + INPUT_TYPES contract, the pure-Python ``relabel_glb`` path
(via a rigged glb built from the committed skeleton fixtures), the standalone
``SkinTokensRelabel`` node, and the VRAM-wrapper's no-ComfyUI fallback + size
estimator. The GPU paths (Loader/Rig) are exercised on the server (Gate F).
"""

import json
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
    assert it["glb"][0] == "STRING"
    assert it["convention"][0] == nodes.CONVENTIONS
    assert nodes.SkinTokensRig.RETURN_TYPES == ("STRING",)


def test_loader_outputs_model_type():
    assert nodes.SkinTokensLoader.RETURN_TYPES == ("SKINTOKENS_MODEL",)


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


# --- standalone relabel node ------------------------------------------------


def test_relabel_node_end_to_end(tmp_path, monkeypatch):
    src = _rigged_glb_from_fixture("peasant", str(tmp_path / "peasant.glb"))
    monkeypatch.setattr(nodes, "_output_dir", lambda: str(tmp_path))

    node = nodes.SkinTokensRelabel()
    (out_path,) = node.relabel(glb=src, convention="Mixamo", filename_prefix="out")

    assert Path(out_path).parent == tmp_path
    names = {n.name for n in GLTF2().load_binary(out_path).nodes}
    assert "mixamorig:Hips" in names


def test_relabel_node_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(nodes, "_output_dir", lambda: str(tmp_path))
    with pytest.raises(FileNotFoundError):
        nodes.SkinTokensRelabel().relabel(glb=str(tmp_path / "nope.glb"))


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
