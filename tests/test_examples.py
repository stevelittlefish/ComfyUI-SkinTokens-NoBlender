"""Phase 7 regression: the shipped example workflow stays valid.

The example under ``examples/`` is what users load in ComfyUI, so it must keep
referencing node types we actually register and keep its ``SkinTokensRig`` widget
values aligned with the node's INPUT_TYPES (ComfyUI maps widgets by position, so a
drifted list silently loads the wrong parameters). Pure JSON checks — no GPU.
"""

import json
from pathlib import Path

import nodes

EXAMPLES = Path(__file__).parent.parent / "examples"


def _load(name):
    return json.loads((EXAMPLES / name).read_text())


def test_example_workflow_present():
    assert (EXAMPLES / "skintokens_rig.json").is_file()


def test_example_nodes_are_registered_or_core():
    d = _load("skintokens_rig.json")
    core = {"Load3D", "Preview3DAdvanced", "Save3D"}  # shipped by ComfyUI core
    for n in d["nodes"]:
        t = n["type"]
        assert t in nodes.NODE_CLASS_MAPPINGS or t in core, t


def _rig_widget_names():
    """The SkinTokensRig widget names, in the order ComfyUI renders them."""
    it = nodes.SkinTokensRig.INPUT_TYPES()
    names = []
    for section in ("required", "optional"):
        for name, spec in it.get(section, {}).items():
            if name == "mesh":
                continue  # a socket input, not a widget
            names.append(name)
    return names


def test_example_rig_widget_values_match_node_contract():
    d = _load("skintokens_rig.json")
    rig = next(n for n in d["nodes"] if n["type"] == "SkinTokensRig")
    widget_names = _rig_widget_names()
    # model is a socket; the rest are widgets in declaration order.
    expected = [w for w in widget_names if w != "model"]
    assert len(rig["widgets_values"]) == len(expected), (
        f"example has {len(rig['widgets_values'])} widget values, "
        f"node declares {len(expected)}: {expected}"
    )
    # spot-check the Phase-6 toggles sit where the node declares them.
    values = dict(zip(expected, rig["widgets_values"]))
    assert values["use_transfer"] is True
    assert values["use_postprocess"] is False
    assert values["use_skeleton"] is False
