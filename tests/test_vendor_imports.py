"""Phase 0 gate: the vendored torch core imports cleanly with no Blender/server.

These are pure import smoke tests — no GPU, no model weights, no ComfyUI. They
prove the vendoring + import fixes (see skintokens/vendor/UPSTREAM.md) hold.
"""

import importlib

import pytest


def test_engine_package_imports():
    import skintokens

    assert skintokens.__version__


def test_vendor_core_modules_import():
    # The modules on the inference path (spec/02 "copy verbatim" set).
    for name in [
        "skintokens.vendor.rig_package.info.asset",
        "skintokens.vendor.tokenizer.parse",
        "skintokens.vendor.tokenizer.spec",
        "skintokens.vendor.data.transform",
        "skintokens.vendor.data.order",
        "skintokens.vendor.data.vertex_group",
        "skintokens.vendor.data.augment",
        "skintokens.vendor.model.tokenrig",
        "skintokens.vendor.model.skin_vae_model",
    ]:
        importlib.import_module(name)


def test_bpy_and_server_are_gone():
    # The Blender parser and HTTP server must not be importable.
    for name in [
        "skintokens.vendor.rig_package.parser.bpy",
        "skintokens.vendor.server.bpy_server",
    ]:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(name)


def test_no_bpy_dependency_pulled_in():
    import sys

    importlib.import_module("skintokens.vendor.model.tokenrig")
    assert "bpy" not in sys.modules, "bpy must never be imported"
