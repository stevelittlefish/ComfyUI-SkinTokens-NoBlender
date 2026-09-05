"""ComfyUI-SkinTokens-NoBlender — custom node registration.

ComfyUI imports this package and reads NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS.
The node classes live in ``nodes.py`` (spec/05); they import ``comfy.*``/``torch``
lazily so importing this package never requires a GPU or the model weights.

ComfyUI loads this ``__init__.py`` by file spec and does NOT put the pack directory
on ``sys.path``, so the pack's own ``skintokens`` package would not be importable by
its absolute name. Add the pack dir to ``sys.path`` here so ``import skintokens``
resolves both under ComfyUI and under local pytest.
"""

import os
import sys

_PACK_DIR = os.path.dirname(os.path.abspath(__file__))
if _PACK_DIR not in sys.path:
    sys.path.insert(0, _PACK_DIR)

# Import our node module by file path, NOT as the bare name ``nodes`` — under
# ComfyUI ``nodes`` is ComfyUI's own core module, so a bare ``import nodes`` would
# silently pull in the wrong mappings. A relative import resolves to our submodule
# when loaded as a package (ComfyUI); the except covers being imported without a
# package context (e.g. pytest importing the repo root), where ``nodes`` on the
# path is unambiguously ours.
try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:
    from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
