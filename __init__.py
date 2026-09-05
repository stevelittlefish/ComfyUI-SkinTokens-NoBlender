"""ComfyUI-SkinTokens-NoBlender — custom node registration.

ComfyUI imports this package and reads NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS.
The node classes live in ``nodes.py`` (spec/05); they import ``comfy.*``/``torch``
lazily so importing this package never requires a GPU or the model weights.
"""

try:
    # Normal case: ComfyUI imports this folder as a package.
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:
    # Imported without package context (e.g. pytest importing the rootdir).
    from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
