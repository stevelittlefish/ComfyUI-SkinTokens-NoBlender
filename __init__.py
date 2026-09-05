"""ComfyUI-SkinTokens-NoBlender — custom node registration.

ComfyUI imports this package and reads NODE_CLASS_MAPPINGS. The nodes themselves
are added in Phase 5 (see spec/05); for now this registers nothing so the pack
loads cleanly while the engine is being built.
"""

NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
