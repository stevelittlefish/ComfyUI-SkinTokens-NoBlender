"""ComfyUI node classes for SkinTokens auto-rigging (spec/05).

Three nodes:
  * ``SkinTokensLoader`` — load the TokenRig model, wrapped for ComfyUI's VRAM
    management (``SKINTOKENS_MODEL`` output).
  * ``SkinTokensRig``    — static mesh glb -> rigged skinned glb (+ optional relabel).
  * ``SkinTokensRelabel``— standalone structural relabel of an already-rigged glb.

I/O uses **STRING filepaths** for meshes (spec/05 I/O decision: the safe default
for the automation use case; a custom MESH socket can be added later).

ComfyUI-only imports (``comfy.*``, ``folder_paths``, ``torch``) are done lazily
inside methods so this module imports cleanly under local pytest, where the node
*logic* that needs them is not exercised. ComfyUI reads NODE_CLASS_MAPPINGS at
startup; that must never require a GPU or the model.
"""

from __future__ import annotations

import os
from typing import Tuple

CONVENTIONS = ["Mixamo", "UE5"]
_CONVENTION_KEY = {"Mixamo": "mixamo", "UE5": "ue5"}

CATEGORY = "SkinTokens"


def _output_dir() -> str:
    """ComfyUI output directory, or the CWD when running outside ComfyUI."""
    try:
        import folder_paths

        return folder_paths.get_output_directory()
    except Exception:
        return os.getcwd()


def _resolve_input(path: str) -> str:
    """Resolve a possibly-relative input mesh path against ComfyUI's input dir."""
    if os.path.isabs(path) or os.path.exists(path):
        return path
    try:
        import folder_paths

        candidate = os.path.join(folder_paths.get_input_directory(), path)
        if os.path.exists(candidate):
            return candidate
    except Exception:
        pass
    return path


def _unique_output_path(filename: str) -> str:
    """An output path in the ComfyUI output dir, avoiding clobbering existing files."""
    out_dir = _output_dir()
    os.makedirs(out_dir, exist_ok=True)
    base, ext = os.path.splitext(filename)
    ext = ext or ".glb"
    path = os.path.join(out_dir, base + ext)
    n = 1
    while os.path.exists(path):
        path = os.path.join(out_dir, f"{base}_{n:03d}{ext}")
        n += 1
    return path


class SkinTokensLoader:
    """Load the SkinTokens/TokenRig model, wrapped for ComfyUI VRAM management."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "download": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                # Empty string -> shared HF cache; a path -> private copy there.
                "models_dir": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("SKINTOKENS_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, download: bool = True, models_dir: str = ""):
        import comfy.model_management as mm

        from skintokens.comfy_model import wrap_for_comfy
        from skintokens.model_loader import load_model

        offload_device = mm.unet_offload_device()
        bundle = load_model(
            models_dir=(models_dir or None),
            device=offload_device,  # load to CPU; ComfyUI moves it to GPU on demand
            download=download,
        )
        return (wrap_for_comfy(bundle),)


class SkinTokensRig:
    """Rig a static mesh glb -> rigged skinned glb (+ optional Mixamo/UE5 relabel)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SKINTOKENS_MODEL",),
                "glb": ("STRING", {"default": "input.glb"}),
                "filename_prefix": ("STRING", {"default": "skintokens_rigged"}),
                "relabel": ("BOOLEAN", {"default": True}),
                "convention": (CONVENTIONS, {"default": "Mixamo"}),
                "relabel_fingers": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "top_k": ("INT", {"default": 5, "min": 1, "max": 200}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05}),
                "repetition_penalty": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 5.0, "step": 0.1}),
                "num_beams": ("INT", {"default": 10, "min": 1, "max": 50}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("rigged_glb",)
    FUNCTION = "rig"
    CATEGORY = CATEGORY

    def rig(
        self,
        model,
        glb: str,
        filename_prefix: str = "skintokens_rigged",
        relabel: bool = True,
        convention: str = "Mixamo",
        relabel_fingers: bool = True,
        top_k: int = 5,
        top_p: float = 0.95,
        temperature: float = 1.0,
        repetition_penalty: float = 2.0,
        num_beams: int = 10,
    ) -> Tuple[str]:
        from skintokens import glb_io, infer, relabel as relabel_mod

        in_path = _resolve_input(glb)
        if not os.path.exists(in_path):
            raise FileNotFoundError(f"input mesh not found: {glb}")

        bundle = model.prepare()  # ComfyUI moves weights to the compute device
        generate_kwargs = dict(
            top_k=int(top_k),
            top_p=float(top_p),
            temperature=float(temperature),
            repetition_penalty=float(repetition_penalty),
            num_beams=int(num_beams),
        )

        rigged = infer.rig_glb(bundle, in_path, generate_kwargs=generate_kwargs)

        if relabel:
            relabel_mod.relabel_asset(
                rigged,
                convention=_CONVENTION_KEY[convention],
                with_fingers=relabel_fingers,
            )

        out_path = _unique_output_path(filename_prefix + ".glb")
        glb_io.export_glb(rigged, out_path)
        return (out_path,)


class SkinTokensRelabel:
    """Structurally relabel an already-rigged glb's joints (Mixamo/UE5). No model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "glb": ("STRING", {"default": "rigged.glb"}),
                "filename_prefix": ("STRING", {"default": "skintokens_relabeled"}),
                "convention": (CONVENTIONS, {"default": "Mixamo"}),
                "relabel_fingers": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("relabeled_glb",)
    FUNCTION = "relabel"
    CATEGORY = CATEGORY

    def relabel(
        self,
        glb: str,
        filename_prefix: str = "skintokens_relabeled",
        convention: str = "Mixamo",
        relabel_fingers: bool = True,
    ) -> Tuple[str]:
        from skintokens import glb_io

        in_path = _resolve_input(glb)
        if not os.path.exists(in_path):
            raise FileNotFoundError(f"input glb not found: {glb}")

        out_path = _unique_output_path(filename_prefix + ".glb")
        glb_io.relabel_glb(
            in_path,
            out_path,
            convention=_CONVENTION_KEY[convention],
            with_fingers=relabel_fingers,
        )
        return (out_path,)


NODE_CLASS_MAPPINGS = {
    "SkinTokensLoader": SkinTokensLoader,
    "SkinTokensRig": SkinTokensRig,
    "SkinTokensRelabel": SkinTokensRelabel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SkinTokensLoader": "SkinTokens Loader",
    "SkinTokensRig": "SkinTokens Rig",
    "SkinTokensRelabel": "SkinTokens Relabel",
}
