"""ComfyUI node classes for SkinTokens auto-rigging (spec/05).

Three nodes:
  * ``SkinTokensLoader`` — load the TokenRig model, wrapped for ComfyUI's VRAM
    management (``SKINTOKENS_MODEL`` output).
  * ``SkinTokensRig``    — static mesh -> rigged skinned glb (+ optional relabel).
  * ``SkinTokensRelabel``— standalone structural relabel of an already-rigged glb.

Mesh I/O uses ComfyUI's **native 3D types** so the nodes drop into the standard
3D graph (Load3D / Trellis / remesh / Preview3DAdvanced / Save3D):
  * ``SkinTokensRig`` accepts a native ``MESH`` (from the Trellis/remesh chain) or
    a ``File3D`` (``FILE_3D_GLB``/``GLTF``/``OBJ``/``STL``, from Load3D).
  * The rigged output is a ``FILE_3D_GLB`` — the glb file carries the skeleton +
    skin weights (the ``MESH`` type has no armature), and it feeds
    ``Preview3DAdvanced`` (showSkeleton) / ``Save3D`` directly.

ComfyUI-only imports (``comfy.*``, ``folder_paths``, ``torch``, ``comfy_api``) are
done lazily inside methods / the bridge module so this file imports cleanly under
local pytest, where the paths that need them are not exercised. ComfyUI reads
NODE_CLASS_MAPPINGS at startup; that must never require a GPU or the model.
"""

from __future__ import annotations

import os
from typing import Tuple

from skintokens.comfy_types import (
    MESH_OR_FILE3D_TYPES,
    RIGGED_FILE3D_TYPES,
)

CONVENTIONS = ["Mixamo", "UE5"]
_CONVENTION_KEY = {"Mixamo": "mixamo", "UE5": "ue5"}

# Selectable TokenRig checkpoints in the HF repo (VAST-AI/SkinTokens). Friendly
# name -> checkpoint path. Auto-downloaded to the HF cache (HF_HOME) on first use.
# Currently the repo ships one; kept as a dict so more can be added later.
MODELS = {
    "articulation (GRPO)": "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt",
}
_DEFAULT_MODEL = next(iter(MODELS))

CATEGORY = "SkinTokens"


def _output_dir() -> str:
    """ComfyUI output directory, or the CWD when running outside ComfyUI."""
    try:
        import folder_paths

        return folder_paths.get_output_directory()
    except Exception:
        return os.getcwd()


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
    """Load a SkinTokens/TokenRig model, wrapped for ComfyUI VRAM management.

    Pick a model from the dropdown; it is downloaded to the HuggingFace cache
    (``HF_HOME``) on first use and reused thereafter — the standard "auto-download
    from a preselected list" loader pattern. No path to configure.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (list(MODELS.keys()), {"default": _DEFAULT_MODEL}),
            },
        }

    RETURN_TYPES = ("SKINTOKENS_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, model: str = _DEFAULT_MODEL):
        import comfy.model_management as mm

        from skintokens.comfy_model import wrap_for_comfy
        from skintokens.model_loader import load_model

        offload_device = mm.unet_offload_device()
        bundle = load_model(
            device=offload_device,  # load to CPU; ComfyUI moves it to GPU on demand
            tokenrig_ckpt=MODELS[model],
            # models_dir=None -> shared HF cache (respects HF_HOME); auto-downloads.
        )
        return (wrap_for_comfy(bundle),)


class SkinTokensRig:
    """Rig a mesh (native MESH or File3D) -> rigged skinned glb (FILE_3D_GLB)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SKINTOKENS_MODEL",),
                "mesh": (MESH_OR_FILE3D_TYPES,),
                "filename_prefix": ("STRING", {"default": "skintokens_rigged"}),
                "relabel": ("BOOLEAN", {"default": True}),
                "convention": (CONVENTIONS, {"default": "Mixamo"}),
                "relabel_fingers": ("BOOLEAN", {"default": True}),
                "clean_weights": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Repair geometrically-impossible skin weights (e.g. a hand "
                               "vertex leaking onto a toe) after rigging. Fixes 'exploding "
                               "hand' tearing during animation. Deterministic.",
                }),
            },
            "optional": {
                "use_transfer": ("BOOLEAN", {"default": True}),
                "use_postprocess": ("BOOLEAN", {"default": False}),
                "use_skeleton": ("BOOLEAN", {"default": False}),
                "top_k": ("INT", {"default": 5, "min": 1, "max": 200}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05}),
                "repetition_penalty": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 5.0, "step": 0.1}),
                "num_beams": ("INT", {"default": 10, "min": 1, "max": 50}),
            },
        }

    RETURN_TYPES = ("FILE_3D_GLB",)
    RETURN_NAMES = ("rigged_glb",)
    FUNCTION = "rig"
    CATEGORY = CATEGORY

    def rig(
        self,
        model,
        mesh,
        filename_prefix: str = "skintokens_rigged",
        relabel: bool = True,
        convention: str = "Mixamo",
        relabel_fingers: bool = True,
        clean_weights: bool = True,
        use_transfer: bool = True,
        use_postprocess: bool = False,
        use_skeleton: bool = False,
        top_k: int = 5,
        top_p: float = 0.95,
        temperature: float = 1.0,
        repetition_penalty: float = 2.0,
        num_beams: int = 10,
    ) -> Tuple:
        from skintokens import comfy_types, glb_io, infer, relabel as relabel_mod

        bundle = model.prepare()  # ComfyUI moves weights to the compute device
        generate_kwargs = dict(
            top_k=int(top_k),
            top_p=float(top_p),
            temperature=float(temperature),
            repetition_penalty=float(repetition_penalty),
            num_beams=int(num_beams),
        )

        # A native MESH has no source file, so texture transfer / existing-armature
        # skinning (both need the original glb) only apply to a File3D input.
        in_path = None
        if comfy_types.is_comfy_mesh(mesh):
            verts, faces, normals = comfy_types.comfy_mesh_to_arrays(mesh)
            rigged = infer.rig_mesh(
                bundle, verts, faces, normals=normals,
                generate_kwargs=generate_kwargs, use_postprocess=use_postprocess,
            )
        elif comfy_types.is_file3d(mesh):
            in_path = comfy_types.file3d_to_path(mesh)
            rigged = infer.rig_glb(
                bundle, in_path, generate_kwargs=generate_kwargs,
                use_skeleton=use_skeleton, use_postprocess=use_postprocess,
            )
        else:
            raise TypeError(
                f"SkinTokensRig: unsupported mesh input {type(mesh)!r} "
                "(expected a ComfyUI MESH or a FILE_3D_* object)"
            )

        if relabel:
            relabel_mod.relabel_asset(
                rigged,
                convention=_CONVENTION_KEY[convention],
                with_fingers=relabel_fingers,
            )

        out_path = _unique_output_path(filename_prefix + ".glb")
        if use_transfer and in_path is not None:
            from skintokens.transfer import transfer_rigging

            transfer_rigging(rigged, in_path, out_path)
        else:
            glb_io.export_glb(rigged, out_path)

        # Deterministic post-rig cleanup: strip cross-body weight contamination.
        if clean_weights:
            from skintokens import weight_repair

            stats = weight_repair.repair_glb(out_path, out_path)
            if stats.n_influences_removed:
                print(
                    f"[SkinTokens] Rig: cleaned {stats.n_vertices_repaired} vertices "
                    f"({stats.repaired_fraction * 100:.2f}%), removed "
                    f"{stats.n_influences_removed} stray influences.",
                    flush=True,
                )
        return (comfy_types.make_file3d(out_path, "glb"),)


class SkinTokensRelabel:
    """Structurally relabel an already-rigged glb's joints (Mixamo/UE5). No model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "glb": (RIGGED_FILE3D_TYPES,),
                "filename_prefix": ("STRING", {"default": "skintokens_relabeled"}),
                "convention": (CONVENTIONS, {"default": "Mixamo"}),
                "relabel_fingers": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("FILE_3D_GLB",)
    RETURN_NAMES = ("relabeled_glb",)
    FUNCTION = "relabel"
    CATEGORY = CATEGORY

    def relabel(
        self,
        glb,
        filename_prefix: str = "skintokens_relabeled",
        convention: str = "Mixamo",
        relabel_fingers: bool = True,
    ) -> Tuple:
        from skintokens import comfy_types, glb_io

        if not comfy_types.is_file3d(glb):
            raise TypeError(
                f"SkinTokensRelabel: expected a FILE_3D_* object, got {type(glb)!r}"
            )
        in_path = comfy_types.file3d_to_path(glb)

        out_path = _unique_output_path(filename_prefix + ".glb")
        glb_io.relabel_glb(
            in_path,
            out_path,
            convention=_CONVENTION_KEY[convention],
            with_fingers=relabel_fingers,
        )
        return (comfy_types.make_file3d(out_path, "glb"),)


class SkinTokensCleanWeights:
    """Repair geometrically-impossible skin weights on a rigged glb. No model.

    Removes influences that blend across the skeleton (e.g. a hand vertex weighted
    to a toe) and renormalizes, fixing the "exploding hand" tearing without losing
    finger articulation. Deterministic, offline; a sibling of SkinTokensRelabel.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "glb": (RIGGED_FILE3D_TYPES,),
                "filename_prefix": ("STRING", {"default": "skintokens_cleaned"}),
            },
            "optional": {
                "min_hops": ("INT", {
                    "default": 5, "min": 2, "max": 20,
                    "tooltip": "Flag an influence when a vertex is weighted to two bones "
                               "at least this many skeleton-graph hops apart. 5 is above "
                               "the widest anatomical blend (leg<->leg via hips = 4).",
                }),
            },
        }

    RETURN_TYPES = ("FILE_3D_GLB",)
    RETURN_NAMES = ("cleaned_glb",)
    FUNCTION = "clean"
    CATEGORY = CATEGORY

    def clean(self, glb, filename_prefix: str = "skintokens_cleaned", min_hops: int = 5) -> Tuple:
        from skintokens import comfy_types, weight_repair

        if not comfy_types.is_file3d(glb):
            raise TypeError(
                f"SkinTokensCleanWeights: expected a FILE_3D_* object, got {type(glb)!r}"
            )
        in_path = comfy_types.file3d_to_path(glb)
        out_path = _unique_output_path(filename_prefix + ".glb")
        stats = weight_repair.repair_glb(in_path, out_path, min_hops=int(min_hops))
        print(
            f"[SkinTokens] CleanWeights: repaired {stats.n_vertices_repaired} vertices "
            f"({stats.repaired_fraction * 100:.2f}%), removed {stats.n_influences_removed} "
            f"stray influences. Top contaminants: "
            + ", ".join(f"{b}({c})" for b, c in
                        sorted(stats.removed_by_bone.items(), key=lambda kv: -kv[1])[:5]),
            flush=True,
        )
        return (comfy_types.make_file3d(out_path, "glb"),)


NODE_CLASS_MAPPINGS = {
    "SkinTokensLoader": SkinTokensLoader,
    "SkinTokensRig": SkinTokensRig,
    "SkinTokensRelabel": SkinTokensRelabel,
    "SkinTokensCleanWeights": SkinTokensCleanWeights,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SkinTokensLoader": "SkinTokens Loader",
    "SkinTokensRig": "SkinTokens Rig",
    "SkinTokensRelabel": "SkinTokens Relabel",
    "SkinTokensCleanWeights": "SkinTokens Clean Weights",
}
