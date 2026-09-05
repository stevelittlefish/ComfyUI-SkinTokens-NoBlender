"""Load the SkinTokens / TokenRig model for inference.

Wraps the vendored upstream loader (``TokenRig.load_from_system_checkpoint``) and
the HF weight download logic (from upstream ``download.py``), and fixes up the
config paths so the model loads regardless of the working directory or where the
weights were downloaded.

Nothing here is GPU-specific: pass whatever device you want. ComfyUI wiring
(``comfy.model_management``) comes in Phase 5. The actual run of the ~14 GB model
is a server-side gate (Gate B); this module is the code that gets it into memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch

from .vendor.data.transform import Transform
from .vendor.model.tokenrig import TokenRig
from .vendor.tokenizer.parse import get_tokenizer
from .vendor.tokenizer.spec import Tokenizer

# --- upstream HF locations (see references/SkinTokens/download.py) ---
HF_REPO_ID = "VAST-AI/SkinTokens"
DEFAULT_TOKENRIG_CKPT = "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt"
DEFAULT_SKIN_VAE_CKPT = "experiments/skin_vae_2_10_32768/last.ckpt"
QWEN_REPO_ID = "Qwen/Qwen3-0.6B"

DeviceLike = Union[str, torch.device]


@dataclass
class SkinTokensModel:
    """Everything inference needs: the model plus its tokenizer and transform."""

    model: TokenRig
    tokenizer: Tokenizer
    transform: Transform
    device: torch.device
    dtype: torch.dtype

    def to(self, device: DeviceLike) -> "SkinTokensModel":
        self.device = torch.device(device)
        self.model.to(self.device)
        return self


def default_models_dir() -> Path:
    """Where weights are cached when the caller does not specify a directory.

    Phase 5 will route this through ComfyUI's ``folder_paths``; until then it is a
    ``models/`` directory next to the repo (override with ``models_dir=``).
    """
    return Path(__file__).resolve().parent.parent / "models"


def download_weights(models_dir: Optional[Path] = None) -> dict:
    """Download the TokenRig ckpt, the skin-VAE ckpt, and the Qwen3 config.

    Weights are large (~14 GB) and are NOT bundled; this mirrors upstream
    ``download.py``. Returns absolute paths to the two checkpoints and the Qwen
    config directory. Requires network access (and HF auth if the repo is gated).
    """
    from huggingface_hub import hf_hub_download, snapshot_download

    models_dir = Path(models_dir) if models_dir is not None else default_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)

    tokenrig_path = hf_hub_download(
        repo_id=HF_REPO_ID, filename=DEFAULT_TOKENRIG_CKPT, local_dir=str(models_dir)
    )
    skin_vae_path = hf_hub_download(
        repo_id=HF_REPO_ID, filename=DEFAULT_SKIN_VAE_CKPT, local_dir=str(models_dir)
    )
    # Config only (no weights) — the transformer weights come from the ckpt.
    qwen_dir = snapshot_download(
        repo_id=QWEN_REPO_ID,
        local_dir=str(models_dir / "Qwen3-0.6B"),
        ignore_patterns=["*.bin", "*.safetensors"],
    )
    return {
        "tokenrig": str(Path(tokenrig_path).resolve()),
        "skin_vae": str(Path(skin_vae_path).resolve()),
        "qwen_dir": str(Path(qwen_dir).resolve()),
    }


def load_model(
    models_dir: Optional[Path] = None,
    device: DeviceLike = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    download: bool = True,
) -> SkinTokensModel:
    """Load TokenRig (+ skin-VAE + Qwen backbone) ready for ``predict_step``.

    The checkpoint bakes in *relative* paths to the skin-VAE ckpt and the Qwen
    config (resolved against the original training CWD). We rewrite those to the
    downloaded absolute locations before constructing the model, so loading works
    from any directory.
    """
    device = torch.device(device)

    if download:
        paths = download_weights(models_dir)
    else:
        # Assume already downloaded into the standard layout.
        root = Path(models_dir) if models_dir is not None else default_models_dir()
        paths = {
            "tokenrig": str((root / DEFAULT_TOKENRIG_CKPT).resolve()),
            "skin_vae": str((root / DEFAULT_SKIN_VAE_CKPT).resolve()),
            "qwen_dir": str((root / "Qwen3-0.6B").resolve()),
        }

    ckpt = torch.load(paths["tokenrig"], map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]
    model_config = dict(hp["model_config"])
    transform_config = hp["transform_config"]
    tokenizer_config = hp["tokenizer_config"]

    # Point the skin-VAE and Qwen config at the downloaded absolute paths.
    model_config["pretrained_vae"] = paths["skin_vae"]
    llm_cfg = dict(model_config["llm"])
    llm_cfg["pretrained_model_name_or_path"] = paths["qwen_dir"]
    model_config["llm"] = llm_cfg

    # Reproduces ModelSpec.load_from_system_checkpoint, but with our patched
    # config and only one torch.load of the (large) checkpoint.
    model = TokenRig(
        model_config=model_config,
        transform_config=transform_config,
        tokenizer_config=tokenizer_config,
    )
    state_dict = {}
    for k, v in ckpt["state_dict"].items():
        k = k.replace("_orig_mod.", "")
        if k.startswith("model."):
            k = k[len("model.") :]
        state_dict[k] = v
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[SkinTokens] Missing keys when loading: {missing}")
    if unexpected:
        print(f"[SkinTokens] Unexpected keys when loading: {unexpected}")
    model.on_load_checkpoint(ckpt)

    model = model.to(device).to(dtype)
    model.eval()

    tokenizer = get_tokenizer(**model.tokenizer_config)
    transform = Transform.parse(**model.transform_config["predict_transform"])

    return SkinTokensModel(
        model=model, tokenizer=tokenizer, transform=transform, device=device, dtype=dtype
    )
