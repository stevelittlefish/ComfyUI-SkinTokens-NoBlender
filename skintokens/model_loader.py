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

# Vendored skeleton part-order configs (upstream configs/skeleton/*.yaml). The
# checkpoint's transform config references these by a CWD-relative path
# (./configs/skeleton/*.yaml); we rewrite those to here at load time.
VENDOR_CONFIGS_DIR = Path(__file__).resolve().parent / "vendor" / "configs"


def _rewrite_skeleton_paths(transform_config: dict) -> None:
    """Point every transform's Order.skeleton_path at the vendored yaml files.

    Mutates ``transform_config`` in place. Matches by filename, so the original
    relative paths in the checkpoint no longer depend on the working directory.
    """
    for key in ("predict_transform", "validate_transform", "train_transform"):
        section = transform_config.get(key)
        if not isinstance(section, dict):
            continue
        order = section.get("order")
        if not isinstance(order, dict):
            continue
        skeleton_path = order.get("skeleton_path")
        if not isinstance(skeleton_path, dict):
            continue
        for cls_name, path in list(skeleton_path.items()):
            resolved = VENDOR_CONFIGS_DIR / "skeleton" / Path(str(path)).name
            skeleton_path[cls_name] = str(resolved.resolve())


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


def resolve_weights(
    models_dir: Optional[Path] = None,
    download: bool = True,
    tokenrig_ckpt: str = DEFAULT_TOKENRIG_CKPT,
) -> dict:
    """Locate the TokenRig ckpt, skin-VAE ckpt, and Qwen3 config, downloading if needed.

    Weights are large (~14 GB) and are NOT bundled; this mirrors upstream
    ``download.py``. Returns absolute paths to the two checkpoints and the Qwen
    config directory.

    ``models_dir``:
      - ``None`` (default): use the standard HuggingFace cache (``$HF_HOME`` /
        ``~/.cache/huggingface``), shared with other HF tools — no duplicate copy.
      - a path: download a private copy into that directory (e.g. a
        ComfyUI-visible ``models/skintokens`` dir).

    ``download``: when ``False``, resolve from cache/disk only (``local_files_only``)
    and never hit the network — for airgapped or pre-staged installs.

    HF auth (for gated repos) is picked up automatically from the ambient token
    (``huggingface-cli login`` / ``$HF_TOKEN``); we never handle it directly.
    """
    from huggingface_hub import hf_hub_download, snapshot_download

    local_files_only = not download
    # When models_dir is given, download copies into it; otherwise use the HF cache.
    if models_dir is not None:
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        ckpt_kwargs = {"local_dir": str(models_dir)}
        qwen_kwargs = {"local_dir": str(models_dir / "Qwen3-0.6B")}
    else:
        ckpt_kwargs = {}
        qwen_kwargs = {}

    tokenrig_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=tokenrig_ckpt,
        local_files_only=local_files_only,
        **ckpt_kwargs,
    )
    skin_vae_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=DEFAULT_SKIN_VAE_CKPT,
        local_files_only=local_files_only,
        **ckpt_kwargs,
    )
    # Config only (no weights) — the transformer weights come from the ckpt.
    qwen_dir = snapshot_download(
        repo_id=QWEN_REPO_ID,
        ignore_patterns=["*.bin", "*.safetensors"],
        local_files_only=local_files_only,
        **qwen_kwargs,
    )
    return {
        "tokenrig": str(Path(tokenrig_path).resolve()),
        "skin_vae": str(Path(skin_vae_path).resolve()),
        "qwen_dir": str(Path(qwen_dir).resolve()),
    }


# Backwards-compatible alias.
def download_weights(models_dir: Optional[Path] = None) -> dict:
    return resolve_weights(models_dir=models_dir, download=True)


def load_model(
    models_dir: Optional[Path] = None,
    device: DeviceLike = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    download: bool = True,
    tokenrig_ckpt: str = DEFAULT_TOKENRIG_CKPT,
) -> SkinTokensModel:
    """Load TokenRig (+ skin-VAE + Qwen backbone) ready for ``predict_step``.

    The checkpoint bakes in *relative* paths to the skin-VAE ckpt and the Qwen
    config (resolved against the original training CWD). We rewrite those to the
    resolved absolute locations before constructing the model, so loading works
    from any directory.

    ``models_dir`` / ``download`` behave as in :func:`resolve_weights` — default
    is the shared HF cache with auto-download.
    """
    device = torch.device(device)

    paths = resolve_weights(
        models_dir=models_dir, download=download, tokenrig_ckpt=tokenrig_ckpt
    )

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
    _rewrite_skeleton_paths(model.transform_config)
    transform = Transform.parse(**model.transform_config["predict_transform"])

    return SkinTokensModel(
        model=model, tokenizer=tokenizer, transform=transform, device=device, dtype=dtype
    )
