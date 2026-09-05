#!/usr/bin/env bash
#
# Run the tests that need the GPU + full model weights (the "server" marker).
#
# Everything else in this project is tested on a laptop with no GPU. This script
# is for the bits that can only run on the GPU box (currently: the end-to-end
# inference smoke test). It is idempotent — safe to run repeatedly.
#
# Quick start (from anywhere on the server):
#
#   PYTHON=/path/to/comfyui/python ./scripts/server/run-server-tests.sh
#
# See scripts/server/README.md for details and configuration.
#
set -euo pipefail

# --- resolve repo root (this script lives in scripts/server/) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# --- configuration (override via environment) ---
# PYTHON: the interpreter to base the venv on. Point this at ComfyUI's Python so
#         the venv inherits its CUDA-enabled torch (and flash-attn, if present).
PYTHON="${PYTHON:-python3}"
# VENV: where to create the server venv (kept separate from the laptop .venv).
VENV="${VENV:-$REPO_ROOT/.venv-server}"
# SKINTOKENS_DEVICE: torch device to run inference on.
export SKINTOKENS_DEVICE="${SKINTOKENS_DEVICE:-cuda}"
# SKINTOKENS_MODELS_DIR: where to cache the ~14 GB weights (optional).
#   If set, it is passed through; load_model() defaults to ./models otherwise.

echo "==> repo:    $REPO_ROOT"
echo "==> python:  $PYTHON  ($($PYTHON --version 2>&1))"
echo "==> venv:    $VENV"
echo "==> device:  $SKINTOKENS_DEVICE"

# --- create the venv, inheriting the base interpreter's site-packages ---
# --system-site-packages is what lets us reuse ComfyUI's torch/CUDA/flash-attn
# instead of downloading a second multi-GB torch build.
if [ ! -d "$VENV" ]; then
  echo "==> creating venv (inheriting system site-packages)"
  "$PYTHON" -m venv --system-site-packages "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# --- install deps into the venv (not into ComfyUI's env) ---
# torch is intentionally NOT installed here: it must come from the base
# interpreter (ComfyUI). We only add the SkinTokens runtime deps + pytest.
echo "==> installing runtime deps + pytest (torch comes from the base env)"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt pytest
python -m pip install --quiet -e . --no-deps

# --- sanity check: torch must see CUDA, or the tests can't run ---
echo "==> checking torch / CUDA"
python - <<'PY'
import sys
try:
    import torch
except Exception as e:
    sys.exit(f"ERROR: torch is not importable in this env: {e}\n"
             "Point PYTHON= at ComfyUI's interpreter so the venv inherits its torch.")
print(f"    torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("ERROR: torch cannot see a GPU. Base the venv on ComfyUI's Python "
             "(PYTHON=/path/to/comfyui/python) or check your CUDA setup.")
print(f"    device: {torch.cuda.get_device_name(0)}")
PY

# --- run the server-only tests ---
# SKINTOKENS_RUN_MODEL=1 un-skips them; -m server selects only those tests.
# The first run downloads the weights (~14 GB) via huggingface_hub.
echo "==> running server tests (first run downloads ~14 GB of weights)"
SKINTOKENS_RUN_MODEL=1 python -m pytest -m server -v -s "$@"
