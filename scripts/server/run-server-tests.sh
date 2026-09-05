#!/usr/bin/env bash
#
# Run the tests that need a GPU + the full model weights (the "server" marker).
#
# These tests do NOT need ComfyUI. They need a GPU and a CUDA-enabled torch.
# This runs directly on the host using uv: it creates an isolated venv, installs
# a CUDA torch wheel + the runtime deps, checks the GPU is visible, and runs the
# server tests. Nothing global is touched; ComfyUI's container is not involved.
#
# Quick start (from the repo checkout on the GPU host):
#
#   ./scripts/server/run-server-tests.sh
#
# Idempotent — safe to run repeatedly. See scripts/server/README.md.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# --- configuration (override via environment) ---
VENV="${VENV:-$REPO_ROOT/.venv-server}"
export SKINTOKENS_DEVICE="${SKINTOKENS_DEVICE:-cuda}"
# TORCH_BACKEND: uv auto-detects the CUDA build from the driver by default.
# Override if detection is wrong, e.g. TORCH_BACKEND=cu124 (or cpu to force CPU).
TORCH_BACKEND="${TORCH_BACKEND:-auto}"

echo "==> repo:    $REPO_ROOT"
echo "==> venv:    $VENV"
echo "==> device:  $SKINTOKENS_DEVICE"
echo "==> torch:   backend=$TORCH_BACKEND"

# --- venv (via uv) ---
if [ ! -d "$VENV" ]; then
  echo "==> creating venv"
  uv venv "$VENV"
fi

# --- deps: CUDA torch first, then runtime deps + pytest, then the package ---
# uv --torch-backend picks the right CUDA wheel from the host driver.
echo "==> installing CUDA torch + deps into $VENV"
VIRTUAL_ENV="$VENV" uv pip install --python "$VENV/bin/python" \
  torch --torch-backend="$TORCH_BACKEND"
VIRTUAL_ENV="$VENV" uv pip install --python "$VENV/bin/python" \
  -r requirements.txt pytest
VIRTUAL_ENV="$VENV" uv pip install --python "$VENV/bin/python" -e . --no-deps

# --- sanity check: torch must see the GPU ---
echo "==> checking torch / CUDA"
"$VENV/bin/python" - <<'PY'
import sys
try:
    import torch
except Exception as e:
    sys.exit(f"ERROR: torch not importable: {e}")
print(f"    torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("ERROR: torch cannot see a GPU. Check the NVIDIA driver (nvidia-smi), "
             "or set TORCH_BACKEND=cuXXX to match it.")
print(f"    device: {torch.cuda.get_device_name(0)}")
PY

# --- run the server-only tests ---
# SKINTOKENS_RUN_MODEL=1 un-skips them; -m server selects only those tests.
# First run downloads the weights (~14 GB) via huggingface_hub into ./models/.
echo "==> running server tests (first run downloads ~14 GB of weights)"
SKINTOKENS_RUN_MODEL=1 "$VENV/bin/python" -m pytest -m server -v -s "$@"
