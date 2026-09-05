# Running tests on the GPU server

Most of this project is tested on a laptop with no GPU. A few tests can only run
on the GPU box because they need the full SkinTokens model (~14 GB) on CUDA.
Those are tagged with the `server` pytest marker. This directory has one script
to run them.

**These tests do not need ComfyUI.** They only need a GPU and a CUDA-enabled
`torch`. ComfyUI happens to run on the same box (in Docker), but it is not
involved here — ComfyUI *integration* is a separate, later gate (Gate F).

## TL;DR

From a checkout of this repo **on the GPU host**:

```bash
./scripts/server/run-server-tests.sh
```

That's it. The script uses `uv` to build an isolated venv with a CUDA `torch`,
checks the GPU is visible, downloads the weights on first run, and runs the
server tests. Nothing global is touched; ComfyUI's container is left alone.

## Requirements on the host

- `uv` and `python` (both already installed).
- The NVIDIA driver — check with `nvidia-smi`. The CUDA `torch` wheel bundles its
  own CUDA runtime, so only the driver needs to be on the host (the same driver
  Docker's `--gpus` uses). No system CUDA toolkit required.

## What it does (idempotent — re-run freely)

1. Creates `.venv-server/` via `uv` (once).
2. Installs a CUDA `torch` wheel (`uv` auto-detects the build from your driver),
   then `requirements.txt` + `pytest`, then this package (editable).
3. Verifies `torch` imports and `torch.cuda.is_available()`; prints the GPU name.
4. Runs `pytest -m server` with `SKINTOKENS_RUN_MODEL=1`. The first run downloads
   the weights (~14 GB) via `huggingface_hub` into `./models/`.

## Configuration (environment variables)

| Variable                | Default        | Meaning                                                        |
|-------------------------|----------------|----------------------------------------------------------------|
| `TORCH_BACKEND`         | `auto`         | CUDA wheel selection for `uv` (e.g. `cu124`; `cpu` to force CPU). |
| `VENV`                  | `.venv-server` | Where to create the server venv.                               |
| `SKINTOKENS_DEVICE`     | `cuda`         | Torch device (e.g. `cuda:0`).                                   |
| `SKINTOKENS_MODELS_DIR` | `./models`     | Where the weights are cached (via `load_model`).               |

If `uv`'s auto CUDA detection picks the wrong build, set `TORCH_BACKEND` to match
your driver, e.g. `TORCH_BACKEND=cu124 ./scripts/server/run-server-tests.sh`.

## Extra pytest arguments

Anything after the script name is passed straight to pytest:

```bash
./scripts/server/run-server-tests.sh tests/test_infer.py::test_rig_mesh_end_to_end -vv
```

## HuggingFace access

If the `VAST-AI/SkinTokens` repo is gated, authenticate first:

```bash
huggingface-cli login          # or: export HF_TOKEN=hf_xxx
```

## What runs today

Right now the only `server` test is `test_rig_mesh_end_to_end` — a **smoke
test**: it rigs a synthetic box mesh and asserts the model returns a skeleton
(`parents`, `J > 0`) and per-vertex skin weights. It proves inference *runs* and
returns a plausibly-shaped result.

It is **not** full Gate B parity (comparing joints/parents/weights against
upstream `demo.py`) — that comes later. If this smoke test throws, the traceback
is exactly what's needed to fix the inference wiring, so capture and share it.
