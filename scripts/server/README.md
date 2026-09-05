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
   the weights (~14 GB) via `huggingface_hub` into the shared HF cache
   (`$HF_HOME` / `~/.cache/huggingface`) — the same default the real ComfyUI node
   uses. To reuse a copy you already downloaded elsewhere, set
   `SKINTOKENS_MODELS_DIR` to that directory.

## Configuration (environment variables)

| Variable                | Default        | Meaning                                                        |
|-------------------------|----------------|----------------------------------------------------------------|
| `TORCH_BACKEND`         | `auto`         | CUDA wheel selection for `uv` (e.g. `cu124`; `cpu` to force CPU). |
| `VENV`                  | `.venv-server` | Where to create the server venv.                               |
| `SKINTOKENS_DEVICE`     | `cuda`         | Torch device (e.g. `cuda:0`).                                   |
| `SKINTOKENS_MODELS_DIR` | *(unset)*      | Unset = use the shared HF cache (`$HF_HOME`). Set it to reuse a local copy and avoid re-downloading (e.g. `SKINTOKENS_MODELS_DIR=./models`). |

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

`pytest -m server` currently runs, end to end on the GPU:

- `test_rig_mesh_end_to_end` (test_infer.py) — rigs a synthetic box; asserts a
  skeleton (`parents`, `J > 0`) + per-vertex skin weights come back. Gate B smoke.
- `test_rig_glb_end_to_end` (test_infer.py) — rigs the committed sample mesh via
  the pure-Python importer + inference.
- `test_rig_glb_to_file_roundtrip` (test_infer.py) — full engine pipeline: glb in
  -> skinned glb out, re-importable, with JOINTS_0/WEIGHTS_0. Gate C server half.
- `test_rig_node_end_to_end` (test_nodes.py) — the **Phase 5 ComfyUI node glue**:
  `SkinTokensRig` with a File3D input -> rigged `FILE_3D_GLB`, exercising the
  MESH/File3D bridge + relabel + export around real inference (no ComfyUI needed).
  The sample is humanoid, so it also checks the relabeler emits `mixamorig:*`.

These use the committed sample mesh (`tests/fixtures/meshes/dummy.glb`), so there
is nothing to download first — no `references/pull.sh` needed.

### What this does NOT cover

- **Full Gate B parity** (comparing joints/parents/weights against upstream
  `demo.py` on the same mesh) — smoke only, for now.
- **Gate F (ComfyUI VRAM lifecycle)** — load-on-demand / offload / eviction with
  no leak. That runs *inside* ComfyUI (the `ModelPatcher` path in
  `skintokens/comfy_model.py`), which this script deliberately does not launch.
- **Gate E (end-to-end through Kimodo)** — the real animation acceptance test.

If any server test throws, the traceback is exactly what's needed — capture and
share it.
