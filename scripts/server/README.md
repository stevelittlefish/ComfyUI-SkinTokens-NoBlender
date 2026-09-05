# Running tests on the GPU server

Most of this project is tested on a laptop with no GPU. A few tests can only run
on the GPU box because they need the full SkinTokens model (~14 GB) on CUDA.
Those are tagged with the `server` pytest marker. This directory has one script
to run them.

The GPU server is https://comfy.seaslug.ai/ (ComfyUI). These tests run in a
shell on that machine, not through the ComfyUI web UI.

## TL;DR

From a checkout of this repo **on the server**:

```bash
PYTHON=/path/to/comfyui/python ./scripts/server/run-server-tests.sh
```

That's it. The script sets up an isolated environment, checks the GPU is
visible, downloads the weights on first run, and runs the server tests.

## What `PYTHON` should be

Point `PYTHON` at **ComfyUI's own Python interpreter**. The script builds its
venv with `--system-site-packages`, so it reuses ComfyUI's CUDA-enabled `torch`
(and `flash-attn`, if installed) instead of downloading a second multi-GB torch
build. Only the extra SkinTokens deps (`transformers`, `trimesh`, …) and
`pytest` get installed into the venv — ComfyUI's environment is left untouched.

To find ComfyUI's Python, on the server:

```bash
# examples — the real path depends on how ComfyUI was installed
ls /opt/ComfyUI/venv/bin/python
# or, if ComfyUI runs under conda/uv, use that env's python
```

If you omit `PYTHON`, it defaults to `python3`. That only works if that
interpreter already has a CUDA `torch` — otherwise the script stops with a clear
error telling you to set `PYTHON`.

## What it does

1. Creates `.venv-server/` (once), inheriting the base interpreter's packages.
2. Installs `requirements.txt` + `pytest` + this package (editable) into it.
   **`torch` is not installed here** — it comes from the base env.
3. Verifies `torch` imports and `torch.cuda.is_available()` is true; prints the
   GPU name.
4. Runs `pytest -m server` with `SKINTOKENS_RUN_MODEL=1`. The first run
   downloads the weights (~14 GB) via `huggingface_hub` into `./models/`.

The script is **idempotent** — run it as often as you like. Setup steps are
skipped when already done.

## Configuration (environment variables)

| Variable                  | Default        | Meaning                                              |
|---------------------------|----------------|------------------------------------------------------|
| `PYTHON`                  | `python3`      | Interpreter to base the venv on (use ComfyUI's).     |
| `VENV`                    | `.venv-server` | Where to create the server venv.                     |
| `SKINTOKENS_DEVICE`       | `cuda`         | Torch device (e.g. `cuda:0`).                        |
| `SKINTOKENS_MODELS_DIR`   | `./models`     | Where the weights are cached (via `load_model`).     |

## Extra pytest arguments

Anything after the script name is passed straight to pytest:

```bash
# run one specific test, extra-verbose
PYTHON=/path/to/comfyui/python ./scripts/server/run-server-tests.sh \
  tests/test_infer.py::test_rig_mesh_end_to_end -vv
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
