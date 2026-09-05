# ComfyUI-SkinTokens-NoBlender

Auto-rigging and skinning nodes for ComfyUI - **no Blender / `bpy` required**.

Wraps the [SkinTokens / TokenRig](https://huggingface.co/VAST-AI/SkinTokens)
auto-rigger: a static mesh goes in, a **rigged `.glb`** (skeleton + skin weights)
comes out, with an optional step that relabels the skeleton with Mixamo or UE5
bone names. All glb import/export is pure Python (`trimesh` + `pygltflib`) - the
upstream project needs Blender for this, and this pack removes that dependency so
it runs on a headless GPU server.

> **Status:** feature-complete. Rigging, relabeling, skinned-glb export, texture
> transfer, the optional voxel post-process, and skin-only-against-an-existing-
> skeleton all work and are tested. The one open item - animating a rigged output
> in **Kimodo** - is waiting on a fix to the Kimodo node itself, not on this pack.

## Nodes

| Node | Does |
|------|------|
| **SkinTokens Loader** | Loads the TokenRig model (auto-downloaded from HuggingFace on first use) and wraps it for ComfyUI's VRAM manager. Pick a checkpoint from the dropdown - no path to configure. |
| **SkinTokens Rig** | Static mesh → rigged skinned `.glb`. Accepts a native `MESH` (from Trellis/remesh) or a `File3D` (`FILE_3D_GLB`/`GLTF`/`OBJ`/`STL`, from Load3D). Optional relabel, texture transfer, voxel post-process, skin-only-against-existing-skeleton. |
| **SkinTokens Relabel** | Standalone structural relabel of an already-rigged glb (Mixamo/UE5). Model-free. |

The rigged output is a `FILE_3D_GLB`, so it drops straight into ComfyUI core's
`Preview3DAdvanced` (which has a **showSkeleton** toggle) and `Save3D` - no custom
preview node needed.

## Install

**Via ComfyUI Manager** (recommended): search for *SkinTokens-NoBlender* and
install, then restart ComfyUI.

**Manually:**

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/stevelittlefish/ComfyUI-SkinTokens-NoBlender
pip install -r ComfyUI-SkinTokens-NoBlender/requirements.txt
```

`torch` is **not** pinned - ComfyUI provides it. `flash-attn` is optional (the
code falls back to standard attention). The model weights (~14 GB) are **not**
bundled; they download from `VAST-AI/SkinTokens` to your HuggingFace cache
(`$HF_HOME`) the first time the Loader runs. A gated/private model needs the usual
HF auth (`huggingface-cli login` or `$HF_TOKEN`).

## Usage

Load the example workflow at
[`examples/skintokens_rig.json`](examples/skintokens_rig.json):

```
Load3D ─▶ SkinTokens Rig ─▶ Preview3DAdvanced (showSkeleton)
             ▲
SkinTokens Loader
```

**SkinTokens Rig options:**

- **relabel** / **convention** (`Mixamo`/`UE5`) / **relabel_fingers** - rename the
  generated humanoid bones to a standard convention (structural, topology-driven).
- **use_transfer** (default on) - attach the rig to the *original* glb, preserving
  its materials, textures, UVs and scale. Only applies to a `File3D` input (a
  native `MESH` has no source file). Off = write the rigged proxy with a flat
  default material.
- **use_postprocess** (default on) - refine the skin weights with a voxel-heat
  heuristic. Slower, but gives cleaner deformation; turn off for a faster pass.
- **use_skeleton** (default off) - skin-only: keep an armature already present in
  the input glb and only predict skin weights for it, instead of generating a new
  skeleton.
- **top_k / top_p / temperature / repetition_penalty / num_beams** - generation
  params (defaults match upstream: 5 / 0.95 / 1.0 / 2.0 / 10).

## How it works

- **Import** (`skintokens/glb_io.py`): `trimesh` loads the mesh; multi-part scenes
  are concatenated with correct face offsets. `import_skeleton` reads an existing
  armature for the skin-only path.
- **Inference** (`skintokens/infer.py`): builds the upstream `Asset`, runs
  `predict_step` on the model (moved to the GPU by ComfyUI on demand, evicted when
  the GPU is needed elsewhere).
- **Export** (`skintokens/glb_io.py`): writes a skinned glb - translation-only
  joint nodes, `translate(-joint)` inverse bind matrices, top-4 `JOINTS_0`/
  `WEIGHTS_0` - so the file poses correctly in any glTF viewer.
- **Relabel** (`skintokens/relabel.py`): topology-driven humanoid recognizer.
- **Transfer** (`skintokens/transfer.py`): similarity-align the proxy to the
  original mesh, nearest-neighbour skin, inject skeleton+skin into the target
  glTF in place so all materials survive.

The upstream torch core is vendored under `skintokens/vendor/` (see
`skintokens/vendor/UPSTREAM.md` for the commit and the portability edits).

## Development

Local dev is GPU-less and Blender-less; the highest-risk paths (glb export,
relabeler, transfer) are covered by pure-Python `pytest`:

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements-dev.txt   # CPU torch + the runtime stack + pytest
uv pip install -e . --no-deps
python -m pytest -q                        # ~84 passed, ~4 `server` skipped
```

The `.venv` is **not** in git, so a fresh clone (or a machine you haven't set up
yet) starts with no dependencies - run the block above before `pytest`, or it
can't even import `torch`/`transformers`. No `uv`? `python -m pip install -r
requirements-dev.txt` inside the activated venv works too.

Tests marked `server` need the GPU + full weights and are skipped locally; see
`scripts/server/`. The build is spec-driven - see [`spec/`](spec/) for the design
and [`spec/TODO.md`](spec/TODO.md) for the phased task list and validation gates.

## Releasing (Comfy Registry)

Publishing is automated. A GitHub Action
([`.github/workflows/publish.yml`](.github/workflows/publish.yml)) publishes to
[registry.comfy.org](https://registry.comfy.org) on any push to `main` that
changes `pyproject.toml`. ComfyUI Manager then picks the new version up from the
registry on its own sync cadence.

To cut a release:

1. Bump `version` in `pyproject.toml` (e.g. `0.0.1` -> `0.0.2`). **This is the
   trigger** - the registry rejects re-publishing an existing version, so nothing
   publishes until the number changes.
2. Update the description (`[project] description`, the one-line text shown in
   Manager) and/or `README.md` (the registry page body) if needed.
3. Commit and push to `main`. Watch the run under the repo's **Actions** tab.

The registry API key lives only as the `REGISTRY_ACCESS_TOKEN` repo secret (never
in the repo). Rotate it on registry.comfy.org, then
`gh secret set REGISTRY_ACCESS_TOKEN` - nothing in the workflow changes.

## Credits

- [VAST-AI/SkinTokens](https://huggingface.co/VAST-AI/SkinTokens) - the model and
  torch reference implementation.
- Prior-art ComfyUI node pack: `Aero-Ex/ComfyUI-SkinTokens`.

## License

See [LICENSE](LICENSE).
