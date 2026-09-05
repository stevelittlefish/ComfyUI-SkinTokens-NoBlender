# ComfyUI Node Pack Spec

## Repository layout (proposed)

```
comfyui-skintokens/                 # repo root (name TBD by user)
├── __init__.py                     # NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS
├── pyproject.toml                  # ComfyUI-registry metadata (for Manager)
├── requirements.txt
├── nodes.py                        # the ComfyUI node classes
├── skintokens/                     # the pure-Python engine
│   ├── model_loader.py             # load TokenRig + wrap for model_management
│   ├── infer.py                    # mesh → Asset (predict)
│   ├── glb_io.py                   # trimesh import + pygltflib skinned export (03)
│   ├── relabel.py                  # structural relabeler (04)
│   ├── transfer.py                 # texture-preserving transfer (phase 2)
│   └── vendor/                     # copied-verbatim torch core from upstream SkinTokens
│       ├── model/ tokenizer/ data/ rig_package/ ...
├── reference/                      # NOT shipped; local read-only refs (gitignored or submodule)
│   ├── SkinTokens/                 # upstream torch repo
│   └── skin-tokens.cpp/            # C++ reference
├── spec/                           # THIS spec directory
└── examples/                       # sample workflows (.json) + a tiny test mesh
```

> `vendor/` vs a pip dependency: upstream SkinTokens is not published as a clean pip package,
> so vendor the needed `src/**` (minus bpy) into `skintokens/vendor/` and fix imports. Keep a
> note of the upstream commit it was copied from.

## Nodes

### 1. `SkinTokensLoader`
- **Inputs**: `model` (dropdown of available checkpoints; default the recommended GRPO one),
  optional generation params exposed later.
- **Output**: `SKINTOKENS_MODEL` (an object holding the model + tokenizer + transform, wrapped
  for `model_management`).
- Auto-downloads weights from HF on first use (like upstream `download.py`). Loads lazily.

### 2. `SkinTokensRig`
- **Inputs**:
  - `model`: `SKINTOKENS_MODEL`
  - `glb` : input mesh — accept a filepath string and/or bytes (see I/O note below)
  - `relabel`: BOOL (default True) + `convention`: dropdown `Mixamo`/`UE5` (see `04`/`08`)
  - `use_transfer`: BOOL (default True) — preserve original textures/scale (phase 2)
  - `use_postprocess`: BOOL (default False) — voxel skin heuristic
  - generation params: `top_k`, `top_p`, `temperature`, `repetition_penalty`, `num_beams`
    (defaults from upstream: 5 / 0.95 / 1.0 / 2.0 / 10)
  - optional `use_skeleton`: BOOL — skin-only against an existing armature in the input
- **Output**: rigged `glb` (filepath and/or a ComfyUI-appropriate type).
- **Behavior**: import mesh → predict Asset → (optional relabel) → export skinned glb
  (+ optional transfer).

### 3. (optional) `SkinTokensRelabel`
- Standalone node that takes a rigged glb and applies the relabel, for reuse on rigs produced
  elsewhere. Same logic as the toggle in `SkinTokensRig`.
- **`convention` dropdown**: `Mixamo` (default) / `UE5` / (extensible). See `04`/`08`.

### 4. (optional, UX) `SkinTokensPreview3D`
- Inline 3D preview of a rigged glb inside the ComfyUI graph (idea from prior art, `08`).
- `OUTPUT_NODE`; input `mesh_path` (STRING). Resolve the path relative to
  `folder_paths.get_output_directory()` / `get_input_directory()` and serve via ComfyUI's
  `/view?filename=...&type=output&subfolder=...` endpoint. Return
  `{"ui": {"skintokens_mesh": [url]}, "result": (mesh_path,)}`.
- Web extension under `web/js/` renders with **three.js + GLTFLoader** (and FBXLoader if we
  ever preview animation). **Bundle three.js and the loaders locally in the package — do NOT
  load from a CDN** (must work on an airgapped LAN). This is the one concrete fix over the
  prior-art previewer.

> **I/O type decision to make during build:** ComfyUI has no native "3D mesh" socket in the
> base install. Options: (a) pass filepaths as STRING (simplest, matches automation use), (b)
> define a custom `GLB`/`MESH` type, (c) integrate with an existing 3D node pack's type if the
> user already uses one. Given the automation goal, **STRING filepaths are the safe default**;
> add a custom type only if the user wants graph-native chaining. Confirm with the user.

## Model loading & VRAM (see `01`)

- Use `comfy.model_management.get_torch_device()` for placement — never hardcode `cuda:0`.
- Wrap the model so ComfyUI can offload/evict it (ModelPatcher or the version's idiomatic
  equivalent). Requirement is behavioral: **loads on demand, frees when GPU needed elsewhere,
  survives across runs without leaking.**
- Respect `--lowvram`/`--cpu` ComfyUI flags where reasonable.
- Validate the exact `model_management` API against the ComfyUI version pinned in
  `requirements.txt` / the target install — the API changes over time.

## Dependencies (`requirements.txt` — verify versions against upstream)

- `torch` (provided by ComfyUI; do not pin hard)
- `transformers>=4.57.0`, `diffusers>=0.35.0`, `einops`, `omegaconf`, `lightning`,
  `python-box`, `addict`, `huggingface_hub`, `numpy>=1.26.0`
- `trimesh` (mesh import), `pygltflib` (skinned export)
- **NOT** `bpy`, **NOT** `gradio`/`bottle`/`tornado` (server pieces dropped)
- `flash-attn` is optional and painful to build — make it optional, fall back to
  standard attention.

## Packaging for ComfyUI Manager

- `pyproject.toml` with the ComfyUI registry fields (`[tool.comfy]` publisher/name/etc.) per
  current Comfy registry docs.
- Node auto-registers via `NODE_CLASS_MAPPINGS` in `__init__.py`.
- Weights are NOT bundled; downloaded on first use to the ComfyUI models dir.
- Keep `reference/` out of the published package (gitignore or git submodule).

## Non-goals

- No animation (that is Kimodo, separate).
- No Blender, no subprocess engine, no C++ runtime dependency.
- No speed optimization work beyond "not gratuitously slow."
</content>
