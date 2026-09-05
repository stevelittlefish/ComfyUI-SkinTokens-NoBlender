# Scope: What to Port vs What to Copy

The single insight that bounds this project: **in upstream SkinTokens, Blender (`bpy`) is
confined to one file — `src/rig_package/parser/bpy.py` (`BpyParser`) — plus the thin
`src/server/bpy_server.py` that exposes it over HTTP.** Everything else (model, tokenizer,
transforms, FSQ, SkinVAE, surface-point sampling, normalization) is pure torch/numpy.

Therefore: **port `BpyParser`'s functionality to pure Python; copy the rest verbatim.**

## `bpy` touchpoints found in upstream (the complete list)

- `src/server/bpy_server.py` — bottle/tornado HTTP server exposing `/load`, `/export`,
  `/transfer`, `/ping`. **Delete/replace entirely** (we call functions directly, no server).
- `src/rig_package/parser/bpy.py` — `BpyParser.load`, `BpyParser` export path,
  `transfer_rigging`. **This is the thing to reimplement.**
- `src/data/datapath.py` — has a `BpyServerLazyAsset` (loads via `GET /load`) and a
  `BpyLazyAsset` (direct `BpyParser.load`). **Replace with a pure-Python loader asset.**
- `src/model/tokenrig.py:313` — imports `BpyParser` inside `predict_step` (make_asset).
  **Swap for the pure-Python exporter/asset builder.**
- `src/rig_package/info/asset.py:158` — comment documenting the Blender bone convention
  (bones extruded along local Y). **Not code to change, but the convention you MUST match on
  export.** See `03`.

## The 3 operations to reimplement

| Op (BpyParser) | Purpose | Python replacement | Difficulty |
|----------------|---------|--------------------|-----------|
| `load` (mesh) | glb/obj → vertices, faces, normals, mesh_names | `trimesh` | trivial |
| `load` (armature) | read existing skeleton: joint names, parents, bind matrices, lengths — only needed for `use_skeleton` mode | `pygltflib` (nodes + skins + accessors) | moderate |
| `export` | Asset (mesh + skeleton + dense skin weights) → skinned glb (one-frame rest pose) | `pygltflib` | **hard — see `03`** |
| `transfer` (`--use_transfer`) | bake generated rig onto the ORIGINAL glb to preserve textures/scale | `pygltflib`, carry original buffers/materials | moderate (phase 2) |

### Notes on `load`
- Read `BpyParser.load` to see exactly which `Asset` fields it populates
  (vertices, faces, normals, mesh_names, and — if an armature exists — joints/joint_names/
  parents/matrix_local/lengths/matrix_world). Reproduce the same `Asset` fields so the
  downstream torch code is unchanged.
- The heavy preprocessing (54k farthest-point sampling, normalization to [-1,1]) is **already
  pure torch/numpy** (in `src/model/skin_vae/.../autoencoder_kl_tripo2.py` `_sample_features`
  via `fps`, and `src/data/vertex_group.py`). Do NOT reimplement those — they are copied.
- Upstream removes a `glTF_not_exported` collection on import; irrelevant without Blender.

### Notes on `export`
- The model output `Asset` has: `joints` (J×3 positions), `parents` (J), dense skin weights
  (per-vertex × per-joint), `vertices`, `faces`. Export must:
  1. Build a glTF node per joint with correct local transforms (parent-relative).
  2. Create a `skin` with `inverseBindMatrices` consistent with those node transforms.
  3. Reduce dense weights to **top-4 influences per vertex** (glTF max), normalized —
     matching upstream's `group_per_vertex=4`.
  4. Write `JOINTS_0` / `WEIGHTS_0` vertex attributes.
  5. Store a **one-frame rest pose** so the file opens as a normal skinned glTF.
- **This must reproduce the Blender bone convention** or the mesh detaches when posed.
  `skin-tokens.cpp/src/glb.cpp` is the authoritative spec. See `03`.

### Notes on `transfer` (phase 2, don't skip permanently)
- Purpose: keep the original mesh's textures/materials/scale on the rigged output. Important
  because the whole point is glb-in → rigged-glb-out that still looks right.
- Approach: read the original glb (materials, images, UVs, buffers) with `pygltflib`, attach
  the generated skeleton + skin weights to it, write back. Reference: upstream
  `transfer_rigging` and `skin-tokens.cpp` handling of embedded materials/textures.

## What to copy verbatim from upstream (do NOT rewrite)

- `src/model/**` (TokenRig, Qwen3 backbone, Michelangelo mesh encoder, SkinVAE, FSQ)
- `src/tokenizer/**`
- `src/data/transform.py`, `src/data/order.py`, `src/data/vertex_group.py`,
  `src/data/augment.py` (minus any bpy-loader wiring in `dataset.py`/`datapath.py`)
- `src/rig_package/info/asset.py` (the `Asset` datatype; keep its math, drop bpy-only helpers)
- `download.py` logic for fetching checkpoints from HF

Adjust imports where they reference the removed bpy modules.
</content>
