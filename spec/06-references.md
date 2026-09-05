# References & Setup for the Building Session

Place these under `reference/` in the new repo (gitignored or as submodules) so the building
Claude Code session can read them directly.

## 1. Upstream SkinTokens (torch) — the code to copy

- GitHub: https://github.com/VAST-AI-Research/SkinTokens
- HF weights: https://huggingface.co/VAST-AI/SkinTokens
  - `articulation_xl_quantization_256_token_4` — TokenRig autoregressive model (recommended;
    the GRPO-refined `grpo_1400.ckpt`).
  - `skin_vae_2_10_32768` — FSQ-CVAE (SkinTokens skin-weight tokenizer).
- HF Space (browser demo, for behavior reference): https://huggingface.co/spaces/VAST-AI/SkinTokens
- Paper: arXiv 2602.04805 ("SkinTokens: A Learned Compact Representation for Unified
  Autoregressive Rigging"). Predecessor: UniRig (https://github.com/VAST-AI-Research/UniRig).

### Files to COPY verbatim (see `02` for the full list)
`src/model/**`, `src/tokenizer/**`, `src/data/{transform,order,vertex_group,augment}.py`,
`src/rig_package/info/asset.py`, and the checkpoint-download logic from `download.py`.

### Files whose BEHAVIOR to reimplement (pure Python)
`src/rig_package/parser/bpy.py` (`BpyParser.load/export`, `transfer_rigging`). Delete
`src/server/bpy_server.py` and the bpy-loader wiring in `src/data/datapath.py`. Swap the
`BpyParser` import at `src/model/tokenrig.py:313`.

### Key upstream detail to remember
- `configs/skeleton/mixamo.yaml` and `vroid.yaml` — named skeleton templates + canonical
  part order. Confirms the generated skeleton is Mixamo-topology (body 22 + hands). Useful
  cross-check for the relabeler; NOT required at runtime.
- Generic `bone_N` naming is just `f"bone_{i}"` fallback (`src/rig_package/info/asset.py`,
  `src/tokenizer/tokenizer_part.py`, `src/data/order.py`).

## 2. skin-tokens.cpp — the spec for glb export / relabel / retarget (REFERENCE ONLY)

- C++23/GGML port; CPU or Vulkan; GGUF F16 weights. **Not a runtime dependency here.**
- Key files to read:
  - `src/glb.cpp` — native glb import + **skinned export** (the export spec, `03`).
  - `src/skinning.cpp`, `src/binding.cpp` — dense-weight → top-4 packing, bind integration.
  - `src/retarget.cpp` — structural humanoid recognition + SOMA30 retargeting (compare vs
    the relabeler in `04`; adopt robustness ideas).
  - `README.md` `## Status` — documents what is implemented and the parity tolerances.
  - `TODO.md` — notes on generalizing structural recognition beyond a VROID-like core, and
    constrained skeleton generation ideas.
- GGUF bundle (only if someone wants to run the C++ ref): `hf download LocalAI-io/SkinTokens-GGUF --include "F16/*"`.

## 3. Kimodo (context only — DO NOT modify from this repo)

- The animation stage. Lives in the user's fork `ComfyUI-Kimodo-Enhanced`.
- Relevant file: `kimodo_retarget_fbx.py` — `SOMA_TO_MIXAMO` mapping (line ~146) and the
  name-suffix bone matching (`SkeletonData.get_bone`). This is WHY relabeling to `mixamorig:*`
  makes the rigged output animatable with no downstream changes.
- Kimodo generates motion from text on its SOMA skeleton (30 joints), then retargets onto a
  Mixamo-named target. It needs Llama-3-8B; runs as a separate ComfyUI workflow. Not this repo.

## 4. Sample assets used during investigation (for regression tests)

Copy a couple of these into `examples/` if available from the old workspace:
- `sci-fi-dude.glb` (unrigged Trellis mesh) and `sci-fi-dude-rigged.glb` (52 bones)
- `peasant.glb` (34), `knight.glb` (34), `robot_industrial.glb` (41) — the determinism set.
- (For end-to-end only) a Kimodo-animated Mixamo FBX as a retarget sanity reference.

## Environment notes

- Target GPU server: `ai.lemon.com`, 4× RTX 3090; ComfyUI gets ONE GPU (24 GB). Rig and
  animate run as separate workflows. Do not hardcode devices.
- Python >= 3.11. SkinTokens needs ~14 GB VRAM for inference.
</content>
