# SkinTokens ComfyUI Node Pack — Project Overview

## What we are building

A **pure-Python ComfyUI custom-node pack** that wraps the **SkinTokens / TokenRig**
auto-rigging model. It takes a static mesh (`.glb`) and outputs a **rigged `.glb`**
(skeleton + skin weights), with an **optional step that relabels the generated skeleton
with Mixamo bone names** so the output is drop-in compatible with Mixamo-based animation
tools.

The node pack must:
- Load the SkinTokens torch model and run inference (mesh → skeleton + skin weights).
- Import/export `.glb` in **pure Python** (no Blender / `bpy`).
- Optionally relabel the humanoid skeleton to `mixamorig:*` names.
- Integrate with ComfyUI's VRAM manager (`comfy.model_management` / `ModelPatcher`) so the
  model loads on demand and unloads when other workflows need the GPU.
- Install like a normal ComfyUI node (pip requirements, weights auto-download), so it can be
  distributed via ComfyUI Manager.

## Why this exists — the bigger pipeline

The end goal is a fully automated, no-user-interaction pipeline running on a LAN GPU server:

```
Trellis.2 (generate mesh) → SkinTokens (auto-rig) → Kimodo (text→motion) → animated character
```

**This repo is only the SkinTokens (auto-rig) stage.** The animation stage (Kimodo) already
exists as a separate, working ComfyUI package and is **out of scope** here. The two stages
**always run as separate ComfyUI workflows** — never simultaneously — so there is no
same-GPU VRAM contention between them to design around.

The relabel step is what connects this stage to the next: Kimodo animates **Mixamo-named**
skeletons, and SkinTokens by default emits generic `bone_N` names. Relabeling bridges them.

## Two tools — do not confuse them (glossary)

| | **SkinTokens** (this repo) | **Kimodo** (separate, out of scope) |
|---|---|---|
| Job | **Rig** a mesh (skeleton + skin weights) | **Animate** (text prompt → motion) |
| Input | static mesh (`.glb`) | text prompt + a rigged character |
| Output | rigged `.glb` | animated file |
| Author | VAST-AI (Tripo/Trellis) | NVIDIA |
| Heavy model | Qwen3-0.6B + shape encoder (~14 GB) | Llama-3-8B text encoder (~16 GB) |

- **SkinTokens = TokenRig**, the successor to **UniRig**. Autoregressive: mesh → (skeleton
  tokens + skin-weight "SkinTokens") as one sequence.

## Source material (references to be placed in the new repo)

1. **Upstream SkinTokens (torch)** — the reference implementation. The torch **model,
   tokenizer, transforms, FSQ, SkinVAE, sampling, normalization are copied verbatim**. The
   only part NOT reused is its Blender-based glb I/O (`BpyParser` + `bpy_server`).
   - Upstream repo: https://github.com/VAST-AI-Research/SkinTokens
   - HF weights: https://huggingface.co/VAST-AI/SkinTokens
2. **skin-tokens.cpp** — a C++/GGML port. **Reference only, NOT a runtime dependency.** It is
   the **spec** for the pure-Python glb export, the structural skeleton relabeler, and
   SOMA30 retargeting. Key files: `src/glb.cpp`, `src/retarget.cpp`, `src/skinning.cpp`,
   `src/binding.cpp`.
   - Repo: https://github.com/(LocalAI-io fork or wherever the user cloned it)

> When the new project is set up, place clones/copies of both under a `reference/` directory
> so the building session can read them. See `06-references.md`.

## Guiding priorities (in order)

1. **Flexibility & fitting the ComfyUI ecosystem** — idiomatic node, Manager-installable.
2. **Correctness** — the exported rig must survive being posed/animated (see `03`).
3. **Speed is explicitly NOT a priority.** Pure Python is fine.

## Documents in this spec

- `00-overview.md` — this file.
- `01-architecture.md` — decisions, rejected alternatives, VRAM/ComfyUI integration.
- `02-scope-bpyparser-port.md` — exact code to port vs copy.
- `03-glb-export-spec.md` — the hard part: bone convention, inverse bind matrices, weights.
- `04-relabeler-spec.md` — the structural humanoid relabeler (validated algorithm + code).
- `05-comfyui-node-spec.md` — node interface, model loading, ModelPatcher, repo layout.
- `06-references.md` — what to copy, what to reference, links, weights.
- `07-validation.md` — how to prove it works.
- `TODO.md` — ordered, phased task list.
</content>
