# SkinTokens ComfyUI Node Pack — Specification

This directory is a **complete, self-contained build spec** for a fresh Claude Code session.
Read the documents in order; `TODO.md` is the build plan.

## What we're building (one paragraph)
A pure-Python ComfyUI custom-node pack that runs the **SkinTokens / TokenRig** auto-rigging
model: static mesh (`.glb`) in → **rigged `.glb`** out (skeleton + skin weights), with an
optional step that **relabels the humanoid skeleton to Mixamo bone names** so the result is
drop-in animatable by the (separate) Kimodo pipeline. **No Blender, no subprocess, no C++
runtime** — just torch + `trimesh` + `pygltflib`, integrated with ComfyUI's VRAM manager and
installable via ComfyUI Manager.

## Read in this order
1. `00-overview.md` — purpose, the bigger pipeline, glossary (SkinTokens vs Kimodo).
2. `01-architecture.md` — decisions + rejected alternatives + ComfyUI/VRAM integration.
3. `02-scope-bpyparser-port.md` — exactly what to port (BpyParser, 3 ops) vs copy verbatim.
4. `03-glb-export-spec.md` — the critical correctness path (bone convention, IBM, weights).
5. `04-relabeler-spec.md` — the validated structural relabeler (algorithm + reference code).
6. `05-comfyui-node-spec.md` — node interface, model loading, repo layout, packaging.
7. `06-references.md` — what to clone into `reference/`, links, weights, sample assets.
8. `07-validation.md` — the gates that define "done".
9. `08-prior-art.md` — the existing Aero-Ex node: why we don't fork it, what to borrow.
10. `TODO.md` — phased task list.

## The three things most likely to go wrong (read these twice)
- **glb export bone convention / inverse bind matrices** (`03`): get it wrong and the rig
  looks fine in bind pose but explodes when animated. `skin-tokens.cpp/src/glb.cpp` is the spec.
- **Top-4 skin weight packing** (`03`): glTF allows max 4 influences/vertex; the model emits
  dense weights. Match upstream `group_per_vertex=4`.
- **Axis calibration** (`03`/`04`): determine the Asset's up/side axes empirically once; the
  exporter and the relabeler both depend on it.

## Prerequisites the human sets up before building
- Clone `reference/SkinTokens` (upstream torch) and `reference/skin-tokens.cpp` (C++ ref).
- Copy sample assets into `examples/` if available (sci-fi-dude, peasant, knight, robot).
- Target: Python ≥3.11, a CUDA GPU with ≥14 GB, a working ComfyUI to test in.
</content>
