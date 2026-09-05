# Architecture & Decisions

This document records the architecture and, importantly, the alternatives that were
**considered and rejected**, so they are not relitigated.

## Core decisions

1. **Pure-Python ComfyUI node pack, separate repository.**
   - Not folded into the Kimodo fork. Its own repo, so it can be distributed independently
     via ComfyUI Manager.
2. **No Blender. No `bpy`. No `bpy_server`.**
   - All glb I/O is done in pure Python (`trimesh` for import, `pygltflib` for skinned
     export). See `02` and `03`.
3. **No CLI subprocess / no wrapping the C++ port as the runtime.**
   - The torch model is loaded in-process so ComfyUI's memory manager can see and manage it.
4. **Rig and animate are separate workflows, never concurrent.**
   - The SkinTokens model (~14 GB) never needs to coexist on the GPU with Kimodo's Llama
     (~16 GB). A single 24 GB card is comfortable for the rig stage alone.
5. **VRAM handled by ComfyUI's `ModelPatcher` / `comfy.model_management`.**
   - This gives the desired "unload other models → load this → unload when done" behavior
     natively.
6. **Relabel to Mixamo is an optional toggle** on the node (default: on, but exposed).

## Rejected alternatives and why

- **Wrap `skin-tokens.cpp` CLI as a subprocess node.**
  Rejected because: (a) a subprocess's VRAM (CUDA or Vulkan) is **invisible to ComfyUI's
  memory accounting** — ComfyUI won't free room for it and won't count it, so the wanted
  automatic load/unload doesn't happen (you'd have to manually call `unload_all_models()`);
  (b) distributing a **compiled native binary + GGUF + Vulkan runtime** via ComfyUI Manager
  is not the normal pip-based flow and is high-friction. Its clean process-exit VRAM cleanup
  was the one upside, but not enough.
- **In-process C API (ctypes/pybind) + custom ModelPatcher shim.**
  Rejected as unnecessary complexity: you'd hand-roll memory accounting for a Vulkan
  allocator ComfyUI still can't truly see. Pure-Python torch is simpler and idiomatic.
- **Shared-VRAM coexistence of SkinTokens + Kimodo on one card.**
  Moot: the stages run separately. Do not design for coexistence.
- **Keeping Blender via a bundled `bpy_server` subprocess.**
  Rejected by explicit decision — no Blender in this project at all.

## Target deployment environment (context, not a constraint to hardcode)

- GPU server `ai.lemon.com`: 4× RTX 3090 (24 GB each). **ComfyUI only has access to one GPU
  (GPU1).** The other three run unrelated services and are not available to this pipeline.
- Because rig/animate are separate workflows, one 24 GB card suffices for the rig stage.
- The node must respect ComfyUI's device selection (do not hardcode `cuda:0`; use
  `comfy.model_management.get_torch_device()`).

## ComfyUI VRAM integration (how, concretely)

ComfyUI only manages models it knows about. To get automatic load/unload:

- Wrap the SkinTokens torch model in a **`comfy.model_patcher.ModelPatcher`** (or the
  current idiomatic equivalent for the installed ComfyUI version), reporting its memory
  footprint, so ComfyUI can offload it to CPU / evict it when other workflows need VRAM.
- Load lazily (first use), on `comfy.model_management.get_torch_device()`.
- Let ComfyUI drive eviction; do not manually pin the model to the GPU forever.
- Verify against the ComfyUI version pinned for the project — the `model_management` API
  evolves. If the exact `ModelPatcher` contract is awkward for a non-UNet model, an
  acceptable fallback is a lighter-weight registration that still calls
  `model_management.free_memory(...)` appropriately and offloads to CPU between runs. The
  requirement is behavioral: **model loads on demand, frees when the GPU is needed
  elsewhere.**

## High-level data flow inside the node

```
glb bytes/path
   │  trimesh import  (02)
   ▼
mesh (verts, faces, normals)  ──►  torch preprocessing (copied verbatim: sample + normalize)
   ▼
SkinTokens model.predict_step  ──►  Asset(joints, parents, skin weights, vertices, faces)
   ▼  (optional) structural relabel to mixamorig:*  (04)
   ▼
pygltflib skinned-glb export  (03)   [+ optional texture transfer, phase 2]
   ▼
rigged glb bytes/path
```
</content>
