# TODO — Build Order

Phased task list for building the SkinTokens ComfyUI node pack. Do phases in order; each
phase ends at a runnable/verifiable checkpoint. Gates refer to `07-validation.md`.

Status legend: `[x]` done · `[~]` partial / blocked · `[ ]` not started.
GPU/server-only gates (B, E, F) are deferred until we have Comfy-server access; the
laptop has no GPU. Keep this file updated as work lands (see CLAUDE.md).

## Phase 0 — Project setup ✅
- [x] Create the repo; add `spec/` (this package) and `reference/` (upstream SkinTokens +
      skin-tokens.cpp, gitignored or submodules). See `06`. (`references/`, gitignored.)
- [x] `pyproject.toml` + `requirements.txt` (trimesh, pygltflib, transformers, etc.; NO bpy,
      NO gradio/bottle/tornado). Make `flash-attn` optional. (Also `requirements-dev.txt`
      with CPU torch + pytest for GPU-less local dev.)
- [x] Vendor upstream torch core into `skintokens/vendor/` (whole `src/` minus bpy/server).
      Upstream commit `273b691` recorded in `skintokens/vendor/UPSTREAM.md`.
- [x] Fix imports in vendored code to drop bpy/server references. Get it importing.
      (Also: flash-attn→SDPA fallbacks, idempotent OmegaConf resolvers, CUDA-absent guard.
      Verified by `tests/test_vendor_imports.py`.)

## Phase 1 — Inference (no I/O yet)
- [x] `model_loader.py`: load TokenRig + skin-vae from HF checkpoints; auto-download.
      (Rewrites ckpt's baked-in relative paths for pretrained_vae + Qwen config.)
- [x] `infer.py`: given an in-memory mesh (verts/faces/normals), build the `Asset`, run
      `predict_step`, return the rigged `Asset` (joints, parents, dense weights). Reuse the
      copied transform/sampling/normalization.
- [x] Temporary harness: `tests/test_infer.py` (CPU) covers `build_asset`, the sampler
      transform, and input guards. Full run is `test_rig_mesh_end_to_end` (`server` marker).
- [~] **Gate B** (inference parity vs upstream demo.py). Smoke test PASSES on the server
      (`test_rig_mesh_end_to_end` + `test_rig_glb_end_to_end` via
      `scripts/server/run-server-tests.sh`, all 4 server tests green 2026-09-05): model loads
      and `predict_step` returns a rigged Asset (skeleton + skin weights) on both a synthetic
      box and a real humanoid glb. Confirms the `predict_transform` config,
      `cls="articulation"`, and batch assembly are correct. Still TODO for full parity:
      compare joints/parents/weights against upstream `demo.py` on the same mesh (needs a
      shared input + upstream run on the server).

## Phase 2 — glb import (pure Python)
- [x] `glb_io.py` import: trimesh → verts/faces/normals/mesh_names + vertex_bias/face_bias,
      matching the fields `BpyParser.load` produced. Multi-part scenes concatenated (faces
      offset by running vertex count); normals recomputed by trimesh (as upstream did).
      `load_mesh` (arrays) + `load_asset` (unrigged Asset). `infer.rig_glb` wires it to
      inference. (Armature-in / `use_skeleton` import deferred to Phase 6. UV not extracted —
      only needed for texture transfer, Phase 6.)
- [~] **Gate A** (import parity). Self-consistency half DONE (`tests/test_glb_io.py`: counts,
      multi-part bias, face-offset validity, normals, Asset fields; verified on the real
      3-part giraffe.glb). Blender-parity (vs `BpyParser.load`) still TODO: needs a committed
      golden fixture generated on a Blender box. AXIS: trimesh loads glTF-native Y-up vs
      upstream's Blender Z-up — calibration deferred to Phase 3 (must match export).

## Phase 3 — glb export (THE hard part — read `03` + glb.cpp first)
- [~] Calibrate the Asset's native axis convention (up axis, side axis); document constants.
      PARTIAL: skinning is frame-agnostic (verts + joints share the Asset's frame), so the
      posing gate passes without a rotation. The empirical up-axis constant (for upright
      display in a Y-up viewer + the relabeler) still needs a real rigged asset — dump one
      on the server and set the constant. Not blocking export correctness.
- [x] `glb_io.py` export (`export_glb`): joint nodes + hierarchy, skin with
      inverseBindMatrices, top-4 JOINTS_0/WEIGHTS_0 (`pack_top4`, group_per_vertex=4),
      one-frame rest pose. Matches glb.cpp: translation-only joints, IBM = translate(-joint).
- [x] Wire to the exporter: `infer.rig_glb_to_file` (glb in -> rigged skinned glb out).
      (The vendored `tokenrig.py:predict_step` still lazily imports BpyParser only for its
      own file-export convenience path, which we don't call — our pipeline uses `export_glb`.)
- [x] **Gate C** (export correctness — incl. the POSING gate) — LOCAL half PASSES
      (`tests/test_glb_export.py`): skin structure, bind-pose identity (world@IBM==I), weight
      sanity (>=0, sum==1, valid indices), `pack_top4` selection/normalization, and the
      POSING GATE via a numpy reference LBS (rest reproduces verts; rotating a bone deforms
      upper verts locally at fixed pivot radius while lower stay put; no collapse/explosion).
      Server round-trip PASSES (`test_rig_glb_to_file_roundtrip`, 2026-09-05): real glb ->
      rigged skinned glb out, re-importable with JOINTS_0/WEIGHTS_0 and J joints. Still TODO:
      parity vs upstream Blender export (golden fixture) and the real Kimodo animation
      round-trip (E).

## Phase 4 — Relabeler (pure Python)
- [x] `relabel.py`: topology-driven recognizer per `04`/`08` (descendant-count based;
      position only for L/R + thumb). Multi-convention via name tables: **Mixamo + UE5**
      (`NAME_TABLES`, extensible). Pelvis = first bone with >=3 children (survives an
      armature root above it). Pure numpy; `label_humanoid` returns `{index: name}`.
- [x] Apply names to exported joint node names (keep indices stable). `relabel_asset`
      writes into `asset.joint_names` in place (creating `bone_N` first if absent);
      indices never reorder, so exported JOINTS_0 stays consistent (export reads
      `asset.joint_names` for the glTF node names).
- [x] Fingers: optional (`with_fingers`), per `04` (thumb=min up-axis; remaining sorted
      by side axis; 3-finger special case -> [thumb, index, pinky]).
- [x] **Gate D** PASSES (`tests/test_relabel.py`, 22 tests). Synthetic half: Mixamo + UE5
      core labeling, mirror test (L/R by position), 5/3/off fingers, asset in-place rename,
      unrecognized-joint preservation, armature-root-above-pelvis. REAL half: the full core-22
      resolves correctly on **all 4 validation rigs** (knight/peasant 34, robot 41, sci-fi 52 —
      bone counts match spec/04) in both conventions, and the first-10 indices match the
      documented mapping. Skeletons committed as `tests/fixtures/rigs/*.json` (joints+parents
      extracted from the rigged .glb, no mesh; ~6 KB total).

## Phase 5 — ComfyUI nodes
- [x] `nodes.py`: `SkinTokensLoader`, `SkinTokensRig` (relabel + generation params toggles),
      `SkinTokensRelabel` (standalone, model-free). `__init__.py` registration (relative
      import with an absolute fallback so pytest can import it without package context).
      `comfy.*`/`torch`/`folder_paths` imported lazily inside methods, so importing the pack
      (what ComfyUI does at startup to read NODE_CLASS_MAPPINGS) needs no GPU/model.
- [x] Mesh I/O socket type = ComfyUI's **native 3D types** (decided with user after
      inspecting the Trellis workflow + ComfyUIDocker core): `SkinTokensRig` accepts a native
      `MESH` (rigs straight from its vertex/face/normal tensors) OR a `File3D`
      (`FILE_3D_GLB/GLTF/OBJ/STL`); rigged output is `FILE_3D_GLB` (the glb carries the
      skeleton+skin; the `MESH` type has no armature). Drops into Load3D/Trellis/remesh →
      Rig → Preview3DAdvanced(showSkeleton)/Save3D. Bridge in `skintokens/comfy_types.py`
      (duck-typed, `comfy_api`/`torch` lazy; `make_file3d` falls back to a path string
      without ComfyUI so it stays testable).
- [x] Wrap the model for `comfy.model_management` (`skintokens/comfy_model.py`): a
      `ModelPatcher` over the nn.Module with `get_torch_device()`/`unet_offload_device()`,
      loaded to CPU by the Loader and moved to GPU on demand via `load_models_gpu` in
      `SkinTokensModelWrapper.prepare()` (ComfyUI drives eviction; never hardcodes cuda:0).
      Falls back to a passthrough wrapper when `comfy` is absent (local dev). Size estimator +
      no-comfy fallback unit-tested.
- [x] **Gate F** (ComfyUI integration) — WORKS END TO END in real ComfyUI on the server
      (2026-09-05, comfy.seaslug.ai): the workflow `Load3D -> SkinTokens Rig -> Preview3DAdvanced`
      (with SkinTokens Loader) runs a full rig and previews the skinned result with skeleton.
      Two real integration bugs found + fixed only by running in ComfyUI: (1) pack failed to
      register because ComfyUI doesn't put the pack dir on sys.path and our __init__ fallback
      silently imported ComfyUI's own `nodes` — fixed by inserting the pack dir into sys.path
      (`2f4588b`); (2) `ModelPatcher.load()` assigns `model.device` directly but Lightning's
      TokenRig has a read-only `device` property — fixed with a runtime settable-device
      subclass (`7f55a97`). Loader redesigned to the idiomatic auto-download dropdown
      (`fdec169`). LOCAL coverage: `tests/test_nodes.py` (22 tests) + the GPU
      `test_rig_node_end_to_end`. VRAM EVICTION CONFIRMED (2026-09-05): the Manager's
      "clear all models and cache" frees the model and VRAM returns to 0 — proving the model
      is properly registered with `model_management` (via ModelPatcher) and evictable with no
      leak. Gate F complete.

## Phase 6 — Texture transfer & extras (phase 2 features)
- [ ] `transfer.py`: `use_transfer` — attach generated rig to the ORIGINAL glb preserving
      materials/textures/scale (ref: upstream `transfer_rigging`, glb.cpp material handling).
- [ ] `use_postprocess` (voxel skin) toggle, opt-in only.
- [ ] `use_skeleton` (skin-only against an existing armature): armature import in glb_io.
- [x] ~~`SkinTokensPreview3D` node + `web/js/` three.js extension~~ **SKIPPED** (decided
      with user, Phase 5): core ComfyUI's `Preview3DAdvanced` already previews our
      `FILE_3D_GLB` output and has a `showSkeleton` toggle, so a custom preview node is
      redundant. Users wire Rig -> Preview3DAdvanced directly. Revisit only if a
      rig-specific preview (bone picking, weight paint) is ever wanted.

## Phase 7 — End-to-end + packaging
- [ ] **Gate E** (end-to-end through Kimodo on ai.lemon.com — the real acceptance test).
- [ ] `examples/`: sample workflow .json + regression fixtures wired into pytest (Gates C/D).
- [ ] README (install, usage, Manager registry metadata). Publish path via Manager.

## Cross-cutting reminders
- No Blender, no subprocess engine, no C++ runtime dependency.
- Don't hardcode `cuda:0`; honor ComfyUI device/flags.
- Speed is not a priority; correctness + ecosystem fit are.
- The export bone/IBM convention and the top-4 weight packing are the highest-risk details.
</content>
