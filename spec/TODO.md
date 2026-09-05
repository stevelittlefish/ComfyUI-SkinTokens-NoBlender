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
      (`test_rig_mesh_end_to_end` via `scripts/server/run-server-tests.sh`): model loads and
      `predict_step` returns a rigged Asset (skeleton + skin weights). Confirms the
      `predict_transform` config, `cls="articulation"`, and batch assembly are correct.
      Still TODO for full parity: compare joints/parents/weights against upstream `demo.py`
      on the same mesh (needs a shared input + upstream run on the server).

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
      Server round-trip wired (`test_rig_glb_to_file_roundtrip`). Still TODO: parity vs
      upstream Blender export (golden fixture) and the real Kimodo animation round-trip (E).

## Phase 4 — Relabeler (pure Python)
- [ ] `relabel.py`: topology-driven recognizer per `04`/`08` (descendant-count based;
      position only for L/R + thumb). Multi-convention via name tables: **Mixamo + UE5**.
- [ ] Apply names to exported joint node names (keep indices stable).
- [ ] Fingers: optional, per `04` (thumb=min-y; 3-finger special case).
- [ ] **Gate D** (relabel correctness on the 4 sample rigs + mirror test; both conventions).

## Phase 5 — ComfyUI nodes
- [ ] `nodes.py`: `SkinTokensLoader`, `SkinTokensRig` (relabel + params toggles), optional
      `SkinTokensRelabel`. `__init__.py` registration.
- [ ] Decide the mesh I/O socket type (default: STRING filepaths — confirm with user).
- [ ] Wrap the model for `comfy.model_management` (ModelPatcher or version-idiomatic
      equivalent); use `get_torch_device()`; verify load-on-demand + eviction.
- [ ] **Gate F** (ComfyUI integration + VRAM eviction, no leak).

## Phase 6 — Texture transfer & extras (phase 2 features)
- [ ] `transfer.py`: `use_transfer` — attach generated rig to the ORIGINAL glb preserving
      materials/textures/scale (ref: upstream `transfer_rigging`, glb.cpp material handling).
- [ ] `use_postprocess` (voxel skin) toggle, opt-in only.
- [ ] `use_skeleton` (skin-only against an existing armature): armature import in glb_io.
- [ ] `SkinTokensPreview3D` node + `web/js/` three.js extension (`05`/`08`). **Bundle
      three.js + GLTFLoader locally — no CDN.** Serve via `/view` + `folder_paths`.

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
