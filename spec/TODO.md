# TODO — Build Order

Phased task list for building the SkinTokens ComfyUI node pack. Do phases in order; each
phase ends at a runnable/verifiable checkpoint. Gates refer to `07-validation.md`.

## Phase 0 — Project setup
- [ ] Create the repo; add `spec/` (this package) and `reference/` (upstream SkinTokens +
      skin-tokens.cpp, gitignored or submodules). See `06`.
- [ ] `pyproject.toml` + `requirements.txt` (trimesh, pygltflib, transformers, etc.; NO bpy,
      NO gradio/bottle/tornado). Make `flash-attn` optional.
- [ ] Vendor upstream torch core into `skintokens/vendor/` (model, tokenizer, data/{transform,
      order,vertex_group,augment}, rig_package/info/asset.py). Record the upstream commit.
- [ ] Fix imports in vendored code to drop bpy/server references. Get it importing.

## Phase 1 — Inference (no I/O yet)
- [ ] `model_loader.py`: load TokenRig + skin-vae from HF checkpoints; auto-download.
- [ ] `infer.py`: given an in-memory mesh (verts/faces/normals), build the `Asset`, run
      `predict_step`, return the rigged `Asset` (joints, parents, dense weights). Reuse the
      copied transform/sampling/normalization.
- [ ] Temporary harness: feed a mesh loaded any quick way; confirm an Asset comes out.
- [ ] **Gate B** (inference parity vs upstream demo.py).

## Phase 2 — glb import (pure Python)
- [ ] `glb_io.py` import: trimesh → verts/faces/normals/uv/mesh_names, matching the fields
      `BpyParser.load` produced. (Armature-in / `use_skeleton` import can wait to Phase 6.)
- [ ] **Gate A** (import parity).

## Phase 3 — glb export (THE hard part — read `03` + glb.cpp first)
- [ ] Calibrate the Asset's native axis convention (up axis, side axis); document constants.
- [ ] `glb_io.py` export: build joint nodes + hierarchy, skin with inverseBindMatrices,
      top-4 JOINTS_0/WEIGHTS_0 (group_per_vertex=4), one-frame rest pose. Match glb.cpp.
- [ ] Wire `tokenrig.py` make_asset to the new exporter (replace the BpyParser import).
- [ ] **Gate C** (export correctness — including the POSING gate). Do not proceed until the
      posing gate passes.

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
