# Prior Art — Aero-Ex/ComfyUI-SkinTokens

An existing ComfyUI node pack for SkinTokens exists: **`Aero-Ex/ComfyUI-SkinTokens`**
(https://github.com/Aero-Ex/ComfyUI-SkinTokens). We evaluated it and **chose NOT to fork it**.
This doc records why, and what to borrow as reference.

## Decision: build fresh, do not fork
- It **requires Blender 4.2+** (headless subprocess *or* embedded `bpy`) for all glb I/O AND
  its relabeler (a `bpy` script string). Our project is **pip-only, no Blender** — so its
  core mechanism is exactly what we're removing. Forking means gutting the core; a fork saves
  almost nothing on the hard parts (pure-Python skinned-glb export, VRAM management).
- It has **no VRAM/`model_management` integration** — hardcodes `.to("cuda")`. That is our
  main value-add and must be built regardless.
- The genuinely reusable parts are **algorithms/UX**, which we lift as reference without
  inheriting a Blender-centric codebase.

Note (correcting an earlier assumption): in that repo Blender does **CPU-only mesh I/O**; the
torch model runs in-process on GPU. So Blender-subprocess did not actually break VRAM
management there. The sole reason we drop Blender is **deployment purity (pip-only)** — a firm
project requirement.

## Ideas to ADOPT (fold into our spec)

1. **Multi-convention relabeler.** It relabels to **Mixamo** and **UE5** (and could extend to
   others) from the same structural walk, using per-convention name tables. → Our relabeler
   (`04`) is generalized to a `convention` parameter. UE5 example names:
   `pelvis`, `spine_01…`, `neck_01`, `head`, `clavicle_l/r`, `upperarm_l`, `lowerarm_l`,
   `hand_l`, `thumb_01_l`, `index_01_l`, … Mixamo: `Hips`, `Spine/Spine1/Spine2`, `Neck`,
   `Head`, `LeftShoulder`, `LeftArm`, `LeftForeArm`, `LeftHand`, `LeftHandThumb1`, …

2. **Topology-driven recognition (more axis-robust than our first draft).** Their walk:
   - **pelvis = first bone with ≥3 children** (root-agnostic; survives a root/armature bone
     above the pelvis). Fall back to the root if none.
   - split pelvis children by **descendant count**: largest subtree = spine root; the rest =
     legs (L/R by `head.x`).
   - walk spine until a bone with ≥3 children = chest; that split's children by descendant
     count: smallest = neck chain; the rest = arms (L/R by `head.x`).
   - neck chain walked until a childless bone = head.
   - arm chain: shoulder→arm/upperarm→forearm/lowerarm→hand (single-child walk).
   Use **descendant count** for spine/legs and neck/arms disambiguation, position (`head.x`)
   only for left/right and thumb. This reduces dependence on axis calibration.

3. **Finger naming (optional).** thumb = finger with **min `head.y`**; remaining fingers
   sorted by `head.x` (reversed per side); special-case a **3-finger hand** →
   `[thumb, index, pinky]`. Names: Mixamo `LeftHandThumb1/2/3`, UE5 `thumb_01_l`, etc.

4. **Inline 3D preview node.** An `OUTPUT_NODE` that serves the result glb via ComfyUI's
   `/view` endpoint (resolved with `folder_paths` output/input dirs) and renders it with
   three.js + GLTFLoader/FBXLoader in the graph, returning `{"ui": {...}, "result": (...)}`.
   → Adopt, but **bundle three.js + loaders locally** (their version loads from CDN, which
   breaks on an airgapped LAN). See `05`.

5. **STRING filepath mesh sockets + `folder_paths`.** Confirms our default I/O choice; use
   `folder_paths.get_output_directory()` / `get_input_directory()` for path handling.

## Things to AVOID (their mistakes)
- Blender dependency (headless or embedded `bpy`) — removed entirely.
- CDN-loaded three.js in the previewer — bundle locally.
- No VRAM management — we add `model_management`/ModelPatcher.
- Leftover server deps (`gradio`, `bottle`, `tornado`) — not needed.

## Reference files in their repo
- `nodes.py` — node scaffolding; `rename_joints_in_blender` (the multi-convention relabel
  algorithm, lines ~157–330); `SkinTokensRigPreviewer` (preview node, ~727).
- `web/js/preview3d.js` — the three.js previewer web extension.
- `install.py`, `pyproject.toml` — packaging/Manager scaffolding reference.
</content>
