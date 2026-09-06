# HANDOFF — animation retarget (A-pose/T-pose) & next steps

**Written:** 2026-09-06. **Purpose:** resume the animation-debugging thread in a fresh
conversation. Read this top-to-bottom and you have the full context.

---

## TL;DR

- Our node pack is **working correctly**. The bone **naming/left-right bug is fixed** and
  verified. Do not keep debugging our export/labels for the animation glitch.
- The animation "arm flapping" is a **Kimodo retarget bug**, root-caused and **confirmed**:
  Kimodo assumes a T-pose; our rigs are A-posed. A **T-pose input mesh animates cleanly.**
- **Decision for next session:** do **both**, Kimodo first:
  1. **Fix Kimodo** `retarget_animation` to be rest-*direction* aware (the real fix).
  2. **Add an optional T-pose bake** to this pack (portability feature, default off).

---

## How we got here (the story so far)

1. User reported a rigged android's **arm bones labelled for the wrong arm**, intermittently.
2. Found: SkinTokens emits a **random facing per run**; the old relabeler used a fixed
   `+x == character-Left` constant, correct for only one facing.
3. **Fixed the labeling** anatomically: `skintokens/orient.py::detect_forward` reads the
   **toe→foot** offset (thumbs deliberately unused — a confused rig drops the thumb too), and
   the relabeler picks L/R by projecting onto `up × forward`. Correct at any facing. ✅ landed,
   tested (`tests/test_orient.py`).
4. Also **tried** yawing the whole rig to a canonical -z facing on export. **Reverted** — on
   real rigs it faced the character backwards in previews and made it walk backwards; the
   **native SkinTokens orientation is correct**. Only the geometry-neutral labeling uses
   `detect_forward` now.
5. Remaining symptom: **arms "flap"** in the finished animation (both transfer and non-transfer,
   post-processed or not, with labels verified correct). Traced it into Kimodo (below).

---

## Root cause of the flapping (CONFIRMED)

**Kimodo's retarget is rotation-only and assumes a T-pose.**

- Kimodo (`~/git/ComfyUI-Kimodo-Enhanced/`) generates motion on its own **SOMA** skeleton
  (T-pose, identity global rotations) and retargets it onto the target character.
- `kimodo_retarget_glb.py` → reuses `retarget_animation` from `kimodo_retarget_fbx.py`.
  The source→target correction is computed **from rest *rotations* only**:
  ```
  # kimodo_retarget_fbx.py, retarget_animation, ~line 511
  off = _quat_mul(_quat_inv(s_bone.rest_rotation), t_bone.rest_rotation)
  ```
- **SOMA source** rest rotations are identity (its own comment: "SOMA's standard T-pose has
  identity global rotations", ~line 232).
- **Our exported rig** also has **identity bind rotations** — `glb_io.export_glb` writes
  translation-only joint nodes (the deliberate spec/03 convention). So Kimodo sees identity
  rest rotations on both sides and assumes **we are also T-posed**.
- **But our rig is A-posed** — the pose lives entirely in bone *translations*, which the
  rotation-only retarget never looks at. Measured on `android_rigged_post_processed.glb`:
  - arms sit **~65° below horizontal** (T-pose arm = 0°),
  - legs **~85°** (≈ SOMA's straight-down legs, ~90°).
- So SOMA "raise arm from horizontal" motion is applied to an arm already 65° down, **with no
  correction → flapping**. Legs barely mismatch (~5°) so they walk roughly fine. This
  arms-flap / legs-ok fingerprint is exactly this bug.

**Confirmation:** rigging a **T-pose mesh** and animating it through Kimodo **works cleanly.**

---

## The two fixes (decision: do BOTH, Kimodo first)

### A. Fix Kimodo retarget (PRIMARY — the correct fix)
- **Where:** `~/git/ComfyUI-Kimodo-Enhanced/kimodo_retarget_fbx.py`, `retarget_animation`
  (shared by the `.glb` path in `kimodo_retarget_glb.py`).
- **What:** make the source→target offset account for each bone's **rest direction** (the
  vector from the bone to its child / its `head`→child), not just `rest_rotation`. Then a
  T-pose source maps correctly onto an A-pose (or any-pose) target.
- **Why primary:** it's the true root cause, needs **no mesh deformation** (lossless), unblocks
  immediately, and benefits *every* character fed to Kimodo — not just ours. This is precisely
  the Gate E "Kimodo needs a fix" the spec has always referred to.
- **Caveat:** different repo; Kimodo has been called possibly-unreliable. Start by reading the
  full `retarget_animation` (and `_skeleton_height`, `kimodo_to_source_skeleton`) to scope it
  before changing anything. Add a test.

### B. Optional T-pose bake in THIS pack (SECONDARY — portability)
- **What:** after rigging, re-pose the rigged mesh to a canonical **T-pose** (rotate each
  limb bone from its current rest direction to the canonical Mixamo T-pose direction,
  propagate down the hierarchy, apply linear-blend-skinning to reposition the vertices, and
  re-export with the new rest pose). We already know the bones (relabeled) and have LBS
  reference code in `tests/test_glb_export.py`.
- **Why secondary / default off:** it's a heuristic re-pose; a ~65° shoulder rotation via
  skinning can introduce artifacts. But T-posed rigs are the **universal** retargeting
  standard (Mixamo, Unreal, Unity), so this makes our output portable to any engine and
  aligns with the original "drops into the pipeline unchanged" design goal.
- **Node surface:** a new `to_tpose` boolean on `SkinTokensRig` (default False).

**Sequence:** implement A, verify a previously-flapping A-pose rig now animates; then add B as
a robustness/portability feature.

---

## Also resolved this session (no action needed)

- **Post-process + transfer:** verified there is **no bug** — `use_postprocess` is passed in
  both node paths, `apply_voxel_postprocess` refines `rigged.skin` in place *before* the
  export/transfer branch, and `transfer_rigging` carries those weights via nearest-neighbour.
  User confirmed no symptoms. (What earlier looked like "post-process skips transfer" was the
  orientation *canonicalization*, which only ran on the non-transfer path — now removed.)
- **`use_postprocess` now defaults ON** (node + example workflow + docs).

---

## Key files / pointers

- This pack — labeling: `skintokens/orient.py` (`detect_forward`, `left_dir`),
  `skintokens/relabel.py` (`pick_side` uses `up × forward`). Export: `skintokens/glb_io.py`
  (`export_glb`, translation-only joint nodes, pure-translation IBMs — the spec/03 convention).
  Transfer: `skintokens/transfer.py`. Post-process: `skintokens/postprocess.py`.
- Kimodo — `~/git/ComfyUI-Kimodo-Enhanced/kimodo_retarget_fbx.py` (`retarget_animation`,
  `SOMA_TO_MIXAMO` ~L146), `kimodo_retarget_glb.py` (glb read/write of the animation).
- Deploy note: the ComfyUI server bakes this pack into its image via a `--depth=1` clone at
  **build time** (`ComfyUIDocker/custom-nodes.yaml`). To pick up new commits, **rebuild the
  image**, not just restart. Check running hashes:
  `docker compose exec comfyui sh -c 'git -C /srv/app/custom_nodes/ComfyUI-SkinTokens-NoBlender rev-parse HEAD'`.
- Test data: `~/Documents/meshes/` (android rigs). Local tests: `source .venv/bin/activate &&
  python -m pytest -q` (expect ~97 passed, ~4 server-skipped).

## First actions next session
1. Read `retarget_animation` in full (Kimodo) → scope the rest-direction change.
2. Implement fix A; test against a known A-pose rig that currently flaps.
3. Then add optional T-pose bake (B) to `SkinTokensRig`.
