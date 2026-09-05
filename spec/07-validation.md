# Validation Plan

Correctness is priority #2 (after ecosystem fit). The export path (`03`) is the main risk.
Do NOT declare the node "done" until the gates below pass — especially the posing gate, since
a rig can look correct in bind pose and still explode when animated.

## Gate A — Import parity (cheap, do first)
- Load a mesh via the new `trimesh` importer and via upstream `BpyParser.load`.
- Assert matching vertex count, face count, and (within tolerance) vertex positions/normals.

## Gate B — Inference parity
- Run the SAME input mesh through upstream `demo.py` and through the new `infer.py`.
- Compare the resulting `Asset`: joint count, `parents` array (exact), joint positions
  (small tolerance), and dense skin weights (small tolerance). Fix seeds / beams to make the
  autoregressive output comparable; if sampling makes exact parity impossible, compare
  topology (parents) exactly and positions/weights statistically.

## Gate C — Export correctness (the critical one)
1. **Re-import**: exported glb re-opens in `pygltflib`/`trimesh` AND a real glTF viewer with
   the mesh bound to the skeleton (has `skin`, `JOINTS_0`, `WEIGHTS_0`, `inverseBindMatrices`).
2. **Bind-pose identity**: for every joint, `worldMatrix(joint) @ inverseBindMatrix(joint)`
   ≈ identity. Assert numerically.
3. **Weight sanity**: each vertex's 4 weights are ≥0 and sum ≈ 1.0; indices are valid joints.
4. **Posing gate (must pass)**: programmatically rotate a few bones (e.g. an elbow, a knee)
   and confirm the mesh deforms **locally and plausibly** — no collapse to origin, no
   detachment, no whole-mesh warp. This catches the bone-convention / IBM bug from `03`.
5. **Parity vs upstream export**: rig the same mesh with upstream (Blender export) and with
   the new exporter; compare the two glbs' skin weights (top-4 per vertex) and joint
   transforms within tolerance.

## Gate D — Relabeler correctness
- Run the relabeler on the 4 sample rigs (sci-fi-dude 52, peasant 34, knight 34, robot 41).
- Assert it produces exactly the validated 22-bone `mixamorig:*` mapping for each (see `04`).
- Assert Hips=root, the first-10-bones body chain, arms located regardless of finger count,
  and legs = last 8 bones.
- Add a mirrored-mesh test to confirm the Left/Right convention (flip constant if needed).

## Gate E — End-to-end through Kimodo (on ai.lemon.com)
- Take a freshly rigged + relabeled glb, feed it to the existing Kimodo retarget path
  (`kimodo_retarget_fbx.py` / the Kimodo ComfyUI workflow), generate a short motion.
- Confirm: bones match by name (no "0 bone pairs matched" warning), motion plays, and the
  **mesh does not explode** during animation. This is the true acceptance test for the whole
  reason the project exists.

## Gate F — ComfyUI integration
- Node appears in ComfyUI, runs in a workflow, produces a rigged glb.
- VRAM: after the node runs, and when another model needs the GPU, ComfyUI can evict the
  SkinTokens model (watch `nvidia-smi` on the target box). No leak across repeated runs.
- Installs cleanly from a fresh checkout via requirements (and, ideally, via Manager).

## Regression fixtures
- Keep the 4 sample rigs + 1 unrigged mesh in `examples/` and wire Gates C/D into `pytest` so
  future changes can't silently break export or relabeling.
</content>
