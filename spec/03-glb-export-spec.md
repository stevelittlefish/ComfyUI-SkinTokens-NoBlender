# glb Export Spec — The Critical Correctness Path

Exporting the rigged glb is where the project succeeds or fails. A subtly wrong export
produces a file that **looks** rigged in a viewer's bind pose but **explodes / detaches when
posed or animated**. Treat `skin-tokens.cpp/src/glb.cpp` (and `src/skinning.cpp`,
`src/binding.cpp`) as the authoritative reference and match it.

## Required reading before writing any export code

1. `skin-tokens.cpp/src/glb.cpp` — the export path (node hierarchy, skin,
   inverseBindMatrices, JOINTS_0/WEIGHTS_0, rest pose).
2. Upstream `SkinTokens/src/rig_package/parser/bpy.py` — the `export` path and
   `transfer_rigging`, to see what the Blender exporter actually wrote.
3. Upstream `SkinTokens/src/rig_package/info/asset.py` — the `Asset` type and especially the
   comment at line ~158: bones are extruded along **local Y "in accordance with Blender"**.

## The bone-convention trap (read this twice)

- SkinTokens' skeleton was authored/exported through Blender, whose **bone local space** has
  a specific convention (bone points along its local **+Y**, with a roll). The dense skin
  weights and the joint transforms are consistent with **that** convention.
- glTF joints are plain nodes with TRS transforms plus a `skin.inverseBindMatrices` array.
- The exporter must produce joint node world transforms + inverse bind matrices such that:
  `worldMatrix(joint) @ inverseBindMatrix(joint)` = identity at bind time, for every joint.
  If the bind matrices are inconsistent with the node transforms, the mesh deforms wrong.
- Practically: derive each joint's **bind world matrix** from the `Asset` (positions +
  parent hierarchy + whatever orientation upstream used), set `inverseBindMatrices[i] =
  inverse(bindWorld[i])`, and set node local transforms so that recomputing world transforms
  from the hierarchy reproduces `bindWorld[i]`. `glb.cpp` shows exactly how it builds these.

## glTF skinned-mesh structure to emit (via `pygltflib`)

- **Nodes**: one node per joint, arranged in the parent hierarchy from `Asset.parents`
  (root joint's parent = -1). Each node gets a local TRS (translation at minimum; include
  rotation if the convention requires it — follow `glb.cpp`).
- **Skeleton root node**: the root joint (or a wrapper) referenced by `skin.skeleton`.
- **Skin**:
  - `joints`: array of node indices in a fixed order (index i ↔ joint i).
  - `inverseBindMatrices`: accessor to J × mat4 (column-major, as glTF requires).
- **Mesh primitive attributes**:
  - `POSITION`, `NORMAL`, (`TEXCOORD_0` if present), `indices` — from `Asset`.
  - `JOINTS_0`: per-vertex 4× joint indices (unsigned short/byte accessor).
  - `WEIGHTS_0`: per-vertex 4× normalized float weights.
- **Node referencing the mesh** must set `skin` = the skin index.
- **Rest pose**: write joint node transforms as the bind (rest) pose so the file opens as a
  static, correctly-shaped skinned asset. (Upstream stores "a one-frame rest pose".)

## Dense weights → top-4 (must match `group_per_vertex=4`)

The model emits **dense** per-vertex weights over all joints. glTF allows **max 4 influences
per vertex**. For each vertex:
1. Take the 4 joints with the largest weights.
2. Renormalize those 4 so they sum to 1.0.
3. Fill `JOINTS_0` (the 4 joint indices) and `WEIGHTS_0` (the 4 weights).
Upstream uses `group_per_vertex=4`; match its selection/normalization exactly. If upstream
applies optional `voxel_skin` postprocessing (`--postprocess` / `use_postprocess`), keep it
**opt-in** (a node toggle), not silently on.

## Coordinate system / up-axis

- glTF is **Y-up, right-handed, meters**. The raw `Asset` coming from the torch model is in
  the model's normalized frame; Blender import/export previously handled axis conversion.
  Since Blender is gone, **determine the Asset's native axis convention empirically** (dump a
  known rig's joints and check which axis is vertical) and convert to glTF Y-up on export.
- This axis fact also feeds the relabeler (`04`), which needs to know the up-axis and the
  left-right axis. **Calibrate once, document it in code constants.**

## Validation gates (see `07` for the full plan)

- The exported glb re-imports (trimesh/pygltflib and a glTF viewer) with the mesh **attached**
  to the skeleton.
- Posing any bone deforms the mesh **locally and correctly** (no collapse, no detachment).
- Numerical parity: same input mesh through upstream `demo.py` vs this exporter yields
  matching joints, parents, and top-4 weights (small tolerance).
- A full round-trip through Kimodo retarget animates without the mesh exploding.
</content>
