# Structural Skeleton Relabeler Spec

## Purpose

SkinTokens emits generic bone names `bone_0 … bone_N`. Kimodo (and Mixamo tooling in
general) animate skeletons named `mixamorig:Hips`, `mixamorig:LeftArm`, etc. The relabeler
renames the generated humanoid skeleton's core body bones to `mixamorig:*` so the rigged
output is drop-in animatable — **no retargeting engine changes needed downstream**.

This was empirically validated (below). It is an **optional node toggle** (default on).

## Key finding: the skeleton is structurally deterministic

Four SkinTokens outputs were inspected (sci-fi-dude 52 bones, peasant 34, knight 34,
industrial robot 41). Findings:

- **Bone counts vary** — a static "bone_5 = Head" index map does NOT work. The variation is
  entirely in **finger count** (peasant/knight got 2 fingers/hand, robot 3+stub, sci-fi 5).
- **Topology and depth-first traversal order are identical every time.** The first 10 bones
  are the same across all characters:
  ```
  bone_0 Hips → bone_1 Spine → bone_2 Spine1 → bone_3 Spine2 (branch)
  bone_4 Neck → bone_5 Head
  bone_6 LeftShoulder → 7 LeftArm → 8 LeftForeArm → 9 LeftHand
  ```
  then the right-arm chain, then the **legs are always the last 8 bones** (the only downward
  children of Hips). Only the finger fan-out off each Hand varies.
- Conclusion: identify bones by **hierarchy + position**, not by index. This labels the ~22
  core body bones — exactly the ones Kimodo drives (its SOMA skeleton has ~30 joints and
  barely uses fingers). Fingers can be left unnamed/unmapped.

## Multi-convention output (adopted from prior art — see `08`)

The relabeler takes a **`convention`** parameter and emits names from a per-convention table.
Support at least **Mixamo** and **UE5**; keep it table-driven so more can be added.

| Role | Mixamo | UE5 |
|------|--------|-----|
| pelvis/root | `mixamorig:Hips` | `pelvis` |
| spine chain | `Spine`, `Spine1`, `Spine2` | `spine_01`, `spine_02`, … |
| neck / head | `Neck` / `Head` | `neck_01` / `head` |
| shoulder→hand (L) | `LeftShoulder`,`LeftArm`,`LeftForeArm`,`LeftHand` | `clavicle_l`,`upperarm_l`,`lowerarm_l`,`hand_l` |
| upleg→toe (L) | `LeftUpLeg`,`LeftLeg`,`LeftFoot`,`LeftToeBase` | `thigh_l`,`calf_l`,`foot_l`,`ball_l` |
| fingers (L) | `LeftHandThumb1…`,`LeftHandIndex1…` | `thumb_01_l`,`index_01_l`,… |

(Right side mirrors: `Right*` / `_r`. Confirm exact UE5 leg/ball names against a UE5 rig.)
The `mixamorig:` prefix on Mixamo names is what Kimodo matches on — keep it.

## The algorithm (rules — topology-driven for axis robustness)

Operate directly on the model's `joints` (J×3 positions) and `parents` (J,) arrays — **no
Blender needed**. Prefer **hierarchy/descendant-count** for structural decisions and use
position only for left/right and thumb; this minimizes dependence on axis calibration. Let
`side` = the left/right axis index (calibrate to the Asset's native frame; see `03`).

1. **Pelvis** = the first bone with **≥3 children** (root-agnostic; survives a root/armature
   bone above the pelvis). Fall back to the root (parent == -1) if none qualifies.
2. Pelvis children split by **descendant count**: the **largest subtree** is the **Spine**
   root; the remaining children are the two **legs** (Left = larger `side`, Right = smaller).
3. **Spine chain**: from the spine root, follow single-child links until a bone with ≥3
   children — that branch bone is the **chest**; assign spine names in order along the way.
4. Chest children split by **descendant count**: the **smallest subtree** is the **Neck**
   chain; the rest are the **arms** (Left = larger `side`, Right = smaller).
5. **Neck chain**: walk until a childless bone = **Head**; intermediate bones are neck joints.
6. **Arms**: walk each arm root down single-child links →
   `Shoulder/clavicle, Arm/upperarm, ForeArm/lowerarm, Hand`. (Use deepest-subtree child if a
   step branches early.)
7. **Legs**: walk each leg root → `UpLeg/thigh, Leg/calf, Foot, ToeBase/ball`.
8. **Fingers (optional)**: off each Hand, thumb = finger with **min `head.y`**; remaining
   fingers sorted by `head.x` (reversed per side). Special-case a **3-finger hand** →
   `[thumb, index, pinky]`. Name per convention (`LeftHandThumb1/2/3`, `thumb_01_l`, …).

> The earlier draft used the up-axis to find the spine; descendant-count (above) is more
> robust and is the adopted approach. Keep the reference code below but prefer topology.

### Left/Right caveat
For all 4 validated models, **+side == character-Left**. If a first real animation comes out
mirrored, flip the L/R rule once — it is then permanent. Expose it as a constant/param.

## Reference implementation (pure Python / numpy — adapt, don't blindly paste)

```python
import numpy as np
from functools import lru_cache

def build_children(parents):
    children = {i: [] for i in range(len(parents))}
    root = None
    for i, p in enumerate(parents):
        if p is None or p < 0:
            root = i
        else:
            children[int(p)].append(i)
    return children, root

def label_humanoid(joints, parents, up=2, side=0, left_is_positive=True):
    """joints: (J,3) float; parents: (J,) int, -1 for root.
       up/side: axis indices in the Asset's native frame (CALIBRATE — see 03).
       Returns {bone_index: 'mixamorig:Name'} for the ~22 core body bones."""
    joints = np.asarray(joints)
    children, root = build_children(parents)

    @lru_cache(maxsize=None)
    def depth(n):
        return 0 if not children[n] else 1 + max(depth(c) for c in children[n])

    def walk(start, n):
        chain, cur = [start], start
        while len(chain) < n and children[cur]:
            cur = max(children[cur], key=depth)
            chain.append(cur)
        return chain

    m = {root: "mixamorig:Hips"}
    kids = children[root]
    spine_start = max(kids, key=lambda b: joints[b][up])
    legs = [b for b in kids if b != spine_start]

    spine, cur = [spine_start], spine_start
    while len(children[cur]) < 2 and children[cur]:
        cur = children[cur][0]; spine.append(cur)
    chest = cur
    for i, b in enumerate(spine):
        m[b] = ["mixamorig:Spine", "mixamorig:Spine1", "mixamorig:Spine2"][min(i, 2)]

    cc = children[chest]
    neck = min(cc, key=lambda b: abs(joints[b][side]))
    arms = [b for b in cc if b != neck]
    m[neck] = "mixamorig:Neck"
    if children[neck]:
        m[walk(neck, 2)[-1]] = "mixamorig:Head"

    def pick(bones, want_left):
        return (max if want_left == left_is_positive else min)(bones, key=lambda b: joints[b][side])
    # left = larger `side` when left_is_positive
    L_arm = max(arms, key=lambda b: joints[b][side]); R_arm = min(arms, key=lambda b: joints[b][side])
    for name, rb in (("Left", L_arm), ("Right", R_arm)):
        for b, nm in zip(walk(rb, 4),
                         [f"mixamorig:{name}Shoulder", f"mixamorig:{name}Arm",
                          f"mixamorig:{name}ForeArm", f"mixamorig:{name}Hand"]):
            m[b] = nm
    L_leg = max(legs, key=lambda b: joints[b][side]); R_leg = min(legs, key=lambda b: joints[b][side])
    for name, rb in (("Left", L_leg), ("Right", R_leg)):
        for b, nm in zip(walk(rb, 4),
                         [f"mixamorig:{name}UpLeg", f"mixamorig:{name}Leg",
                          f"mixamorig:{name}Foot", f"mixamorig:{name}ToeBase"]):
            m[b] = nm
    return m
```

> Note: if `left_is_positive` handling looks redundant above, simplify — the intent is: Left
> = the arm/leg root on the +side, Right = the -side. Keep one clear convention.

## Validated output (all four models produced this mapping correctly)

```
bone_0  Hips        bone_1 Spine   bone_2 Spine1   bone_3 Spine2
bone_4  Neck        bone_5 Head
(Left)  Shoulder→Arm→ForeArm→Hand
(Right) Shoulder→Arm→ForeArm→Hand   (indices shift with finger count)
(Left)  UpLeg→Leg→Foot→ToeBase      (always the last 8 bones)
(Right) UpLeg→Leg→Foot→ToeBase
```

## Where the names must be applied

- Rename the **joint/bone names** in the exported skeleton, AND
- Rename the corresponding **skin weight groups / joint references** consistently, or the
  skin detaches (same class of bug as `03`). Since export builds JOINTS_0 from joint indices
  (not names), the practical requirement is: the exported glTF **node names** for the joints
  become `mixamorig:*`, and any name-based lookups downstream (Kimodo) resolve. Keep indices
  stable; only names change.

## Reference in the C++ port

`skin-tokens.cpp/src/retarget.cpp` implements "structural recognition" of the humanoid core
plus SOMA30 retargeting. Compare its recognizer against the rules above; adopt any
robustness improvements (its TODO notes it currently supports a "VROID-like humanoid core"
and wants to generalize). Do not fall back to nearest-joint guessing.

## Relationship to Kimodo (context)

Kimodo's `kimodo_retarget_fbx.py` maps its SOMA skeleton to Mixamo via a hardcoded
`SOMA_TO_MIXAMO` dict, and matches bones by **name suffix**. Once this relabeler has named
the SkinTokens skeleton `mixamorig:*`, Kimodo's existing retarget consumes it unchanged.
This is why relabeling — not writing a new retargeter — is the chosen bridge.
</content>
