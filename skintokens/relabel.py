"""Structural humanoid skeleton relabeler (pure Python / numpy).

SkinTokens emits generic bone names ``bone_0 … bone_N``. Downstream animation
tooling (Kimodo/Mixamo) matches bones by name (``mixamorig:Hips``, …). This
module recognizes the humanoid core of a generated skeleton **by topology**
(parents + descendant counts), using joint position only for left/right and
thumb disambiguation, and renames the ~22 core body bones per a convention.

See ``spec/04-relabeler-spec.md`` for the algorithm and the empirical validation
(4 rigs, identical DFS topology; only finger fan-out varies). The reference C++
recognizer is ``references/skin-tokens.cpp/src/retarget.cpp``.

Convention is table-driven (``NAME_TABLES``); Mixamo and UE5 ship here. Indices
are never reordered — only names change — so exported JOINTS_0 (built from joint
indices) stays consistent with the renamed glTF node names.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Axis convention. joints are (J, 3) in the Asset's native frame.
#   side = the left/right axis index (x by default).
#   up   = the head/foot axis index (y by default; only used for thumb pick).
# For all 4 validated rigs, +side == character-Left. If a first real animation
# comes out mirrored, flip LEFT_IS_POSITIVE once (spec/04, "Left/Right caveat").
# ---------------------------------------------------------------------------
SIDE_AXIS = 0
UP_AXIS = 1
LEFT_IS_POSITIVE = True


# ---------------------------------------------------------------------------
# Per-convention name tables. Each maps a structural role -> the bone name.
# Chains (spine, neck, arm, leg, finger) are ordered proximal -> distal. ``{side}``
# is filled with the side token; finger names are ``.format(side=..., i=joint#)``.
# ---------------------------------------------------------------------------

_MIXAMO = {
    "prefix": "mixamorig:",
    "hips": "Hips",
    "spine": ["Spine", "Spine1", "Spine2"],
    "neck": ["Neck"],
    "head": "Head",
    "side_token": {"L": "Left", "R": "Right"},
    "arm": ["{side}Shoulder", "{side}Arm", "{side}ForeArm", "{side}Hand"],
    "leg": ["{side}UpLeg", "{side}Leg", "{side}Foot", "{side}ToeBase"],
    "fingers": {
        "thumb": "{side}HandThumb{i}",
        "index": "{side}HandIndex{i}",
        "middle": "{side}HandMiddle{i}",
        "ring": "{side}HandRing{i}",
        "pinky": "{side}HandPinky{i}",
    },
    "finger_start": 1,  # LeftHandThumb1, ...
}

_UE5 = {
    "prefix": "",
    "hips": "pelvis",
    "spine": ["spine_01", "spine_02", "spine_03"],
    "neck": ["neck_01"],
    "head": "head",
    "side_token": {"L": "l", "R": "r"},
    "arm": ["clavicle_{side}", "upperarm_{side}", "lowerarm_{side}", "hand_{side}"],
    "leg": ["thigh_{side}", "calf_{side}", "foot_{side}", "ball_{side}"],
    "fingers": {
        "thumb": "thumb_{i:02d}_{side}",
        "index": "index_{i:02d}_{side}",
        "middle": "middle_{i:02d}_{side}",
        "ring": "ring_{i:02d}_{side}",
        "pinky": "pinky_{i:02d}_{side}",
    },
    "finger_start": 1,  # thumb_01_l, ...
}

NAME_TABLES: Dict[str, dict] = {"mixamo": _MIXAMO, "ue5": _UE5}

# Fingers in x-sorted order for the standard 5-finger hand (thumb handled by y).
_FINGER_ORDER_5 = ["index", "middle", "ring", "pinky"]
# Special case: a 3-finger hand is [thumb, index, pinky] (spec/04 step 8).
_FINGER_ORDER_3 = ["index", "pinky"]


def build_children(parents) -> tuple:
    """Return ({parent_index: [child_indices]}, root_index) from a parents array."""
    parents = np.asarray(parents)
    children: Dict[int, List[int]] = {i: [] for i in range(len(parents))}
    root = None
    for i, p in enumerate(parents):
        p = int(p)
        if p < 0:
            root = i
        else:
            children[p].append(i)
    if root is None:
        raise ValueError("no root joint (parent == -1) found")
    return children, root


def label_humanoid(
    joints,
    parents,
    convention: str = "mixamo",
    with_fingers: bool = True,
) -> Dict[int, str]:
    """Recognize the humanoid core and return ``{joint_index: name}``.

    Only recognized core body bones (and optionally fingers) are in the map;
    unrecognized joints are omitted (caller keeps their original names).

    joints:  (J, 3) float positions in the Asset's native frame.
    parents: (J,)   int, -1 for the root.
    convention: key into ``NAME_TABLES`` ('mixamo' or 'ue5').
    """
    if convention not in NAME_TABLES:
        raise ValueError(
            f"unknown convention {convention!r} (have {sorted(NAME_TABLES)})"
        )
    table = NAME_TABLES[convention]
    joints = np.asarray(joints, dtype=np.float64)
    children, root = build_children(parents)

    @lru_cache(maxsize=None)
    def n_desc(n: int) -> int:
        """Number of descendants (subtree size excluding n itself)."""
        return sum(1 + n_desc(c) for c in children[n])

    def walk_chain(start: int, deepest: bool = True) -> List[int]:
        """Follow links from ``start`` until a branch (>=2 children) or a leaf.

        Includes the branch/leaf bone as the last element. When a step has
        multiple children (shouldn't for a clean chain), follow the deepest.
        """
        chain = [start]
        cur = start
        while len(children[cur]) == 1:
            cur = children[cur][0]
            chain.append(cur)
        return chain

    def pick_side(bones: List[int], want_left: bool) -> int:
        """The bone on the character's left/right, by side-axis position."""
        want_positive = want_left == LEFT_IS_POSITIVE
        key = (lambda b: joints[b][SIDE_AXIS])
        return (max if want_positive else min)(bones, key=key)

    prefix = table["prefix"]

    def full(name: str) -> str:
        return prefix + name

    m: Dict[int, str] = {}

    # 1. Pelvis = first bone (DFS/index order) with >= 3 children; else root.
    pelvis = next((i for i in range(len(joints)) if len(children[i]) >= 3), root)
    m[pelvis] = full(table["hips"])

    kids = children[pelvis]
    if len(kids) < 3:
        # Not a recognizable humanoid pelvis; return just the hips label.
        return m

    # 2. Pelvis children split by descendant count: largest subtree = spine root,
    #    the rest are legs (there should be exactly two).
    spine_root = max(kids, key=n_desc)
    legs = [b for b in kids if b != spine_root]

    # 3. Spine chain: single-child walk from spine_root until a branch = chest.
    spine = walk_chain(spine_root)
    chest = spine[-1]
    for i, b in enumerate(spine):
        names = table["spine"]
        m[b] = full(names[min(i, len(names) - 1)])

    # 4. Chest children split by descendant count: smallest subtree = neck chain,
    #    the rest are arms.
    cc = children[chest]
    if len(cc) >= 3:
        neck_root = min(cc, key=n_desc)
        arms = [b for b in cc if b != neck_root]

        # 5. Neck chain: walk to a leaf = head; intermediates are neck joints.
        neck = walk_chain(neck_root)
        neck_names = table["neck"]
        for i, b in enumerate(neck[:-1] if len(neck) > 1 else neck):
            m[b] = full(neck_names[min(i, len(neck_names) - 1)])
        if len(neck) > 1:
            m[neck[-1]] = full(table["head"])

        # 6. Arms: left/right by side axis; walk shoulder->arm->forearm->hand.
        _label_limb_pair(m, arms, table, "arm", children, joints, walk_chain, pick_side, full)
        if with_fingers:
            _label_fingers(m, arms, table, children, joints, pick_side, full)

    # 7. Legs: left/right by side axis; walk upleg->leg->foot->toe.
    _label_limb_pair(m, legs, table, "leg", children, joints, walk_chain, pick_side, full)

    return m


def _label_limb_pair(
    m, roots, table, kind, children, joints, walk_chain, pick_side, full,
):
    """Label a symmetric limb pair (arms or legs) from their two root bones."""
    if len(roots) != 2:
        return
    names = table[kind]
    tok = table["side_token"]
    for want_left in (True, False):
        rb = pick_side(roots, want_left)
        side = tok["L"] if want_left else tok["R"]
        chain = walk_chain(rb)
        for b, nm in zip(chain, names):
            m[b] = full(nm.format(side=side))


def _label_fingers(m, arm_roots, table, children, joints, pick_side, full):
    """Label fingers off each hand (thumb = min up-axis; rest by side axis)."""
    if len(arm_roots) != 2:
        return
    tok = table["side_token"]
    fnames = table["fingers"]
    start = table["finger_start"]

    for want_left in (True, False):
        # Rebuild the arm chain to find the hand (last of a 4-step single walk).
        rb = pick_side(arm_roots, want_left)
        cur = rb
        for _ in range(3):
            if len(children[cur]) == 1:
                cur = children[cur][0]
            elif children[cur]:
                cur = max(children[cur], key=lambda c: joints[c][UP_AXIS])
            else:
                break
        hand = cur
        finger_roots = children[hand]
        if not finger_roots:
            continue

        side = tok["L"] if want_left else tok["R"]
        # Thumb = finger root with the minimum up-axis value.
        thumb = min(finger_roots, key=lambda b: joints[b][UP_AXIS])
        others = [b for b in finger_roots if b != thumb]
        # Remaining fingers sorted by side axis, reversed per side so index is
        # nearest the thumb on both hands.
        others.sort(key=lambda b: joints[b][SIDE_AXIS], reverse=not want_left)

        if len(finger_roots) == 3:
            order = ["thumb"] + _FINGER_ORDER_3
            roots_in_order = [thumb] + others
        else:
            order = ["thumb"] + _FINGER_ORDER_5[: len(others)]
            roots_in_order = [thumb] + others

        for fname, froot in zip(order, roots_in_order):
            tmpl = fnames.get(fname)
            if tmpl is None:
                continue
            # Walk the finger's own single-child chain, numbering joints.
            cur = froot
            i = start
            while True:
                m[cur] = full(tmpl.format(side=side, i=i))
                kids = children[cur]
                if len(kids) != 1:
                    break
                cur = kids[0]
                i += 1


def relabel_asset(
    asset,
    convention: str = "mixamo",
    with_fingers: bool = True,
) -> Dict[int, str]:
    """Rename the humanoid core bones of ``asset`` in place; return the mapping.

    Reads ``asset.joints`` (from ``matrix_local``) and ``asset.parents``, writes
    the recognized names into ``asset.joint_names`` (creating it as
    ``bone_0 … bone_N`` first if absent). Indices are unchanged.
    """
    if asset.joints is None or asset.parents is None:
        raise ValueError("asset needs joints and parents (rig it first)")
    J = asset.parents.shape[0]
    if asset.joint_names is None:
        asset.joint_names = [f"bone_{i}" for i in range(J)]
    else:
        asset.joint_names = list(asset.joint_names)

    mapping = label_humanoid(
        asset.joints, asset.parents, convention=convention, with_fingers=with_fingers
    )
    for idx, name in mapping.items():
        asset.joint_names[idx] = name
    return mapping
