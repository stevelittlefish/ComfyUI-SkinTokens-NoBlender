"""Phase 4 local tests: structural humanoid relabeler (Gate D).

No GPU, no Blender. We build synthetic humanoid rigs with the empirically
validated DFS topology (spec/04) and check that the topology-driven recognizer
labels the ~22 core body bones (+ fingers) correctly, in both Mixamo and UE5
conventions, and that left/right is decided by position (mirror test).
"""

import numpy as np
import pytest

from skintokens import relabel
from skintokens.vendor.rig_package.info.asset import Asset


# --- rig builder -----------------------------------------------------------


def _humanoid(with_fingers=True, n_fingers=5, mirror=False):
    """Build a T-pose humanoid: joints (J,3, y-up, +x=left) and parents (J,).

    DFS order matches the validated skeleton: Hips, spine x3, neck, head, then
    left arm, right arm, then legs last. Fingers (optional) fan off each hand,
    spread along the side (x) axis with the thumb lower in y.
    """
    joints = []
    parents = []
    names = []  # for readability in failures

    def add(pos, parent, tag):
        joints.append(pos)
        parents.append(parent)
        names.append(tag)
        return len(joints) - 1

    s = -1.0 if mirror else 1.0  # flip the side axis to test L/R by position

    hips = add([0, 1.0, 0], -1, "hips")
    sp0 = add([0, 1.2, 0], hips, "spine")
    sp1 = add([0, 1.4, 0], sp0, "spine1")
    chest = add([0, 1.6, 0], sp1, "spine2")
    neck = add([0, 1.8, 0], chest, "neck")
    add([0, 2.0, 0], neck, "head")

    def arm(sign, side_tag):
        sh = add([sign * 0.2, 1.7, 0], chest, f"{side_tag}shoulder")
        up = add([sign * 0.5, 1.7, 0], sh, f"{side_tag}arm")
        fore = add([sign * 0.8, 1.7, 0], up, f"{side_tag}forearm")
        hand = add([sign * 1.1, 1.7, 0], fore, f"{side_tag}hand")
        if with_fingers:
            # thumb: lower in y. others: spread along x above the thumb.
            add([sign * 1.2, 1.55, 0], hand, f"{side_tag}thumb")
            spread = ["index", "middle", "ring", "pinky"][: n_fingers - 1]
            for k, fn in enumerate(spread):
                add([sign * (1.2 + 0.1 * (k + 1)), 1.7, 0], hand, f"{side_tag}{fn}")
        return hand

    def leg(sign, side_tag):
        thigh = add([sign * 0.1, 0.9, 0], hips, f"{side_tag}upleg")
        calf = add([sign * 0.1, 0.5, 0], thigh, f"{side_tag}leg")
        foot = add([sign * 0.1, 0.1, 0], calf, f"{side_tag}foot")
        add([sign * 0.1, 0.0, 0.1], foot, f"{side_tag}toe")

    # left first (matches DFS: left arm, right arm, then legs)
    arm(s * 1.0, "L_")
    arm(s * -1.0, "R_")
    leg(s * 1.0, "L_")
    leg(s * -1.0, "R_")

    return np.array(joints, dtype=np.float64), np.array(parents, dtype=np.int64), names


def _make_asset(joints, parents):
    J = parents.shape[0]
    asset = Asset(parents=parents)
    ml = np.tile(np.eye(4, dtype=np.float32), (J, 1, 1))
    ml[:, :3, 3] = joints
    asset.matrix_local = ml
    return asset


# --- core body labeling ----------------------------------------------------


def test_mixamo_core_bones():
    joints, parents, _ = _humanoid(with_fingers=False)
    m = relabel.label_humanoid(joints, parents, convention="mixamo")

    assert m[0] == "mixamorig:Hips"
    assert m[1] == "mixamorig:Spine"
    assert m[2] == "mixamorig:Spine1"
    assert m[3] == "mixamorig:Spine2"
    assert m[4] == "mixamorig:Neck"
    assert m[5] == "mixamorig:Head"
    # left arm (indices 6..9), +x side
    assert m[6] == "mixamorig:LeftShoulder"
    assert m[7] == "mixamorig:LeftArm"
    assert m[8] == "mixamorig:LeftForeArm"
    assert m[9] == "mixamorig:LeftHand"
    # right arm (10..13)
    assert m[10] == "mixamorig:RightShoulder"
    assert m[13] == "mixamorig:RightHand"
    # legs last 8: left 14..17, right 18..21
    assert m[14] == "mixamorig:LeftUpLeg"
    assert m[15] == "mixamorig:LeftLeg"
    assert m[16] == "mixamorig:LeftFoot"
    assert m[17] == "mixamorig:LeftToeBase"
    assert m[18] == "mixamorig:RightUpLeg"
    assert m[21] == "mixamorig:RightToeBase"


def test_ue5_core_bones():
    joints, parents, _ = _humanoid(with_fingers=False)
    m = relabel.label_humanoid(joints, parents, convention="ue5")

    assert m[0] == "pelvis"
    assert m[1] == "spine_01"
    assert m[3] == "spine_03"
    assert m[4] == "neck_01"
    assert m[5] == "head"
    assert m[6] == "clavicle_l"
    assert m[9] == "hand_l"
    assert m[10] == "clavicle_r"
    assert m[14] == "thigh_l"
    assert m[17] == "ball_l"
    assert m[21] == "ball_r"


def test_mirror_swaps_left_right():
    """Flipping the side axis must swap all Left/Right labels."""
    joints, parents, _ = _humanoid(with_fingers=False, mirror=True)
    m = relabel.label_humanoid(joints, parents, convention="mixamo")

    # index 6 was built on the (now negated) "left" side -> should read Right.
    assert m[6] == "mixamorig:RightShoulder"
    assert m[9] == "mixamorig:RightHand"
    assert m[10] == "mixamorig:LeftShoulder"
    assert m[14] == "mixamorig:RightUpLeg"
    assert m[18] == "mixamorig:LeftUpLeg"


# --- fingers ---------------------------------------------------------------


def test_fingers_5():
    joints, parents, _ = _humanoid(with_fingers=True, n_fingers=5)
    m = relabel.label_humanoid(joints, parents, convention="mixamo", with_fingers=True)

    # left hand fingers follow the hand (index 9). thumb is min-y.
    labels = {m[i] for i in m if "Hand" in m[i] and m[i] != "mixamorig:LeftHand"
              and m[i] != "mixamorig:RightHand"}
    for base in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
        assert f"mixamorig:LeftHand{base}1" in labels, base
        assert f"mixamorig:RightHand{base}1" in labels, base


def test_fingers_3_special_case():
    """A 3-finger hand -> [thumb, index, pinky] (spec/04 step 8)."""
    joints, parents, _ = _humanoid(with_fingers=True, n_fingers=3)
    m = relabel.label_humanoid(joints, parents, convention="ue5", with_fingers=True)

    vals = set(m.values())
    assert "thumb_01_l" in vals
    assert "index_01_l" in vals
    assert "pinky_01_l" in vals
    # middle/ring must NOT appear on a 3-finger hand
    assert "middle_01_l" not in vals
    assert "ring_01_l" not in vals


def test_fingers_off_disables():
    joints, parents, _ = _humanoid(with_fingers=True, n_fingers=5)
    m = relabel.label_humanoid(joints, parents, with_fingers=False)
    assert not any("Thumb" in v or "Index" in v for v in m.values())


# --- asset integration -----------------------------------------------------


def test_relabel_asset_in_place():
    joints, parents, _ = _humanoid(with_fingers=False)
    asset = _make_asset(joints, parents)
    mapping = relabel.relabel_asset(asset, convention="mixamo")

    assert asset.joint_names[0] == "mixamorig:Hips"
    assert asset.joint_names[9] == "mixamorig:LeftHand"
    # unrecognized joints keep bone_N; but this rig is fully recognized -> all named.
    assert len(mapping) >= 22
    # indices are stable: joint_names length == joint count.
    assert len(asset.joint_names) == parents.shape[0]


def test_relabel_asset_preserves_unrecognized():
    """Extra non-humanoid joints keep their default bone_N name."""
    joints, parents, _ = _humanoid(with_fingers=False)
    # append a stray joint past the left toe (index 17): the 4-name leg chain
    # zips to the first 4 bones, leaving this 5th one unrecognized.
    extra = np.vstack([joints, [[0.1, -0.1, 0.2]]])
    extra_parents = np.append(parents, 17)
    asset = _make_asset(extra, extra_parents)
    relabel.relabel_asset(asset, convention="mixamo")
    assert asset.joint_names[-1] == f"bone_{extra_parents.shape[0] - 1}"


def test_unknown_convention_raises():
    joints, parents, _ = _humanoid(with_fingers=False)
    with pytest.raises(ValueError):
        relabel.label_humanoid(joints, parents, convention="bogus")


def test_pelvis_below_root_armature_bone():
    """Pelvis = first bone with >=3 children, surviving a root/armature bone."""
    joints, parents, _ = _humanoid(with_fingers=False)
    # insert an armature root above the hips: shift everyone, add root at index 0.
    J = parents.shape[0]
    new_joints = np.vstack([[[0, 0.0, 0]], joints])
    new_parents = np.array([-1] + [(p + 1 if p >= 0 else 1) for p in parents])
    # old hips (was root) now parents to the new armature root.
    new_parents[1] = 0
    m = relabel.label_humanoid(new_joints, new_parents, convention="mixamo")
    # the humanoid hips is old index 0 -> now index 1.
    assert m[1] == "mixamorig:Hips"
    assert 0 not in m  # the armature root is not labeled
