"""Orientation detection + canonicalization (toe-based forward).

SkinTokens emits humanoids at a random facing each run; a back-facing rig walks
backwards and its left/right read mirrored. These tests build a humanoid, yaw it
to arbitrary facings, and check that:

  * forward is recovered from the toe->foot offset,
  * the relabeler tags the *same physical bone* Left/Right at any facing, and
  * export canonicalization yaws the rig back to the convention's forward.

See spec/04 (Left/Right caveat) and skintokens/orient.py.
"""

import numpy as np
import pytest

from skintokens import orient, relabel

# Reuse the validated humanoid builder from the relabel tests.
from tests.test_relabel import _humanoid


def _yaw(joints, deg):
    """Rotate joints about the up (y) axis by ``deg`` degrees."""
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    return joints @ R.T


def test_detect_forward_points_at_toes():
    joints, parents, _ = _humanoid(with_fingers=False)
    fwd = orient.detect_forward(joints, parents)
    # Builder puts toes at foot + [0,0,0.1] -> forward is +z.
    assert fwd is not None
    np.testing.assert_allclose(fwd, [0, 0, 1], atol=1e-6)


def test_detect_forward_follows_yaw():
    joints, parents, _ = _humanoid(with_fingers=False)
    fwd = orient.detect_forward(_yaw(joints, 180), parents)
    np.testing.assert_allclose(fwd, [0, 0, -1], atol=1e-6)
    fwd90 = orient.detect_forward(_yaw(joints, 90), parents)
    np.testing.assert_allclose(fwd90, [1, 0, 0], atol=1e-6)


def test_detect_forward_none_without_legs():
    # A lone root (no pelvis with >=3 children) has no usable toe cue.
    joints = np.array([[0.0, 1.0, 0.0]])
    parents = np.array([-1])
    assert orient.detect_forward(joints, parents) is None


@pytest.mark.parametrize("deg", [0, 90, 180, 270, 37])
def test_labels_track_physical_bone_at_any_facing(deg):
    """The bone labeled LeftShoulder must be the same index at every facing."""
    joints, parents, _ = _humanoid(with_fingers=False)
    base = relabel.label_humanoid(joints, parents)
    left0 = next(i for i, n in base.items() if n == "mixamorig:LeftShoulder")
    right0 = next(i for i, n in base.items() if n == "mixamorig:RightShoulder")

    m = relabel.label_humanoid(_yaw(joints, deg), parents)
    left = next(i for i, n in m.items() if n == "mixamorig:LeftShoulder")
    right = next(i for i, n in m.items() if n == "mixamorig:RightShoulder")
    assert (left, right) == (left0, right0), f"L/R flipped at {deg} deg"


def test_canonicalize_yaws_back_to_forward():
    # Builder faces +z (non-canonical) -> canonicalize should yaw it to -z.
    joints, parents, _ = _humanoid(with_fingers=False)
    verts = joints.copy()  # stand in for mesh verts (share the frame)
    Vr, Nr, Jr, rotated = orient.canonicalize(verts, None, joints.copy(), parents)
    assert rotated
    np.testing.assert_allclose(
        orient.detect_forward(Jr, parents), orient.CANONICAL_FORWARD, atol=1e-6
    )
    # Mesh verts and joints rotate by the same rigid transform.
    np.testing.assert_allclose(Vr, Jr, atol=1e-6)


def test_canonicalize_noop_when_already_forward():
    # A rig already facing -z (canonical): yaw the +z builder by 180.
    joints, parents, _ = _humanoid(with_fingers=False)
    canon = _yaw(joints, 180)
    _, _, Jr, rotated = orient.canonicalize(canon.copy(), None, canon.copy(), parents)
    assert not rotated
    np.testing.assert_allclose(Jr, canon, atol=1e-6)
