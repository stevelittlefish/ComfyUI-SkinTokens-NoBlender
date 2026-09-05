"""Rig orientation detection + canonicalization (pure Python / numpy).

SkinTokens emits a generated humanoid in a **random facing** along the up-plane:
some runs the character faces one way, some the other. This has two visible
consequences downstream:

* **Left/Right labels flip.** The relabeler's side test is only correct for one
  facing (see ``relabel.py`` and the old ``LEFT_IS_POSITIVE`` caveat).
* **The character walks backwards.** An animation clip drives the rig along the
  convention's forward axis; if the body faces the opposite way it moonwalks.

Both are the *same* confusion. We resolve it from geometry the model does get
right: **the toes point forward**. ``detect_forward`` reads the toe-vs-foot
offset of the leg chains (thumbs are deliberately NOT used — a mis-oriented rig
tends to drop the thumb too, so it is unreliable exactly when needed). From that
we can (a) label left/right anatomically at any facing and (b) yaw the whole rig
to a canonical facing before export so it never walks backwards.

Only the toe cue is used; if the legs are missing/degenerate we return ``None``
and callers fall back to prior behavior rather than guessing.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# Axis convention (matches relabel.py): up = head/foot axis, side = left/right.
UP_AXIS = 1
# Canonical facing the exported rig is rotated to, in the up-plane. Empirically
# the animation engine (Kimodo/Mixamo) drives the character along -Z, so a rig
# whose toes point -Z walks forward. Rigs generated facing +Z are the ones that
# came out walking backwards.
CANONICAL_FORWARD = np.array([0.0, 0.0, -1.0])


def _children_root(parents) -> Tuple[Dict[int, List[int]], int]:
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


def _leg_roots(joints, children, root) -> List[int]:
    """The two leg root joints off the pelvis (topology only, no names)."""
    pelvis = next((i for i in range(len(joints)) if len(children[i]) >= 3), root)
    kids = children[pelvis]
    if len(kids) < 3:
        return []

    # Descendant count per subtree; the largest is the spine, the rest are legs.
    memo: Dict[int, int] = {}

    def n_desc(n: int) -> int:
        if n not in memo:
            memo[n] = sum(1 + n_desc(c) for c in children[n])
        return memo[n]

    spine_root = max(kids, key=n_desc)
    return [b for b in kids if b != spine_root]


def _chain_to_leaf(start: int, children) -> List[int]:
    chain = [start]
    cur = start
    while len(children[cur]) == 1:
        cur = children[cur][0]
        chain.append(cur)
    return chain


def detect_forward(joints, parents, up_axis: int = UP_AXIS) -> Optional[np.ndarray]:
    """Unit forward vector (in the up-plane) from the toe->foot offset, or None.

    Uses only the legs: for each leg chain, ``toe - foot`` (leaf minus the joint
    before it) points forward. Averaged over both legs, projected onto the
    horizontal plane and normalized. Returns ``None`` if the legs are missing or
    the cue is too weak/ambiguous to trust.
    """
    joints = np.asarray(joints, dtype=np.float64)
    try:
        children, root = _children_root(parents)
    except ValueError:
        return None

    legs = _leg_roots(joints, children, root)
    if len(legs) != 2:
        return None

    fwd = np.zeros(3, dtype=np.float64)
    for leg in legs:
        chain = _chain_to_leaf(leg, children)
        if len(chain) < 2:
            return None  # no toe distinct from foot
        toe, foot = chain[-1], chain[-2]
        fwd += joints[toe] - joints[foot]

    fwd[up_axis] = 0.0  # keep only the horizontal component
    n = np.linalg.norm(fwd)
    if n < 1e-6:
        return None  # toes sit directly under the feet -> no usable direction
    return fwd / n


def left_dir(forward, up_axis: int = UP_AXIS) -> np.ndarray:
    """Character-left unit vector = up x forward (right-handed, Y-up)."""
    up = np.zeros(3)
    up[up_axis] = 1.0
    return np.cross(up, np.asarray(forward, dtype=np.float64))


def yaw_to_canonical(
    forward, up_axis: int = UP_AXIS, target=CANONICAL_FORWARD
) -> np.ndarray:
    """3x3 rotation about the up axis mapping ``forward`` onto ``target``.

    Both are projected to the up-plane, so this is a pure yaw (no tilt): it never
    changes which way is up, only which way the character faces.
    """
    forward = np.asarray(forward, dtype=np.float64).copy()
    target = np.asarray(target, dtype=np.float64).copy()
    forward[up_axis] = 0.0
    target[up_axis] = 0.0
    fn, tn = np.linalg.norm(forward), np.linalg.norm(target)
    if fn < 1e-9 or tn < 1e-9:
        return np.eye(3)
    f = forward / fn
    t = target / tn
    cos = float(np.clip(f @ t, -1.0, 1.0))
    axis = np.zeros(3)
    axis[up_axis] = 1.0
    # Signed angle from f to t about the up axis.
    sin = float(np.dot(axis, np.cross(f, t)))
    angle = np.arctan2(sin, cos)
    if abs(angle) < 1e-4:
        return np.eye(3)  # already facing canonically (ignore float dust)
    c, s = np.cos(angle), np.sin(angle)
    R = np.eye(3)
    # Rotation about the up axis (Rodrigues specialized to a principal axis).
    other = [i for i in range(3) if i != up_axis]
    a, b = other  # the two in-plane axes, in order
    R[a, a] = c
    R[b, b] = c
    R[a, b] = s
    R[b, a] = -s
    return R


def canonicalize(
    vertices,
    normals,
    joints,
    parents,
    up_axis: int = UP_AXIS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Yaw mesh + rig to the canonical facing. Returns (V, N, J, rotated).

    ``rotated`` is False (arrays returned unchanged) when forward can't be
    detected or the rig already faces canonically, so a rig we can't read is
    never silently mis-rotated.
    """
    V = np.asarray(vertices, dtype=np.float64)
    N = None if normals is None else np.asarray(normals, dtype=np.float64)
    J = np.asarray(joints, dtype=np.float64)

    forward = detect_forward(J, parents, up_axis=up_axis)
    if forward is None:
        return V, N, J, False

    R = yaw_to_canonical(forward, up_axis=up_axis)
    if np.allclose(R, np.eye(3), atol=1e-9):
        return V, N, J, False

    Vr = V @ R.T
    Nr = None if N is None else N @ R.T
    Jr = J @ R.T
    return Vr, Nr, Jr, True
