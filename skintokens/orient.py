"""Rig orientation detection (pure Python / numpy).

Used only to decide **left vs right** anatomically. A humanoid skeleton is
bilaterally symmetric, so which arm is "left" depends on which way the character
faces — which SkinTokens does not fix (it can emit either facing). We recover the
facing from geometry the model gets right: **the toes point forward**.
``detect_forward`` reads the toe-vs-foot offset of the leg chains (thumbs are
deliberately NOT used — a mis-oriented rig tends to drop the thumb too, so it is
unreliable exactly when needed), and ``left_dir`` turns that into the
character-left axis for the relabeler.

Only the toe cue is used; if the legs are missing/degenerate ``detect_forward``
returns ``None`` and the relabeler falls back to its fixed side-axis constant.

NB: we do NOT rotate geometry to a "canonical" facing. That was tried and
reverted — forcing a facing turned the character backwards in previews and made
it walk backwards; the native SkinTokens orientation is the correct one.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# Axis convention (matches relabel.py): up = head/foot axis, side = left/right.
UP_AXIS = 1


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
