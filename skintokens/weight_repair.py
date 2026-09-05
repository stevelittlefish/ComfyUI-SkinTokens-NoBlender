"""Deterministic skin-weight coherence repair for rigged glb files.

The generative rigger occasionally leaks a skin influence onto a bone on the far
side of the body — e.g. right-hand finger vertices weighted partly to
``RightToeBase``. At rest this is invisible (nothing has moved), but as soon as
the far bone animates independently, those vertices are dragged toward it and the
mesh tears (the classic "exploding hand"). This module detects and removes such
influences purely geometrically — no model, no template, fully offline.

Principle
---------
Good skinning is *local*: a vertex is only influenced by bones near it, which are
also near each other in the skeleton graph. So an influence is *incoherent* when a
vertex is weighted to two bones that are far apart in the skeleton — measured in
**graph hops**, which is scale-free and needs no thresholds in metres. A hand and
a toe are the maximally distant pair on a humanoid; a hand and its own fingers are
1–2 hops. ``min_hops`` cleanly separates the two (anatomical blends span at most
~4 hops: left-leg↔right-leg through the hips; genuine contamination spans ≥5).

Repair
------
For each incoherent vertex, drop the influence on whichever contaminating bone is
**geometrically farther from the vertex**, then renormalize the surviving weights.
The near, correct bone always survives, so finger articulation is preserved (unlike
a blunt "collapse all fingers to the wrist"). This is a sibling of
:mod:`skintokens.relabel`: a deterministic structural correction on a rigged glb.

This module deliberately depends only on ``numpy`` + ``pygltflib`` (like
``glb_io``), so it imports and unit-tests without ComfyUI / torch present.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

PathLike = Union[str, Path]

# glTF componentType -> numpy dtype (the ones mesh attributes actually use).
_COMPONENT_DTYPE = {
    5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16,
    5125: np.uint32, 5126: np.float32,
}
_TYPE_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
_FLOAT = 5126
_UNSIGNED_SHORT = 5123
_ARRAY_BUFFER = 34962

# A vertex weighted to two bones at least this many skeleton-graph hops apart is
# structurally impossible for good skinning. 5 sits above the widest legitimate
# anatomical blend (left-leg <-> right-leg through the hips = 4 hops).
DEFAULT_MIN_HOPS = 5


@dataclass
class RepairStats:
    """What a repair pass changed — returned for logging / tests."""

    n_vertices: int = 0
    n_vertices_repaired: int = 0
    n_influences_removed: int = 0
    removed_by_bone: Dict[str, int] = field(default_factory=dict)
    comingled_pairs: Dict[Tuple[str, str], int] = field(default_factory=dict)

    @property
    def repaired_fraction(self) -> float:
        return self.n_vertices_repaired / self.n_vertices if self.n_vertices else 0.0


# ---------------------------------------------------------------------------
# glTF plumbing (self-contained; numpy + pygltflib only)
# ---------------------------------------------------------------------------

def _read_accessor(gltf, blob: bytes, accessor_index: int) -> np.ndarray:
    """Decode accessor ``accessor_index`` to a (count, ncomp) numpy array."""
    acc = gltf.accessors[accessor_index]
    if acc.sparse is not None:
        raise NotImplementedError("sparse accessors are not supported")
    dtype = _COMPONENT_DTYPE[acc.componentType]
    ncomp = _TYPE_NCOMP[acc.type]
    comp_bytes = np.dtype(dtype).itemsize
    bv = gltf.bufferViews[acc.bufferView]
    base = (bv.byteOffset or 0) + (acc.byteOffset or 0)
    stride = bv.byteStride or (ncomp * comp_bytes)
    out = np.empty((acc.count, ncomp), dtype=dtype)
    for i in range(acc.count):
        out[i] = np.frombuffer(blob, dtype=dtype, count=ncomp, offset=base + i * stride)
    return out


def _append_accessor(gltf, blob: bytearray, array: np.ndarray, component_type: int,
                     type_str: str, target: Optional[int] = None) -> int:
    """Append ``array`` to the binary blob; register a bufferView + accessor."""
    from pygltflib import Accessor, BufferView

    data = np.ascontiguousarray(array).tobytes()
    byte_offset = len(blob)
    blob.extend(data)
    while len(blob) % 4 != 0:
        blob.append(0)
    bv = BufferView(buffer=0, byteOffset=byte_offset, byteLength=len(data))
    if target is not None:
        bv.target = target
    gltf.bufferViews.append(bv)
    gltf.accessors.append(Accessor(
        bufferView=len(gltf.bufferViews) - 1,
        componentType=component_type,
        count=array.shape[0],
        type=type_str,
    ))
    return len(gltf.accessors) - 1


def _local_matrix(node) -> np.ndarray:
    if node.matrix:
        return np.array(node.matrix, dtype=np.float64).reshape(4, 4).T
    m = np.eye(4, dtype=np.float64)
    if node.rotation:
        x, y, z, w = node.rotation
        m[:3, :3] = [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    if node.scale:
        m[:3, :3] = m[:3, :3] @ np.diag(node.scale)
    m[:3, 3] = node.translation or [0.0, 0.0, 0.0]
    return m


def _world_matrices(gltf) -> List[np.ndarray]:
    parent_of = {}
    for i, node in enumerate(gltf.nodes):
        for c in (node.children or []):
            parent_of[c] = i
    cache: Dict[int, np.ndarray] = {}

    def world(i: int) -> np.ndarray:
        if i in cache:
            return cache[i]
        m = _local_matrix(gltf.nodes[i])
        if i in parent_of:
            m = world(parent_of[i]) @ m
        cache[i] = m
        return m

    return [world(i) for i in range(len(gltf.nodes))]


# ---------------------------------------------------------------------------
# Skeleton graph
# ---------------------------------------------------------------------------

def _hop_distance_matrix(gltf, joints: List[int]) -> np.ndarray:
    """All-pairs skeleton-graph hop distance between joints (BFS per joint)."""
    node_to_j = {n: j for j, n in enumerate(joints)}
    parent_of = {}
    for i, node in enumerate(gltf.nodes):
        for c in (node.children or []):
            parent_of[c] = i
    J = len(joints)
    adj: List[set] = [set() for _ in range(J)]
    for j, n in enumerate(joints):
        p = parent_of.get(n)
        if p in node_to_j:
            pj = node_to_j[p]
            adj[j].add(pj)
            adj[pj].add(j)

    INF = J + 1
    dist = np.full((J, J), INF, dtype=np.int32)
    for s in range(J):
        dist[s, s] = 0
        q = deque([s])
        while q:
            x = q.popleft()
            for y in adj[x]:
                if dist[s, y] == INF:
                    dist[s, y] = dist[s, x] + 1
                    q.append(y)
    return dist


# ---------------------------------------------------------------------------
# Core repair (pure numpy — the unit-testable heart)
# ---------------------------------------------------------------------------

def repair_weights_array(
    positions: np.ndarray,      # (N, 3) vertex world positions
    joints0: np.ndarray,        # (N, 4) joint indices
    weights0: np.ndarray,       # (N, 4) weights
    joint_world: np.ndarray,    # (J, 3) bind-pose joint world positions
    hop_dist: np.ndarray,       # (J, J) skeleton-graph hop distances
    min_hops: int = DEFAULT_MIN_HOPS,
    weight_eps: float = 0.0,    # ignore contaminating influences with weight <= eps
    joint_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, RepairStats]:
    """Remove geometrically-impossible skin influences; renormalize. Pure numpy.

    For each vertex, while two active influences are ``>= min_hops`` apart, drop
    the contaminating bone that is farther from the vertex. Weights are renormalized
    to sum to 1. Returns ``(new_joints0, new_weights0, stats)``. Inputs are not
    mutated.
    """
    N = positions.shape[0]
    newJ = np.array(joints0, dtype=np.int64)
    newW = np.array(weights0, dtype=np.float64)
    stats = RepairStats(n_vertices=N)
    names = joint_names or []

    def nm(j: int) -> str:
        return names[j] if j < len(names) else f"joint_{j}"

    for vi in range(N):
        active = [c for c in range(newW.shape[1]) if newW[vi, c] > weight_eps]
        if len(active) < 2:
            continue
        jd = {c: newJ[vi, c] for c in active}
        # bones participating in at least one incoherent pair
        removed_here: List[int] = []
        while True:
            keep = [c for c in active if c not in removed_here]
            bad_bones = {c for c in keep for d in keep
                         if c != d and hop_dist[jd[c], jd[d]] >= min_hops}
            if not bad_bones:
                break
            # drop the contaminating bone geometrically farthest from the vertex
            worst = max(bad_bones,
                        key=lambda c: np.linalg.norm(positions[vi] - joint_world[jd[c]]))
            removed_here.append(worst)

        if not removed_here:
            continue
        for c in removed_here:
            b = nm(int(jd[c]))
            stats.removed_by_bone[b] = stats.removed_by_bone.get(b, 0) + 1
            newW[vi, c] = 0.0
        # record co-mingled pairs (contaminant vs the nearest surviving bone) for reporting
        survivors = [c for c in active if c not in removed_here]
        if survivors:
            near = min(survivors, key=lambda c: np.linalg.norm(positions[vi] - joint_world[jd[c]]))
            for c in removed_here:
                a, b = sorted((nm(int(jd[near])), nm(int(jd[c]))))
                stats.comingled_pairs[(a, b)] = stats.comingled_pairs.get((a, b), 0) + 1

        s = newW[vi].sum()
        if s > 1e-12:
            newW[vi] /= s
        else:  # should not happen: the nearest bone always survives
            keep0 = active[int(np.argmin([np.linalg.norm(positions[vi] - joint_world[jd[c]]) for c in active]))]
            newW[vi] = 0.0
            newW[vi, keep0] = 1.0
        stats.n_vertices_repaired += 1
        stats.n_influences_removed += len(removed_here)

    return newJ.astype(joints0.dtype), newW.astype(np.float32), stats


# ---------------------------------------------------------------------------
# glb-level entry point
# ---------------------------------------------------------------------------

def repair_glb(
    in_path: PathLike,
    out_path: PathLike,
    min_hops: int = DEFAULT_MIN_HOPS,
    weight_eps: float = 0.0,
) -> RepairStats:
    """Repair skin-weight contamination in a rigged glb; write to ``out_path``.

    Detects influences that blend across ``>= min_hops`` of skeleton graph and
    removes the geometrically-farther contaminant per vertex, renormalizing. Only
    ``JOINTS_0`` / ``WEIGHTS_0`` are rewritten; geometry and skeleton are untouched.
    """
    from pygltflib import GLTF2

    g = GLTF2().load_binary(str(in_path))
    if not g.skins:
        raise ValueError("glb has no skin (not a rigged mesh)")

    blob_ro = g.binary_blob() or b""
    world = _world_matrices(g)

    total = RepairStats()

    # Process each node that binds a mesh to a skin.
    edits: List[Tuple[object, np.ndarray, np.ndarray]] = []
    for node in g.nodes:
        if node.skin is None or node.mesh is None:
            continue
        skin = g.skins[node.skin]
        joints = list(skin.joints)
        joint_names = [g.nodes[n].name or f"joint_{k}" for k, n in enumerate(joints)]
        ibm = _read_accessor(g, blob_ro, skin.inverseBindMatrices).astype(np.float64)
        joint_world = np.array([
            np.linalg.inv(ibm[k].reshape(4, 4).T)[:3, 3] for k in range(len(joints))
        ])
        hop = _hop_distance_matrix(g, joints)
        node_world = world[g.nodes.index(node)]

        for prim in g.meshes[node.mesh].primitives:
            if prim.attributes.JOINTS_0 is None or prim.attributes.WEIGHTS_0 is None:
                continue
            pos_local = _read_accessor(g, blob_ro, prim.attributes.POSITION).astype(np.float64)
            pos_world = pos_local @ node_world[:3, :3].T + node_world[:3, 3]
            J0 = _read_accessor(g, blob_ro, prim.attributes.JOINTS_0)
            W0 = _read_accessor(g, blob_ro, prim.attributes.WEIGHTS_0).astype(np.float64)

            newJ, newW, stats = repair_weights_array(
                pos_world, J0, W0, joint_world, hop,
                min_hops=min_hops, weight_eps=weight_eps, joint_names=joint_names,
            )
            _accumulate(total, stats)
            edits.append((prim, newJ, newW))

    # Append rewritten accessors and repoint the primitives.
    blob = bytearray(blob_ro)
    while len(blob) % 4 != 0:
        blob.append(0)
    for prim, newJ, newW in edits:
        a_j = _append_accessor(g, blob, newJ.astype(np.uint16), _UNSIGNED_SHORT, "VEC4", _ARRAY_BUFFER)
        a_w = _append_accessor(g, blob, newW.astype(np.float32), _FLOAT, "VEC4", _ARRAY_BUFFER)
        prim.attributes.JOINTS_0 = a_j
        prim.attributes.WEIGHTS_0 = a_w

    while len(blob) % 4 != 0:
        blob.append(0)
    g.buffers[0].byteLength = len(blob)
    g.buffers[0].uri = None
    g.set_binary_blob(bytes(blob))
    g.save_binary(str(out_path))
    return total


def _accumulate(total: RepairStats, s: RepairStats) -> None:
    total.n_vertices += s.n_vertices
    total.n_vertices_repaired += s.n_vertices_repaired
    total.n_influences_removed += s.n_influences_removed
    for b, c in s.removed_by_bone.items():
        total.removed_by_bone[b] = total.removed_by_bone.get(b, 0) + c
    for p, c in s.comingled_pairs.items():
        total.comingled_pairs[p] = total.comingled_pairs.get(p, 0) + c
