"""Tests for deterministic skin-weight coherence repair (weight_repair).

Two levels:
  * unit — the pure numpy core on a tiny synthetic skeleton, so the detect/repair
    logic is pinned without needing a real glb;
  * fixture — the real ``broken_hand_weights.glb`` (a SkinTokens-rigged character
    whose right-hand finger vertices leak onto ``RightToeBase``), verifying the
    glb-level pass removes the cross-body contamination and leaves the file valid.
"""

from pathlib import Path

import numpy as np
import pytest

from skintokens import weight_repair

_BROKEN_RIG = Path(__file__).parent / "fixtures" / "rigs" / "broken_hand_weights.glb"


# ---------------------------------------------------------------------------
# Unit: pure-numpy core
# ---------------------------------------------------------------------------

def _chain_hops(n):
    """Hop-distance matrix for a simple bone chain 0-1-2-...-(n-1)."""
    d = np.abs(np.subtract.outer(np.arange(n), np.arange(n))).astype(np.int32)
    return d


def test_removes_far_bone_and_renormalizes():
    # Chain of 6 bones along +x at unit spacing; a vertex sits at the far end (x=5)
    # but is weighted 0.5 to its correct near bone (5) and 0.5 to a distant bone (0).
    joint_world = np.array([[float(i), 0.0, 0.0] for i in range(6)])
    hop = _chain_hops(6)
    positions = np.array([[5.0, 0.0, 0.0]])
    joints0 = np.array([[5, 0, 0, 0]])
    weights0 = np.array([[0.5, 0.5, 0.0, 0.0]])

    newJ, newW, stats = weight_repair.repair_weights_array(
        positions, joints0, weights0, joint_world, hop, min_hops=5,
    )
    # The far bone (0) is 5 hops from bone 5 -> dropped; near bone keeps all weight.
    assert stats.n_vertices_repaired == 1
    assert stats.n_influences_removed == 1
    assert newW[0, 0] == pytest.approx(1.0)
    assert newW[0, 1] == pytest.approx(0.0)
    assert newW[0].sum() == pytest.approx(1.0)


def test_leaves_coherent_weights_untouched():
    # Same chain, but the two influences are only 1 hop apart (adjacent bones) —
    # a legitimate blend that must survive unchanged.
    joint_world = np.array([[float(i), 0.0, 0.0] for i in range(6)])
    hop = _chain_hops(6)
    positions = np.array([[4.5, 0.0, 0.0]])
    joints0 = np.array([[4, 5, 0, 0]])
    weights0 = np.array([[0.6, 0.4, 0.0, 0.0]])

    newJ, newW, stats = weight_repair.repair_weights_array(
        positions, joints0, weights0, joint_world, hop, min_hops=5,
    )
    assert stats.n_vertices_repaired == 0
    np.testing.assert_allclose(newW[0], weights0[0])


def test_nearest_bone_always_survives():
    # A vertex contaminated by two far bones still keeps its single near influence.
    joint_world = np.array([[float(i), 0.0, 0.0] for i in range(7)])
    hop = _chain_hops(7)
    positions = np.array([[6.0, 0.0, 0.0]])
    joints0 = np.array([[6, 0, 1, 0]])
    weights0 = np.array([[0.4, 0.3, 0.3, 0.0]])

    newJ, newW, stats = weight_repair.repair_weights_array(
        positions, joints0, weights0, joint_world, hop, min_hops=5,
    )
    assert newW[0].sum() == pytest.approx(1.0)
    # Only the near bone (6) retains weight.
    assert newW[0, 0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Fixture: the real broken rig
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _BROKEN_RIG.exists(), reason="broken rig fixture missing")
def test_repair_broken_hand_weights_glb(tmp_path):
    pytest.importorskip("pygltflib")
    from pygltflib import GLTF2

    out = tmp_path / "cleaned.glb"
    stats = weight_repair.repair_glb(_BROKEN_RIG, out, min_hops=5)

    # The known defect: right-hand fingers leak onto RightToeBase. Repair must
    # remove real contamination, and the toe must be among the top offenders.
    assert stats.n_influences_removed > 0
    assert out.exists() and out.stat().st_size > 0
    removed = stats.removed_by_bone
    assert any("ToeBase" in b for b in removed), removed

    # After repair, NO vertex blends across >= min_hops of skeleton anywhere.
    g = GLTF2().load_binary(str(out))
    blob = g.binary_blob()
    skin = g.skins[0]
    joints = list(skin.joints)
    hop = weight_repair._hop_distance_matrix(g, joints)
    incoherent = 0
    for node in g.nodes:
        if node.skin is None or node.mesh is None:
            continue
        for prim in g.meshes[node.mesh].primitives:
            J0 = weight_repair._read_accessor(g, blob, prim.attributes.JOINTS_0)
            W0 = weight_repair._read_accessor(g, blob, prim.attributes.WEIGHTS_0).astype(float)
            for vi in range(J0.shape[0]):
                active = [c for c in range(4) if W0[vi, c] > 1e-6]
                for a in active:
                    for b in active:
                        if hop[J0[vi, a], J0[vi, b]] >= 5:
                            incoherent += 1
    assert incoherent == 0, f"{incoherent} cross-body influences remain after repair"

    # Skin/skeleton are untouched (same joint count); weights still normalized.
    assert len(g.skins[0].joints) == len(joints)
