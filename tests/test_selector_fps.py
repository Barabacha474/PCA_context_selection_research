"""Tests for FPSSelector -- Block 8c."""
from __future__ import annotations

import numpy as np
import pytest

from pcadiv.selection.selectors import (
    FPSSelector,
    MMRSelector,
    get_selector,
    list_selectors,
)


# ---- registry -------------------------------------------------------

def test_registry_contains_fps():
    assert "fps" in list_selectors()


def test_get_fps_returns_instance():
    assert isinstance(get_selector("fps"), FPSSelector)


# ---- validation -----------------------------------------------------

def test_unknown_metric_rejected_at_init():
    with pytest.raises(ValueError, match="metric must be"):
        FPSSelector(metric="manhattan")


# ---- ordering on a toy example --------------------------------------

def test_fps_picks_far_points_first():
    """Three clusters on a line: after the first pick from cluster A,
    FPS must jump to a far cluster (C), not stay in A.
    """
    # clusters at -10, 0, +10; the anchor freezes FPS to start at -10
    emb = np.array(
        [[-10.0, 0.0], [-9.5, 0.0], [0.0, 0.0], [9.5, 0.0], [10.0, 0.0]],
        dtype=np.float32,
    )
    toks = np.ones(5, dtype=np.int64)
    # No anchor -> random warm-up; force via seed=0 which permutes 0..4.
    # To make the test robust, use core_emb=emb[0] to anchor at index 0.
    core = emb[0].copy()
    res = get_selector("fps", seed=0, metric="euclidean").select(
        emb, toks, budget=100, core_emb=core
    )
    # After index 0, the farthest point is index 4 (+10). Second pick.
    assert res.indices[0] == 0
    assert res.indices[1] == 4


def test_fps_second_pick_is_farthest_from_first():
    emb = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 5.0], [3.0, 0.0]], dtype=np.float32
    )
    toks = np.ones(4, dtype=np.int64)
    core = emb[0].copy()
    res = get_selector("fps", metric="euclidean").select(
        emb, toks, budget=100, core_emb=core
    )
    # index 0 -> farthest is index 2 (dist 5)
    assert res.indices[0] == 0
    assert res.indices[1] == 2


def test_fps_differs_from_mmr_on_multi_cluster_data():
    """FPS and MMR (λ=0) use different criteria -- they must not coincide
    on a pool with several clusters and non-trivial geometry.
    """
    rng = np.random.default_rng(1)
    emb = np.concatenate(
        [
            rng.normal(loc=[5, 0], scale=0.1, size=(10, 2)),
            rng.normal(loc=[-5, 0], scale=0.1, size=(10, 2)),
            rng.normal(loc=[0, 5], scale=0.1, size=(10, 2)),
        ]
    ).astype(np.float32)
    toks = np.ones(30, dtype=np.int64)
    core = emb[0].copy()

    fps = get_selector("fps", seed=0, metric="cosine").select(
        emb, toks, budget=100, core_emb=core
    )
    mmr = get_selector("mmr", lambda_=0.0, seed=0).select(
        emb, toks, budget=100, core_emb=core
    )
    # Both should pick one per cluster within the first three picks;
    # the specific order may differ but the sequences as a whole
    # should not be elementwise identical (unless the methods happen
    # to agree -- allow skipping that pathological case).
    assert fps.indices.tolist() != mmr.indices.tolist()


# ---- warm-up behavior ----------------------------------------------

def test_warmup_uses_anchor_when_given():
    emb = np.eye(4, dtype=np.float32)
    toks = np.ones(4, dtype=np.int64)
    core = emb[2].copy()
    res = get_selector("fps").select(emb, toks, budget=100, core_emb=core)
    assert res.indices[0] == 2


def test_warmup_random_when_no_anchor():
    emb = np.eye(10, dtype=np.float32)
    toks = np.ones(10, dtype=np.int64)
    a = get_selector("fps", seed=0).select(emb, toks, budget=100)
    b = get_selector("fps", seed=0).select(emb, toks, budget=100)
    np.testing.assert_array_equal(a.indices, b.indices)
    c = get_selector("fps", seed=1).select(emb, toks, budget=100)
    # Different seeds should give different first picks (with high
    # probability on N=10 and unit vectors).
    assert c.indices[0] != a.indices[0]


# ---- determinism ----------------------------------------------------

def test_fps_deterministic():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((20, 6)).astype(np.float32)
    toks = np.ones(20, dtype=np.int64)
    core = emb.mean(axis=0).astype(np.float32)
    a = get_selector("fps", seed=0).select(emb, toks, budget=50, core_emb=core)
    b = get_selector("fps", seed=0).select(emb, toks, budget=50, core_emb=core)
    np.testing.assert_array_equal(a.indices, b.indices)


# ---- budget ---------------------------------------------------------

def test_fps_budget_respected():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((20, 6)).astype(np.float32)
    toks = rng.integers(10, 30, size=20).astype(np.int64)
    core = emb.mean(axis=0).astype(np.float32)
    res = get_selector("fps").select(emb, toks, budget=60, core_emb=core)
    assert res.total_tokens <= 60
    assert len(set(res.indices.tolist())) == len(res)


def test_budget_zero_returns_empty():
    emb = np.eye(3, dtype=np.float32)
    toks = np.ones(3, dtype=np.int64)
    res = get_selector("fps", seed=0).select(emb, toks, budget=0)
    assert len(res) == 0


def test_fps_empty_pool():
    emb = np.zeros((0, 4), dtype=np.float32)
    toks = np.zeros(0, dtype=np.int64)
    res = get_selector("fps", seed=0).select(emb, toks, budget=100)
    assert len(res) == 0


# ---- metric choice --------------------------------------------------

def test_euclidean_and_cosine_differ_on_unnormalized_data():
    """With unnormalized inputs the two metrics must not give identical orders."""
    emb = np.array(
        [[1.0, 0.0], [10.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
        dtype=np.float32,
    )
    toks = np.ones(4, dtype=np.int64)
    core = emb[0].copy()
    e = get_selector("fps", metric="euclidean").select(
        emb, toks, budget=100, core_emb=core
    )
    c = get_selector("fps", metric="cosine").select(
        emb, toks, budget=100, core_emb=core
    )
    # euclidean: index 1 (dist 9) is farthest from index 0
    # cosine:    index 2 (cos dist 1) is farthest from index 0; index 1
    #            has cos dist 0 (same direction)
    assert e.indices[1] == 1
    assert c.indices[1] == 2


# ---- step_scores shape ---------------------------------------------

def test_step_scores_length_matches_indices():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((10, 4)).astype(np.float32)
    toks = np.ones(10, dtype=np.int64)
    res = get_selector("fps", seed=0).select(emb, toks, budget=100)
    assert len(res.meta["step_scores"]) == len(res)