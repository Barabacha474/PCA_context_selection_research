"""Tests for MMRSelector and DynamicMMRSelector -- Block 8b."""
from __future__ import annotations

import numpy as np
import pytest

from pcadiv.selection.selectors import (
    DynamicMMRSelector,
    MMRSelector,
    RelevanceSelector,
    get_selector,
    list_selectors,
)


# ---- registry -------------------------------------------------------

def test_registry_contains_mmr_variants():
    names = list_selectors()
    assert "mmr" in names and "mmr_dynamic" in names


def test_get_returns_instances():
    assert isinstance(get_selector("mmr"), MMRSelector)
    assert isinstance(get_selector("mmr_dynamic"), DynamicMMRSelector)


# ---- validation -----------------------------------------------------

def test_lambda_out_of_range_raises():
    with pytest.raises(ValueError, match="lambda_ must be in"):
        MMRSelector(lambda_=1.5)
    with pytest.raises(ValueError, match="lambda_ must be in"):
        DynamicMMRSelector(lambda_=-0.1)


def test_lambda_positive_requires_anchor():
    emb = np.eye(3, dtype=np.float32)
    toks = np.ones(3, dtype=np.int64)
    with pytest.raises(ValueError, match="requires core_emb"):
        get_selector("mmr", lambda_=0.5).select(emb, toks, budget=100)


def test_lambda_zero_needs_no_anchor():
    emb = np.eye(3, dtype=np.float32)
    toks = np.ones(3, dtype=np.int64)
    res = get_selector("mmr", lambda_=0.0, seed=0).select(emb, toks, budget=100)
    assert len(res) == 3


# ---- λ=1 == RelevanceSelector ---------------------------------------

def test_lambda_one_equals_relevance():
    """With λ=1 MMR must reproduce RelevanceSelector's selection exactly."""
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((30, 8)).astype(np.float32)
    toks = rng.integers(5, 20, size=30).astype(np.int64)
    core = emb.mean(axis=0).astype(np.float32)
    budget = 100

    rel = get_selector("relevance").select(emb, toks, budget=budget, core_emb=core)
    mmr = get_selector("mmr", lambda_=1.0).select(emb, toks, budget=budget, core_emb=core)
    np.testing.assert_array_equal(mmr.indices, rel.indices)
    np.testing.assert_array_equal(mmr.cum_tokens, rel.cum_tokens)


def test_lambda_one_with_tight_budget_matches_relevance():
    """Same as above but budget excludes several items -- skip semantics must match."""
    emb = np.eye(5, dtype=np.float32)
    toks = np.array([100, 5, 5, 5, 5], dtype=np.int64)
    core = emb[0].copy()  # top relevance = index 0, but it's too expensive
    rel = get_selector("relevance").select(emb, toks, budget=10, core_emb=core)
    mmr = get_selector("mmr", lambda_=1.0).select(emb, toks, budget=10, core_emb=core)
    np.testing.assert_array_equal(mmr.indices, rel.indices)


# ---- λ=0 behavior ---------------------------------------------------

def test_lambda_zero_first_pick_deterministic_by_seed():
    emb = np.eye(10, dtype=np.float32)
    toks = np.ones(10, dtype=np.int64)
    a = get_selector("mmr", lambda_=0.0, seed=7).select(emb, toks, budget=100)
    b = get_selector("mmr", lambda_=0.0, seed=7).select(emb, toks, budget=100)
    np.testing.assert_array_equal(a.indices, b.indices)


def test_lambda_zero_prefers_diverse_after_warmup():
    """After one pick from a tight cluster, λ=0 should jump to the far cluster."""
    emb = np.array(
        [[1.0, 0.0], [1.0, 0.01], [1.0, -0.01], [-1.0, 0.0]],
        dtype=np.float32,
    )
    toks = np.ones(4, dtype=np.int64)
    # Force warm-up to index 0 by seeding a permutation that starts there.
    # Easier: use λ=0, seed=0, and just check that the far point comes
    # within the first two picks (it must, since it has the smallest
    # max_sim to any cluster point).
    res = get_selector("mmr", lambda_=0.0, seed=0).select(emb, toks, budget=100)
    assert 3 in res.indices[:2].tolist()


# ---- MMR sanity -----------------------------------------------------

def test_mmr_budget_respected():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((20, 6)).astype(np.float32)
    toks = rng.integers(10, 30, size=20).astype(np.int64)
    core = emb.mean(axis=0).astype(np.float32)
    res = get_selector("mmr", lambda_=0.5).select(emb, toks, budget=60, core_emb=core)
    assert res.total_tokens <= 60
    assert len(set(res.indices.tolist())) == len(res)


def test_mmr_step_scores_non_increasing_with_lambda_zero():
    """With λ=0 the marginal diversity scores should be non-increasing.

    score at step k = -(1-λ)·max_sim(best). Since max_sim is a running
    maximum (non-decreasing per candidate), and we pick the minimum
    max_sim each time, the minimum is non-decreasing -> score is
    non-increasing.
    """
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((15, 4)).astype(np.float32)
    toks = np.ones(15, dtype=np.int64)
    res = get_selector("mmr", lambda_=0.0, seed=0).select(emb, toks, budget=100)
    scores = res.meta["step_scores"]
    # First entry is 0 (warm-up by convention); the rest must be non-increasing.
    tail = scores[1:]
    assert all(a >= b - 1e-9 for a, b in zip(tail, tail[1:]))


def test_mmr_deterministic():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((20, 6)).astype(np.float32)
    toks = np.ones(20, dtype=np.int64)
    core = emb.mean(axis=0).astype(np.float32)
    a = get_selector("mmr", lambda_=0.7).select(emb, toks, budget=50, core_emb=core)
    b = get_selector("mmr", lambda_=0.7).select(emb, toks, budget=50, core_emb=core)
    np.testing.assert_array_equal(a.indices, b.indices)


# ---- DynamicMMR -----------------------------------------------------

def test_dynamic_mmr_budget_respected():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((20, 6)).astype(np.float32)
    toks = rng.integers(10, 30, size=20).astype(np.int64)
    res = get_selector("mmr_dynamic", lambda_=0.5, seed=0).select(
        emb, toks, budget=60
    )
    assert res.total_tokens <= 60


def test_dynamic_mmr_differs_from_static_on_diverse_pool():
    """Dynamic anchor must not always produce the same sequence as static."""
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((20, 6)).astype(np.float32)
    toks = np.ones(20, dtype=np.int64)
    core = emb.mean(axis=0).astype(np.float32)
    static = get_selector("mmr", lambda_=0.5).select(emb, toks, budget=100, core_emb=core)
    dynamic = get_selector("mmr_dynamic", lambda_=0.5, seed=0).select(
        emb, toks, budget=100, core_emb=core
    )
    # Warm-up must match (both use argmax relevance to anchor).
    assert static.indices[0] == dynamic.indices[0]
    # Divergence somewhere is expected; if identical, the test is a hint
    # that dynamic anchor was not wired.
    assert not np.array_equal(static.indices, dynamic.indices)


def test_dynamic_mmr_deterministic():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((20, 6)).astype(np.float32)
    toks = np.ones(20, dtype=np.int64)
    a = get_selector("mmr_dynamic", lambda_=0.5, seed=0).select(emb, toks, budget=100)
    b = get_selector("mmr_dynamic", lambda_=0.5, seed=0).select(emb, toks, budget=100)
    np.testing.assert_array_equal(a.indices, b.indices)


def test_dynamic_mmr_warmup_uses_anchor_when_given():
    emb = np.eye(4, dtype=np.float32)
    toks = np.ones(4, dtype=np.int64)
    core = emb[2].copy()
    res = get_selector("mmr_dynamic", lambda_=0.5).select(emb, toks, budget=100, core_emb=core)
    assert res.indices[0] == 2


def test_dynamic_mmr_warmup_random_without_anchor():
    emb = np.eye(10, dtype=np.float32)
    toks = np.ones(10, dtype=np.int64)
    a = get_selector("mmr_dynamic", lambda_=0.5, seed=3).select(emb, toks, budget=100)
    b = get_selector("mmr_dynamic", lambda_=0.5, seed=3).select(emb, toks, budget=100)
    np.testing.assert_array_equal(a.indices, b.indices)
    # and a different seed gives a different first pick (probabilistically)
    c = get_selector("mmr_dynamic", lambda_=0.5, seed=4).select(emb, toks, budget=100)
    assert c.indices[0] != a.indices[0]


# ---- edge cases ----------------------------------------------------

def test_empty_pool_returns_empty():
    emb = np.zeros((0, 4), dtype=np.float32)
    toks = np.zeros(0, dtype=np.int64)
    core = np.zeros(4, dtype=np.float32)
    res = get_selector("mmr", lambda_=0.5).select(emb, toks, budget=100, core_emb=core)
    assert len(res) == 0
    res = get_selector("mmr_dynamic", lambda_=0.5, seed=0).select(emb, toks, budget=100)
    assert len(res) == 0


def test_budget_zero_returns_empty():
    emb = np.eye(3, dtype=np.float32)
    toks = np.ones(3, dtype=np.int64)
    core = emb[0].copy()
    res = get_selector("mmr", lambda_=0.5).select(emb, toks, budget=0, core_emb=core)
    assert len(res) == 0