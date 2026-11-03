"""Tests for pcadiv.selection.selectors -- Block 4."""
from __future__ import annotations

import numpy as np
import pytest

from pcadiv.selection.selectors import (
    BaseSelector,
    RandomSelector,
    SelectionResult,
    get_selector,
    list_selectors,
    register_selector,
)


# ---- fixtures -------------------------------------------------------

@pytest.fixture
def synthetic_pool():
    rng = np.random.default_rng(0)
    N, D = 100, 16
    emb = rng.standard_normal((N, D)).astype(np.float32)
    toks = rng.integers(1, 50, size=N).astype(np.int64)
    return emb, toks


# ---- registry -------------------------------------------------------

def test_registry_contains_random():
    assert "random" in list_selectors()


def test_get_selector_returns_instance():
    assert isinstance(get_selector("random"), RandomSelector)


def test_get_selector_unknown_raises():
    with pytest.raises(KeyError, match="unknown selector"):
        get_selector("does-not-exist")


def test_register_duplicate_name_raises():
    class Fake(BaseSelector):
        name = "random"
        def _select_indices(self, pool_emb, pool_tokens, budget, core_emb):
            return [], {}

    with pytest.raises(ValueError, match="already registered"):
        register_selector(Fake)


# ---- SelectionResult invariants -------------------------------------

def test_select_indices_unique_and_in_range(synthetic_pool):
    emb, toks = synthetic_pool
    res = get_selector("random", seed=0).select(emb, toks, budget=500)
    assert len(set(res.indices.tolist())) == len(res)
    assert (res.indices >= 0).all()
    assert (res.indices < emb.shape[0]).all()


def test_cum_tokens_monotone(synthetic_pool):
    emb, toks = synthetic_pool
    res = get_selector("random", seed=0).select(emb, toks, budget=500)
    assert (np.diff(res.cum_tokens) >= 0).all()


def test_budget_not_exceeded(synthetic_pool):
    emb, toks = synthetic_pool
    res = get_selector("random", seed=0).select(emb, toks, budget=500)
    assert res.total_tokens <= 500


def test_cum_tokens_match_pool_tokens(synthetic_pool):
    emb, toks = synthetic_pool
    res = get_selector("random", seed=0).select(emb, toks, budget=500)
    expected = np.cumsum(toks[res.indices])
    np.testing.assert_array_equal(res.cum_tokens, expected)


def test_result_lengths_match(synthetic_pool):
    emb, toks = synthetic_pool
    res = get_selector("random", seed=0).select(emb, toks, budget=500)
    assert len(res.indices) == len(res.cum_tokens) == len(res)


# ---- determinism ----------------------------------------------------

def test_same_seed_same_selection(synthetic_pool):
    emb, toks = synthetic_pool
    a = get_selector("random", seed=42).select(emb, toks, budget=500)
    b = get_selector("random", seed=42).select(emb, toks, budget=500)
    np.testing.assert_array_equal(a.indices, b.indices)


def test_different_seeds_differ(synthetic_pool):
    emb, toks = synthetic_pool
    a = get_selector("random", seed=0).select(emb, toks, budget=500)
    b = get_selector("random", seed=1).select(emb, toks, budget=500)
    assert not np.array_equal(a.indices, b.indices)


# ---- budget edge cases ----------------------------------------------

def test_budget_zero_returns_empty(synthetic_pool):
    emb, toks = synthetic_pool
    res = get_selector("random", seed=0).select(emb, toks, budget=0)
    assert len(res) == 0
    assert res.total_tokens == 0


def test_budget_below_min_token_returns_empty(synthetic_pool):
    emb, toks = synthetic_pool
    res = get_selector("random", seed=0).select(
        emb, toks, budget=int(toks.min()) - 1
    )
    assert len(res) == 0


def test_huge_budget_selects_all(synthetic_pool):
    emb, toks = synthetic_pool
    res = get_selector("random", seed=0).select(
        emb, toks, budget=int(toks.sum())
    )
    assert len(res) == emb.shape[0]
    assert res.total_tokens == int(toks.sum())


def test_skips_expensive_then_fits_cheap():
    """_pack must skip items that do not fit, not stop at first failure."""
    emb = np.zeros((5, 4), dtype=np.float32)
    toks = np.array([100, 1, 1, 1, 1], dtype=np.int64)
    class Fixed(BaseSelector):
        name = "_fixed_tmp"
        def _select_indices(self, pool_emb, pool_tokens, budget, core_emb):
            return [0, 1, 2, 3, 4], {}

    res = Fixed().select(emb, toks, budget=3)
    np.testing.assert_array_equal(res.indices, [1, 2, 3])
    assert res.total_tokens == 3


# ---- input validation ----------------------------------------------

def test_pool_emb_must_be_2d():
    with pytest.raises(ValueError, match="pool_emb must be 2-D"):
        get_selector("random").select(np.zeros(5), np.zeros(5, dtype=np.int64), 10)


def test_pool_tokens_must_be_1d():
    with pytest.raises(ValueError, match="pool_tokens must be 1-D"):
        get_selector("random").select(
            np.zeros((5, 3)), np.zeros((5, 1), dtype=np.int64), 10
        )


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        get_selector("random").select(
            np.zeros((5, 3)), np.zeros(4, dtype=np.int64), 10
        )


def test_pool_tokens_must_be_integer():
    with pytest.raises(ValueError, match="integer dtype"):
        get_selector("random").select(
            np.zeros((3, 2)), np.array([1.0, 2.0, 3.0]), 10
        )


def test_pool_tokens_must_be_non_negative():
    with pytest.raises(ValueError, match="non-negative"):
        get_selector("random").select(
            np.zeros((3, 2)), np.array([1, -2, 3], dtype=np.int64), 10
        )


def test_core_emb_wrong_dim_raises():
    with pytest.raises(ValueError, match="core_emb dim mismatch"):
        get_selector("random").select(
            np.zeros((3, 4)),
            np.array([1, 1, 1], dtype=np.int64),
            10,
            core_emb=np.zeros(3, dtype=np.float32),
        )


def test_negative_budget_raises():
    with pytest.raises(ValueError, match="budget must be >= 0"):
        get_selector("random").select(
            np.zeros((3, 2)), np.array([1, 1, 1], dtype=np.int64), -1
        )


# ---- core_emb is explicitly ignored --------------------------------

def test_random_ignores_anchor(synthetic_pool):
    """Random is anchor-free: passing any core_emb must not change selection."""
    emb, toks = synthetic_pool
    core = np.zeros(emb.shape[1], dtype=np.float32)
    a = get_selector("random", seed=0).select(emb, toks, budget=500)
    b = get_selector("random", seed=0).select(emb, toks, budget=500, core_emb=core)
    np.testing.assert_array_equal(a.indices, b.indices)


# ---- empty pool ----------------------------------------------------

def test_empty_pool_returns_empty():
    emb = np.zeros((0, 4), dtype=np.float32)
    toks = np.zeros(0, dtype=np.int64)
    res = get_selector("random").select(emb, toks, budget=100)
    assert len(res) == 0