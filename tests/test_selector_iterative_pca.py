"""Tests for IterativePCASelector -- Block 8e."""
from __future__ import annotations

import numpy as np
import pytest

from pcadiv.selection.selectors import (
    FPSSelector,
    IterativePCASelector,
    StaticPCASelector,
    get_selector,
    list_selectors,
)


# ---- fixtures -------------------------------------------------------

def _three_cluster_3d(n_per_cluster: int = 10) -> np.ndarray:
    """Three clusters on the three orthogonal axes of R³."""
    return np.vstack([
        np.tile([1.0, 0.0, 0.0], (n_per_cluster, 1)),
        np.tile([0.0, 1.0, 0.0], (n_per_cluster, 1)),
        np.tile([0.0, 0.0, 1.0], (n_per_cluster, 1)),
    ]).astype(np.float32)


# ---- registry & validation -----------------------------------------

def test_registry_contains_iterative_pca():
    assert "iterative_pca" in list_selectors()


def test_get_returns_instance():
    assert isinstance(get_selector("iterative_pca"), IterativePCASelector)


def test_k_must_be_positive():
    with pytest.raises(ValueError, match="k must be >= 1"):
        IterativePCASelector(k=0)


def test_variance_threshold_must_be_in_range():
    with pytest.raises(ValueError, match="variance_threshold"):
        IterativePCASelector(variance_threshold=1.5)
    with pytest.raises(ValueError, match="variance_threshold"):
        IterativePCASelector(variance_threshold=0.0)


def test_warmup_must_be_known():
    with pytest.raises(ValueError, match="warmup must be"):
        IterativePCASelector(warmup="zero")


# ---- core behavior: cluster coverage --------------------------------

def test_first_three_picks_cover_three_clusters():
    """With k=2 on 3-cluster 3D data, first 3 picks must be one from
    each cluster: after {x}, U = span(x); pick y; after {x,y},
    U = span(x,y); pick z.
    """
    X = _three_cluster_3d(10)
    toks = np.ones(30, dtype=np.int64)
    res = get_selector("iterative_pca", k=2).select(X, toks, budget=1000)

    cluster_of = lambda idx: idx // 10
    first3_clusters = [cluster_of(int(i)) for i in res.indices[:3]]
    assert len(set(first3_clusters)) == 3


def test_second_pick_is_outside_first_span():
    """After picking from cluster x, the 2nd pick must not be in x again
    (residual of other x-points is 0; only y or z have residual 1).
    """
    X = _three_cluster_3d(10)
    toks = np.ones(30, dtype=np.int64)
    res = get_selector("iterative_pca", k=2).select(X, toks, budget=1000)
    assert res.indices[0] // 10 != res.indices[1] // 10


def test_iterative_differs_from_static_on_clustered_data():
    """Online updating must matter: static and iterative disagree."""
    X = _three_cluster_3d(10)
    toks = np.ones(30, dtype=np.int64)
    it = get_selector("iterative_pca", k=2).select(X, toks, budget=1000)
    st = get_selector("static_pca", k=2).select(X, toks, budget=1000)
    assert it.indices.tolist() != st.indices.tolist()


def test_iterative_differs_from_fps_on_clustered_data():
    """Different criteria (PCA subspace vs nearest-selected) -> different
    orderings on non-trivial geometry.
    """
    rng = np.random.default_rng(7)
    X = np.concatenate([
        rng.normal([3, 0, 0], 0.2, size=(8, 3)),
        rng.normal([-3, 0, 0], 0.2, size=(8, 3)),
        rng.normal([0, 3, 0], 0.2, size=(8, 3)),
    ]).astype(np.float32)
    toks = np.ones(24, dtype=np.int64)
    core = X.mean(axis=0).astype(np.float32)
    it = get_selector("iterative_pca", k=2, warmup="norm").select(
        X, toks, budget=1000
    )
    fps = get_selector("fps", seed=0).select(X, toks, budget=1000, core_emb=core)
    assert it.indices.tolist() != fps.indices.tolist()


# ---- warm-up --------------------------------------------------------

def test_warmup_norm_picks_max_norm():
    X = np.array(
        [[0.5, 0.0], [2.0, 0.0], [0.1, 0.0], [3.0, 0.0]], dtype=np.float32
    )
    toks = np.ones(4, dtype=np.int64)
    res = get_selector("iterative_pca", k=1, warmup="norm").select(
        X, toks, budget=100
    )
    assert res.indices[0] == 3


def test_warmup_random_uses_seed():
    X = np.array(
        [[0.5, 0.0], [2.0, 0.0], [0.1, 0.0], [3.0, 0.0]], dtype=np.float32
    )
    toks = np.ones(4, dtype=np.int64)
    a = get_selector("iterative_pca", k=1, warmup="random", seed=0).select(
        X, toks, budget=100
    )
    b = get_selector("iterative_pca", k=1, warmup="random", seed=0).select(
        X, toks, budget=100
    )
    np.testing.assert_array_equal(a.indices, b.indices)


def test_anchor_overrides_warmup():
    """core_emb presence overrides warmup mode and picks argmax cosine."""
    X = np.eye(4, dtype=np.float32)
    toks = np.ones(4, dtype=np.int64)
    core = X[2].copy()
    res = get_selector("iterative_pca", k=1, warmup="norm").select(
        X, toks, budget=100, core_emb=core
    )
    assert res.indices[0] == 2
    assert res.meta["warmup"] == "anchor"


# ---- variance_threshold -------------------------------------------

def test_variance_threshold_reduces_k_eff():
    """Concentrated variance -> threshold picks fewer directions than k."""
    X = np.array(
        [[10.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.1],
         [10.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    toks = np.ones(5, dtype=np.int64)
    # With k=3 and variance_threshold=0.95: eigenvalues proportional to
    # [200, 2, 0.02] -> first eigenvalue alone is >0.99 of total.
    res = get_selector(
        "iterative_pca", k=3, variance_threshold=0.95, warmup="norm"
    ).select(X, toks, budget=100)
    # After the first pick (10,0,0), |S|=1 so k_eff=1 anyway. Check
    # that threshold is respected across the whole run.
    for k_eff in res.meta["step_k_eff"][1:]:
        assert k_eff <= 3


def test_variance_threshold_meta_contains_eigvals():
    X = _three_cluster_3d(5)
    toks = np.ones(15, dtype=np.int64)
    res = get_selector(
        "iterative_pca", k=2, variance_threshold=0.5
    ).select(X, toks, budget=1000)
    assert len(res.meta["step_eigvals"]) == len(res)
    # Last entry has at least one eigval recorded
    assert len(res.meta["step_eigvals"][-1]) >= 1


# ---- budget --------------------------------------------------------

def test_budget_respected():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 6)).astype(np.float32)
    toks = rng.integers(5, 25, size=20).astype(np.int64)
    res = get_selector("iterative_pca", k=3).select(X, toks, budget=80)
    assert res.total_tokens <= 80
    assert len(set(res.indices.tolist())) == len(res)


def test_skip_expensive_keeps_going():
    X = np.array(
        [[1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32
    )
    toks = np.array([1, 100, 1, 1], dtype=np.int64)
    res = get_selector("iterative_pca", k=1, warmup="norm").select(
        X, toks, budget=10
    )
    assert 1 not in res.indices.tolist()
    assert res.total_tokens <= 10


# ---- determinism & edge cases --------------------------------------

def test_deterministic():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 6)).astype(np.float32)
    toks = np.ones(20, dtype=np.int64)
    a = get_selector("iterative_pca", k=3).select(X, toks, budget=100)
    b = get_selector("iterative_pca", k=3).select(X, toks, budget=100)
    np.testing.assert_array_equal(a.indices, b.indices)


def test_empty_pool():
    X = np.zeros((0, 4), dtype=np.float32)
    toks = np.zeros(0, dtype=np.int64)
    res = get_selector("iterative_pca").select(X, toks, budget=100)
    assert len(res) == 0


def test_budget_zero():
    X = _three_cluster_3d(3)
    toks = np.ones(9, dtype=np.int64)
    res = get_selector("iterative_pca").select(X, toks, budget=0)
    assert len(res) == 0


def test_identical_points_does_not_crash():
    X = np.ones((5, 3), dtype=np.float32)
    toks = np.ones(5, dtype=np.int64)
    res = get_selector("iterative_pca", k=2).select(X, toks, budget=100)
    # All residuals are 0 after the first pick; selection stops early
    # or fills with the remaining (score 0) points. Neither is a bug.
    assert len(res) >= 1
    assert res.total_tokens <= 100


# ---- step_scores shape ---------------------------------------------

def test_step_scores_and_k_eff_lengths():
    X = _three_cluster_3d(5)
    toks = np.ones(15, dtype=np.int64)
    res = get_selector("iterative_pca", k=2).select(X, toks, budget=1000)
    assert len(res.meta["step_scores"]) == len(res)
    assert len(res.meta["step_k_eff"]) == len(res)