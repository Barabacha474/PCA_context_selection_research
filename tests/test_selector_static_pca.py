"""Tests for StaticPCASelector -- Block 8d."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from pcadiv.selection.selectors import (
    FPSSelector,
    StaticPCASelector,
    _fit_top_k_subspace,
    _residuals_sq,
    get_selector,
    list_selectors,
)


# ---- registry & validation -----------------------------------------

def test_registry_contains_static_pca():
    assert "static_pca" in list_selectors()


def test_get_returns_instance():
    assert isinstance(get_selector("static_pca"), StaticPCASelector)


def test_k_must_be_positive():
    with pytest.raises(ValueError, match="k must be >= 1"):
        StaticPCASelector(k=0)


# ---- helper correctness --------------------------------------------

def test_residuals_match_manual_formula():
    """_residuals_sq must equal ||x - U Uᵀx||² computed naively."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 6))
    U, _ = _fit_top_k_subspace(X, 3)
    manual = np.array([np.sum((x - U @ (U.T @ x)) ** 2) for x in X])
    np.testing.assert_allclose(_residuals_sq(X, U), manual, atol=1e-9)


def test_residuals_are_non_negative():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((15, 8))
    U, _ = _fit_top_k_subspace(X, 4)
    assert (_residuals_sq(X, U) >= 0).all()


def test_top_k_subspace_orthonormal():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 6))
    U, _ = _fit_top_k_subspace(X, 3)
    np.testing.assert_allclose(U.T @ U, np.eye(3), atol=1e-12)


def test_top_k_eigenvalues_descending():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 6))
    _, eigvals = _fit_top_k_subspace(X, 4)
    assert (np.diff(eigvals) <= 1e-12).all()


# ---- semantic behavior: minority direction wins --------------------

def test_static_pca_ranks_minority_direction_highest():
    """Pool with 3 orthogonal directions of unequal size: points in the
    minority direction have highest residual and are selected first.
    """
    X = np.vstack([
        np.tile([1.0, 0.0, 0.0], (20, 1)),  # majority: x
        np.tile([0.0, 1.0, 0.0], (20, 1)),  # majority: y
        np.tile([0.0, 0.0, 1.0], (2, 1)),   # minority: z
    ]).astype(np.float32)
    toks = np.ones(X.shape[0], dtype=np.int64)

    res = get_selector("static_pca", k=2).select(X, toks, budget=1000)
    # The two z-direction points must come first (residual ~1; others ~0).
    assert set(res.indices[:2].tolist()) == {40, 41}


def test_static_pca_residuals_isolate_minority():
    """Direct check on residuals in the same setup."""
    X = np.vstack([
        np.tile([1.0, 0.0, 0.0], (20, 1)),
        np.tile([0.0, 1.0, 0.0], (20, 1)),
        np.tile([0.0, 0.0, 1.0], (2, 1)),
    ])
    U, _ = _fit_top_k_subspace(X, 2)
    residuals = _residuals_sq(X, U)
    np.testing.assert_allclose(residuals[40:], 1.0, atol=1e-9)
    np.testing.assert_allclose(residuals[:40], 0.0, atol=1e-9)


# ---- degenerate cases ----------------------------------------------

def test_degenerate_k_ge_dim_collapses_residuals():
    """When k >= min(N, D), residuals collapse to ~0 and everything is
    selected. The exact order is not guaranteed: residuals are not
    exactly 0 in floating point, and the cancellation error scales
    with ||x||², so a point with larger norm can get a tiny (but
    non-zero) residual and outrank a point with a smaller norm.
    """
    X = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32
    )
    toks = np.ones(3, dtype=np.int64)
    with pytest.warns(RuntimeWarning, match="capping"):
        res = get_selector("static_pca", k=5).select(X, toks, budget=100)
    # All three points must be selected -- that is the semantic claim
    # of "residuals collapse". Order is set-valued, not order-valued.
    assert set(res.indices.tolist()) == {0, 1, 2}


def test_empty_pool_returns_empty():
    X = np.zeros((0, 4), dtype=np.float32)
    toks = np.zeros(0, dtype=np.int64)
    res = get_selector("static_pca", k=3).select(X, toks, budget=100)
    assert len(res) == 0


# ---- budget --------------------------------------------------------

def test_budget_respected():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 6)).astype(np.float32)
    toks = rng.integers(5, 25, size=20).astype(np.int64)
    res = get_selector("static_pca", k=3).select(X, toks, budget=80)
    assert res.total_tokens <= 80
    assert len(set(res.indices.tolist())) == len(res)


def test_budget_skips_expensive_top():
    """If the top-residual candidate doesn't fit, next one is taken."""
    X = np.vstack([
        np.tile([1.0, 0.0, 0.0], (10, 1)),
        np.array([[0.0, 0.0, 1.0]]),  # highest residual, cost 100
    ]).astype(np.float32)
    toks = np.array([1] * 10 + [100], dtype=np.int64)
    res = get_selector("static_pca", k=2).select(X, toks, budget=50)
    assert 10 not in res.indices.tolist()
    assert res.total_tokens <= 50


# ---- determinism ----------------------------------------------------

def test_deterministic():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 6)).astype(np.float32)
    toks = np.ones(20, dtype=np.int64)
    a = get_selector("static_pca", k=3).select(X, toks, budget=100)
    b = get_selector("static_pca", k=3).select(X, toks, budget=100)
    np.testing.assert_array_equal(a.indices, b.indices)


# ---- anchor is ignored ---------------------------------------------

def test_ignores_core_emb():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((15, 4)).astype(np.float32)
    toks = np.ones(15, dtype=np.int64)
    core = X.mean(axis=0).astype(np.float32)
    a = get_selector("static_pca", k=2).select(X, toks, budget=100)
    b = get_selector("static_pca", k=2).select(X, toks, budget=100, core_emb=core)
    np.testing.assert_array_equal(a.indices, b.indices)


# ---- differs from FPS (different criterion) -----------------------

def test_differs_from_fps():
    """Static PCA and FPS are different criteria and must not coincide."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((40, 6)).astype(np.float32)
    toks = np.ones(40, dtype=np.int64)
    core = X.mean(axis=0).astype(np.float32)
    sp = get_selector("static_pca", k=3).select(X, toks, budget=1000)
    fps = get_selector("fps", seed=0).select(X, toks, budget=1000, core_emb=core)
    assert sp.indices.tolist() != fps.indices.tolist()


# ---- meta content ---------------------------------------------------

def test_meta_records_k_and_eigenvalues():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((15, 8)).astype(np.float32)
    toks = np.ones(15, dtype=np.int64)
    res = get_selector("static_pca", k=4).select(X, toks, budget=100)
    assert res.meta["k_eff"] == 4
    assert res.meta["k_requested"] == 4
    assert len(res.meta["top_eigenvalues"]) == 4
    # desc order
    eigs = res.meta["top_eigenvalues"]
    assert all(eigs[i] >= eigs[i+1] - 1e-12 for i in range(len(eigs)-1))