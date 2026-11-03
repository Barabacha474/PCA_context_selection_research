"""Tests for VendiScore -- Block 9b."""
from __future__ import annotations

import numpy as np
import pytest

from pcadiv.evaluation.metrics import VendiScore, get_metric, list_metrics


# ---- registry & validation -----------------------------------------

def test_registry_contains_vendi():
    assert "vendi" in list_metrics()


def test_get_returns_instance():
    assert isinstance(get_metric("vendi"), VendiScore)


def test_unknown_kernel_rejected():
    with pytest.raises(ValueError, match="kernel must be"):
        VendiScore(kernel="poly")


def test_negative_gamma_rejected():
    with pytest.raises(ValueError, match="gamma must be positive"):
        VendiScore(gamma=-1.0)


def test_zero_q_rejected():
    with pytest.raises(ValueError, match="q must be positive"):
        VendiScore(q=0.0)


# ---- degenerate cases ----------------------------------------------

def test_empty_returns_zero():
    assert VendiScore()(np.zeros((0, 4), dtype=np.float32)) == {"vendi_score": 0.0}


def test_single_point_returns_one():
    assert VendiScore()(np.ones((1, 4), dtype=np.float32)) == {"vendi_score": 1.0}


def test_identical_points_returns_one():
    """K(x,x)=1 for all, eigenvalues collapse to one nonzero -> Vendi = 1."""
    X = np.ones((10, 3), dtype=np.float32)
    out = VendiScore(gamma=1.0)(X)  # fixed gamma for determinism
    assert out["vendi_score"] == pytest.approx(1.0, abs=1e-6)


# ---- known behavior ------------------------------------------------

def test_orthogonal_points_approach_n():
    """N orthogonal unit vectors with fixed RBF gamma -> near N."""
    N = 4
    X = np.eye(N, dtype=np.float32)
    # Fixed moderate gamma: exp(-g * 2) for distinct points, exp(0) = 1 on diag.
    # With gamma=10, off-diagonal tiny, eigenvalues ~ 1/N -> Vendi ~ N.
    out = VendiScore(gamma=10.0)(X)
    assert out["vendi_score"] == pytest.approx(N, rel=0.05)


def test_vendi_monotone_in_spread_cosine():
    """Two clusters far apart have higher Vendi than one tight cluster."""
    m = VendiScore(kernel="cosine", gamma=1.0)
    one_cluster = np.array(
        [[1.0, 0.0], [0.99, 0.1], [0.98, 0.05], [0.97, 0.08]],
        dtype=np.float32,
    )
    two_clusters = np.array(
        [[1.0, 0.0], [0.99, 0.1], [-1.0, 0.0], [-0.99, 0.1]],
        dtype=np.float32,
    )
    assert m(two_clusters)["vendi_score"] > m(one_cluster)["vendi_score"]


def test_median_gamma_runs():
    """Default gamma=None must not crash on varied-scale data."""
    X = np.random.default_rng(0).standard_normal((20, 4)).astype(np.float32)
    out = VendiScore()(X)
    assert np.isfinite(out["vendi_score"])
    assert out["vendi_score"] >= 1.0 - 1e-6
    assert out["vendi_score"] <= 20.0 + 1e-6


def test_renyi_q_2_returns_finite():
    X = np.random.default_rng(0).standard_normal((10, 4)).astype(np.float32)
    out = VendiScore(q=2.0)(X)
    assert np.isfinite(out["vendi_score"])


# ---- contract ------------------------------------------------------

def test_returns_plain_dict():
    X = np.random.default_rng(0).standard_normal((5, 3)).astype(np.float32)
    out = VendiScore()(X)
    assert type(out) is dict
    assert set(out) == {"vendi_score"}
    assert isinstance(out["vendi_score"], float)


def test_accepts_core_emb_and_ignores_it():
    X = np.random.default_rng(0).standard_normal((5, 3)).astype(np.float32)
    m = VendiScore()
    a = m(X)
    b = m(X, core_emb=np.zeros(3, dtype=np.float32))
    assert a == b


def test_deterministic_with_fixed_gamma():
    X = np.random.default_rng(0).standard_normal((10, 4)).astype(np.float32)
    m = VendiScore(gamma=0.5)
    a = m(X)["vendi_score"]
    b = m(X)["vendi_score"]
    assert a == b


def test_zero_norm_row_does_not_crash_cosine():
    X = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    out = VendiScore(kernel="cosine", gamma=1.0)(X)
    assert np.isfinite(out["vendi_score"])