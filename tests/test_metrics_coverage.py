"""Tests for Coverage metric -- Block 9a."""
from __future__ import annotations

import numpy as np
import pytest

from pcadiv.evaluation.metrics import Coverage, get_metric, list_metrics


# ---- registry & validation -----------------------------------------

def test_registry_contains_coverage():
    assert "coverage" in list_metrics()


def test_get_returns_instance():
    assert isinstance(get_metric("coverage"), Coverage)


# ---- degenerate cases ----------------------------------------------

def test_empty_returns_zero_both():
    m = Coverage()
    out = m(np.zeros((0, 4), dtype=np.float32))
    assert out == {"coverage_radius": 0.0, "coverage_logdet": 0.0}


def test_single_point_returns_zero_both():
    m = Coverage()
    out = m(np.ones((1, 4), dtype=np.float32))
    assert out == {"coverage_radius": 0.0, "coverage_logdet": 0.0}


def test_identical_points_radius_zero_logdet_zero():
    """Fully collapsed set: 0 radius, 0 logdet by convention."""
    m = Coverage()
    X = np.ones((5, 3), dtype=np.float32)
    out = m(X)
    assert out["coverage_radius"] == pytest.approx(0.0, abs=1e-6)
    assert out["coverage_logdet"] == 0.0


# ---- radius sanity --------------------------------------------------

def test_radius_known_1d():
    """Points at 0, 1, 2: centroid at 1, max distance 1."""
    X = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    out = Coverage()(X)
    assert out["coverage_radius"] == pytest.approx(1.0, rel=1e-6)


def test_radius_grows_with_spread():
    m = Coverage()
    tight = np.array([[0.0, 0.0], [0.1, 0.0], [-0.1, 0.0]], dtype=np.float32)
    wide = np.array([[0.0, 0.0], [10.0, 0.0], [-10.0, 0.0]], dtype=np.float32)
    assert m(wide)["coverage_radius"] > m(tight)["coverage_radius"]


# ---- logdet sanity -------------------------------------------------

def test_logdet_positive_for_spread_2d():
    X = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=np.float32,
    )
    out = Coverage()(X)
    # 2D full-rank; logdet should be finite, non-zero
    assert np.isfinite(out["coverage_logdet"])
    assert out["coverage_logdet"] != 0.0


def test_logdet_monotone_in_spread_1d():
    """Wider spread of 1D points -> larger logdet (finite, not -inf)."""
    m = Coverage()
    narrow = np.array([[0.0], [0.1], [0.2]], dtype=np.float32)
    wide = np.array([[0.0], [5.0], [10.0]], dtype=np.float32)
    assert m(wide)["coverage_logdet"] > m(narrow)["coverage_logdet"]


def test_logdet_rank_deficient_is_finite():
    """N < D: covariance is rank-deficient, logdet must not be -inf."""
    X = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    out = Coverage()(X)
    assert np.isfinite(out["coverage_logdet"])


def test_logdet_orthogonal_beats_collinear():
    """Orthogonal points cover more volume than collinear points."""
    m = Coverage()
    collinear = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32
    )
    orthogonal = np.array(
        [[0.0, 0.0], [3.0, 0.0], [0.0, 3.0], [3.0, 3.0]], dtype=np.float32
    )
    assert m(orthogonal)["coverage_logdet"] > m(collinear)["coverage_logdet"]


# ---- contract ------------------------------------------------------

def test_returns_plain_dict_with_both_keys():
    out = Coverage()(np.ones((3, 2), dtype=np.float32))
    assert type(out) is dict
    assert set(out) == {"coverage_radius", "coverage_logdet"}
    for v in out.values():
        assert isinstance(v, float)


def test_accepts_core_emb_and_ignores_it():
    X = np.random.default_rng(0).standard_normal((5, 3)).astype(np.float32)
    m = Coverage()
    a = m(X)
    b = m(X, core_emb=np.zeros(3, dtype=np.float32))
    assert a == b


def test_core_emb_wrong_dim_raises():
    with pytest.raises(ValueError, match="core_emb dim mismatch"):
        Coverage()(np.ones((3, 4), dtype=np.float32),
                    core_emb=np.zeros(3, dtype=np.float32))


def test_float32_input_does_not_crash_and_is_finite():
    X = np.random.default_rng(0).standard_normal((50, 16)).astype(np.float32)
    out = Coverage()(X)
    assert np.isfinite(out["coverage_radius"])
    assert np.isfinite(out["coverage_logdet"])