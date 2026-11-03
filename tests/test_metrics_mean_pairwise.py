"""Tests for pcadiv.evaluation.metrics -- Block 5."""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from pcadiv.evaluation.metrics import (
    BaseMetric,
    MeanPairwiseDistance,
    get_metric,
    list_metrics,
    pairwise_distances,
    register_metric,
)


# ---- registry -------------------------------------------------------

def test_registry_contains_mean_pairwise():
    assert "mean_pairwise" in list_metrics()


def test_get_metric_returns_instance():
    m = get_metric("mean_pairwise")
    assert isinstance(m, MeanPairwiseDistance)


def test_get_metric_unknown_raises():
    with pytest.raises(KeyError, match="unknown metric"):
        get_metric("does-not-exist")


def test_register_duplicate_name_raises():
    class Fake(BaseMetric):
        name = "mean_pairwise"
        def __call__(self, emb, *, core_emb=None):
            return {}

    with pytest.raises(ValueError, match="already registered"):
        register_metric(Fake)


def test_invalid_metric_name_rejected_at_init():
    with pytest.raises(ValueError, match="unknown metric"):
        MeanPairwiseDistance(metric="euclidian")


# ---- degenerate cases ----------------------------------------------

def test_empty_input_returns_zero():
    m = MeanPairwiseDistance()
    assert m(np.zeros((0, 4), dtype=np.float32)) == {"mean_pairwise": 0.0}


def test_single_point_returns_zero():
    m = MeanPairwiseDistance()
    assert m(np.ones((1, 4), dtype=np.float32)) == {"mean_pairwise": 0.0}


def test_two_identical_points_returns_zero():
    m = MeanPairwiseDistance()
    X = np.ones((2, 4), dtype=np.float32)
    assert m(X)["mean_pairwise"] == pytest.approx(0.0, abs=1e-7)


# ---- known distances -----------------------------------------------

def test_two_orthogonal_unit_vectors_euclidean():
    m = MeanPairwiseDistance(metric="euclidean")
    X = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    assert m(X)["mean_pairwise"] == pytest.approx(np.sqrt(2.0), rel=1e-6)


def test_two_orthogonal_unit_vectors_cosine():
    m = MeanPairwiseDistance(metric="cosine")
    X = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    assert m(X)["mean_pairwise"] == pytest.approx(1.0, rel=1e-6)


def test_two_points_known_distance():
    m = MeanPairwiseDistance()
    X = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
    assert m(X)["mean_pairwise"] == pytest.approx(5.0, rel=1e-6)


def test_three_points_manual_euclidean():
    # distances: (0,0)-(1,0)=1, (0,0)-(0,2)=2, (1,0)-(0,2)=sqrt(5)
    # mean = (1 + 2 + sqrt(5)) / 3
    X = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]], dtype=np.float32
    )
    m = MeanPairwiseDistance()
    expected = (1.0 + 2.0 + np.sqrt(5.0)) / 3.0
    assert m(X)["mean_pairwise"] == pytest.approx(expected, rel=1e-6)


def test_euclidean_and_cosine_differ():
    X = np.array(
        [[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]], dtype=np.float32
    )
    e = MeanPairwiseDistance(metric="euclidean")(X)["mean_pairwise"]
    c = MeanPairwiseDistance(metric="cosine")(X)["mean_pairwise"]
    assert e != pytest.approx(c)


# ---- float64 accumulation ------------------------------------------

def test_accepts_float32_and_float64():
    X32 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    X64 = X32.astype(np.float64)
    m = MeanPairwiseDistance()
    np.testing.assert_allclose(
        m(X32)["mean_pairwise"], m(X64)["mean_pairwise"], rtol=1e-6
    )


def test_input_not_mutated():
    X = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    X_copy = X.copy()
    MeanPairwiseDistance()(X)
    np.testing.assert_array_equal(X, X_copy)


# ---- output contract -----------------------------------------------

def test_returns_plain_dict():
    out = MeanPairwiseDistance()(np.ones((3, 2), dtype=np.float32))
    assert type(out) is dict
    assert set(out) == {"mean_pairwise"}
    assert isinstance(out["mean_pairwise"], float)


def test_metric_does_not_accept_budget_kwarg():
    """Metrics must not know about the budget (see module docstring)."""
    sig = inspect.signature(MeanPairwiseDistance.__call__)
    params = set(sig.parameters)
    assert "budget" not in params
    assert "cum_tokens" not in params


# ---- core_emb handling ---------------------------------------------

def test_accepts_core_emb_and_ignores_it():
    X = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    m = MeanPairwiseDistance()
    core = np.zeros(2, dtype=np.float32)
    a = m(X)["mean_pairwise"]
    b = m(X, core_emb=core)["mean_pairwise"]
    assert a == b


def test_core_emb_wrong_dim_raises():
    m = MeanPairwiseDistance()
    X = np.ones((3, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="core_emb dim mismatch"):
        m(X, core_emb=np.zeros(3, dtype=np.float32))


# ---- input validation ----------------------------------------------

def test_1d_input_raises():
    m = MeanPairwiseDistance()
    with pytest.raises(ValueError, match="emb must be 2-D"):
        m(np.zeros(5, dtype=np.float32))


# ---- pairwise_distances kernel -------------------------------------

def test_pairwise_distances_symmetric_zero_diagonal():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 6))
    D = pairwise_distances(X, "euclidean")
    np.testing.assert_allclose(D, D.T, atol=1e-12)
    np.testing.assert_array_equal(np.diag(D), 0.0)


def test_pairwise_distances_rejects_unknown_metric():
    with pytest.raises(ValueError, match="unknown metric"):
        pairwise_distances(np.zeros((3, 2)), "manhattan")


def test_pairwise_cosine_zero_vector_convention():
    """A zero row must be maximally distant from everything (cos dist = 1)."""
    X = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    D = pairwise_distances(X, "cosine")
    assert D[0, 1] == pytest.approx(1.0)
    assert D[1, 0] == pytest.approx(1.0)
    assert D[1, 1] == pytest.approx(0.0)