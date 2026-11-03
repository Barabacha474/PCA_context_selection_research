"""Tests for spectral.py -- Block 13a."""
from __future__ import annotations

import numpy as np
import pytest

from pcadiv.analysis.spectral import (
    classify_signal_noise,
    explained_variance_at_k,
    marchenko_pastur_bounds,
    participation_ratio,
    pool_eigenvalues,
    spectral_summary,
)


# ---- pool_eigenvalues ----------------------------------------------

def test_pool_eigenvalues_shape_N_ge_D():
    X = np.random.default_rng(0).standard_normal((100, 20))
    eigs = pool_eigenvalues(X)
    assert eigs.shape == (20,)


def test_pool_eigenvalues_shape_N_lt_D():
    X = np.random.default_rng(0).standard_normal((10, 50))
    eigs = pool_eigenvalues(X)
    assert eigs.shape == (10,)


def test_pool_eigenvalues_descending_and_nonnegative():
    X = np.random.default_rng(0).standard_normal((50, 10))
    eigs = pool_eigenvalues(X)
    assert (np.diff(eigs) <= 1e-12).all()
    assert (eigs >= 0).all()


def test_pool_eigenvalues_rank1_structure():
    """One dominant direction -> exactly one large eigenvalue."""
    rng = np.random.default_rng(0)
    n = 100
    direction = np.array([1.0, 0.0, 0.0])
    X = direction * rng.standard_normal((n, 1)) + 1e-4 * rng.standard_normal((n, 3))
    eigs = pool_eigenvalues(X)
    assert eigs[0] > 100 * eigs[1]  # gap of two orders of magnitude


def test_pool_eigenvalues_identity_covariance():
    """White noise with variance 1 -> all eigenvalues near 1."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2000, 20))
    eigs = pool_eigenvalues(X)
    # top should be close to MP upper, not huge
    assert eigs[0] < 3.0  # quite loose; MP upper for gamma=0.01 is ~1.27


def test_pool_eigenvalues_empty_input():
    X = np.zeros((0, 10))
    eigs = pool_eigenvalues(X)
    assert eigs.shape == (0,)


def test_pool_eigenvalues_1d_input_raises():
    with pytest.raises(ValueError, match="2-D"):
        pool_eigenvalues(np.zeros(10))


def test_pool_eigenvalues_center_false_gives_ones_mode():
    """Without centering a non-zero-mean pool has a large 'ones' eigenvalue."""
    X = np.random.default_rng(0).standard_normal((100, 5)) + 10.0
    eigs_centered = pool_eigenvalues(X, center=True)
    eigs_raw = pool_eigenvalues(X, center=False)
    assert eigs_raw[0] > 10 * eigs_centered[0]


# ---- marchenko_pastur_bounds ---------------------------------------

def test_mp_bounds_known_gamma_1():
    """gamma = 1 -> MP support is [0, 4σ²]."""
    lower, upper = marchenko_pastur_bounds(100, 100, 1.0)
    assert lower == pytest.approx(0.0)
    assert upper == pytest.approx(4.0)


def test_mp_bounds_known_gamma_small():
    """gamma = 0.01 -> support ≈ [0.81σ², 1.21σ²]."""
    lower, upper = marchenko_pastur_bounds(1000, 10, 1.0)
    assert lower == pytest.approx(0.9 ** 2)
    assert upper == pytest.approx(1.1 ** 2)


def test_mp_bounds_gamma_gt_1_lower_zero():
    """gamma > 1 -> lower edge is 0 (D - N zero eigenvalues)."""
    lower, upper = marchenko_pastur_bounds(10, 50, 1.0)
    assert lower == 0.0
    assert upper > 0


def test_mp_bounds_zero_sigma():
    lower, upper = marchenko_pastur_bounds(100, 10, 0.0)
    assert lower == 0.0 and upper == 0.0


def test_mp_bounds_negative_sigma_raises():
    with pytest.raises(ValueError, match="non-negative"):
        marchenko_pastur_bounds(100, 10, -1.0)


# ---- classify_signal_noise -----------------------------------------

def test_classify_all_noise():
    """Pure iid noise: essentially no eigenvalues above MP upper."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2000, 20))
    eigs = pool_eigenvalues(X)
    cls = classify_signal_noise(eigs, 2000, 20)
    # For gamma = 0.01, expect at most a handful of outliers.
    assert cls["n_signal"] < 5
    assert cls["n_noise"] == len(eigs) - cls["n_signal"]


def test_classify_with_clear_signal():
    """One strong rank-1 direction -> at least 1 signal above MP."""
    rng = np.random.default_rng(0)
    n, d = 1000, 10
    X = rng.standard_normal((n, d)) * 0.1  # noise
    X[:, 0] += rng.standard_normal(n) * 10  # signal
    eigs = pool_eigenvalues(X)
    cls = classify_signal_noise(eigs, n, d)
    assert cls["n_signal"] >= 1
    assert cls["frac_signal"] > 0


def test_classify_sigma_sq_supplied():
    eigs = np.array([10.0, 1.0, 1.0, 1.0])
    cls = classify_signal_noise(eigs, 100, 4, sigma_sq=1.0)
    assert cls["sigma_sq"] == 1.0
    # MP upper for gamma=0.04: (1.2)^2 = 1.44; eigs[0]=10 > 1.44
    assert cls["n_signal"] == 1


def test_classify_empty_eigs():
    cls = classify_signal_noise(np.empty(0), 10, 10)
    assert cls["n_signal"] == 0
    assert cls["frac_signal"] == 0.0


# ---- participation_ratio --------------------------------------------

def test_participation_ratio_rank1():
    """Rank-1 spectrum -> PR = 1."""
    eigs = np.array([10.0, 0.0, 0.0, 0.0])
    assert participation_ratio(eigs) == pytest.approx(1.0)


def test_participation_ratio_uniform():
    """Uniform spectrum -> PR = number of dimensions."""
    eigs = np.ones(8)
    assert participation_ratio(eigs) == pytest.approx(8.0)


def test_participation_ratio_empty():
    assert participation_ratio(np.empty(0)) == 0.0


def test_participation_ratio_all_zero():
    assert participation_ratio(np.zeros(5)) == 0.0


# ---- explained_variance_at_k ---------------------------------------

def test_explained_variance_rank1_k1_is_one():
    eigs = np.array([10.0, 0.0, 0.0])
    out = explained_variance_at_k(eigs, [1, 2, 3])
    assert out[1] == pytest.approx(1.0)
    assert out[2] == pytest.approx(1.0)
    assert out[3] == pytest.approx(1.0)


def test_explained_variance_uniform():
    eigs = np.ones(10)
    out = explained_variance_at_k(eigs, [1, 5, 10])
    assert out[1] == pytest.approx(0.1)
    assert out[5] == pytest.approx(0.5)
    assert out[10] == pytest.approx(1.0)


def test_explained_variance_k_capped():
    eigs = np.array([3.0, 1.0])
    out = explained_variance_at_k(eigs, [5])
    assert out[5] == pytest.approx(1.0)


def test_explained_variance_zero_total():
    out = explained_variance_at_k(np.zeros(5), [1, 3])
    assert out[1] == 0.0 and out[3] == 0.0


# ---- spectral_summary ----------------------------------------------

def test_summary_returns_expected_keys():
    X = np.random.default_rng(0).standard_normal((100, 10))
    s = spectral_summary(X, ks=(1, 5))
    for key in (
        "n", "d", "top_eigenvalues", "total_variance",
        "sigma_sq", "mp_lower", "mp_upper",
        "n_signal", "n_noise", "frac_signal",
        "participation_ratio", "explained_variance_at_k",
    ):
        assert key in s
    assert s["n"] == 100 and s["d"] == 10


def test_summary_1d_input_raises():
    with pytest.raises(ValueError, match="2-D"):
        spectral_summary(np.zeros(10))


def test_summary_clear_signal_detected():
    rng = np.random.default_rng(0)
    n, d = 500, 10
    X = rng.standard_normal((n, d)) * 0.1
    X[:, 0] += rng.standard_normal(n) * 10
    s = spectral_summary(X)
    assert s["n_signal"] >= 1