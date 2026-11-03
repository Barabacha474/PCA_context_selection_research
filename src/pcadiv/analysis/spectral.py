"""Spectral diagnostics for the pool embedding matrix.

Purpose
-------
Answer, before any selection is run, whether the embedding pool has
**low-rank structure at all** -- i.e., whether a method based on
principal subspaces (iterative PCA, static PCA, FPS) can in principle
find anything meaningful.

If the top eigenvalues of the pool covariance lie inside the
Marchenko-Pastur bulk, then the pool is indistinguishable from iid
noise from the subspace viewpoint, and our proposed method cannot beat
random by construction. Documenting this is the fallback / negative
result branch (see proposal, section 7).

Conventions
-----------
``pool_eigenvalues`` returns eigenvalues of ``XᵀX / N`` (note ``N``,
not ``N-1``). This is the normalisation under which Marchenko-Pastur
is stated, so the bound comparison is exact. ``Coverage`` uses
``N-1`` -- that is fine because it does not compare to MP.

Noise variance ``σ²`` is estimated as ``median(eigenvalues)`` by
default. Rationale: if the pool has signal, it lives in the top of the
spectrum; the median of the bulk is robust to that. ``mean`` would be
inflated by the signal and would over-estimate σ², shrinking the
apparent "signal count".

Marchenko-Pastur
----------------
For an ``N × D`` matrix with iid entries of variance ``σ²``, the
eigenvalues of ``XᵀX / N`` concentrate on

    [ σ² (1 − √γ)² ,  σ² (1 + √γ)² ],   γ = D / N

For ``γ > 1`` the lower bound is 0 and there are ``D − N`` zero
eigenvalues; the function still returns the correct bounds, and the
zero eigenvalues are correctly classified as noise.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "pool_eigenvalues",
    "marchenko_pastur_bounds",
    "classify_signal_noise",
    "participation_ratio",
    "explained_variance_at_k",
    "spectral_summary",
]


def pool_eigenvalues(pool_emb: np.ndarray, center: bool = True) -> np.ndarray:
    """Return the eigenvalue spectrum of ``pool_emb``, descending.

    Parameters
    ----------
    pool_emb
        ``(N, D)`` embedding matrix. Upcast to ``float64`` internally.
    center
        If ``True`` (default), subtract the column mean first. Centered
        data is what PCA assumes; for raw embeddings the mean is small
        but non-zero, and including it would put a large spurious
        eigenvalue on the "all-ones" direction. Set to ``False`` only
        to reproduce a specific uncentered baseline.

    Returns
    -------
    numpy.ndarray
        Length ``min(N, D)`` eigenvalues, non-negative, descending.
        Uses the ``N×N`` Gram matrix when ``N < D`` (eigendecomposing
        ``D×D`` would be wasteful and rank-deficient anyway).

    Notes
    -----
    Normalisation is ``1/N`` (MP convention), not ``1/(N-1)``. See
    module docstring.
    """
    X = np.asarray(pool_emb, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"pool_emb must be 2-D, got ndim={X.ndim}")
    N, D = X.shape
    if N == 0 or D == 0:
        return np.empty(0, dtype=np.float64)
    if center:
        X = X - X.mean(axis=0, keepdims=True)

    if N >= D:
        G = X.T @ X / N
        eigvals = np.linalg.eigvalsh(G)  # ascending
    else:
        G = X @ X.T / N
        eigvals = np.linalg.eigvalsh(G)  # N eigenvalues

    eigvals = eigvals[::-1].copy()
    # Numerical noise can give tiny negatives; clamp.
    return np.maximum(eigvals, 0.0)


def marchenko_pastur_bounds(
    n: int, d: int, sigma_sq: float
) -> tuple[float, float]:
    """Return the Marchenko-Pastur ``(lower, upper)`` edges.

    Parameters
    ----------
    n, d
        Matrix shape: ``n`` samples, ``d`` dimensions.
    sigma_sq
        Per-entry noise variance. Typically estimated as
        ``median(pool_eigenvalues(...))``.

    Returns
    -------
    (lower, upper)
        Floating-point edges of the MP bulk. ``lower`` is 0 when
        ``d > n``; ``upper`` is always positive for ``sigma_sq > 0``.
        When ``sigma_sq == 0`` both are 0.
    """
    if n <= 0 or d <= 0:
        raise ValueError(f"n and d must be positive, got n={n}, d={d}")
    if sigma_sq < 0:
        raise ValueError(f"sigma_sq must be non-negative, got {sigma_sq}")
    if sigma_sq == 0:
        return 0.0, 0.0
    gamma = d / n
    sqrt_gamma = np.sqrt(gamma)
    lower = sigma_sq * max(0.0, 1.0 - sqrt_gamma) ** 2
    upper = sigma_sq * (1.0 + sqrt_gamma) ** 2
    return float(lower), float(upper)


def classify_signal_noise(
    eigvals: np.ndarray,
    n: int,
    d: int,
    sigma_sq: float | None = None,
) -> dict:
    """Classify eigenvalues as "above MP bulk" (signal) or "in bulk" (noise).

    Parameters
    ----------
    eigvals
        Descending spectrum from :func:`pool_eigenvalues`.
    n, d
        Original matrix shape. Must match ``eigvals``'s source.
    sigma_sq
        Noise variance. If ``None``, estimated as ``median(eigvals)``.

    Returns
    -------
    dict with keys:
        ``sigma_sq``        -- estimated or supplied variance
        ``mp_lower``        -- MP lower edge
        ``mp_upper``        -- MP upper edge
        ``n_signal``        -- count of eigenvalues strictly above ``mp_upper``
        ``n_noise``         -- count strictly below or equal
        ``frac_signal``     -- ``n_signal / len(eigvals)``
    """
    eigs = np.asarray(eigvals, dtype=np.float64)
    if sigma_sq is None:
        if eigs.size == 0:
            sigma_sq = 0.0
        else:
            sigma_sq = float(np.median(eigs))
    lower, upper = marchenko_pastur_bounds(n, d, sigma_sq)
    n_signal = int((eigs > upper).sum())
    return {
        "sigma_sq": float(sigma_sq),
        "mp_lower": float(lower),
        "mp_upper": float(upper),
        "n_signal": n_signal,
        "n_noise": int(len(eigs) - n_signal),
        "frac_signal": (
            float(n_signal) / len(eigs) if len(eigs) > 0 else 0.0
        ),
    }


def participation_ratio(eigvals: np.ndarray) -> float:
    """Participation ratio: ``(Σλ)² / Σλ²``.

    Effective number of dimensions carrying energy. Equals 1 for a
    rank-1 spectrum and ``K`` for ``K`` equal eigenvalues. Robust to
    small eigenvalues near zero (they contribute nothing to either
    numerator or denominator).
    """
    eigs = np.asarray(eigvals, dtype=np.float64)
    if eigs.size == 0:
        return 0.0
    s1 = float(eigs.sum())
    s2 = float((eigs ** 2).sum())
    if s2 <= 0:
        return 0.0
    return (s1 * s1) / s2


def explained_variance_at_k(
    eigvals: np.ndarray, ks: list[int] | tuple[int, ...]
) -> dict[int, float]:
    """Fraction of total variance explained by the top ``k`` directions.

    For each ``k`` in ``ks`` returns ``sum(eigvals[:k]) / sum(eigvals)``.
    ``k`` larger than ``len(eigvals)`` is capped. Total variance zero
    (all eigenvalues zero) returns 0 for every ``k``.
    """
    eigs = np.asarray(eigvals, dtype=np.float64)
    total = float(eigs.sum())
    out: dict[int, float] = {}
    for k in ks:
        kk = min(int(k), len(eigs))
        out[int(k)] = float(eigs[:kk].sum() / total) if total > 0 else 0.0
    return out


def spectral_summary(
    pool_emb: np.ndarray,
    ks: list[int] | tuple[int, ...] = (1, 5, 10, 50),
) -> dict:
    """One-shot spectral report for a pool.

    Combines :func:`pool_eigenvalues`, :func:`classify_signal_noise`,
    :func:`participation_ratio` and :func:`explained_variance_at_k`.
    Intended for logging and for the CLI in
    ``experiments/run_spectral_analysis.py``.
    """
    X = np.asarray(pool_emb)
    if X.ndim != 2:
        raise ValueError(f"pool_emb must be 2-D, got ndim={X.ndim}")
    n, d = X.shape
    eigs = pool_eigenvalues(X)
    cls = classify_signal_noise(eigs, n, d)
    return {
        "n": int(n),
        "d": int(d),
        "top_eigenvalues": eigs[: min(10, len(eigs))].tolist(),
        "total_variance": float(eigs.sum()),
        **cls,
        "participation_ratio": participation_ratio(eigs),
        "explained_variance_at_k": explained_variance_at_k(eigs, ks),
    }