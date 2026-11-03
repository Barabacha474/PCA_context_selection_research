"""Plotting utilities for diversity-vs-budget and spectral reports.

All plots are saved to file (no interactive backend). The function
imports ``matplotlib`` lazily so that ``import pcadiv.analysis`` stays
cheap when only spectral analysis is used.

Design notes
------------
* Every plot function accepts ``output_path`` and returns it.
  Determinism is enforced by passing ``dpi`` and ``bbox_inches``
  explicitly, so repeated calls produce identical files at the same
  path.
* Bands are drawn from pre-computed CI columns in the aggregated /
  comparison DataFrames. The plot functions do **not** compute CIs --
  that is done by ``stats.aggregate_seeds`` and ``stats.paired_compare``
  upstream, so that the same CIs are used for tables and figures.
* An empty / single-row DataFrame is not an error: the plot is
  produced with just what is available. This keeps the fallback
  analysis pipeline robust to degenerate sweeps.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "plot_diversity_vs_budget",
    "plot_delta_vs_reference",
    "plot_spectrum",
]


_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


def _mpl():
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:
        raise ImportError(
            "plots require 'matplotlib'. Install with: pip install matplotlib"
        ) from e


def plot_diversity_vs_budget(
    agg_df: pd.DataFrame,
    metric: str,
    output_path: str | Path,
    *,
    x: str = "target_budget",
    method_col: str = "method",
    dpi: int = 140,
    title: str | None = None,
) -> Path:
    """Line plot of ``metric`` vs ``target_budget`` with CI bands.

    Expects the output of :func:`~pcadiv.evaluation.stats.aggregate_seeds`
    with columns ``{metric}_mean``, ``{metric}_ci_low``,
    ``{metric}_ci_high`` alongside ``method`` and ``x``. If CI columns
    are missing, draws only the mean line.

    Parameters
    ----------
    agg_df
        Aggregated DataFrame (one row per ``method, x``).
    metric
        Base metric name (``"vendi_score"``, ``"mean_pairwise"``, ...).
    output_path
        Destination ``.png``.
    x
        Column to use as x-axis (default ``target_budget``).
    method_col
        Column with method labels.
    dpi
        Figure DPI.
    title
        If given, set as title; else a default based on ``metric``.
    """
    plt = _mpl()
    mean_col = f"{metric}_mean"
    lo_col = f"{metric}_ci_low"
    hi_col = f"{metric}_ci_high"
    if mean_col not in agg_df.columns:
        raise KeyError(
            f"{mean_col!r} not in aggregated DataFrame; "
            f"available: {sorted(agg_df.columns)}"
        )
    if x not in agg_df.columns:
        raise KeyError(f"{x!r} not in aggregated DataFrame")

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=dpi)
    methods = sorted(agg_df[method_col].unique())
    has_ci = lo_col in agg_df.columns and hi_col in agg_df.columns

    for i, m in enumerate(methods):
        sub = agg_df[agg_df[method_col] == m].sort_values(x)
        color = _PALETTE[i % len(_PALETTE)]
        ax.plot(
            sub[x], sub[mean_col],
            marker="o", markersize=4, linewidth=1.6,
            color=color, label=m,
        )
        if has_ci:
            ax.fill_between(
                sub[x], sub[lo_col], sub[hi_col],
                color=color, alpha=0.15, linewidth=0,
            )

    ax.set_xlabel("Token budget")
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} vs token budget")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_delta_vs_reference(
    compare_df: pd.DataFrame,
    metric: str,
    output_path: str | Path,
    *,
    x: str = "target_budget",
    method_col: str = "method",
    dpi: int = 140,
    title: str | None = None,
) -> Path:
    """Plot ``delta = method − reference`` with CI bands and a zero line.

    Expects the output of :func:`~pcadiv.evaluation.stats.paired_compare`
    with columns ``{metric}_delta``, ``{metric}_ci_low``,
    ``{metric}_ci_high``. The zero horizontal line is drawn so that
    "does the CI cross zero?" is visible at a glance.
    """
    plt = _mpl()
    delta_col = f"{metric}_delta"
    lo_col = f"{metric}_ci_low"
    hi_col = f"{metric}_ci_high"
    if delta_col not in compare_df.columns:
        raise KeyError(
            f"{delta_col!r} not in comparison DataFrame; "
            f"available: {sorted(compare_df.columns)}"
        )
    if x not in compare_df.columns:
        raise KeyError(f"{x!r} not in comparison DataFrame")

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=dpi)
    methods = sorted(compare_df[method_col].unique())
    has_ci = lo_col in compare_df.columns and hi_col in compare_df.columns

    for i, m in enumerate(methods):
        sub = compare_df[compare_df[method_col] == m].sort_values(x)
        color = _PALETTE[i % len(_PALETTE)]
        ax.plot(
            sub[x], sub[delta_col],
            marker="o", markersize=4, linewidth=1.6,
            color=color, label=m,
        )
        if has_ci:
            ax.fill_between(
                sub[x], sub[lo_col], sub[hi_col],
                color=color, alpha=0.15, linewidth=0,
            )

    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("Token budget")
    ax.set_ylabel(f"Δ {metric} (method − reference)")
    ax.set_title(title or f"Δ {metric} vs reference")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_spectrum(
    eigvals: np.ndarray,
    n: int,
    d: int,
    output_path: str | Path,
    *,
    sigma_sq: float | None = None,
    dpi: int = 140,
    title: str | None = None,
) -> Path:
    """Plot the eigenvalue spectrum with MP bounds overlaid.

    On log-y so that the tail and the MP bulk are both visible. The
    MP bounds are drawn as horizontal dashed lines; eigenvalues above
    the upper bound are highlighted in red (the "signal" part).
    """
    plt = _mpl()
    from pcadiv.analysis.spectral import (
        classify_signal_noise,
        marchenko_pastur_bounds,
    )

    eigs = np.asarray(eigvals, dtype=np.float64)
    if eigs.size == 0:
        # Empty spectrum: still write a file, so downstream code can
        # expect a consistent artefact.
        fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=dpi)
        ax.text(0.5, 0.5, "empty spectrum", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        return out

    if sigma_sq is None:
        sigma_sq = float(np.median(eigs))
    lower, upper = marchenko_pastur_bounds(n, d, sigma_sq)
    cls = classify_signal_noise(eigs, n, d, sigma_sq=sigma_sq)

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=dpi)
    ranks = np.arange(1, len(eigs) + 1)
    in_bulk = eigs <= upper
    above = ~in_bulk
    ax.scatter(ranks[in_bulk], eigs[in_bulk],
               s=8, color="#1f77b4", label="inside MP bulk")
    ax.scatter(ranks[above], eigs[above],
               s=8, color="#d62728", label="above MP upper bound")

    ax.axhline(lower, color="black", linewidth=0.8, linestyle="--",
               alpha=0.6, label=f"MP lower = {lower:.3g}")
    ax.axhline(upper, color="black", linewidth=0.8, linestyle=":",
               alpha=0.6, label=f"MP upper = {upper:.3g}")

    ax.set_xlabel("Rank")
    ax.set_ylabel("Eigenvalue of XᵀX/N")
    ax.set_yscale("log")
    ax.set_title(
        title
        or f"pool spectrum (n={n}, d={d}, "
           f"{cls['n_signal']}/{len(eigs)} above MP)"
    )
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out