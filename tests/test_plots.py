"""Tests for plots.py -- Block 13b."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)

from pcadiv.analysis.plots import (
    plot_delta_vs_reference,
    plot_diversity_vs_budget,
    plot_spectrum,
)


@pytest.fixture
def agg_df():
    rows = []
    for method in ("random", "iterative_pca", "mmr"):
        for budget in (512, 1024, 2048):
            base = {"random": 5.0, "iterative_pca": 8.0, "mmr": 7.0}[method]
            mean = base + 0.001 * budget
            rows.append({
                "method": method,
                "target_budget": budget,
                "vendi_score_mean": mean,
                "vendi_score_ci_low": mean - 0.3,
                "vendi_score_ci_high": mean + 0.3,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def compare_df():
    rows = []
    for method in ("iterative_pca", "mmr"):
        for budget in (512, 1024, 2048):
            delta = {"iterative_pca": 3.0, "mmr": 2.0}[method]
            rows.append({
                "method": method,
                "target_budget": budget,
                "vendi_score_delta": delta,
                "vendi_score_ci_low": delta - 0.2,
                "vendi_score_ci_high": delta + 0.2,
            })
    return pd.DataFrame(rows)


# ---- plot_diversity_vs_budget --------------------------------------

def test_diversity_plot_creates_file(agg_df, tmp_path):
    p = plot_diversity_vs_budget(
        agg_df, "vendi_score", tmp_path / "div.png"
    )
    assert p.exists()
    assert p.stat().st_size > 1000  # not an empty file


def test_diversity_plot_without_ci_columns(tmp_path):
    df = pd.DataFrame({
        "method": ["a", "a", "b", "b"],
        "target_budget": [512, 1024, 512, 1024],
        "vendi_score_mean": [1.0, 2.0, 3.0, 4.0],
    })
    p = plot_diversity_vs_budget(df, "vendi_score", tmp_path / "no_ci.png")
    assert p.exists()


def test_diversity_plot_missing_mean_column_raises(agg_df, tmp_path):
    with pytest.raises(KeyError, match="not in aggregated"):
        plot_diversity_vs_budget(agg_df, "unknown_metric", tmp_path / "x.png")


def test_diversity_plot_missing_x_column_raises(agg_df, tmp_path):
    with pytest.raises(KeyError, match="not_a_column"):
        plot_diversity_vs_budget(
            agg_df, "vendi_score", tmp_path / "x.png", x="not_a_column"
        )


def test_diversity_plot_deterministic(agg_df, tmp_path):
    p1 = plot_diversity_vs_budget(agg_df, "vendi_score", tmp_path / "a.png")
    p2 = plot_diversity_vs_budget(agg_df, "vendi_score", tmp_path / "b.png")
    assert p1.read_bytes() == p2.read_bytes()


# ---- plot_delta_vs_reference ---------------------------------------

def test_delta_plot_creates_file(compare_df, tmp_path):
    p = plot_delta_vs_reference(
        compare_df, "vendi_score", tmp_path / "delta.png"
    )
    assert p.exists()
    assert p.stat().st_size > 1000


def test_delta_plot_missing_delta_raises(compare_df, tmp_path):
    with pytest.raises(KeyError, match="not in comparison"):
        plot_delta_vs_reference(
            compare_df, "unknown_metric", tmp_path / "x.png"
        )


# ---- plot_spectrum -------------------------------------------------

def test_spectrum_plot_creates_file(tmp_path):
    eigs = np.array([100.0, 50.0, 10.0, 5.0, 2.0, 1.5, 1.1, 1.0, 0.9])
    p = plot_spectrum(eigs, n=100, d=9, output_path=tmp_path / "spec.png")
    assert p.exists()
    assert p.stat().st_size > 1000


def test_spectrum_plot_empty_spectrum(tmp_path):
    p = plot_spectrum(np.empty(0), n=100, d=9, output_path=tmp_path / "empty.png")
    assert p.exists()


def test_spectrum_plot_with_explicit_sigma(tmp_path):
    eigs = np.array([100.0] + [1.0] * 20)
    p = plot_spectrum(
        eigs, n=100, d=20, output_path=tmp_path / "spec.png", sigma_sq=1.0
    )
    assert p.exists()