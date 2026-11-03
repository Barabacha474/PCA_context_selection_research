"""Tests for budget_curve with multiple metrics -- Block 9c."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pcadiv.evaluation import budget_curve, get_metric
from pcadiv.evaluation.metrics import BaseMetric
from pcadiv.selection import get_selector
from pcadiv.selection.selectors import SelectionResult


@pytest.fixture
def setup():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((60, 8)).astype(np.float32)
    toks = rng.integers(5, 25, size=60).astype(np.int64)
    res = get_selector("random", seed=0).select(emb, toks, budget=300)
    return emb, toks, res


def _three_metrics():
    return [
        get_metric("mean_pairwise"),
        get_metric("coverage"),
        get_metric("vendi", gamma=1.0),
    ]


def test_three_metrics_all_columns_present(setup):
    emb, _, res = setup
    df = budget_curve(res, emb, _three_metrics())
    expected = {
        "n_selected", "token_budget",
        "mean_pairwise",
        "coverage_radius", "coverage_logdet",
        "vendi_score",
    }
    assert set(df.columns) == expected


def test_column_order_is_first_seen(setup):
    """mean_pairwise appears first, then coverage's two, then vendi."""
    emb, _, res = setup
    df = budget_curve(res, emb, _three_metrics())
    assert list(df.columns) == [
        "n_selected", "token_budget",
        "mean_pairwise",
        "coverage_radius", "coverage_logdet",
        "vendi_score",
    ]


def test_row_count_matches_len_plus_one(setup):
    emb, _, res = setup
    df = budget_curve(res, emb, _three_metrics())
    assert len(df) == len(res) + 1


def test_no_nan_with_three_metrics(setup):
    emb, _, res = setup
    df = budget_curve(res, emb, _three_metrics())
    assert not df.isna().any().any()


def test_first_row_all_metrics_defined(setup):
    """Empty prefix must have well-defined values for all three metrics."""
    emb, _, res = setup
    df = budget_curve(res, emb, _three_metrics())
    first = df.iloc[0]
    assert first["n_selected"] == 0
    assert first["token_budget"] == 0
    assert first["mean_pairwise"] == 0.0
    assert first["coverage_radius"] == 0.0
    assert first["coverage_logdet"] == 0.0
    assert first["vendi_score"] == 0.0


def test_last_row_matches_direct_metric_calls(setup):
    emb, _, res = setup
    metrics = _three_metrics()
    df = budget_curve(res, emb, metrics)
    sel_emb = emb[res.indices]
    for m in metrics:
        for key, val in m(sel_emb).items():
            assert df[key].iloc[-1] == pytest.approx(val)


def test_core_emb_forwarded_to_all_metrics(setup):
    """Every metric receives the same core_emb on every prefix."""
    emb, _, res = setup
    seen: list[dict] = []

    class Recorder(BaseMetric):
        name = "_rec"
        def __init__(self, tag): super().__init__(tag=tag); self.tag = tag
        def __call__(self, e, *, core_emb=None):
            seen.append({"tag": self.tag, "core_is_none": core_emb is None})
            return {f"rec_{self.tag}": 0.0}

    core = np.ones(emb.shape[1], dtype=np.float32)
    budget_curve(
        res, emb,
        [Recorder("a"), Recorder("b"), Recorder("c")],
        core_emb=core,
    )
    assert len(seen) == 3 * (len(res) + 1)
    assert all(not r["core_is_none"] for r in seen)


def test_known_growth_direction(setup):
    """Adding more points should increase Vendi and coverage radius
    (not guaranteed in general, but on this well-separated synthetic
    pool with random selection it holds)."""
    emb, _, res = setup
    df = budget_curve(res, emb, _three_metrics())
    assert df["coverage_radius"].iloc[-1] > df["coverage_radius"].iloc[1]
    assert df["vendi_score"].iloc[-1] > df["vendi_score"].iloc[1]