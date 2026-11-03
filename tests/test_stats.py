"""Tests for stats.py -- Block 9d."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pcadiv.evaluation import (
    aggregate_seeds,
    interpolate_at_budgets,
    paired_compare,
    pivot_summary,
)


# ---- interpolate_at_budgets ----------------------------------------

@pytest.fixture
def one_curve():
    return pd.DataFrame({
        "n_selected": [0, 1, 2, 3, 4],
        "token_budget": [0, 54, 171, 250, 485],
        "mean_pairwise": [0.0, 0.0, 1.1, 1.05, 1.09],
    })


def test_interpolate_step_known_values(one_curve):
    out = interpolate_at_budgets(
        one_curve,
        budgets=[0, 100, 200, 500],
        value_cols=["mean_pairwise"],
        method="step",
    )
    # target 0   -> row at budget 0   -> 0.0
    # target 100 -> largest <= 100 is 54 -> 0.0
    # target 200 -> largest <= 200 is 171 -> 1.1
    # target 500 -> largest <= 500 is 485 -> 1.09
    assert out["mean_pairwise"].tolist() == [0.0, 0.0, 1.1, 1.09]
    assert out["token_budget"].tolist() == [0, 100, 200, 500]


def test_interpolate_linear_known_values(one_curve):
    out = interpolate_at_budgets(
        one_curve,
        budgets=[112],  # halfway between 54 and 171
        value_cols=["mean_pairwise"],
        method="linear",
    )
    # 0.0 + (1.1 - 0.0) * (112 - 54) / (171 - 54) = 1.1 * 58/117 ≈ 0.545
    expected = 1.1 * 58 / 117
    assert out["mean_pairwise"].iloc[0] == pytest.approx(expected, rel=1e-6)


def test_interpolate_linear_extrapolates_flat(one_curve):
    """Beyond max budget linear mode is flat (np.interp default)."""
    out = interpolate_at_budgets(
        one_curve, budgets=[10000], value_cols=["mean_pairwise"], method="linear"
    )
    assert out["mean_pairwise"].iloc[0] == pytest.approx(1.09)


def test_interpolate_rejects_negative_budget(one_curve):
    with pytest.raises(ValueError, match="non-negative"):
        interpolate_at_budgets(
            one_curve, budgets=[-1], value_cols=["mean_pairwise"]
        )


def test_interpolate_rejects_unknown_method(one_curve):
    with pytest.raises(ValueError, match="method must be"):
        interpolate_at_budgets(
            one_curve, budgets=[0], value_cols=["mean_pairwise"],
            method="cubic",
        )


def test_interpolate_missing_value_col(one_curve):
    with pytest.raises(KeyError, match="not in DataFrame"):
        interpolate_at_budgets(
            one_curve, budgets=[0], value_cols=["not_a_column"]
        )


def test_interpolate_empty_df():
    df = pd.DataFrame(columns=["token_budget", "x"])
    with pytest.raises(ValueError, match="df is empty"):
        interpolate_at_budgets(df, budgets=[0], value_cols=["x"])


def test_interpolate_preserves_order_of_columns(one_curve):
    out = interpolate_at_budgets(
        one_curve, budgets=[0, 200], value_cols=["mean_pairwise"]
    )
    assert list(out.columns) == ["token_budget", "mean_pairwise"]


# ---- aggregate_seeds -----------------------------------------------

@pytest.fixture
def long_df():
    """Two methods, three seeds, three budgets each."""
    rows = []
    for method, base in (("A", 0.0), ("B", 1.0)):
        for s in (0, 1, 2):
            for b in (100, 200, 300):
                rows.append({
                    "method": method,
                    "seed": s,
                    "token_budget": b,
                    "metric": base + 0.1 * b / 100 + 0.01 * s,
                })
    return pd.DataFrame(rows)


def test_aggregate_shape_and_columns(long_df):
    out = aggregate_seeds(
        long_df,
        by=["method", "token_budget"],
        value_cols=["metric"],
        n_resamples=200,
        seed=0,
    )
    assert len(out) == 2 * 3  # 2 methods × 3 budgets
    expected = {
        "method", "token_budget", "n_seeds",
        "metric_mean", "metric_std", "metric_ci_low", "metric_ci_high",
    }
    assert set(out.columns) == expected


def test_aggregate_mean_matches_manual(long_df):
    out = aggregate_seeds(
        long_df,
        by=["method", "token_budget"],
        value_cols=["metric"],
        n_resamples=200,
        seed=0,
    )
    # For method A at budget 100, seeds give 0.1 + {0, 0.01, 0.02}
    sub = long_df[(long_df.method == "A") & (long_df.token_budget == 100)]
    expected_mean = float(sub["metric"].mean())
    got = out[(out.method == "A") & (out.token_budget == 100)]["metric_mean"].iloc[0]
    assert got == pytest.approx(expected_mean)


def test_aggregate_ci_contains_mean(long_df):
    out = aggregate_seeds(
        long_df,
        by=["method", "token_budget"],
        value_cols=["metric"],
        n_resamples=500,
        seed=0,
    )
    for _, row in out.iterrows():
        assert row["metric_ci_low"] <= row["metric_mean"] <= row["metric_ci_high"]


def test_aggregate_std_matches_numpy(long_df):
    out = aggregate_seeds(
        long_df,
        by=["method", "token_budget"],
        value_cols=["metric"],
    )
    sub = long_df[(long_df.method == "A") & (long_df.token_budget == 100)]
    expected_std = float(sub["metric"].std(ddof=1))
    got = out[(out.method == "A") & (out.token_budget == 100)]["metric_std"].iloc[0]
    assert got == pytest.approx(expected_std)


def test_aggregate_single_seed_std_zero_and_ci_equal_mean():
    df = pd.DataFrame({
        "method": ["A"],
        "seed": [0],
        "token_budget": [100],
        "metric": [1.5],
    })
    out = aggregate_seeds(df, by=["method", "token_budget"], value_cols=["metric"])
    row = out.iloc[0]
    assert row["metric_std"] == 0.0
    assert row["metric_ci_low"] == 1.5
    assert row["metric_ci_high"] == 1.5


def test_aggregate_seed_in_by_raises(long_df):
    with pytest.raises(ValueError, match="'seed' must not be in"):
        aggregate_seeds(long_df, by=["seed", "token_budget"], value_cols=["metric"])


def test_aggregate_missing_seed_col():
    df = pd.DataFrame({"method": ["A"], "token_budget": [100], "metric": [1.0]})
    with pytest.raises(KeyError, match="'seed' column"):
        aggregate_seeds(df, by=["method"], value_cols=["metric"])


def test_aggregate_deterministic_given_seed(long_df):
    a = aggregate_seeds(long_df, by=["method", "token_budget"], value_cols=["metric"], seed=7)
    b = aggregate_seeds(long_df, by=["method", "token_budget"], value_cols=["metric"], seed=7)
    pd.testing.assert_frame_equal(a, b)


# ---- paired_compare ------------------------------------------------

def test_paired_compare_excludes_reference_from_output(long_df):
    """reference must not appear as a compared method in the result."""
    out = paired_compare(
        long_df, reference="A", by=["token_budget"], value_cols=["metric"]
    )
    assert "A" not in out["method"].unique()
    assert set(out["method"].unique()) == {"B"}


def test_paired_compare_zero_delta_for_identical_duplicate(long_df):
    """A method with values identical to the reference has delta = 0."""
    dup = long_df[long_df.method == "A"].copy()
    dup["method"] = "A_copy"
    df = pd.concat([long_df, dup], ignore_index=True)
    out = paired_compare(
        df, reference="A", by=["token_budget"], value_cols=["metric"]
    )
    a_copy = out[out.method == "A_copy"]
    assert len(a_copy) > 0
    for _, row in a_copy.iterrows():
        assert row["metric_delta"] == pytest.approx(0.0, abs=1e-12)
        assert row["metric_ci_low"] == pytest.approx(0.0, abs=1e-12)
        assert row["metric_ci_high"] == pytest.approx(0.0, abs=1e-12)


def test_paired_compare_shift(long_df):
    """B is A + 1.0 per seed/budget cell -> delta = 1.0."""
    out = paired_compare(
        long_df, reference="A", by=["token_budget"], value_cols=["metric"]
    )
    for _, row in out.iterrows():
        assert row["metric_delta"] == pytest.approx(1.0)


def test_paired_compare_unknown_reference(long_df):
    with pytest.raises(ValueError, match="reference"):
        paired_compare(
            long_df, reference="Z", by=["token_budget"], value_cols=["metric"]
        )


def test_paired_compare_no_seed_overlap():
    df = pd.DataFrame({
        "method": ["A", "A", "B", "B"],
        "seed": [0, 1, 2, 3],  # disjoint
        "token_budget": [100, 100, 100, 100],
        "metric": [0.0, 0.1, 1.0, 1.1],
    })
    out = paired_compare(df, reference="A", by=["token_budget"], value_cols=["metric"])
    assert len(out) == 0


def test_paired_compare_partial_seed_overlap():
    df = pd.DataFrame({
        "method": ["A", "A", "A", "B", "B"],
        "seed": [0, 1, 2, 1, 2],
        "token_budget": [100] * 5,
        "metric": [0.0, 0.1, 0.2, 1.0, 1.1],
    })
    out = paired_compare(df, reference="A", by=["token_budget"], value_cols=["metric"])
    assert len(out) == 1
    assert out["n_seeds"].iloc[0] == 2  # only seeds 1, 2 overlap


def test_paired_compare_multi_by():
    df = pd.DataFrame({
        "method": ["A", "A", "A", "A", "B", "B", "B", "B"],
        "seed": [0, 1, 0, 1, 0, 1, 0, 1],
        "token_budget": [100, 100, 200, 200, 100, 100, 200, 200],
        "metric": [0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 2.0, 2.1],
    })
    out = paired_compare(
        df, reference="A", by=["token_budget"], value_cols=["metric"]
    )
    # One row per budget (100, 200)
    assert sorted(out["token_budget"].tolist()) == [100, 200]
    # delta at 100 = (1.0-0.0 + 1.1-0.1)/2 = 1.0
    # delta at 200 = (2.0-0.2 + 2.1-0.3)/2 = 1.8
    assert out[out.token_budget == 100]["metric_delta"].iloc[0] == pytest.approx(1.0)
    assert out[out.token_budget == 200]["metric_delta"].iloc[0] == pytest.approx(1.8)


# ---- pivot_summary -------------------------------------------------

def test_pivot_summary_shape(long_df):
    agg = aggregate_seeds(
        long_df, by=["method", "token_budget"], value_cols=["metric"]
    )
    wide = pivot_summary(agg, values="metric_mean")
    assert wide.shape == (3, 2)  # 3 budgets × 2 methods
    assert sorted(wide.columns.tolist()) == ["A", "B"]
    assert sorted(wide.index.tolist()) == [100, 200, 300]


def test_pivot_summary_values(long_df):
    agg = aggregate_seeds(
        long_df, by=["method", "token_budget"], value_cols=["metric"]
    )
    wide = pivot_summary(agg, values="metric_mean")
    sub_A_100 = long_df[(long_df.method == "A") & (long_df.token_budget == 100)]
    assert wide.loc[100, "A"] == pytest.approx(float(sub_A_100["metric"].mean()))