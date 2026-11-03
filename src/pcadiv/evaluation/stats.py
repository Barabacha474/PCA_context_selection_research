"""Aggregation and comparison utilities for sweep results.

Typical pipeline
----------------
1. Run a sweep: for each (method, seed), produce a ``budget_curve``
   DataFrame.
2. Concatenate all of them into one long DataFrame with a ``method``
   and ``seed`` column.
3. :func:`interpolate_at_budgets` -- bring every run onto a common
   budget grid (methods stop at different token counts because item
   costs differ).
4. :func:`aggregate_seeds` -- mean, std, bootstrap CI per
   ``(method, budget)``.
5. :func:`paired_compare` -- for each method vs a reference, paired
   delta with bootstrap CI.
6. :func:`pivot_summary` -- readable table for plotting or a paper.

Why interpolation is mandatory
------------------------------
Methods stop at different ``token_budget`` values because the next
candidate they would pick has a different cost. Comparing two methods
row-by-row on ``token_budget`` is therefore wrong: row 5 of method A
may be at 300 tokens while row 5 of method B is at 340. We
interpolate both onto a shared grid, then every downstream step
(aggregate, compare) is well-defined.

Interpolation mode
------------------
``method="step"`` (default) is the honest read: the metric value at
budget ``B`` is the value achieved by the largest prefix whose
``token_budget <= B``. This never extrapolates above what was actually
selected.

``method="linear"`` interpolates linearly between adjacent prefixes.
It can exceed or undershoot the true prefix values and is provided
only for plotting when step functions look too jagged.

Bootstrap, not t-tests
----------------------
With 3-5 seeds per (method, budget) cell, a t-test has no power and
its normality assumption is indefensible. We use nonparametric
bootstrap over the seed-level values: resample with replacement,
compute the mean, take percentiles. This gives a CI that means what it
says even for tiny n.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "interpolate_at_budgets",
    "aggregate_seeds",
    "paired_compare",
    "pivot_summary",
]


# ---------------------------------------------------------------------
# Interpolation onto a common budget grid
# ---------------------------------------------------------------------

def interpolate_at_budgets(
    df: pd.DataFrame,
    budgets: Sequence[int],
    value_cols: Sequence[str],
    method: str = "step",
    budget_col: str = "token_budget",
) -> pd.DataFrame:
    """Interpolate ``value_cols`` from ``df`` onto a fixed budget grid.

    Parameters
    ----------
    df
        Output of :func:`~pcadiv.evaluation.budget_curve.budget_curve`,
        one row per prefix, sorted by ``budget_col`` (that is what
        ``budget_curve`` produces).
    budgets
        Target budget grid, non-negative, will be sorted internally.
    value_cols
        Columns to interpolate.
    method
        ``"step"`` (default) or ``"linear"``. See module docstring.
    budget_col
        Name of the budget column. Default ``"token_budget"``.

    Returns
    -------
    pandas.DataFrame
        One row per target budget, with ``budget_col`` plus one column
        per ``value_cols``. Non-metric columns from ``df`` (e.g.
        ``method``, ``seed``) are not carried -- caller is expected to
        add them back after the call (they are constant within one
        ``df`` anyway).
    """
    if method not in ("step", "linear"):
        raise ValueError(f"method must be 'step' or 'linear', got {method!r}")
    if len(budgets) == 0:
        raise ValueError("budgets must be non-empty")

    budgets_arr = np.asarray(sorted(budgets), dtype=np.int64)
    if (budgets_arr < 0).any():
        raise ValueError("budgets must be non-negative")

    if budget_col not in df.columns:
        raise KeyError(f"{budget_col!r} not in DataFrame columns")
    src_budgets = df[budget_col].to_numpy(dtype=np.int64)
    if len(src_budgets) == 0:
        raise ValueError("df is empty; cannot interpolate")

    for col in value_cols:
        if col not in df.columns:
            raise KeyError(f"{col!r} not in DataFrame columns")

    src_values = {col: df[col].to_numpy(dtype=np.float64) for col in value_cols}

    rows = []
    for target in budgets_arr:
        row: dict[str, float | int] = {budget_col: int(target)}
        if method == "step":
            # Largest prefix with budget <= target. df[0] has budget 0,
            # so this is always well-defined for target >= 0.
            idx = int(np.searchsorted(src_budgets, target, side="right") - 1)
            idx = max(idx, 0)
            for col in value_cols:
                row[col] = float(src_values[col][idx])
        else:  # linear
            for col in value_cols:
                row[col] = float(
                    np.interp(target, src_budgets, src_values[col])
                )
        rows.append(row)

    return pd.DataFrame(rows, columns=[budget_col] + list(value_cols))


# ---------------------------------------------------------------------
# Aggregation across seeds
# ---------------------------------------------------------------------

def aggregate_seeds(
    df: pd.DataFrame,
    by: Sequence[str],
    value_cols: Sequence[str],
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """Aggregate values across seeds for each group, with bootstrap CI.

    Parameters
    ----------
    df
        Long-format DataFrame, one row per ``(seed, *by)`` cell. Must
        contain a ``seed`` column so that duplicates within a group are
        interpretable as seeds.
    by
        Grouping columns, e.g. ``["method", "token_budget"]``. The
        ``seed`` column must NOT be in ``by`` -- it is the resampling
        unit.
    value_cols
        Metric columns to aggregate.
    n_resamples
        Bootstrap resamples per group.
    ci
        Confidence level for the interval.
    seed
        RNG seed for the bootstrap resampling.

    Returns
    -------
    pandas.DataFrame
        One row per group. Columns: ``by``, ``n_seeds``, and for each
        ``col`` in ``value_cols``: ``{col}_mean``, ``{col}_std``,
        ``{col}_ci_low``, ``{col}_ci_high``. Std is computed with
        ``ddof=1`` (sample std); for a single seed it is set to 0.
    """
    if "seed" not in df.columns:
        raise KeyError("df must have a 'seed' column")
    if "seed" in by:
        raise ValueError(
            "'seed' must not be in 'by' -- it is the resampling unit"
        )
    if not 0.0 < ci < 1.0:
        raise ValueError(f"ci must be in (0, 1), got {ci}")

    rng = np.random.default_rng(seed)
    alpha = (1.0 - ci) / 2.0
    by = list(by)
    value_cols = list(value_cols)

    rows = []
    for group_key, g in df.groupby(by, sort=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row: dict[str, object] = dict(zip(by, group_key))
        n = len(g)
        row["n_seeds"] = int(n)
        for col in value_cols:
            vals = g[col].to_numpy(dtype=np.float64)
            mean = float(vals.mean())
            std = float(vals.std(ddof=1)) if n > 1 else 0.0
            if n > 1:
                idx = rng.integers(0, n, size=(n_resamples, n))
                boot = vals[idx].mean(axis=1)
                low = float(np.quantile(boot, alpha))
                high = float(np.quantile(boot, 1.0 - alpha))
            else:
                low = high = mean
            row[f"{col}_mean"] = mean
            row[f"{col}_std"] = std
            row[f"{col}_ci_low"] = low
            row[f"{col}_ci_high"] = high
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Paired comparison against a reference method
# ---------------------------------------------------------------------

def paired_compare(
    df: pd.DataFrame,
    reference: str,
    by: Sequence[str],
    value_cols: Sequence[str],
    method_col: str = "method",
    seed_col: str = "seed",
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """Paired delta of each method vs a reference across matching seeds.

    For each ``(method != reference, *by, col)``, computes
    ``delta = value[method] - value[reference]`` on seed-matched rows,
    then reports the mean delta with a bootstrap CI over seeds.

    Pairs are matched on ``seed_col`` **and** on all ``by`` columns. If
    a seed is missing in either side, that seed is excluded for that
    group.

    Returns
    -------
    pandas.DataFrame
        One row per ``(method, *by)``. Columns: ``method``, ``by``,
        ``n_seeds``, and for each ``col``: ``{col}_delta``,
        ``{col}_ci_low``, ``{col}_ci_high``.
    """
    if reference not in df[method_col].unique():
        raise ValueError(
            f"reference {reference!r} not found in {method_col!r} column; "
            f"available: {sorted(df[method_col].unique())}"
        )
    if method_col in by or seed_col in by:
        raise ValueError(
            f"{method_col!r} and {seed_col!r} must not be in 'by'"
        )
    if not 0.0 < ci < 1.0:
        raise ValueError(f"ci must be in (0, 1), got {ci}")

    rng = np.random.default_rng(seed)
    alpha = (1.0 - ci) / 2.0
    by = list(by)
    value_cols = list(value_cols)

    index_cols = [seed_col] + by

    ref_df = df[df[method_col] == reference].set_index(index_cols)
    others = df[df[method_col] != reference]

    rows = []
    for (method, *group_vals), g in others.groupby(
        [method_col] + by, sort=False
    ):
        m_idx = g.set_index(index_cols)
        common = ref_df.index.intersection(m_idx.index)
        if len(common) == 0:
            continue

        row: dict[str, object] = {method_col: method}
        for b, v in zip(by, group_vals):
            row[b] = v
        row["n_seeds"] = int(len(common))

        for col in value_cols:
            m_vals = m_idx.loc[common, col].to_numpy(dtype=np.float64)
            r_vals = ref_df.loc[common, col].to_numpy(dtype=np.float64)
            deltas = m_vals - r_vals
            mean_delta = float(deltas.mean())
            if len(deltas) > 1:
                idx = rng.integers(0, len(deltas), size=(n_resamples, len(deltas)))
                boot = deltas[idx].mean(axis=1)
                low = float(np.quantile(boot, alpha))
                high = float(np.quantile(boot, 1.0 - alpha))
            else:
                low = high = mean_delta
            row[f"{col}_delta"] = mean_delta
            row[f"{col}_ci_low"] = low
            row[f"{col}_ci_high"] = high
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Human-readable pivot
# ---------------------------------------------------------------------

def pivot_summary(
    df: pd.DataFrame,
    values: str,
    index: str = "token_budget",
    columns: str = "method",
) -> pd.DataFrame:
    """Pivot long aggregated stats into a readable wide table.

    Typical use::

        agg = aggregate_seeds(..., value_cols=["vendi_score"])
        wide = pivot_summary(agg, values="vendi_score_mean")
        # rows = budgets, columns = methods, cells = mean vendi

    Parameters
    ----------
    df
        Output of :func:`aggregate_seeds` (or any DataFrame with the
        needed columns).
    values
        Name of the column to pivot (e.g. ``"vendi_score_mean"``).
    index, columns
        Row and column labels for the pivot.

    Returns
    -------
    pandas.DataFrame
        Pivoted table. Missing cells (e.g. a method that never reached
        a budget) are ``NaN`` -- visible, not silently dropped.
    """
    return df.pivot(index=index, columns=columns, values=values)