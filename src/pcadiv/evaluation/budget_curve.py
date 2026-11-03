"""Diversity-vs-budget curves.

Given a :class:`~pcadiv.selection.SelectionResult` and the pool
embeddings it was selected from, evaluate one or more metrics on every
**prefix** of the selection and return a tidy :class:`pandas.DataFrame`.

Why prefixes
------------
Selectors return an ordered list (see ``selection/selectors.py``).
Diversity-vs-budget curves are exactly the sequence "metric on the first
``i`` selected passages, at token cost ``cum_tokens[i-1]``". This module
turns that definition into code so that no selector has to know about
metrics and no metric has to know about selection.

Included rows
-------------
Rows go from ``i = 0`` (empty selection, zero tokens) up to
``i = len(result)`` (full selection). Including the empty prefix makes
the curve start at ``(0, 0)`` and forces metrics to handle ``N = 0`` --
which is why ``BaseMetric`` documents that requirement.

Metric key uniqueness
---------------------
The keys returned by metrics are used verbatim as DataFrame columns.
They must be **globally unique across all metrics in the project**
(``mean_pairwise``, ``coverage_radius``, ``coverage_logdet``,
``vendi_score``, ...). :func:`budget_curve` validates collisions and
raises, because two metrics writing the same column is a silent data
corruption bug -- one of them wins and the other is invisible in the
parquet file.

Metric key completeness
-----------------------
A metric must return the **same set of keys for every N**, including
``N = 0`` and ``N = 1``. :func:`budget_curve` validates that all rows
have the full key set and raises otherwise. This keeps the resulting
DataFrame rectangular and free of NaN -- important because
``.groupby(...).mean()`` and parquet readers both silently propagate
NaN.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from pcadiv.evaluation.metrics import BaseMetric
from pcadiv.selection.selectors import SelectionResult

__all__ = ["budget_curve"]


def budget_curve(
    result: SelectionResult,
    pool_emb: np.ndarray,
    metrics: Sequence[BaseMetric],
    core_emb: np.ndarray | None = None,
) -> pd.DataFrame:
    """Evaluate ``metrics`` on every prefix of ``result``.

    Parameters
    ----------
    result
        Ordered selection produced by a selector. ``result.indices``
        must be valid indices into ``pool_emb``.
    pool_emb
        ``float32`` array of shape ``(N, D)`` from which the selection
        was made.
    metrics
        Non-empty sequence of :class:`~pcadiv.evaluation.metrics.BaseMetric`
        instances. All metrics receive the same prefix and the same
        ``core_emb`` on every row.
    core_emb
        Optional anchor vector, forwarded unchanged to every metric.
        ``None`` is passed through as ``None``.

    Returns
    -------
    pandas.DataFrame
        One row per prefix (``len(result) + 1`` rows). Columns:
        ``n_selected`` (int), ``token_budget`` (int), then one float
        column per metric key, in first-seen order.

    Raises
    ------
    ValueError
        If ``metrics`` is empty; if ``pool_emb`` is not 2-D; if
        ``result.indices`` contains out-of-range indices; if two
        metrics return the same key; if a metric returns an incomplete
        key set on some prefix.
    """
    if len(metrics) == 0:
        raise ValueError("metrics must be a non-empty sequence")

    emb = np.asarray(pool_emb)
    if emb.ndim != 2:
        raise ValueError(f"pool_emb must be 2-D, got ndim={emb.ndim}")
    n_pool = emb.shape[0]
    if result.indices.size and int(result.indices.max()) >= n_pool:
        raise ValueError(
            f"result contains index {int(result.indices.max())} "
            f"but pool size is {n_pool}"
        )

    rows: list[dict[str, float | int]] = []
    column_order: list[str] = []

    for i in range(len(result) + 1):
        if i == 0:
            sel_emb = emb[:0]
            budget = 0
        else:
            sel_emb = emb[result.indices[:i]]
            budget = int(result.cum_tokens[i - 1])

        row: dict[str, float | int] = {"n_selected": i, "token_budget": budget}

        for m in metrics:
            out = m(sel_emb, core_emb=core_emb)
            if not isinstance(out, dict):
                raise TypeError(
                    f"{m.name} returned {type(out).__name__}, expected dict"
                )
            for key, val in out.items():
                if key in row and key not in ("n_selected", "token_budget"):
                    raise ValueError(
                        f"metric key {key!r} (from {m.name}) collides with "
                        f"an earlier metric; metric keys must be globally "
                        f"unique across the project"
                    )
                if key in ("n_selected", "token_budget"):
                    raise ValueError(
                        f"metric key {key!r} (from {m.name}) shadows a "
                        f"reserved budget_curve column"
                    )
                row[key] = float(val)
                if key not in column_order:
                    column_order.append(key)

        rows.append(row)

    # Every row must carry the full key set -- otherwise pandas will
    # silently insert NaN, which downstream aggregation hides.
    expected_keys = set(column_order)
    for r_idx, row in enumerate(rows):
        actual = set(row) - {"n_selected", "token_budget"}
        if actual != expected_keys:
            missing = expected_keys - actual
            extra = actual - expected_keys
            raise ValueError(
                f"metric key set mismatch at prefix {r_idx}: "
                f"missing={sorted(missing)}, extra={sorted(extra)}; "
                f"metrics must return the same keys for every N"
            )

    columns = ["n_selected", "token_budget"] + column_order
    return pd.DataFrame(rows, columns=columns)