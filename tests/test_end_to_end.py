"""Walking-skeleton end-to-end tests -- Block 6.

Exercises the full pipeline with offline components only:
HashingEncoder + RegexTokenizer + RandomSelector + MeanPairwiseDistance
+ budget_curve. If these break, the contract between components is
wrong -- fix the contract before touching concrete selectors or
metrics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pcadiv.data.tokenizer import get_tokenizer
from pcadiv.embeddings.encoders import get_encoder
from pcadiv.evaluation import budget_curve, get_metric
from pcadiv.evaluation.metrics import BaseMetric
from pcadiv.selection import get_selector
from pcadiv.selection.selectors import SelectionResult


# ---- fixtures -------------------------------------------------------

@pytest.fixture
def pipeline():
    """Fully offline pipeline fixture."""
    rng = np.random.default_rng(0)
    vocab = [f"w{i}" for i in range(200)]
    texts = [
        " ".join(rng.choice(vocab, size=int(rng.integers(10, 60))).tolist())
        for _ in range(80)
    ]
    enc = get_encoder("hashing", dim=32)
    tok = get_tokenizer("regex")
    emb = enc.encode(texts)
    toks = tok.count_many(texts)
    res = get_selector("random", seed=0).select(emb, toks, budget=200)
    return emb, toks, res


# ---- full pipeline -------------------------------------------------

def test_full_pipeline_produces_dataframe(pipeline):
    emb, _, res = pipeline
    df = budget_curve(res, emb, [get_metric("mean_pairwise")])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(res) + 1
    assert list(df.columns) == ["n_selected", "token_budget", "mean_pairwise"]


def test_n_selected_is_0_to_len_result(pipeline):
    emb, _, res = pipeline
    df = budget_curve(res, emb, [get_metric("mean_pairwise")])
    assert df["n_selected"].tolist() == list(range(len(res) + 1))


def test_token_budget_matches_cum_tokens(pipeline):
    emb, _, res = pipeline
    df = budget_curve(res, emb, [get_metric("mean_pairwise")])
    expected = [0] + res.cum_tokens.tolist()
    assert df["token_budget"].tolist() == expected


def test_no_nan_values(pipeline):
    emb, _, res = pipeline
    df = budget_curve(res, emb, [get_metric("mean_pairwise")])
    assert not df.isna().any().any()


def test_last_row_matches_metric_on_full_selection(pipeline):
    emb, _, res = pipeline
    metric = get_metric("mean_pairwise")
    df = budget_curve(res, emb, [metric])
    expected = metric(emb[res.indices])["mean_pairwise"]
    assert df["mean_pairwise"].iloc[-1] == pytest.approx(expected)


def test_first_row_is_empty_selection(pipeline):
    emb, _, res = pipeline
    df = budget_curve(res, emb, [get_metric("mean_pairwise")])
    assert df["n_selected"].iloc[0] == 0
    assert df["token_budget"].iloc[0] == 0
    assert df["mean_pairwise"].iloc[0] == 0.0


def test_deterministic(pipeline):
    emb, _, res = pipeline
    m = get_metric("mean_pairwise")
    a = budget_curve(res, emb, [m])
    b = budget_curve(res, emb, [m])
    pd.testing.assert_frame_equal(a, b)


# ---- edge cases ----------------------------------------------------

def test_empty_result_yields_one_row():
    emb = np.zeros((5, 4), dtype=np.float32)
    res = SelectionResult(
        indices=np.empty(0, dtype=np.int64),
        cum_tokens=np.empty(0, dtype=np.int64),
    )
    df = budget_curve(res, emb, [get_metric("mean_pairwise")])
    assert len(df) == 1
    assert df["n_selected"].iloc[0] == 0
    assert df["token_budget"].iloc[0] == 0
    assert df["mean_pairwise"].iloc[0] == 0.0


def test_empty_metrics_raises(pipeline):
    emb, _, res = pipeline
    with pytest.raises(ValueError, match="non-empty sequence"):
        budget_curve(res, emb, [])


def test_pool_emb_must_be_2d(pipeline):
    _, _, res = pipeline
    with pytest.raises(ValueError, match="pool_emb must be 2-D"):
        budget_curve(res, np.zeros(10), [get_metric("mean_pairwise")])


def test_out_of_range_index_raises():
    emb = np.zeros((5, 4), dtype=np.float32)
    res = SelectionResult(
        indices=np.array([0, 1, 7], dtype=np.int64),
        cum_tokens=np.array([1, 2, 3], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="pool size is 5"):
        budget_curve(res, emb, [get_metric("mean_pairwise")])


def test_metric_key_collision_raises(pipeline):
    emb, _, res = pipeline

    class CollidingMetric(BaseMetric):
        name = "colliding"
        def __call__(self, emb, *, core_emb=None):
            return {"mean_pairwise": 1.0}

    with pytest.raises(ValueError, match="collides with an earlier metric"):
        budget_curve(
            res, emb,
            [get_metric("mean_pairwise"), CollidingMetric()],
        )


def test_metric_key_shadowing_reserved_column_raises(pipeline):
    emb, _, res = pipeline

    class BadMetric(BaseMetric):
        name = "bad"
        def __call__(self, emb, *, core_emb=None):
            return {"n_selected": 1.0}

    with pytest.raises(ValueError, match="shadows a reserved"):
        budget_curve(res, emb, [BadMetric()])


def test_metric_returning_non_dict_raises(pipeline):
    emb, _, res = pipeline

    class ScalarMetric(BaseMetric):
        name = "scalar"
        def __call__(self, emb, *, core_emb=None):
            return 1.0

    with pytest.raises(TypeError, match="expected dict"):
        budget_curve(res, emb, [ScalarMetric()])


def test_metric_with_inconsistent_keys_raises(pipeline):
    emb, _, res = pipeline

    class InconsistentMetric(BaseMetric):
        name = "inconsistent"
        def __call__(self, emb, *, core_emb=None):
            # Missing key on the empty prefix, present otherwise.
            if emb.shape[0] == 0:
                return {}
            return {"inconsistent_score": 0.0}

    with pytest.raises(ValueError, match="key set mismatch"):
        budget_curve(res, emb, [InconsistentMetric()])


# ---- core_emb forwarding -------------------------------------------

def test_core_emb_forwarded_to_metric(pipeline):
    emb, _, res = pipeline
    seen: list[np.ndarray | None] = []

    class Recorder(BaseMetric):
        name = "recorder"
        def __call__(self, emb, *, core_emb=None):
            seen.append(None if core_emb is None else core_emb.copy())
            return {"recorder_value": 0.0}

    core = np.ones(emb.shape[1], dtype=np.float32)
    budget_curve(res, emb, [Recorder()], core_emb=core)
    assert all(s is not None for s in seen)
    for s in seen:
        np.testing.assert_array_equal(s, core)


def test_core_emb_none_forwarded(pipeline):
    emb, _, res = pipeline
    seen: list[bool] = []

    class Recorder(BaseMetric):
        name = "recorder2"
        def __call__(self, emb, *, core_emb=None):
            seen.append(core_emb is None)
            return {"recorder2_value": 0.0}

    budget_curve(res, emb, [Recorder()])
    assert all(seen)