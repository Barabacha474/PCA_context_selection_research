"""Smoke tests for run_sweep -- Block 12b."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pcadiv.embeddings.encoders import get_encoder
from pcadiv.sweep import run_sweep


@pytest.fixture
def tiny_pool(tmp_path) -> Path:
    """Small deterministic pool with mixed token costs."""
    rng = np.random.default_rng(0)
    n = 40
    rows = []
    for i in range(n):
        rows.append({
            "passage_id": f"{i // 4}::{i % 4}",
            "story_id": i // 4,
            "prompt_id": f"p{i // 4}",
            "text": f"passage number {i} with some words",
            "n_tokens": int(rng.integers(20, 60)),
            "position": i % 4,
        })
    df = pd.DataFrame(rows)
    p = tmp_path / "pool.parquet"
    df.to_parquet(p, index=False)
    return p


@pytest.fixture
def two_encoders():
    """Two distinct HashingEncoders (different dims -> different fingerprints)."""
    sel = get_encoder("hashing", dim=32, model_id="sel")
    ev = get_encoder("hashing", dim=64, model_id="eval")
    return sel, ev


@pytest.fixture
def sweep_cfg(tmp_path, tiny_pool):
    return {
        "data": {
            "pool_path": str(tiny_pool),
            "cache_dir": None,
        },
        "encoders": {
            "selection": "unused",
            "eval": "unused",
        },
        "selectors": [
            {"name": "random", "params": {}},
            {"name": "fps", "params": {"metric": "euclidean"}},
        ],
        "budgets": [100, 200],
        "seeds": [0, 1],
        "metrics": [
            {"name": "mean_pairwise", "params": {}},
            {"name": "coverage", "params": {}},
            {"name": "vendi", "params": {"gamma": 1.0}},
        ],
        "reference_method": "random",
        "output_dir": str(tmp_path / "results"),
        "sweep_name": "test",
    }


# ---- success path --------------------------------------------------

def test_sweep_produces_all_outputs(sweep_cfg, two_encoders, tmp_path):
    sel, ev = two_encoders
    out = run_sweep(
        sweep_cfg,
        sel_encoder_override=sel,
        eval_encoder_override=ev,
    )
    for key in ("results_path", "agg_path", "meta_path"):
        assert Path(out[key]).exists()
    assert out["compare_path"] is not None
    assert Path(out["compare_path"]).exists()


def test_results_shape(sweep_cfg, two_encoders):
    sel, ev = two_encoders
    out = run_sweep(sweep_cfg, sel_encoder_override=sel, eval_encoder_override=ev)
    results = pd.read_parquet(out["results_path"])
    # 2 methods × 2 seeds × 2 budgets = 8 rows
    assert len(results) == 8
    expected_cols = {
        "method", "seed", "target_budget", "token_budget_actual", "n_selected",
        "mean_pairwise", "coverage_radius", "coverage_logdet", "vendi_score",
    }
    assert set(results.columns) == expected_cols


def test_all_methods_covered(sweep_cfg, two_encoders):
    sel, ev = two_encoders
    out = run_sweep(sweep_cfg, sel_encoder_override=sel, eval_encoder_override=ev)
    results = pd.read_parquet(out["results_path"])
    assert set(results["method"].unique()) == {"random", "fps"}


def test_all_budgets_and_seeds_covered(sweep_cfg, two_encoders):
    sel, ev = two_encoders
    out = run_sweep(sweep_cfg, sel_encoder_override=sel, eval_encoder_override=ev)
    results = pd.read_parquet(out["results_path"])
    assert sorted(results["target_budget"].unique()) == [100, 200]
    assert sorted(results["seed"].unique()) == [0, 1]


def test_agg_has_expected_columns(sweep_cfg, two_encoders):
    sel, ev = two_encoders
    out = run_sweep(sweep_cfg, sel_encoder_override=sel, eval_encoder_override=ev)
    agg = pd.read_parquet(out["agg_path"])
    for col in ("mean_pairwise_mean", "vendi_score_mean",
                "coverage_radius_std", "n_selected_mean", "n_seeds"):
        assert col in agg.columns


def test_compare_against_random(sweep_cfg, two_encoders):
    sel, ev = two_encoders
    out = run_sweep(sweep_cfg, sel_encoder_override=sel, eval_encoder_override=ev)
    cmp_df = pd.read_parquet(out["compare_path"])
    # One non-reference method ('fps') × 2 budgets = 2 rows
    assert set(cmp_df["method"].unique()) == {"fps"}
    assert sorted(cmp_df["target_budget"].unique()) == [100, 200]
    assert "vendi_score_delta" in cmp_df.columns


def test_meta_json_contents(sweep_cfg, two_encoders):
    sel, ev = two_encoders
    out = run_sweep(sweep_cfg, sel_encoder_override=sel, eval_encoder_override=ev)
    meta = json.loads(Path(out["meta_path"]).read_text())
    assert meta["sweep_name"] == "test"
    assert meta["n_rows"] == 8
    assert meta["sel_encoder"]["fingerprint"] == sel.fingerprint()
    assert meta["eval_encoder"]["fingerprint"] == ev.fingerprint()
    assert meta["reference_method"] == "random"


# ---- safety --------------------------------------------------------

def test_same_encoder_for_selection_and_eval_raises(sweep_cfg):
    shared = get_encoder("hashing", dim=32, model_id="shared")
    with pytest.raises(ValueError, match="same fingerprint"):
        run_sweep(
            sweep_cfg,
            sel_encoder_override=shared,
            eval_encoder_override=shared,
        )


def test_missing_pool_raises(sweep_cfg, two_encoders, tmp_path):
    sel, ev = two_encoders
    with pytest.raises(FileNotFoundError, match="pool file not found"):
        run_sweep(
            sweep_cfg,
            pool_override=tmp_path / "does-not-exist.parquet",
            sel_encoder_override=sel,
            eval_encoder_override=ev,
        )


def test_empty_pool_raises(sweep_cfg, two_encoders, tmp_path):
    sel, ev = two_encoders
    empty = tmp_path / "empty.parquet"
    pd.DataFrame({
        "passage_id": [], "story_id": [], "prompt_id": [],
        "text": [], "n_tokens": [], "position": [],
    }).astype({"n_tokens": "int64", "position": "int64"}).to_parquet(empty, index=False)
    with pytest.raises(ValueError, match="pool is empty"):
        run_sweep(
            sweep_cfg,
            pool_override=empty,
            sel_encoder_override=sel,
            eval_encoder_override=ev,
        )


def test_empty_budgets_raises(sweep_cfg, two_encoders):
    sel, ev = two_encoders
    sweep_cfg["budgets"] = []
    with pytest.raises(ValueError, match="budgets.*empty"):
        run_sweep(sweep_cfg, sel_encoder_override=sel, eval_encoder_override=ev)


def test_empty_seeds_raises(sweep_cfg, two_encoders):
    sel, ev = two_encoders
    sweep_cfg["seeds"] = []
    with pytest.raises(ValueError, match="seeds.*empty"):
        run_sweep(sweep_cfg, sel_encoder_override=sel, eval_encoder_override=ev)


def test_duplicate_metric_keys_raises(sweep_cfg, two_encoders):
    sel, ev = two_encoders
    sweep_cfg["metrics"] = [
        {"name": "mean_pairwise", "params": {}},
        {"name": "mean_pairwise", "params": {"metric": "cosine"}},
    ]
    with pytest.raises(ValueError, match="more than one metric"):
        run_sweep(sweep_cfg, sel_encoder_override=sel, eval_encoder_override=ev)


# ---- determinism ---------------------------------------------------

def test_deterministic_given_seeds(sweep_cfg, two_encoders):
    sel, ev = two_encoders
    out1 = run_sweep(
        sweep_cfg,
        out_dir=sweep_cfg["output_dir"], sweep_name="a",
        sel_encoder_override=sel, eval_encoder_override=ev,
    )
    out2 = run_sweep(
        sweep_cfg,
        out_dir=sweep_cfg["output_dir"], sweep_name="a",
        sel_encoder_override=sel, eval_encoder_override=ev,
    )
    r1 = pd.read_parquet(out1["results_path"])
    r2 = pd.read_parquet(out2["results_path"])
    pd.testing.assert_frame_equal(r1, r2)


# ---- no reference -----------------------------------------------

def test_no_reference_no_compare_file(sweep_cfg, two_encoders):
    sel, ev = two_encoders
    sweep_cfg["reference_method"] = None
    out = run_sweep(sweep_cfg, sel_encoder_override=sel, eval_encoder_override=ev)
    assert out["compare_path"] is None


def test_reference_not_in_methods_no_compare_file(sweep_cfg, two_encoders):
    sel, ev = two_encoders
    sweep_cfg["reference_method"] = "iterative_pca"  # not in selectors
    out = run_sweep(sweep_cfg, sel_encoder_override=sel, eval_encoder_override=ev)
    assert out["compare_path"] is None


# ---- metric values are finite ---------------------------------------

def test_all_metric_values_finite(sweep_cfg, two_encoders):
    sel, ev = two_encoders
    out = run_sweep(sweep_cfg, sel_encoder_override=sel, eval_encoder_override=ev)
    results = pd.read_parquet(out["results_path"])
    for col in ("mean_pairwise", "coverage_radius", "coverage_logdet", "vendi_score"):
        assert np.isfinite(results[col]).all(), f"{col} has non-finite values"


def test_vendi_between_one_and_n(sweep_cfg, two_encoders):
    """Vendi for N selected points must be in [1, N] (up to fp)."""
    sel, ev = two_encoders
    out = run_sweep(sweep_cfg, sel_encoder_override=sel, eval_encoder_override=ev)
    results = pd.read_parquet(out["results_path"])
    for _, row in results.iterrows():
        if row["n_selected"] >= 1:
            assert row["vendi_score"] >= 1.0 - 1e-6
            assert row["vendi_score"] <= row["n_selected"] + 1e-6

def test_selector_seed_in_params_is_stripped(sweep_cfg, two_encoders, caplog):
    """If a selector's params contain 'seed', sweep warns and strips it,
    using cfg['seeds'] instead. Regression guard for double-seed bug."""
    sel, ev = two_encoders
    sweep_cfg["selectors"] = [
        {"name": "random", "params": {"seed": 999}},
    ]
    sweep_cfg["seeds"] = [0, 1]
    with caplog.at_level("WARNING"):
        out = run_sweep(
            sweep_cfg,
            sel_encoder_override=sel,
            eval_encoder_override=ev,
        )
    assert any("seed" in r.message.lower() for r in caplog.records)
    results = pd.read_parquet(out["results_path"])
    # Still produces rows for both sweep seeds (0 and 1)
    assert sorted(results["seed"].unique()) == [0, 1]