"""Tests for pools.py -- Block 11c."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pcadiv.data.pools import (
    build_pool,
    compute_pool_centroid,
    encode_pool,
    load_pool,
    save_pool,
)


@pytest.fixture
def passages():
    """Passage table with 3 stories × 4 passages = 12 rows."""
    rows = []
    for sid in range(3):
        for pos in range(4):
            rows.append({
                "passage_id": f"{sid}::{pos}",
                "story_id": sid,
                "prompt_id": f"p{sid}",
                "text": f"story{sid} part{pos}",
                "n_tokens": 10 * (pos + 1),  # 10, 20, 30, 40
                "position": pos,
            })
    return pd.DataFrame(rows)


# ---- build_pool ----------------------------------------------------

def test_basic_shape_and_columns(passages):
    pool = build_pool(passages, n_passages=8, seed=0)
    assert len(pool) == 8
    assert list(pool.columns) == [
        "passage_id", "story_id", "prompt_id", "text", "n_tokens", "position",
    ]
    assert pool["passage_id"].is_unique


def test_deterministic_by_seed(passages):
    a = build_pool(passages, n_passages=8, seed=0)
    b = build_pool(passages, n_passages=8, seed=0)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_different_selection(passages):
    a = build_pool(passages, n_passages=8, seed=0)
    b = build_pool(passages, n_passages=8, seed=1)
    assert a["passage_id"].tolist() != b["passage_id"].tolist()


def test_n_passages_zero_rejected(passages):
    with pytest.raises(ValueError, match="n_passages must be >= 1"):
        build_pool(passages, n_passages=0)


def test_story_filter_restricts_eligibility(passages):
    pool = build_pool(passages, n_passages=4, seed=0, story_ids={0, 1})
    assert set(pool["story_id"].unique()) <= {0, 1}


def test_two_pools_disjoint_by_story_ids(passages):
    train_ids = {0, 1}
    ood_ids = {2}
    p_train = build_pool(passages, n_passages=4, seed=0, story_ids=train_ids)
    p_ood = build_pool(passages, n_passages=4, seed=0, story_ids=ood_ids)
    assert set(p_train["story_id"]) & set(p_ood["story_id"]) == set()


def test_min_tokens_filter(passages):
    pool = build_pool(passages, n_passages=6, seed=0, min_tokens=30)
    assert (pool["n_tokens"] >= 30).all()


def test_max_tokens_filter(passages):
    pool = build_pool(passages, n_passages=6, seed=0, max_tokens=20)
    assert (pool["n_tokens"] <= 20).all()


def test_min_gt_max_rejected(passages):
    with pytest.raises(ValueError, match="min_tokens .* <= max_tokens"):
        build_pool(passages, n_passages=1, min_tokens=30, max_tokens=10)


def test_not_enough_passages_raises(passages):
    with pytest.raises(ValueError, match="only 12 available"):
        build_pool(passages, n_passages=100)


def test_allow_short_warns_and_truncates(passages):
    with pytest.warns(RuntimeWarning, match="truncated"):
        pool = build_pool(passages, n_passages=100, allow_short=True)
    assert len(pool) == 12


def test_empty_after_filters_raises(passages):
    with pytest.raises(ValueError, match="no passages survived"):
        build_pool(passages, n_passages=1, min_tokens=10_000)


def test_missing_required_col():
    bad = pd.DataFrame({"passage_id": ["a"], "text": ["x"]})
    with pytest.raises(KeyError, match="missing"):
        build_pool(bad, n_passages=1)


def test_dtypes(passages):
    pool = build_pool(passages, n_passages=4, seed=0)
    assert pool["n_tokens"].dtype == "int64"
    assert pool["position"].dtype == "int64"


# ---- compute_pool_centroid -----------------------------------------

def test_centroid_shape_and_dtype():
    X = np.random.default_rng(0).standard_normal((10, 8)).astype(np.float32)
    c = compute_pool_centroid(X)
    assert c.shape == (8,)
    assert c.dtype == np.float32


def test_centroid_normalized_unit_norm():
    X = np.random.default_rng(0).standard_normal((10, 8)).astype(np.float32)
    c = compute_pool_centroid(X, normalize=True)
    assert np.linalg.norm(c) == pytest.approx(1.0, rel=1e-5)


def test_centroid_unnormalized_not_unit_norm():
    X = np.random.default_rng(0).standard_normal((10, 8)).astype(np.float32) * 10
    c = compute_pool_centroid(X, normalize=False)
    assert np.linalg.norm(c) != pytest.approx(1.0, rel=1e-3)


def test_centroid_empty_pool_returns_zeros():
    c = compute_pool_centroid(np.zeros((0, 5), dtype=np.float32))
    np.testing.assert_array_equal(c, np.zeros(5, dtype=np.float32))


def test_centroid_collinear_pool_normalized_well():
    """All points identical and non-zero -> normalized centroid is that point."""
    X = np.tile([3.0, 0.0, 4.0], (5, 1)).astype(np.float32)
    c = compute_pool_centroid(X, normalize=True)
    np.testing.assert_allclose(c, [0.6, 0.0, 0.8], atol=1e-6)


def test_centroid_zero_pool_no_nan():
    X = np.zeros((5, 3), dtype=np.float32)
    c = compute_pool_centroid(X, normalize=True)
    assert np.isfinite(c).all()


# ---- encode_pool ---------------------------------------------------

def test_encode_pool_uses_cache(tmp_path):
    from pcadiv.embeddings.encoders import get_encoder

    enc = get_encoder("hashing", dim=16)
    pool = pd.DataFrame({"text": ["a b c", "d e f"]})
    emb1 = encode_pool(pool, enc, cache_dir=tmp_path)
    emb2 = encode_pool(pool, enc, cache_dir=tmp_path)
    np.testing.assert_array_equal(emb1, emb2)
    assert emb1.shape == (2, 16)


def test_encode_pool_missing_text_raises():
    from pcadiv.embeddings.encoders import get_encoder
    enc = get_encoder("hashing", dim=8)
    pool = pd.DataFrame({"not_text": ["a"]})
    with pytest.raises(KeyError, match="text"):
        encode_pool(pool, enc, cache_dir=None)


# ---- save_pool / load_pool -----------------------------------------

def test_save_and_load_roundtrip(tmp_path, passages):
    pool = build_pool(passages, n_passages=4, seed=0)
    p = tmp_path / "pool.parquet"
    save_pool(pool, p, seed=0)
    loaded, meta = load_pool(p)
    pd.testing.assert_frame_equal(loaded, pool)
    assert meta["n_passages"] == 4
    assert meta["seed"] == 0
    assert "story_ids_sha1" in meta


def test_meta_json_written(tmp_path, passages):
    pool = build_pool(passages, n_passages=4, seed=0)
    p = tmp_path / "pool.parquet"
    save_pool(pool, p, seed=0, extra_meta={"encoder": "hashing_16"})
    meta_path = p.with_suffix(".meta.json")
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["encoder"] == "hashing_16"


def test_story_ids_sha1_stable_across_row_order(tmp_path, passages):
    """story_ids_sha1 depends only on the set of story ids, not their order."""
    pool = build_pool(passages, n_passages=4, seed=0)
    save_pool(pool, tmp_path / "a.parquet", seed=0)
    shuffled = pool.sample(frac=1, random_state=1).reset_index(drop=True)
    save_pool(shuffled, tmp_path / "b.parquet", seed=0)

    meta_a = json.loads((tmp_path / "a.meta.json").read_text())
    meta_b = json.loads((tmp_path / "b.meta.json").read_text())
    assert meta_a["story_ids_sha1"] == meta_b["story_ids_sha1"]


def test_load_missing_pool_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_pool(tmp_path / "nope.parquet")


def test_load_pool_without_meta(tmp_path, passages):
    pool = build_pool(passages, n_passages=3, seed=0)
    p = tmp_path / "pool.parquet"
    pool.to_parquet(p, index=False)  # bypass save_pool -> no meta
    loaded, meta = load_pool(p)
    pd.testing.assert_frame_equal(loaded, pool)
    assert meta == {}