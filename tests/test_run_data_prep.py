"""Smoke test for the full data pipeline -- Block 12a."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pcadiv.data.pipeline import run_data_prep
from pcadiv.utils.config import load_config


@pytest.fixture
def tiny_stories() -> pd.DataFrame:
    """30 short stories, deterministic."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(30):
        n_words = int(rng.integers(40, 100))
        words = [f"w{int(rng.integers(0, 200))}" for _ in range(n_words)]
        rows.append({
            "story_id": i,
            "prompt": f"prompt_{i}",
            "story": " ".join(words),
        })
    return pd.DataFrame(rows)


@pytest.fixture
def tiny_cfg(tmp_path, tiny_stories):
    """Config wired to use ``RegexTokenizer`` and tmp paths."""
    return {
        "source": {
            "hf_dataset": "unused",
            "cache_path": str(tmp_path / "raw.parquet"),
            "split": "train",
            "limit": None,
            "force_download": False,
        },
        "tokenizer": {"name": "regex", "params": {}},
        "passageize": {
            "passage_tokens": 20,
            "min_passage_tokens": 10,
            "out_path": str(tmp_path / "passages.parquet"),
        },
        "split": {"train_frac": 0.7, "seed": 0},
        "pools": {
            "primary": {
                "n_passages": 40,
                "seed": 0,
                "min_tokens": 10,
                "max_tokens": 20,
                "allow_short": True,
                "out_path": str(tmp_path / "primary.parquet"),
            },
            "ood": {
                "n_passages": 15,
                "seed": 0,
                "min_tokens": 10,
                "max_tokens": 20,
                "allow_short": True,
                "out_path": str(tmp_path / "ood.parquet"),
            },
        },
    }


def test_pipeline_produces_all_artifacts(tiny_cfg, tiny_stories, tmp_path):
    out = run_data_prep(
        tiny_cfg,
        load_fn=lambda **kw: tiny_stories,
    )
    assert Path(out["passages_path"]).exists()
    assert Path(out["primary_pool_path"]).exists()
    assert Path(out["ood_pool_path"]).exists()
    assert out["disjoint"] is True
    assert out["n_passages"] > 0
    assert out["n_primary"] > 0
    assert out["n_ood"] > 0


def test_pipeline_meta_files_written(tiny_cfg, tiny_stories, tmp_path):
    out = run_data_prep(tiny_cfg, load_fn=lambda **kw: tiny_stories)
    for pool_name in ("primary", "ood"):
        meta_path = Path(out[f"{pool_name}_pool_path"]).with_suffix(".meta.json")
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["split"] == pool_name
        assert "story_ids_sha1" in meta


def test_pipeline_idempotent_reuses_passages(tiny_cfg, tiny_stories, tmp_path):
    out1 = run_data_prep(tiny_cfg, load_fn=lambda **kw: tiny_stories)
    passages_mtime = Path(out1["passages_path"]).stat().st_mtime_ns

    # Second run: no force, should reuse existing passages.
    out2 = run_data_prep(tiny_cfg, load_fn=lambda **kw: tiny_stories)
    assert Path(out2["passages_path"]).stat().st_mtime_ns == passages_mtime
    assert out1["n_passages"] == out2["n_passages"]


def test_pipeline_disjoint_pools(tiny_cfg, tiny_stories, tmp_path):
    out = run_data_prep(tiny_cfg, load_fn=lambda **kw: tiny_stories)
    primary = pd.read_parquet(out["primary_pool_path"])
    ood = pd.read_parquet(out["ood_pool_path"])
    assert set(primary["story_id"]).isdisjoint(set(ood["story_id"]))


def test_pipeline_no_stories_raises(tiny_cfg):
    empty = pd.DataFrame(columns=["story_id", "prompt", "story"])
    with pytest.raises(RuntimeError, match="no stories loaded"):
        run_data_prep(tiny_cfg, load_fn=lambda **kw: empty)


def test_pipeline_missing_passage_tokens_raises(tiny_cfg, tiny_stories):
    del tiny_cfg["passageize"]["passage_tokens"]
    with pytest.raises(KeyError, match="passage_tokens"):
        run_data_prep(tiny_cfg, load_fn=lambda **kw: tiny_stories)


def test_pipeline_skip_download_requires_cache(tiny_cfg, tmp_path):
    # cache_path points to a non-existent file
    with pytest.raises(FileNotFoundError, match="skip_download"):
        run_data_prep(tiny_cfg, skip_download=True)


def test_pipeline_force_passageize_reruns(tiny_cfg, tiny_stories, tmp_path):
    out1 = run_data_prep(tiny_cfg, load_fn=lambda **kw: tiny_stories)
    mtime1 = Path(out1["passages_path"]).stat().st_mtime_ns

    # force should produce a new file
    import time
    time.sleep(0.01)
    out2 = run_data_prep(
        tiny_cfg, load_fn=lambda **kw: tiny_stories, force_passageize=True,
    )
    mtime2 = Path(out2["passages_path"]).stat().st_mtime_ns
    assert mtime2 > mtime1