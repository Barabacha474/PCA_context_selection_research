"""Tests for download.py -- Block 11a.

Network-dependent tests are marked ``slow`` and skipped unless
PCADIV_RUN_SLOW=1. The rest use a locally-produced parquet as the
cache and verify offline behaviour.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from pcadiv.data.download import load_writingprompts

SLOW = os.environ.get("PCADIV_RUN_SLOW", "0") == "1"


@pytest.fixture
def fake_cache(tmp_path) -> Path:
    """A parquet file the loader can read as if it were the real dataset."""
    df = pd.DataFrame({
        "story_id": list(range(20)),
        "prompt": [f"prompt_{i}" for i in range(20)],
        "story": [f"story_{i} " * 10 for i in range(20)],
    })
    p = tmp_path / "writingprompts.parquet"
    df.to_parquet(p, index=False)
    return p


def test_load_from_cache_does_not_download(fake_cache):
    df = load_writingprompts(cache_path=fake_cache)
    assert len(df) == 20
    assert list(df.columns) == ["story_id", "prompt", "story"]


def test_limit_truncates_after_load(fake_cache):
    df = load_writingprompts(cache_path=fake_cache, limit=5)
    assert len(df) == 5
    assert df["story_id"].tolist() == [0, 1, 2, 3, 4]


def test_limit_none_keeps_all(fake_cache):
    df = load_writingprompts(cache_path=fake_cache, limit=None)
    assert len(df) == 20


def test_story_id_is_int64(fake_cache):
    df = load_writingprompts(cache_path=fake_cache)
    assert df["story_id"].dtype == "int64"


def test_no_datasets_raises_import_error(tmp_path, monkeypatch):
    """If 'datasets' is unavailable and cache is missing, ImportError.

    This is the only error path we can test deterministically: network
    failures propagate the underlying exception unchanged, which is
    correct behaviour but not reproducible in a test.
    """
    import sys
    monkeypatch.setitem(sys.modules, "datasets", None)
    missing = tmp_path / "does-not-exist.parquet"
    with pytest.raises(ImportError, match="datasets"):
        load_writingprompts(cache_path=missing, force_download=True)


@pytest.mark.skipif(not SLOW, reason="set PCADIV_RUN_SLOW=1 to run")
def test_real_download(tmp_path):
    p = tmp_path / "wp.parquet"
    df = load_writingprompts(cache_path=p, limit=50, force_download=True)
    assert len(df) == 50
    assert set(df.columns) >= {"story_id", "prompt", "story"}
    assert p.exists()