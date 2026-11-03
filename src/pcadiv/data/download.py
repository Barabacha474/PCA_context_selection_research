"""Download the WritingPrompts dataset from Hugging Face.

The dataset (``euclaise/writingprompts``) is ~600 MB and is fetched
through :mod:`datasets`. To avoid re-downloading on every run and to
give the rest of the pipeline a stable artifact, we materialise it as a
single parquet file under ``data/raw/``. Subsequent runs load from that
file unless ``force_download=True``.

Offline behaviour
-----------------
If the cache file exists and ``force_download=False``, no network call
is made. If the file is missing and the ``datasets`` package is not
installed, a clear ``ImportError`` is raised pointing at the
``[data]`` extra. If ``datasets`` is installed but the download fails
(no network, HF outage), the underlying exception propagates unchanged
-- we do not swallow it, because "fallback to empty DataFrame" is a
worse failure mode than a loud error.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

__all__ = ["load_writingprompts", "DEFAULT_CACHE"]


DEFAULT_CACHE = Path("data/raw/writingprompts.parquet")


def load_writingprompts(
    cache_path: str | Path = DEFAULT_CACHE,
    split: str = "train",
    limit: int | None = None,
    force_download: bool = False,
    hf_dataset: str = "euclaise/writingprompts",
) -> pd.DataFrame:
    """Load WritingPrompts, downloading on first use.

    Parameters
    ----------
    cache_path
        Where to store/load the materialised parquet. Parent
        directories are created as needed.
    split
        HF split name (``"train"`` or ``"validation"``).
    limit
        If given, truncate to the first ``limit`` stories after load.
        Applied *after* reading from cache, so the parquet on disk is
        always the full split.
    force_download
        Re-download and overwrite the cache even if it exists.
    hf_dataset
        HF dataset id. Override for forks/mirrors.

    Returns
    -------
    pandas.DataFrame
        Columns: ``story_id`` (int64), ``prompt`` (str), ``story`` (str).
    """
    path = Path(cache_path)
    if path.exists() and not force_download:
        df = pd.read_parquet(path)
        if limit is not None:
            df = df.head(limit).copy()
        return df

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "load_writingprompts requires the 'datasets' package. "
            "Install with: pip install -e \".[data]\""
        ) from e

    ds = load_dataset(hf_dataset, split=split)
    df = ds.to_pandas()[["prompt", "story"]].copy()
    df.insert(0, "story_id", range(len(df)))
    df["story_id"] = df["story_id"].astype("int64")

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

    if limit is not None:
        df = df.head(limit).copy()
    return df