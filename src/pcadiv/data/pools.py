"""Assemble selection pools from passages, with reproducible sampling.

A *pool* is a fixed, reproducible subset of passages -- the set from
which a selector chooses under a token budget. To keep experiments
comparable, pools are:

* **Sampled deterministically** from a parent passage table, via a
  seeded random permutation (not ``DataFrame.sample``, which depends on
  row order).
* **Disjoint across splits** -- when ``story_ids`` is given, only
  passages from those stories are eligible. ``splits.py`` produces
  disjoint ``train_ids`` and ``ood_ids``; ``build_pool`` respects them,
  so the primary and OOD pools never share a story.
* **Self-contained** -- the returned DataFrame has everything a sweep
  needs: ``passage_id``, ``text``, ``n_tokens``, plus ``story_id``,
  ``prompt_id``, ``position`` for tracing back.
* **Traceable** -- ``save_pool`` writes a ``.meta.json`` next to the
  parquet with the seed, the source passage-file fingerprint, and the
  story-id set digest, so that two runs on different pools are
  auditable.

Centroid of the pool
--------------------
``compute_pool_centroid`` returns the (optionally L2-normalised) mean
embedding. This is the "centrality anchor" for the relevance baseline
(Block 8a) in the primary sweep. It is computed on demand and not
cached: it is a single ``.mean(0)`` on an array that is already cached
by the embedding cache (Block 10). Having exactly one function for it
guarantees every sweep uses the same formula -- the alternative
(letting each sweep normalise or not) would silently change the
centrality baseline between experiments.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from pcadiv.embeddings.cache import load_or_encode
from pcadiv.embeddings.encoders import BaseEncoder
from pcadiv.utils.seeds import make_rng

__all__ = [
    "build_pool",
    "compute_pool_centroid",
    "encode_pool",
    "save_pool",
    "load_pool",
]


def build_pool(
    passages: pd.DataFrame,
    n_passages: int,
    seed: int = 0,
    story_ids: set[int] | None = None,
    min_tokens: int | None = None,
    max_tokens: int | None = None,
    allow_short: bool = False,
) -> pd.DataFrame:
    """Sample a fixed pool of passages, deterministically.

    Parameters
    ----------
    passages
        Output of :func:`~pcadiv.data.passageize.passageize`. Must have
        ``passage_id``, ``story_id``, ``text``, ``n_tokens``.
    n_passages
        Target pool size. If fewer passages survive the filters:
        raise if ``allow_short=False``, truncate with a warning if
        ``True``.
    seed
        Sampling seed. A pool is the same set of passages for the same
        ``(passages, filters, seed, n_passages)``.
    story_ids
        If given, only passages whose ``story_id`` is in this set are
        eligible. Used to enforce disjointness between primary and OOD
        pools.
    min_tokens, max_tokens
        Optional inclusive filter on ``n_tokens``. Useful to exclude
        pathological single-word passages (``max_tokens``) or very short
        tails (``min_tokens``).
    allow_short
        If ``True``, return fewer than ``n_passages`` with a
        ``RuntimeWarning`` instead of raising.

    Returns
    -------
    pandas.DataFrame
        With columns ``passage_id, story_id, prompt_id, text, n_tokens,
        position``, in the sampled order (deterministic given the
        seed).
    """
    if n_passages < 1:
        raise ValueError(f"n_passages must be >= 1, got {n_passages}")
    if min_tokens is not None and max_tokens is not None and min_tokens > max_tokens:
        raise ValueError(
            f"min_tokens ({min_tokens}) must be <= max_tokens ({max_tokens})"
        )
    required = {"passage_id", "story_id", "text", "n_tokens"}
    missing = required - set(passages.columns)
    if missing:
        raise KeyError(f"passages is missing columns: {sorted(missing)}")

    # 1. Filter.
    eligible = passages
    if story_ids is not None:
        eligible = eligible[eligible["story_id"].isin(story_ids)]
    if min_tokens is not None:
        eligible = eligible[eligible["n_tokens"] >= min_tokens]
    if max_tokens is not None:
        eligible = eligible[eligible["n_tokens"] <= max_tokens]

    n_avail = len(eligible)
    if n_avail == 0:
        raise ValueError(
            "no passages survived the filters; pool cannot be built"
        )
    if n_avail < n_passages and not allow_short:
        raise ValueError(
            f"requested {n_passages} passages but only {n_avail} "
            f"available after filters; pass allow_short=True to truncate"
        )
    if n_avail < n_passages and allow_short:
        import warnings
        warnings.warn(
            f"only {n_avail} passages available, pool truncated from "
            f"requested {n_passages}",
            RuntimeWarning,
            stacklevel=2,
        )

    # 2. Deterministic permutation of eligible rows. We permute row
    # positions (not values) so the result does not depend on
    # DataFrame's internal ordering for a given seed and input.
    rng = make_rng(seed)
    perm = rng.permutation(n_avail)
    take = perm[:n_passages]

    out = eligible.iloc[take].reset_index(drop=True)

    # 3. Column order and dtypes.
    cols = ["passage_id", "story_id", "prompt_id", "text", "n_tokens", "position"]
    for c in cols:
        if c not in out.columns:
            out[c] = None
    out = out[cols].copy()
    out["n_tokens"] = out["n_tokens"].astype("int64")
    out["position"] = out["position"].astype("int64")
    return out


def encode_pool(
    pool: pd.DataFrame,
    encoder: BaseEncoder,
    cache_dir: str | Path | None = None,
    *,
    force_recompute: bool = False,
) -> np.ndarray:
    """Encode all passages in ``pool`` (via the embedding cache)."""
    if "text" not in pool.columns:
        raise KeyError("pool is missing 'text' column")
    return load_or_encode(
        encoder,
        pool["text"].tolist(),
        cache_dir=cache_dir,
        force_recompute=force_recompute,
    )


def compute_pool_centroid(
    pool_emb: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """Return the pool centroid as a 1-D ``float32`` vector.

    Parameters
    ----------
    pool_emb
        ``(N, D)`` embeddings of the pool.
    normalize
        If ``True`` (default), L2-normalise the centroid to unit length.
        This matters for the centrality baseline: a non-normalised
        centroid has a length that depends on the pool, and cosine
        similarity to it is only interpretable if the length is
        comparable across pools. All sweeps must use the same setting.

    Returns
    -------
    numpy.ndarray
        ``(D,)`` ``float32``. If the pool is empty or the centroid is a
        zero vector, a zero vector is returned (not NaN) so that
        downstream code does not need a special case.
    """
    X = np.asarray(pool_emb)
    if X.ndim != 2:
        raise ValueError(f"pool_emb must be 2-D, got ndim={X.ndim}")
    if X.shape[0] == 0:
        return np.zeros(X.shape[1], dtype=np.float32)
    centroid = X.mean(axis=0).astype(np.float64)
    if normalize:
        n = float(np.linalg.norm(centroid))
        if n > 0:
            centroid = centroid / n
    return centroid.astype(np.float32)


def save_pool(
    pool: pd.DataFrame,
    path: str | Path,
    *,
    seed: int,
    source_passages_path: str | Path | None = None,
    source_passages_sha1: str | None = None,
    extra_meta: dict | None = None,
) -> Path:
    """Write ``pool`` to parquet and a ``.meta.json`` next to it.

    The meta file records the seed, source passages digest, the number
    of passages, and a digest of the ``story_id`` set (so that two
    pools can be checked for disjointness without loading the full
    parquet). ``extra_meta`` is merged in as-is.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(p, index=False)

    story_ids = sorted(set(pool["story_id"].dropna().astype(int).tolist()))
    story_ids_sha1 = hashlib.sha1(
        ",".join(str(s) for s in story_ids).encode("utf-8")
    ).hexdigest()

    meta = {
        "n_passages": int(len(pool)),
        "seed": int(seed),
        "story_ids_sha1": story_ids_sha1,
        "n_unique_stories": len(story_ids),
        "source_passages_path": (
            str(source_passages_path) if source_passages_path is not None else None
        ),
        "source_passages_sha1": source_passages_sha1,
    }
    if extra_meta:
        meta.update(extra_meta)

    meta_path = p.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
    )
    return p


def load_pool(
    path: str | Path,
) -> tuple[pd.DataFrame, dict]:
    """Load a pool parquet and (if present) its ``.meta.json``."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"pool file not found: {p}")
    pool = pd.read_parquet(p)
    meta_path = p.with_suffix(".meta.json")
    meta: dict = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return pool, meta