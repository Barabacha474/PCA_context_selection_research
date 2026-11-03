"""Disk cache for encoder outputs.

Layout
------
::

    <cache_dir>/<encoder_fingerprint>/<sha1_of_texts>.npz

The path *is* the cache key. Nothing else is stored inside the
``.npz`` except the embedding matrix itself:

* ``encoder_fingerprint`` covers ``(name, model_id, normalize, dim)``
  -- change any of these and the old cache is invisible to the new
  encoder. It also gives a convenient unit for cleanup: nuking one
  subdirectory invalidates exactly one encoder's outputs.
* ``sha1_of_texts`` is order-sensitive (see ``utils.io``). Reordering
  the pool is a *different* cache entry, deliberately: the cache
  cannot return a permuted embedding matrix under a linear key.

Integrity
---------
No metadata is stored in the ``.npz``, and no pickling is involved.
On load we verify ``emb.shape == (len(texts), encoder.dim)`` and
``emb.dtype == float32``. A mismatch means the cache was corrupted or
written by a different encoder with a colliding fingerprint -- both
are programmer/config errors, but they must not silently poison
results. A ``RuntimeWarning`` is emitted and the cache is
recomputed.

Concurrency
-----------
Two processes encoding the same texts write the **same bytes** to the
**same path** (content hash), so a race is benign: ``os.replace`` is
atomic and the loser's write is overwritten by identical content.
No locks are needed.

Disabling the cache
-------------------
Pass ``cache_dir=None`` to skip disk entirely; ``load_or_encode``
becomes a thin wrapper around ``encoder.encode(texts)``.
"""
from __future__ import annotations

import shutil
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np

from pcadiv.embeddings.encoders import BaseEncoder
from pcadiv.utils.io import ensure_dir, load_npz, save_npz, sha1_of_texts

__all__ = ["cache_path", "load_or_encode", "clear_cache"]


def cache_path(
    encoder: BaseEncoder,
    texts: Sequence[str],
    cache_dir: str | Path,
) -> Path:
    """Return the cache file path for ``(encoder, texts)``.

    Does not touch the filesystem. Useful in tests and in logs.
    """
    base = Path(cache_dir) / encoder.fingerprint()
    return base / f"{sha1_of_texts(texts)}.npz"


def load_or_encode(
    encoder: BaseEncoder,
    texts: Sequence[str],
    cache_dir: str | Path | None,
    *,
    force_recompute: bool = False,
    batch_size: int = 64,
) -> np.ndarray:
    """Load embeddings from cache, or encode and cache them.

    Parameters
    ----------
    encoder
        Any :class:`~pcadiv.embeddings.encoders.BaseEncoder`. Its
        :meth:`fingerprint` is used as the per-encoder cache namespace.
    texts
        Sequence of strings. Order matters: it is part of the cache key.
    cache_dir
        Root directory for the cache. ``None`` disables the cache.
    force_recompute
        If ``True``, ignore any existing file and re-encode. The new
        result is written back to the same path.
    batch_size
        Forwarded to ``encoder.encode``.

    Returns
    -------
    numpy.ndarray
        ``float32`` array of shape ``(len(texts), encoder.dim)``.
    """
    if cache_dir is None:
        return encoder.encode(texts, batch_size=batch_size)

    path = cache_path(encoder, texts, cache_dir)
    n = len(texts)
    dim = encoder.dim

    if path.exists() and not force_recompute:
        try:
            cached = load_npz(path)["emb"]
        except (KeyError, OSError, ValueError) as e:
            warnings.warn(
                f"cache read failed for {path}: {e}; recomputing",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            if (
                cached.dtype == np.float32
                and cached.shape == (n, dim)
            ):
                return cached
            warnings.warn(
                f"cache at {path} has shape {cached.shape} dtype "
                f"{cached.dtype}, expected ({n}, {dim}) float32; "
                f"recomputing",
                RuntimeWarning,
                stacklevel=2,
            )

    emb = encoder.encode(texts, batch_size=batch_size)
    ensure_dir(path.parent)
    save_npz(path, emb=emb)
    return emb


def clear_cache(
    cache_dir: str | Path,
    fingerprint: str | None = None,
) -> int:
    """Delete cached embeddings.

    Parameters
    ----------
    cache_dir
        Root cache directory. If it does not exist, returns 0.
    fingerprint
        If given, delete only that encoder's subdirectory. If ``None``,
        delete the entire cache directory tree.

    Returns
    -------
    int
        Number of ``.npz`` files removed.
    """
    base = Path(cache_dir)
    if not base.exists():
        return 0

    if fingerprint is None:
        n = sum(1 for _ in base.rglob("*.npz"))
        shutil.rmtree(base)
        return n

    target = base / fingerprint
    if not target.exists():
        return 0
    n = sum(1 for _ in target.rglob("*.npz"))
    shutil.rmtree(target)
    return n