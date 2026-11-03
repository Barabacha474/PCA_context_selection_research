"""Deterministic train / out-of-domain splits by story_id.

What "distinct prompt subsets" means in this project
----------------------------------------------------
The proposal reserves "distinct prompt subsets for out-of-domain
robustness checks". In WritingPrompts each story has its own unique
prompt, so a split by ``story_id`` is also a split by ``prompt_id``.
We hash ``(seed, story_id)`` to a float in ``[0, 1)`` and threshold at
``train_frac``. This is deterministic, order-independent, and
reproducible across processes -- unlike Python's builtin ``hash()``,
which is salted per-run.

The split is a *partition*: every story goes to exactly one side, and
``train_ids & ood_ids == set()`` by construction. ``pools.py`` relies
on that invariant when it builds non-overlapping pools.
"""
from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np
import pandas as pd

__all__ = ["assign_story_split", "split_ids"]


def _stable_unit_hash(story_id: int, seed: int) -> float:
    """Deterministic float in ``[0, 1)`` from ``(seed, story_id)``.

    Uses SHA-1 of the string ``"<seed>:<story_id>"`` -- NOT Python's
    builtin ``hash()``, which is salted by ``PYTHONHASHSEED`` and
    changes between processes.
    """
    h = hashlib.sha1(f"{seed}:{int(story_id)}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def assign_story_split(
    story_ids: Sequence[int],
    train_frac: float = 0.8,
    seed: int = 0,
    train_label: str = "train",
    ood_label: str = "ood",
) -> pd.Series:
    """Assign each ``story_id`` to a split via stable hash.

    Parameters
    ----------
    story_ids
        Iterable of integer story ids. Duplicates receive the same
        split (the assignment depends only on the id and the seed).
    train_frac
        Fraction of the id space assigned to ``train``. Must be in
        ``(0, 1)``.
    seed
        Split seed. Changing it reshuffles the partition without
        changing its shape.
    train_label, ood_label
        Split labels to emit.

    Returns
    -------
    pandas.Series
        Split label per position in ``story_ids``, same length and
        index as the input.
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")

    ids = pd.Series(list(story_ids), dtype="int64")
    thresholds = ids.map(lambda sid: _stable_unit_hash(int(sid), seed))
    return pd.Series(
        np.where(thresholds.to_numpy() < train_frac, train_label, ood_label),
        index=ids.index,
        name="split",
    )


def split_ids(
    df: pd.DataFrame,
    train_frac: float = 0.8,
    seed: int = 0,
    id_col: str = "story_id",
) -> tuple[set[int], set[int]]:
    """Return ``(train_ids, ood_ids)`` for the unique stories in ``df``.

    Convenience wrapper around :func:`assign_story_split`. The two sets
    are disjoint by construction.
    """
    if id_col not in df.columns:
        raise KeyError(f"{id_col!r} not in DataFrame columns")
    unique_ids = df[id_col].drop_duplicates().to_numpy()
    labels = assign_story_split(unique_ids, train_frac=train_frac, seed=seed)
    train_ids = set(unique_ids[labels.to_numpy() == "train"].tolist())
    ood_ids = set(unique_ids[labels.to_numpy() == "ood"].tolist())
    return train_ids, ood_ids