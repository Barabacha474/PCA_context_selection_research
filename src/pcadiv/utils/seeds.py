"""Seed utilities for reproducible randomness across the project.

All stochastic components (selectors, splits, sampling) should obtain
their randomness via :func:`make_rng` rather than using the global
:mod:`numpy.random` or :mod:`random` state. This keeps determinism local
and avoids the classic "one component globally seeds, everything else
inherits" bug.
"""
from __future__ import annotations

import numpy as np

__all__ = ["make_rng"]


def make_rng(seed: int | None = None) -> np.random.Generator:
    """Return a fresh :class:`numpy.random.Generator`.

    Parameters
    ----------
    seed
        Seed for the generator. ``None`` yields a non-deterministic
        generator seeded from OS entropy.

    Returns
    -------
    numpy.random.Generator
        A new generator. Two calls with the same integer seed produce
        identical streams; two generators from the same seed are
        independent (advancing one does not advance the other).
    """
    return np.random.default_rng(seed)