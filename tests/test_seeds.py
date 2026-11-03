"""Tests for pcadiv.utils.seeds."""
from __future__ import annotations

import numpy as np

from pcadiv.utils.seeds import make_rng


def test_same_seed_same_stream():
    a = make_rng(42).random(100)
    b = make_rng(42).random(100)
    np.testing.assert_array_equal(a, b)


def test_different_seeds_differ():
    a = make_rng(0).random(100)
    b = make_rng(1).random(100)
    assert not np.array_equal(a, b)


def test_independent_generators():
    """Advancing one generator must not advance another from the same seed."""
    a = make_rng(7)
    b = make_rng(7)
    a.random(10)  # advance a, leave b untouched
    np.testing.assert_array_equal(b.random(10), make_rng(7).random(10))