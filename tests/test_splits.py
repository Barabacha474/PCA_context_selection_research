"""Tests for splits.py -- Block 11d."""
from __future__ import annotations

import pandas as pd
import pytest

from pcadiv.data.splits import assign_story_split, split_ids


def test_assign_returns_series_same_length():
    ids = list(range(100))
    labels = assign_story_split(ids, train_frac=0.8, seed=0)
    assert len(labels) == 100
    assert set(labels.unique()) == {"train", "ood"}


def test_assign_deterministic_by_seed():
    ids = list(range(100))
    a = assign_story_split(ids, train_frac=0.8, seed=0)
    b = assign_story_split(ids, train_frac=0.8, seed=0)
    assert (a == b).all()


def test_assign_different_seeds_different_partition():
    ids = list(range(1000))
    a = assign_story_split(ids, train_frac=0.5, seed=0)
    b = assign_story_split(ids, train_frac=0.5, seed=1)
    # Two independent hashes on 1000 ids: expect disagreement on ~half.
    disagreement = (a != b).sum()
    assert disagreement > 100  # loose bound, statistical


def test_duplicates_same_split():
    ids = [7, 7, 7, 7]
    labels = assign_story_split(ids, train_frac=0.5, seed=0)
    assert labels.nunique() == 1


def test_train_frac_extremes_rejected():
    with pytest.raises(ValueError, match="train_frac must be in"):
        assign_story_split([1, 2], train_frac=0.0)
    with pytest.raises(ValueError, match="train_frac must be in"):
        assign_story_split([1, 2], train_frac=1.0)


def test_train_frac_controls_proportion():
    """Roughly (not exactly) the requested fraction goes to train."""
    ids = list(range(10_000))
    labels = assign_story_split(ids, train_frac=0.3, seed=0)
    frac_train = (labels == "train").mean()
    assert 0.27 < frac_train < 0.33


def test_custom_labels():
    labels = assign_story_split(
        [1, 2, 3], train_frac=0.5, seed=0,
        train_label="primary", ood_label="holdout",
    )
    assert set(labels.unique()) <= {"primary", "holdout"}


# ---- split_ids ------------------------------------------------------

def test_split_ids_disjoint_and_cover():
    df = pd.DataFrame({"story_id": list(range(100))})
    train, ood = split_ids(df, train_frac=0.8, seed=0)
    assert train & ood == set()
    assert train | ood == set(range(100))


def test_split_ids_deterministic():
    df = pd.DataFrame({"story_id": list(range(100))})
    a = split_ids(df, train_frac=0.5, seed=0)
    b = split_ids(df, train_frac=0.5, seed=0)
    assert a == b


def test_split_ids_missing_col():
    df = pd.DataFrame({"not_story_id": [1, 2]})
    with pytest.raises(KeyError, match="story_id"):
        split_ids(df)


def test_split_ids_drops_duplicates_silently():
    df = pd.DataFrame({"story_id": [1, 1, 2, 2, 3]})
    train, ood = split_ids(df, train_frac=0.5, seed=0)
    assert len(train) + len(ood) == 3  # 3 unique stories, not 5