"""Tests for RelevanceSelector -- Block 8a."""
from __future__ import annotations

import numpy as np
import pytest

from pcadiv.selection.selectors import (
    RelevanceSelector,
    get_selector,
    list_selectors,
)


# ---- registry -------------------------------------------------------

def test_registry_contains_relevance():
    assert "relevance" in list_selectors()


def test_get_relevance_returns_instance():
    assert isinstance(get_selector("relevance"), RelevanceSelector)


# ---- anchor requirement --------------------------------------------

def test_requires_core_emb():
    emb = np.ones((3, 4), dtype=np.float32)
    toks = np.ones(3, dtype=np.int64)
    with pytest.raises(ValueError, match="requires core_emb"):
        get_selector("relevance").select(emb, toks, budget=100)


# ---- ordering -------------------------------------------------------

def test_anchor_matching_pool_index_first():
    """If the anchor equals one of the pool rows, that row is selected first."""
    emb = np.array(
        [[0.0, 1.0], [1.0, 0.0], [0.7, 0.7]], dtype=np.float32
    )
    toks = np.ones(3, dtype=np.int64)
    core = emb[1].copy()
    res = get_selector("relevance").select(emb, toks, budget=100, core_emb=core)
    assert res.indices[0] == 1


def test_descending_similarity_order():
    """With anchor [1,0], similarities are 1, 0, -1 -> order [0, 1, 2]."""
    emb = np.array(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32
    )
    toks = np.ones(3, dtype=np.int64)
    core = np.array([1.0, 0.0], dtype=np.float32)
    res = get_selector("relevance").select(emb, toks, budget=100, core_emb=core)
    assert res.indices.tolist() == [0, 1, 2]


def test_most_similar_selected_first_most_dissimilar_last():
    emb = np.array(
        [[0.0, 1.0], [-1.0, 0.0], [0.9, 0.1], [0.1, 0.9]], dtype=np.float32
    )
    toks = np.ones(4, dtype=np.int64)
    core = np.array([1.0, 0.0], dtype=np.float32)
    res = get_selector("relevance").select(emb, toks, budget=100, core_emb=core)
    # [0.9, 0.1] should be first; [-1, 0] last
    assert res.indices[0] == 2
    assert res.indices[-1] == 1


def test_anchor_is_pool_centroid_ranks_typical_first():
    """Centrality baseline: centroid anchor ranks a 'typical' point first."""
    emb = np.array(
        [[1.0, 0.0], [1.0, 0.1], [0.9, -0.05], [-1.0, 0.0]], dtype=np.float32
    )
    toks = np.ones(4, dtype=np.int64)
    centroid = emb.mean(axis=0).astype(np.float32)
    res = get_selector("relevance").select(
        emb, toks, budget=100, core_emb=centroid
    )
    # The outlier [-1, 0] must be last
    assert res.indices[-1] == 3


# ---- determinism & ties --------------------------------------------

def test_deterministic():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((20, 8)).astype(np.float32)
    toks = np.ones(20, dtype=np.int64)
    core = emb.mean(axis=0).astype(np.float32)
    a = get_selector("relevance").select(emb, toks, budget=100, core_emb=core)
    b = get_selector("relevance").select(emb, toks, budget=100, core_emb=core)
    np.testing.assert_array_equal(a.indices, b.indices)


def test_tie_breaking_by_index():
    """Identical vectors -> identical similarity -> lower index first."""
    emb = np.array(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32
    )
    toks = np.ones(4, dtype=np.int64)
    core = np.array([1.0, 0.0], dtype=np.float32)
    res = get_selector("relevance").select(emb, toks, budget=100, core_emb=core)
    assert res.indices.tolist() == [0, 1, 2, 3]


# ---- cosine, not dot product ---------------------------------------

def test_uses_cosine_not_dot_product():
    """Unnormalized embeddings: cosine and dot give different orders.

    emb = [[10, 0], [1, 1]], anchor = [1, 1]:
      dot:    [10, 2]       -> 0 first
      cosine: [0.707, 1.0]  -> 1 first
    RelevanceSelector must use cosine.
    """
    emb = np.array([[10.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    toks = np.ones(2, dtype=np.int64)
    core = np.array([1.0, 1.0], dtype=np.float32)
    res = get_selector("relevance").select(emb, toks, budget=100, core_emb=core)
    assert res.indices[0] == 1


# ---- zero-norm handling --------------------------------------------

def test_zero_norm_candidate_does_not_crash():
    emb = np.array(
        [[1.0, 0.0], [0.0, 0.0], [0.5, 0.5]], dtype=np.float32
    )
    toks = np.ones(3, dtype=np.int64)
    core = np.array([1.0, 0.0], dtype=np.float32)
    res = get_selector("relevance").select(emb, toks, budget=100, core_emb=core)
    # sim: [1, 0, 0.707] -> order [0, 2, 1]
    assert res.indices.tolist() == [0, 2, 1]


def test_zero_norm_anchor_returns_source_order():
    """Zero anchor -> all sim 0 -> stable sort keeps original order."""
    emb = np.array(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32
    )
    toks = np.ones(3, dtype=np.int64)
    core = np.zeros(2, dtype=np.float32)
    res = get_selector("relevance").select(emb, toks, budget=100, core_emb=core)
    assert res.indices.tolist() == [0, 1, 2]


# ---- budget ---------------------------------------------------------

def test_budget_respected():
    emb = np.eye(5, dtype=np.float32)
    toks = np.array([10, 10, 10, 10, 10], dtype=np.int64)
    core = emb[0].copy()
    res = get_selector("relevance").select(emb, toks, budget=25, core_emb=core)
    assert res.total_tokens <= 25
    assert len(res) == 2


def test_skips_expensive_anchor_match_and_keeps_fitting():
    """If the top-relevance passage is too expensive, keep next-best fitting."""
    emb = np.eye(3, dtype=np.float32)
    toks = np.array([100, 5, 5], dtype=np.int64)
    core = emb[0].copy()  # best match is index 0 (cost 100)
    res = get_selector("relevance").select(emb, toks, budget=10, core_emb=core)
    # index 0 doesn't fit; next by similarity is either 1 or 2 (both sim 0)
    assert 0 not in res.indices.tolist()
    assert res.total_tokens <= 10


def test_empty_pool_returns_empty():
    emb = np.zeros((0, 4), dtype=np.float32)
    toks = np.zeros(0, dtype=np.int64)
    core = np.zeros(4, dtype=np.float32)
    res = get_selector("relevance").select(emb, toks, budget=100, core_emb=core)
    assert len(res) == 0