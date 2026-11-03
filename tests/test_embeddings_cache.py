"""Tests for embeddings cache -- Block 10."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pcadiv.embeddings.cache import cache_path, clear_cache, load_or_encode
from pcadiv.embeddings.encoders import BaseEncoder, get_encoder
from pcadiv.utils.io import sha1_of_texts


# ---- counting encoder for tests ------------------------------------

class CountingEncoder(BaseEncoder):
    """Deterministic mock encoder that counts ``encode`` calls."""

    name = "counting_test"

    def __init__(self, model_id: str = "test", dim: int = 4, normalize: bool = False):
        super().__init__(model_id=model_id, device="cpu", normalize=normalize)
        self._dim = dim
        self.calls = 0

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts, batch_size: int = 64):
        self.calls += 1
        seed = int(sha1_of_texts(texts)[:8], 16)
        rng = np.random.default_rng(seed)
        return rng.standard_normal((len(texts), self._dim)).astype(np.float32)


# ---- cache_path ----------------------------------------------------

def test_cache_path_layout(tmp_path):
    enc = CountingEncoder(dim=8)
    texts = ["a", "b"]
    p = cache_path(enc, texts, tmp_path)
    assert p.parent.name == enc.fingerprint()
    assert p.name == f"{sha1_of_texts(texts)}.npz"
    assert p.parent.parent == tmp_path


def test_cache_path_order_sensitive(tmp_path):
    enc = CountingEncoder()
    p1 = cache_path(enc, ["a", "b"], tmp_path)
    p2 = cache_path(enc, ["b", "a"], tmp_path)
    assert p1 != p2


def test_cache_path_depends_on_encoder(tmp_path):
    e1 = CountingEncoder(dim=4)
    e2 = CountingEncoder(dim=8)
    texts = ["a"]
    assert cache_path(e1, texts, tmp_path) != cache_path(e2, texts, tmp_path)


# ---- load_or_encode: hit / miss ------------------------------------

def test_first_call_encodes_and_writes(tmp_path):
    enc = CountingEncoder()
    emb = load_or_encode(enc, ["a", "b"], tmp_path)
    assert enc.calls == 1
    assert emb.shape == (2, 4)
    assert cache_path(enc, ["a", "b"], tmp_path).exists()


def test_second_call_reads_from_cache(tmp_path):
    enc = CountingEncoder()
    emb1 = load_or_encode(enc, ["a", "b"], tmp_path)
    emb2 = load_or_encode(enc, ["a", "b"], tmp_path)
    assert enc.calls == 1  # only first call hit the encoder
    np.testing.assert_array_equal(emb1, emb2)


def test_different_texts_new_cache_entry(tmp_path):
    enc = CountingEncoder()
    load_or_encode(enc, ["a"], tmp_path)
    load_or_encode(enc, ["b"], tmp_path)
    assert enc.calls == 2
    assert cache_path(enc, ["a"], tmp_path).exists()
    assert cache_path(enc, ["b"], tmp_path).exists()


def test_different_encoder_separate_namespace(tmp_path):
    e1 = CountingEncoder(dim=4)
    e2 = CountingEncoder(dim=8)
    load_or_encode(e1, ["a"], tmp_path)
    load_or_encode(e2, ["a"], tmp_path)
    assert e1.calls == 1
    assert e2.calls == 1
    assert cache_path(e1, ["a"], tmp_path) != cache_path(e2, ["a"], tmp_path)


def test_order_matters_for_cache(tmp_path):
    enc = CountingEncoder()
    load_or_encode(enc, ["a", "b"], tmp_path)
    load_or_encode(enc, ["b", "a"], tmp_path)
    assert enc.calls == 2


# ---- force_recompute -----------------------------------------------

def test_force_recompute_ignores_existing(tmp_path):
    enc = CountingEncoder()
    load_or_encode(enc, ["a"], tmp_path)
    load_or_encode(enc, ["a"], tmp_path, force_recompute=True)
    assert enc.calls == 2


# ---- cache disabled -------------------------------------------------

def test_cache_dir_none_always_encodes():
    enc = CountingEncoder()
    load_or_encode(enc, ["a"], cache_dir=None)
    load_or_encode(enc, ["a"], cache_dir=None)
    assert enc.calls == 2


def test_cache_dir_none_returns_same_as_direct_encode():
    enc = CountingEncoder()
    direct = enc.encode(["x", "y"])
    via_cache = load_or_encode(enc, ["x", "y"], cache_dir=None)
    np.testing.assert_array_equal(direct, via_cache)


# ---- corruption handling -------------------------------------------

def test_corrupted_shape_triggers_recompute(tmp_path):
    enc = CountingEncoder(dim=4)
    texts = ["a", "b"]
    load_or_encode(enc, texts, tmp_path)
    assert enc.calls == 1

    # Corrupt the cache with wrong-shaped array.
    path = cache_path(enc, texts, tmp_path)
    np.savez(path, emb=np.zeros((999, 999), dtype=np.float32))

    with pytest.warns(RuntimeWarning, match="shape"):
        emb = load_or_encode(enc, texts, tmp_path)
    assert enc.calls == 2
    assert emb.shape == (2, 4)


def test_corrupted_missing_key_triggers_recompute(tmp_path):
    enc = CountingEncoder()
    texts = ["a"]
    load_or_encode(enc, texts, tmp_path)
    assert enc.calls == 1

    path = cache_path(enc, texts, tmp_path)
    np.savez(path, not_emb=np.zeros((1, 4), dtype=np.float32))

    with pytest.warns(RuntimeWarning, match="cache read failed"):
        load_or_encode(enc, texts, tmp_path)
    assert enc.calls == 2


def test_corrupted_wrong_dtype_triggers_recompute(tmp_path):
    enc = CountingEncoder(dim=4)
    texts = ["a"]
    load_or_encode(enc, texts, tmp_path)
    path = cache_path(enc, texts, tmp_path)
    np.savez(path, emb=np.zeros((1, 4), dtype=np.float64))

    with pytest.warns(RuntimeWarning, match="dtype"):
        load_or_encode(enc, texts, tmp_path)
    assert enc.calls == 2


# ---- round-trip fidelity -------------------------------------------

def test_cached_values_equal_fresh_encode(tmp_path):
    enc1 = CountingEncoder(dim=4)
    texts = ["foo", "bar", "baz"]
    fresh = enc1.encode(texts)
    via_cache_cold = load_or_encode(enc1, texts, tmp_path)

    enc2 = CountingEncoder(dim=4)
    via_cache_warm = load_or_encode(enc2, texts, tmp_path)

    np.testing.assert_array_equal(fresh, via_cache_cold)
    np.testing.assert_array_equal(fresh, via_cache_warm)
    assert enc2.calls == 0  # served from cache, encoder untouched


# ---- clear_cache ---------------------------------------------------

def test_clear_cache_specific_fingerprint(tmp_path):
    e1 = CountingEncoder(dim=4)
    e2 = CountingEncoder(dim=8)
    load_or_encode(e1, ["a"], tmp_path)
    load_or_encode(e2, ["b"], tmp_path)

    n = clear_cache(tmp_path, fingerprint=e1.fingerprint())
    assert n == 1
    assert not (tmp_path / e1.fingerprint()).exists()
    assert (tmp_path / e2.fingerprint()).exists()


def test_clear_cache_all(tmp_path):
    e1 = CountingEncoder(dim=4)
    e2 = CountingEncoder(dim=8)
    load_or_encode(e1, ["a"], tmp_path)
    load_or_encode(e2, ["b"], tmp_path)

    n = clear_cache(tmp_path)
    assert n == 2
    assert not tmp_path.exists()


def test_clear_cache_unknown_dir_returns_zero(tmp_path):
    assert clear_cache(tmp_path / "nonexistent") == 0


def test_clear_cache_unknown_fingerprint_returns_zero(tmp_path):
    e = CountingEncoder()
    load_or_encode(e, ["a"], tmp_path)
    assert clear_cache(tmp_path, fingerprint="deadbeef0000") == 0
    assert (tmp_path / e.fingerprint()).exists()


# ---- integration with real HashingEncoder --------------------------

def test_integration_with_hashing_encoder(tmp_path):
    enc = get_encoder("hashing", dim=32)
    texts = ["hello world", "goodbye moon"]
    e1 = load_or_encode(enc, texts, tmp_path)
    e2 = load_or_encode(enc, texts, tmp_path)
    np.testing.assert_array_equal(e1, e2)
    assert e1.shape == (2, 32)
    assert e1.dtype == np.float32