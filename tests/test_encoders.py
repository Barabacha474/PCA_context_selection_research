"""Tests for pcadiv.embeddings.encoders (Block 2)."""
from __future__ import annotations

import numpy as np
import pytest

from pcadiv.embeddings.encoders import (
    BaseEncoder,
    HashingEncoder,
    get_encoder,
    list_encoders,
    register_encoder,
)


# ---- registry --------------------------------------------------------

def test_registry_contains_hashing():
    assert "hashing" in list_encoders()


def test_get_encoder_returns_instance():
    enc = get_encoder("hashing", dim=64)
    assert isinstance(enc, HashingEncoder)
    assert enc.dim == 64


def test_get_encoder_unknown_raises():
    with pytest.raises(KeyError, match="unknown encoder"):
        get_encoder("does-not-exist")


def test_register_duplicate_name_different_class_raises():
    class Fake(BaseEncoder):
        name = "hashing"  # collide with HashingEncoder
        @property
        def dim(self): return 1
        def encode(self, texts, batch_size=64): return np.zeros((len(texts), 1), dtype=np.float32)

    with pytest.raises(ValueError, match="already registered"):
        register_encoder(Fake)


def test_register_requires_name():
    class NoName(BaseEncoder):
        @property
        def dim(self): return 1
        def encode(self, texts, batch_size=64): return np.zeros((len(texts), 1), dtype=np.float32)

    with pytest.raises(ValueError, match="non-empty string"):
        register_encoder(NoName)


# ---- encode: shape / dtype / normalization ---------------------------

def test_encode_shape_and_dtype():
    enc = HashingEncoder(dim=128)
    texts = ["hello world", "foo bar baz", ""]
    X = enc.encode(texts)
    assert X.shape == (3, 128)
    assert X.dtype == np.float32


def test_encode_normalized_rows():
    enc = HashingEncoder(dim=64, normalize=True)
    X = enc.encode(["alpha beta gamma", "delta epsilon"])
    norms = np.linalg.norm(X, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_encode_unnormalized_preserves_counts():
    enc = HashingEncoder(dim=256, normalize=False)
    X = enc.encode(["a b c"])
    assert X.sum() == pytest.approx(3.0)


def test_encode_empty_string_is_zero_vector():
    enc = HashingEncoder(dim=32, normalize=True)
    X = enc.encode([""])
    np.testing.assert_array_equal(X, np.zeros((1, 32), dtype=np.float32))


def test_encode_empty_batch():
    enc = HashingEncoder(dim=16)
    X = enc.encode([])
    assert X.shape == (0, 16)


# ---- determinism -----------------------------------------------------

def test_encode_deterministic_across_instances():
    a = HashingEncoder(dim=64).encode(["hello world"])
    b = HashingEncoder(dim=64).encode(["hello world"])
    np.testing.assert_array_equal(a, b)


# ---- fingerprint -----------------------------------------------------

def test_fingerprint_stable():
    a = HashingEncoder(dim=64).fingerprint()
    b = HashingEncoder(dim=64).fingerprint()
    assert a == b
    assert len(a) == 12


def test_fingerprint_changes_with_dim():
    assert HashingEncoder(dim=64).fingerprint() != HashingEncoder(dim=128).fingerprint()


def test_fingerprint_changes_with_normalize():
    assert HashingEncoder(dim=64, normalize=True).fingerprint() != \
           HashingEncoder(dim=64, normalize=False).fingerprint()


def test_fingerprint_changes_with_model_id():
    assert HashingEncoder(dim=64, model_id="a").fingerprint() != \
           HashingEncoder(dim=64, model_id="b").fingerprint()


# ---- input validation ------------------------------------------------

def test_negative_dim_raises():
    with pytest.raises(ValueError, match="dim must be positive"):
        HashingEncoder(dim=0)


def test_repr_contains_key_fields():
    r = repr(HashingEncoder(dim=64))
    assert "HashingEncoder" in r
    assert "dim=64" in r
    assert "normalize=True" in r