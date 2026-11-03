"""Slow tests for real (network-downloaded) encoders and tokenizers -- Block 7.

These tests download models. They are skipped by default; enable with:

    PowerShell:  $env:PCADIV_RUN_SLOW = "1"; pytest tests/test_encoders_real.py -v
    bash:        PCADIV_RUN_SLOW=1 pytest tests/test_encoders_real.py -v

Override the models used with:

    PCADIV_TEST_ST_MODEL          (default: sentence-transformers/all-MiniLM-L6-v2)
    PCADIV_TEST_HF_MODEL          (default: prajjwal1/bert-tiny)
    PCADIV_TEST_HF_TOKENIZER_MODEL (default: NousResearch/Meta-Llama-3-8B)
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from pcadiv.data.tokenizer import get_tokenizer
from pcadiv.embeddings.encoders import get_encoder

SLOW = os.environ.get("PCADIV_RUN_SLOW", "0") == "1"
pytestmark = pytest.mark.skipif(
    not SLOW, reason="set PCADIV_RUN_SLOW=1 to run (downloads models)"
)

ST_MODEL = os.environ.get(
    "PCADIV_TEST_ST_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
HF_MODEL = os.environ.get("PCADIV_TEST_HF_MODEL", "google/bert_uncased_L-2_H-128_A-2")
HF_TOKENIZER_MODEL = os.environ.get(
    "PCADIV_TEST_HF_TOKENIZER_MODEL", "NousResearch/Meta-Llama-3-8B"
)

SAMPLE_TEXTS = [
    "The dragon circled the tower at dusk.",
    "A wizard whispered the binding spell.",
    "Three moons rose over the silver forest.",
]


# ---- shared assertion helpers --------------------------------------

def _check_encode_contract(X: np.ndarray, n: int, dim: int) -> None:
    assert X.shape == (n, dim)
    assert X.dtype == np.float32
    assert np.isfinite(X).all()


def _check_normalized(X: np.ndarray) -> None:
    norms = np.linalg.norm(X, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-4)


# ---- SentenceTransformerEncoder -------------------------------------

def _st_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not _st_available(), reason="sentence-transformers not installed"
)
def test_st_encoder_contract():
    enc = get_encoder("st", model_id=ST_MODEL)
    X = enc.encode(SAMPLE_TEXTS)
    _check_encode_contract(X, len(SAMPLE_TEXTS), enc.dim)
    _check_normalized(X)


@pytest.mark.skipif(
    not _st_available(), reason="sentence-transformers not installed"
)
def test_st_encoder_dim_positive():
    enc = get_encoder("st", model_id=ST_MODEL)
    assert enc.dim > 0


@pytest.mark.skipif(
    not _st_available(), reason="sentence-transformers not installed"
)
def test_st_encoder_empty_input():
    enc = get_encoder("st", model_id=ST_MODEL)
    X = enc.encode([])
    assert X.shape == (0, enc.dim)
    assert X.dtype == np.float32


@pytest.mark.skipif(
    not _st_available(), reason="sentence-transformers not installed"
)
def test_st_encoder_unnormalized_does_not_crash():
    """normalize=False must return finite float32; norms depend on the model.

    Some ST models (e.g. all-MiniLM-L6-v2) carry a Normalize module and
    always output unit vectors -- that is a model property, not a bug
    in this project.
    """
    enc = get_encoder("st", model_id=ST_MODEL, normalize=False)
    X = enc.encode(SAMPLE_TEXTS)
    assert X.shape == (len(SAMPLE_TEXTS), enc.dim)
    assert X.dtype == np.float32
    assert np.isfinite(X).all()


# ---- HFMeanPoolEncoder ---------------------------------------------

def test_hf_mean_encoder_contract():
    enc = get_encoder("hf_mean", model_id=HF_MODEL)
    X = enc.encode(SAMPLE_TEXTS)
    _check_encode_contract(X, len(SAMPLE_TEXTS), enc.dim)
    _check_normalized(X)


def test_hf_mean_encoder_batching_consistency():
    """Splitting into batches must not change the output."""
    enc = get_encoder("hf_mean", model_id=HF_MODEL)
    X_full = enc.encode(SAMPLE_TEXTS, batch_size=8)
    X_b1 = enc.encode(SAMPLE_TEXTS, batch_size=1)
    np.testing.assert_allclose(X_full, X_b1, atol=1e-5)


def test_hf_mean_encoder_empty_input():
    enc = get_encoder("hf_mean", model_id=HF_MODEL)
    X = enc.encode([])
    assert X.shape == (0, enc.dim)


def test_hf_mean_encoder_handles_empty_string():
    """Empty string must produce a finite row (BERT pads it to [CLS][SEP])."""
    enc = get_encoder("hf_mean", model_id=HF_MODEL)
    X = enc.encode([""])
    assert np.isfinite(X).all()
    assert X.shape == (1, enc.dim)


# ---- HFCLSEncoder --------------------------------------------------

def test_hf_cls_encoder_contract():
    enc = get_encoder("hf_cls", model_id=HF_MODEL)
    X = enc.encode(SAMPLE_TEXTS)
    _check_encode_contract(X, len(SAMPLE_TEXTS), enc.dim)
    _check_normalized(X)


def test_hf_cls_encoder_differs_from_mean():
    """CLS and mean pooling must not give identical embeddings."""
    cls_enc = get_encoder("hf_cls", model_id=HF_MODEL)
    mean_enc = get_encoder("hf_mean", model_id=HF_MODEL)
    X_cls = cls_enc.encode(SAMPLE_TEXTS)
    X_mean = mean_enc.encode(SAMPLE_TEXTS)
    assert not np.allclose(X_cls, X_mean, atol=1e-3)


# ---- HFTokenizerFromModel ------------------------------------------

def test_hf_tokenizer_count_basic():
    tok = get_tokenizer("hf", model_id=HF_TOKENIZER_MODEL)
    assert tok.count("") == 0
    assert tok.count("hello") >= 1


def test_hf_tokenizer_count_many_matches_count():
    tok = get_tokenizer("hf", model_id=HF_TOKENIZER_MODEL)
    texts = ["hello", "hello world", "", "The quick brown fox."]
    batch = tok.count_many(texts)
    singles = np.array([tok.count(t) for t in texts], dtype=np.int64)
    np.testing.assert_array_equal(batch, singles)


def test_hf_tokenizer_count_many_dtype():
    tok = get_tokenizer("hf", model_id=HF_TOKENIZER_MODEL)
    out = tok.count_many(["a", "b"])
    assert out.dtype == np.int64


def test_hf_tokenizer_count_many_empty():
    tok = get_tokenizer("hf", model_id=HF_TOKENIZER_MODEL)
    out = tok.count_many([])
    assert out.shape == (0,)
    assert out.dtype == np.int64


def test_hf_tokenizer_fingerprint_depends_on_add_special():
    a = get_tokenizer(
        "hf", model_id=HF_TOKENIZER_MODEL, add_special_tokens=False
    ).fingerprint()
    b = get_tokenizer(
        "hf", model_id=HF_TOKENIZER_MODEL, add_special_tokens=True
    ).fingerprint()
    assert a != b


def test_hf_tokenizer_add_special_tokens_changes_count():
    no_spec = get_tokenizer(
        "hf", model_id=HF_TOKENIZER_MODEL, add_special_tokens=False
    )
    with_spec = get_tokenizer(
        "hf", model_id=HF_TOKENIZER_MODEL, add_special_tokens=True
    )
    text = "hello world"
    assert with_spec.count(text) > no_spec.count(text)


def test_hf_tokenizer_empty_model_id_raises():
    with pytest.raises(ValueError, match="model_id must be a non-empty"):
        get_tokenizer("hf", model_id="")