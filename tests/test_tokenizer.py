"""Tests for pcadiv.data.tokenizer (Block 3)."""
from __future__ import annotations

import numpy as np
import pytest

from pcadiv.data.tokenizer import (
    BaseBudgetTokenizer,
    HFTokenizerFromModel,
    RegexTokenizer,
    get_tokenizer,
    list_tokenizers,
    register_tokenizer,
)

# tiktoken is an optional dependency; skip its tests if absent.
tiktoken = pytest.importorskip("tiktoken", reason="optional dep [data]")


# ---- registry --------------------------------------------------------

def test_registry_contains_core_names():
    names = list_tokenizers()
    for expected in ("regex", "tiktoken", "hf"):
        assert expected in names


def test_get_tokenizer_returns_instance():
    tok = get_tokenizer("regex")
    assert isinstance(tok, RegexTokenizer)


def test_get_tokenizer_unknown_raises():
    with pytest.raises(KeyError, match="unknown tokenizer"):
        get_tokenizer("nope")


def test_register_duplicate_name_raises():
    class Fake(BaseBudgetTokenizer):
        name = "regex"
        def count(self, text): return 0

    with pytest.raises(ValueError, match="already registered"):
        register_tokenizer(Fake)


# ---- RegexTokenizer --------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("", 0),
    ("   ", 0),
    ("hello", 1),
    ("hello world", 2),
    ("hello, world!", 2),
    ("a-b-c", 3),
    ("one  two\tthree\nfour", 4),
    ("123 456", 2),
    ("Привет, мир!", 2),
    ("snake_case is one", 3),  # underscore is \w
])
def test_regex_count(text, expected):
    assert RegexTokenizer().count(text) == expected


def test_regex_count_many_matches_count():
    texts = ["", "hi", "hi there", "a, b, c.", "Привет мир"]
    tok = RegexTokenizer()
    got = tok.count_many(texts)
    expected = np.array([tok.count(t) for t in texts], dtype=np.int64)
    np.testing.assert_array_equal(got, expected)


def test_regex_count_many_dtype_and_shape():
    tok = RegexTokenizer()
    got = tok.count_many(["a b c", "d e"])
    assert got.dtype == np.int64
    assert got.shape == (2,)


def test_regex_count_many_empty():
    tok = RegexTokenizer()
    got = tok.count_many([])
    assert got.shape == (0,)
    assert got.dtype == np.int64


def test_regex_fingerprint_stable():
    assert RegexTokenizer().fingerprint() == RegexTokenizer().fingerprint()


# ---- TiktokenTokenizer ----------------------------------------------

def test_tiktoken_count_simple():
    tok = get_tokenizer("tiktoken", encoding="cl100k_base")
    # Known cl100k_base behaviour: single ASCII words = 1 token each
    assert tok.count("hello") == 1
    assert tok.count("hello world") == 2
    assert tok.count("") == 0


def test_tiktoken_count_many_matches_count():
    tok = get_tokenizer("tiktoken", encoding="cl100k_base")
    texts = ["hello", "hello world", "", "The quick brown fox."]
    got = tok.count_many(texts)
    expected = np.array([tok.count(t) for t in texts], dtype=np.int64)
    np.testing.assert_array_equal(got, expected)


def test_tiktoken_fingerprint_depends_on_encoding():
    a = get_tokenizer("tiktoken", encoding="cl100k_base").fingerprint()
    b = get_tokenizer("tiktoken", encoding="o200k_base").fingerprint()
    assert a != b


def test_tiktoken_empty_encoding_raises():
    with pytest.raises(ValueError, match="encoding must be a non-empty"):
        get_tokenizer("tiktoken", encoding="")


# ---- HFTokenizerFromModel (stub) ------------------------------------

def test_hf_stub_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="Block 7"):
        get_tokenizer("hf", model_id="bert-base-uncased")