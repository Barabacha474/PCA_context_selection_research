"""Tests for passageize.py -- Block 11b."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pcadiv.data.passageize import passageize
from pcadiv.data.tokenizer import RegexTokenizer


@pytest.fixture
def tok():
    return RegexTokenizer()


@pytest.fixture
def simple_stories():
    return pd.DataFrame({
        "story_id": [0, 1],
        "prompt": ["p0", "p1"],
        "story": [
            "The quick brown fox jumps over the lazy dog. " * 10,
            "Another story with different words to slice." * 5,
        ],
    })


# ---- basic behaviour -----------------------------------------------

def test_returns_dataframe_with_expected_columns(simple_stories, tok):
    df = passageize(simple_stories, tok, passage_tokens=20)
    expected = {
        "passage_id", "story_id", "prompt_id",
        "text", "n_tokens", "position",
    }
    assert set(df.columns) == expected


def test_n_tokens_never_exceeds_passage_tokens(simple_stories, tok):
    df = passageize(simple_stories, tok, passage_tokens=20)
    assert (df["n_tokens"] <= 20).all()


def test_n_tokens_matches_tokenizer_count(simple_stories, tok):
    df = passageize(simple_stories, tok, passage_tokens=20)
    for _, row in df.iterrows():
        assert row["n_tokens"] == tok.count(row["text"])


def test_passage_ids_unique(simple_stories, tok):
    df = passageize(simple_stories, tok, passage_tokens=20)
    assert df["passage_id"].is_unique


def test_positions_zero_based_per_story(simple_stories, tok):
    df = passageize(simple_stories, tok, passage_tokens=20)
    for sid, g in df.groupby("story_id"):
        assert g["position"].tolist() == list(range(len(g)))


def test_story_id_and_prompt_preserved(simple_stories, tok):
    df = passageize(simple_stories, tok, passage_tokens=20)
    assert set(df["story_id"].unique()) == {0, 1}
    assert set(df["prompt_id"].unique()) == {"p0", "p1"}


# ---- content preservation -------------------------------------------

def test_all_story_words_appear_in_some_passage(simple_stories, tok):
    """No word is lost: the union of passage words equals the story words."""
    df = passageize(simple_stories, tok, passage_tokens=20, min_passage_tokens=0)
    for _, story in simple_stories.iterrows():
        story_words = " ".join(story["story"].split()).split()
        passages = df[df["story_id"] == story["story_id"]]["text"].tolist()
        passage_words = " ".join(passages).split()
        # The reassembled sequence must equal the original word sequence.
        assert passage_words == story_words


# ---- tail filtering -------------------------------------------------

def test_short_tail_dropped_by_default(tok):
    """A story whose token count is not a multiple of passage_tokens
    will produce a final short passage; with default min = pt//2 it
    is dropped if shorter than that.
    """
    story = pd.DataFrame({
        "story_id": [0],
        "prompt": ["p"],
        "story": [" ".join(f"w{i}" for i in range(15))],  # 15 tokens
    })
    # passage_tokens=10 -> first chunk ~10 tokens, tail ~5 tokens.
    # Default min_passage_tokens = 5, so the tail is *kept*.
    df_kept = passageize(story, tok, passage_tokens=10)
    assert len(df_kept) == 2

    # With min=6, the 5-token tail is dropped.
    df_dropped = passageize(story, tok, passage_tokens=10, min_passage_tokens=6)
    assert len(df_dropped) == 1
    assert df_dropped.iloc[0]["n_tokens"] >= 6


def test_min_zero_keeps_everything(tok):
    story = pd.DataFrame({
        "story_id": [0], "prompt": ["p"],
        "story": [" ".join(f"w{i}" for i in range(7))],
    })
    df = passageize(story, tok, passage_tokens=5, min_passage_tokens=0)
    # Should produce 2 passages (5 + 2), both kept.
    assert len(df) == 2


# ---- edge cases -----------------------------------------------------

def test_empty_stories_df(tok):
    df = passageize(pd.DataFrame({"story_id": [], "prompt": [], "story": []}),
                    tok, passage_tokens=10)
    assert len(df) == 0
    assert set(df.columns) == {
        "passage_id", "story_id", "prompt_id",
        "text", "n_tokens", "position",
    }


def test_empty_story_text_produces_no_passages(tok):
    df = passageize(
        pd.DataFrame({"story_id": [0], "prompt": ["p"], "story": [""]}),
        tok, passage_tokens=10,
    )
    assert len(df) == 0


def test_whitespace_only_story_produces_no_passages(tok):
    df = passageize(
        pd.DataFrame({"story_id": [0], "prompt": ["p"], "story": ["   \n  "]}),
        tok, passage_tokens=10,
    )
    assert len(df) == 0


class CharTokenizer:
    """Mock budget tokenizer: one token per character.

    Models a BPE tokenizer on an adversarial input (a long space-free
    string becomes many tokens). Used to exercise the pathological
    branch of passageize that RegexTokenizer cannot reach -- under
    RegexTokenizer, any single whitespace-delimited word is exactly one
    ``\\w+`` run, so its count is always 1 and the branch is dead.
    """

    name = "char_test"
    def count(self, text):
        return len(text)
    def count_many(self, texts):
        return np.fromiter((len(t) for t in texts), dtype=np.int64, count=len(texts))
    def fingerprint(self):
        return "char_test_0000"


def test_single_huge_word_kept_with_warning():
    """A word whose token count exceeds passage_tokens goes alone,
    with a warning. Requires a tokenizer where one word can be many
    tokens (BPE-like); RegexTokenizer cannot trigger this branch.
    """
    char_tok = CharTokenizer()
    story = pd.DataFrame({
        "story_id": [0], "prompt": ["p"],
        # Words: "aa", "bb", then a 60-char monster, then "cc", "dd"
        "story": ["aa bb " + "x" * 60 + " cc dd"],
    })
    with pytest.warns(RuntimeWarning, match="single word exceeds"):
        df = passageize(
            story, char_tok, passage_tokens=10, min_passage_tokens=0
        )
    # The monster word is its own passage with n_tokens == 60.
    huge = df[df["n_tokens"] > 10]
    assert len(huge) == 1
    assert huge.iloc[0]["text"] == "x" * 60
    assert int(huge.iloc[0]["n_tokens"]) == 60
    # And no passage exceeds 10 tokens except the pathological one.
    normal = df[df["n_tokens"] <= 10]
    assert (normal["n_tokens"] > 0).all()


def test_regex_tokenizer_cannot_trigger_huge_word(tok):
    """Sanity: under RegexTokenizer a single space-free word is one
    ``\\w+`` run, so the huge-word branch is unreachable. This is why
    the pathological test above needs a different tokenizer.
    """
    long_word = "w" * 500  # 500 chars, one word
    assert tok.count(long_word) == 1


def test_huge_word_branch_not_reachable_for_regex(tok):
    """End-to-end: with RegexTokenizer a long word never produces a
    passage whose n_tokens exceeds passage_tokens, because one word is
    one token.
    """
    story = pd.DataFrame({
        "story_id": [0], "prompt": ["p"],
        "story": ["w0 w1 " + "x" * 500 + " w2 w3"],
    })
    df = passageize(story, tok, passage_tokens=10, min_passage_tokens=0)
    # No passage has n_tokens > 10.
    assert (df["n_tokens"] <= 10).all()


# ---- input validation ----------------------------------------------

def test_passage_tokens_must_be_positive(tok):
    df = pd.DataFrame({"story_id": [], "prompt": [], "story": []})
    with pytest.raises(ValueError, match="passage_tokens must be >= 1"):
        passageize(df, tok, passage_tokens=0)


def test_min_passage_tokens_negative_rejected(tok):
    df = pd.DataFrame({"story_id": [], "prompt": [], "story": []})
    with pytest.raises(ValueError, match="min_passage_tokens must be >= 0"):
        passageize(df, tok, passage_tokens=10, min_passage_tokens=-1)


def test_missing_story_col_raises(tok):
    df = pd.DataFrame({"story_id": [0], "prompt": ["p"]})
    with pytest.raises(KeyError, match="story"):
        passageize(df, tok, passage_tokens=10)


def test_missing_id_col_raises(tok):
    df = pd.DataFrame({"story": ["x"], "prompt": ["p"]})
    with pytest.raises(KeyError, match="story_id"):
        passageize(df, tok, passage_tokens=10)


# ---- without prompt column ------------------------------------------

def test_no_prompt_col_sets_prompt_id_none(tok):
    df = pd.DataFrame({
        "story_id": [0],
        "story": ["one two three four five six seven eight"],
    })
    out = passageize(df, tok, passage_tokens=4, min_passage_tokens=0)
    assert (out["prompt_id"].isna()).all()


# ---- determinism ----------------------------------------------------

def test_deterministic(simple_stories, tok):
    a = passageize(simple_stories, tok, passage_tokens=20)
    b = passageize(simple_stories, tok, passage_tokens=20)
    pd.testing.assert_frame_equal(a, b)


# ---- larger passage_tokens -> fewer, longer passages ---------------

def test_larger_passage_tokens_yields_fewer_passages(simple_stories, tok):
    small = passageize(simple_stories, tok, passage_tokens=10, min_passage_tokens=0)
    large = passageize(simple_stories, tok, passage_tokens=40, min_passage_tokens=0)
    assert len(small) > len(large)