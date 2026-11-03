"""Slice stories into fixed-token passages.

Why word-greedy, not token-ID windows
-------------------------------------
Two approaches for turning a story into passages of ``passage_tokens``
tokens:

1. **Token-ID windowing**: encode the whole story with the budget
   tokenizer, slice the resulting ID list into fixed-size windows,
   decode each window back to text. Exact on token counts but requires
   ``encode``/``decode`` on ``BaseBudgetTokenizer`` and can lose
   whitespace on byte-level BPE, so ``count(decoded) != slice_size``
   in edge cases.
2. **Word-greedy accumulation**: split into words, add words until the
   *count of the accumulated text* would exceed ``passage_tokens``;
   then finalise the passage. ``n_tokens = tokenizer.count(text)`` is
   then **exactly what is in the text**, not "how many we intended to
   cut". Works uniformly for ``RegexTokenizer``, ``TiktokenTokenizer``,
   ``HFTokenizerFromModel`` -- no interface changes.

We use (2). The cost is ``O(words²)`` ``count`` calls per story --
acceptable for WritingPrompts (~100-500 words per story, ~1 μs per
``count``). Passages can end up slightly under ``passage_tokens`` (a
passage of 250 when target is 256), which is fine: the hard budget
invariant ``n_tokens <= passage_tokens`` is what matters downstream,
and a uniformly tighter pack is not worth the interface surface.

Pathological case
-----------------
A single word whose token count exceeds ``passage_tokens`` on its own
(e.g. a long URL, a chemical formula) cannot be split further without
changing the text content. Such words are kept as their own passage
with ``n_tokens > passage_tokens``, and a ``RuntimeWarning`` is
emitted once per call. The selection stage filters by budget anyway,
so over-budget passages cannot enter a selection.

Tail filtering
--------------
The trailing passage of a story is often much shorter than
``passage_tokens``. Passages with ``n_tokens < min_passage_tokens``
are dropped. Default ``min_passage_tokens = passage_tokens // 2``
(so a 256-token target drops tails shorter than 128 tokens). Set
``min_passage_tokens=0`` to keep everything.
"""
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd

from pcadiv.data.tokenizer import BaseBudgetTokenizer

__all__ = ["passageize"]


def _slice_story(
    text: str,
    tokenizer: BaseBudgetTokenizer,
    passage_tokens: int,
    min_passage_tokens: int,
) -> list[tuple[str, int]]:
    """Slice one story into ``(passage_text, n_tokens)`` pairs.

    Greedy word accumulation. Returns empty list for an empty story.
    """
    words = text.split()
    if not words:
        return []

    passages: list[tuple[str, int]] = []
    acc: list[str] = []
    acc_count = 0

    def _finalise():
        nonlocal acc, acc_count
        if not acc:
            return
        passage_text = " ".join(acc)
        n = tokenizer.count(passage_text)
        if n >= min_passage_tokens and n > 0:
            passages.append((passage_text, int(n)))
        acc = []
        acc_count = 0

    for word in words:
        candidate_text = word if not acc else " ".join(acc) + " " + word
        candidate_count = tokenizer.count(candidate_text)
        if candidate_count <= passage_tokens:
            acc.append(word)
            acc_count = candidate_count
            continue

        # Adding this word would overflow. Finalise current passage,
        # then deal with this word on its own -- its own count, not the
        # count of (acc + word), which is what pushed us here.
        _finalise()

        word_count = tokenizer.count(word)
        if word_count <= passage_tokens:
            acc = [word]
            acc_count = word_count
            continue

        # Pathological: the word alone exceeds passage_tokens.
        warnings.warn(
            f"single word exceeds passage_tokens={passage_tokens} "
            f"(n_tokens={word_count}); keeping it as its own passage",
            RuntimeWarning,
            stacklevel=3,
        )
        if word_count >= min_passage_tokens:
            passages.append((word, int(word_count)))
        # acc stays empty; loop continues with the next word

    _finalise()
    return passages


def passageize(
    stories: pd.DataFrame,
    tokenizer: BaseBudgetTokenizer,
    passage_tokens: int,
    min_passage_tokens: int | None = None,
    *,
    id_col: str = "story_id",
    text_col: str = "story",
    prompt_col: str = "prompt",
) -> pd.DataFrame:
    """Slice stories into fixed-token passages.

    Parameters
    ----------
    stories
        DataFrame with at least ``id_col``, ``text_col``. ``prompt_col``
        is carried through if present, dropped silently if absent.
    tokenizer
        Budget tokenizer. Its ``count`` defines ``n_tokens``.
    passage_tokens
        **Target** number of tokens per passage. Required; there is no
        default on purpose -- the "right" value is an experimental
        choice and should be made explicitly in the config.
    min_passage_tokens
        Drop passages shorter than this. ``None`` means
        ``passage_tokens // 2``. Set to ``0`` to keep everything.
    id_col, text_col, prompt_col
        Column names in the input DataFrame.

    Returns
    -------
    pandas.DataFrame
        Columns: ``passage_id`` (str ``"<story_id>::<position>"``),
        ``story_id``, ``prompt_id``, ``text``, ``n_tokens`` (int64),
        ``position`` (int64, 0-based within the story). Prompt text is
        renamed to ``prompt_id`` only -- the actual prompt string is
        kept on the story and joined later in ``pools.py``; storing it
        on every passage would multiply memory by passages/story.
    """
    if passage_tokens < 1:
        raise ValueError(f"passage_tokens must be >= 1, got {passage_tokens}")
    if min_passage_tokens is None:
        min_passage_tokens = passage_tokens // 2
    if min_passage_tokens < 0:
        raise ValueError(
            f"min_passage_tokens must be >= 0, got {min_passage_tokens}"
        )

    missing = {id_col, text_col} - set(stories.columns)
    if missing:
        raise KeyError(f"stories is missing columns: {sorted(missing)}")
    has_prompt = prompt_col in stories.columns

    rows: list[dict] = []
    for _, story in stories.iterrows():
        story_id = int(story[id_col])
        text = "" if pd.isna(story[text_col]) else str(story[text_col])
        slices = _slice_story(text, tokenizer, passage_tokens, min_passage_tokens)
        for pos, (passage_text, n) in enumerate(slices):
            rows.append({
                "passage_id": f"{story_id}::{pos}",
                "story_id": story_id,
                "prompt_id": story[prompt_col] if has_prompt else None,
                "text": passage_text,
                "n_tokens": int(n),
                "position": pos,
            })

    if not rows:
        return pd.DataFrame(
            columns=[
                "passage_id", "story_id", "prompt_id",
                "text", "n_tokens", "position",
            ]
        ).astype({"n_tokens": "int64", "position": "int64"})

    out = pd.DataFrame(rows)
    out["n_tokens"] = out["n_tokens"].astype("int64")
    out["position"] = out["position"].astype("int64")
    return out