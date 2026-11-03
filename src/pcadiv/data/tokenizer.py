"""Budget tokenizer family: base contract, registry, and implementations.

What "budget tokenizer" means here
----------------------------------
The token budget is the hard limit on how much selected text can be fed
to the target LLM. Its unit is **tokens of the target tokenizer** --
NOT tokens of the selection encoder, and NOT characters. This family
exists so that "price of a passage in budget units" is a first-class,
swappable concept.

This is why ``RegexTokenizer`` is explicitly a *proxy* for offline
tests only. Using it in final experiments would make diversity-vs-budget
curves incomparable to published results.

Design notes
------------
* Same contract-first, registry-based pattern as encoders.
* ``count_many`` returns ``np.int64`` array, so it feeds directly into
  selectors and cumsum without casting noise.
* Heavy imports (``tiktoken``, ``transformers``) are lazy: importing
  this module must stay cheap. ``HFTokenizerFromModel`` is a registered
  stub until Block 7.
"""
from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from typing import ClassVar, Sequence

import numpy as np

__all__ = [
    "BaseBudgetTokenizer",
    "RegexTokenizer",
    "TiktokenTokenizer",
    "HFTokenizerFromModel",
    "register_tokenizer",
    "get_tokenizer",
    "list_tokenizers",
]


# ---------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------

_TOKENIZERS: dict[str, type["BaseBudgetTokenizer"]] = {}


def register_tokenizer(
    cls: type["BaseBudgetTokenizer"],
) -> type["BaseBudgetTokenizer"]:
    """Register a tokenizer class by its ``name`` attribute."""
    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name:
        raise ValueError(f"{cls.__name__}.name must be a non-empty string")
    if name in _TOKENIZERS and _TOKENIZERS[name] is not cls:
        raise ValueError(
            f"tokenizer name {name!r} already registered by "
            f"{_TOKENIZERS[name].__name__}"
        )
    _TOKENIZERS[name] = cls
    return cls


def get_tokenizer(name: str, **cfg) -> "BaseBudgetTokenizer":
    """Instantiate a registered tokenizer by name."""
    if name not in _TOKENIZERS:
        raise KeyError(
            f"unknown tokenizer {name!r}; available: {sorted(_TOKENIZERS)}"
        )
    return _TOKENIZERS[name](**cfg)


def list_tokenizers() -> list[str]:
    """Return registered tokenizer names, sorted."""
    return sorted(_TOKENIZERS)


# ---------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------

class BaseBudgetTokenizer(ABC):
    """Abstract base for budget tokenizers.

    Subclasses must set ``name: ClassVar[str]`` and implement
    :meth:`count`. The default :meth:`count_many` loops over
    :meth:`count`; override it when the backend has a batch API.
    """

    name: ClassVar[str]

    @abstractmethod
    def count(self, text: str) -> int:
        """Return the number of budget tokens in ``text``."""

    def count_many(self, texts: Sequence[str]) -> np.ndarray:
        """Return token counts for a sequence of texts as ``int64`` array.

        Default implementation calls :meth:`count` per item. Subclasses
        with a native batch API should override this for speed.
        """
        return np.fromiter(
            (self.count(t) for t in texts),
            dtype=np.int64,
            count=len(texts),
        )

    def fingerprint(self) -> str:
        """Short stable hash of the tokenizer configuration.

        Used as part of cache keys for passages and pool token counts.
        """
        key = f"{self.name}|{self._fingerprint_key()}"
        return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]

    def _fingerprint_key(self) -> str:
        """Backend-specific config string. Default: empty."""
        return ""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ---------------------------------------------------------------------
# RegexTokenizer (offline proxy -- tests only)
# ---------------------------------------------------------------------

@register_tokenizer
class RegexTokenizer(BaseBudgetTokenizer):
    """Offline word-count proxy for tests.

    Counts ``\\w+`` runs (letters, digits, underscore) under Unicode
    rules. Punctuation and whitespace do not count. This is **not** a
    subword tokenizer and must NOT be used for real budget accounting
    in experiments -- only as a fast, deterministic stand-in in tests.
    """

    name = "regex"

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(re.findall(r"\w+", text, flags=re.UNICODE))


# ---------------------------------------------------------------------
# TiktokenTokenizer (lazy import)
# ---------------------------------------------------------------------

@register_tokenizer
class TiktokenTokenizer(BaseBudgetTokenizer):
    """Tiktoken-backed budget tokenizer.

    ``encoding`` must be specified explicitly (e.g. ``"cl100k_base"``);
    there is no sensible default because the correct encoding depends on
    the target LLM.

    Requires the ``tiktoken`` package (``pip install -e ".[data]"``).
    """

    name = "tiktoken"

    def __init__(self, encoding: str) -> None:
        if not encoding:
            raise ValueError("encoding must be a non-empty string")
        try:
            import tiktoken  # noqa: F401  (lazy, optional dep)
        except ImportError as e:  # pragma: no cover -- exercised via skip
            raise ImportError(
                "TiktokenTokenizer requires the 'tiktoken' package. "
                "Install with: pip install -e \".[data]\""
            ) from e
        import tiktoken

        self.encoding_name = encoding
        self._enc = tiktoken.get_encoding(encoding)

    def _fingerprint_key(self) -> str:
        return f"enc={self.encoding_name}"

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._enc.encode(text))

    def count_many(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.empty(0, dtype=np.int64)
        # encode_batch exists in recent tiktoken; fall back if absent.
        if hasattr(self._enc, "encode_batch"):
            batch = self._enc.encode_batch(list(texts))
            return np.fromiter(
                (len(toks) for toks in batch), dtype=np.int64, count=len(texts)
            )
        return super().count_many(texts)


# ---------------------------------------------------------------------
# HFTokenizerFromModel (stub until Block 7)
# ---------------------------------------------------------------------

@register_tokenizer
class HFTokenizerFromModel(BaseBudgetTokenizer):
    """Budget tokenizer using an HF model's tokenizer.

    **Not implemented yet** -- scheduled for Block 7 alongside HF
    encoders. Registered now so configs can reference ``name: hf``
    without breaking, and so the test suite can assert the stub raises
    a clear error rather than silently doing the wrong thing.
    """

    name = "hf"

    def __init__(
        self, model_id: str, add_special_tokens: bool = False
    ) -> None:
        raise NotImplementedError(
            "HFTokenizerFromModel is a stub; implemented in Block 7."
        )

    def count(self, text: str) -> int:  # pragma: no cover
        raise NotImplementedError