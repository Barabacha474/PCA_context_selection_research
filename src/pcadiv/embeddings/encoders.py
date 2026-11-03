"""Encoder family: base contract, registry, and concrete implementations.

Design notes
------------
* **Contract first.** All encoders subclass :class:`BaseEncoder` and
  implement ``encode(texts) -> np.ndarray[N, dim]``. Nothing else is
  required to plug a new encoder into the pipeline -- just decorate the
  class with :func:`register_encoder` and give it a unique ``name``.

* **Lazy heavy imports.** Concrete encoders that need ``torch``,
  ``transformers`` or ``sentence_transformers`` must import those
  libraries **inside** ``__init__``/``encode``, not at module top. This
  keeps ``import pcadiv.embeddings.encoders`` cheap and lets tests run
  without the heavy stack installed.

* **Fingerprint for cache keys.** :meth:`BaseEncoder.fingerprint` returns
  a short stable hash of ``(name, model_id, normalize, dim)``. The
  embedding cache (Block 10) keys on this -- if you change any of those
  four fields, the cache **must** invalidate. Getting this wrong is the
  classic silent-corruption bug in retrieval pipelines.

* **``HashingEncoder`` is the workhorse for offline tests.** It is
  deterministic, dependency-free, and returns an L2-normalised
  bag-of-hashed-tokens embedding. Dimension is configurable. It is not
  a good semantic encoder -- that is not its job -- but it exercises
  every downstream contract without network access or model downloads.
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import ClassVar, Sequence

import numpy as np

__all__ = [
    "BaseEncoder",
    "HashingEncoder",
    "register_encoder",
    "get_encoder",
    "list_encoders",
]


# ---------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------

_ENCODERS: dict[str, type["BaseEncoder"]] = {}


def register_encoder(cls: type["BaseEncoder"]) -> type["BaseEncoder"]:
    """Register an encoder class by its ``name`` attribute.

    Raises
    ------
    ValueError
        If the class lacks ``name``, or a different class is already
        registered under the same name. Silent overwrite would hide
        typos like two classes both named ``"st"``.
    """
    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name:
        raise ValueError(f"{cls.__name__}.name must be a non-empty string")
    if name in _ENCODERS and _ENCODERS[name] is not cls:
        raise ValueError(
            f"encoder name {name!r} already registered by "
            f"{_ENCODERS[name].__name__}"
        )
    _ENCODERS[name] = cls
    return cls


def get_encoder(name: str, **cfg) -> "BaseEncoder":
    """Instantiate a registered encoder by name, forwarding ``**cfg``."""
    if name not in _ENCODERS:
        raise KeyError(
            f"unknown encoder {name!r}; available: {sorted(_ENCODERS)}"
        )
    return _ENCODERS[name](**cfg)


def list_encoders() -> list[str]:
    """Return registered encoder names, sorted."""
    return sorted(_ENCODERS)


# ---------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------

class BaseEncoder(ABC):
    """Abstract base for all encoders.

    Subclasses must set ``name: ClassVar[str]`` and implement
    :meth:`encode` and the :attr:`dim` property.
    """

    name: ClassVar[str]

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        normalize: bool = True,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.normalize = normalize

    @property
    @abstractmethod
    def dim(self) -> int:
        """Output embedding dimension."""

    @abstractmethod
    def encode(
        self, texts: Sequence[str], batch_size: int = 64
    ) -> np.ndarray:
        """Encode ``texts`` into a float32 array of shape ``[len(texts), dim]``.

        If ``self.normalize`` is ``True``, rows are L2-normalised.
        """

    def fingerprint(self) -> str:
        """Short stable hash of the encoder configuration.

        Used as part of the embedding-cache key. Includes ``name``,
        ``model_id``, ``normalize``, and ``dim`` -- changing any of
        them must invalidate cached embeddings.
        """
        key = f"{self.name}|{self.model_id}|{self.normalize}|{self.dim}"
        return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]

    # -- convenience ---------------------------------------------------

    def _maybe_normalize(self, X: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return X
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        # Avoid division by zero: a zero row stays zero.
        return X / np.where(norms > 0, norms, 1.0)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(model_id={self.model_id!r}, "
            f"normalize={self.normalize}, dim={self.dim})"
        )


# ---------------------------------------------------------------------
# Concrete: HashingEncoder (offline, deterministic)
# ---------------------------------------------------------------------

@register_encoder
class HashingEncoder(BaseEncoder):
    """Deterministic bag-of-hashed-tokens encoder.

    Each token is hashed into one of ``dim`` buckets; the resulting
    count vector is L2-normalised. This is *not* a semantic encoder --
    it is a fast, offline, dependency-free stand-in that exercises the
    rest of the pipeline (selectors, metrics, cache) without downloading
    models. Its embeddings **are** deterministic across runs and
    platforms (SHA-1 based).
    """

    name = "hashing"

    def __init__(
        self,
        model_id: str = "hashed-words-v1",
        dim: int = 256,
        normalize: bool = True,
        tokenizer: str = "whitespace",
    ) -> None:
        super().__init__(model_id=model_id, device="cpu", normalize=normalize)
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if tokenizer != "whitespace":
            raise ValueError(
                f"HashingEncoder only supports tokenizer='whitespace', "
                f"got {tokenizer!r}"
            )
        self._dim = dim
        self._tokenizer = tokenizer

    @property
    def dim(self) -> int:
        return self._dim

    def _bucket(self, token: str) -> int:
        h = hashlib.sha1(token.encode("utf-8")).digest()
        # Take 8 bytes as unsigned int, mod dim.
        return int.from_bytes(h[:8], "big") % self._dim

    def encode(
        self, texts: Sequence[str], batch_size: int = 64
    ) -> np.ndarray:
        # batch_size is accepted for API symmetry but has no effect here.
        del batch_size
        n = len(texts)
        X = np.zeros((n, self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in t.split():
                X[i, self._bucket(tok)] += 1.0
        return self._maybe_normalize(X)