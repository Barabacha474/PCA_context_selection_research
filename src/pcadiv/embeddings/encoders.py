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

# ---------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------

def _resolve_device(device: str) -> str:
    """Resolve ``"auto"`` to an available torch device; pass others through."""
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------
# HF common helpers
# ---------------------------------------------------------------------

def _load_hf_model_and_tokenizer(model_id: str, device: str):
    """Load HF model + tokenizer with lazy import."""
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "HF encoders require 'transformers' and 'torch'. "
            "Install with: pip install -e \".[encoders]\""
        ) from e
    model = AutoModel.from_pretrained(model_id)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


def _extract_last_hidden_state(output):
    """Extract ``last_hidden_state`` from a transformers output object.

    Handles both the modern ``ModelOutput`` dataclass (with
    ``.last_hidden_state``) and the legacy tuple form. Raises a clear
    error if neither is present, rather than silently returning None.
    """
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, (tuple, list)) and len(output) > 0:
        return output[0]
    raise RuntimeError(
        f"model output has no last_hidden_state and is not a tuple: "
        f"{type(output).__name__}"
    )


def _get_hf_hidden_dim(model) -> int:
    """Return the hidden dimension of an HF model."""
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "hidden_size"):
        return int(cfg.hidden_size)
    emb = model.get_input_embeddings()
    return int(emb.weight.shape[1])


# ---------------------------------------------------------------------
# SentenceTransformerEncoder
# ---------------------------------------------------------------------

@register_encoder
class SentenceTransformerEncoder(BaseEncoder):
    """Encoder backed by :mod:`sentence_transformers`.

    Pooling and per-model normalization are handled inside the ST
    library. We disable ST's own normalization and apply ours via
    :meth:`BaseEncoder._maybe_normalize`, so that every encoder in the
    project normalizes identically -- otherwise ``st`` and ``hf_mean``
    would silently disagree about what "normalized" means.

    Requires: ``pip install -e ".[encoders]"``.
    """

    name = "st"

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        normalize: bool = True,
    ) -> None:
        super().__init__(
            model_id=model_id,
            device=_resolve_device(device),
            normalize=normalize,
        )
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "SentenceTransformerEncoder requires 'sentence-transformers'. "
                "Install with: pip install -e \".[encoders]\""
            ) from e
        self._model = SentenceTransformer(self.model_id, device=self.device)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def dim(self) -> int:
        return self._dim

    def encode(
        self, texts: Sequence[str], batch_size: int = 64
    ) -> np.ndarray:
        if len(texts) == 0:
            return np.empty((0, self._dim), dtype=np.float32)
        X = self._model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        return self._maybe_normalize(np.asarray(X, dtype=np.float32))


# ---------------------------------------------------------------------
# HF encoder base + concrete variants
# ---------------------------------------------------------------------

class _HFEncoderBase(BaseEncoder):
    """Shared machinery for HF encoder variants (mean / CLS pooling).

    Handles lazy imports, model/tokenizer loading, batching, and the
    output extraction dance. Subclasses only implement :meth:`_pool`.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        normalize: bool = True,
    ) -> None:
        super().__init__(
            model_id=model_id,
            device=_resolve_device(device),
            normalize=normalize,
        )
        try:
            import torch
        except ImportError as e:
            raise ImportError(
                "HF encoders require 'torch'. "
                "Install with: pip install -e \".[encoders]\""
            ) from e
        self._torch = torch
        self._model, self._tokenizer = _load_hf_model_and_tokenizer(
            self.model_id, self.device
        )
        self._dim = _get_hf_hidden_dim(self._model)

    @property
    def dim(self) -> int:
        return self._dim

    def _pool(self, last_hidden_state, attention_mask):
        raise NotImplementedError

    def encode(
        self, texts: Sequence[str], batch_size: int = 64
    ) -> np.ndarray:
        if len(texts) == 0:
            return np.empty((0, self._dim), dtype=np.float32)
        torch = self._torch
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = list(texts[start:start + batch_size])
                enc = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in enc.items()}
                out = self._model(**inputs)
                hidden = _extract_last_hidden_state(out)
                pooled = self._pool(hidden, inputs["attention_mask"])
                chunks.append(pooled.detach().cpu().numpy())
        X = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
        return self._maybe_normalize(X)


@register_encoder
class HFMeanPoolEncoder(_HFEncoderBase):
    """Mean-pooled HF encoder.

    Mean of ``last_hidden_state`` over non-padding tokens, then L2-norm.
    Works for encoder models (BERT family) and for decoder models
    (Llama, Mistral, etc.) where mean pooling over content tokens is a
    common embedding-from-LLM convention.
    """

    name = "hf_mean"

    def _pool(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts


@register_encoder
class HFCLSEncoder(_HFEncoderBase):
    """CLS-token HF encoder.

    Takes ``last_hidden_state[:, 0]``. For BERT-family encoder models
    this is the ``[CLS]`` token, trained to be meaningful. For
    decoder-only models (Llama, Mistral) the "first token" is the BOS
    token, which is typically not tuned for sentence embedding -- use
    :class:`HFMeanPoolEncoder` for those instead.
    """

    name = "hf_cls"

    def _pool(self, last_hidden_state, attention_mask):
        del attention_mask  # CLS pooling ignores the mask
        return last_hidden_state[:, 0]