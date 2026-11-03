"""Metric family: base contract, registry, and MeanPairwiseDistance.

Two principles baked in
------------------------
1. **Input float32, compute float64.** Metrics receive embeddings as
   ``float32`` (that is what encoders produce and what is stored on
   disk), but any accumulation -- sums, log-determinants, kernel
   eigenvalues -- happens in ``float64``. On 768-d embeddings the
   float32 accumulation error is not theoretical; it silently biases
   ``logdet`` and Vendi toward zero.

2. **Metrics do not know about the budget.** A metric is a pure
   function of an embedding matrix (and optionally an anchor). The
   diversity-vs-budget curve is built by slicing a
   :class:`~pcadiv.selection.SelectionResult` prefix by prefix and
   calling the same metric on each slice -- see
   ``evaluation/budget_curve.py``. This keeps metric code free of any
   awareness of selection methods, and makes every metric reusable on
   arbitrary subsets (e.g. for ablations on random subsets).

Why ``core_emb`` is in the base signature
-----------------------------------------
Same reason as for selectors: some future metrics may want to score
"diversity relative to an anchor" (e.g. centrality-aware coverage).
The base signature accepts it; every metric must call
_validate_inputs(emb, core_emb) at the top of __call__, so shape
mismatches are caught identically across all metrics (a config bug
that only one metric complains about is worse than one that all
metrics complain about). Metrics that do not use the anchor's values
simply discard the second return value: X, _ = self._validate_inputs(emb, core_emb).

Output shape
------------
Every metric returns ``dict[str, float]``, never a bare float. Some
metrics naturally produce several numbers (coverage returns both a
radius and a log-determinant). Uniform ``dict`` output keeps
``budget_curve`` and ``stats`` from branching on metric type.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

__all__ = [
    "BaseMetric",
    "MeanPairwiseDistance",
    "register_metric",
    "get_metric",
    "list_metrics",
]


# ---------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------

_METRICS: dict[str, type["BaseMetric"]] = {}


def register_metric(cls: type["BaseMetric"]) -> type["BaseMetric"]:
    """Register a metric class by its ``name`` attribute."""
    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name:
        raise ValueError(f"{cls.__name__}.name must be a non-empty string")
    if name in _METRICS and _METRICS[name] is not cls:
        raise ValueError(
            f"metric name {name!r} already registered by "
            f"{_METRICS[name].__name__}"
        )
    _METRICS[name] = cls
    return cls


def get_metric(name: str, **cfg) -> "BaseMetric":
    """Instantiate a registered metric by name."""
    if name not in _METRICS:
        raise KeyError(
            f"unknown metric {name!r}; available: {sorted(_METRICS)}"
        )
    return _METRICS[name](**cfg)


def list_metrics() -> list[str]:
    """Return registered metric names, sorted."""
    return sorted(_METRICS)


# ---------------------------------------------------------------------
# Pairwise distance kernels (shared helpers)
# ---------------------------------------------------------------------

def _pairwise_euclidean(X: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix for rows of ``X`` (float64, N×N).

    Uses ``||x||² + ||y||² − 2⟨x,y⟩`` rather than direct broadcasting:
    broadcasting materialises an ``N×N×D`` tensor (19 GB at N=5000,
    D=768), while this formulation needs only ``N×N``. The tiny
    negative values on the diagonal due to floating-point cancellation
    are clamped to zero.
    """
    sq = np.einsum("ij,ij->i", X, X)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(D2, 0.0, out=D2)
    np.fill_diagonal(D2, 0.0)
    return np.sqrt(D2)


def _pairwise_cosine(X: np.ndarray) -> np.ndarray:
    """Cosine distance matrix (``1 - cos_sim``) for rows of ``X``.

    Zero-norm rows are treated as maximally distant from everything
    (including each other), i.e. their cosine distance is fixed at 1.
    This is a deliberate convention -- a zero vector has no direction,
    so "similarity" to it is undefined; treating it as maximally
    dissimilar is the conservative default and keeps downstream metrics
    finite.
    """
    norms = np.linalg.norm(X, axis=1)
    denom = np.outer(norms, norms)
    safe_denom = np.where(denom > 0.0, denom, 1.0)
    sim = (X @ X.T) / safe_denom
    np.clip(sim, -1.0, 1.0, out=sim)
    D = 1.0 - sim
    zero = norms == 0.0
    if zero.any():
        D[zero, :] = 1.0
        D[:, zero] = 1.0
        np.fill_diagonal(D, 0.0)
    return D


_KERNELS = {
    "euclidean": _pairwise_euclidean,
    "cosine": _pairwise_cosine,
}


def pairwise_distances(X: np.ndarray, metric: str) -> np.ndarray:
    """Return the ``N×N`` distance matrix for ``X`` under ``metric``.

    ``X`` must already be ``float64``; this function does not upcast.
    Supported metrics: ``"euclidean"``, ``"cosine"``.
    """
    if metric not in _KERNELS:
        raise ValueError(
            f"unknown metric {metric!r}; supported: {sorted(_KERNELS)}"
        )
    return _KERNELS[metric](X)


# ---------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------

class BaseMetric(ABC):
    """Abstract base for all metrics.

    Subclasses set ``name: ClassVar[str]`` and implement :meth:`__call__`.
    """

    name: ClassVar[str]

    def __init__(self, **cfg) -> None:
        self.cfg = dict(cfg)

    @abstractmethod
    def __call__(
        self,
        emb: np.ndarray,
        *,
        core_emb: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Compute the metric on a set of embeddings.

        Parameters
        ----------
        emb
            ``float32`` or ``float64`` array of shape ``(N, D)``.
        core_emb
            Optional anchor vector of shape ``(D,)``. Ignored by
            metrics that do not need one; must be ignored explicitly
            (``del core_emb``) so the omission is visible.

        Returns
        -------
        dict[str, float]
            One or more named scalar values. Names are stable across
            calls and across subsets of the same embedding space.
        """

    # -- helpers ------------------------------------------------------

    @staticmethod
    def _validate_inputs(
        emb: np.ndarray,
        core_emb: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Validate inputs and upcast to float64.

        Returns the embeddings as ``float64`` (a no-copy view when the
        input is already ``float64``; otherwise a copy) and the anchor
        as ``float64`` 1-D or ``None``.
        """
        X = np.asarray(emb)
        if X.ndim != 2:
            raise ValueError(f"emb must be 2-D, got ndim={X.ndim}")
        if X.dtype != np.float64:
            X = X.astype(np.float64, copy=False)

        if core_emb is None:
            core: np.ndarray | None = None
        else:
            core = np.asarray(core_emb, dtype=np.float64)
            if core.ndim != 1:
                raise ValueError(
                    f"core_emb must be 1-D, got ndim={core.ndim}"
                )
            if core.shape[0] != X.shape[1]:
                raise ValueError(
                    f"core_emb dim mismatch: {core.shape[0]} vs {X.shape[1]}"
                )
        return X, core

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.cfg})"


# ---------------------------------------------------------------------
# MeanPairwiseDistance
# ---------------------------------------------------------------------

@register_metric
class MeanPairwiseDistance(BaseMetric):
    """Mean distance over all unordered pairs of embeddings.

    For ``N`` embeddings this is the average of the ``N(N-1)/2``
    pairwise distances under ``metric``. For ``N < 2`` the value is
    defined as ``0.0``: "no diversity" is the correct reading of a set
    with zero or one element, and this convention keeps
    diversity-vs-budget curves well-defined at their first prefix.

    Parameters
    ----------
    metric
        ``"euclidean"`` (default) or ``"cosine"``. Euclidean is the
        convention in most diversity-from-embeddings literature; cosine
        is provided because for L2-normalised embeddings euclidean is
        a monotone transform of cosine, and reviewers often ask for
        both.
    """

    name = "mean_pairwise"

    def __init__(self, metric: str = "euclidean") -> None:
        super().__init__(metric=metric)
        if metric not in _KERNELS:
            raise ValueError(
                f"unknown metric {metric!r}; supported: {sorted(_KERNELS)}"
            )
        self.metric = metric

    def __call__(
        self,
        emb: np.ndarray,
        *,
        core_emb: np.ndarray | None = None,
    ) -> dict[str, float]:
        # Validate both inputs for consistency across all metrics; the
        # anchor's values are not used here (see module docstring).
        X, _ = self._validate_inputs(emb, core_emb)
        n = X.shape[0]
        if n < 2:
            return {"mean_pairwise": 0.0}
        D = pairwise_distances(X, self.metric)
        iu = np.triu_indices(n, k=1)
        return {"mean_pairwise": float(D[iu].mean())}