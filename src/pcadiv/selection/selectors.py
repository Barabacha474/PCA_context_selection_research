"""Selector family: base contract, registry, and RandomSelector.

Contract in one paragraph
-------------------------
A selector receives a pool of embeddings, their token costs, and a hard
token budget, and returns a :class:`SelectionResult` -- an **ordered**
sequence of pool indices with precomputed cumulative token costs. It is a
pure function of its inputs and ``seed``: no state is carried between
calls, no global RNG is touched.

Why ordered, not a set
----------------------
Diversity-vs-budget curves are prefix slices of the selection. If we
returned a set, every metric would have to re-derive an ordering, and
every method would invent its own -- making curves incomparable across
methods. Returning an ordered list makes the budget curve a pure
slicing operation and keeps metrics orthogonal to the selection logic.

What is ``core_emb``
--------------------
``core_emb`` is an **anchor vector** in the selection-embedding space
(shape ``(D,)``), used to score "relevance to a reference" by those
selectors that need one. It has three legitimate instantiations, all
covered by the same field:

* **Centroid anchor** -- the centroid of the pool embeddings. Used in
  the primary sweep to build the "centrality / typicality" baseline.
  Tests the hypothesis "diverse beats typical under a fixed budget"
  without a confound from query choice.

* **Real query** -- an embedding of a held-out prompt (e.g. a Dungeon
  Master situation, a memory summary, or the leading part of a long
  prompt). Used in the secondary sweep for the deployment-relevant
  regime. Must be embedded by the **selection** encoder so that cosine
  similarity against the pool is meaningful.

* **``None``** -- anchor-free selection. Diversity-only methods
  (random, FPS, PCA-based) accept and ignore ``core_emb``; selectors
  that *require* an anchor (relevance, MMR) must raise ``ValueError``
  when it is ``None``.

Some methods use ``core_emb`` as a **fixed** anchor; others (future
``mmr_dynamic``) additionally use the centroid of the currently
selected set as a moving reference. The base contract only fixes the
entry point, not how the anchor is composed.

Why ``core_emb`` is in the base signature
-----------------------------------------
Some selectors need an anchor to score relevance (relevance, MMR),
some only use it as an optional FPS seed, some ignore it entirely
(random, PCA-based). Putting it in the base signature keeps one
interface for all methods instead of forking the contract later.
Selectors that do not use it must explicitly ignore it; selectors that
require it must raise ``ValueError`` when it is ``None``.

Budget handling
---------------
``cum_tokens[-1] <= budget`` is a hard invariant, enforced by
:meth:`BaseSelector._pack`. Selectors only propose candidate indices;
packing (cumsum + uniqueness + budget check) is centralized. This is
deliberate: budget-off-by-one is the single most common bug in retrieval
pipelines, and one implementation means one place to test it.
"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, Sequence

import numpy as np

from pcadiv.utils.seeds import make_rng

__all__ = [
    "SelectionResult",
    "BaseSelector",
    "RandomSelector",
    "register_selector",
    "get_selector",
    "list_selectors",
]


# ---------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------

@dataclass
class SelectionResult:
    """Output of a selector.

    Attributes
    ----------
    indices
        1-D ``int64`` array of pool indices, in selection order. Unique.
    cum_tokens
        1-D ``int64`` array of the same length as ``indices``; entry ``i``
        is the cumulative token cost of ``indices[:i+1]``. Monotone
        non-decreasing. ``cum_tokens[-1] <= budget`` always holds.
    meta
        Free-form dict for diagnostics (``k_used``, ``eigenvalues``,
        ``n_candidates``, ``step_scores``, etc.). Never used for
        control flow.
    """

    indices: np.ndarray
    cum_tokens: np.ndarray
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.indices = np.asarray(self.indices, dtype=np.int64)
        self.cum_tokens = np.asarray(self.cum_tokens, dtype=np.int64)
        if self.indices.shape != self.cum_tokens.shape:
            raise ValueError(
                f"indices and cum_tokens shape mismatch: "
                f"{self.indices.shape} vs {self.cum_tokens.shape}"
            )
        if self.indices.ndim != 1:
            raise ValueError(f"indices must be 1-D, got ndim={self.indices.ndim}")

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    @property
    def total_tokens(self) -> int:
        return int(self.cum_tokens[-1]) if len(self) else 0


# ---------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------

_SELECTORS: dict[str, type["BaseSelector"]] = {}


def register_selector(cls: type["BaseSelector"]) -> type["BaseSelector"]:
    """Register a selector class by its ``name`` attribute."""
    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name:
        raise ValueError(f"{cls.__name__}.name must be a non-empty string")
    if name in _SELECTORS and _SELECTORS[name] is not cls:
        raise ValueError(
            f"selector name {name!r} already registered by "
            f"{_SELECTORS[name].__name__}"
        )
    _SELECTORS[name] = cls
    return cls


def get_selector(name: str, **cfg) -> "BaseSelector":
    """Instantiate a registered selector by name."""
    if name not in _SELECTORS:
        raise KeyError(
            f"unknown selector {name!r}; available: {sorted(_SELECTORS)}"
        )
    return _SELECTORS[name](**cfg)


def list_selectors() -> list[str]:
    """Return registered selector names, sorted."""
    return sorted(_SELECTORS)


# ---------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------

class BaseSelector(ABC):
    """Abstract base for all selectors.

    Subclasses set ``name: ClassVar[str]`` and implement
    :meth:`_select_indices`. They must not reimplement budget packing --
    use :meth:`_pack`.
    """

    name: ClassVar[str]

    def __init__(self, seed: int = 0, **cfg) -> None:
        self.seed = int(seed)
        self.cfg = dict(cfg)

    # -- abstract -----------------------------------------------------

    @abstractmethod
    def _select_indices(
        self,
        pool_emb: np.ndarray,
        pool_tokens: np.ndarray,
        budget: int,
        core_emb: np.ndarray | None,
    ) -> tuple[list[int], dict]:
        """Return ``(ordered_indices, meta)`` without enforcing the budget.

        Implementations may return more indices than fit; :meth:`_pack`
        will truncate to the budget. This keeps subclasses free of
        off-by-one logic.
        """

    # -- public API ---------------------------------------------------

    def select(
        self,
        pool_emb: np.ndarray,
        pool_tokens: np.ndarray,
        budget: int,
        core_emb: np.ndarray | None = None,
    ) -> SelectionResult:
        """Run selection under a hard token budget.

        Parameters
        ----------
        pool_emb
            ``float32`` array of shape ``(N, D)``.
        pool_tokens
            Integer array of shape ``(N,)`` -- per-passage token cost in
            **budget tokens**. Non-negative.
        budget
            Hard upper bound on ``sum(pool_tokens[indices])``.
        core_emb
            Optional anchor vector of shape ``(D,)`` in the same space
            as ``pool_emb``. May be a pool centroid (primary sweep), a
            real query embedding (secondary sweep), or ``None``.
            Selectors that require it must raise ``ValueError`` when it
            is ``None``; selectors that ignore it must do so explicitly.

        Returns
        -------
        SelectionResult
        """
        pool_emb, pool_tokens, core_emb = self._validate_inputs(
            pool_emb, pool_tokens, core_emb
        )
        if budget < 0:
            raise ValueError(f"budget must be >= 0, got {budget}")

        ordered, meta = self._select_indices(
            pool_emb, pool_tokens, int(budget), core_emb
        )
        return self._pack(ordered, pool_tokens, budget=int(budget), meta=meta)

    # -- helpers ------------------------------------------------------

    @staticmethod
    def _validate_inputs(
        pool_emb: np.ndarray,
        pool_tokens: np.ndarray,
        core_emb: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        emb = np.asarray(pool_emb)
        if emb.ndim != 2:
            raise ValueError(f"pool_emb must be 2-D, got ndim={emb.ndim}")
        toks = np.asarray(pool_tokens)
        if toks.ndim != 1:
            raise ValueError(f"pool_tokens must be 1-D, got ndim={toks.ndim}")
        if toks.shape[0] != emb.shape[0]:
            raise ValueError(
                f"pool_emb and pool_tokens length mismatch: "
                f"{emb.shape[0]} vs {toks.shape[0]}"
            )
        if not np.issubdtype(toks.dtype, np.integer):
            raise ValueError(
                f"pool_tokens must be integer dtype, got {toks.dtype}"
            )
        if toks.size and (toks < 0).any():
            raise ValueError("pool_tokens must be non-negative")

        core: np.ndarray | None
        if core_emb is None:
            core = None
        else:
            core = np.asarray(core_emb)
            if core.ndim != 1:
                raise ValueError(
                    f"core_emb must be 1-D, got ndim={core.ndim}"
                )
            if core.shape[0] != emb.shape[1]:
                raise ValueError(
                    f"core_emb dim mismatch: {core.shape[0]} vs {emb.shape[1]}"
                )
        return emb, toks, core

    @staticmethod
    def _pack(
        ordered: Sequence[int],
        pool_tokens: np.ndarray,
        budget: int,
        meta: dict,
    ) -> SelectionResult:
        """Enforce uniqueness and budget; compute ``cum_tokens``.

        Walks ``ordered`` in order, keeping an index only if it has not
        been seen and if adding it does not exceed ``budget``. An item
        that does not fit is skipped, not treated as a terminator,
        because a cheaper later item might fit.
        """
        seen: set[int] = set()
        kept: list[int] = []
        cum: list[int] = []
        running = 0

        for raw in ordered:
            i = int(raw)
            if i < 0 or i >= pool_tokens.shape[0]:
                raise IndexError(
                    f"selector returned out-of-range index {i} "
                    f"for pool of size {pool_tokens.shape[0]}"
                )
            if i in seen:
                continue
            cost = int(pool_tokens[i])
            if running + cost > budget:
                continue
            seen.add(i)
            kept.append(i)
            running += cost
            cum.append(running)

        return SelectionResult(
            indices=np.asarray(kept, dtype=np.int64),
            cum_tokens=np.asarray(cum, dtype=np.int64),
            meta=dict(meta),
        )


# ---------------------------------------------------------------------
# RandomSelector
# ---------------------------------------------------------------------

@register_selector
class RandomSelector(BaseSelector):
    """Uniform random subset of the pool, truncated to the budget.

    Order is the random permutation order, truncated when the budget is
    exhausted. ``core_emb`` is accepted for interface symmetry but
    ignored: random selection has no anchor.
    """

    name = "random"

    def _select_indices(
        self,
        pool_emb: np.ndarray,
        pool_tokens: np.ndarray,
        budget: int,
        core_emb: np.ndarray | None,
    ) -> tuple[list[int], dict]:
        del core_emb  # explicitly ignored
        n = pool_emb.shape[0]
        rng = make_rng(self.seed)
        perm = rng.permutation(n).tolist()
        meta = {"n_candidates": int(n), "budget": int(budget)}
        return perm, meta

# ---------------------------------------------------------------------
# Cosine helper (shared by Relevance and MMR)
# ---------------------------------------------------------------------

def _cosine_sim_to_anchor(X: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Cosine similarity of each row of ``X`` to a 1-D ``anchor``.

    * Works for unnormalized inputs -- this is cosine, not dot product.
    * Zero-norm rows and a zero-norm anchor yield similarity ``0.0``
      (no direction, no signal) rather than ``NaN``.
    * Non-finite values are also mapped to ``0.0`` so a single corrupt
      row cannot poison the whole selection.
    """
    X = np.asarray(X, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)
    x_norms = np.linalg.norm(X, axis=1)
    a_norm = float(np.linalg.norm(anchor))
    denom = x_norms * a_norm
    safe = np.where(denom > 0, denom, 1.0)
    sim = (X @ anchor) / safe
    np.clip(sim, -1.0, 1.0, out=sim)
    sim = np.where(denom > 0, sim, 0.0)
    sim = np.where(np.isfinite(sim), sim, 0.0)
    return sim


# ---------------------------------------------------------------------
# RelevanceSelector
# ---------------------------------------------------------------------

@register_selector
class RelevanceSelector(BaseSelector):
    """Top-``k`` by cosine similarity to an anchor vector.

    In the **primary sweep** this is the *centrality* baseline: the
    anchor is the pool centroid, so the method selects the most typical
    passages -- the opposite of diversity. In the **secondary sweep**
    the anchor is a real query embedding, so the method becomes
    classic top-``k`` relevance retrieval.

    Requires ``core_emb``: raises ``ValueError`` when it is ``None``.
    Order is deterministic: descending similarity, ties broken by lower
    index (stable sort).
    """

    name = "relevance"

    def _select_indices(
        self,
        pool_emb: np.ndarray,
        pool_tokens: np.ndarray,
        budget: int,
        core_emb: np.ndarray | None,
    ) -> tuple[list[int], dict]:
        del pool_tokens, budget  # not used -- _pack enforces the budget
        if core_emb is None:
            raise ValueError(
                "RelevanceSelector requires core_emb (an anchor vector); "
                "got None. In the primary sweep pass the pool centroid "
                "('centrality' baseline); in the secondary sweep pass "
                "the query embedding."
            )
        sim = _cosine_sim_to_anchor(pool_emb, core_emb)
        order = np.argsort(-sim, kind="stable").tolist()
        meta = {
            "n_candidates": int(pool_emb.shape[0]),
            "anchor_norm": float(np.linalg.norm(core_emb)),
        }
        return order, meta

# ---------------------------------------------------------------------
# MMRSelector
# ---------------------------------------------------------------------

@register_selector
class MMRSelector(BaseSelector):
    """Maximal Marginal Relevance with a fixed anchor.

    At each step, choose::

        score(x) = λ · sim(q, x) − (1−λ) · max_{s∈S} sim(s, x)

    where ``q`` is ``core_emb`` and ``S`` is the already selected set.
    Higher λ favors relevance to the anchor; lower λ favors diversity
    against the selected set.

    Edge cases
    ----------
    * ``λ = 1`` degenerates to :class:`RelevanceSelector` (verified by
      a test).
    * ``λ = 0`` needs no anchor; ``core_emb`` may be ``None``.
    * ``λ > 0`` requires ``core_emb``; ``None`` raises ``ValueError``.
    * Warm-up (``S = ∅``): with an anchor, pick argmax relevance;
      without, pick a random item seeded by ``seed``.
    * Budget is enforced *inside* the greedy loop: a candidate that
      would exceed the remaining budget is excluded from ``argmax``,
      not silently dropped afterwards. This keeps ``max_sim`` consistent
      with the actual selected set.
    """

    name = "mmr"

    def __init__(self, seed: int = 0, lambda_: float = 0.7, **cfg) -> None:
        super().__init__(seed=seed, **cfg)
        if not 0.0 <= lambda_ <= 1.0:
            raise ValueError(f"lambda_ must be in [0, 1], got {lambda_}")
        self.lambda_ = float(lambda_)

    def _select_indices(
        self,
        pool_emb: np.ndarray,
        pool_tokens: np.ndarray,
        budget: int,
        core_emb: np.ndarray | None,
    ) -> tuple[list[int], dict]:
        n = pool_emb.shape[0]
        if n == 0:
            return [], {"n_candidates": 0, "lambda": self.lambda_, "step_scores": []}
        if core_emb is None and self.lambda_ > 0:
            raise ValueError(
                "MMRSelector with lambda_ > 0 requires core_emb; "
                "pass the pool centroid (centrality mode) or a query "
                "embedding, or set lambda_=0 for diversity-only MMR."
            )

        X = np.asarray(pool_emb, dtype=np.float64)
        rel = (
            _cosine_sim_to_anchor(X, core_emb)
            if core_emb is not None
            else np.zeros(n, dtype=np.float64)
        )

        max_sim = np.full(n, -np.inf, dtype=np.float64)
        selected_mask = np.zeros(n, dtype=bool)
        selected: list[int] = []
        step_scores: list[float] = []
        running = 0

        # ---- warm-up ---------------------------------------------------
        warmup = "relevance" if (self.lambda_ > 0 and core_emb is not None) else "random"
        if warmup == "relevance":
            order = np.argsort(-rel, kind="stable")
        else:
            order = make_rng(self.seed).permutation(n)

        first: int | None = None
        for raw in order:
            i = int(raw)
            if running + int(pool_tokens[i]) <= budget:
                first = i
                break
        if first is None:
            return [], {
                "n_candidates": n,
                "lambda": self.lambda_,
                "step_scores": [],
                "warmup": warmup,
            }

        selected.append(first)
        selected_mask[first] = True
        running += int(pool_tokens[first])
        step_scores.append(float(rel[first]))
        np.maximum(max_sim, _cosine_sim_to_anchor(X, X[first]), out=max_sim)
        max_sim[first] = -np.inf

        # ---- greedy loop ----------------------------------------------
        while True:
            scores = self.lambda_ * rel - (1.0 - self.lambda_) * max_sim
            scores[selected_mask] = -np.inf
            fits = (pool_tokens + running) <= budget
            scores[~fits] = -np.inf

            best = int(np.argmax(scores))
            if not np.isfinite(scores[best]):
                break

            selected.append(best)
            selected_mask[best] = True
            running += int(pool_tokens[best])
            step_scores.append(float(scores[best]))
            np.maximum(max_sim, _cosine_sim_to_anchor(X, X[best]), out=max_sim)
            max_sim[best] = -np.inf

        meta = {
            "n_candidates": n,
            "lambda": self.lambda_,
            "step_scores": step_scores,
            "warmup": warmup,
        }
        return selected, meta


# ---------------------------------------------------------------------
# DynamicMMRSelector
# ---------------------------------------------------------------------

@register_selector
class DynamicMMRSelector(BaseSelector):
    """MMR whose relevance anchor is the centroid of the selected set.

    At each step, choose::

        score(x) = λ · sim(centroid(S), x) − (1−λ) · max_{s∈S} sim(s, x)

    The anchor moves as ``S`` grows. This makes the relevance term a
    "typicality within S" signal, complementary to the diversity term
    "distance from any member of S".

    Warm-up (``S = ∅``): if ``λ > 0`` and ``core_emb`` is provided, use
    argmax relevance to it; otherwise pick randomly by ``seed``. This
    keeps the first step deterministic and independent of the moving
    centroid (which is undefined when ``S`` is empty).

    Budget is enforced inside the greedy loop, same as for
    :class:`MMRSelector`.
    """

    name = "mmr_dynamic"

    def __init__(self, seed: int = 0, lambda_: float = 0.7, **cfg) -> None:
        super().__init__(seed=seed, **cfg)
        if not 0.0 <= lambda_ <= 1.0:
            raise ValueError(f"lambda_ must be in [0, 1], got {lambda_}")
        self.lambda_ = float(lambda_)

    def _select_indices(
        self,
        pool_emb: np.ndarray,
        pool_tokens: np.ndarray,
        budget: int,
        core_emb: np.ndarray | None,
    ) -> tuple[list[int], dict]:
        n = pool_emb.shape[0]
        if n == 0:
            return [], {"n_candidates": 0, "lambda": self.lambda_, "step_scores": []}

        X = np.asarray(pool_emb, dtype=np.float64)
        max_sim = np.full(n, -np.inf, dtype=np.float64)
        selected_mask = np.zeros(n, dtype=bool)
        selected: list[int] = []
        step_scores: list[float] = []
        running = 0
        centroid_sum = np.zeros(X.shape[1], dtype=np.float64)

        # ---- warm-up ---------------------------------------------------
        if core_emb is not None and self.lambda_ > 0:
            warmup = "anchor"
            order = np.argsort(-_cosine_sim_to_anchor(X, core_emb), kind="stable")
        else:
            warmup = "random"
            order = make_rng(self.seed).permutation(n)

        first: int | None = None
        for raw in order:
            i = int(raw)
            if running + int(pool_tokens[i]) <= budget:
                first = i
                break
        if first is None:
            return [], {
                "n_candidates": n,
                "lambda": self.lambda_,
                "step_scores": [],
                "warmup": warmup,
            }

        selected.append(first)
        selected_mask[first] = True
        running += int(pool_tokens[first])
        centroid_sum += X[first]
        step_scores.append(0.0)
        np.maximum(max_sim, _cosine_sim_to_anchor(X, X[first]), out=max_sim)
        max_sim[first] = -np.inf

        # ---- greedy loop ----------------------------------------------
        while True:
            centroid = centroid_sum / len(selected)
            rel = _cosine_sim_to_anchor(X, centroid)
            scores = self.lambda_ * rel - (1.0 - self.lambda_) * max_sim
            scores[selected_mask] = -np.inf
            fits = (pool_tokens + running) <= budget
            scores[~fits] = -np.inf

            best = int(np.argmax(scores))
            if not np.isfinite(scores[best]):
                break

            selected.append(best)
            selected_mask[best] = True
            running += int(pool_tokens[best])
            centroid_sum += X[best]
            step_scores.append(float(scores[best]))
            np.maximum(max_sim, _cosine_sim_to_anchor(X, X[best]), out=max_sim)
            max_sim[best] = -np.inf

        meta = {
            "n_candidates": n,
            "lambda": self.lambda_,
            "step_scores": step_scores,
            "warmup": warmup,
        }
        return selected, meta

# ---------------------------------------------------------------------
# Distance helper (FPS-local, to keep selection/ free of evaluation/)
# ---------------------------------------------------------------------

def _distance_to_point(
    X: np.ndarray, point: np.ndarray, metric: str
) -> np.ndarray:
    """Distance from each row of ``X`` to a single ``point`` (1-D).

    ``metric`` is ``"euclidean"`` or ``"cosine"``. Cosine follows the
    same convention as ``evaluation.metrics``: a zero-norm row or a
    zero-norm point has distance ``1.0`` (maximally dissimilar) rather
    than ``NaN``.
    """
    X = np.asarray(X, dtype=np.float64)
    p = np.asarray(point, dtype=np.float64)

    if metric == "euclidean":
        diff = X - p
        return np.sqrt(np.einsum("ij,ij->i", diff, diff))

    if metric == "cosine":
        x_norms = np.linalg.norm(X, axis=1)
        p_norm = float(np.linalg.norm(p))
        denom = x_norms * p_norm
        safe = np.where(denom > 0, denom, 1.0)
        sim = (X @ p) / safe
        np.clip(sim, -1.0, 1.0, out=sim)
        dist = 1.0 - sim
        return np.where(denom > 0, dist, 1.0)

    raise ValueError(f"unknown metric {metric!r}; supported: 'euclidean', 'cosine'")


# ---------------------------------------------------------------------
# FPSSelector
# ---------------------------------------------------------------------

@register_selector
class FPSSelector(BaseSelector):
    """Farthest Point Sampling.

    At each step, choose the candidate whose **distance to the nearest
    already-selected point is largest**. This is the greedy
    ``min-distance`` counterpart of MMR's ``λ=0`` mode, with one crucial
    difference: MMR uses *similarity* (cosine) while FPS uses an
    arbitrary distance metric, configurable as ``euclidean`` or
    ``cosine``. Euclidean is the convention in the FPS literature; on
    L2-normalised embeddings the two orderings differ only mildly, but
    on unnormalised embeddings they can diverge substantially.

    Relationship to IterativePCASelector
    ------------------------------------
    FPS and iterative PCA are **not** the same method dressed
    differently: FPS ranks candidates by ``min_{s∈S} d(x, s)`` (nearest
    neighbour in ``S``), while PCA ranks them by
    ``||x - U_k U_kᵀ x||`` (residual outside the subspace spanned by
    ``S``). They coincide only when ``S`` is well-approximated by a
    single point (``|S| = 1``); as ``S`` grows, PCA captures
    correlation structure that FPS cannot see. The gap between them is
    the empirical question this project is built around.

    Warm-up
    -------
    If ``core_emb`` is provided, the first pick is ``argmax`` cosine
    similarity to it (same as :class:`RelevanceSelector`'s first pick).
    Otherwise the first pick is random by ``seed``. This makes FPS
    deterministic given ``(core_emb, seed)`` and consistent with MMR's
    warm-up semantics.
    """

    name = "fps"

    def __init__(
        self,
        seed: int = 0,
        metric: str = "euclidean",
        **cfg,
    ) -> None:
        super().__init__(seed=seed, **cfg)
        if metric not in ("euclidean", "cosine"):
            raise ValueError(
                f"metric must be 'euclidean' or 'cosine', got {metric!r}"
            )
        self.metric = metric

    def _select_indices(
        self,
        pool_emb: np.ndarray,
        pool_tokens: np.ndarray,
        budget: int,
        core_emb: np.ndarray | None,
    ) -> tuple[list[int], dict]:
        n = pool_emb.shape[0]
        if n == 0:
            return [], {
                "n_candidates": 0,
                "metric": self.metric,
                "step_scores": [],
            }

        X = np.asarray(pool_emb, dtype=np.float64)

        # ---- warm-up ---------------------------------------------------
        if core_emb is not None:
            warmup = "anchor"
            order = np.argsort(
                -_cosine_sim_to_anchor(X, core_emb), kind="stable"
            )
        else:
            warmup = "random"
            order = make_rng(self.seed).permutation(n)

        first: int | None = None
        running = 0
        for raw in order:
            i = int(raw)
            if running + int(pool_tokens[i]) <= budget:
                first = i
                break
        if first is None:
            return [], {
                "n_candidates": n,
                "metric": self.metric,
                "step_scores": [],
                "warmup": warmup,
            }

        selected: list[int] = [first]
        selected_mask = np.zeros(n, dtype=bool)
        selected_mask[first] = True
        running += int(pool_tokens[first])

        # Running minimum distance from each candidate to any selected.
        # Initialised to the distance to the first pick; updated by
        # element-wise minimum on each subsequent pick.
        min_dist = _distance_to_point(X, X[first], self.metric)
        min_dist[first] = -np.inf  # exclude the first pick itself
        step_scores: list[float] = [0.0]  # warm-up by convention

        # ---- greedy loop ----------------------------------------------
        while True:
            scores = min_dist.copy()
            scores[selected_mask] = -np.inf
            fits = (pool_tokens + running) <= budget
            scores[~fits] = -np.inf

            best = int(np.argmax(scores))
            if not np.isfinite(scores[best]):
                break

            selected.append(best)
            selected_mask[best] = True
            running += int(pool_tokens[best])
            step_scores.append(float(scores[best]))
            np.minimum(
                min_dist,
                _distance_to_point(X, X[best], self.metric),
                out=min_dist,
            )
            min_dist[best] = -np.inf

        meta = {
            "n_candidates": n,
            "metric": self.metric,
            "step_scores": step_scores,
            "warmup": warmup,
        }
        return selected, meta


# ---------------------------------------------------------------------
# PCA helpers (shared by Static and Iterative PCA selectors)
# ---------------------------------------------------------------------

def _fit_top_k_subspace(
    X: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return top-``k`` principal directions of ``X`` and their eigenvalues.

    Two computation paths, chosen by shape:

    * ``N >= D``: eigendecompose the ``D×D`` gram matrix ``XᵀX``.
      Standard approach for "tall" matrices (the pool in static PCA).
    * ``N < D``:  eigendecompose the ``N×N`` gram matrix ``XXᵀ``, then
      project back via ``U = Xᵀ V / √λ``. For the iterative selector,
      ``X`` is the **selected set** with ``|S| << D``; this path is
      ``O(N³ + N·D²)`` instead of ``O(D³ + N·D²)`` -- a factor
      ``(D/N)³`` speedup, which at ``N = 100, D = 768`` is ~450x.

    Near-zero eigenvalues are dropped (relative threshold
    ``1e-10 · λ_max``), so the returned ``U`` never contains numerical
    noise directions whose contribution to ``U Uᵀ`` would be garbage.
    Directions are orthonormal; eigenvalues are in **descending** order;
    sign is arbitrary but ``U Uᵀ`` is sign-invariant.

    Parameters
    ----------
    X
        ``float64`` array ``(N, D)``. No centering.
    k
        Requested directions. Capped by ``min(N, D)`` and by the numeric
        rank of ``X``.

    Returns
    -------
    U
        ``(D, k_eff)`` orthonormal-column matrix.
    eigvals
        ``(k_eff,)`` non-increasing eigenvalues.
    """
    N, D = X.shape
    if N == 0 or D == 0:
        return np.zeros((D, 0)), np.empty(0)
    k = min(k, D, N)

    if N >= D:
        G = X.T @ X
        eigvals, eigvecs = np.linalg.eigh(G)
        eigvals = eigvals[::-1][:k].copy()
        eigvecs = eigvecs[:, ::-1][:, :k].copy()
        if len(eigvals) > 0:
            eps = max(float(eigvals[0]) * 1e-10, 1e-15)
            keep = eigvals > eps
            eigvals = eigvals[keep]
            eigvecs = eigvecs[:, keep]
        return eigvecs, eigvals

    G_small = X @ X.T
    eigvals, V = np.linalg.eigh(G_small)
    eigvals = eigvals[::-1][:k].copy()
    V = V[:, ::-1][:, :k].copy()
    if len(eigvals) > 0:
        eps = max(float(eigvals[0]) * 1e-10, 1e-15)
        keep = eigvals > eps
        eigvals = eigvals[keep]
        V = V[:, keep]
    if len(eigvals) == 0:
        return np.zeros((D, 0)), np.empty(0)
    U = X.T @ V
    U = U / np.sqrt(eigvals)[None, :]
    return U, eigvals


def _residuals_sq(X: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Squared residual norms of ``X`` outside ``span(U)``.

    Uses ``||x||² − ||Uᵀx||²`` (valid because ``U`` has orthonormal
    columns). Clamped at zero to absorb floating-point cancellation --
    a point already in ``span(U)`` has exact residual ``0``, but the
    subtraction could otherwise give ``-1e-16``.
    """
    if U.shape[1] == 0:
        return np.einsum("ij,ij->i", X, X)
    norms_sq = np.einsum("ij,ij->i", X, X)
    proj = X @ U
    proj_sq = np.einsum("ij,ij->i", proj, proj)
    return np.maximum(norms_sq - proj_sq, 0.0)

# ---------------------------------------------------------------------
# StaticPCASelector
# ---------------------------------------------------------------------

@register_selector
class StaticPCASelector(BaseSelector):
    """Static PCA ablation: fixed subspace, rank all candidates by residual.

    Fit PCA once on the **entire pool**, keep top-``k`` directions
    ``U_k``, and rank every pool item by ``||x − U_k U_kᵀ x||²``.
    Selection is then simply: walk candidates in descending residual
    order, pack to budget. The ranking ignores what has already been
    selected -- that is precisely what distinguishes this ablation from
    :class:`IterativePCASelector`, which refits ``U_k`` after every pick.

    Why this is an ablation
    -----------------------
    If IterativePCA beats StaticPCA in the experiments, the difference
    isolates the value of **online subspace updating**. If they tie, the
    cheaper static version wins by parsimony.

    No centering
    ------------
    PCA here is applied to **raw** embeddings (no mean subtraction).
    This matches the proposal's formula literally, avoids an ill-defined
    "mean of what" question, and keeps the residual interpretation
    simple: how much of ``x`` lives outside the pool's dominant
    directions. For L2-normalised embeddings the mean is often close to
    zero anyway.

    No warm-up
    ----------
    Residuals are defined for every candidate from the start, so there
    is no special first pick. ``core_emb`` is accepted for interface
    symmetry but ignored.

    Degenerate case
    ---------------
    When ``k >= min(N, D)`` the subspace spans the data and every
    residual is ≈ 0; selection degenerates to index order. A
    ``RuntimeWarning`` is emitted so that a misconfigured sweep does
    not silently produce trivial selections.
    """

    name = "static_pca"

    def __init__(self, seed: int = 0, k: int = 5, **cfg) -> None:
        super().__init__(seed=seed, **cfg)
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = int(k)

    def _select_indices(
        self,
        pool_emb: np.ndarray,
        pool_tokens: np.ndarray,
        budget: int,
        core_emb: np.ndarray | None,
    ) -> tuple[list[int], dict]:
        del core_emb  # anchor-free by design

        n = pool_emb.shape[0]
        if n == 0:
            return [], {
                "n_candidates": 0,
                "k_requested": self.k,
                "k_eff": 0,
            }

        X = np.asarray(pool_emb, dtype=np.float64)
        D = X.shape[1]
        k_eff = min(self.k, D, n)
        if self.k > k_eff:
            warnings.warn(
                f"StaticPCASelector: k={self.k} > min(N, D)={k_eff}; "
                f"capping to {k_eff}. Residuals will be near zero and "
                f"selection will degenerate.",
                RuntimeWarning,
                stacklevel=2,
            )

        U, eigvals = _fit_top_k_subspace(X, k_eff)
        residuals = _residuals_sq(X, U)
        order = np.argsort(-residuals, kind="stable").tolist()

        meta = {
            "n_candidates": int(n),
            "k_requested": int(self.k),
            "k_eff": int(k_eff),
            "top_eigenvalues": eigvals.tolist(),
        }
        return order, meta

# ---------------------------------------------------------------------
# IterativePCASelector  (the proposed method)
# ---------------------------------------------------------------------

@register_selector
class IterativePCASelector(BaseSelector):
    """Iterative PCA novelty selection -- the proposed method.

    At each step, compute the top-``k`` principal directions of the
    **already selected** embeddings, and pick the candidate with the
    largest squared residual outside that subspace::

        score(x) = || x − U_k U_kᵀ x ||²

    where ``U_k`` is refit from ``S`` on every step. Higher score means
    ``x`` covers a direction the selected set does not yet span.

    Relationship to other methods
    -----------------------------
    * :class:`StaticPCASelector` fits ``U_k`` once on the full pool.
      The gap between them isolates the value of **online updating**.
    * :class:`FPSSelector` ranks by ``min_{s∈S} d(x, s)`` (nearest
      selected neighbour). PCA ranks by residual **outside the
      subspace** spanned by ``S``. They differ once ``|S| > 1`` -- FPS
      is local, PCA is global over ``S``.

    Warm-up (``S = ∅``)
    -------------------
    Residual against an empty subspace is ``||x||²``, so the natural
    first pick is ``argmax ||x||²``. This is ``warmup="norm"`` (the
    default). ``warmup="random"`` picks a seeded random point instead;
    useful for robustness checks. If ``core_emb`` is provided, it takes
    precedence: first pick is ``argmax`` cosine to the anchor, matching
    :class:`MMRSelector`/:class:`FPSSelector` semantics.

    Effective ``k``
    ---------------
    ``k_eff = min(k, |S|, D)`` -- cannot have more principal directions
    than selected points. With ``variance_threshold`` set (e.g. ``0.9``),
    ``k_eff`` is additionally shrunk to the smallest count whose top
    eigenvalues explain that fraction of total variance.

    Over-picking caveat
    -------------------
    Once ``|S|`` grows, a tight cluster of points not in ``span(U_k)``
    all get near-equal high residual, so the selector may pick several
    from that cluster before another direction "wins". This is a real
    property of the residual criterion, not a bug; it shows up in
    diversity curves and in the qualitative coherence check (Block 14).
    Documented here so that readings of the results do not treat it as
    unexpected.
    """

    name = "iterative_pca"

    def __init__(
        self,
        seed: int = 0,
        k: int = 5,
        variance_threshold: float | None = None,
        warmup: str = "norm",
        **cfg,
    ) -> None:
        super().__init__(seed=seed, **cfg)
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if variance_threshold is not None and not 0.0 < variance_threshold <= 1.0:
            raise ValueError(
                f"variance_threshold must be in (0, 1], got {variance_threshold}"
            )
        if warmup not in ("norm", "random"):
            raise ValueError(
                f"warmup must be 'norm' or 'random', got {warmup!r}"
            )
        self.k = int(k)
        self.variance_threshold = variance_threshold
        self.warmup = warmup

    # -- subspace at each step ---------------------------------------

    def _compute_subspace(
        self, sel_emb: np.ndarray
    ) -> tuple[np.ndarray, int, np.ndarray]:
        """Return ``(U, k_eff, eigvals)`` for the current selected set."""
        n_sel, D = sel_emb.shape
        max_k = min(self.k, n_sel, D)
        if max_k == 0:
            return np.zeros((D, 0)), 0, np.empty(0)

        U, eigvals = _fit_top_k_subspace(sel_emb, max_k)
        k_eff = U.shape[1]

        if self.variance_threshold is not None and len(eigvals) > 0:
            total = float(eigvals.sum())
            if total > 1e-12:
                ratios = np.cumsum(eigvals) / total
                k_from_thresh = int(np.searchsorted(ratios, self.variance_threshold) + 1)
                k_from_thresh = min(k_from_thresh, k_eff)
                if k_from_thresh < k_eff:
                    k_eff = k_from_thresh
                    U = U[:, :k_eff]
                    eigvals = eigvals[:k_eff]
            # else: total variance ~0 (all selected points identical);
            # keep whatever _fit_top_k_subspace returned.

        return U, k_eff, eigvals

    # -- main entry point --------------------------------------------

    def _select_indices(
        self,
        pool_emb: np.ndarray,
        pool_tokens: np.ndarray,
        budget: int,
        core_emb: np.ndarray | None,
    ) -> tuple[list[int], dict]:
        n = pool_emb.shape[0]
        if n == 0:
            return [], {
                "n_candidates": 0,
                "k_requested": self.k,
                "variance_threshold": self.variance_threshold,
                "step_scores": [],
                "step_k_eff": [],
            }

        X = np.asarray(pool_emb, dtype=np.float64)
        D = X.shape[1]

        # ---- warm-up ---------------------------------------------------
        if core_emb is not None:
            warmup_mode = "anchor"
            order = np.argsort(
                -_cosine_sim_to_anchor(X, core_emb), kind="stable"
            )
        elif self.warmup == "norm":
            warmup_mode = "norm"
            norms_sq = np.einsum("ij,ij->i", X, X)
            order = np.argsort(-norms_sq, kind="stable")
        else:
            warmup_mode = "random"
            order = make_rng(self.seed).permutation(n)

        selected: list[int] = []
        running = 0
        first: int | None = None
        for raw in order:
            i = int(raw)
            if running + int(pool_tokens[i]) <= budget:
                first = i
                break
        if first is None:
            return [], {
                "n_candidates": n,
                "k_requested": self.k,
                "variance_threshold": self.variance_threshold,
                "step_scores": [],
                "step_k_eff": [],
                "warmup": warmup_mode,
            }

        selected_mask = np.zeros(n, dtype=bool)
        selected.append(first)
        selected_mask[first] = True
        running += int(pool_tokens[first])

        step_scores: list[float] = [0.0]
        step_k_eff: list[int] = [0]
        step_eigvals: list[list[float]] = [[]]

        # ---- greedy loop ----------------------------------------------
        while True:
            sel_emb = X[selected]
            U, k_eff, eigvals = self._compute_subspace(sel_emb)

            if k_eff == 0:
                # No directions available (all selected points are zero
                # vectors). Fall back to raw norm², which is residual
                # against the empty subspace.
                scores = np.einsum("ij,ij->i", X, X).copy()
            else:
                scores = _residuals_sq(X, U)

            scores[selected_mask] = -np.inf
            fits = (pool_tokens + running) <= budget
            scores[~fits] = -np.inf

            best = int(np.argmax(scores))
            if not np.isfinite(scores[best]):
                break

            selected.append(best)
            selected_mask[best] = True
            running += int(pool_tokens[best])
            step_scores.append(float(scores[best]))
            step_k_eff.append(int(k_eff))
            step_eigvals.append(eigvals.tolist())

        meta = {
            "n_candidates": int(n),
            "k_requested": int(self.k),
            "variance_threshold": self.variance_threshold,
            "warmup": warmup_mode,
            "step_scores": step_scores,
            "step_k_eff": step_k_eff,
            "step_eigvals": step_eigvals,
        }
        return selected, meta