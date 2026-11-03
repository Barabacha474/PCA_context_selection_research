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