"""Selection sweep: run all (method × seed × budget) combinations.

Responsibility
--------------
Take a prepared pool, encode it with two independent encoders, run every
selector under every (seed, budget), and emit tidy result tables:

* ``<name>.parquet``          -- one row per (method, seed, target_budget)
* ``<name>_agg.parquet``      -- mean/std/CI across seeds per (method, budget)
* ``<name>_vs_<ref>.parquet`` -- paired delta vs a reference method
* ``<name>.meta.json``        -- config, encoder fingerprints, timestamp

Why one run per budget, not "max budget + interpolation"
--------------------------------------------------------
MMR, FPS and both PCA selectors are greedy with budget checks **inside**
the loop: when the next candidate does not fit, they skip it rather than
terminate. As a result the set selected at ``budget=512`` is not
necessarily a prefix of the set selected at ``budget=1024`` -- a
candidate that fit at 512 may be displaced by a higher-scoring candidate
that only fits once the budget grows. Interpolating from a single
max-budget run would therefore report numbers the method never achieves.
We run each ``(method, seed, budget)`` independently and record the
actually selected set.

Selection vs evaluation embeddings
----------------------------------
Selectors score on **selection** embeddings. Metrics measure on **eval**
embeddings. Using the same encoder for both would let a method optimise
the exact quantity it is being graded on, and the comparison would be
meaningless. The two encoders MUST have different fingerprints; we
enforce this and refuse to run otherwise.

Centrality anchor
-----------------
``core_emb`` passed to selectors is the L2-normalised centroid of the
pool in **selection** embedding space (see
:func:`pcadiv.data.pools.compute_pool_centroid`). This is the anchor for
the relevance baseline and for the relevance term of MMR in the primary
sweep. A selector spec may set ``pass_anchor: false`` to withhold the
anchor (``core_emb=None``); deterministic selectors then warm up from
their seeded anchor-free mode, which is what produces a non-degenerate
bootstrap CI across seeds. Metrics receive ``core_emb=None`` -- current metrics are
anchor-free, and future anchor-aware metrics will be wired in when they
exist.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pcadiv.data.pools import compute_pool_centroid
from pcadiv.embeddings.cache import load_or_encode
from pcadiv.embeddings.encoders import BaseEncoder, get_encoder
from pcadiv.evaluation.metrics import BaseMetric, get_metric
from pcadiv.evaluation.stats import aggregate_seeds, paired_compare
from pcadiv.selection.selectors import get_selector
from pcadiv.utils.config import load_config
from pcadiv.utils.logging import get_logger

__all__ = ["build_encoder_from_cfg", "run_sweep"]


def build_encoder_from_cfg(cfg: dict) -> BaseEncoder:
    """Instantiate an encoder from a parsed YAML config dict."""
    if "name" not in cfg:
        raise KeyError("encoder config must have a 'name' field")
    name = cfg["name"]
    params = {k: v for k, v in cfg.items() if k != "name"}
    return get_encoder(name, **params)


def _load_encoder(
    maybe_cfg: dict | BaseEncoder,
    label: str,
) -> BaseEncoder:
    if isinstance(maybe_cfg, BaseEncoder):
        return maybe_cfg
    return build_encoder_from_cfg(maybe_cfg)


def run_sweep(
    cfg: dict,
    *,
    out_dir: str | Path | None = None,
    sweep_name: str | None = None,
    pool_override: str | Path | None = None,
    sel_encoder_override: BaseEncoder | None = None,
    eval_encoder_override: BaseEncoder | None = None,
) -> dict:
    """Run all (method × seed × budget) combinations.

    Parameters
    ----------
    cfg
        Parsed experiment config (see ``configs/experiments/main_sweep.yaml``).
    out_dir
        Override ``cfg["output_dir"]``. Created if missing.
    sweep_name
        Override ``cfg["sweep_name"]``.
    pool_override
        Override ``cfg["data"]["pool_path"]``. Useful for tests and for
        running against a small debug pool without editing config.
    sel_encoder_override, eval_encoder_override
        If given, used directly instead of loading from the YAML files
        referenced in ``cfg["encoders"]``. For tests and notebook use.

    Returns
    -------
    dict
        ``{"results_path", "agg_path", "compare_path" (or None),
        "n_rows", "n_methods", "n_seeds", "n_budgets"}``.
    """
    log = get_logger("sweep")

    pool_path = Path(pool_override) if pool_override is not None else Path(
        cfg["data"]["pool_path"]
    )
    if not pool_path.exists():
        raise FileNotFoundError(f"pool file not found: {pool_path}")
    pool = pd.read_parquet(pool_path)
    if len(pool) == 0:
        raise ValueError(f"pool is empty: {pool_path}")
    log.info("pool: %d passages from %s", len(pool), pool_path)
    texts = pool["text"].tolist()
    pool_tokens = pool["n_tokens"].to_numpy(dtype=np.int64)

    cache_dir: str | Path | None = cfg["data"].get("cache_dir")

    # ---- encoders ---------------------------------------------------
    if sel_encoder_override is not None:
        sel_enc = sel_encoder_override
    else:
        sel_enc = build_encoder_from_cfg(
            load_config(cfg["encoders"]["selection"])
        )
    if eval_encoder_override is not None:
        eval_enc = eval_encoder_override
    else:
        eval_enc = build_encoder_from_cfg(
            load_config(cfg["encoders"]["eval"])
        )

    if sel_enc.fingerprint() == eval_enc.fingerprint():
        raise ValueError(
            "selection and eval encoders have the same fingerprint "
            f"({sel_enc.fingerprint()}); evaluation would be biased. "
            "Use different (name, model_id, normalize, dim) for the two."
        )
    log.info(
        "selection encoder: %s (fp=%s)",
        sel_enc.__class__.__name__, sel_enc.fingerprint(),
    )
    log.info(
        "eval encoder: %s (fp=%s)",
        eval_enc.__class__.__name__, eval_enc.fingerprint(),
    )

    t0 = time.perf_counter()
    sel_emb = load_or_encode(sel_enc, texts, cache_dir)
    eval_emb = load_or_encode(eval_enc, texts, cache_dir)
    log.info(
        "embeddings: sel %s, eval %s (%.1fs)",
        sel_emb.shape, eval_emb.shape, time.perf_counter() - t0,
    )

    core_emb = compute_pool_centroid(sel_emb, normalize=True)

    # ---- instantiate selectors and metrics -------------------------
    # Each spec is (name, params, pass_anchor). ``pass_anchor`` defaults
    # to True (backward compatible: every pre-existing config passed the
    # pool centroid to every selector). With ``pass_anchor: false`` the
    # selector receives ``core_emb=None`` and its warm-up falls back to
    # its anchor-free mode (seeded random / norm) -- this is what gives
    # deterministic selectors (fps, iterative_pca, mmr_dynamic) a real
    # variance across seeds. Selectors that require an anchor
    # (relevance, mmr with lambda_>0) raise ValueError if it is withheld.
    selector_specs: list[tuple[str, dict, bool]] = []
    for s in cfg["selectors"]:
        params = dict(s.get("params", {}))
        pass_anchor = s.get("pass_anchor", True)
        if not isinstance(pass_anchor, bool):
            raise ValueError(
                f"selector {s['name']!r}: pass_anchor must be a bool, "
                f"got {pass_anchor!r}"
            )
        if "seed" in params:
            # 'seed' is controlled by the sweep loop (cfg['seeds']); a
            # per-selector seed in params is a config bug -- silently
            # it would be overwritten anyway, so make the intent clear.
            log.warning(
                "selector %r has 'seed' in params; sweep overrides it "
                "with each value from cfg['seeds'] -- removing",
                s["name"],
            )
            params.pop("seed")
        selector_specs.append((s["name"], params, pass_anchor))
    metrics = [
        get_metric(m["name"], **m.get("params", {}))
        for m in cfg["metrics"]
    ]
    metric_keys: list[str] = []
    # Dry probe: call each metric on empty input to learn its key set.
    for m in metrics:
        probe = m(eval_emb[:0], core_emb=None)
        for k in probe:
            if k in metric_keys:
                raise ValueError(
                    f"metric key {k!r} is returned by more than one metric"
                )
            metric_keys.append(k)

    budgets = [int(b) for b in cfg["budgets"]]
    seeds = [int(s) for s in cfg["seeds"]]
    if not budgets:
        raise ValueError("cfg['budgets'] is empty")
    if not seeds:
        raise ValueError("cfg['seeds'] is empty")

    # ---- main loop -------------------------------------------------
    rows: list[dict[str, Any]] = []
    n_total = len(selector_specs) * len(seeds) * len(budgets)
    n_done = 0
    t_sweep = time.perf_counter()
    for sel_name, sel_params, pass_anchor in selector_specs:
        anchor = core_emb if pass_anchor else None
        for seed in seeds:
            for budget in budgets:
                selector = get_selector(sel_name, seed=seed, **sel_params)
                res = selector.select(
                    sel_emb, pool_tokens, budget=budget, core_emb=anchor,
                )
                sel_eval = (
                    eval_emb[res.indices] if len(res) > 0 else eval_emb[:0]
                )
                row: dict[str, Any] = {
                    "method": sel_name,
                    "seed": seed,
                    "target_budget": budget,
                    "token_budget_actual": int(res.total_tokens),
                    "n_selected": int(len(res)),
                }
                for m in metrics:
                    row.update(m(sel_eval, core_emb=None))
                rows.append(row)
                n_done += 1
                if n_done % max(1, n_total // 10) == 0 or n_done == n_total:
                    log.info(
                        "progress %d/%d (%.1fs)",
                        n_done, n_total, time.perf_counter() - t_sweep,
                    )

    results = pd.DataFrame(rows)
    # Column order: identity first, then n_selected/token_budget, then metrics.
    identity = ["method", "seed", "target_budget",
                "token_budget_actual", "n_selected"]
    results = results[identity + metric_keys].copy()

    # ---- aggregate across seeds ------------------------------------
    agg = aggregate_seeds(
        results,
        by=["method", "target_budget"],
        value_cols=["n_selected"] + metric_keys,
    )

    # ---- paired comparison vs reference ---------------------------
    compare_df: pd.DataFrame | None = None
    reference = cfg.get("reference_method")
    if reference is not None and reference in results["method"].unique():
        compare_df = paired_compare(
            results,
            reference=reference,
            by=["target_budget"],
            value_cols=metric_keys,
        )

    # ---- save ------------------------------------------------------
    out = Path(out_dir) if out_dir is not None else Path(cfg["output_dir"])
    name = sweep_name or cfg.get("sweep_name", "sweep")
    out.mkdir(parents=True, exist_ok=True)

    results_path = out / f"{name}.parquet"
    agg_path = out / f"{name}_agg.parquet"
    compare_path: Path | None = None
    meta_path = out / f"{name}.meta.json"

    results.to_parquet(results_path, index=False)
    agg.to_parquet(agg_path, index=False)
    if compare_df is not None:
        compare_path = out / f"{name}_vs_{reference}.parquet"
        compare_df.to_parquet(compare_path, index=False)

    meta = {
        "sweep_name": name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pool_path": str(pool_path),
        "n_pool_passages": int(len(pool)),
        "sel_encoder": {
            "class": sel_enc.__class__.__name__,
            "fingerprint": sel_enc.fingerprint(),
        },
        "eval_encoder": {
            "class": eval_enc.__class__.__name__,
            "fingerprint": eval_enc.fingerprint(),
        },
        "selectors": [s[0] for s in selector_specs],
        "selector_specs": [
            {"name": n, "params": p, "pass_anchor": a}
            for n, p, a in selector_specs
        ],
        "budgets": budgets,
        "seeds": seeds,
        "metric_keys": metric_keys,
        "n_rows": int(len(results)),
        "reference_method": reference,
    }
    meta_path.write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
    )

    log.info(
        "sweep done: %d rows, %.1fs total; saved to %s",
        len(results), time.perf_counter() - t_sweep, out,
    )

    return {
        "results_path": str(results_path),
        "agg_path": str(agg_path),
        "compare_path": str(compare_path) if compare_path else None,
        "meta_path": str(meta_path),
        "n_rows": int(len(results)),
        "n_methods": len(selector_specs),
        "n_seeds": len(seeds),
        "n_budgets": len(budgets),
    }