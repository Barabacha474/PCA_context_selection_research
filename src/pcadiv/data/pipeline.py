"""Full data-preparation pipeline: download -> passageize -> split -> pools.

Kept in the package (not in ``experiments/``) so that tests can call
:func:`run_data_prep` directly, without importing an un-packaged
script.

The pipeline is idempotent: existing artefacts are reused unless
``force_*`` is set. Every step logs its outcome and every output path
is returned, so the caller (CLI, test, notebook) does not need to
recompute paths.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable

import pandas as pd

from pcadiv.data.download import load_writingprompts
from pcadiv.data.passageize import passageize
from pcadiv.data.pools import build_pool, save_pool
from pcadiv.data.splits import split_ids
from pcadiv.data.tokenizer import get_tokenizer
from pcadiv.utils.io import ensure_dir
from pcadiv.utils.logging import get_logger

__all__ = ["run_data_prep"]


def run_data_prep(
    cfg: dict,
    *,
    skip_download: bool = False,
    force_passageize: bool = False,
    load_fn: Callable[..., pd.DataFrame] | None = None,
) -> dict:
    """Run the data-preparation pipeline.

    Parameters
    ----------
    cfg
        Parsed data config (see ``configs/data/writingprompts.yaml``).
    skip_download
        If ``True``, read stories directly from ``source.cache_path``
        instead of calling the downloader.
    force_passageize
        If ``True``, re-run passageize even if ``passageize.out_path``
        already exists.
    load_fn
        Override for the downloader -- used by tests to inject a mock.
        Defaults to :func:`load_writingprompts`.

    Returns
    -------
    dict
        Keys: ``passages_path``, ``primary_pool_path``, ``ood_pool_path``,
        ``n_passages``, ``n_primary``, ``n_ood``, ``n_train_stories``,
        ``n_ood_stories``, ``disjoint`` (always ``True``; the function
        raises on overlap rather than returning a false flag).
    """
    log = get_logger("data_pipeline")
    load_fn = load_fn or load_writingprompts

    # ---- 1. Stories --------------------------------------------------
    src = cfg["source"]
    if skip_download:
        cache_path = Path(src["cache_path"])
        if not cache_path.exists():
            raise FileNotFoundError(
                f"skip_download=True but {cache_path} does not exist"
            )
        stories = pd.read_parquet(cache_path)
        if src.get("limit") is not None:
            stories = stories.head(int(src["limit"])).copy()
        log.info(
            "skipped download; loaded %d stories from %s",
            len(stories), cache_path,
        )
    else:
        stories = load_fn(
            cache_path=src["cache_path"],
            split=src.get("split", "train"),
            limit=src.get("limit"),
            force_download=src.get("force_download", False),
            hf_dataset=src["hf_dataset"],
        )
        log.info("loaded %d stories", len(stories))

    if len(stories) == 0:
        raise RuntimeError("no stories loaded; check source configuration")

    # ---- 2. Passages -------------------------------------------------
    ps = cfg["passageize"]
    if "passage_tokens" not in ps:
        raise KeyError(
            "passageize.passage_tokens is required (no default by design)"
        )
    passages_path = Path(ps["out_path"])
    if passages_path.exists() and not force_passageize:
        passages = pd.read_parquet(passages_path)
        log.info(
            "reusing existing passages: %d rows from %s",
            len(passages), passages_path,
        )
    else:
        tok_cfg = cfg["tokenizer"]
        tokenizer = get_tokenizer(tok_cfg["name"], **tok_cfg.get("params", {}))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            passages = passageize(
                stories,
                tokenizer,
                passage_tokens=int(ps["passage_tokens"]),
                min_passage_tokens=ps.get("min_passage_tokens"),
            )
            for w in caught:
                log.warning("passageize: %s", w.message)
        if len(passages) == 0:
            raise RuntimeError(
                "passageize produced zero passages; check "
                "passage_tokens / min_passage_tokens"
            )
        ensure_dir(passages_path.parent)
        passages.to_parquet(passages_path, index=False)
        log.info(
            "passageized: %d passages -> %s",
            len(passages), passages_path,
        )

    # ---- 3. Split ----------------------------------------------------
    sp = cfg["split"]
    train_ids, ood_ids = split_ids(
        passages,
        train_frac=sp["train_frac"],
        seed=sp["seed"],
    )
    log.info(
        "split: %d train stories, %d ood stories",
        len(train_ids), len(ood_ids),
    )

    # ---- 4. Pools ----------------------------------------------------
    outputs: dict = {
        "passages_path": str(passages_path),
        "n_passages": int(len(passages)),
        "n_train_stories": len(train_ids),
        "n_ood_stories": len(ood_ids),
    }
    for pool_name, pool_ids in (("primary", train_ids), ("ood", ood_ids)):
        pool_cfg = cfg["pools"][pool_name]
        pool = build_pool(
            passages,
            n_passages=int(pool_cfg["n_passages"]),
            seed=int(pool_cfg["seed"]),
            story_ids=pool_ids,
            min_tokens=pool_cfg.get("min_tokens"),
            max_tokens=pool_cfg.get("max_tokens"),
            allow_short=pool_cfg.get("allow_short", False),
        )
        pool_path = Path(pool_cfg["out_path"])
        save_pool(
            pool, pool_path,
            seed=int(pool_cfg["seed"]),
            source_passages_path=passages_path,
            extra_meta={"split": pool_name},
        )
        log.info(
            "%s pool: %d passages, %d unique stories -> %s",
            pool_name, len(pool), pool["story_id"].nunique(), pool_path,
        )
        outputs[f"{pool_name}_pool_path"] = str(pool_path)
        outputs[f"n_{pool_name}"] = int(len(pool))

    # ---- 5. Disjointness --------------------------------------------
    primary = pd.read_parquet(outputs["primary_pool_path"])
    ood = pd.read_parquet(outputs["ood_pool_path"])
    overlap = set(primary["story_id"]) & set(ood["story_id"])
    if overlap:
        raise RuntimeError(
            f"primary and ood pools share {len(overlap)} stories; "
            f"split is not disjoint -- check splits configuration"
        )
    outputs["disjoint"] = True
    return outputs