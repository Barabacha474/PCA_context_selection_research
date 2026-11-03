"""CLI wrapper for the data-preparation pipeline.

Thin on purpose: argument parsing only. All real work is in
:func:`pcadiv.data.pipeline.run_data_prep`, which is exercised by tests
directly.

Usage::

    python experiments/run_data_prep.py --config configs/data/writingprompts.yaml

    # debug: tiny pool, tiny dataset, skip re-download
    python experiments/run_data_prep.py \
        --override source.limit=2000 \
        --override pools.primary.n_passages=200 \
        --skip-download
"""
from __future__ import annotations

import argparse

from pcadiv.data.pipeline import run_data_prep
from pcadiv.utils.config import load_config
from pcadiv.utils.logging import configure, get_logger


def main() -> None:
    configure()
    log = get_logger("run_data_prep")

    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/data/writingprompts.yaml")
    p.add_argument(
        "--override", action="append", default=[],
        help="dot-path override, e.g. pools.primary.n_passages=1000",
    )
    p.add_argument(
        "--skip-download", action="store_true",
        help="read stories from source.cache_path instead of calling HF",
    )
    p.add_argument(
        "--force-passageize", action="store_true",
        help="re-run passageize even if passages.parquet exists",
    )
    args = p.parse_args()

    cfg = load_config(args.config, args.override)
    out = run_data_prep(
        cfg,
        skip_download=args.skip_download,
        force_passageize=args.force_passageize,
    )
    log.info("done: %s", out)


if __name__ == "__main__":
    main()