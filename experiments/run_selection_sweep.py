"""CLI wrapper for the selection sweep.

Thin on purpose: argument parsing only. All work is in
:func:`pcadiv.sweep.run_sweep`, exercised by tests directly.

Usage::

    python experiments/run_selection_sweep.py \
        --config configs/experiments/main_sweep.yaml

    # debug on a small pool, override pool path and sweep name
    python experiments/run_selection_sweep.py \
        --config configs/experiments/main_sweep.yaml \
        --pool data/pools/primary_debug.parquet \
        --sweep-name debug

    # debug with tiny encoders via YAML overrides
    python experiments/run_selection_sweep.py \
        --config configs/experiments/main_sweep.yaml \
        --override encoders.selection=configs/embed/selection_encoder.yaml \
        --sweep-name debug
"""
from __future__ import annotations

import argparse

from pcadiv.sweep import run_sweep
from pcadiv.utils.config import load_config
from pcadiv.utils.logging import configure, get_logger


def main() -> None:
    configure()
    log = get_logger("run_selection_sweep")

    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiments/main_sweep.yaml")
    p.add_argument(
        "--override", action="append", default=[],
        help="dot-path override, e.g. budgets='[512]'",
    )
    p.add_argument(
        "--pool", default=None,
        help="override data.pool_path (debug on a smaller pool)",
    )
    p.add_argument(
        "--sweep-name", default=None,
        help="override cfg['sweep_name'] (used as output file stem)",
    )
    p.add_argument(
        "--out-dir", default=None,
        help="override cfg['output_dir']",
    )
    args = p.parse_args()

    cfg = load_config(args.config, args.override)
    out = run_sweep(
        cfg,
        out_dir=args.out_dir,
        sweep_name=args.sweep_name,
        pool_override=args.pool,
    )
    log.info("done: %s", out)


if __name__ == "__main__":
    main()