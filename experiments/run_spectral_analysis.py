"""Spectral analysis of a prepared pool: eigenvalues + MP comparison + plot.

Usage::

    python experiments/run_spectral_analysis.py \
        --pool data/pools/primary.parquet \
        --encoder configs/embed/selection_encoder.yaml \
        --out-dir results/spectral
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pcadiv.analysis.plots import plot_spectrum
from pcadiv.analysis.spectral import pool_eigenvalues, spectral_summary
from pcadiv.embeddings.cache import load_or_encode
from pcadiv.sweep import build_encoder_from_cfg
from pcadiv.utils.config import load_config
from pcadiv.utils.logging import configure, get_logger


def main() -> None:
    configure()
    log = get_logger("run_spectral_analysis")

    p = argparse.ArgumentParser()
    p.add_argument("--pool", required=True)
    p.add_argument(
        "--encoder", default="configs/embed/selection_encoder.yaml",
        help="encoder YAML -- typically the selection encoder",
    )
    p.add_argument("--out-dir", default="results/spectral")
    p.add_argument("--cache-dir", default="data/embeddings")
    args = p.parse_args()

    pool = pd.read_parquet(args.pool)
    log.info("pool: %d passages", len(pool))

    enc = build_encoder_from_cfg(load_config(args.encoder))
    log.info("encoder: %s fp=%s", enc.__class__.__name__, enc.fingerprint())
    emb = load_or_encode(enc, pool["text"].tolist(), args.cache_dir)
    log.info("embeddings: %s", emb.shape)

    eigs = pool_eigenvalues(emb)
    summary = spectral_summary(emb)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"rank": range(1, len(eigs) + 1), "eigenvalue": eigs}).to_parquet(
        out / "eigenvalues.parquet", index=False
    )
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    plot_spectrum(
        eigs, n=summary["n"], d=summary["d"],
        output_path=out / "spectrum.png",
    )
    log.info(
        "wrote %s: %d signal / %d noise, PR=%.2f",
        out, summary["n_signal"], summary["n_noise"],
        summary["participation_ratio"],
    )


if __name__ == "__main__":
    main()