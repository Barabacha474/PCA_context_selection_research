"""Summarise essential-experiment sweeps: paired deltas vs fps / random.

For every sweep given on the command line, reads the raw
``results/metrics/<name>.parquet`` and computes seed-paired deltas
(bootstrap CI over seeds, via :func:`pcadiv.evaluation.stats.paired_compare`)
of every method against ``fps`` and against ``random``. Writes:

* ``results/essential/<name>_vs_fps.csv`` / ``_vs_random.csv``
* ``results/essential/summary_ipa_vs_fps.csv`` -- one row per
  (sweep, budget): ipa mean, fps mean, delta, CI, significance flag, for
  Vendi and coverage radius. This is the table the report is built from.
* ``results/essential/summary_vendi_wide.csv`` -- mean Vendi, method x
  budget, one block per sweep.

Usage::

    python experiments/analyze_essential.py randomized sel_mpnet eval_mpnet ...
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pcadiv.evaluation.stats import paired_compare

METRICS = ["vendi_score", "coverage_radius", "mean_pairwise"]


def summarise(name: str, metrics_dir: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_parquet(metrics_dir / f"{name}.parquet")
    for ref in ("fps", "random"):
        cmp_ = paired_compare(raw, reference=ref, by=["target_budget"],
                              value_cols=METRICS, n_resamples=10000)
        cmp_.insert(0, "sweep", name)
        cmp_.to_csv(out_dir / f"{name}_vs_{ref}.csv", index=False)
        if ref == "fps":
            vs_fps = cmp_

    means = raw.groupby(["method", "target_budget"])[METRICS + ["n_selected"]].mean()
    rows = []
    ipa = vs_fps[vs_fps["method"] == "iterative_pca"]
    for _, r in ipa.iterrows():
        b = r["target_budget"]
        row = {"sweep": name, "budget": int(b), "n_seeds": int(r["n_seeds"]),
               "n_sel_ipa": means.loc[("iterative_pca", b), "n_selected"]}
        for m in ("vendi_score", "coverage_radius"):
            row[f"{m}_ipa"] = means.loc[("iterative_pca", b), m]
            row[f"{m}_fps"] = means.loc[("fps", b), m]
            row[f"{m}_delta"] = r[f"{m}_delta"]
            row[f"{m}_ci_low"] = r[f"{m}_ci_low"]
            row[f"{m}_ci_high"] = r[f"{m}_ci_high"]
            lo, hi = r[f"{m}_ci_low"], r[f"{m}_ci_high"]
            row[f"{m}_sig"] = "+" if lo > 0 else ("-" if hi < 0 else "0")
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("budget")

    wide = means["vendi_score"].unstack("method").reset_index()
    wide.insert(0, "sweep", name)
    return summary, wide


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("sweeps", nargs="+")
    p.add_argument("--metrics-dir", default="results/metrics")
    p.add_argument("--out-dir", default="results/essential")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summaries, wides = [], []
    for name in args.sweeps:
        s, w = summarise(name, Path(args.metrics_dir), out)
        summaries.append(s)
        wides.append(w)
    summary = pd.concat(summaries, ignore_index=True)
    wide = pd.concat(wides, ignore_index=True)
    summary.to_csv(out / "summary_ipa_vs_fps.csv", index=False)
    wide.to_csv(out / "summary_vendi_wide.csv", index=False)

    pd.set_option("display.width", 200)
    cols = ["sweep", "budget", "n_sel_ipa", "vendi_score_ipa", "vendi_score_fps",
            "vendi_score_delta", "vendi_score_ci_low", "vendi_score_ci_high",
            "vendi_score_sig", "coverage_radius_delta", "coverage_radius_sig"]
    print(summary[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
