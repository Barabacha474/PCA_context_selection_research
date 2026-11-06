"""Diversity-vs-budget curves (all methods) + JSON export for the report page.

Figure: ``results/figures/vendi_vs_budget.png`` -- mean Vendi with 95%
bootstrap CI band over seeds, one line per method, two panels: the base
eval space (bert-tiny) and an independent sentence encoder (mpnet).
Both from the randomized warm-up sweeps (selection = MiniLM-L6).

JSON: ``results/essential/report_data.json`` -- the same curves for four
sweeps plus every ipa-vs-fps relative delta, consumed by the HTML report.

Usage::

    python experiments/plot_curves.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

MET = Path("results/metrics")
ESS = Path("results/essential")
OUT = Path("results/figures")
ORDER = ["iterative_pca", "fps", "static_pca", "random", "mmr", "relevance", "mmr_dynamic"]
LABEL = {"iterative_pca": "iterative PCA", "fps": "FPS", "static_pca": "static PCA",
         "random": "random", "mmr": "MMR λ=0.7", "relevance": "relevance",
         "mmr_dynamic": "MMR dynamic"}
COLOR = dict(zip(ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                         "#e87ba4", "#008300", "#6250d6"]))
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"
CURVE_SWEEPS = {"randomized": "primary, eval bert-tiny",
                "eval_mpnet": "primary, eval mpnet",
                "ood_2000": "OOD 2000, eval bert-tiny",
                "clean": "primary без zalgo, eval bert-tiny"}


def curves(sweep: str) -> pd.DataFrame:
    return pd.read_parquet(MET / f"{sweep}_agg.parquet")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, sweep, title in zip(axes, ["randomized", "eval_mpnet"],
                                ["eval = bert-tiny (база)", "eval = all-mpnet-base-v2"]):
        a = curves(sweep)
        for m in ORDER:
            d = a[a["method"] == m].sort_values("target_budget")
            x = d["target_budget"]
            ax.fill_between(x, d["vendi_score_ci_low"], d["vendi_score_ci_high"],
                            color=COLOR[m], alpha=0.15, lw=0)
            ax.plot(x, d["vendi_score_mean"], color=COLOR[m], lw=2, marker="o", ms=4,
                    label=LABEL[m])
        ax.set_xscale("log", base=2)
        ax.set_xticks([2048, 4096, 8192, 16384, 32768])
        ax.set_xticklabels(["2k", "4k", "8k", "16k", "32k"])
        ax.grid(axis="y", color=GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("token budget")
    axes[0].set_ylabel("Vendi score (mean, 95% CI)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Diversity vs token budget, selection = MiniLM-L6, random warm-up, 10 seeds",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "vendi_vs_budget.png", dpi=150)
    print("wrote", OUT / "vendi_vs_budget.png")

    data = {"curves": {}, "deltas": []}
    for sweep, label in CURVE_SWEEPS.items():
        a = curves(sweep)
        data["curves"][sweep] = {"label": label, "series": {
            m: [{"b": int(r.target_budget), "mean": round(r.vendi_score_mean, 4),
                 "lo": round(r.vendi_score_ci_low, 4), "hi": round(r.vendi_score_ci_high, 4),
                 "n": round(r.n_selected_mean, 1)}
                for r in a[a["method"] == m].sort_values("target_budget").itertuples()]
            for m in ORDER}}
    s = pd.read_csv(ESS / "summary_ipa_vs_fps.csv")
    for r in s.itertuples():
        f = 100.0 / r.vendi_score_fps
        data["deltas"].append({"sweep": r.sweep, "b": int(r.budget),
                               "d": round(r.vendi_score_delta * f, 2),
                               "lo": round(r.vendi_score_ci_low * f, 2),
                               "hi": round(r.vendi_score_ci_high * f, 2),
                               "sig": r.vendi_score_sig})
    (ESS / "report_data.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print("wrote", ESS / "report_data.json")


if __name__ == "__main__":
    main()
