"""Figure for docs/05_essential_results.md: relative delta Vendi (ipa - fps) vs budget.

Four small multiples, one per experiment group; each line is one sweep,
error bars are the 95% bootstrap CI over the 10 seed-paired deltas
(``results/essential/<sweep>_vs_fps.csv``, written by analyze_essential.py).
The anchored debug_wide reproduction has no CI (deterministic) and is drawn
with hollow markers.

Usage::

    python experiments/plot_essential.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ESS = Path("results/essential")
OUT = Path("results/figures")
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # fixed order
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"

PANELS = [
    ("3.1  Warm-up: anchor vs random", [
        ("debug_wide_repro", "anchor (debug_wide)"),
        ("randomized", "random warm-up, 10 seeds")]),
    ("1.1  Selection encoder (eval = bert-tiny)", [
        ("randomized", "MiniLM-L6 (base)"),
        ("sel_minilm_l3", "MiniLM-L3"),
        ("sel_mpnet", "mpnet-base"),
        ("sel_bge_large", "bge-large"),
        ("sel_e5_large", "e5-large")]),
    ("1.2  Eval encoder (selection = MiniLM-L6)", [
        ("randomized", "bert-tiny (base)"),
        ("eval_mpnet", "mpnet-base"),
        ("eval_bge_small", "bge-small")]),
    ("1.8 / 3.4  OOD pools and artifact-filtered pool", [
        ("randomized", "primary 2000 (base)"),
        ("ood_2000", "OOD 2000"),
        ("ood_200", "OOD 200"),
        ("clean", "primary minus 4 zalgo")]),
]


def load(sweep: str) -> pd.DataFrame:
    """ipa-vs-fps deltas, rescaled to % of the FPS mean Vendi.

    Vendi lives on a different scale in every eval space (~4 under
    bert-tiny, ~90 under mpnet at 32k), so absolute deltas are not
    comparable across panels; relative deltas are.
    """
    df = pd.read_csv(ESS / f"{sweep}_vs_fps.csv")
    df = df[df["method"] == "iterative_pca"].sort_values("target_budget")
    wide = pd.read_csv(ESS / "summary_vendi_wide.csv")
    fps = wide[wide["sweep"] == sweep].set_index("target_budget")["fps"]
    scale = 100.0 / df["target_budget"].map(fps)
    for c in ("vendi_score_delta", "vendi_score_ci_low", "vendi_score_ci_high"):
        df[c] = df[c] * scale
    return df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharey=True)
    for ax, (title, series) in zip(axes.flat, PANELS):
        n = len(series)
        for j, (sweep, label) in enumerate(series):
            d = load(sweep)
            # small horizontal dodge so CI bars do not overlap
            x = d["target_budget"] * (2 ** ((j - (n - 1) / 2) * 0.06))
            y = d["vendi_score_delta"]
            lo = y - d["vendi_score_ci_low"]
            hi = d["vendi_score_ci_high"] - y
            hollow = sweep == "debug_wide_repro"
            ax.errorbar(x, y, yerr=[lo, hi], color=SERIES[j], lw=2, capsize=0,
                        elinewidth=1.5, marker="o", ms=6,
                        mfc="white" if hollow else SERIES[j], mew=1.5,
                        label=label, zorder=3)
        ax.axhline(0, color=MUTED, lw=1, zorder=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks([2048, 4096, 8192, 16384, 32768])
        ax.set_xticklabels(["2k", "4k", "8k", "16k", "32k"])
        ax.grid(axis="y", color=GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    for ax in axes[1]:
        ax.set_xlabel("token budget")
    for ax in axes[:, 0]:
        ax.set_ylabel("Δ Vendi, % of FPS")
    fig.suptitle("Iterative PCA vs FPS: seed-paired Δ Vendi (% of FPS) with 95% bootstrap CI",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "essential_delta_vendi.png", dpi=150)
    print("wrote", OUT / "essential_delta_vendi.png")


if __name__ == "__main__":
    main()
