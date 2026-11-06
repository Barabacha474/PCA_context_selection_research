"""Experiment 3.4 -- qualitative coherence check.

Two sub-commands:

``extract``
    Selects passages at a large budget (default 32768) with random,
    relevance, fps and iterative_pca on the primary pool, using the same
    selector settings as ``configs/experiments/randomized.yaml``
    (fps / iterative_pca: no anchor, seeded random warm-up). Writes

    * ``results/qualitative/<method>.parquet`` -- seed-0 selection, in
      selection order, with automatic artifact features per passage;
    * ``results/qualitative/artifact_stats.csv`` -- per-method mean of
      every artifact feature over seeds 0..9 (random/fps/ipa) and the
      pool baseline, so "does ipa pick formatting junk?" is answered on
      all seeds, not just the one we read;
    * ``results/qualitative/overlap_seed0.csv`` -- ipa-only / fps-only /
      shared split at seed 0;
    * ``results/qualitative/blind_sample.md`` + ``blind_key.csv`` -- 30
      passages per method, shuffled, method hidden, for blind reading.
      Passages selected by several methods appear once and count for
      each of them.

``score``
    Joins ``blind_annotations.csv`` (filled in by the reader: id, label,
    theme) with the key and prints per-method label shares. Themes are
    free-text notes for the report; every read passage got a distinct
    theme, so they do not discriminate between methods.

Labels used in annotations (scale from docs, experiment 3.4):
    ok       -- meaningful fiction passage
    style    -- stylistic outlier (odd format: poem, list, script, all-caps,
                chat log, meta-commentary...), but readable
    garbage  -- broken / truncated / non-text / off-topic meta (e.g. "Edit:
                thanks for the gold"), carries no fiction content
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from pcadiv.data.pools import compute_pool_centroid
from pcadiv.embeddings.cache import load_or_encode
from pcadiv.selection.selectors import get_selector
from pcadiv.sweep import build_encoder_from_cfg
from pcadiv.utils.config import load_config

OUT = Path("results/qualitative")
METHODS = ("random", "relevance", "fps", "iterative_pca")
SELECTOR_CFG = {
    "random": ({}, False),
    "relevance": ({}, True),
    "fps": ({"metric": "euclidean"}, False),
    "iterative_pca": ({"k": 5, "warmup": "random"}, False),
}
WORD = re.compile(r"[A-Za-z']+")


def artifact_features(text: str) -> dict[str, float]:
    """Cheap surface features that flag formatting artifacts / junk."""
    n = max(len(text), 1)
    words = WORD.findall(text)
    nw = max(len(words), 1)
    stripped = text.strip()
    return {
        "n_chars": float(len(text)),
        "frac_alpha": sum(c.isalpha() for c in text) / n,
        "frac_upper": sum(c.isupper() for c in text) / max(sum(c.isalpha() for c in text), 1),
        "frac_digit": sum(c.isdigit() for c in text) / n,
        "frac_punct": sum((not c.isalnum()) and (not c.isspace()) for c in text) / n,
        "frac_non_ascii": sum(ord(c) > 127 for c in text) / n,
        "n_newline_tokens": float(text.count("<newline>")),
        "type_token_ratio": len({w.lower() for w in words}) / nw,
        "mean_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        "frac_quote_chars": (text.count('"') + text.count("“") + text.count("”")) / n,
        "starts_lowercase": float(bool(stripped) and stripped[0].islower()),
        "has_meta_words": float(bool(re.search(
            r"\b(edit|upvote|reddit|gold|thanks for reading|part \d|wp|prompt)\b",
            text, re.I))),
    }


def _select(method: str, emb, toks, budget: int, seed: int, core):
    params, pass_anchor = SELECTOR_CFG[method]
    sel = get_selector(method, seed=seed, **params)
    return sel.select(emb, toks, budget=budget, core_emb=core if pass_anchor else None)


def extract(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pool = pd.read_parquet(args.pool).reset_index(drop=True)
    enc = build_encoder_from_cfg(load_config(args.encoder))
    emb = load_or_encode(enc, pool["text"].tolist(), args.cache_dir)
    toks = pool["n_tokens"].to_numpy(dtype=np.int64)
    core = compute_pool_centroid(emb, normalize=True)

    feats = pd.DataFrame([artifact_features(t) for t in pool["text"]])
    feat_cols = list(feats.columns)

    stat_rows = [{"method": "POOL", "seed": -1, **feats.mean().to_dict()}]
    seed0: dict[str, np.ndarray] = {}
    for method in METHODS:
        seeds = [0] if method == "relevance" else list(range(10))
        for seed in seeds:
            res = _select(method, emb, toks, args.budget, seed, core)
            idx = res.indices
            stat_rows.append({"method": method, "seed": seed,
                              **feats.iloc[idx].mean().to_dict()})
            if seed == 0:
                seed0[method] = idx
                sub = pool.iloc[idx].copy()
                sub.insert(0, "select_order", range(len(idx)))
                sub.insert(1, "pool_index", idx)
                sub = pd.concat([sub.reset_index(drop=True),
                                 feats.iloc[idx].reset_index(drop=True)], axis=1)
                sub.to_parquet(OUT / f"{method}.parquet", index=False)
                print(f"{method}: {len(idx)} passages, {int(res.total_tokens)} tokens")

    stats = pd.DataFrame(stat_rows)
    per_method = stats.groupby("method", sort=False)[feat_cols].mean()
    per_method.insert(0, "n_seeds", stats.groupby("method", sort=False).size())
    per_method.to_csv(OUT / "artifact_stats.csv")
    pd.set_option("display.width", 220)
    print(per_method.round(3).to_string())

    f, i = set(seed0["fps"].tolist()), set(seed0["iterative_pca"].tolist())
    ov = pd.DataFrame([
        {"group": "shared", "n": len(f & i), **feats.iloc[sorted(f & i)].mean().to_dict()},
        {"group": "ipa_only", "n": len(i - f), **feats.iloc[sorted(i - f)].mean().to_dict()},
        {"group": "fps_only", "n": len(f - i), **feats.iloc[sorted(f - i)].mean().to_dict()},
    ])
    ov.to_csv(OUT / "overlap_seed0.csv", index=False)
    print(ov.round(3).to_string(index=False))

    # ---- blind sample ---------------------------------------------
    rng = np.random.default_rng(12345)
    membership: dict[int, set[str]] = {}
    for method in METHODS:
        idx = seed0[method]
        take = rng.choice(idx, size=min(args.n_read, len(idx)), replace=False)
        for t in take.tolist():
            membership.setdefault(int(t), set()).add(method)
    # A passage counts for every method that selected it at seed 0,
    # not only the one it was sampled for -- the label is a property of
    # the passage.
    for t in membership:
        for method in METHODS:
            if t in set(seed0[method].tolist()):
                membership[t].add(method)
    items = list(membership)
    rng.shuffle(items)
    key_rows, md = [], ["# Blind sample (methods hidden)\n",
                        "Label each id: ok / style / garbage, plus a 2-5 word theme.\n"]
    for j, t in enumerate(items, 1):
        qid = f"Q{j:03d}"
        key_rows.append({"id": qid, "pool_index": t,
                         "methods": ",".join(sorted(membership[t]))})
        md.append(f"\n## {qid}\n\n{pool.loc[t, 'text']}\n")
    pd.DataFrame(key_rows).to_csv(OUT / "blind_key.csv", index=False)
    (OUT / "blind_sample.md").write_text("\n".join(md), encoding="utf-8")
    print(f"blind sample: {len(items)} unique passages")


def score(args) -> None:
    key = pd.read_csv(OUT / "blind_key.csv")
    ann = pd.read_csv(OUT / "blind_annotations.csv")
    df = key.merge(ann, on="id", how="inner")
    rows = []
    for method in METHODS:
        sub = df[df["methods"].str.split(",").apply(lambda m: method in m)]
        shares = sub["label"].value_counts(normalize=True)
        rows.append({
            "method": method, "n_read": len(sub),
            "ok": shares.get("ok", 0.0), "style": shares.get("style", 0.0),
            "garbage": shares.get("garbage", 0.0),
            "n_garbage": int((sub["label"] == "garbage").sum()),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "blind_scores.csv", index=False)
    print(out.round(3).to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--pool", default="data/pools/primary.parquet")
    e.add_argument("--encoder", default="configs/embed/selection_encoder.yaml")
    e.add_argument("--cache-dir", default="data/embeddings")
    e.add_argument("--budget", type=int, default=32768)
    e.add_argument("--n-read", type=int, default=30)
    sub.add_parser("score")
    args = p.parse_args()
    extract(args) if args.cmd == "extract" else score(args)


if __name__ == "__main__":
    main()
