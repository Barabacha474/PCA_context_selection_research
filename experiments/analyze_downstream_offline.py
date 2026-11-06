"""Offline (no-API) analysis of the exp 3.3 generations.

The LLM-judge phase of run_downstream.py stopped when the API quota ran
out, but all generations (6 tasks x 5 conditions) are cached. This
script measures what can be measured without a judge:

* ``ctx_used``  -- number of context passages whose *rare* content words
  reappear in the answer (>= 2 shared words that occur in < 1% of pool
  passages: names, invented terms, unusual nouns). A conservative,
  lexical proxy for "the answer drew on this passage".
* ``ctx_used_frac`` -- the same divided by the number of passages.
* ``distinct2`` -- distinct bigram ratio of the answer (lexical variety).
* ``sim_to_none`` -- cosine similarity (eval encoder all-mpnet-base-v2,
  independent of the MiniLM selection encoder) between the answer and
  the no-context answer for the same task: how far the context moved
  the output away from the model's default.
* ``answer_vendi`` -- Vendi score (cosine kernel) of a method's 6
  answers: are answers produced from this context more varied?

Writes ``results/downstream/offline_metrics.csv`` and prints a summary.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from pcadiv.embeddings.cache import load_or_encode
from pcadiv.evaluation.metrics import get_metric
from pcadiv.sweep import build_encoder_from_cfg
from pcadiv.utils.config import load_config

import sys

sys.path.insert(0, str(Path(__file__).parent))
from run_downstream import METHODS, TASKS, build_contexts, context_block  # noqa: E402

CACHE = Path("results/downstream/cache")
WORD = re.compile(r"[A-Za-z][A-Za-z'-]{4,}")
PREFIX = re.compile(r"^\s*I am Claude[^\n]*\n+", re.I)  # proxy-injected line


def words(text: str) -> set[str]:
    return {w.lower().strip("'-") for w in WORD.findall(text)}


def main() -> None:
    class A:  # same defaults as run_downstream
        pool = "data/pools/primary.parquet"
        encoder = "configs/embed/selection_encoder.yaml"
        cache_dir = "data/embeddings"
        budget = 8192
        seed = 0
    pool, ctx = build_contexts(A)

    gens = {}
    for f in CACHE.glob("*.json"):
        rec = json.loads(f.read_text(encoding="utf-8"))
        if rec["max_tokens"] != 1200:
            continue
        task = next(t for t, txt in TASKS.items() if txt in rec["user"])
        if "library of fiction" not in rec["user"]:
            cond = "none"
        else:
            # random / fps / ipa share the seed-0 first passage, so match
            # on the whole numbered context block, not its first entry.
            cond = next(m for m in METHODS
                        if context_block(pool, ctx[m]) in rec["user"])
        gens[(task, cond)] = PREFIX.sub("", rec["text"]).strip()
    assert len(gens) == len(TASKS) * (len(METHODS) + 1), len(gens)

    df_counts = Counter()
    for t in pool["text"]:
        df_counts.update(words(t))
    rare_cut = 0.01 * len(pool)

    enc = build_encoder_from_cfg(load_config("configs/embed/alt/mpnet.yaml"))
    keys = sorted(gens)
    emb = load_or_encode(enc, [gens[k] for k in keys], None)
    E = dict(zip(keys, emb))

    rows = []
    for (task, cond), ans in gens.items():
        aw = words(ans)
        toks = ans.lower().split()
        bigrams = list(zip(toks, toks[1:]))
        row = {"task": task, "method": cond, "answer_words": len(toks),
               "distinct2": len(set(bigrams)) / max(len(bigrams), 1),
               "sim_to_none": float(E[(task, cond)] @ E[(task, "none")])}
        if cond != "none":
            used = 0
            for i in ctx[cond]:
                rare = {w for w in words(pool.loc[i, "text"]) if df_counts[w] < rare_cut}
                used += len(rare & aw) >= 2
            row.update(n_ctx=len(ctx[cond]), ctx_used=used,
                       ctx_used_frac=used / len(ctx[cond]))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv("results/downstream/offline_metrics.csv", index=False)

    vendi = get_metric("vendi", kernel="cosine", gamma=1.0)
    av = {c: vendi(np.stack([E[(t, c)] for t in TASKS]))["vendi_score"]
          for c in list(METHODS) + ["none"]}
    summ = df.groupby("method")[["n_ctx", "ctx_used", "ctx_used_frac", "distinct2",
                                 "sim_to_none", "answer_words"]].mean()
    summ["answer_vendi"] = pd.Series(av)
    summ = summ.loc[["none", "relevance", "random", "fps", "iterative_pca"]]
    summ.to_csv("results/downstream/offline_summary.csv")
    pd.set_option("display.width", 200)
    print(summ.round(3).to_string())
    piv = df[df.method != "none"].pivot(index="task", columns="method", values="ctx_used")
    print(piv.to_string())


if __name__ == "__main__":
    main()
