"""Experiment 3.3 -- downstream LLM evaluation (simplified, LLM-judge).

Question: does a context with higher embedding diversity (Vendi) give an
LLM better material for a game-master style creative task?

Design
------
* Contexts: random / relevance / fps / iterative_pca on the primary pool
  at budget 8192 (~32 passages), seed 0, selector settings identical to
  ``configs/experiments/randomized.yaml``. Contexts are task-agnostic
  (the anchor is the pool centroid, not the task) -- this is the setup
  of the primary sweep, and the question is whether a *broader* library
  sample helps across tasks. A no-context condition ("none") is added
  as a floor.
* Generation: the same model answers every (task, condition); output is
  a ~400-word piece. Temperature 0.7 (creative task), responses cached.
* Judging (blind to condition; judge model/endpoint configurable via
  ``--judge-model`` / ``--judge-base-url``, defaults to the generator):
  (A) pairwise preference for every pair of the four context methods,
      both presentation orders (controls position bias);
  (B) absolute 1-10 scores: idea_variety, originality, coherence,
      overall -- also for "none";
  (C) context utilisation: given the numbered context and the answer,
      list the passages the answer clearly draws on -> breadth of use.
* With a judge different from the generator there is no self-preference
  bias; with the same model it would affect all conditions equally
  (every answer comes from the same generator). Either way an LLM judge
  is not a human.

API: any OpenAI-compatible endpoint. Key from env ``LLM_API_KEY``,
endpoint from ``--base-url``. Every call is cached under
``results/downstream/cache/<sha1>.json``, so re-running is free.

Outputs: ``results/downstream/{task}_{method}.json`` (task, context ids,
answer, scores, utilisation), ``pairwise.csv``, ``absolute.csv``,
``summary.md``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from pcadiv.data.pools import compute_pool_centroid
from pcadiv.embeddings.cache import load_or_encode
from pcadiv.selection.selectors import get_selector
from pcadiv.sweep import build_encoder_from_cfg
from pcadiv.utils.config import load_config

OUT = Path("results/downstream")
CACHE = OUT / "cache"
METHODS = ("random", "relevance", "fps", "iterative_pca")
SELECTOR_CFG = {
    "random": ({}, False),
    "relevance": ({}, True),
    "fps": ({"metric": "euclidean"}, False),
    "iterative_pca": ({"k": 5, "warmup": "random"}, False),
}

TASKS = {
    "dungeon": "Describe, as a game master, the moment the party enters an "
               "ancient flooded dungeon beneath a ruined monastery. Introduce "
               "at least three distinct points of interest or threats.",
    "city": "Invent a city the players arrive in: its history, factions, "
            "a rumour circulating in the market, and one strange local custom.",
    "villain": "Create a memorable antagonist for a long campaign: their "
               "background, motive, methods, and how the players first "
               "encounter their influence.",
    "tavern": "Write a tavern scene with three non-player characters, each "
              "with a hook that could start a different adventure.",
    "journey": "Narrate a week-long journey across hostile terrain, with "
               "varied events on different days.",
    "twist": "Propose a plot twist that recontextualises a simple 'rescue the "
             "merchant's daughter' quest, and describe how the reveal plays out.",
}

GEN_SYSTEM = ("You are an experienced, imaginative tabletop RPG game master. "
              "Write vivid, concrete prose.")
JUDGE_SYSTEM = ("You are a strict, impartial editor evaluating creative writing "
                "for tabletop RPG game masters. Respond with JSON only.")


# ---------------------------------------------------------------------
# cached LLM call
# ---------------------------------------------------------------------

class LLM:
    """Cached chat-completion client.

    The client is created lazily, so a fully cached phase (e.g. the
    generations, once done) never touches the network or needs a valid
    key. ``temperature=None`` omits the field (some endpoints reject it).
    The cache key includes the model name, so answers of different
    models never collide.
    """

    def __init__(self, base_url: str, model: str, key_env: str = "LLM_API_KEY"):
        self.base_url, self.model, self.key_env = base_url, model, key_env
        self._client = None
        CACHE.mkdir(parents=True, exist_ok=True)
        self.usage = {"prompt": 0, "completion": 0, "calls": 0, "cached": 0}

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            key = os.environ.get(self.key_env) or os.environ["LLM_API_KEY"]
            self._client = OpenAI(base_url=self.base_url, api_key=key, timeout=300)
        return self._client

    def __call__(self, system: str, user: str, max_tokens: int,
                 temperature: float | None) -> str:
        payload = {"model": self.model, "system": system, "user": user,
                   "max_tokens": max_tokens, "temperature": temperature}
        key = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        path = CACHE / f"{key}.json"
        if path.exists():
            self.usage["cached"] += 1
            return json.loads(path.read_text(encoding="utf-8"))["text"]
        for attempt in range(5):
            try:
                extra = {} if temperature is None else {"temperature": temperature}
                r = self.client.chat.completions.create(
                    model=self.model, max_tokens=max_tokens, **extra,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}])
                text = r.choices[0].message.content or ""
                break
            except Exception as e:  # network / rate limit: back off
                if attempt == 4:
                    raise
                time.sleep(5 * 2 ** attempt)
                print(f"retry {attempt + 1}: {e!r}")
        self.usage["calls"] += 1
        if r.usage:
            self.usage["prompt"] += r.usage.prompt_tokens or 0
            self.usage["completion"] += r.usage.completion_tokens or 0
        path.write_text(json.dumps({**payload, "text": text}, ensure_ascii=False),
                        encoding="utf-8")
        return text


def parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON in judge output: {text[:200]!r}")
    return json.loads(m.group(0))


# ---------------------------------------------------------------------
# contexts
# ---------------------------------------------------------------------

def build_contexts(args) -> dict[str, list[int]]:
    pool = pd.read_parquet(args.pool).reset_index(drop=True)
    enc = build_encoder_from_cfg(load_config(args.encoder))
    emb = load_or_encode(enc, pool["text"].tolist(), args.cache_dir)
    toks = pool["n_tokens"].to_numpy(dtype=np.int64)
    core = compute_pool_centroid(emb, normalize=True)
    ctx = {}
    for m in METHODS:
        params, pass_anchor = SELECTOR_CFG[m]
        res = get_selector(m, seed=args.seed, **params).select(
            emb, toks, budget=args.budget, core_emb=core if pass_anchor else None)
        ctx[m] = res.indices.tolist()
    return pool, ctx


def context_block(pool: pd.DataFrame, idx: list[int]) -> str:
    return "\n\n".join(f"[{j}] {pool.loc[i, 'text'].strip()}"
                       for j, i in enumerate(idx, 1))


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pool", default="data/pools/primary.parquet")
    p.add_argument("--encoder", default="configs/embed/selection_encoder.yaml")
    p.add_argument("--cache-dir", default="data/embeddings")
    p.add_argument("--budget", type=int, default=8192)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base-url", default="https://www.ai-funpay.ru/v1")
    p.add_argument("--model", default="claude-opus-5-5",
                   help="generator model")
    p.add_argument("--judge-base-url", default=None,
                   help="judge endpoint (default: --base-url); key from "
                        "env LLM_JUDGE_API_KEY, falling back to LLM_API_KEY")
    p.add_argument("--judge-model", default=None,
                   help="judge model (default: same as --model)")
    p.add_argument("--judge-no-temperature", action="store_true",
                   help="omit temperature in judge calls (endpoint rejects it)")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    llm = LLM(args.base_url, args.model)
    args.judge_model = args.judge_model or args.model
    judge = LLM(args.judge_base_url or args.base_url, args.judge_model,
                key_env="LLM_JUDGE_API_KEY")
    jt = None if args.judge_no_temperature else 0
    pool, ctx = build_contexts(args)
    conditions = list(METHODS) + ["none"]

    # ---- (1) generation -------------------------------------------
    def gen(job):
        task, cond = job
        if cond == "none":
            user = f"Task: {TASKS[task]}\n\nWrite about 400 words."
        else:
            user = ("Below are passages from a library of fiction. Use them as "
                    "inspiration and source material for the task.\n\n"
                    f"{context_block(pool, ctx[cond])}\n\n"
                    f"Task: {TASKS[task]}\n\nWrite about 400 words.")
        return job, llm(GEN_SYSTEM, user, max_tokens=1200, temperature=0.7)

    jobs = [(t, c) for t in TASKS for c in conditions]
    with ThreadPoolExecutor(args.workers) as ex:
        answers = dict(ex.map(gen, jobs))
    print(f"generated {len(answers)} answers; usage so far {llm.usage}")

    # ---- (2) absolute scores --------------------------------------
    def absolute(job):
        task, cond = job
        user = (f"Task given to the writer:\n{TASKS[task]}\n\nResponse:\n"
                f"{answers[job]}\n\nScore the response from 1 to 10 on each "
                "criterion: idea_variety (number and range of distinct ideas, "
                "places, characters, hooks), originality (avoids cliches), "
                "coherence (consistent, readable), overall (usefulness to a game "
                'master). Return JSON: {"idea_variety": int, "originality": int, '
                '"coherence": int, "overall": int}')
        return job, parse_json(judge(JUDGE_SYSTEM, user, max_tokens=200, temperature=jt))

    with ThreadPoolExecutor(args.workers) as ex:
        abs_scores = dict(ex.map(absolute, jobs))

    # ---- (3) pairwise, both orders --------------------------------
    pair_jobs = [(t, a, b) for t in TASKS for a, b in combinations(METHODS, 2)]
    pair_jobs += [(t, b, a) for t, a, b in pair_jobs]

    def pairwise(job):
        task, first, second = job
        user = (f"Task given to two writers:\n{TASKS[task]}\n\n"
                f"Response A:\n{answers[(task, first)]}\n\n"
                f"Response B:\n{answers[(task, second)]}\n\n"
                "Which response gives a game master richer, more varied and more "
                "original material while staying coherent? Return JSON: "
                '{"winner": "A" | "B" | "tie", "reason": "<one sentence>"}')
        out = parse_json(judge(JUDGE_SYSTEM, user, max_tokens=200, temperature=jt))
        w = str(out.get("winner", "tie")).strip().upper()
        winner = first if w == "A" else second if w == "B" else "tie"
        return {"task": task, "first": first, "second": second,
                "winner": winner, "reason": out.get("reason", "")}

    with ThreadPoolExecutor(args.workers) as ex:
        pairs = pd.DataFrame(list(ex.map(pairwise, pair_jobs)))
    pairs.to_csv(OUT / "pairwise.csv", index=False)

    # ---- (4) context utilisation ----------------------------------
    util_jobs = [(t, m) for t in TASKS for m in METHODS]

    def utilisation(job):
        task, m = job
        user = (f"Numbered context passages:\n\n{context_block(pool, ctx[m])}\n\n"
                f"Response written using this context:\n{answers[job]}\n\n"
                "List the numbers of passages whose specific content (an image, "
                "character, setting, event or idea) is clearly reflected in the "
                "response. Be conservative: do not count generic overlap. Return "
                'JSON: {"used": [int, ...]}')
        out = parse_json(judge(JUDGE_SYSTEM, user, max_tokens=300, temperature=jt))
        used = sorted({int(u) for u in out.get("used", [])
                       if 1 <= int(u) <= len(ctx[m])})
        return job, used

    with ThreadPoolExecutor(args.workers) as ex:
        util = dict(ex.map(utilisation, util_jobs))

    # ---- save per-item JSON + tables -----------------------------
    abs_rows = []
    for (task, cond), sc in abs_scores.items():
        rec = {"task": task, "method": cond, "task_text": TASKS[task],
               "context_pool_indices": ctx.get(cond, []),
               "answer": answers[(task, cond)], "scores_llm_judge": sc,
               "context_used": util.get((task, cond)),
               "judge_model": args.judge_model, "generator_model": args.model}
        (OUT / f"{task}_{cond}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        abs_rows.append({"task": task, "method": cond, **sc,
                         "n_ctx": len(ctx.get(cond, [])),
                         "n_used": len(util[(task, cond)]) if (task, cond) in util else np.nan,
                         "answer_words": len(answers[(task, cond)].split())})
    absdf = pd.DataFrame(abs_rows)
    absdf.to_csv(OUT / "absolute.csv", index=False)

    # win rates (tie = 0.5) over all pairwise judgements involving a method
    win = {m: [] for m in METHODS}
    for _, r in pairs.iterrows():
        for m in (r["first"], r["second"]):
            win[m].append(1.0 if r["winner"] == m else 0.5 if r["winner"] == "tie" else 0.0)
    head = {}
    for a, b in combinations(METHODS, 2):
        sub = pairs[((pairs["first"] == a) & (pairs["second"] == b)) |
                    ((pairs["first"] == b) & (pairs["second"] == a))]
        sa = ((sub["winner"] == a) + 0.5 * (sub["winner"] == "tie")).mean()
        head[f"{a} vs {b}"] = (sa, len(sub))
    pos_bias = (pairs["winner"] == pairs["first"]).mean()

    lines = ["# Downstream LLM evaluation (exp 3.3)\n",
             f"Generator: {args.model}; judge: {args.judge_model}; budget {args.budget}; "
             f"seed {args.seed}; tasks {len(TASKS)}\n",
             "## Absolute scores (mean over tasks, 1-10)\n",
             absdf.groupby("method")[["idea_variety", "originality", "coherence",
                                       "overall", "n_used", "n_ctx", "answer_words"]]
             .mean().round(2).to_markdown(), "\n## Pairwise win rate (tie = 0.5)\n",
             pd.Series({m: np.mean(v) for m, v in win.items()}, name="win_rate")
             .round(3).to_frame().to_markdown(), "\n## Head-to-head (share for first-named)\n",
             pd.DataFrame(head, index=["share_first", "n"]).T.round(3).to_markdown(),
             f"\nPosition bias: first-presented response wins {pos_bias:.2%} "
             "of pairwise judgements.\n",
             f"\nAPI usage this run: generator {llm.usage}, judge {judge.usage}\n"]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
