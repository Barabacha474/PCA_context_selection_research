"""Build an artifact-filtered copy of a pool (robustness check for exp 3.4).

The qualitative check found that diversity selectors pick a handful of
"zalgo" / unicode-styled passages (combining diacritics, small-caps
letters) very early. To test whether the ipa-vs-fps result depends on
them, we drop every passage whose share of non-ASCII characters exceeds
``--max-non-ascii`` (default 0.1; this removes exactly the 4 such
passages from the 2000-passage primary pool, i.e. 0.2%) and re-run the
randomized sweep on the result. The rule is mechanical, not hand-picked.

Usage::

    python experiments/make_clean_pool.py --pool data/pools/primary.parquet \
        --out data/pools/primary_clean.parquet
"""
from __future__ import annotations

import argparse

import pandas as pd

from pcadiv.data.pools import save_pool


def non_ascii_frac(text: str) -> float:
    return sum(ord(c) > 127 for c in text) / max(len(text), 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pool", default="data/pools/primary.parquet")
    p.add_argument("--out", default="data/pools/primary_clean.parquet")
    p.add_argument("--max-non-ascii", type=float, default=0.1)
    args = p.parse_args()

    pool = pd.read_parquet(args.pool)
    frac = pool["text"].map(non_ascii_frac)
    keep = frac <= args.max_non_ascii
    dropped = pool.loc[~keep, "passage_id"].tolist()
    clean = pool.loc[keep].reset_index(drop=True)
    save_pool(clean, args.out, seed=0, extra_meta={
        "derived_from": args.pool,
        "filter": f"non_ascii_frac <= {args.max_non_ascii}",
        "dropped_passage_ids": dropped,
    })
    print(f"kept {len(clean)} / {len(pool)}; dropped {dropped}")


if __name__ == "__main__":
    main()
