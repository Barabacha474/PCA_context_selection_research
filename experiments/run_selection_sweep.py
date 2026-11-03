"""Minimal end-to-end sweep: encoder -> tokenizer -> selector -> metrics.

This is the walking-skeleton entry point. It uses only offline
components (HashingEncoder, RegexTokenizer) so it runs without network
access or model downloads. The real sweep will grow config
loading around this, but the pipeline shape -- the order of calls and
the boundary between the components -- is already final here.

Run from the project root:

    python experiments/run_selection_sweep.py
"""
from __future__ import annotations

import numpy as np

from pcadiv.data.tokenizer import get_tokenizer
from pcadiv.embeddings.encoders import get_encoder
from pcadiv.evaluation import budget_curve, get_metric
from pcadiv.selection import get_selector
from pcadiv.utils.logging import configure, get_logger


def _synthetic_passages(n: int, vocab_size: int, seed: int) -> list[str]:
    """Generate ``n`` synthetic passages as word sequences."""
    rng = np.random.default_rng(seed)
    vocab = [f"w{i}" for i in range(vocab_size)]
    passages: list[str] = []
    for _ in range(n):
        length = int(rng.integers(20, 120))
        tokens = rng.choice(vocab, size=length).tolist()
        passages.append(" ".join(tokens))
    return passages


def main() -> None:
    configure()
    log = get_logger("run_selection_sweep")

    texts = _synthetic_passages(n=200, vocab_size=500, seed=0)
    log.info("generated %d synthetic passages", len(texts))

    encoder = get_encoder("hashing", dim=128)
    tokenizer = get_tokenizer("regex")

    pool_emb = encoder.encode(texts)
    pool_tokens = tokenizer.count_many(texts)
    log.info(
        "pool_emb shape=%s, pool_tokens total=%d, min=%d, max=%d",
        pool_emb.shape, int(pool_tokens.sum()),
        int(pool_tokens.min()), int(pool_tokens.max()),
    )

    selector = get_selector("random", seed=0)
    result = selector.select(pool_emb, pool_tokens, budget=500)
    log.info(
        "selected %d / %d passages, %d / 500 tokens",
        len(result), len(texts), result.total_tokens,
    )

    metrics = [get_metric("mean_pairwise")]
    df = budget_curve(result, pool_emb, metrics)
    log.info("budget_curve rows=%d, columns=%s", len(df), list(df.columns))

    print("\n--- head ---")
    print(df.head(3).to_string(index=False))
    print("\n--- tail ---")
    print(df.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()