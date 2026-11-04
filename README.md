# pcadiv

PCA-guided selection of diverse fictional context under a fixed token budget.

Research project on whether iteratively updated PCA subspaces can be used
as a novelty measure for selecting semantically diverse context for LLMs.

## Key finding so far

Iterative PCA beats Farthest Point Sampling on diversity metrics (Vendi
Score, coverage radius) at token budgets >= 8192, while selecting 60-65%
different passages. Below 8192 tokens the two methods are mathematically
equivalent on standard embedding spaces. See `docs/04_findings.md` for
details.

## Quick install

```bash
git clone <repo-url> pca_context_selection
cd pca_context_selection
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev,data,encoders]"
pytest tests/ -v
```

Requires Python 3.10-3.12 (3.13 works with caveats, see `docs/02_project.md`).

## Quickstart

```bash
# Prepare data (downloads WritingPrompts, ~5 min)
python experiments/run_data_prep.py --override source.limit=5000

# Run a selection sweep
python experiments/run_selection_sweep.py --sweep-name my_run

# Inspect results
python -c "
import pandas as pd
agg = pd.read_parquet('results/metrics/my_run_agg.parquet')
print(agg.pivot(index='target_budget', columns='method',
                values='vendi_score_mean').round(3))
"
```

Full walkthrough in `docs/02_project.md`, section 2.

## What's inside

Seven selectors, three metrics, four encoder backends — all swappable
via YAML, no code changes required:

- **Selectors**: `random`, `relevance`, `mmr`, `mmr_dynamic`, `fps`,
  `static_pca`, `iterative_pca`
- **Metrics**: `mean_pairwise`, `coverage`, `vendi`
- **Encoders**: `hashing` (offline), `st`, `hf_mean`, `hf_cls`
- **Tokenizers**: `regex` (offline), `tiktoken`, `hf`

## Documentation

NOT SUBMITED YET

| File | Contents |
|------|----------|
| `docs/01_proposal.md` | Scientific description: question, method, baselines, risks (RU) |
| `docs/02_project.md` | Technical reference: architecture, contracts, how to extend (RU) |
| `docs/03_experiments.md` | Planned experiments with priorities and recipes (RU) |
| `docs/04_findings.md` | What's already established — read before running anything (RU) |

**Read `docs/04_findings.md` before running experiments.** It lists what
has already been done and what not to re-run.

## Requirements

- Python 3.10-3.12
- ~5 GB disk (models, cache, dataset)
- CPU sufficient; GPU optional (`device: auto` in encoder config)
- Internet for Hugging Face (model and dataset downloads)

## Testing

```bash
pytest tests/ -v                          # fast, offline
PCADIV_RUN_SLOW=1 pytest tests/ -v        # includes real models
```

~415 tests. Tests that need network or models are skipped by default.

## Project status

Research in progress. Main results in `docs/04_findings.md`. Not yet
published.

## Caveats

Two things that trip people up:

1. **Small budgets hide the effect.** The iterative PCA vs FPS difference
   only appears at budgets >= 8192 tokens (with `passage_tokens=256`).
   If you run with 512-4096, you'll see all methods give nearly identical
   results — that is mathematically expected, not a bug. See
   `docs/04_findings.md`, section 3.

2. **Selection encoder must differ from eval encoder.** The code enforces
   this via fingerprint check and will refuse to run if they match. Using
   the same encoder for both makes the evaluation biased — the method
   would optimize the exact metric it is being measured on.

## Contributing

Internal team project. See `docs/02_project.md`, section 13 (recipes)
for how to add new encoders, tokenizers, selectors, or metrics.

## License

TBD
