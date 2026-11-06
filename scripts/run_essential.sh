#!/usr/bin/env bash
# Essential experiments (docs/05_essential_results.md): 3.1, 1.1, 1.2, 1.8.
# Prerequisite: data prepared as in docs Findings section 1, plus the
# 2000-passage OOD pool:
#   python experiments/run_data_prep.py --override source.limit=5000 \
#       --override pools.primary.n_passages=2000 --override pools.ood.n_passages=200
#   python experiments/run_data_prep.py --skip-download --override source.limit=5000 \
#       --override pools.primary.n_passages=2000 \
#       --override pools.primary.out_path=data/pools/primary_check.parquet \
#       --override pools.ood.n_passages=2000 --override pools.ood.out_path=data/pools/ood_2000.parquet
# Every sweep uses configs/experiments/randomized.yaml (pass_anchor: false,
# 10 seeds, reference fps). Each sweep takes ~20 s once embeddings are cached.
set -euo pipefail
PY=${PY:-python}
CFG=configs/experiments/randomized.yaml
A=configs/embed/alt
sweep() {  # name sel_cfg eval_cfg pool
  echo "=== $1"
  $PY experiments/run_selection_sweep.py --config $CFG --sweep-name "$1" --pool "$4" \
      --override encoders.selection="$2" --override encoders.eval="$3"
}
SEL0=configs/embed/selection_encoder.yaml
EV0=configs/embed/eval_encoder.yaml
P=data/pools/primary.parquet

# 3.1 randomized warm-up (baseline encoders)
sweep randomized              $SEL0               $EV0               $P
# 1.1 other selection encoders (eval = bert-tiny)
sweep sel_minilm_l3           $A/sel_minilm_l3.yaml $EV0             $P
sweep sel_mpnet               $A/mpnet.yaml         $EV0             $P
sweep sel_bge_large           $A/sel_bge_large.yaml $EV0             $P
sweep sel_e5_large            $A/sel_e5_large.yaml  $EV0             $P
# 1.2 other eval encoders (selection = MiniLM-L6)
sweep eval_mpnet              $SEL0               $A/mpnet.yaml        $P
sweep eval_bge_small          $SEL0               $A/eval_bge_small.yaml $P
# 1.1 x 1.2 cross: every other selection encoder under the stronger evals
sweep x_minilm_l3__mpnet      $A/sel_minilm_l3.yaml $A/mpnet.yaml          $P
sweep x_minilm_l3__bge_small  $A/sel_minilm_l3.yaml $A/eval_bge_small.yaml $P
sweep x_mpnet__bge_small      $A/mpnet.yaml         $A/eval_bge_small.yaml $P
sweep x_bge_large__mpnet      $A/sel_bge_large.yaml $A/mpnet.yaml          $P
sweep x_bge_large__bge_small  $A/sel_bge_large.yaml $A/eval_bge_small.yaml $P
sweep x_e5_large__mpnet       $A/sel_e5_large.yaml  $A/mpnet.yaml          $P
sweep x_e5_large__bge_small   $A/sel_e5_large.yaml  $A/eval_bge_small.yaml $P
# 1.8 OOD pools
sweep ood_200                 $SEL0               $EV0               data/pools/ood.parquet
sweep ood_2000                $SEL0               $EV0               data/pools/ood_2000.parquet
sweep ood_2000__eval_mpnet    $SEL0               $A/mpnet.yaml      data/pools/ood_2000.parquet

# spectra of every selection space (mechanistic context for 1.1 / 1.8)
for e in sel_minilm_l3 mpnet sel_bge_large sel_e5_large; do
  $PY experiments/run_spectral_analysis.py --pool $P --encoder $A/$e.yaml --out-dir results/spectral_$e
done
$PY experiments/run_spectral_analysis.py --pool data/pools/ood_2000.parquet --encoder $SEL0 --out-dir results/spectral_ood_2000
echo ALL_DONE

# 3.4 follow-up: artifact-filtered pool (drops 4 zalgo / unicode-styled passages)
$PY experiments/make_clean_pool.py --pool data/pools/primary.parquet --out data/pools/primary_clean.parquet
sweep clean                   $SEL0               $EV0               data/pools/primary_clean.parquet
sweep clean__eval_mpnet       $SEL0               $A/mpnet.yaml      data/pools/primary_clean.parquet
