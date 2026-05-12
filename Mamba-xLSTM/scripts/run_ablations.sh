#!/usr/bin/env bash
# Run all 7 ablations × N seeds for a given dataset.
#
# Usage:
#   bash scripts/run_ablations.sh phm2012 3
#   bash scripts/run_ablations.sh xjtu_sy 3
#
# The runner streams progress to stdout; per-run summary.json files land in
# results/runs/<timestamp>_<run_id>/.
set -euo pipefail

DATASET="${1:-phm2012}"
N_SEEDS="${2:-3}"

DATA_CFG="configs/data/${DATASET}.yaml"
TRAIN_CFG="configs/train/default.yaml"

case "$DATASET" in
  phm2012) ;;
  xjtu_sy|xjtusy) DATASET="xjtu_sy"; DATA_CFG="configs/data/xjtu_sy.yaml" ;;
  *) echo "Unknown dataset: $DATASET" >&2; exit 1 ;;
esac

echo "Dataset: ${DATASET}"
echo "Seeds  : ${N_SEEDS}"
echo

ABLATIONS=(a1_no_mamba a2_no_xlstm a3_unidir a4_concat a5_lstm a6_no_smooth a7_short_ctx)

for abl in "${ABLATIONS[@]}"; do
  ABL_CFG="configs/ablation/${abl}.yaml"
  for s in $(seq 0 $((N_SEEDS - 1))); do
    SEED=$((42 + s))
    RUN_ID="${abl}_${DATASET}_s${SEED}"
    echo "=== ${RUN_ID} ==="
    python scripts/train.py \
      --data "$DATA_CFG" \
      --model configs/model/mamba_xlstm_net.yaml \
      --ablation "$ABL_CFG" \
      --train "$TRAIN_CFG" \
      --seed "$SEED" \
      --run-id "$RUN_ID"
    echo
  done
done
