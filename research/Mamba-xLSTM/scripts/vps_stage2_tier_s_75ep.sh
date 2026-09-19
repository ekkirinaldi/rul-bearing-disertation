#!/usr/bin/env bash
# Stage 2 — Tier-S deep dive on the GPU VPS.
#
# Models     : mamba_xlstm_net + nbeats_xlstm_rul (run in parallel, batch 256)
#              sparse_gate_tcn_rul (run alone afterwards, batch 512)
# Datasets   : PHM2012 then XJTU-SY (sequential between datasets)
# Budget     : 75 epochs, bf16-mixed (configs/train/cloud_full_75.yaml)
# Seed       : 42
# Total runs : 6 (3 models × 2 datasets)
#
# Usage (on the VPS, from Mamba-xLSTM/ after ``source .venv/bin/activate``):
#   chmod +x scripts/vps_stage2_tier_s_75ep.sh
#   nohup ./scripts/vps_stage2_tier_s_75ep.sh > ~/vps_stage2_tier_s_75ep.log 2>&1 &
#
# Laptop side: poll the remote log; once each report line appears, rsync
# Mamba-xLSTM/results/ back down.

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

if [[ ! -d "${ROOT}/.venv" ]]; then
  echo "ERROR: ${ROOT}/.venv missing — run scripts/bootstrap_gpu_vps.sh first." >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${ROOT}/.venv/bin/activate"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TRAIN="${ROOT}/configs/train/cloud_full_75.yaml"
ABL="${ROOT}/configs/ablation/gpu_throughput.yaml"
SEED="${SEED:-42}"

if [[ ! -f "${TRAIN}" ]]; then
  echo "ERROR: ${TRAIN} not found. Sync the latest Mamba-xLSTM/ from the laptop." >&2
  exit 1
fi

_pair_parallel() {
  local dataset_key="$1"
  echo "=== [${dataset_key}] phase A: mamba_xlstm_net + nbeats_xlstm_rul (parallel, batch 256) ==="
  python -u scripts/run_algorithm_comparison.py \
    --datasets "${dataset_key}" \
    --models mamba_xlstm_net nbeats_xlstm_rul \
    --train "${TRAIN}" \
    --ablation "${ABL}" \
    --data-batch-size 256 \
    --parallel-models \
    --seed "${SEED}" \
    --report-name "stage2_pair_${dataset_key}_s${SEED}" \
    --no-figures
}

_sparse_solo() {
  local dataset_key="$1"
  echo "=== [${dataset_key}] phase B: sparse_gate_tcn_rul (solo, batch 512) ==="
  python -u scripts/run_algorithm_comparison.py \
    --datasets "${dataset_key}" \
    --models sparse_gate_tcn_rul \
    --train "${TRAIN}" \
    --ablation "${ABL}" \
    --data-batch-size 512 \
    --seed "${SEED}" \
    --report-name "stage2_sparse_${dataset_key}_s${SEED}" \
    --no-figures
}

for dataset_key in phm2012 xjtusy; do
  _pair_parallel "${dataset_key}"
  _sparse_solo  "${dataset_key}"
done

echo "=== Stage 2 deep-dive complete (seed=${SEED}). ==="
echo "Per-dataset reports:"
ls -1 "${ROOT}/results/reports/" | grep -E "^stage2_(pair|sparse)_(phm2012|xjtusy)_s${SEED}\.html$" || true
