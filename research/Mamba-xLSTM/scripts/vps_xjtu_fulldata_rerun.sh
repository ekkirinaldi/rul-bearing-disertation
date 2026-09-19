#!/usr/bin/env bash
# XJTU-SY Tier-S rerun on CUDA VPS (repaired 3-condition dataset).
# Uses configs/data/xjtu_sy_available_full.yaml (9 train / 3 val / 3 test).
# GPU recipe: cloud_full_75.yaml + gpu_throughput.yaml (bf16-mixed, num_workers=8).
#
# Usage (on VPS, from Mamba-xLSTM/ after bootstrap_gpu_vps.sh):
#   chmod +x scripts/vps_xjtu_fulldata_rerun.sh
#   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#   nohup ./scripts/vps_xjtu_fulldata_rerun.sh > ~/vps_xjtu_fulldata_rerun.log 2>&1 &

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

_pair_parallel() {
  local seed="$1"
  echo "=== [xjtusy seed=${seed}] mamba_xlstm_net + nbeats_xlstm_rul (parallel, batch 256) ==="
  python -u scripts/run_algorithm_comparison.py \
    --datasets xjtusy \
    --models mamba_xlstm_net nbeats_xlstm_rul \
    --train "${TRAIN}" \
    --ablation "${ABL}" \
    --data-batch-size 256 \
    --parallel-models \
    --seed "${seed}" \
    --report-name "xjtu_fulldata_pair_s${seed}" \
    --no-figures \
    --skip-report
}

_sparse_solo() {
  local seed="$1"
  echo "=== [xjtusy seed=${seed}] sparse_gate_tcn_rul (solo, batch 512) ==="
  python -u scripts/run_algorithm_comparison.py \
    --datasets xjtusy \
    --models sparse_gate_tcn_rul \
    --train "${TRAIN}" \
    --ablation "${ABL}" \
    --data-batch-size 512 \
    --seed "${seed}" \
    --report-name "xjtu_fulldata_sparse_s${seed}" \
    --no-figures \
    --skip-report
}

for seed in 42 43 44; do
  _pair_parallel "${seed}"
  _sparse_solo  "${seed}"
done

for seed in 42 43 44; do
  python -u scripts/build_report.py \
    --name "xjtu_fulldata_pair_s${seed}" \
    --runs \
      "$(ls -dt results/runs/*_algorithm_comparison_xjtusy_mamba_xlstm_net_s${seed} | head -1)" \
      "$(ls -dt results/runs/*_algorithm_comparison_xjtusy_nbeats_xlstm_rul_s${seed} | head -1)" \
    --no-figures || true
  python -u scripts/build_report.py \
    --name "xjtu_fulldata_sparse_s${seed}" \
    --runs \
      "$(ls -dt results/runs/*_algorithm_comparison_xjtusy_sparse_gate_tcn_rul_s${seed} | head -1)" \
    --no-figures || true
done

echo "=== XJTU full-data VPS rerun complete (seeds 42/43/44). ==="
ls -1 "${ROOT}/results/reports/" | grep -E '^xjtu_fulldata_(pair|sparse)_s(42|43|44)\.html$' || true
