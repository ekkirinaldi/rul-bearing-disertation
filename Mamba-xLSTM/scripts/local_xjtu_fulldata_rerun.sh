#!/usr/bin/env bash
# Re-run XJTU-SY Tier-S experiments with repaired full dataset (all 3 conditions).
# Uses configs/data/xjtu_sy_available_full.yaml (9 train / 3 val / 3 test bearings).
# Matches dissertation stage-2 GPU recipe (scripts/vps_stage2_tier_s_75ep.sh) for XJTU only.
#
# Reference configs (frozen snapshots):
#   results/_comparison_configs/xjtusy_mamba_xlstm_net_s{42,43,44}_merged.yaml
#   results/_comparison_configs/xjtusy_nbeats_xlstm_rul_s{42,43,44}_merged.yaml
#   results/_comparison_configs/xjtusy_sparse_gate_tcn_rul_s{42,43,44}_merged.yaml
#
# Usage:
#   cd Mamba-xLSTM && source .venv/bin/activate
#   chmod +x scripts/local_xjtu_fulldata_rerun.sh
#   nohup ./scripts/local_xjtu_fulldata_rerun.sh > ~/xjtu_fulldata_rerun.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

# shellcheck source=/dev/null
source "${ROOT}/.venv/bin/activate"
export PYTHONUNBUFFERED=1

TRAIN="${ROOT}/configs/train/cloud_full_75.yaml"
# Mac MPS: keep num_workers=0 from configs/data/xjtu_sy_available_full.yaml.
# On CUDA VPS use configs/ablation/gpu_throughput.yaml instead (see vps_stage2_tier_s_75ep.sh).
ABL=""

_pair_serial() {
  local seed="$1"
  echo "=== [xjtusy seed=${seed}] mamba_xlstm_net (batch 256) ==="
  python -u scripts/run_algorithm_comparison.py \
    --datasets xjtusy \
    --models mamba_xlstm_net \
    --train "${TRAIN}" \
    ${ABL:+--ablation "${ABL}"} \
    --data-batch-size 256 \
    --seed "${seed}" \
    --report-name "xjtu_fulldata_pair_s${seed}" \
    --no-figures \
    --skip-report
  echo "=== [xjtusy seed=${seed}] nbeats_xlstm_rul (batch 256) ==="
  python -u scripts/run_algorithm_comparison.py \
    --datasets xjtusy \
    --models nbeats_xlstm_rul \
    --train "${TRAIN}" \
    ${ABL:+--ablation "${ABL}"} \
    --data-batch-size 256 \
    --seed "${seed}" \
    --report-name "xjtu_fulldata_pair_s${seed}" \
    --no-figures \
    --skip-report
}

_sparse_solo() {
  local seed="$1"
  echo "=== [xjtusy seed=${seed}] sparse_gate_tcn_rul (batch 512) ==="
  python -u scripts/run_algorithm_comparison.py \
    --datasets xjtusy \
    --models sparse_gate_tcn_rul \
    --train "${TRAIN}" \
    ${ABL:+--ablation "${ABL}"} \
    --data-batch-size 512 \
    --seed "${seed}" \
    --report-name "xjtu_fulldata_sparse_s${seed}" \
    --no-figures \
    --skip-report
}

for seed in 42 43 44; do
  _pair_serial "${seed}"
  _sparse_solo  "${seed}"
done

# Build combined HTML reports after all runs finish.
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

echo "=== XJTU full-data rerun complete (seeds 42/43/44). ==="
ls -1 "${ROOT}/results/reports/" | grep -E '^xjtu_fulldata_(pair|sparse)_s(42|43|44)\.html$' || true
