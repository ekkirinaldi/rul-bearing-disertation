#!/usr/bin/env bash
# Run on the GPU VPS (CUDA), not on a laptop — keeps long jobs off MPS and uses
# ``gpu_throughput.yaml`` + ``cloud_full_75.yaml`` (default cloud budget).
#
# Usage (on VPS, from ``Mamba-xLSTM/`` after ``source .venv/bin/activate``):
#   ./scripts/vps_nbeats_physics_cloud.sh smoke     # fast_dev_run, ~1–2 min GPU
#   ./scripts/vps_nbeats_physics_cloud.sh compare   # 30-epoch × 3 models × 2 datasets + HTML report
#   ./scripts/vps_nbeats_physics_cloud.sh cloud75   # 75-epoch × physics + N-BEATS-xLSTM × 2 datasets (parallel)
#   ./scripts/vps_nbeats_physics_cloud.sh all       # smoke then compare
#
# From laptop (push code first — see ``.cursor/rules/vps-ssh-key-access.mdc``):
#   rsync -az --delete .../Mamba-xLSTM/ root@HOST:.../Mamba-xLSTM/
#   ssh ... 'cd .../Mamba-xLSTM && chmod +x scripts/vps_nbeats_physics_cloud.sh && ./scripts/vps_nbeats_physics_cloud.sh compare'

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
if [[ ! -d "${ROOT}/.venv" ]]; then
  echo "ERROR: ${ROOT}/.venv missing — run scripts/bootstrap_gpu_vps.sh on this host first." >&2
  exit 1
fi
# shellcheck source=/dev/null
source "${ROOT}/.venv/bin/activate"
export PYTHONUNBUFFERED=1

cmd="${1:-smoke}"

_run_smoke() {
  echo "=== Physics-N-BEATS smoke (fast_dev_run, PHM2012) ==="
  python -u scripts/train.py \
    --data "${ROOT}/configs/data/phm2012.yaml" \
    --model "${ROOT}/configs/model/physics_nbeats_rul.yaml" \
    --train "${ROOT}/configs/train/algorithm_comparison.yaml" \
    --ablation "${ROOT}/configs/ablation/gpu_throughput.yaml" \
    --seed 42 \
    --run-id "vps_smoke_physics_nbeats_s${SEED:-42}" \
    --fast-dev-run \
    --no-figures
}

_run_compare() {
  echo "=== N-BEATS vs Physics-N-BEATS vs N-BEATS-xLSTM (30 epochs, PHM2012 + XJTU-SY) ==="
  python -u scripts/run_algorithm_comparison.py \
    --datasets phm2012 xjtusy \
    --models nbeats_rul physics_nbeats_rul nbeats_xlstm_rul \
    --train "${ROOT}/configs/train/cloud_full_75.yaml" \
    --ablation "${ROOT}/configs/ablation/gpu_throughput.yaml" \
    --data-batch-size 512 \
    --max-epochs 30 \
    --seed "${SEED:-42}" \
    --report-name nbeats_physics_hybrid_s"${SEED:-42}" \
    --no-figures
}

_run_cloud75() {
  echo "=== Physics-N-BEATS + N-BEATS-xLSTM v3 (75 ep, PHM2012 + XJTU-SY, cloud_full_75, parallel models) ==="
  python -u scripts/run_algorithm_comparison.py \
    --datasets phm2012 xjtusy \
    --models physics_nbeats_rul nbeats_xlstm_rul \
    --train "${ROOT}/configs/train/cloud_full_75.yaml" \
    --ablation "${ROOT}/configs/ablation/gpu_throughput.yaml" \
    --data-batch-size 384 \
    --parallel-models \
    --seed "${SEED:-42}" \
    --report-name physics_nbeats_xlstm_cloud75_v3_parallel_s"${SEED:-42}" \
    --no-figures
}

case "${cmd}" in
  smoke)  _run_smoke ;;
  compare) _run_compare ;;
  cloud75) _run_cloud75 ;;
  all)     _run_smoke && _run_compare ;;
  *)
    echo "usage: $0 {smoke|compare|cloud75|all}" >&2
    exit 2
    ;;
esac

echo "=== Done (${cmd}) ==="
