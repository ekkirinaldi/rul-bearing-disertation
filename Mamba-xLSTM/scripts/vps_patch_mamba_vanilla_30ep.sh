#!/usr/bin/env bash
# Run on the GPU VPS from ``Mamba-xLSTM/`` after ``source .venv/bin/activate``.
#
# Serial 30-epoch comparison: PatchTST-RUL, Mamba-RUL, Vanilla xLSTM-RUL
# on PHM2012 + XJTU-SY (6 runs total). Does NOT use --parallel-models.
#
# From laptop (after rsync code):
#   ssh ... 'cd .../Mamba-xLSTM && chmod +x scripts/vps_patch_mamba_vanilla_30ep.sh && ./scripts/vps_patch_mamba_vanilla_30ep.sh'

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

echo "=== PatchTST + Mamba-RUL + Vanilla xLSTM (30 ep each, serial, PHM2012 + XJTU-SY) ==="
python -u scripts/run_algorithm_comparison.py \
  --datasets phm2012 xjtusy \
  --models patch_tst_rul mamba_rul vanilla_xlstm_rul \
  --train "${ROOT}/configs/train/algorithm_comparison.yaml" \
  --ablation "${ROOT}/configs/ablation/gpu_throughput.yaml" \
  --data-batch-size 512 \
  --max-epochs 30 \
  --seed "${SEED:-42}" \
  --report-name patch_mamba_vanilla_30ep_serial_s"${SEED:-42}" \
  --no-figures \
  --no-pdf

echo "=== Done (patch_mamba_vanilla 30ep serial) ==="
