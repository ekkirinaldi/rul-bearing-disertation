#!/usr/bin/env bash
# Sequential 30-epoch jobs: PhaseMoE PHM2012 resumes from last.ckpt; others fresh.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
source .venv/bin/activate
export PYTHONUNBUFFERED=1

PHM_DATA="${ROOT}/results/_comparison_configs/phm2012_batch512.yaml"
XJTU_DATA="${ROOT}/results/_comparison_configs/xjtusy_batch512.yaml"
TRAIN="${ROOT}/configs/train/cloud_full_75.yaml"
ABL="${ROOT}/configs/ablation/gpu_throughput.yaml"
PHASE_CKPT="${ROOT}/results/runs/20260510_204127_algorithm_comparison_phm2012_phase_moe_xlstm_rul_s42/checkpoints/last.ckpt"

echo "=== 1/4 PhaseMoE PHM2012: resume -> max 30 epochs total ==="
python -u scripts/train.py \
  --data "${PHM_DATA}" \
  --model "${ROOT}/configs/model/phase_moe_xlstm_rul.yaml" \
  --train "${TRAIN}" \
  --ablation "${ABL}" \
  --seed 42 \
  --run-id algorithm_comparison_phm2012_phase_moe_xlstm_rul_s42 \
  --max-epochs 30 \
  --ckpt-path "${PHASE_CKPT}" \
  --no-figures

echo "=== 2/4 SparseGate PHM2012: 30 epochs ==="
python -u scripts/train.py \
  --data "${PHM_DATA}" \
  --model "${ROOT}/configs/model/sparse_gate_tcn_rul.yaml" \
  --train "${TRAIN}" \
  --ablation "${ABL}" \
  --seed 42 \
  --run-id algorithm_comparison_phm2012_sparse_gate_tcn_rul_s42 \
  --max-epochs 30 \
  --no-figures

echo "=== 3/4 PhaseMoE XJTU-SY: 30 epochs ==="
python -u scripts/train.py \
  --data "${XJTU_DATA}" \
  --model "${ROOT}/configs/model/phase_moe_xlstm_rul.yaml" \
  --train "${TRAIN}" \
  --ablation "${ABL}" \
  --seed 42 \
  --run-id algorithm_comparison_xjtusy_phase_moe_xlstm_rul_s42 \
  --max-epochs 30 \
  --no-figures

echo "=== 4/4 SparseGate XJTU-SY: 30 epochs ==="
python -u scripts/train.py \
  --data "${XJTU_DATA}" \
  --model "${ROOT}/configs/model/sparse_gate_tcn_rul.yaml" \
  --train "${TRAIN}" \
  --ablation "${ABL}" \
  --seed 42 \
  --run-id algorithm_comparison_xjtusy_sparse_gate_tcn_rul_s42 \
  --max-epochs 30 \
  --no-figures

echo "=== All four jobs finished ==="
for rid in \
  algorithm_comparison_phm2012_phase_moe_xlstm_rul_s42 \
  algorithm_comparison_phm2012_sparse_gate_tcn_rul_s42 \
  algorithm_comparison_xjtusy_phase_moe_xlstm_rul_s42 \
  algorithm_comparison_xjtusy_sparse_gate_tcn_rul_s42
do
  d="$(ls -td "${ROOT}/results/runs/"*"_${rid}" 2>/dev/null | head -1)"
  echo "--- ${rid} -> ${d} ---"
  if [[ -f "${d}/summary.json" ]]; then cat "${d}/summary.json"; fi
done
