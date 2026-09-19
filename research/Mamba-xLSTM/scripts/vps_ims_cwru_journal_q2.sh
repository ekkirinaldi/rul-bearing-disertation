#!/usr/bin/env bash
# Train mamba_xlstm_net, nbeats_xlstm_rul, sparse_gate_tcn_rul on IMS and CWRU,
# then run the full journal_q2 experiment suite on all four datasets.
#
# If a run stopped mid-way (e.g. after Mamba on IMS seed 42), use instead:
#   scripts/vps_resume_ims_cwru_journal_q2.sh
#
# Run on a GPU VPS (CUDA) after bootstrapping the environment:
#   source .venv/bin/activate
#
# Usage (from Mamba-xLSTM/):
#   chmod +x scripts/vps_ims_cwru_journal_q2.sh
#   nohup bash scripts/vps_ims_cwru_journal_q2.sh > ~/ims_cwru_q2.log 2>&1 &
#
# After it completes, pull results back to laptop:
#   rsync -az root@HOST:/root/disertation-rul-prediction/Mamba-xLSTM/results/ \
#     ./Mamba-xLSTM/results/
#
# Env-var overrides:
#   SEEDS="42 43 44"         — seeds for backbone training (default: 42 43 44)
#   DATA_BATCH_SIZE=512     — passed to --data-batch-size (default: 512; use 128 on MPS)
#   SKIP_TRAIN_IMS=1         — skip IMS training (use existing checkpoints)
#   SKIP_TRAIN_CWRU=1        — skip CWRU training (use existing checkpoints)
#   SKIP_JOURNAL_Q2=1        — skip run_all.sh

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

if [[ ! -d "${ROOT}/.venv" ]]; then
  echo "ERROR: ${ROOT}/.venv missing — run scripts/bootstrap_gpu_vps.sh first." >&2
  exit 1
fi
source "${ROOT}/.venv/bin/activate"
export PYTHONUNBUFFERED=1

SEEDS="${SEEDS:-42 43 44}"
SKIP_TRAIN_IMS="${SKIP_TRAIN_IMS:-0}"
SKIP_TRAIN_CWRU="${SKIP_TRAIN_CWRU:-0}"
SKIP_JOURNAL_Q2="${SKIP_JOURNAL_Q2:-0}"
# Lower on laptop MPS / limited VRAM (e.g. DATA_BATCH_SIZE=128).
DATA_BATCH_SIZE="${DATA_BATCH_SIZE:-512}"

TRAIN_CFG="${ROOT}/configs/train/cloud_full_75.yaml"
ABLATION_CFG="${ROOT}/configs/ablation/gpu_throughput.yaml"
MODELS="mamba_xlstm_net nbeats_xlstm_rul sparse_gate_tcn_rul"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

# ---------------------------------------------------------------------------
# Phase 1 — Train on IMS
# ---------------------------------------------------------------------------
if [[ "${SKIP_TRAIN_IMS}" == "0" ]]; then
  log "=== Phase 1: Training on IMS (all seeds: ${SEEDS}) ==="
  for SEED in ${SEEDS}; do
    log "  IMS seed ${SEED}..."
    python -u scripts/run_algorithm_comparison.py \
      --datasets ims \
      --models ${MODELS} \
      --train "${TRAIN_CFG}" \
      --ablation "${ABLATION_CFG}" \
      --data-batch-size "${DATA_BATCH_SIZE}" \
      --seed "${SEED}" \
      --report-name "ims_three_arch_s${SEED}" \
      --no-figures
    log "  IMS seed ${SEED} done."
  done
  log "IMS training complete."
else
  log "Skipping IMS training (SKIP_TRAIN_IMS=1)."
fi

# ---------------------------------------------------------------------------
# Phase 2 — Train on CWRU
# ---------------------------------------------------------------------------
if [[ "${SKIP_TRAIN_CWRU}" == "0" ]]; then
  log "=== Phase 2: Training on CWRU (all seeds: ${SEEDS}) ==="
  for SEED in ${SEEDS}; do
    log "  CWRU seed ${SEED}..."
    python -u scripts/run_algorithm_comparison.py \
      --datasets cwru \
      --models ${MODELS} \
      --train "${TRAIN_CFG}" \
      --ablation "${ABLATION_CFG}" \
      --data-batch-size "${DATA_BATCH_SIZE}" \
      --seed "${SEED}" \
      --report-name "cwru_three_arch_s${SEED}" \
      --no-figures
    log "  CWRU seed ${SEED} done."
  done
  log "CWRU training complete."
else
  log "Skipping CWRU training (SKIP_TRAIN_CWRU=1)."
fi

# ---------------------------------------------------------------------------
# Phase 3 — Journal Q2 full experiment suite (all four datasets)
# ---------------------------------------------------------------------------
if [[ "${SKIP_JOURNAL_Q2}" == "0" ]]; then
  log "=== Phase 3: Journal Q2 experiments (all four datasets) ==="
  DATASETS="phm2012 xjtusy ims cwru" \
    N_BOOT=1000 N_PERM=1000 EPOCHS=50 N_HIDDEN=5000 MAX_REC=300 \
    bash scripts/journal_q2/run_all.sh
  log "Journal Q2 experiments complete."
else
  log "Skipping journal Q2 experiments (SKIP_JOURNAL_Q2=1)."
fi

log "ALL DONE — results under results/journal_q2/"
