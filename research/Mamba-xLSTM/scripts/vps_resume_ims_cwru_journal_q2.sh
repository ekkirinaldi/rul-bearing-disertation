#!/usr/bin/env bash
# Resume IMS + CWRU training after a partial VPS run (e.g. IMS seed 42 Mamba done,
# N-BEATS failed before physics_nbeats_core supported ims/cwru).
#
# From Mamba-xLSTM/ on the VPS (venv active):
#   screen -dmS ims_cwru_resume bash -c '
#     cd /root/disertation-rul-prediction/Mamba-xLSTM && source .venv/bin/activate &&
#     bash scripts/vps_resume_ims_cwru_journal_q2.sh 2>&1 | tee results/_logs/vps_resume_ims_cwru.log
#   '
#
# Env:
#   DATA_BATCH_SIZE   default 512
#   SKIP_JOURNAL_Q2   default 1 (set 0 to run journal_q2/run_all.sh after training)

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

if [[ ! -d "${ROOT}/.venv" ]]; then
  echo "ERROR: ${ROOT}/.venv missing" >&2
  exit 1
fi
source "${ROOT}/.venv/bin/activate"
export PYTHONUNBUFFERED=1

mkdir -p "${ROOT}/results/_logs"

TRAIN_CFG="${ROOT}/configs/train/cloud_full_75.yaml"
ABLATION_CFG="${ROOT}/configs/ablation/gpu_throughput.yaml"
DATA_BATCH_SIZE="${DATA_BATCH_SIZE:-512}"
SKIP_JOURNAL_Q2="${SKIP_JOURNAL_Q2:-1}"

MODELS_ALL="mamba_xlstm_net nbeats_xlstm_rul sparse_gate_tcn_rul"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

# ---------------------------------------------------------------------------
# IMS seed 42 — only architectures not yet finished (Mamba already OK).
# ---------------------------------------------------------------------------
log "=== Resume: IMS seed 42 — N-BEATS-xLSTM-RUL + SparseGate-TCN-RUL ==="
python -u scripts/run_algorithm_comparison.py \
  --datasets ims \
  --models nbeats_xlstm_rul sparse_gate_tcn_rul \
  --train "${TRAIN_CFG}" \
  --ablation "${ABLATION_CFG}" \
  --data-batch-size "${DATA_BATCH_SIZE}" \
  --seed 42 \
  --report-name ims_resume_s42_nbeats_sparse \
  --no-figures

# ---------------------------------------------------------------------------
# IMS seeds 43, 44 — full three-way comparison
# ---------------------------------------------------------------------------
for SEED in 43 44; do
  log "=== IMS seed ${SEED} — all three models ==="
  python -u scripts/run_algorithm_comparison.py \
    --datasets ims \
    --models ${MODELS_ALL} \
    --train "${TRAIN_CFG}" \
    --ablation "${ABLATION_CFG}" \
    --data-batch-size "${DATA_BATCH_SIZE}" \
    --seed "${SEED}" \
    --report-name "ims_three_arch_s${SEED}" \
    --no-figures
done

# ---------------------------------------------------------------------------
# CWRU seeds 42, 43, 44 — full three-way comparison
# ---------------------------------------------------------------------------
for SEED in 42 43 44; do
  log "=== CWRU seed ${SEED} — all three models ==="
  python -u scripts/run_algorithm_comparison.py \
    --datasets cwru \
    --models ${MODELS_ALL} \
    --train "${TRAIN_CFG}" \
    --ablation "${ABLATION_CFG}" \
    --data-batch-size "${DATA_BATCH_SIZE}" \
    --seed "${SEED}" \
    --report-name "cwru_three_arch_s${SEED}" \
    --no-figures
done

# ---------------------------------------------------------------------------
# Journal Q2 (optional)
# ---------------------------------------------------------------------------
if [[ "${SKIP_JOURNAL_Q2}" == "0" ]]; then
  log "=== Journal Q2 experiments (phm2012 xjtusy ims cwru) ==="
  DATASETS="phm2012 xjtusy ims cwru" \
    N_BOOT=1000 N_PERM=1000 EPOCHS=50 N_HIDDEN=5000 MAX_REC=300 \
    bash scripts/journal_q2/run_all.sh
  log "Journal Q2 complete."
else
  log "SKIP_JOURNAL_Q2=${SKIP_JOURNAL_Q2} — skipping journal_q2/run_all.sh (set SKIP_JOURNAL_Q2=0 to run)."
fi

log "RESUME ALL DONE — checkpoints under results/runs/"
