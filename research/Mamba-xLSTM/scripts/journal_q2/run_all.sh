#!/usr/bin/env bash
# Orchestrator for the JETS journal Q2 extra experiments.
#
# Runs in sequence:
#   1. Bootstrap CI + permutation + per-bearing breakdown (run_stats.py)
#   2. Negative controls (untrained backbone + Gaussian noise)
#   3. Sparsity sweep
#   4. Cross-architecture BPFx mapping
#
# Total wall-clock estimate (single GPU, 5k hidden states, ~300 recordings):
#   ~15 min stats, ~30 min controls, ~30 min sweep, ~45 min cross-arch
#   ≈ 2 hours end-to-end on one A40.
#
# Logs are written to results/journal_q2/_logs/.
#
# Usage:
#   cd Mamba-xLSTM
#   bash scripts/journal_q2/run_all.sh
#
# Override defaults via env vars, e.g.:
#   N_BOOT=2000 N_PERM=2000 bash scripts/journal_q2/run_all.sh

set -euo pipefail

cd "$(dirname "$0")/../.."   # repo / Mamba-xLSTM root

if [[ -z "${VIRTUAL_ENV:-}" ]] && [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

LOG_DIR="results/journal_q2/_logs"
mkdir -p "$LOG_DIR"

DATASETS="${DATASETS:-phm2012 xjtusy ims cwru}"
THRESHOLD="${THRESHOLD:-0.30}"
N_BOOT="${N_BOOT:-1000}"
N_PERM="${N_PERM:-1000}"
EPOCHS="${EPOCHS:-30}"
N_HIDDEN="${N_HIDDEN:-5000}"
MAX_REC="${MAX_REC:-300}"
KS="${KS:-10 51 102 205}"
ARCHS="${ARCHS:-mamba_xlstm_net nbeats_xlstm_rul sparse_gate_tcn_rul}"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

log() { echo "[$(ts)] $*"; }

run_step() {
  local name="$1"; shift
  local logf="$LOG_DIR/${name}.log"
  log "START ${name}  (log → ${logf})"
  python -u "$@" 2>&1 | tee "$logf"
  log "DONE  ${name}"
}

run_step "01_stats" -m scripts.journal_q2.run_stats \
  --datasets ${DATASETS} \
  --threshold "${THRESHOLD}" --n-boot "${N_BOOT}" --n-perm "${N_PERM}" \
  --max-recordings "${MAX_REC}" --n-hidden "${N_HIDDEN}"

run_step "02_negative_controls" -m scripts.journal_q2.run_negative_controls \
  --datasets ${DATASETS} \
  --threshold "${THRESHOLD}" --epochs "${EPOCHS}" \
  --max-recordings "${MAX_REC}" --n-hidden "${N_HIDDEN}"

run_step "03_sparsity_sweep" -m scripts.journal_q2.run_sparsity_sweep \
  --datasets ${DATASETS} --ks ${KS} \
  --threshold "${THRESHOLD}" --epochs "${EPOCHS}" \
  --max-recordings "${MAX_REC}" --n-hidden "${N_HIDDEN}"

run_step "04_cross_arch" -m scripts.journal_q2.run_cross_arch \
  --datasets ${DATASETS} --architectures ${ARCHS} \
  --threshold "${THRESHOLD}" --epochs "${EPOCHS}" \
  --max-recordings "${MAX_REC}" --n-hidden "${N_HIDDEN}"

# Re-run stats now that cross_arch has written explain/sae.pt for each dataset.
# Datasets that already had a SAE are re-processed (idempotent); datasets that
# were skipped in step 01 (IMS, CWRU) now have their SAE available.
run_step "05_stats_post_cross_arch" -m scripts.journal_q2.run_stats \
  --datasets ${DATASETS} \
  --threshold "${THRESHOLD}" --n-boot "${N_BOOT}" --n-perm "${N_PERM}" \
  --max-recordings "${MAX_REC}" --n-hidden "${N_HIDDEN}"

log "ALL DONE — see results/journal_q2/"
