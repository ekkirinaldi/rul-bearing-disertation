#!/usr/bin/env bash
#
# Preflight + launch full comparative training for RunPod (or any CUDA/Linux host).
#
# Usage (from repo Mamba-xLSTM/, after venv + pip install -e ".[report_pure,...]"):
#
#   # Full PHM2012 + XJTU available sweep, five models, cloud 200 epoch recipe
#   ./scripts/run_runpod_full_experiments.sh full
#
#   # Same but skip aggregated reports (manual build_cloud_full_reports.sh later)
#   ./scripts/run_runpod_full_experiments.sh train-only
#
#   # Only regenerate reports after training (same SEED env)
#   ./scripts/run_runpod_full_experiments.sh reports-only
#
#   # Lightweight wiring check (tiny train/val, no figures, no reports)
#   ./scripts/run_runpod_full_experiments.sh smoke
#
# Env:
#   SEED            default 42
#   TRAIN_YAML      default configs/train/cloud_full_75.yaml
#   PYTHON          default .venv/bin/python
#   DATA_ROOT       override path to ../data-bearing if layouts differ on pod
#
# Dataset on a fresh pod: .cursor/rules/vps-ssh-key-access.mdc §6 (data-bearing.zip)
#   + §6.3 (xtju-sy.zip — full three-condition XJTU, 9216 CSVs).
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

MODE="${1:-full}"
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
SEED="${SEED:-42}"
TRAIN_YAML="${TRAIN_YAML:-configs/train/cloud_full_75.yaml}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/../data-bearing}"

case "${MODE}" in
  full|train-only|smoke|reports-only) ;;
  *) echo "[run_runpod_full_experiments] ERROR: unknown mode '${MODE}'. Use: full | train-only | smoke | reports-only" >&2; exit 2 ;;
esac

_die() {
  echo "[run_runpod_full_experiments] ERROR: $*" >&2
  exit 1
}

if [[ ! -f "${PYTHON}" ]]; then
  _die "Python interpreter not found (expected project venv): ${PYTHON}"
fi

if [[ ! -d "${DATA_ROOT}" ]]; then
  _die "data-bearing missing: ${DATA_ROOT} (set DATA_ROOT or mount dataset volume)"
fi
if [[ ! -d "${DATA_ROOT}/ieee-phm-2012" ]]; then
  _die "PHM2012 folder missing under ${DATA_ROOT}/ieee-phm-2012"
fi
if [[ ! -d "${DATA_ROOT}/xtju-sy" ]]; then
  _die "XJTU-SY folder missing under ${DATA_ROOT}/xtju-sy"
fi
if [[ ! -d "${DATA_ROOT}/xtju-sy/35Hz12kN" || ! -d "${DATA_ROOT}/xtju-sy/37.5Hz11kN" || ! -d "${DATA_ROOT}/xtju-sy/40Hz10kN" ]]; then
  _die "XJTU expected condition folders missing (need 35Hz12kN, 37.5Hz11kN, 40Hz10kN)"
fi

# Basic CUDA readiness (recommended for bf16 cloud config)
_run_py_quickcheck() {
  "${PYTHON}" - <<'PY'
import sys

try:
    import torch
except Exception as exc:
    print(f"[precheck cuda] WARN: torch import failed ({exc!r}) — install torch before cloud training.")
    sys.exit(0)

if torch.cuda.is_available():
    print(
        f"[precheck cuda] GPU={torch.cuda.get_device_name(0)} "
        f"capability={torch.cuda.get_device_capability(0)}"
    )
else:
    print("[precheck cuda] WARN: CUDA not available — bf16 GPU config may fall back awkwardly.")
PY
}
_run_py_quickcheck

mkdir -p "${ROOT}/results/_logs"

if [[ "${MODE}" == "reports-only" ]]; then
  exec "${ROOT}/scripts/build_cloud_full_reports.sh" "${SEED}"
fi

MODELS=(
  diffusion_rul
  liquid_wave_rul
  mamba_xlstm_net
  nbeats_rul
  xlstm_transformer
)

STDBUF=""
if command -v stdbuf >/dev/null 2>&1; then
  STDBUF="stdbuf -oL -eL"
fi

if [[ "${MODE}" == "smoke" ]]; then
  LOG="${ROOT}/results/_logs/runpod_cloud_smoke_s${SEED}_$(date +%Y%m%d_%H%M%S).log"
  echo "[run_runpod_full_experiments] Smoke → tee ${LOG}"
  ${STDBUF} "${PYTHON}" -u scripts/run_algorithm_comparison.py \
    --datasets phm2012 xjtu_available \
    --models nbeats_rul \
    --train configs/train/algorithm_comparison.yaml \
    --seed "${SEED}" \
    --fast-dev-run \
    --data-batch-size 32 \
    --skip-report \
    --no-figures 2>&1 | tee "${LOG}"
  exit "${PIPESTATUS[0]}"
fi

ARGS=(
  -u scripts/run_algorithm_comparison.py
  --datasets phm2012 xjtu_available
  --models "${MODELS[@]}"
  --train "${TRAIN_YAML}"
  --seed "${SEED}"
)

if [[ "${MODE}" == "train-only" ]] || [[ "${MODE}" == "full" ]]; then
  ARGS+=(--skip-report)
fi

LOG="${ROOT}/results/_logs/runpod_cloud_full_s${SEED}_$(date +%Y%m%d_%H%M%S).log"
echo "[run_runpod_full_experiments] Mode=${MODE} train=${TRAIN_YAML} SEED=${SEED}"
echo "[run_runpod_full_experiments] Tee log → ${LOG}"

${STDBUF} "${PYTHON}" "${ARGS[@]}" 2>&1 | tee "${LOG}"
train_status="${PIPESTATUS[0]}"
if [[ "${train_status}" -ne 0 ]]; then
  exit "${train_status}"
fi

if [[ "${MODE}" == "full" ]]; then
  echo "[run_runpod_full_experiments] Building report bundle (global + per-model + dataset rollups)"
  exec "${ROOT}/scripts/build_cloud_full_reports.sh" "${SEED}"
else
  echo "[run_runpod_full_experiments] train-only finished. Build reports:"
  echo "  ./scripts/build_cloud_full_reports.sh ${SEED}"
fi
