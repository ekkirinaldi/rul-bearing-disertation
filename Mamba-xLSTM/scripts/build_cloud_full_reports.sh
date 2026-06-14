#!/usr/bin/env bash
#
# Build HTML/PDF reports after a cloud full sweep (10 runs × two datasets × five models).
# Resolves newest matching run dirs by timestamp suffix.
#
# Usage (from repo Mamba-xLSTM/):
#   ./scripts/build_cloud_full_reports.sh [SEED]
#
# Prerequisites: scripts/run_algorithm_comparison.py has finished for the same seed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
SEED="${1:-42}"

_latest() {
  # shellcheck disable=SC2086
  ls -dt "${ROOT}/results/runs/"${1} 2>/dev/null | head -1 || true
}

model_label() {
  case "$1" in
    diffusion_rul) echo "Diffusion-RUL" ;;
    liquid_wave_rul) echo "LiquidWave-RUL" ;;
    mamba_xlstm_net) echo "Mamba-xLSTM-Net" ;;
    nbeats_rul) echo "N-BEATS-RUL" ;;
    xlstm_transformer) echo "Baseline (xLSTM–Transformer)" ;;
    *) echo "$1" ;;
  esac
}

MODELS=(
  diffusion_rul
  liquid_wave_rul
  mamba_xlstm_net
  nbeats_rul
  xlstm_transformer
)

ALL_RUNS=()
MISSING=()
for m in "${MODELS[@]}"; do
  p="$(_latest "*_algorithm_comparison_phm2012_${m}_s${SEED}")"
  x="$(_latest "*_algorithm_comparison_xjtu_available_${m}_s${SEED}")"
  if [[ -z "${p}" || ! -f "${p}/summary.json" ]]; then
    MISSING+=("phm2012+${m}")
  else
    ALL_RUNS+=("${p}")
  fi
  if [[ -z "${x}" || ! -f "${x}/summary.json" ]]; then
    MISSING+=("xjtu_available+${m}")
  else
    ALL_RUNS+=("${x}")
  fi
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
  echo "[build_cloud_full_reports] Missing run dirs (need summary.json):" >&2
  printf '  - %s\n' "${MISSING[@]}" >&2
  exit 1
fi

mkdir -p "${ROOT}/results/reports"

echo "[build_cloud_full_reports] Global report (10 runs) → cloud_full_all_algorithms_s${SEED}"

"${PYTHON}" -u scripts/build_report.py \
  --reports-dir results/reports \
  --runs "${ALL_RUNS[@]}" \
  --name "cloud_full_all_algorithms_s${SEED}" \
  --title "Cloud full sweep — all algorithms" \
  --subtitle "PHM2012 + XJTU-SY (3 conditions, dense eval), seed ${SEED}"

for m in "${MODELS[@]}"; do
  p="$(_latest "*_algorithm_comparison_phm2012_${m}_s${SEED}")"
  x="$(_latest "*_algorithm_comparison_xjtu_available_${m}_s${SEED}")"
  lbl="$(model_label "${m}")"
  echo "[build_cloud_full_reports] Pair report ${m} → cloud_full_${m}_s${SEED}"
  "${PYTHON}" -u scripts/build_report.py \
    --reports-dir results/reports \
    --runs "${p}" "${x}" \
    --name "cloud_full_${m}_s${SEED}" \
    --title "${lbl}" \
    --subtitle "PHM2012 vs XJTU-SY (available), seed ${SEED}"
done

PHM_ONLY=()
XJTU_ONLY=()
for m in "${MODELS[@]}"; do
  PHM_ONLY+=("$(_latest "*_algorithm_comparison_phm2012_${m}_s${SEED}")")
  XJTU_ONLY+=("$(_latest "*_algorithm_comparison_xjtu_available_${m}_s${SEED}")")
done

echo "[build_cloud_full_reports] Dataset-only rollup → cloud_full_dataset_phm2012_s${SEED}"
"${PYTHON}" -u scripts/build_report.py \
  --reports-dir results/reports \
  --runs "${PHM_ONLY[@]}" \
  --name "cloud_full_dataset_phm2012_s${SEED}" \
  --title "PHM2012 — all five architectures" \
  --subtitle "Seed ${SEED}"

echo "[build_cloud_full_reports] Dataset-only rollup → cloud_full_dataset_xjtu_available_s${SEED}"
"${PYTHON}" -u scripts/build_report.py \
  --reports-dir results/reports \
  --runs "${XJTU_ONLY[@]}" \
  --name "cloud_full_dataset_xjtu_available_s${SEED}" \
  --title "XJTU-SY (available) — all five architectures" \
  --subtitle "Seed ${SEED}"

echo "[build_cloud_full_reports] Done."
