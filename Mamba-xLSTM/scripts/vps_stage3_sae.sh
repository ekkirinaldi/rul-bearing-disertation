#!/usr/bin/env bash
# Stage 3 — SAE interpretability on best Mamba-xLSTM-Net checkpoints.
#
# Scope (minimal, for Pilar 2 core claim):
#   Mamba-xLSTM-Net × PHM2012 (best seed 42) + Mamba-xLSTM-Net × XJTU-SY (best seed 44)
#
# Optional extension: set INCLUDE_SPARSE=1 to also run SparseGate-TCN-RUL for
#   cross-model comparison (both datasets, best seeds 42 PHM / 42 XJTU).
#
# Produces per-run: <run_dir>/explain/
#   sae_config.json, sae_weights.pt, sae_hidden.npz,
#   shap_values.npz, ig_attribution.npz, umap_embedding.npz,
#   figures/  (shap_bar, ig_heatmap, umap_scatter, sae_activation, …)
#
# Usage (on the VPS, from Mamba-xLSTM/ after ``source .venv/bin/activate``):
#   chmod +x scripts/vps_stage3_sae.sh
#   nohup ./scripts/vps_stage3_sae.sh > ~/vps_stage3_sae.log 2>&1 &
#
# Wall-clock estimate per run: ~5–15 min on A40 depending on dataset size.

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

if [[ ! -d "${ROOT}/.venv" ]]; then
  echo "ERROR: ${ROOT}/.venv missing — run scripts/bootstrap_gpu_vps.sh first." >&2
  exit 1
fi

source "${ROOT}/.venv/bin/activate"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUNS_ROOT="${ROOT}/results/runs"

# ---------------------------------------------------------------------------
# Best run directories (Stage 2 output, lowest test/rmse per model × dataset).
# Update these if new runs with lower RMSE are generated.
# ---------------------------------------------------------------------------
MAMBA_PHM_RUN="${RUNS_ROOT}/20260512_151550_algorithm_comparison_phm2012_mamba_xlstm_net_s42"
MAMBA_XJTU_RUN="${RUNS_ROOT}/20260611_104213_algorithm_comparison_xjtusy_mamba_xlstm_net_s42"

# Optional cross-model runs (set INCLUDE_SPARSE=1 to enable)
SPARSE_PHM_RUN="${RUNS_ROOT}/20260512_163040_algorithm_comparison_phm2012_sparse_gate_tcn_rul_s42"
SPARSE_XJTU_RUN="${RUNS_ROOT}/20260512_163817_algorithm_comparison_xjtusy_sparse_gate_tcn_rul_s42"

INCLUDE_SPARSE="${INCLUDE_SPARSE:-0}"

# ---------------------------------------------------------------------------
# SAE hyperparameters (matches experiment-design.md Stage 3 plan)
# ---------------------------------------------------------------------------
SAE_EPOCHS=50
SAE_EXPANSION=8
N_SHAP_SAMPLES=64

_run_interp() {
  local label="$1"
  local run_dir="$2"
  if [[ ! -d "${run_dir}" ]]; then
    echo "WARN: run dir not found, skipping: ${run_dir}" >&2
    return 0
  fi
  local ckpt
  ckpt="$(ls "${run_dir}/checkpoints/"*.ckpt 2>/dev/null | head -1 || true)"
  if [[ -z "${ckpt}" ]]; then
    echo "WARN: no checkpoint in ${run_dir}/checkpoints/, skipping." >&2
    return 0
  fi
  echo "=== [${label}] Starting SAE + SHAP + IG + UMAP ==="
  echo "    run_dir : ${run_dir}"
  echo "    ckpt    : ${ckpt}"
  python -u scripts/run_interpretability.py \
    --from-run    "${run_dir}" \
    --out-dir     "${run_dir}/explain" \
    --sae-epochs  "${SAE_EPOCHS}" \
    --sae-expansion "${SAE_EXPANSION}" \
    --n-shap-samples "${N_SHAP_SAMPLES}"
  echo "=== [${label}] Done. Outputs in ${run_dir}/explain/ ==="
}

# ---------------------------------------------------------------------------
# Core runs (always)
# ---------------------------------------------------------------------------
_run_interp "Mamba-xLSTM-Net / PHM2012 / seed42" "${MAMBA_PHM_RUN}"
_run_interp "Mamba-xLSTM-Net / XJTU-SY  / seed44" "${MAMBA_XJTU_RUN}"

# ---------------------------------------------------------------------------
# Optional cross-model extension
# ---------------------------------------------------------------------------
if [[ "${INCLUDE_SPARSE}" == "1" ]]; then
  _run_interp "SparseGate-TCN-RUL / PHM2012 / seed42" "${SPARSE_PHM_RUN}"
  _run_interp "SparseGate-TCN-RUL / XJTU-SY  / seed42" "${SPARSE_XJTU_RUN}"
fi

echo ""
echo "=== Stage 3 SAE complete. Explain directories: ==="
for run_dir in "${MAMBA_PHM_RUN}" "${MAMBA_XJTU_RUN}"; do
  echo "  ${run_dir}/explain/"
  ls "${run_dir}/explain/" 2>/dev/null | sed 's/^/    /' || true
done
