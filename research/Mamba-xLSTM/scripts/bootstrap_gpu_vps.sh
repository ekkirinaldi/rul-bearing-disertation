#!/usr/bin/env bash
# One-shot environment prep for Mamba-xLSTM on an NVIDIA VPS (RunPod / generic CUDA host).
#
# From this directory after cloning/syncing the repo + data-bearing:
#   cd Mamba-xLSTM && bash scripts/bootstrap_gpu_vps.sh
#
# Env:
#   TORCH_CUDA   PyTorch wheel flavour. Default: cu128 (Blackwell sm_120 needs 2.7+ / cu128 wheels;
#                use TORCH_CUDA=cu124 only for older GPUs when you know it works).
#   INSTALL_MAMBASSM  If 1, build mamba-ssm from Git (slow; CUDA Mamba3). Default: 0
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TORCH_CUDA="${TORCH_CUDA:-cu128}"

if command -v nvidia-smi &>/dev/null; then
  nvidia-smi
else
  echo "warning: nvidia-smi not found — continue only if intentional" >&2
fi

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" &>/dev/null; then
  echo "error: $PYTHON not found" >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  "$PYTHON" -m venv .venv
fi
# shellcheck source=/dev/null
source .venv/bin/activate

python -m pip install --upgrade pip wheel setuptools

# Install CUDA Torch first so ``pip install -e .`` does not pull CPU-only builds.
EXTRA="https://download.pytorch.org/whl/${TORCH_CUDA}"
python -m pip install "torch>=2.7" torchvision --index-url "${EXTRA}"

python -m pip install -e ".[xlstm]"

python << 'PYCHECK'
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PYCHECK

INSTALL_MAMBASSM="${INSTALL_MAMBASSM:-0}"
if [[ "${INSTALL_MAMBASSM}" == "1" ]]; then
  echo "[bootstrap] Installing mamba-ssm from source (may take several minutes)…"
  # Packages often required for CMake/CUDA extensions; ignore apt failures on non-Debian images.
  if command -v apt-get &>/dev/null; then
    apt-get update -qq && apt-get install -y -qq build-essential git ninja-build cmake \
      || true
  fi
  MAMBA_FORCE_BUILD=TRUE python -m pip install --no-cache-dir --force-reinstall \
    git+https://github.com/state-spaces/mamba.git --no-build-isolation
  python -c "import mamba_ssm; print('mamba_ssm OK')"
fi

echo
echo "[bootstrap] Done. Example smoke + full comparison on GPU:"
echo "  cd $ROOT && source .venv/bin/activate"
echo "  python scripts/train.py --data configs/data/phm2012.yaml \\"
echo "    --model configs/model/nbeats_rul.yaml \\"
echo "    --train configs/train/algorithm_comparison.yaml \\"
echo "    --ablation configs/ablation/gpu_throughput.yaml \\"
echo "    --fast-dev-run"
echo ""
echo "  Increase batch (--data-batch-size on run_algorithm_comparison, or bump data YAML) "
echo "  while watching nvidia-smi until VRAM is well used without OOM."
