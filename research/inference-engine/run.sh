#!/usr/bin/env bash
# Run the RUL streaming inference dashboard (uses Mamba-xLSTM venv + package).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${ROOT}/Mamba-xLSTM/.venv"
export PYTHONPATH="${ROOT}/Mamba-xLSTM/src:${PYTHONPATH:-}"
cd "$(dirname "$0")"
exec "${VENV}/bin/uvicorn" app.server:app --reload --host 0.0.0.0 --port 8800
