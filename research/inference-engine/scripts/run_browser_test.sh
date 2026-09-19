#!/usr/bin/env bash
# Boot the inference server (model venv) and drive it with Playwright (test venv).
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${PORT:-8813}"
MODEL_PY="../Mamba-xLSTM/.venv/bin/python"
TEST_PY=".test-venv/bin/python"

echo "[run] starting server on :$PORT"
PYTHONPATH=".:../Mamba-xLSTM/src" "$MODEL_PY" -m uvicorn app.server:app \
  --host 127.0.0.1 --port "$PORT" --log-level warning > /tmp/rul_browser_srv.log 2>&1 &
SRV_PID=$!
trap 'kill $SRV_PID 2>/dev/null || true' EXIT

echo "[run] waiting for server…"
for i in $(seq 1 60); do
  if curl -s -o /dev/null "http://127.0.0.1:$PORT/api/runs"; then echo "[run] up"; break; fi
  sleep 1
done

RUL_BASE="http://127.0.0.1:$PORT" "$TEST_PY" scripts/browser_test.py
echo "[run] server log tail:"; tail -3 /tmp/rul_browser_srv.log
