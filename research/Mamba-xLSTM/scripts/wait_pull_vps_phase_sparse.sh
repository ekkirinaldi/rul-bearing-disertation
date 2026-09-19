#!/usr/bin/env bash
# Wait for remote run_algorithm_comparison log to finish (HTML report line),
# then rsync Mamba-xLSTM/results from VPS to this repo copy.
#
# Env:
#   VPS_HOST   default 69.30.85.101
#   VPS_PORT   default 22046
#   VPS_USER   default root
#   VPS_KEY    default ~/.ssh/id_ed25519
#   REMOTE_LOG default /root/vps_phase_sparse_75ep.log
#   POLL_SEC   default 900 (15 minutes between checks)

set -euo pipefail

VPS_HOST="${VPS_HOST:-69.30.85.101}"
VPS_PORT="${VPS_PORT:-22046}"
VPS_USER="${VPS_USER:-root}"
VPS_KEY="${VPS_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_LOG="${REMOTE_LOG:-/root/vps_phase_sparse_75ep.log}"
POLL_SEC="${POLL_SEC:-900}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH=(ssh -o BatchMode=yes -o ServerAliveInterval=60 -i "${VPS_KEY}" -p "${VPS_PORT}" "${VPS_USER}@${VPS_HOST}")

echo "[wait_pull] Watching ${VPS_USER}@${VPS_HOST}:${REMOTE_LOG} every ${POLL_SEC}s → ${ROOT}/results/"
i=0
while true; do
  i=$((i + 1))
  if "${SSH[@]}" "test -f '${REMOTE_LOG}' && grep -q 'HTML report:' '${REMOTE_LOG}'"; then
    echo "[wait_pull] Completed (poll #${i}). Tail:"
    "${SSH[@]}" "tail -40 '${REMOTE_LOG}'"
    break
  fi
  if ! "${SSH[@]}" "pgrep -f run_algorithm_comparison.py >/dev/null 2>&1" \
    && ! "${SSH[@]}" "pgrep -f scripts/train.py >/dev/null 2>&1"; then
    echo "[wait_pull] No training processes but no HTML report — check log:" >&2
    "${SSH[@]}" "tail -80 '${REMOTE_LOG}'" >&2 || true
    exit 1
  fi
  echo "[wait_pull] poll #${i} $(date -u +"%Y-%m-%dT%H:%M:%SZ") — still running"
  "${SSH[@]}" "tail -2 '${REMOTE_LOG}'" || true
  sleep "${POLL_SEC}"
done

echo "[wait_pull] rsync results → ${ROOT}/results/"
rsync -avz -e "ssh -o BatchMode=yes -i ${VPS_KEY} -p ${VPS_PORT}" \
  "${VPS_USER}@${VPS_HOST}:/root/disertation-rul-prediction/Mamba-xLSTM/results/" \
  "${ROOT}/results/"

echo "[wait_pull] Done."
