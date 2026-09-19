#!/usr/bin/env bash
# Poll VPS until vps_xjtu_fulldata_rerun.sh finishes, then rsync results locally.
#
# Env:
#   VPS_HOST   default 194.68.245.201
#   VPS_PORT   default 22161
#   VPS_USER   default root
#   VPS_KEY    default ~/.ssh/id_ed25519
#   REMOTE_LOG default /root/vps_xjtu_fulldata_rerun.log
#   POLL_SEC   default 900 (15 minutes)

set -euo pipefail

VPS_HOST="${VPS_HOST:-194.68.245.201}"
VPS_PORT="${VPS_PORT:-22161}"
VPS_USER="${VPS_USER:-root}"
VPS_KEY="${VPS_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_LOG="${REMOTE_LOG:-/root/vps_xjtu_fulldata_rerun.log}"
POLL_SEC="${POLL_SEC:-900}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH=(ssh -o BatchMode=yes -o ServerAliveInterval=60 -i "${VPS_KEY}" -p "${VPS_PORT}" "${VPS_USER}@${VPS_HOST}")

echo "[wait_pull] Watching ${VPS_USER}@${VPS_HOST}:${REMOTE_LOG} every ${POLL_SEC}s → ${ROOT}/results/"

while true; do
  if "${SSH[@]}" "grep -q 'XJTU full-data VPS rerun complete' '${REMOTE_LOG}' 2>/dev/null"; then
    echo "[wait_pull] Remote log shows completion."
    break
  fi
  if ! "${SSH[@]}" "pgrep -f 'vps_xjtu_fulldata_rerun|run_algorithm_comparison.*xjtusy' >/dev/null 2>&1"; then
    if "${SSH[@]}" "test -f '${REMOTE_LOG}' && grep -q 'XJTU full-data VPS rerun complete' '${REMOTE_LOG}'"; then
      break
    fi
    echo "[wait_pull] WARN: no training procs and log not complete — check ${REMOTE_LOG} on VPS."
  fi
  "${SSH[@]}" "tail -3 '${REMOTE_LOG}' 2>/dev/null" || true
  echo "[wait_pull] sleeping ${POLL_SEC}s..."
  sleep "${POLL_SEC}"
done

rsync -az -e "ssh -i ${VPS_KEY} -p ${VPS_PORT}" \
  "${VPS_USER}@${VPS_HOST}:/root/disertation-rul-prediction/Mamba-xLSTM/results/" \
  "${ROOT}/results/"

echo "[wait_pull] Done. Local results: ${ROOT}/results/"
