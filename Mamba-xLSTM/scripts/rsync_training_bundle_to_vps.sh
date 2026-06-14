#!/usr/bin/env bash
# Minimal upload to run scripts/train.py and scripts/run_algorithm_comparison.py on a VPS.
# Syncs:
#   - Mamba-xLSTM (code + configs + tests; excludes venv, caches, bulky results)
#   - data-bearing (optional; see below)
#
# **VPS default (see .cursor/rules/vps-ssh-key-access.mdc §6 + §6.3):** on the server, download
#   https://dataset-bearing-rul.s3.ap-southeast-2.amazonaws.com/data-bearing/data-bearing.zip
#   and https://dataset-bearing-rul.s3.ap-southeast-2.amazonaws.com/data-bearing/xtju-sy.zip
#   into REMOTE_BASE and unzip — do not rsync the dataset from the laptop.
#   From the laptop use: RSYNC_SKIP_DATA=1 … bash rsync_training_bundle_to_vps.sh
#
# Layout on the remote MUST be:
#   <REMOTE_BASE>/Mamba-xLSTM/
#   <REMOTE_BASE>/data-bearing/
# Train from REMOTE_BASE/Mamba-xLSTM after sync.
#
# Usage (from laptop):
#   bash Mamba-xLSTM/scripts/rsync_training_bundle_to_vps.sh
#   RSYNC_REMOTE=root@YOUR_IP:/root/disertation-rul-prediction \
#   RSYNC_RSH='ssh -i ~/.ssh/id_ed25519 -p 15891' \
#     bash Mamba-xLSTM/scripts/rsync_training_bundle_to_vps.sh
#
# Code-only (VPS + S3 dataset on server):
#   RSYNC_SKIP_DATA=1 RSYNC_REMOTE=... RSYNC_RSH='...' bash Mamba-xLSTM/scripts/rsync_training_bundle_to_vps.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MBA="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT="$(cd "${MBA}/.." && pwd)"

REMOTE_BASE="${RSYNC_REMOTE:-}"
RSYNC_RSH="${RSYNC_RSH:-ssh -p 22}"
EXCLUDES=(
  '--exclude=.venv/'
  '--exclude=__pycache__/'
  '--exclude=*/__pycache__/'
  '--exclude=*.py[cod]'
  '--exclude=.pytest_cache/'
  '--exclude=.ruff_cache/'
  '--exclude=htmlcov/'
  '--exclude=results/runs/'
  '--exclude=results/reports/'
  '--exclude=results/_logs/'
  '--exclude=results/_comparison_configs/'
  '--exclude=results/_chapter_assets/'
  '--exclude=results/tables/'
  '--exclude=results/v1_initial/'
  '--exclude=results/journal_q2/'
  '--exclude=results/bpfx_mapping/'
)

if [[ -z "${REMOTE_BASE}" ]]; then
  echo "Set RSYNC_REMOTE to user@host:path (remote parent of Mamba-xLSTM and data-bearing)." >&2
  echo "Example: RSYNC_REMOTE=root@1.2.3.4:/root/disertation-rul-prediction bash $0" >&2
  exit 2
fi

RSYNC_SKIP_DATA="${RSYNC_SKIP_DATA:-0}"
if [[ "${RSYNC_SKIP_DATA}" != "1" ]] && [[ ! -d "${ROOT}/data-bearing" ]]; then
  echo "error: missing ${ROOT}/data-bearing (datasets). Clone or symlink it beside Mamba-xLSTM, or set RSYNC_SKIP_DATA=1 if the VPS already has data-bearing from S3." >&2
  exit 1
fi

export RSYNC_RSH

# RSYNC_REMOTE is user@host:/abs/path ; rsync accepts that form; echoed hints need path only (after SSH).
REMOTE_HOSTPATH="${REMOTE_BASE#*:}"

echo "[rsync] Mamba-xLSTM -> ${REMOTE_BASE%/}/Mamba-xLSTM/"
/usr/bin/rsync -avz "${EXCLUDES[@]}" \
  "${MBA}/" "${REMOTE_BASE%/}/Mamba-xLSTM/"

if [[ "${RSYNC_SKIP_DATA}" == "1" ]]; then
  echo "[rsync] Skipping data-bearing (RSYNC_SKIP_DATA=1). On VPS unzip S3 archive into ${REMOTE_HOSTPATH%/}/ — see vps-ssh-key-access.mdc §6."
else
  echo "[rsync] data-bearing -> ${REMOTE_BASE%/}/data-bearing/"
  /usr/bin/rsync -avz \
    "${ROOT}/data-bearing/" "${REMOTE_BASE%/}/data-bearing/"
fi

echo "[rsync] Done. After SSH (${REMOTE_BASE%%:*}):"
echo "  cd ${REMOTE_HOSTPATH%/}/Mamba-xLSTM && bash scripts/bootstrap_gpu_vps.sh"
echo "  source .venv/bin/activate && python scripts/run_algorithm_comparison.py --help"
