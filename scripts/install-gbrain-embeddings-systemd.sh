#!/usr/bin/env bash
set -euo pipefail

UNIT_NAME="gbrain-embeddings.service"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_DIR="${REPO_ROOT}/services/gbrain-embeddings"
UNIT_SRC="${SERVICE_DIR}/${UNIT_NAME}"
UNIT_DEST="/etc/systemd/system/${UNIT_NAME}"

if [[ ! -f "${UNIT_SRC}" ]]; then
  echo "Missing unit file: ${UNIT_SRC}" >&2
  exit 1
fi

if [[ "${SERVICE_DIR}" != "/home/sailorjoe6/Code/vllm-sleeper-proxy/services/gbrain-embeddings" ]]; then
  echo "This unit is pinned to /home/sailorjoe6/Code/vllm-sleeper-proxy." >&2
  echo "Move the repo there or update services/gbrain-embeddings/${UNIT_NAME} deliberately." >&2
  exit 1
fi

sudo install -m 0644 "${UNIT_SRC}" "${UNIT_DEST}"
sudo systemctl daemon-reload
sudo systemctl enable "${UNIT_NAME}"

echo "Installed ${UNIT_DEST}"
echo "Start with: sudo systemctl start gbrain-embeddings"
echo "Check with: sudo systemctl status gbrain-embeddings --no-pager"
