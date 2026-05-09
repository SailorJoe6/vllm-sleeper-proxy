#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_DIR="${REPO_ROOT}/services/gbrain-embeddings"

cd "${SERVICE_DIR}"

echo "== systemd =="
if command -v systemctl >/dev/null 2>&1; then
  systemctl is-active --quiet gbrain-embeddings
  echo "active: yes"
  if systemctl is-enabled --quiet gbrain-embeddings; then
    echo "enabled: yes"
  else
    echo "enabled: no" >&2
    exit 1
  fi
  systemctl status gbrain-embeddings --no-pager
else
  echo "systemctl not found"
fi

echo
echo "== compose =="
docker compose ps

echo
echo "== endpoints =="
curl -fsS "http://127.0.0.1:${LITELLM_HOST_PORT:-4000}/v1/models"
echo
curl -fsS "http://127.0.0.1:${VLLM_HOST_PORT:-8888}/v1/models"
echo
