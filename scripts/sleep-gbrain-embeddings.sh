#!/usr/bin/env bash
set -euo pipefail

VLLM_CONTROL_URL="${VLLM_CONTROL_URL:-http://127.0.0.1:8888}"
SLEEP_LEVEL="${SLEEP_LEVEL:-2}"

need() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

need curl

curl -fsS -X POST "${VLLM_CONTROL_URL%/}/sleep?level=${SLEEP_LEVEL}" >/dev/null
printf 'Slept vLLM at %s with level=%s\n' "${VLLM_CONTROL_URL%/}" "${SLEEP_LEVEL}"
