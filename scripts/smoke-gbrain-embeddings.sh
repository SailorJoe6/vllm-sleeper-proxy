#!/usr/bin/env bash
set -euo pipefail

LITELLM_BASE_URL="${LITELLM_BASE_URL:-http://127.0.0.1:4000/v1}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8888/v1}"
VLLM_CONTROL_URL="${VLLM_CONTROL_URL:-${VLLM_BASE_URL%/v1}}"
SLEEPER_PROXY_BASE_URL="${SLEEPER_PROXY_BASE_URL:-http://127.0.0.1:8889/v1}"
LITELLM_MODEL_ALIAS="${LITELLM_MODEL_ALIAS:-Qwen3-Embedding-8B}"
VLLM_MODEL_ID="${VLLM_MODEL_ID:-Qwen/Qwen3-Embedding-8B}"
EXPECTED_DIMENSIONS="${EXPECTED_DIMENSIONS:-4096}"
SLEEPER_WAKE_PATH_SMOKE="${SLEEPER_WAKE_PATH_SMOKE:-0}"
SLEEP_LEVEL="${SLEEP_LEVEL:-1}"

need() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

check_models() {
  local label="$1"
  local url="$2"
  local expected="$3"
  local output="$4"

  curl -fsS "${url}/models" > "${output}"
  python3 - "${output}" "${expected}" "${label}" <<'PY'
import json
import sys

path, expected, label = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
ids = [model.get("id") for model in data.get("data", [])]
if expected not in ids:
    raise SystemExit(f"{label} model missing: {expected}; saw {ids}")
print(f"PASS {label} model listed: {expected}")
PY
}

need curl
need python3

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

if [[ "${SLEEPER_WAKE_PATH_SMOKE}" == "1" ]]; then
  curl -fsS -X POST "${VLLM_CONTROL_URL%/}/sleep?level=${SLEEP_LEVEL}" >/dev/null
  echo "PASS vLLM sleep requested before LiteLLM embedding: level=${SLEEP_LEVEL}"
else
  check_models "vLLM" "${VLLM_BASE_URL}" "${VLLM_MODEL_ID}" "${tmpdir}/vllm-models.json"
fi

check_models "sleeper proxy" "${SLEEPER_PROXY_BASE_URL}" "${LITELLM_MODEL_ALIAS}" "${tmpdir}/sleeper-proxy-models.json"
check_models "LiteLLM alias" "${LITELLM_BASE_URL}" "${LITELLM_MODEL_ALIAS}" "${tmpdir}/litellm-models.json"

cat > "${tmpdir}/embedding-request.json" <<JSON
{"model":"${LITELLM_MODEL_ALIAS}","input":"gbrain local embedding smoke test"}
JSON

curl -fsS \
  -H "content-type: application/json" \
  -d @"${tmpdir}/embedding-request.json" \
  "${LITELLM_BASE_URL}/embeddings" > "${tmpdir}/embedding-response.json"

python3 - "${tmpdir}/embedding-response.json" "${EXPECTED_DIMENSIONS}" <<'PY'
import json
import sys

path, expected_s = sys.argv[1], sys.argv[2]
expected = int(expected_s)
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
embedding = data["data"][0]["embedding"]
actual = len(embedding)
if actual != expected:
    raise SystemExit(f"embedding length mismatch: expected {expected}, got {actual}")
print(f"PASS embedding vector length: {actual}")
PY

if [[ "${SLEEPER_WAKE_PATH_SMOKE}" == "1" ]]; then
  check_models "vLLM post-wake" "${VLLM_BASE_URL}" "${VLLM_MODEL_ID}" "${tmpdir}/vllm-models-post-wake.json"
fi
