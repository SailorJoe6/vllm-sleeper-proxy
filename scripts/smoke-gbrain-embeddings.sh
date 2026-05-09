#!/usr/bin/env bash
set -euo pipefail

LITELLM_BASE_URL="${LITELLM_BASE_URL:-http://127.0.0.1:4000/v1}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8888/v1}"
LITELLM_MODEL_ALIAS="${LITELLM_MODEL_ALIAS:-Qwen3-Embedding-8B}"
VLLM_MODEL_ID="${VLLM_MODEL_ID:-Qwen/Qwen3-Embedding-8B}"
EXPECTED_DIMENSIONS="${EXPECTED_DIMENSIONS:-4096}"

need() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

need curl
need python3

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

curl -fsS "${VLLM_BASE_URL}/models" > "${tmpdir}/vllm-models.json"
python3 - "${tmpdir}/vllm-models.json" "${VLLM_MODEL_ID}" <<'PY'
import json
import sys

path, expected = sys.argv[1], sys.argv[2]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
ids = [model.get("id") for model in data.get("data", [])]
if expected not in ids:
    raise SystemExit(f"vLLM model missing: {expected}; saw {ids}")
print(f"PASS vLLM model listed: {expected}")
PY

curl -fsS "${LITELLM_BASE_URL}/models" > "${tmpdir}/litellm-models.json"
python3 - "${tmpdir}/litellm-models.json" "${LITELLM_MODEL_ALIAS}" <<'PY'
import json
import sys

path, expected = sys.argv[1], sys.argv[2]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
ids = [model.get("id") for model in data.get("data", [])]
if expected not in ids:
    raise SystemExit(f"LiteLLM alias missing: {expected}; saw {ids}")
print(f"PASS LiteLLM alias listed: {expected}")
PY

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
