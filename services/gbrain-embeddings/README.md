# GBrain Embeddings Service

This directory is the production source of truth for the DGX Spark local
embedding stack used by GBrain.

The supported client endpoint is LiteLLM:

```text
http://127.0.0.1:4000/v1
http://<dgx-spark-private-lan-ip>:4000/v1
```

vLLM remains reachable on `:8888` for inspection, but GBrain clients should use
LiteLLM on `:4000`.

## What Runs

```text
GBrain
  -> LiteLLM :4000
  -> vLLM :8888
  -> Qwen/Qwen3-Embedding-8B
```

LiteLLM exposes only the `Qwen3-Embedding-8B` alias for this service. Chat,
coder, and reasoning models intentionally remain outside this always-on stack
so the future sleeper proxy can manage them separately.

The legacy `/home/sailorjoe6/litellm/start` helper is no longer the production
run path. Keep it only as an ad hoc development helper unless an operator
explicitly removes it.

## Files

- `docker-compose.yml` defines the LiteLLM and vLLM containers, ports, restart
  policies, GPU reservation, and readiness health checks.
- `litellm.yaml` maps `Qwen3-Embedding-8B` to the vLLM OpenAI-compatible
  backend.
- `.env.example` documents all host-specific overrides.
- `gbrain-embeddings.service` supervises the Compose project with systemd.
- `../../scripts/install-gbrain-embeddings-systemd.sh` installs and enables the
  unit.
- `../../scripts/status-gbrain-embeddings.sh` prints systemd, Compose, and
  endpoint status.
- `../../scripts/smoke-gbrain-embeddings.sh` validates both model lists and a
  4096-dimensional embedding response.

## Local Compose

From this directory:

```bash
docker compose config
docker compose up -d
docker compose ps
../../scripts/smoke-gbrain-embeddings.sh
```

Expected Compose state after startup:

```text
gbrain-embeddings-vllm      Up ... (healthy)   0.0.0.0:8888->8888/tcp
gbrain-embeddings-litellm   Up ... (healthy)   0.0.0.0:4000->4000/tcp
```

The vLLM container can take a few minutes to become healthy while loading
`Qwen/Qwen3-Embedding-8B`. LiteLLM starts after the vLLM health check passes.

Stop the stack:

```bash
docker compose down
```

The containers use `restart: unless-stopped`, so process exits are restarted by
Docker. Manual operator stops such as `docker stop` or `docker kill` are treated
as intentional stops and should be followed by `docker compose up -d` or
`sudo systemctl restart gbrain-embeddings`.

## systemd Install

The unit is pinned to:

```text
/home/sailorjoe6/Code/vllm-sleeper-proxy/services/gbrain-embeddings
```

Install and enable it:

```bash
../../scripts/install-gbrain-embeddings-systemd.sh
sudo systemctl start gbrain-embeddings
```

Operator commands:

```bash
sudo systemctl start gbrain-embeddings
sudo systemctl stop gbrain-embeddings
sudo systemctl restart gbrain-embeddings
systemctl status gbrain-embeddings --no-pager
journalctl -u gbrain-embeddings -f
```

Validation after install or restart:

```bash
systemctl is-active --quiet gbrain-embeddings && echo active
systemctl is-enabled --quiet gbrain-embeddings && echo enabled
journalctl -u gbrain-embeddings -n 80 --no-pager
docker compose ps
../../scripts/smoke-gbrain-embeddings.sh
```

The systemd unit sets `HOME=/home/sailorjoe6` and
`HF_CACHE_DIR=/home/sailorjoe6/.cache/huggingface` so boot-time Compose runs use
the same Hugging Face cache as interactive validation.

To validate container failure recovery under systemd, terminate the vLLM process
inside the managed container and wait for Docker to restart it:

```bash
docker exec gbrain-embeddings-vllm pkill -TERM -f "vllm serve"
docker inspect gbrain-embeddings-vllm --format '{{.RestartCount}}'
docker compose ps
../../scripts/smoke-gbrain-embeddings.sh
```

Manual `docker stop` and `docker kill` are operator stops under
`restart: unless-stopped`; recover those with
`sudo systemctl restart gbrain-embeddings`.

Reboot validation, when practical:

```bash
sudo reboot
systemctl is-active --quiet gbrain-embeddings && echo active
systemctl is-enabled --quiet gbrain-embeddings && echo enabled
../../scripts/smoke-gbrain-embeddings.sh
```

## Smoke Test

Run:

```bash
../../scripts/smoke-gbrain-embeddings.sh
```

Expected evidence:

```text
PASS vLLM model listed: Qwen/Qwen3-Embedding-8B
PASS LiteLLM alias listed: Qwen3-Embedding-8B
PASS embedding vector length: 4096
```

The direct model endpoints should also list the expected IDs:

```bash
curl -fsS http://127.0.0.1:4000/v1/models
curl -fsS http://127.0.0.1:8888/v1/models
```

For a LAN client, override URLs:

```bash
LITELLM_BASE_URL=http://10.0.4.225:4000/v1 \
VLLM_BASE_URL=http://10.0.4.225:8888/v1 \
../../scripts/smoke-gbrain-embeddings.sh
```

Use the private LAN address chosen by the operator. Do not expose this
unauthenticated endpoint to the public internet.

## GBrain Client Configuration

Host-local GBrain should use:

```text
embedding_model = litellm:Qwen3-Embedding-8B
embedding_dimensions = 4096
LITELLM_BASE_URL = http://127.0.0.1:4000/v1
```

Remote GBrain instances on the private LAN should use the DGX Spark private LAN
address on port `4000`.

## Fresh GBrain Compatibility Smoke

Use an isolated `GBRAIN_HOME` so this does not mutate an existing brain. Unset
cloud API keys to prove the embedding path is local-only:

```bash
rm -rf /tmp/gbrain-litellm-smoke
mkdir -p /tmp/gbrain-litellm-smoke/corpus
printf '%s\n' \
  '# Local Embedding Smoke' \
  '' \
  'GBrain verifies local LiteLLM embeddings using Qwen3-Embedding-8B.' \
  '' \
  'The retrieval keyword is obsidian-papaya.' \
  > /tmp/gbrain-litellm-smoke/corpus/local-embedding-smoke.md

env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY \
  GBRAIN_HOME=/tmp/gbrain-litellm-smoke \
  LITELLM_BASE_URL=http://127.0.0.1:4000/v1 \
  gbrain init --pglite \
    --embedding-model litellm:Qwen3-Embedding-8B \
    --embedding-dimensions 4096

env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY \
  GBRAIN_HOME=/tmp/gbrain-litellm-smoke \
  LITELLM_BASE_URL=http://127.0.0.1:4000/v1 \
  gbrain import /tmp/gbrain-litellm-smoke/corpus --progress-json

env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY \
  GBRAIN_HOME=/tmp/gbrain-litellm-smoke \
  LITELLM_BASE_URL=http://127.0.0.1:4000/v1 \
  gbrain search obsidian-papaya

env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY \
  GBRAIN_HOME=/tmp/gbrain-litellm-smoke \
  LITELLM_BASE_URL=http://127.0.0.1:4000/v1 \
  gbrain query "What page mentions obsidian-papaya?" --no-expand

env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY \
  GBRAIN_HOME=/tmp/gbrain-litellm-smoke \
  LITELLM_BASE_URL=http://127.0.0.1:4000/v1 \
  gbrain doctor --json
```

Expected evidence:

- `gbrain init` creates a PGLite brain with
  `litellm:Qwen3-Embedding-8B` at 4096 dimensions.
- Import reports `1 pages imported` and `1 chunks created`.
- Search and query return `local-embedding-smoke`.
- Doctor reports `embedding_provider` as
  `litellm:Qwen3-Embedding-8B` with `4096 dims, DB aligned`.

Validated on 2026-05-09 with the systemd-managed service active and enabled.
The local test used `oven/bun:1.3.13` with host networking to run the GBrain
checkout because the interactive `gbrain` binary was not on this shell's PATH;
the service-facing inputs were the same as above. The LiteLLM LAN endpoint
`http://10.0.4.225:4000/v1/models` listed `Qwen3-Embedding-8B`, and a LAN-address
embedding request returned a 4096-dimensional vector from this host.

## Image Updates and Rollback

LiteLLM is pinned by repo digest in `docker-compose.yml` and `.env.example`.
Update deliberately:

1. Edit `LITELLM_IMAGE` to the candidate digest.
2. Run `docker compose pull litellm`.
3. Start the stack.
4. Run `../../scripts/smoke-gbrain-embeddings.sh`.
5. Commit the digest only if the smoke test passes.

Rollback:

```bash
git checkout HEAD~1 -- services/gbrain-embeddings/docker-compose.yml services/gbrain-embeddings/.env.example
sudo systemctl restart gbrain-embeddings
../../scripts/smoke-gbrain-embeddings.sh
```

The vLLM image currently uses the locally built `vllm-node:latest` from the
validated `spark-vllm-docker` workflow. A future slice may tag that image with a
service-specific stable tag after runtime validation.
