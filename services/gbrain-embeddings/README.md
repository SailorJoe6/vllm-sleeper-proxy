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
