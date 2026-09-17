# Multimodal Chat Transport

`vllm-sleeper-proxy` forwards OpenAI-compatible chat completions to a
configured vLLM engine:

```text
POST /v1/chat/completions
```

Both buffered JSON and Server-Sent Events (SSE) are supported. This transport
is suitable for text-only and inline multimodal requests. The proxy changes
only the top-level logical `model` value; message content, including ordered
text and `image_url` parts, reaches vLLM unchanged.

This capability does not deploy a VLM by itself. The engine, exact model
revision, media limits, memory settings, LiteLLM mapping, and network policy
remain deployment-owned configuration.

## Model configuration

Configure multiple engines with `SLEEPER_MODELS`:

```json
[
  {
    "name": "Qwen3-Embedding-8B",
    "upstream_model": "Qwen/Qwen3-Embedding-8B",
    "upstream_base_url": "http://embedding-vllm:8888/v1",
    "control_base_url": "http://embedding-vllm:8888"
  },
  {
    "name": "chess-vlm-bootstrap",
    "upstream_model": "Qwen/Qwen3-VL-4B-Instruct",
    "upstream_base_url": "http://vision-vllm:8000/v1",
    "control_base_url": "http://vision-vllm:8000"
  }
]
```

The example describes routing only. Pin the VLM revision and resource envelope
in the engine command or deployment manifest.

Relevant proxy settings are:

| Variable | Default | Meaning |
| --- | --- | --- |
| `SLEEPER_REQUEST_TIMEOUT_SECONDS` | `30` | Upstream connection and read timeout |
| `SLEEPER_WAKE_TIMEOUT_SECONDS` | `300` | Readiness deadline after wake |
| `SLEEPER_DRAIN_TIMEOUT_SECONDS` | `300` | Deadline for in-flight requests to drain before switching |
| `SLEEPER_MAX_REQUEST_BODY_BYTES` | `10485760` | Maximum buffered client request body |
| `SLEEPER_SLEEP_LEVEL` | `2` | vLLM sleep level used before a model switch |
| `SLEEPER_ADMISSION_STATUS_PATH` | unset | Optional host-monitor JSON status; when set, every model request requires a fresh allowed `model_admission` decision before wake |

## Startup reconciliation

The proxy does not treat an empty in-process active-model field as proof that
engines are asleep. Before it binds its HTTP listener, it queries
`/is_sleeping` for every configured engine, sends level-2 `/sleep` to every
awake engine, and verifies that all engines report sleeping. The initial
`/healthz` state is therefore `active_model: null` only after the physical
engine state matches it.

Reconciliation is mandatory and fail-closed. If an engine is unreachable,
returns an error or an ambiguous sleep-state payload, rejects `/sleep`, or does
not enter sleep before the wake timeout, the proxy process exits without
opening its listening port. The service supervisor may retry after the engine
control endpoints become ready.

The request limit includes inline base64 media. Increase it deliberately when
larger local images are required.

## Buffered request

```bash
curl --fail-with-body http://127.0.0.1:8889/v1/chat/completions \
  -H 'content-type: application/json' \
  --data '{
    "model": "chess-vlm-bootstrap",
    "temperature": 0,
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Return the observed chess move in UCI."},
        {"type": "image_url", "image_url": {
          "url": "data:image/jpeg;base64,LOCAL_IMAGE_DATA"
        }}
      ]
    }]
  }'
```

## Streaming request

```bash
curl --no-buffer --fail-with-body \
  http://127.0.0.1:8889/v1/chat/completions \
  -H 'content-type: application/json' \
  --data '{
    "model": "chess-vlm-bootstrap",
    "stream": true,
    "messages": [{"role": "user", "content": "Return e2e4."}]
  }'
```

The proxy forwards upstream SSE bytes in order and flushes each received chunk.
It removes hop-by-hop and length headers and closes the downstream connection
when the upstream stream ends.

## Lifecycle and concurrency

Before inference, the proxy:

1. resolves the logical model or alias;
2. waits for requests using another active model to finish;
3. sleeps the old engine at level 2;
4. wakes the requested engine, resets its multimodal cache after the level-2
   reload, and checks sleep state and model discovery;
5. acquires request ownership;
6. forwards the request;
7. releases ownership on every exit from forwarding.

Ownership covers the full buffered response and the full streaming response,
not only wake-up. Concurrent requests for the active model may proceed.
A request for another model drains current ownership before initiating sleep.
Once a switch is pending, new requests for the old model wait behind it so
steady traffic cannot starve the switch. Drain waits fail at the configured
deadline.
Exceptions and downstream disconnects close the upstream stream and release
ownership.

`POST /sleep` provides a bounded resource-backoff path. It stops new model
acquisitions, waits for existing buffered or streaming ownership to drain,
sleeps the active engine at level 2, verifies sleep, and returns the logical
model name. It never evicts an in-flight response.

Deployments may run `python -m vllm_sleeper_proxy.memory_monitor` beside the
proxy. It samples total-host memory at a short interval and invokes that exact
sleep control when utilization reaches its configured ceiling. The default is
95%, and values above 95% are rejected. Cold-start safety remains deployment-owned: use a host preflight
and serialize engine startup so the proxy and monitor do not need to recover
from an unbudgeted startup overlap.

## Errors and failure semantics

- Unsupported paths return `404`.
- Invalid JSON, missing models, truncated bodies, and invalid content lengths
  return `400`.
- Missing lengths and chunked request uploads return `411`; client request
  bodies must provide `Content-Length`.
- Unknown models return `404` with `type=unknown_model`.
- Oversized bodies return `413` with `type=request_too_large`.
- Wake failures return `503`, `type=wake_failed`, and `Retry-After: 10`.
- A fresh affirmative denied model-specific resource decision returns `503`,
  `type=admission_denied`, before wake. Required models continue through missing,
  stale, malformed, or unknown monitor state; optional models return `503` for
  those uncertainty cases.
- A failed resource-backoff sleep returns `503`, `type=sleep_failed`.
- An upstream connection or timeout failure before response headers returns
  `502` with `type=upstream_unavailable`.
- If an upstream streaming failure occurs after response headers, the proxy
  closes the stream. It cannot replace a partially delivered SSE response with
  a JSON error.

Backpressure is synchronous: a slow client blocks the corresponding proxy
handler's upstream reads. Client disconnect or cancellation closes the
upstream response and frees model ownership.

## Compatibility and rollback

Embedding-only deployments remain supported. Leave `SLEEPER_MODELS` unset (or
configure only the existing embedding engine) and continue using
`POST /v1/embeddings`; chat support does not alter embedding request bodies or
response handling.

To roll back multimodal exposure without reverting the proxy, remove the VLM
from `SLEEPER_MODELS` and from the deployment's LiteLLM and network policy.
Chat requests for that logical model then fail as unknown while embeddings
continue to operate.

## Known limitations

- Request bodies are buffered before forwarding; only response streaming is
  incremental.
- Chunked client request uploads are not supported.
- The proxy accepts inline and remote-looking media references transparently.
  Enforce local-only media by disabling remote URL access in vLLM and at the
  deployment network boundary.
- Read timeouts apply to upstream socket operations; the proxy does not impose
  a separate total streaming-duration deadline.
- A downstream disconnect while waiting for another model to drain is observed
  only after the queued handler acquires the model and begins forwarding.
- Health remains proxy liveness plus active-model reporting. Deployment
  readiness must additionally exercise discovery and an inference request.
- The file admission guard trusts only the status schema and freshness written
  by the deployment's host monitor; it does not calculate cold-start peaks.
