# vllm-sleeper-proxy

`vllm-sleeper-proxy` is a lightweight, OpenAI-compatible proxy that enables **on-demand model switching and discovery** for vLLM on memory-constrained, single-GPU systems (such as NVIDIA DGX Spark).

The proxy sits in front of one or more vLLM engines and provides a **stable HTTP surface** that agent frameworks and clients can treat as a persistent inference endpoint—even though the underlying models may be **asleep, unloaded, or dynamically activated**.

At any given time, the proxy enforces an **at-most-one awake model invariant**, using **vLLM Sleep Mode (Level 2)** to fully release model weights and KV cache before another model is activated.

This allows individual models to consume **nearly all available GPU / unified memory** (for example, very large context windows) while still supporting fast, request-driven switching between models such as *coding*, *reasoning*, or *planning* LLMs—without full cold restarts.

Before binding its HTTP listener, the proxy queries every configured engine,
puts each awake engine into level-2 sleep, and verifies that every engine
reports sleeping. Only then does it begin serving with `active_model: null`.
Startup fails closed if any engine is unavailable, rejects the sleep request,
or cannot provide a boolean `/is_sleeping` state. This startup reconciliation
repairs model state lost across proxy, container, Docker, or host restarts.

---

See [Proxy requirements](docs/requirements.md) for normative lifecycle ownership and deployment traceability.

A deployment must declare its required engine lineup. Startup/readiness fails if
any required engine is absent or cannot be reconciled. For a request, the proxy
routes directly when the requested model is already the sole awake and ready
model; otherwise it drains and sleeps the current model, rechecks admission,
wakes the requested model, waits for full readiness, and only then forwards the
request. Lifecycle control for models outside the required lineup is deferred pending a
separate design sprint; no such model is defined by this deployment contract.

## Documentation map

- [Requirements](docs/requirements.md): normative lifecycle and deployment rules.
- [Multimodal chat transport](docs/multimodal-chat.md): request formats, streaming,
  image handling, and transport limits.
- [Design notes](docs/design%20ideas.md): exploratory implementation notes; the
  requirements document takes precedence when they differ.

## Lifecycle safety lease

The proxy treats model startup as an exclusive lifecycle state, not as an
ordinary wake request. A startup lease is acquired before the first wake or
model switch and is held until the target engine has passed `/is_sleeping`,
`/v1/models` readiness, and configured smoke/resource checks. While that lease
is held:

* all other models remain hibernated;
* competing wake requests wait behind the lease;
* the starting model cannot be hibernated by another request; and
* `/healthz` reports the `starting_model` separately from `active_model`.

The lease is an OS file lock. Set `SLEEPER_STARTUP_LEASE_PATH` to a path shared
by all proxy instances on the host (for example a mounted `/run` directory).
For each model that needs an explicit startup smoke request, add
`startup_smoke_path` and `startup_smoke_body` to its `SLEEPER_MODELS` entry.
The proxy runs that request and re-runs the admission check before releasing
the startup lease. A smoke failure leaves the model unmarked and attempts to
put it back to sleep.
The operating system releases the lock if the proxy crashes, so recovery does
not leave a permanent lease. Proxy-only recovery uses `POST /startup/adopt` to
probe and adopt exactly zero or one already-awake engine without sleeping,
waking, or stopping any engine. Unknown state or multiple awake engines defer
recovery without mutation. Whole-system startup instead uses the same lease
while reconciling every configured engine to level-2 sleep.

## Current implementation

This repo now includes a minimal, dependency-light Python sleeper proxy plus a
GBrain embedding Compose integration:

* `vllm_sleeper_proxy/` exposes `/v1/models`, `/api/tags`, `/healthz`, a
  lifecycle-safe `POST /sleep` control, and
  OpenAI-compatible request forwarding for `/v1/embeddings` and
  `/v1/chat/completions`.
* Before forwarding a request, the proxy calls vLLM `/wake_up`, waits for
  `/is_sleeping` and `/v1/models` readiness, resets vLLM's multimodal cache
  after every level-2 reload, rewrites the logical model alias
  or LiteLLM/vLLM model id to the upstream vLLM model id, then forwards the
  request.
* Chat completions preserve image-and-text message content and support both
  buffered JSON responses and streaming Server-Sent Events (SSE).
* Each request owns its active model until its buffered or streaming response
  ends. A request for another model waits before sleeping the current engine,
  preventing an in-flight response from being evicted.
* Each `SLEEPER_MODELS` entry can declare `"required": true`. File-backed
  resource admission keeps required models available when the host monitor is
  missing, stale, malformed, or reports unknown model state, while fresh
  affirmative pressure still denies. Optional models remain fail-closed.
* A separate optional `SLEEPER_THERMAL_ADMISSION_STATUS_PATH` fast projection
  keeps required acquisitions available during warning or observability-only
  uncertainty. Fresh sleep, cutoff, or danger-derived recovery state rejects
  acquisition before request-body handling with HTTP 503, `Retry-After`, and
  `error.type=thermal_cooldown`. Optional models also reject missing, stale, or
  malformed projections. The fast thermal projection owns thermal denial for
  required models, so a slower resource snapshot cannot extend a cleared
  thermal warning. Existing leases may drain; `POST /sleep` quiesces later
  acquisitions. The consumer does not own deployment thresholds or
  positive-danger provenance.
  The bundled memory monitor polls total-host `MemAvailable` and asks the
  proxy to quiesce and sleep the active model at the configured ceiling.
* `services/gbrain-embeddings/` wires GBrain → LiteLLM → sleeper proxy → vLLM
  for `Qwen/Qwen3-Embedding-8B`.
* `scripts/sleep-gbrain-embeddings.sh` intentionally sleeps Qwen so GPU memory
  can be reclaimed; the next GBrain embedding call should wake it again.

## Key features

* **Single OpenAI-compatible inference endpoint**
* **Strict sleep-before-wake orchestration** (vLLM Sleep Mode Level 2)
* **Request-driven model activation** based on the `model` field
* **At-most-one awake model invariant** (prevents OOM on unified-memory systems)
* **Fast switching** between frequently used models
* **Model discovery APIs** compatible with common local runtimes
* **Multimodal chat forwarding** with streaming and in-flight draining
* Designed for **home labs and single-node setups**, not Kubernetes

## Supported inference endpoints

| Endpoint | Request | Response |
| --- | --- | --- |
| `POST /v1/embeddings` | OpenAI-compatible JSON | Buffered upstream response |
| `POST /v1/chat/completions` | OpenAI-compatible text or inline multimodal JSON | Buffered JSON, or SSE when `"stream": true` |
| `POST /sleep` | Empty body | Quiesces new inference, drains ownership, then sleeps the active model |

The proxy rewrites only the top-level `model` field. It does not transform
`messages`, inline `data:` image URLs, generation parameters, or response
content. Remote-media policy belongs to the vLLM engine and deployment; a
local-only deployment should allow inline content and leave vLLM remote media
access disabled.

See [Multimodal chat transport](docs/multimodal-chat.md) for configuration,
lifecycle, failure behavior, smoke tests, compatibility, and limitations.

---

## Model discovery & compatibility

In addition to proxying inference requests, `vllm-sleeper-proxy` presents **model discovery endpoints** so that clients can programmatically determine which models *can be served*, even if those models are not currently loaded.

The proxy supports:

* **OpenAI-style model enumeration**

  * `GET /v1/models`

  
* **Ollama-style model enumeration**

  * `GET /api/tags`

Both endpoints return a **logical view of available models**, representing models that the proxy knows how to activate on demand—not necessarily models that are currently resident in GPU memory.

This allows:

* Agent frameworks to auto-discover usable models
* IDEs and tools to populate model pickers
* Clients to remain stateless and declarative (“use model X”)

The proxy treats **model availability** as a first-class concern, decoupled from **model liveness**.

---

## Why this exists

On systems like DGX Spark, a single large model (for example FP8/NVFP4 with a 256k context window) can consume ~90–95% of available memory. Running multiple models concurrently is impractical, but fully restarting inference servers on every model switch is too slow for interactive or agentic workflows.

At the same time, modern agent frameworks increasingly expect:

* A stable OpenAI-compatible endpoint
* Dynamic model discovery
* Declarative model selection
* The ability to tolerate temporarily unavailable backends

`vllm-sleeper-proxy` solves this by:

* Treating **model lifecycle management** as a first-class responsibility
* Abstracting away sleep, wake, and unload semantics
* Presenting a **consistent, discoverable model surface** to clients

---

## Conceptual model

From the client’s perspective:

* Models are always “available”
* `/v1/models` or `/api/tags` returns a stable list
* Requests specify the desired model declaratively

From the proxy’s perspective:

* Only one model may be awake at a time
* Models transition between *asleep* and *active* states
* Memory pressure is managed explicitly and deterministically

This separation allows clients and agents to remain simple, while the proxy absorbs the complexity of operating near the hardware limits.

---

## Non-goals

* Serving multiple base models concurrently
* Kubernetes or production-grade orchestration
* Sharing KV cache or weights across models
* Acting as a general model registry or downloader

This project intentionally favors **determinism, simplicity, and maximal hardware utilization** over generality.
