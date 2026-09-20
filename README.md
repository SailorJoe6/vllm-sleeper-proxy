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
repairs state when the Proxy itself bootstraps after a proxy, container, Docker,
or host restart. If a still-running Proxy later loses its owned engine,
`SLP-LIFE-011` revalidates cached ownership under the same lease and exposes a
bounded intentional 503 until positive evidence or supported host repair
converges state.

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

Startup adoption repairs Proxy restart state. Runtime owner validation implements
`SLP-LIFE-011`: before reusing a cached same-model ready path, the Proxy uses one
bounded validation deadline to prove that the owner is awake/listed and every
peer is sleeping. Unknown, conflicting, DNS, control, or forward-open transport
state invalidates cached readiness while retaining the owner, returns HTTP 503
with `error.type=lifecycle_unavailable`, and never wakes a peer. `/healthz`
remains HTTP 200 for a finalized control plane, reports `ok=true`, `ready=false`,
and `lifecycle_state=unknown` so the host can route supported repair without a
restart loop. Health fields are nonblocking observational snapshots so a long
wake or startup-lease wait cannot fail the control-plane health probe. Positive
sleeping evidence makes repeated sleep idempotent; the
supported host lifecycle path owns stopped-engine repair. Thermal recovery does
not remember or preemptively resume the prior model; a normal client retry wakes
its requested model after safe all-sleeping release.

`SLEEPER_OWNER_VALIDATION_TIMEOUT_SECONDS` is a finite positive wall-clock
budget for the complete owner, peer, and readiness validation sequence. It
defaults to two seconds and is separate from the longer wake/load timeout.

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
  uncertainty. Fresh graceful-hold, urgent-hold, hard-cutoff, or danger-derived
  recovery state rejects acquisition at the Sleeper Proxy boundary before Proxy
  request-body handling with HTTP 503, integer `Retry-After`, and stable
  `error.code=thermal_protection_active`. Optional models also reject missing,
  stale, or malformed projections. The fast thermal projection owns thermal
  denial for required models, so a slower resource snapshot cannot extend a
  cleared thermal warning. Existing leases may drain; action-scoped thermal
  sleep quiesces later acquisitions. LiteLLM keeps retries disabled and
  preserves this response; no additional LAN-edge thermal proxy is required.
  The consumer does not own deployment thresholds or positive-danger
  provenance. The denial body uses a constant public message with
  `error.type=service_unavailable`, `error.code=thermal_protection_active`, a
  bounded integer `Retry-After`, and only validated phase/action metadata.
  `/healthz` remains HTTP 200 with `ok=true` for a finalized control plane
  during an intentional hold, while `ready=false` and
  `inference_available=false` distinguish the fenced request plane. It also
  reports startup, lifecycle readiness, selected/starting model, and sanitized
  thermal phase/action fields without waiting for a lifecycle lease.
  The schema-v1 admission latch, root watchdog, and memory-pressure caller remain
  unchanged, including bodyless loopback `POST /sleep`. The separate dormant
  action-authority contract is strict schema v2. `/thermal/actions/hold`,
  `/thermal/actions/sleeping-subset-proof`, and `/thermal/actions/release` remain
  absent unless both
  `SLEEPER_THERMAL_ACTION_CONTROL_ENABLED=1` and
  `SLEEPER_THERMAL_ACTION_STATUS_PATH` are configured. Every request and response
  binds record revision, incident/action/generation/predecessor lineage,
  transition kind, phase revision, containment level, kind-specific deadlines,
  recovery authority, and the exact configured engine set; schema v1 action
  requests are rejected rather than downgraded. `hold` accepts graceful, urgent,
  or held work only inside the immutable containment window, serializes on the
  shared lease, rejects non-level-2 sleep, and returns exact positive all-engine
  proof. Held authority may also expose the dedicated
  `/thermal/actions/sleeping-subset-proof` operation. Root binds its exact sorted
  retained-sleeping subset beside the full configured engine set. Proxy probes
  only that subset under the action lock, lifecycle condition, and shared startup
  lease, repeatedly reauthorizes the root projection, and returns conservative
  proof start/completion times within one two-second budget. It never contacts a
  stopped peer or sleeps, wakes, selects, repairs, persists, or clears ownership.
  `release` accepts ordinary release or `cutoff_recovery_authorized` authority and
  remains the distinct all-engine proof-only final gate. Building this code does
  not activate any action route. Deployment must enable them only with the matching root migration and a
  tested local-only binding; action IDs are correlation values, not credentials.
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
| `POST /thermal/actions/hold` | Exact schema-v2 generation/revision/deadline/engine-set authority; disabled by default | Sleeps only as root-authorized and returns exact identity-bound all-engine positive proof |
| `POST /thermal/actions/sleeping-subset-proof` | Exact held authority, full engine set, and root-derived retained-sleeping subset; disabled by default | Within two seconds, verifies only the exact subset without lifecycle or root-state mutation and never contacts stopped peers |
| `POST /thermal/actions/release` | Exact schema-v2 ordinary or cutoff-recovery proof authority; disabled by default | Verifies all engines sleeping without sleep, wake, selection, repair, or root-state mutation |

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
