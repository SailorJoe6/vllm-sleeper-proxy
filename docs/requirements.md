# vLLM Sleeper Proxy Requirements

This document is the normative requirement source for lifecycle ownership in
`vllm-sleeper-proxy`. Deployment repositories may add stricter requirements,
but must not weaken these invariants.

## Lifecycle requirements

| ID | Requirement | Evidence |
|---|---|---|
| SLP-LIFE-001 | One Sleeper Proxy instance owns every configured vLLM engine in a lineup. A deployment must not split engines across independent proxies when those engines share GPU/unified memory. | `ModelManager`, shared `SLEEPER_MODELS`, integration test |
| SLP-LIFE-002 | Before activating a requested model, the owner must quiesce requests, drain the active model, request level-2 sleep, and verify `/is_sleeping=true`. | `sleep_active_model`, switch tests |
| SLP-LIFE-003 | The owner must not wake a requested model until the previous active model is verified asleep. | `_ensure_awake_startup_locked`, switch tests |
| SLP-LIFE-004 | At most one model may be awake or starting. A shared startup lease must cover reconciliation, sleep, wake, readiness, and admission checks. | startup lease tests |
| SLP-LIFE-005 | Proxy-owned engine lifecycle state must be explicit: active, starting, sleeping, or unknown. Unknown, stale, mismatched, or unavailable engine-control state fails closed for wake/switch so the proxy never wakes a second engine. This is distinct from external monitor-admission uncertainty under SLP-LIFE-009. A required engine that is absent is a startup failure, not a dormant or sleeping state. | `/healthz`, lifecycle-state and guard integration tests |
| SLP-LIFE-006 | A client disconnect or lifecycle timeout must release pending switch ownership and leave later requests able to receive a bounded response. | request-drain regression test |
| SLP-LIFE-007 | If the requested model is already the sole awake and ready model, route the request directly without a wake transition. If it is not active, quiesce and drain the current model, request level-2 sleep, verify it is asleep, perform admission/readiness checks, then wake the requested model and route only after it is fully ready. | `ModelManager` request path and pairwise switch tests |
| SLP-LIFE-008 | A configured required lineup is not ready until every required engine is present, registered, and reconciled to sleeping or the single active ready state. Missing required engines fail startup/readiness; the proxy must not silently omit them or classify them as dormant. | deployment startup/readiness gate and lineup tests |
| SLP-LIFE-009 | Every configured model must explicitly declare whether it is required. For a required model, missing, stale, malformed, warning, or telemetry-failure admission data is degraded observability and must not deny inference; a fresh internally consistent sleep, cutoff, or danger-derived recovery decision still rejects acquisition before request-body processing or wake with HTTP 503, `Retry-After`, and `error.type=thermal_cooldown`. Optional models remain fail-closed on admission uncertainty. An already leased request may drain; `POST /sleep` quiesces later acquisitions before sleeping. The deployment watchdog owns thresholds, positive-danger provenance, and emergency cutoff ordering. | required/optional thermal and resource admission tests, manager drain, server response tests |
| SLP-LIFE-010 | Proxy-only recovery must adopt an existing lineup without changing engine state when exactly zero or one engine is awake. Unknown state or multiple awake engines must defer recovery without issuing sleep, wake, or stop actions. | `adopt_startup_state`, `/startup/adopt`, bootstrap server tests |
| SLP-API-001 | Logical aliases are rewritten only by the owning proxy; clients must not have a direct wake path to an engine. | request rewrite tests |

## Deployment rule

Every deployment must explicitly declare its required engine lineup. The shared
Sleeper Proxy owns every engine in that lineup, and startup/readiness fails if a
required engine is absent or cannot be reconciled. Optional-model lifecycle control is intentionally not defined here. Any
future model that is not part of the required lineup needs a separate design
sprint covering container presence, startup, admission, sleep/reclaim, readiness,
and recovery before it may be added to a deployment.

A second proxy is not an acceptable substitute for shared lifecycle ownership.
If process isolation is required, it must be implemented behind one
coordinating owner or an explicitly tested shared controller that performs the
same sleep-before-wake contract. Container startup and vLLM sleep state are
separate layers: a container may initialize awake, but lineup readiness must
reconcile every required engine before accepting requests.

## Traceability

The proxy unit tests cover model discovery, alias rewriting, request draining,
startup serialization, and admission failure. Deployment repositories must add
lineup tests proving every pairwise switch, including embedding/Vision/Qwen-like
combinations, and must record the exact image and model identity used.
