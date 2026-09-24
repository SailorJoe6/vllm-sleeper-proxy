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
| SLP-LIFE-005 | Proxy-owned engine lifecycle state must be explicit: active, starting, sleeping, or unknown. Unknown, stale, mismatched, or unavailable engine-control state fails closed for wake/switch so the proxy never wakes a second engine. This is distinct from external monitor-admission uncertainty under SLP-LIFE-009. A required engine that is absent is a startup failure, not a dormant or sleeping state. Explicit state must be refreshed or invalidated when the owned engine disappears or cannot be positively revalidated; cached in-process readiness is not durable evidence across engine exit or recreation. | `/healthz`, lifecycle-state and guard integration tests |
| SLP-LIFE-006 | A client disconnect or lifecycle timeout must release pending switch ownership and leave later requests able to receive a bounded response. | request-drain regression test |
| SLP-LIFE-007 | If the requested model is already the sole awake and ready model, route the request directly without a wake transition. If it is not active, quiesce and drain the current model, request level-2 sleep, verify it is asleep, perform admission/readiness checks, then wake the requested model and route only after it is fully ready. | `ModelManager` request path and pairwise switch tests |
| SLP-LIFE-008 | A configured required lineup is not ready until every required engine is present, registered, and reconciled to sleeping or the single active ready state. Missing required engines fail startup/readiness; the proxy must not silently omit them or classify them as dormant. | deployment startup/readiness gate and lineup tests |
| SLP-LIFE-009 | Every configured model must explicitly declare whether it is required. For a required model, missing, stale, malformed, warning, or telemetry-failure admission data is degraded observability and must not deny inference; a fresh internally consistent graceful hold, urgent hold, hard cutoff, or danger-derived recovery decision rejects acquisition at the Sleeper Proxy boundary before Proxy request-body processing or wake with HTTP 503, integer `Retry-After`, and stable `error.code=thermal_protection_active`. Optional models remain fail-closed on admission uncertainty. The independent fast thermal projection and containment latch are authoritative for thermal denial; a slower resource snapshot's duplicate `thermal_admission_denied` reason must not extend a cleared warning into a required-model outage. An already leased request may drain; after fencing, the root watchdog may use the retained bodyless level-2 `POST /sleep` only for a positively awake engine. Deployment routing keeps LiteLLM retries disabled and preserves the Proxy 503 without requiring another LAN-edge thermal proxy. The deployment watchdog owns thresholds, positive-danger provenance, and emergency cutoff ordering. | required/optional thermal and resource admission tests, manager drain, server response tests, LiteLLM propagation test |
| SLP-LIFE-013 | Generic bodyless `POST /sleep` may accept one optional unsigned `X-Sleeper-Drain-Timeout-Ms` value from 1 through 300000. Alone, it preserves the legacy whole-operation bound. A paired unsigned `X-Sleeper-Containment-Timeout-Ms` value in the same range may separate the earlier drain deadline from a later containment deadline. A mutually exclusive absolute mode requires both `X-Sleeper-Drain-Deadline-Epoch-Ms` and `X-Sleeper-Containment-Deadline-Epoch-Ms`. Absolute epoch-ms bounds must be mapped conservatively to monotonic deadlines at manager entry so handler delay never extends them. Every caller-supplied relative or absolute deadline is capped by the configured whole-operation timeout. Concurrent-sleep serialization, condition-lock acquisition, quiesce, and in-flight drain use the effective drain deadline; shared-startup-lease acquisition, positive state validation, level-2 sleep, and convergence use the effective containment deadline. Incomplete pairs, mixed modes, containment earlier than drain, booleans/nonintegers, malformed, duplicate, unsupported, or inconsistent values fail before lifecycle work. No caller may issue a delayed sleep mutation after its applicable deadline or clear another caller's quiesce state. A failed guarded acquisition may publish a short-lived ordinary lifecycle `demand` transition for the requested configured model; this is not a thermal target, action, claim, admission bypass, or Docker authority. | relative/absolute paired-deadline manager/server timeout tests; demand transition and root reconciler tests |
| SLP-LIFE-010 | Proxy-only recovery must adopt an existing lineup without changing engine state when exactly zero or one engine is awake. Unknown state or multiple awake engines must defer recovery without issuing sleep, wake, or stop actions. | `adopt_startup_state`, `/startup/adopt`, bootstrap server tests |
| SLP-LIFE-011 | Runtime lifecycle ownership must converge when an owned engine exits, is recreated, or loses its control endpoint. A stale in-process `active_model` or ready fast path must never permanently block positive containment, return accidental 502 for a known unavailable owner, or prevent required-lineup restoration. Transport or DNS failure alone is not proof that an engine is asleep or stopped. The proxy must invalidate stale readiness, return intentional 503 while state is unresolved, and preserve the at-most-one-awake invariant under the shared startup lease. Ownership may clear when positive vLLM sleep evidence or the supported host lifecycle path establishes stopped/all-sleeping state; unknown or conflicting evidence defers without waking another engine. Thermal recovery does not persist or preemptively resume a prior model: after sleeping-or-stopped release, the next client retry uses the normal guarded wake path. | `ModelManager` bounded owner/peer revalidation and positive sleep idempotence; server same-model/forward-open 503, health-state, exact-once lease, multiple-awake, peer-unknown, and recovery tests |
| SLP-LIFE-012 | A finalized Proxy that is enforcing an affirmative thermal fence remains a healthy control plane. `/healthz` must separate control health, startup finalization, lifecycle readiness, and inference availability; it must report only sanitized thermal phase/action metadata and must not wait for the lifecycle condition or startup lease. Its nested `urgent_drain_snapshot` uses a dedicated short-held request-admission lock, independent of slow lifecycle condition ownership, to expose the exact sanitized action ID, fence state, control health, and integer in-flight count. Final post-activation admission recheck plus in-flight increment and request release use that same linearization authority: either a request is counted before proof, or a same-action fenced zero proof wins and later admission fails. Snapshot `null` is permitted only while that short critical section is held, never merely because engine validation, wake, sleep, convergence, or other lifecycle work is slow. The compatible top-level count is not drain proof. Intentional thermal denial returns a constant OpenAI-compatible HTTP 503 with bounded integer `Retry-After`, `error.type=service_unavailable`, and `error.code=thermal_protection_active` before body read, parsing, model selection, lifecycle wait, or wake. Restart must read the active schema-v1 latch before accepting inference. | atomic final-admission/snapshot order tests; release-under-lifecycle-contention tests; no-owner and active-owner slow-control tests; server early-denial, nonblocking health, restart re-fence, and body-shape tests |
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

The schema-v1 root watchdog and the non-thermal memory-pressure caller retain
compatibility with bodyless loopback `POST /sleep`. A single optional drain
header preserves the legacy whole-operation bound. An urgent caller may pair it
with the later containment timeout, or use the mutually exclusive paired absolute
epoch-ms deadlines, so handler delay cannot reset the immutable drain and
sleep/convergence boundaries. Every caller bound is capped by the configured
whole-operation timeout. The compact production design exposes no action-scoped
thermal lifecycle endpoint. Root owns
the minimal durable fence and may request generic level-2 sleep only for a
positively awake engine; the Proxy continues to consume the sanitized admission
view. A failed acquisition of a stopped engine can publish ordinary demand
through the existing transition file so root may restore the guarded sleeping
lineup after, never before, a client request.

## Traceability

The proxy unit tests cover model discovery, alias rewriting, request draining,
startup serialization, and admission failure. Deployment repositories must add
lineup tests proving every pairwise switch, including embedding/Vision/Qwen-like
combinations, and must record the exact image and model identity used.
