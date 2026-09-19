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
| SLP-LIFE-009 | Every configured model must explicitly declare whether it is required. For a required model, missing, stale, malformed, warning, or telemetry-failure admission data is degraded observability and must not deny inference; a fresh internally consistent graceful hold, urgent hold, hard cutoff, or danger-derived recovery decision rejects acquisition at the Sleeper Proxy boundary before Proxy request-body processing or wake with HTTP 503, integer `Retry-After`, and stable `error.code=thermal_protection_active`. Optional models remain fail-closed on admission uncertainty. The independent fast thermal projection and containment latch are authoritative for thermal denial; a slower resource snapshot's duplicate `thermal_admission_denied` reason must not extend a cleared warning into a required-model outage. An already leased request may drain; action-scoped thermal sleep quiesces later acquisitions before sleeping. Deployment routing keeps LiteLLM retries disabled and preserves the Proxy 503 without requiring another LAN-edge thermal proxy. The deployment watchdog owns thresholds, positive-danger provenance, and emergency cutoff ordering. | required/optional thermal and resource admission tests, manager drain, server response tests, LiteLLM propagation test |
| SLP-LIFE-010 | Proxy-only recovery must adopt an existing lineup without changing engine state when exactly zero or one engine is awake. Unknown state or multiple awake engines must defer recovery without issuing sleep, wake, or stop actions. | `adopt_startup_state`, `/startup/adopt`, bootstrap server tests |
| SLP-LIFE-011 | Runtime lifecycle ownership must converge when an owned engine exits, is recreated, or loses its control endpoint. A stale in-process `active_model` or ready fast path must never permanently block positive containment, return accidental 502 for a known unavailable owner, or prevent required-lineup restoration. Transport or DNS failure alone is not proof that an engine is asleep or stopped. The proxy must invalidate stale readiness, return intentional 503 while state is unresolved, and preserve the at-most-one-awake invariant under the shared startup lease. Ownership may clear when positive vLLM sleep evidence or the supported host lifecycle path establishes stopped/all-sleeping state; unknown or conflicting evidence defers without waking another engine. Thermal recovery does not persist or preemptively resume a prior model: after all-sleeping release, the next client retry uses the normal guarded wake path. | `ModelManager` bounded owner/peer revalidation and positive sleep idempotence; server same-model/forward-open 503, health-state, exact-once lease, multiple-awake, peer-unknown, and recovery tests |
| SLP-LIFE-012 | A finalized Proxy that is enforcing an affirmative thermal fence remains a healthy control plane. `/healthz` must separate control health, startup finalization, lifecycle readiness, and inference availability; it must report only sanitized thermal phase/action metadata and must not wait for the lifecycle condition or startup lease. Intentional thermal denial returns a constant OpenAI-compatible HTTP 503 with bounded integer `Retry-After`, `error.type=service_unavailable`, and `error.code=thermal_protection_active` before body read, parsing, model selection, lifecycle wait, or wake. Restart must read the active schema-v1 latch before accepting inference. | admission snapshots; server early-denial, health, restart re-fence, and body-shape tests |
| SLP-LIFE-013 | Action-scoped thermal control is disabled by default and must be absent unless both the explicit enable flag and a root-owned action projection path are configured. `hold` accepts only an exact active root action ID and `graceful_hold`, `urgent_hold`, or `held` phase whose creation time and immutable finite deadlines bound the total active action window to at most 300 seconds. Every reauthorization binds the same creation time and deadlines. It serializes under the shared lifecycle lease, rechecks the fast thermal fence after lifecycle waits and immediately before any wake, drains only within the original deadline, requires unique configured engine identities, uses level-2 sleep only when the root-authorized phase permits it, and returns success only with exact positive `/is_sleeping=true` proof for every engine. Replaying the same action re-probes state and does not duplicate sleep for already-sleeping engines. `release` is proof-only and uses its own bounded local proof timeout after the hold deadlines: it verifies every engine sleeping, never sleeps, wakes, selects a model, clears root state, or persists prior-model identity. Mismatch, corruption, active-hold deadline expiry, transport loss, unknown state, or a phase flip fails closed without treating HTTP success as containment. Generic bodyless `/sleep` remains unchanged. The action ID is correlation, not authentication; deployment may enable these routes only on the tested local-only control path and LiteLLM must not route them. | strict action-authority tests; manager all-engine deadline/replay/queued-wake/release tests; disabled/enabled server route tests |
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
compatibility with bodyless loopback `POST /sleep`. The dormant
`/thermal/actions/hold` and `/thermal/actions/release` routes are absent unless
`SLEEPER_THERMAL_ACTION_CONTROL_ENABLED=1` and
`SLEEPER_THERMAL_ACTION_STATUS_PATH` names the strict root projection. Merely
building or deploying this code does not activate them. A deployment may enable
them only in the same migration that supplies the root durable record/caller and
enforces a tested local-only control path; an action ID is not a bearer secret.

## Traceability

The proxy unit tests cover model discovery, alias rewriting, request draining,
startup serialization, and admission failure. Deployment repositories must add
lineup tests proving every pairwise switch, including embedding/Vision/Qwen-like
combinations, and must record the exact image and model identity used.
