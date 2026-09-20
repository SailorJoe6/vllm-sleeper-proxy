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
| SLP-LIFE-013 | Action-scoped thermal control is disabled by default and absent unless both the explicit enable flag and a root-owned action projection path are configured. Its separate authority schema v2 rejects v1 rather than downgrading and binds every request, reauthorization, and success response to the exact record revision, incident/action/generation/predecessor lineage, transition kind, phase revision, containment level, applicable nullable deadline tuple, recovery authorization/repair deadline, and sorted configured engine set. `hold` accepts only `graceful_hold`, `urgent_hold`, or `held` authority with immutable containment deadlines bounded to at most 300 seconds, serializes under the shared lifecycle lease, rechecks full authority throughout lifecycle work, uses level-2 sleep only, and succeeds only with exact positive `/is_sleeping=true` proof for every engine. The distinct held-only `sleeping_subset_proof` operation binds the exact sorted root-derived retained-sleeping subset beside the full configured engine set, has one two-second lock/probe/response budget with conservative start/completion times, probes only those peers under action-lock then condition/quiesce then shared-startup-lease order with repeated reauthorization, and never sleeps, wakes, selects, repairs, persists, clears ownership, or contacts a stopped peer. `release` is the separate proof-only final gate for ordinary release or `cutoff_recovery_authorized`: it verifies every engine sleeping and never sleeps, wakes, selects, repairs, exposes readiness, clears root state, or persists prior-model identity. Stale predecessor/revision work, schema mismatch, corruption, expiry, transport loss, unknown state, or phase/lineage change fails closed without treating HTTP success as proof. Private host/container proof, repair targets, and cutoff dispatch metadata are never projected. Generic bodyless `/sleep` and the independent admission schema v1 remain unchanged. The action ID is correlation, not authentication; deployment may enable these routes only on the tested local-only path and LiteLLM must not route them. | strict v2 action-authority golden vectors; lineage/revision/deadline/engine-set mutation tests; manager all-engine deadline/replay/queued-wake/release tests; disabled/enabled server route tests |
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

The schema-v1 admission latch, root watchdog, and non-thermal memory-pressure
caller retain bodyless loopback `POST /sleep`. The separate dormant action
projection/request/response contract is strict schema v2 and rejects v1.
`/thermal/actions/hold`, `/thermal/actions/sleeping-subset-proof`, and
`/thermal/actions/release` remain absent unless both feature settings name the v2
authority path. Building or deploying this code
does not activate them. They may be enabled only in the later atomic root/Proxy
migration with a tested local-only path; an action ID is not a bearer secret.

## Traceability

The proxy unit tests cover model discovery, alias rewriting, request draining,
startup serialization, and admission failure. Deployment repositories must add
lineup tests proving every pairwise switch, including embedding/Vision/Qwen-like
combinations, and must record the exact image and model identity used.
