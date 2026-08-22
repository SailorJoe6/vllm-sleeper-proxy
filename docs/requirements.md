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
| SLP-LIFE-005 | Model state must be explicit: active, starting, sleeping, or unknown. Unknown, stale, mismatched, or unavailable state fails closed. | `/healthz`, guard integration tests |
| SLP-LIFE-006 | A client disconnect or lifecycle timeout must release pending switch ownership and leave later requests able to receive a bounded response. | request-drain regression test |
| SLP-API-001 | Logical aliases are rewritten only by the owning proxy; clients must not have a direct wake path to an engine. | request rewrite tests |

## Deployment rule

A deployment that needs an optional model boundary must add the model to the
existing shared proxy lineup. A second proxy is not an acceptable substitute
for shared lifecycle ownership. If process isolation is required, it must be
implemented behind one coordinating owner or an explicitly tested shared
controller that performs the same sleep-before-wake contract.

## Traceability

The proxy unit tests cover model discovery, alias rewriting, request draining,
startup serialization, and admission failure. Deployment repositories must add
lineup tests proving every pairwise switch, including embedding/Vision/Qwen-like
combinations, and must record the exact image and model identity used.
