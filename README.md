# vllm-sleeper-proxy

`vllm-sleeper-proxy` is a lightweight, OpenAI-compatible proxy that enables **on-demand model switching and discovery** for vLLM on memory-constrained, single-GPU systems (such as NVIDIA DGX Spark).

The proxy sits in front of one or more vLLM engines and provides a **stable HTTP surface** that agent frameworks and clients can treat as a persistent inference endpoint—even though the underlying models may be **asleep, unloaded, or dynamically activated**.

At any given time, the proxy enforces an **at-most-one awake model invariant**, using **vLLM Sleep Mode (Level 2)** to fully release model weights and KV cache before another model is activated.

This allows individual models to consume **nearly all available GPU / unified memory** (for example, very large context windows) while still supporting fast, request-driven switching between models such as *coding*, *reasoning*, or *planning* LLMs—without full cold restarts.

---

## Key features

* **Single OpenAI-compatible inference endpoint**
* **Strict sleep-before-wake orchestration** (vLLM Sleep Mode Level 2)
* **Request-driven model activation** based on the `model` field
* **At-most-one awake model invariant** (prevents OOM on unified-memory systems)
* **Fast switching** between frequently used models
* **Model discovery APIs** compatible with common local runtimes
* Designed for **home labs and single-node setups**, not Kubernetes

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

