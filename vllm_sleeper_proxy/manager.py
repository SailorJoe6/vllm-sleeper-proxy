from __future__ import annotations

import fcntl
import json
import os
import threading
import time
from contextlib import contextmanager
from typing import Callable, Iterable, Iterator
from urllib.parse import urlencode

from .client import HttpClient, sleep_until
from .config import ModelConfig


class UnknownModelError(ValueError):
    pass


class WakeError(RuntimeError):
    pass


@contextmanager
def startup_lease(path: str) -> Iterator[None]:
    """Hold an OS lease across model startup and readiness validation.

    The lock is released automatically if the proxy process crashes. Deployments
    spanning containers should mount the same host path into every proxy and set
    ``SLEEPER_STARTUP_LEASE_PATH`` accordingly.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class ModelLease:
    """Own one active-model request until its response reaches a terminal path."""

    def __init__(self, manager: ModelManager, target: ModelConfig) -> None:
        self.manager = manager
        self.target = target
        self._released = False

    def __enter__(self) -> ModelConfig:
        return self.target

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()

    def release(self) -> None:
        if not self._released:
            self.manager.release(self.target)
            self._released = True


class ModelManager:
    """Serializes vLLM sleep/wake transitions.

    The manager is intentionally conservative: all lifecycle changes pass
    through one mutex, so concurrent cold requests collapse into one wake path
    rather than stampeding the vLLM dev endpoints.
    """

    def __init__(
        self,
        models: Iterable[ModelConfig],
        http: HttpClient,
        *,
        sleep_level: int = 2,
        request_timeout_s: float = 30.0,
        wake_timeout_s: float = 300.0,
        drain_timeout_s: float = 300.0,
        poll_interval_s: float = 1.0,
        always_wake: bool = True,
        wake_strategy: str = "level2",
        admission_check: Callable[[ModelConfig], None] | None = None,
        startup_lease_path: str | None = None,
        transition_state_path: str | None = None,
        bootstrap_mode: bool = False,
    ) -> None:
        self.models = list(models)
        if not self.models:
            raise ValueError("at least one model must be configured")
        self.http = http
        self.sleep_level = sleep_level
        self.request_timeout_s = request_timeout_s
        self.wake_timeout_s = wake_timeout_s
        self.drain_timeout_s = drain_timeout_s
        self.poll_interval_s = poll_interval_s
        self.always_wake = always_wake
        self.wake_strategy = wake_strategy
        self.admission_check = admission_check
        self.startup_lease_path = startup_lease_path or os.environ.get(
            "SLEEPER_STARTUP_LEASE_PATH", "/tmp/vllm-sleeper-proxy-startup.lock"
        )
        self.transition_state_path = transition_state_path or os.environ.get("SLEEPER_TRANSITION_STATE_PATH")
        self.bootstrap_mode = bootstrap_mode
        self._startup_finalized = not bootstrap_mode
        self._condition = threading.Condition()
        self._active_model_name: str | None = None
        self._starting_model_name: str | None = None
        self._active_model_ready = False
        self._inflight_requests = 0
        self._pending_switch_name: str | None = None
        self._quiescing = False

    @property
    def startup_finalized(self) -> bool:
        with self._condition:
            return self._startup_finalized

    def startup_sleep_model(self, requested: str) -> str:
        """Sleep one engine during serialized bootstrap, without waking any engine."""
        target = self.find_model(requested)
        with startup_lease(self.startup_lease_path):
            with self._condition:
                if self._inflight_requests or self._pending_switch_name is not None:
                    raise WakeError("cannot bootstrap sleep while requests are active")
                self._starting_model_name = target.name
                self._publish_transition("bootstrap_sleep", target.name)
            try:
                sleeping = self._is_sleeping(target)
                if sleeping is None:
                    raise WakeError(f"cannot verify bootstrap sleep state for {target.name}")
                if not sleeping:
                    self._sleep(target)
                self._wait_until_sleeping(target)
                with self._condition:
                    if self._active_model_name == target.name:
                        self._active_model_name = None
                        self._active_model_ready = False
                return target.name
            finally:
                with self._condition:
                    self._starting_model_name = None
                    self._publish_transition("idle")
                    self._condition.notify_all()

    def startup_state(self, requested: str | None = None) -> dict[str, object]:
        # The unqualified bootstrap probe is a control-plane liveness check.
        # Before the first engine starts, probing every upstream would turn a
        # missing engine DNS name into a 503/empty reply and deadlock startup.
        # Per-model callers still get a fail-closed sleep-state probe.
        models = [self.find_model(requested)] if requested else self.models
        states: list[dict[str, object]] = []
        for model in models:
            sleeping = None if requested is None else self._is_sleeping(model)
            states.append({"model": model.name, "is_sleeping": sleeping})
        return {"bootstrap": self.bootstrap_mode, "finalized": self.startup_finalized, "models": states}

    def finalize_startup(self) -> None:
        """Reconcile the complete required lineup and unlock inference."""
        with startup_lease(self.startup_lease_path):
            with self._condition:
                self._starting_model_name = "__system_startup__"
                self._publish_transition("finalizing", "__system_startup__")
            try:
                for model in self.models:
                    sleeping = self._is_sleeping(model)
                    if sleeping is None:
                        raise WakeError(f"cannot verify startup sleep state for {model.name}")
                    if not sleeping:
                        self._sleep(model)
                    self._wait_until_sleeping(model)
                with self._condition:
                    self._active_model_name = None
                    self._active_model_ready = False
                    self._startup_finalized = True
            finally:
                with self._condition:
                    self._starting_model_name = None
                    self._publish_transition("idle")
                    self._condition.notify_all()

    @property
    def active_model_name(self) -> str | None:
        return self._active_model_name

    @property
    def starting_model_name(self) -> str | None:
        """Model that owns the startup lease until readiness is verified."""
        with self._condition:
            return self._starting_model_name

    @property
    def inflight_requests(self) -> int:
        with self._condition:
            return self._inflight_requests

    def _publish_transition(self, phase: str, model_name: str | None = None) -> None:
        """Atomically publish lifecycle ownership for the host guard.

        The file is advisory only; the guard still fails closed for stale,
        malformed, or mismatched records. Atomic replacement prevents readers
        from observing a partially written transition.
        """
        if not self.transition_state_path:
            return
        path = self.transition_state_path
        parent = os.path.dirname(path) or "."
        os.makedirs(parent, exist_ok=True)
        payload = {
            "phase": phase,
            "model": model_name,
            "updated_at_epoch": time.time(),
            "pid": os.getpid(),
        }
        temporary = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def reconcile_startup_state(self) -> None:
        """Put every configured engine to sleep before the proxy serves traffic."""

        with startup_lease(self.startup_lease_path):
            with self._condition:
                self._starting_model_name = "__system_startup__"
                self._publish_transition("starting", "__system_startup__")
                try:
                    if self._inflight_requests or self._pending_switch_name is not None:
                        raise WakeError("cannot reconcile startup state while requests are active")

                    for model in self.models:
                        sleeping = self._is_sleeping(model)
                        if sleeping is None:
                            raise WakeError(
                                f"cannot verify startup sleep state for {model.name}: "
                                "/is_sleeping did not return a boolean state"
                            )
                        if not sleeping:
                            self._sleep(model)
                            self._wait_until_sleeping(model)

                    self._active_model_name = None
                    self._active_model_ready = False
                finally:
                    self._starting_model_name = None
                    self._publish_transition("idle")
                    self._condition.notify_all()

    def find_model(self, requested: str) -> ModelConfig:
        for model in self.models:
            if model.matches(requested):
                return model
        raise UnknownModelError(f"unknown model: {requested}")

    def list_openai_models(self) -> dict[str, object]:
        return {
            "object": "list",
            "data": [
                {
                    "id": model.name,
                    "object": "model",
                    "created": 0,
                    "owned_by": model.owned_by,
                }
                for model in self.models
            ],
        }

    def list_ollama_tags(self) -> dict[str, object]:
        return {
            "models": [
                {
                    "name": model.name,
                    "model": model.name,
                    "modified_at": "1970-01-01T00:00:00Z",
                    "size": 0,
                    "digest": "",
                    "details": {"family": "vllm", "families": ["vllm"]},
                }
                for model in self.models
            ]
        }

    def ensure_awake(self, requested: str) -> ModelConfig:
        target = self.find_model(requested)
        with self._condition:
            self._wait_for_switch_safety(target)
            return self._activate_locked(target)

    def acquire(self, requested: str) -> ModelLease:
        """Wake a model and hold it awake for one buffered or streaming request."""

        target = self.find_model(requested)
        with self._condition:
            self._wait_for_switch_safety(target)
            if self.admission_check is not None:
                self.admission_check(target)
            self._activate_locked(target)
            self._inflight_requests += 1
        return ModelLease(self, target)

    def sleep_active_model(self) -> str | None:
        """Quiesce new requests, drain ownership, and sleep the active model.

        Local request state is acquired before the host-wide lease. This keeps
        lock ordering identical to startup and avoids a condition/lease
        deadlock when another proxy instance is starting a model.
        """

        with self._condition:
            deadline = time.monotonic() + self.drain_timeout_s
            self._quiescing = True
            try:
                while self._inflight_requests > 0:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise WakeError("timed out draining requests before resource backoff")
                    self._condition.wait(timeout=remaining)
                if self._active_model_name is None:
                    return None
                current = self.find_model(self._active_model_name)
            except Exception:
                self._quiescing = False
                self._condition.notify_all()
                raise

        try:
            self._publish_transition("sleeping", current.name)
            with startup_lease(self.startup_lease_path):
                self._sleep(current)
                self._wait_until_sleeping(current)
        except Exception:
            with self._condition:
                self._quiescing = False
                self._condition.notify_all()
            raise

        with self._condition:
            self._active_model_name = None
            self._active_model_ready = False
            self._publish_transition("idle")
            self._quiescing = False
            self._condition.notify_all()
        return current.name

    def release(self, target: ModelConfig) -> None:
        with self._condition:
            if self._inflight_requests <= 0:
                raise RuntimeError(f"no in-flight request to release for {target.name}")
            self._inflight_requests -= 1
            self._condition.notify_all()

    def _wait_for_switch_safety(self, target: ModelConfig) -> None:
        deadline = time.monotonic() + self.drain_timeout_s
        while True:
            if self._quiescing:
                self._wait_for_drain(deadline, target)
                continue
            if (
                self._pending_switch_name is not None
                and self._pending_switch_name != target.name
            ):
                self._wait_for_drain(deadline, target)
                continue
            if (
                self._active_model_name is not None
                and self._active_model_name != target.name
                and self._inflight_requests > 0
            ):
                self._pending_switch_name = target.name
                self._wait_for_drain(deadline, target)
                continue
            return

    def _wait_for_drain(self, deadline: float, target: ModelConfig) -> None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            if self._pending_switch_name == target.name:
                self._pending_switch_name = None
                self._condition.notify_all()
            raise WakeError(f"timed out draining requests before switching to {target.name}")
        self._condition.wait(timeout=remaining)

    def _activate_locked(self, target: ModelConfig) -> ModelConfig:
        try:
            with startup_lease(self.startup_lease_path):
                return self._ensure_awake_locked(target)
        except WakeError:
            raise
        except (ConnectionError, OSError, TimeoutError, TypeError, ValueError) as exc:
            raise WakeError(f"lifecycle check failed for {target.name}: {exc}") from exc
        finally:
            if self._pending_switch_name == target.name:
                self._pending_switch_name = None
                self._condition.notify_all()

    def _ensure_awake_locked(self, target: ModelConfig) -> ModelConfig:
        needs_startup = self._active_model_name != target.name or not self._active_model_ready
        if not needs_startup:
            return target
        self._starting_model_name = target.name
        self._publish_transition("starting", target.name)
        try:
            return self._ensure_awake_startup_locked(target)
        except Exception:
            # A failed startup must not leave a partially resident model marked
            # active. Best-effort sleep keeps recovery conservative; preserve
            # the original failure if cleanup itself fails.
            try:
                if self._active_model_name == target.name and self._is_sleeping(target) is False:
                    self._sleep(target)
                    self._wait_until_sleeping(target)
            finally:
                self._active_model_name = None
                self._active_model_ready = False
            raise
        finally:
            self._starting_model_name = None
            self._publish_transition("idle")
            self._condition.notify_all()

    def _ensure_awake_startup_locked(self, target: ModelConfig) -> ModelConfig:
        if self._active_model_name and self._active_model_name != target.name:
            current = self.find_model(self._active_model_name)
            self._sleep(current)
            # Do not sample admission while the previous engine is still
            # asynchronously reclaiming memory. Switching admission is only
            # valid after the proxy verifies the engine's sleep state.
            self._wait_until_sleeping(current)
            self._active_model_name = None
            self._active_model_ready = False

        # The lease is host-wide, so do not trust another proxy instance's
        # in-process active-model bookkeeping. Reconcile every other configured
        # engine before allocating the target.
        for model in self.models:
            if model.name == target.name:
                continue
            sleeping = self._is_sleeping(model)
            if sleeping is None:
                raise WakeError(
                    f"cannot verify that {model.name} is asleep before starting "
                    f"{target.name}"
                )
            if not sleeping:
                self._sleep(model)
                self._wait_until_sleeping(model)

        should_wake = self._active_model_name != target.name
        if self._active_model_name is None:
            sleeping = self._is_sleeping(target)
            if sleeping is False:
                should_wake = False
                self._active_model_name = target.name
                self._active_model_ready = False
        elif self.always_wake and self._active_model_name == target.name:
            sleeping = self._is_sleeping(target)
            should_wake = sleeping is not False

        if should_wake:
            # Recheck after any previous engine has been slept. Sleeping is an
            # asynchronous memory transition, so an admission decision made
            # before the drain can be stale by the time this engine allocates.
            if self.admission_check is not None:
                self.admission_check(target)
            # Conservatively remember the target before wake begins. A partial
            # wake failure can leave weights resident; the next model switch
            # must sleep this engine even when readiness never completed.
            self._active_model_name = target.name
            self._active_model_ready = False
            self._wake(target)
            self._wait_until_not_sleeping(target)

        if not self._active_model_ready:
            self._wait_until_model_listed(target)
            self._run_startup_smoke(target)
            if self.admission_check is not None:
                # Revalidate after startup allocations and warmup, not only
                # before wake, so the lease is released only from a safe state.
                self.admission_check(target)
            self._active_model_ready = True
            self._publish_transition("ready", target.name)

        return target

    def _sleep(self, model: ModelConfig) -> None:
        query = urlencode({"level": str(self.sleep_level)})
        resp = self.http.request(
            "POST",
            f"{model.control_base_url}/sleep?{query}",
            timeout=self.request_timeout_s,
        )
        if resp.status >= 400:
            raise WakeError(f"sleep failed for {model.name}: HTTP {resp.status}: {resp.body[:300]!r}")

    def _wake(self, model: ModelConfig) -> None:
        if self.wake_strategy == "level2":
            self._wake_level2(model)
            return

        if self.admission_check is not None:
            self.admission_check(model)
        resp = self.http.request(
            "POST",
            f"{model.control_base_url}/wake_up",
            timeout=self.request_timeout_s,
        )
        if resp.status >= 400:
            raise WakeError(f"wake_up failed for {model.name}: HTTP {resp.status}: {resp.body[:300]!r}")

    def _wake_level2(self, model: ModelConfig) -> None:
        for url, body, label in (
            (f"{model.control_base_url}/wake_up?tags=weights", None, "wake_up weights"),
            (
                f"{model.control_base_url}/collective_rpc",
                b'{"method":"reload_weights"}',
                "reload_weights",
            ),
            (f"{model.control_base_url}/wake_up?tags=kv_cache", None, "wake_up kv_cache"),
            (
                f"{model.control_base_url}/reset_mm_cache",
                None,
                "reset multimodal cache",
            ),
        ):
            # Check before each allocation phase. The host monitor may have
            # crossed its boundary during an earlier phase of level-2 wake.
            if self.admission_check is not None:
                self.admission_check(model)
            resp = self.http.request(
                "POST",
                url,
                headers={"content-type": "application/json"} if body else None,
                body=body,
                timeout=self.request_timeout_s,
            )
            if resp.status >= 400:
                raise WakeError(
                    f"{label} failed for {model.name}: HTTP {resp.status}: {resp.body[:300]!r}"
                )

    def _is_sleeping(self, model: ModelConfig) -> bool | None:
        resp = self.http.request(
            "GET",
            f"{model.control_base_url}/is_sleeping",
            timeout=self.request_timeout_s,
        )
        if resp.status == 404:
            # Older/alternate builds may not expose this endpoint. The model-list
            # check remains authoritative for inference readiness.
            return None
        if resp.status >= 400:
            raise WakeError(
                f"sleep-state check failed for {model.name}: "
                f"HTTP {resp.status}: {resp.body[:300]!r}"
            )
        parsed = resp.json()
        if isinstance(parsed, bool):
            return parsed
        if isinstance(parsed, dict):
            for key in ("is_sleeping", "sleeping"):
                value = parsed.get(key)
                if isinstance(value, bool):
                    return value
        return None

    def _wait_until_sleeping(self, model: ModelConfig) -> None:
        def sleeping() -> bool:
            return self._is_sleeping(model) is True

        if not sleep_until(
            sleeping,
            timeout_s=self.wake_timeout_s,
            interval_s=self.poll_interval_s,
        ):
            raise WakeError(f"timed out waiting for {model.name} to enter sleep state")

    def _wait_until_not_sleeping(self, model: ModelConfig) -> None:
        def ready() -> bool:
            sleeping = self._is_sleeping(model)
            return sleeping is not True

        if not sleep_until(ready, timeout_s=self.wake_timeout_s, interval_s=self.poll_interval_s):
            raise WakeError(f"timed out waiting for {model.name} to leave sleep state")

    def _run_startup_smoke(self, model: ModelConfig) -> None:
        if not model.startup_smoke_path:
            return
        body = dict(model.startup_smoke_body or {})
        body.setdefault("model", model.upstream_model)
        resp = self.http.request(
            "POST",
            f"{model.upstream_base_url}{model.startup_smoke_path}",
            headers={"content-type": "application/json"},
            body=json.dumps(body, separators=(",", ":")).encode(),
            timeout=self.request_timeout_s,
        )
        if resp.status >= 400:
            raise WakeError(
                f"startup smoke failed for {model.name}: "
                f"HTTP {resp.status}: {resp.body[:300]!r}"
            )

    def _wait_until_model_listed(self, model: ModelConfig) -> None:
        def listed() -> bool:
            resp = self.http.request(
                "GET",
                f"{model.upstream_base_url}/models",
                timeout=self.request_timeout_s,
            )
            if resp.status >= 400:
                return False
            parsed = resp.json()
            if not isinstance(parsed, dict):
                return False
            ids = [item.get("id") for item in parsed.get("data", []) if isinstance(item, dict)]
            return model.upstream_model in ids or model.name in ids

        if not sleep_until(listed, timeout_s=self.wake_timeout_s, interval_s=self.poll_interval_s):
            raise WakeError(f"timed out waiting for {model.name} to appear in /v1/models")

    def rewrite_request_body(self, body: bytes, target: ModelConfig) -> bytes:
        try:
            decoded = json.loads(body.decode("utf-8"))
        except Exception:
            return body
        if isinstance(decoded, dict) and "model" in decoded:
            decoded["model"] = target.upstream_model
            return json.dumps(decoded, separators=(",", ":")).encode("utf-8")
        return body
