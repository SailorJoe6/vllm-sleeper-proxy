from __future__ import annotations

import fcntl
import json
import math
import os
import threading
import time
from contextlib import contextmanager
from typing import Callable, Iterable, Iterator
from urllib.parse import urlencode

from .client import HttpClient, sleep_until
from .config import ModelConfig
from .thermal_control import MAX_SLEEPING_SUBSET_PROOF_SECONDS, ThermalAction


class UnknownModelError(ValueError):
    pass


class WakeError(RuntimeError):
    pass


class LifecycleUnavailableError(WakeError):
    """The cached owner cannot be positively revalidated yet."""


@contextmanager
def startup_lease(path: str, *, timeout_s: float | None = None) -> Iterator[None]:
    """Hold the shared lifecycle lease, optionally within one finite deadline."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a+", encoding="utf-8") as handle:
        if timeout_s is None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        else:
            if not math.isfinite(timeout_s) or timeout_s <= 0:
                raise WakeError("thermal action deadline expired before lifecycle lease")
            deadline = time.monotonic() + timeout_s
            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise WakeError("timed out waiting for shared lifecycle lease")
                    time.sleep(min(0.05, remaining))
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
        owner_validation_timeout_s: float = 2.0,
        wake_timeout_s: float = 300.0,
        drain_timeout_s: float = 300.0,
        poll_interval_s: float = 1.0,
        always_wake: bool = True,
        wake_strategy: str = "level2",
        admission_check: Callable[[ModelConfig], None] | None = None,
        pre_admission_check: Callable[[ModelConfig], None] | None = None,
        startup_lease_path: str | None = None,
        transition_state_path: str | None = None,
        bootstrap_mode: bool = False,
    ) -> None:
        self.models = list(models)
        if not self.models:
            raise ValueError("at least one model must be configured")
        if len(self.models) > 64:
            raise ValueError("at most 64 models may be configured")
        names = [model.name for model in self.models]
        if any(
            not name or len(name) > 256 or not name.isprintable()
            for name in names
        ):
            raise ValueError("model names must be printable and at most 256 characters")
        if len(set(names)) != len(names):
            raise ValueError("configured model names must be unique")
        control_urls = [model.control_base_url for model in self.models]
        if len(set(control_urls)) != len(control_urls):
            raise ValueError("configured engine control URLs must be unique")
        self.http = http
        self.sleep_level = sleep_level
        self.request_timeout_s = request_timeout_s
        if not math.isfinite(owner_validation_timeout_s) or owner_validation_timeout_s <= 0:
            raise ValueError("owner_validation_timeout_s must be finite and greater than zero")
        self.owner_validation_timeout_s = owner_validation_timeout_s
        self.wake_timeout_s = wake_timeout_s
        self.drain_timeout_s = drain_timeout_s
        self.poll_interval_s = poll_interval_s
        self.always_wake = always_wake
        self.wake_strategy = wake_strategy
        self.admission_check = admission_check
        self.pre_admission_check = pre_admission_check
        self.startup_lease_path = startup_lease_path or os.environ.get(
            "SLEEPER_STARTUP_LEASE_PATH", "/tmp/vllm-sleeper-proxy-startup.lock"
        )
        self.transition_state_path = transition_state_path or os.environ.get("SLEEPER_TRANSITION_STATE_PATH")
        self.bootstrap_mode = bootstrap_mode
        self._startup_finalized = not bootstrap_mode
        self._condition = threading.Condition()
        self._thermal_action_lock = threading.Lock()
        self._active_model_name: str | None = None
        self._starting_model_name: str | None = None
        self._active_model_ready = False
        self._inflight_requests = 0
        self._pending_switch_name: str | None = None
        self._quiescing = False
        self._lifecycle_state = "unknown"

    @property
    def startup_finalized(self) -> bool:
        # Health/status requests must remain observable while a lifecycle
        # transition holds the condition lock for a long weight reload.
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
                        self._clear_owner_locked()
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
                    self._clear_owner_locked()
                    self._startup_finalized = True
            finally:
                with self._condition:
                    self._starting_model_name = None
                    self._publish_transition("idle")
                    self._condition.notify_all()

    def adopt_startup_state(self) -> None:
        """Adopt one already-awake engine without changing any engine state.

        This is used only when the proxy control plane restarts around an
        otherwise healthy lineup. Ambiguous state is rejected so recovery can
        retry without sleeping or stopping a healthy engine.
        """
        with startup_lease(self.startup_lease_path):
            with self._condition:
                self._starting_model_name = "__proxy_recovery__"
                self._publish_transition("adopting", "__proxy_recovery__")
            try:
                awake: list[ModelConfig] = []
                for model in self.models:
                    sleeping = self._is_sleeping(model)
                    if sleeping is None:
                        raise WakeError(
                            f"cannot verify proxy recovery state for {model.name}"
                        )
                    if not sleeping:
                        awake.append(model)
                if len(awake) > 1:
                    raise WakeError(
                        "cannot adopt proxy recovery state with multiple awake models"
                    )
                if awake:
                    self._wait_until_model_listed(awake[0])
                with self._condition:
                    self._active_model_name = awake[0].name if awake else None
                    self._active_model_ready = bool(awake)
                    self._lifecycle_state = "active" if awake else "sleeping"
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
    def lifecycle_state(self) -> str:
        """Sanitized lifecycle state for control-plane health and repair routing."""
        return self._lifecycle_state

    @property
    def inference_ready(self) -> bool:
        """Whether lifecycle ownership is safe for request admission."""
        return self._startup_finalized and self._lifecycle_state in {
            "active",
            "sleeping",
        }

    def thermal_admission_snapshot(self):
        """Return a sanitized fast-gate view without lifecycle lock contention."""
        observer = getattr(self.pre_admission_check, "snapshot", None)
        if not callable(observer):
            return None
        return observer(self.models[0])

    @property
    def starting_model_name(self) -> str | None:
        """Model that owns the startup lease until readiness is verified.

        This read is intentionally non-blocking so the host guard can observe
        an in-progress wake while the transition thread owns the condition.
        Writers still serialize all state changes under that condition.
        """
        return self._starting_model_name

    @property
    def inflight_requests(self) -> int:
        """Approximate request count for nonblocking health observation."""
        return self._inflight_requests

    def _publish_transition(self, phase: str, model_name: str | None = None) -> None:
        """Atomically publish lifecycle ownership for the host guard.

        The file is advisory only; the host guard validates and reports stale,
        malformed, or mismatched records. Deployment admission policy decides
        whether that uncertainty may gate an optional or required model. Atomic
        replacement prevents readers from observing a partial transition.
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

                    self._clear_owner_locked()
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

    def _check_pre_admission(self, target: ModelConfig) -> None:
        if self.pre_admission_check is not None:
            self.pre_admission_check(target)

    def check_fast_admission(self) -> None:
        """Check the host-wide fast gate before reading a new request body."""
        self._check_pre_admission(self.models[0])

    def acquire(self, requested: str) -> ModelLease:
        """Wake a model and hold it awake for one buffered or streaming request."""

        target = self.find_model(requested)
        # Check before contending on lifecycle state so a cooldown response is
        # prompt even while another request is draining or switching.
        self._check_pre_admission(target)
        with self._condition:
            try:
                # Recheck under the condition to close the race between the first
                # fast read and lifecycle ownership.
                # The fast thermal projection is checked before waiting for any
                # switch/drain. Cooldown requests must fail promptly instead of
                # joining a queue behind the active request that is draining.
                self._check_pre_admission(target)
                self._wait_for_switch_safety(target)
                # A request can wait across an externally published thermal hold.
                # Re-read the fast fence after every lifecycle wait and again in
                # the activation path so a queued request cannot wake after proof.
                self._check_pre_admission(target)
                if self.admission_check is not None:
                    self.admission_check(target)
                self._check_pre_admission(target)
                self._activate_locked(target)
                self._inflight_requests += 1
            except Exception:
                if self._pending_switch_name == target.name:
                    self._pending_switch_name = None
                    self._condition.notify_all()
                raise
        return ModelLease(self, target)

    def sleep_active_model(self) -> str | None:
        """Quiesce new requests, drain ownership, and sleep the active model.

        Local request state is acquired before the host-wide lease. This keeps
        lock ordering identical to startup and avoids a condition/lease
        deadlock when another proxy instance is starting a model.
        """

        with self._condition:
            if self._quiescing:
                raise WakeError("lifecycle is already quiescing")
            deadline = time.monotonic() + self.drain_timeout_s
            self._quiescing = True
            try:
                while self._inflight_requests > 0:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise WakeError("timed out draining requests before resource backoff")
                    self._condition.wait(timeout=remaining)
                if self._active_model_name is None:
                    try:
                        deadline = time.monotonic() + self.owner_validation_timeout_s
                        with startup_lease(self.startup_lease_path):
                            states = [
                                self._is_sleeping(
                                    model,
                                    timeout_s=self._remaining_owner_validation_timeout(deadline),
                                )
                                for model in self.models
                            ]
                    except Exception as exc:
                        self._lifecycle_state = "unknown"
                        self._active_model_ready = False
                        self._publish_transition("unknown")
                        raise LifecycleUnavailableError(
                            "all-engine sleep state unavailable; retry later"
                        ) from exc
                    if not states or any(state is not True for state in states):
                        self._lifecycle_state = "unknown"
                        self._active_model_ready = False
                        self._publish_transition("unknown")
                        raise LifecycleUnavailableError(
                            "all-engine sleep state unavailable; retry later"
                        )
                    self._lifecycle_state = "sleeping"
                    self._quiescing = False
                    self._publish_transition("idle")
                    self._condition.notify_all()
                    return None
                current = self.find_model(self._active_model_name)
            except Exception:
                self._quiescing = False
                self._condition.notify_all()
                raise

        try:
            self._publish_transition("sleeping", current.name)
            with startup_lease(self.startup_lease_path):
                sleeping = self._is_sleeping(
                    current,
                    timeout_s=self.owner_validation_timeout_s,
                )
                if sleeping is None:
                    raise LifecycleUnavailableError(
                        f"lifecycle state unavailable for {current.name}; retry later"
                    )
                if sleeping is False:
                    self._sleep(current)
                    self._wait_until_sleeping(current)
        except Exception as exc:
            with self._condition:
                self._mark_lifecycle_unknown_locked(current)
                self._quiescing = False
                self._condition.notify_all()
            raise LifecycleUnavailableError(
                f"lifecycle state unavailable for {current.name}; retry later"
            ) from exc

        with self._condition:
            self._clear_owner_locked()
            self._quiescing = False
            self._condition.notify_all()
        return current.name

    @staticmethod
    def _remaining_epoch(deadline_epoch: float, label: str) -> float:
        remaining = deadline_epoch - time.time()
        if not math.isfinite(remaining) or remaining <= 0:
            raise WakeError(f"thermal action {label} deadline expired")
        return remaining

    def _finish_thermal_action(self) -> None:
        with self._condition:
            self._quiescing = False
            self._condition.notify_all()

    def thermal_hold(
        self,
        action: ThermalAction,
        *,
        reauthorize: Callable[[], None],
    ) -> dict[str, object]:
        """Drain and positively prove every configured engine sleeping.

        The root projection owns identity, phase, and immutable deadlines.
        This method never treats an HTTP write as containment proof.
        """
        if action.phase not in {"graceful_hold", "urgent_hold", "held"}:
            raise WakeError("thermal action phase is not a hold phase")
        if self.sleep_level != 2:
            raise WakeError("thermal actions require vLLM sleep level 2")
        lock_timeout = (
            self._remaining_epoch(action.overall_deadline_epoch, "overall")
            if action.phase != "held"
            else self.owner_validation_timeout_s
        )
        if not self._thermal_action_lock.acquire(timeout=lock_timeout):
            raise WakeError("timed out serializing thermal action")
        quiescing = False
        try:
            reauthorize()
            condition_timeout = (
                self._remaining_epoch(action.overall_deadline_epoch, "overall")
                if action.phase != "held"
                else self.owner_validation_timeout_s
            )
            if not self._condition.acquire(timeout=condition_timeout):
                raise WakeError("timed out acquiring lifecycle condition")
            try:
                reauthorize()
                if self._quiescing:
                    raise WakeError("lifecycle is already quiescing")
                self._quiescing = True
                quiescing = True
                if action.phase == "held" and self._inflight_requests:
                    raise WakeError("held action still has in-flight requests")
                while self._inflight_requests > 0:
                    remaining = self._remaining_epoch(
                        action.drain_deadline_epoch, "drain"
                    )
                    self._condition.wait(timeout=remaining)
            finally:
                self._condition.release()

            if action.phase != "held":
                lifecycle_deadline = min(
                    action.sleep_deadline_epoch, action.overall_deadline_epoch
                )
            else:
                lifecycle_deadline = time.time() + self.owner_validation_timeout_s
            with startup_lease(
                self.startup_lease_path,
                timeout_s=self._remaining_epoch(lifecycle_deadline, "sleep"),
            ):
                proofs: dict[str, dict[str, object]] = {}
                for model in self.models:
                    reauthorize()
                    sleeping = self._is_sleeping(
                        model,
                        timeout_s=min(
                            self.request_timeout_s,
                            self._remaining_epoch(lifecycle_deadline, "sleep"),
                        ),
                    )
                    if sleeping is None:
                        raise WakeError(
                            f"cannot verify thermal sleep state for {model.name}"
                        )
                    if sleeping is False:
                        if action.phase == "held":
                            raise WakeError(
                                f"engine is awake during held verification: {model.name}"
                            )
                        reauthorize()
                        self._sleep(
                            model,
                            timeout_s=min(
                                self.request_timeout_s,
                                self._remaining_epoch(lifecycle_deadline, "sleep"),
                            ),
                        )
                        self._wait_until_sleeping(
                            model,
                            timeout_s=self._remaining_epoch(
                                lifecycle_deadline, "sleep"
                            ),
                        )
                    reauthorize()
                    final_state = self._is_sleeping(
                        model,
                        timeout_s=min(
                            self.request_timeout_s,
                            self._remaining_epoch(lifecycle_deadline, "sleep"),
                        ),
                    )
                    if final_state is not True:
                        raise WakeError(
                            f"positive thermal sleep proof unavailable for {model.name}"
                        )
                    proofs[model.name] = {
                        "state": "sleeping",
                        "proof": "vllm_is_sleeping_true",
                    }
                reauthorize()
                with self._condition:
                    self._clear_owner_locked()
            return {"all_sleeping": True, "engine_proofs": proofs}
        finally:
            if quiescing:
                self._finish_thermal_action()
            self._thermal_action_lock.release()

    def thermal_sleeping_subset_proof(
        self,
        action: ThermalAction,
        *,
        reauthorize: Callable[[], None],
    ) -> dict[str, object]:
        """Positively verify only root-declared retained sleeping peers."""
        if action.phase != "held":
            raise WakeError("sleeping subset proof requires held phase")
        configured = tuple(sorted(model.name for model in self.models))
        if action.engine_keys != configured:
            raise WakeError("thermal action engine set does not match configuration")
        if (
            not action.proof_engine_keys
            or action.proof_engine_keys != tuple(sorted(set(action.proof_engine_keys)))
            or any(key not in action.engine_keys for key in action.proof_engine_keys)
        ):
            raise WakeError("thermal sleeping subset is invalid")
        models = {model.name: model for model in self.models}
        deadline = time.monotonic() + MAX_SLEEPING_SUBSET_PROOF_SECONDS

        def remaining(stage: str) -> float:
            value = deadline - time.monotonic()
            if value <= 0:
                raise WakeError(f"thermal sleeping subset {stage} deadline expired")
            return value

        if not self._thermal_action_lock.acquire(timeout=remaining("action lock")):
            raise WakeError("timed out serializing thermal sleeping subset proof")
        quiescing = False
        try:
            reauthorize()
            if not self._condition.acquire(timeout=remaining("condition")):
                raise WakeError("timed out acquiring lifecycle condition")
            try:
                reauthorize()
                if self._quiescing:
                    raise WakeError("lifecycle is already quiescing")
                if self._inflight_requests:
                    raise WakeError("sleeping subset proof still has in-flight requests")
                self._quiescing = True
                quiescing = True
            finally:
                self._condition.release()

            with startup_lease(
                self.startup_lease_path,
                timeout_s=remaining("startup lease"),
            ):
                proofs: dict[str, dict[str, object]] = {}
                for name in action.proof_engine_keys:
                    reauthorize()
                    sleeping = self._is_sleeping(
                        models[name],
                        timeout_s=min(
                            self.request_timeout_s,
                            remaining("engine proof"),
                        ),
                    )
                    reauthorize()
                    if sleeping is not True:
                        raise WakeError(
                            f"sleeping subset proof unavailable for {name}"
                        )
                    proofs[name] = {
                        "state": "sleeping",
                        "proof": "vllm_is_sleeping_true",
                    }
                reauthorize()
            return {
                "sleeping_subset_verified": True,
                "engine_proofs": proofs,
            }
        finally:
            if quiescing:
                self._finish_thermal_action()
            self._thermal_action_lock.release()

    def thermal_release_ready(
        self,
        action: ThermalAction,
        *,
        reauthorize: Callable[[], None],
    ) -> dict[str, object]:
        """Verify all-sleeping release readiness without sleeping or waking."""
        if action.phase not in {
            "release_authorized", "cutoff_recovery_authorized", "releasing"
        }:
            raise WakeError("thermal action phase is not a release phase")
        if not self._thermal_action_lock.acquire(
            timeout=self.owner_validation_timeout_s
        ):
            raise WakeError("timed out serializing thermal release proof")
        quiescing = False
        deadline = time.monotonic() + self.owner_validation_timeout_s
        try:
            reauthorize()
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self._condition.acquire(timeout=remaining):
                raise WakeError("timed out acquiring lifecycle condition")
            try:
                reauthorize()
                if self._quiescing:
                    raise WakeError("lifecycle is already quiescing")
                self._quiescing = True
                quiescing = True
                if self._inflight_requests:
                    raise WakeError("release proof still has in-flight requests")
            finally:
                self._condition.release()
            with startup_lease(
                self.startup_lease_path,
                timeout_s=max(0.001, deadline - time.monotonic()),
            ):
                proofs: dict[str, dict[str, object]] = {}
                for model in self.models:
                    reauthorize()
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise WakeError("thermal release proof deadline expired")
                    sleeping = self._is_sleeping(
                        model,
                        timeout_s=min(self.request_timeout_s, remaining),
                    )
                    if sleeping is not True:
                        raise WakeError(
                            f"all-sleeping release proof unavailable for {model.name}"
                        )
                    proofs[model.name] = {
                        "state": "sleeping",
                        "proof": "vllm_is_sleeping_true",
                    }
                reauthorize()
                with self._condition:
                    self._clear_owner_locked()
            return {"release_ready": True, "engine_proofs": proofs}
        finally:
            if quiescing:
                self._finish_thermal_action()
            self._thermal_action_lock.release()

    def release(self, target: ModelConfig) -> None:
        with self._condition:
            if self._inflight_requests <= 0:
                raise RuntimeError(f"no in-flight request to release for {target.name}")
            self._inflight_requests -= 1
            self._condition.notify_all()

    def mark_owner_unavailable(self, target: ModelConfig) -> None:
        """Invalidate cached readiness after a forward-open transport failure."""
        with self._condition:
            if self._active_model_name == target.name:
                self._mark_lifecycle_unknown_locked(target)
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

    def _mark_lifecycle_unknown_locked(self, owner: ModelConfig) -> None:
        self._active_model_ready = False
        self._lifecycle_state = "unknown"
        self._publish_transition("unknown", owner.name)

    def _clear_owner_locked(self) -> None:
        self._active_model_name = None
        self._active_model_ready = False
        self._lifecycle_state = "sleeping"
        self._publish_transition("idle")

    def _raise_lifecycle_unavailable_locked(
        self,
        owner: ModelConfig,
        cause: BaseException | None = None,
    ) -> None:
        self._mark_lifecycle_unknown_locked(owner)
        error = LifecycleUnavailableError(
            f"lifecycle state unavailable for {owner.name}; retry later"
        )
        if cause is None:
            raise error
        raise error from cause

    @staticmethod
    def _remaining_owner_validation_timeout(deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("owner validation deadline expired")
        return remaining

    def _revalidate_cached_owner_locked(self, owner: ModelConfig) -> bool:
        """Positively validate cached same-model ownership under one deadline.

        ``True`` means the existing owner is awake and inference-ready. ``False``
        means it is positively sleeping and normal guarded startup may proceed.
        Unknown or unreachable state preserves ownership and returns a typed 503
        path instead of forwarding to a vanished upstream.
        """
        deadline = time.monotonic() + self.owner_validation_timeout_s
        try:
            sleeping = self._is_sleeping(
                owner,
                timeout_s=self._remaining_owner_validation_timeout(deadline),
            )
        except (ConnectionError, OSError, TimeoutError, TypeError, ValueError, WakeError) as exc:
            self._raise_lifecycle_unavailable_locked(owner, exc)
        if sleeping is None:
            self._raise_lifecycle_unavailable_locked(owner)
        if sleeping is True:
            self._clear_owner_locked()
            return False
        for peer in self.models:
            if peer.name == owner.name:
                continue
            try:
                peer_sleeping = self._is_sleeping(
                    peer,
                    timeout_s=self._remaining_owner_validation_timeout(deadline),
                )
            except (ConnectionError, OSError, TimeoutError, TypeError, ValueError, WakeError) as exc:
                self._raise_lifecycle_unavailable_locked(owner, exc)
            if peer_sleeping is not True:
                self._raise_lifecycle_unavailable_locked(owner)
        try:
            listed = self._model_is_listed_once(
                owner,
                timeout_s=self._remaining_owner_validation_timeout(deadline),
            )
        except (ConnectionError, OSError, TimeoutError, TypeError, ValueError, WakeError) as exc:
            self._raise_lifecycle_unavailable_locked(owner, exc)
        if not listed:
            self._raise_lifecycle_unavailable_locked(owner)
        self._active_model_ready = True
        self._lifecycle_state = "active"
        self._publish_transition("ready", owner.name)
        return True

    def _activate_locked(self, target: ModelConfig) -> ModelConfig:
        try:
            self._check_pre_admission(target)
            with startup_lease(self.startup_lease_path):
                self._check_pre_admission(target)
                activated = self._ensure_awake_locked(target)
                self._check_pre_admission(target)
                return activated
        except WakeError:
            raise
        except (ConnectionError, OSError, TimeoutError, TypeError, ValueError) as exc:
            raise WakeError(f"lifecycle check failed for {target.name}: {exc}") from exc
        finally:
            if self._pending_switch_name == target.name:
                self._pending_switch_name = None
                self._condition.notify_all()

    def _ensure_awake_locked(self, target: ModelConfig) -> ModelConfig:
        if self._active_model_name == target.name:
            if self._revalidate_cached_owner_locked(target):
                return target
        self._starting_model_name = target.name
        self._lifecycle_state = "starting"
        self._publish_transition("starting", target.name)
        try:
            return self._ensure_awake_startup_locked(target)
        except LifecycleUnavailableError:
            raise
        except Exception as exc:
            owner_name = self._active_model_name
            if owner_name is None:
                self._active_model_ready = False
                self._lifecycle_state = "unknown"
                self._publish_transition("unknown")
            else:
                owner = self.find_model(owner_name)
                if owner.name != target.name:
                    self._raise_lifecycle_unavailable_locked(owner, exc)
                try:
                    sleeping = self._is_sleeping(
                        owner,
                        timeout_s=self.owner_validation_timeout_s,
                    )
                    if sleeping is False:
                        self._sleep(owner)
                        self._wait_until_sleeping(owner)
                        sleeping = True
                except (ConnectionError, OSError, TimeoutError, TypeError, ValueError, WakeError) as cleanup_exc:
                    self._raise_lifecycle_unavailable_locked(owner, cleanup_exc)
                if sleeping is not True:
                    self._raise_lifecycle_unavailable_locked(owner, exc)
                self._clear_owner_locked()
            raise
        finally:
            self._starting_model_name = None
            if self._lifecycle_state != "unknown":
                self._publish_transition(
                    "ready" if self._active_model_ready else "idle",
                    self._active_model_name,
                )
            self._condition.notify_all()

    def _ensure_awake_startup_locked(self, target: ModelConfig) -> ModelConfig:
        if self._active_model_name and self._active_model_name != target.name:
            current = self.find_model(self._active_model_name)
            self._sleep(current)
            # Do not sample admission while the previous engine is still
            # asynchronously reclaiming memory. Switching admission is only
            # valid after the proxy verifies the engine's sleep state.
            self._wait_until_sleeping(current)
            self._clear_owner_locked()

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
                self._lifecycle_state = "starting"
        elif self.always_wake and self._active_model_name == target.name:
            sleeping = self._is_sleeping(target)
            should_wake = sleeping is not False

        if should_wake:
            # Recheck after any previous engine has been slept. Sleeping is an
            # asynchronous memory transition, so an admission decision made
            # before the drain can be stale by the time this engine allocates.
            self._check_pre_admission(target)
            if self.admission_check is not None:
                self.admission_check(target)
            self._check_pre_admission(target)
            # Conservatively remember the target before wake begins. A partial
            # wake failure can leave weights resident; the next model switch
            # must sleep this engine even when readiness never completed.
            self._active_model_name = target.name
            self._active_model_ready = False
            self._lifecycle_state = "starting"
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
            self._lifecycle_state = "active"
            self._publish_transition("ready", target.name)

        return target

    def _sleep(
        self, model: ModelConfig, *, timeout_s: float | None = None
    ) -> None:
        query = urlencode({"level": str(self.sleep_level)})
        resp = self.http.request(
            "POST",
            f"{model.control_base_url}/sleep?{query}",
            timeout=self.request_timeout_s if timeout_s is None else timeout_s,
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

    def _is_sleeping(
        self,
        model: ModelConfig,
        *,
        timeout_s: float | None = None,
    ) -> bool | None:
        resp = self.http.request(
            "GET",
            f"{model.control_base_url}/is_sleeping",
            timeout=self.request_timeout_s if timeout_s is None else timeout_s,
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

    def _wait_until_sleeping(
        self, model: ModelConfig, *, timeout_s: float | None = None
    ) -> None:
        effective_timeout = self.wake_timeout_s if timeout_s is None else timeout_s
        deadline = time.monotonic() + effective_timeout

        def sleeping() -> bool:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            return self._is_sleeping(
                model, timeout_s=min(self.request_timeout_s, remaining)
            ) is True

        if not sleep_until(
            sleeping,
            timeout_s=effective_timeout,
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

    def _model_is_listed_once(
        self,
        model: ModelConfig,
        *,
        timeout_s: float | None = None,
    ) -> bool:
        resp = self.http.request(
            "GET",
            f"{model.upstream_base_url}/models",
            timeout=self.request_timeout_s if timeout_s is None else timeout_s,
        )
        if resp.status >= 400:
            return False
        parsed = resp.json()
        if not isinstance(parsed, dict):
            return False
        ids = [item.get("id") for item in parsed.get("data", []) if isinstance(item, dict)]
        return model.upstream_model in ids or model.name in ids

    def _wait_until_model_listed(self, model: ModelConfig) -> None:
        if not sleep_until(
            lambda: self._model_is_listed_once(model),
            timeout_s=self.wake_timeout_s,
            interval_s=self.poll_interval_s,
        ):
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
