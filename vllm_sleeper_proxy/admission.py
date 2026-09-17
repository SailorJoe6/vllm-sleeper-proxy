from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

from .config import ModelConfig


class AdmissionError(RuntimeError):
    pass


class FileAdmissionGuard:
    """Fail closed when the host resource monitor does not admit a model."""

    def __init__(
        self,
        status_path: Path,
        *,
        now=time.time,
        sleep=time.sleep,
        unknown_state_refreshes: int = 35,
        unknown_state_refresh_delay_seconds: float = 2.0,
    ) -> None:
        self.status_path = status_path
        self.now = now
        self.sleep = sleep
        self.unknown_state_refreshes = max(0, unknown_state_refreshes)
        self.unknown_state_refresh_delay_seconds = unknown_state_refresh_delay_seconds

    def __call__(self, model: ModelConfig) -> None:
        for attempt in range(self.unknown_state_refreshes + 1):
            try:
                status = json.loads(self.status_path.read_text(encoding="utf-8"))
                generated_at = float(status["generated_at_epoch"])
                max_age = float(status["max_age_seconds"])
                dependencies = status["workload_policy"]["model_dependencies"]
                if model.name not in dependencies:
                    raise KeyError(model.name)
                if model.upstream_model == "unsloth/Qwen3.8-27B-NVFP4":
                    sample = status["sample"]
                    if not isinstance(sample.get("gpu_memory_free_bytes"), (int, float)):
                        raise KeyError("gpu_memory_free_bytes")
                admissions = status["model_admission"]
                decision = admissions[model.name]
                allowed = decision["allowed"] is True
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise AdmissionError(
                    f"resource admission unavailable for {model.name}"
                ) from exc

            age = self.now() - generated_at
            if age < -5 or age > max_age:
                raise AdmissionError(f"resource admission stale for {model.name}")
            if allowed:
                return
            reason = str(decision.get("reason", "denied"))
            if reason != "unknown_model_state" or attempt >= self.unknown_state_refreshes:
                raise AdmissionError(f"resource admission denied for {model.name}: {reason}")
            self.sleep(self.unknown_state_refresh_delay_seconds)


THERMAL_STATES = {
    "healthy", "warning", "sleep", "recovering", "cutoff", "telemetry_failure",
}
THERMAL_REASON = re.compile(r"^[a-z0-9_]{1,128}$")


class ThermalCooldownError(AdmissionError):
    """New inference is paused by the independent fast thermal watchdog."""

    def __init__(self, message: str, *, state: str, retry_after_seconds: int) -> None:
        super().__init__(message)
        self.state = state
        self.retry_after_seconds = retry_after_seconds


class FileThermalAdmissionGuard:
    """Fail closed from the watchdog's sanitized fast admission projection."""

    def __init__(
        self,
        status_path: Path,
        *,
        now=time.time,
        maximum_age_seconds: float = 5.0,
        default_retry_after_seconds: int = 10,
    ) -> None:
        self.status_path = status_path
        self.now = now
        self.maximum_age_seconds = maximum_age_seconds
        self.default_retry_after_seconds = default_retry_after_seconds

    def _deny(self, state: str, retry_after: int, detail: str) -> None:
        raise ThermalCooldownError(
            "thermal_cooldown: new inference requests are paused while DGX "
            f"temperatures recover; state={state}; {detail}",
            state=state,
            retry_after_seconds=max(1, min(3600, retry_after)),
        )

    def __call__(self, model: ModelConfig) -> None:
        try:
            status = json.loads(self.status_path.read_text(encoding="utf-8"))
            if not isinstance(status, dict):
                raise TypeError("thermal status root must be an object")
            if type(status.get("schema_version")) is not int or status["schema_version"] != 1:
                raise ValueError("invalid thermal schema")
            generated_at = float(status["generated_at_epoch"])
            published_max_age = float(status["max_age_seconds"])
            raw_allowed = status["allowed"]
            if not isinstance(raw_allowed, bool):
                raise ValueError("thermal allowed must be boolean")
            allowed = raw_allowed
            state = str(status["state"])
            if state not in THERMAL_STATES:
                raise ValueError("invalid thermal state")
            if not math.isfinite(generated_at) or not math.isfinite(published_max_age):
                raise ValueError("nonfinite thermal time")
            if published_max_age <= 0:
                raise ValueError("invalid thermal max age")
            raw_retry = status.get(
                "retry_after_seconds", self.default_retry_after_seconds
            )
            if not isinstance(raw_retry, int) or isinstance(raw_retry, bool):
                raise ValueError("invalid retry-after")
            retry_after = raw_retry
            reason_codes = status.get("reason_codes", [])
            if (
                not isinstance(reason_codes, list)
                or len(reason_codes) > 16
                or any(
                    not isinstance(item, str) or not THERMAL_REASON.fullmatch(item)
                    for item in reason_codes
                )
            ):
                raise ValueError("invalid thermal reason codes")
            expected_reason = "allowed" if state == "healthy" else "thermal_cooldown"
            if status.get("reason") != expected_reason:
                raise ValueError("inconsistent thermal reason")
            if allowed != (state == "healthy"):
                raise ValueError("inconsistent thermal allowed state")
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            self._deny(
                "unavailable",
                self.default_retry_after_seconds,
                "fast thermal status is unavailable",
            )
            return
        age = self.now() - generated_at
        max_age = min(self.maximum_age_seconds, published_max_age)
        if age < -2 or age > max_age:
            self._deny(
                "stale",
                self.default_retry_after_seconds,
                "fast thermal status is stale",
            )
        if not allowed or state != "healthy":
            self._deny(
                state,
                retry_after,
                "retry after the watchdog completes its cool recovery hold",
            )
