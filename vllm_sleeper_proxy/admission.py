from __future__ import annotations

import json
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
        unknown_state_refreshes: int = 3,
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
