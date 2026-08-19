from __future__ import annotations

import json
import time
from pathlib import Path

from .config import ModelConfig


class AdmissionError(RuntimeError):
    pass


class FileAdmissionGuard:
    """Fail closed when the host resource monitor does not admit a model."""

    def __init__(self, status_path: Path, *, now=time.time) -> None:
        self.status_path = status_path
        self.now = now

    def __call__(self, model: ModelConfig) -> None:
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
        if not allowed:
            reason = str(decision.get("reason", "denied"))
            raise AdmissionError(f"resource admission denied for {model.name}: {reason}")
