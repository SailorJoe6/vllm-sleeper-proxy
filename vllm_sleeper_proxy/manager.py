from __future__ import annotations

import json
import threading
from typing import Iterable
from urllib.parse import urlencode

from .client import HttpClient, sleep_until
from .config import ModelConfig


class UnknownModelError(ValueError):
    pass


class WakeError(RuntimeError):
    pass


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
        poll_interval_s: float = 1.0,
        always_wake: bool = True,
        wake_strategy: str = "level2",
    ) -> None:
        self.models = list(models)
        if not self.models:
            raise ValueError("at least one model must be configured")
        self.http = http
        self.sleep_level = sleep_level
        self.request_timeout_s = request_timeout_s
        self.wake_timeout_s = wake_timeout_s
        self.poll_interval_s = poll_interval_s
        self.always_wake = always_wake
        self.wake_strategy = wake_strategy
        self._lock = threading.Lock()
        self._active_model_name: str | None = None

    @property
    def active_model_name(self) -> str | None:
        return self._active_model_name

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
        with self._lock:
            if self._active_model_name and self._active_model_name != target.name:
                current = self.find_model(self._active_model_name)
                self._sleep(current)
                self._active_model_name = None

            should_wake = self._active_model_name != target.name
            if self._active_model_name is None:
                sleeping = self._is_sleeping(target)
                if sleeping is False:
                    should_wake = False
                    self._active_model_name = target.name
            elif self.always_wake and self._active_model_name == target.name:
                sleeping = self._is_sleeping(target)
                should_wake = sleeping is not False

            if should_wake:
                self._wake(target)
                self._wait_until_not_sleeping(target)
                self._wait_until_model_listed(target)
                self._active_model_name = target.name

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
        ):
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
            return True
        parsed = resp.json()
        if isinstance(parsed, bool):
            return parsed
        if isinstance(parsed, dict):
            return bool(parsed.get("is_sleeping", parsed.get("sleeping", False)))
        return None

    def _wait_until_not_sleeping(self, model: ModelConfig) -> None:
        def ready() -> bool:
            sleeping = self._is_sleeping(model)
            return sleeping is not True

        if not sleep_until(ready, timeout_s=self.wake_timeout_s, interval_s=self.poll_interval_s):
            raise WakeError(f"timed out waiting for {model.name} to leave sleep state")

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
