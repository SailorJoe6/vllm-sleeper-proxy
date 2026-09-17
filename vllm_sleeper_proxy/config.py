from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for one logical model exposed by the proxy."""

    name: str
    upstream_model: str
    upstream_base_url: str
    control_base_url: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    owned_by: str = "vllm-sleeper-proxy"
    startup_smoke_path: str | None = None
    startup_smoke_body: dict[str, Any] | None = None
    required: bool = False

    @property
    def all_names(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)

    def matches(self, model: str) -> bool:
        return model in self.all_names or model in {
            self.upstream_model,
            f"hosted_vllm/{self.upstream_model}",
        }


def _strip_trailing_slash(value: str) -> str:
    return value.rstrip("/")


def _model_from_mapping(raw: dict[str, Any]) -> ModelConfig:
    name = str(raw["name"])
    upstream_model = str(raw.get("upstream_model") or name)
    upstream_base_url = _strip_trailing_slash(str(raw["upstream_base_url"]))
    control_base_url = _strip_trailing_slash(
        str(raw.get("control_base_url") or upstream_base_url.removesuffix("/v1"))
    )
    aliases = tuple(str(a) for a in raw.get("aliases", ()))
    owned_by = str(raw.get("owned_by") or "vllm-sleeper-proxy")
    smoke_path = raw.get("startup_smoke_path")
    smoke_body = raw.get("startup_smoke_body")
    required = raw.get("required", False)
    if smoke_path is not None and not isinstance(smoke_path, str):
        raise ValueError("startup_smoke_path must be a string")
    if smoke_body is not None and not isinstance(smoke_body, dict):
        raise ValueError("startup_smoke_body must be an object")
    if not isinstance(required, bool):
        raise ValueError("required must be boolean")
    return ModelConfig(
        name=name,
        upstream_model=upstream_model,
        upstream_base_url=upstream_base_url,
        control_base_url=control_base_url,
        aliases=aliases,
        owned_by=owned_by,
        startup_smoke_path=smoke_path,
        startup_smoke_body=smoke_body,
        required=required,
    )


def load_models_from_env() -> list[ModelConfig]:
    """Load model configuration from environment.

    SLEEPER_MODELS accepts JSON like:

    [
      {
        "name": "Qwen3-Embedding-8B",
        "upstream_model": "Qwen/Qwen3-Embedding-8B",
        "upstream_base_url": "http://vllm:8888/v1",
        "control_base_url": "http://vllm:8888"
      }
    ]

    If unset, a single Qwen embedding model is configured for the existing
    gbrain-embeddings Compose service.
    """

    raw = os.environ.get("SLEEPER_MODELS")
    if raw:
        decoded = json.loads(raw)
        if not isinstance(decoded, list):
            raise ValueError("SLEEPER_MODELS must be a JSON list")
        return [_model_from_mapping(item) for item in decoded]

    name = os.environ.get("SLEEPER_MODEL_NAME", "Qwen3-Embedding-8B")
    upstream_model = os.environ.get("SLEEPER_UPSTREAM_MODEL", "Qwen/Qwen3-Embedding-8B")
    upstream_base_url = _strip_trailing_slash(
        os.environ.get("SLEEPER_UPSTREAM_BASE_URL", "http://vllm:8888/v1")
    )
    control_base_url = _strip_trailing_slash(
        os.environ.get("SLEEPER_CONTROL_BASE_URL", upstream_base_url.removesuffix("/v1"))
    )
    aliases = tuple(
        part.strip()
        for part in os.environ.get("SLEEPER_MODEL_ALIASES", "").split(",")
        if part.strip()
    )
    return [
        ModelConfig(
            name=name,
            upstream_model=upstream_model,
            upstream_base_url=upstream_base_url,
            control_base_url=control_base_url,
            aliases=aliases,
        )
    ]
