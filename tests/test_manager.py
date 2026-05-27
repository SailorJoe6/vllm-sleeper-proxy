from __future__ import annotations

import json
import unittest

from vllm_sleeper_proxy.client import HttpResponse
from vllm_sleeper_proxy.config import ModelConfig
from vllm_sleeper_proxy.manager import ModelManager, UnknownModelError


class FakeHttp:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, bytes | None]] = []
        self.sleeping = True

    def request(self, method, url, *, headers=None, body=None, timeout=None):
        self.calls.append((method, url, body))
        if url.endswith("/wake_up"):
            self.sleeping = False
            return HttpResponse(200, {}, b"{}")
        if "/sleep?" in url:
            self.sleeping = True
            return HttpResponse(200, {}, b"{}")
        if url.endswith("/is_sleeping"):
            return HttpResponse(200, {}, json.dumps({"is_sleeping": self.sleeping}).encode())
        if url.endswith("/v1/models"):
            return HttpResponse(
                200,
                {},
                b'{"object":"list","data":[{"id":"Qwen/Qwen3-Embedding-8B"}]}',
            )
        raise AssertionError(f"unexpected request: {method} {url}")


def qwen_model() -> ModelConfig:
    return ModelConfig(
        name="Qwen3-Embedding-8B",
        upstream_model="Qwen/Qwen3-Embedding-8B",
        upstream_base_url="http://vllm:8888/v1",
        control_base_url="http://vllm:8888",
    )


class ModelManagerTests(unittest.TestCase):
    def test_openai_model_list_is_logical_and_stable(self) -> None:
        manager = ModelManager([qwen_model()], FakeHttp())
        payload = manager.list_openai_models()
        self.assertEqual(payload["object"], "list")
        self.assertEqual(payload["data"][0]["id"], "Qwen3-Embedding-8B")

    def test_ensure_awake_wakes_then_checks_readiness(self) -> None:
        http = FakeHttp()
        manager = ModelManager([qwen_model()], http, poll_interval_s=0, wake_timeout_s=1)
        target = manager.ensure_awake("Qwen3-Embedding-8B")
        self.assertEqual(target.upstream_model, "Qwen/Qwen3-Embedding-8B")
        self.assertEqual(manager.active_model_name, "Qwen3-Embedding-8B")
        urls = [url for _, url, _ in http.calls]
        self.assertIn("http://vllm:8888/wake_up", urls)
        self.assertIn("http://vllm:8888/is_sleeping", urls)
        self.assertIn("http://vllm:8888/v1/models", urls)

    def test_aliases_resolve_to_model(self) -> None:
        model = ModelConfig(
            name="Qwen3-Embedding-8B",
            aliases=("qwen-emb",),
            upstream_model="Qwen/Qwen3-Embedding-8B",
            upstream_base_url="http://vllm:8888/v1",
            control_base_url="http://vllm:8888",
        )
        manager = ModelManager([model], FakeHttp(), poll_interval_s=0, wake_timeout_s=1)
        self.assertEqual(manager.ensure_awake("qwen-emb").name, "Qwen3-Embedding-8B")

    def test_upstream_model_id_resolves_to_model(self) -> None:
        manager = ModelManager([qwen_model()], FakeHttp(), poll_interval_s=0, wake_timeout_s=1)
        self.assertEqual(
            manager.ensure_awake("Qwen/Qwen3-Embedding-8B").name,
            "Qwen3-Embedding-8B",
        )

    def test_litellm_hosted_vllm_model_id_resolves_to_model(self) -> None:
        manager = ModelManager([qwen_model()], FakeHttp(), poll_interval_s=0, wake_timeout_s=1)
        self.assertEqual(
            manager.ensure_awake("hosted_vllm/Qwen/Qwen3-Embedding-8B").name,
            "Qwen3-Embedding-8B",
        )

    def test_unknown_model_is_rejected(self) -> None:
        manager = ModelManager([qwen_model()], FakeHttp())
        with self.assertRaises(UnknownModelError):
            manager.ensure_awake("not-a-model")

    def test_request_body_is_rewritten_to_upstream_model(self) -> None:
        manager = ModelManager([qwen_model()], FakeHttp())
        rewritten = manager.rewrite_request_body(
            b'{"model":"Qwen3-Embedding-8B","input":"hello"}', qwen_model()
        )
        self.assertEqual(
            json.loads(rewritten.decode()),
            {"model": "Qwen/Qwen3-Embedding-8B", "input": "hello"},
        )


if __name__ == "__main__":
    unittest.main()
