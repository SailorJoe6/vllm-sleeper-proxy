from __future__ import annotations

import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from vllm_sleeper_proxy.client import HttpResponse
from vllm_sleeper_proxy.config import ModelConfig
from vllm_sleeper_proxy.manager import ModelManager
from vllm_sleeper_proxy.server import build_server


class ProxyHttp:
    def __init__(self) -> None:
        self.requests: list[tuple[str, str, bytes | None]] = []

    def request(self, method, url, *, headers=None, body=None, timeout=None):
        self.requests.append((method, url, body))
        if url.endswith("/wake_up"):
            return HttpResponse(200, {}, b"{}")
        if url.endswith("/is_sleeping"):
            return HttpResponse(200, {}, b'{"is_sleeping":false}')
        if url.endswith("/v1/models"):
            return HttpResponse(200, {}, b'{"data":[{"id":"Qwen/Qwen3-Embedding-8B"}]}')
        if url.endswith("/v1/embeddings"):
            decoded = json.loads((body or b"{}").decode())
            if decoded["model"] != "Qwen/Qwen3-Embedding-8B":
                return HttpResponse(400, {"content-type": "application/json"}, b'{"bad":"model"}')
            return HttpResponse(
                200,
                {"content-type": "application/json"},
                b'{"object":"list","data":[{"embedding":[1,2,3]}]}',
            )
        raise AssertionError(f"unexpected request {method} {url}")


class ServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.http = ProxyHttp()
        model = ModelConfig(
            name="Qwen3-Embedding-8B",
            upstream_model="Qwen/Qwen3-Embedding-8B",
            upstream_base_url="http://vllm:8888/v1",
            control_base_url="http://vllm:8888",
        )
        manager = ModelManager([model], self.http, poll_interval_s=0, wake_timeout_s=1)
        self.server = build_server("127.0.0.1", 0, manager)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def get_json(self, path: str) -> dict:
        with urlopen(f"{self.base_url}{path}", timeout=2) as resp:  # noqa: S310 - local test server
            return json.loads(resp.read().decode())

    def post_json(self, path: str, payload: dict) -> tuple[int, dict]:
        req = Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode(),
            headers={"content-type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=2) as resp:  # noqa: S310 - local test server
            return resp.status, json.loads(resp.read().decode())

    def test_v1_models(self) -> None:
        payload = self.get_json("/v1/models")
        self.assertEqual(payload["data"][0]["id"], "Qwen3-Embedding-8B")

    def test_embedding_request_wakes_and_forwards(self) -> None:
        status, payload = self.post_json(
            "/v1/embeddings",
            {"model": "Qwen3-Embedding-8B", "input": "hello"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["data"][0]["embedding"], [1, 2, 3])
        urls = [url for _, url, _ in self.http.requests]
        self.assertIn("http://vllm:8888/wake_up", urls)
        self.assertIn("http://vllm:8888/v1/embeddings", urls)

    def test_embedding_request_accepts_upstream_model_id(self) -> None:
        status, payload = self.post_json(
            "/v1/embeddings",
            {"model": "Qwen/Qwen3-Embedding-8B", "input": "hello"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["data"][0]["embedding"], [1, 2, 3])

    def test_chat_completions_not_exposed_until_streaming_is_supported(self) -> None:
        req = Request(
            f"{self.base_url}/v1/chat/completions",
            data=b'{"model":"Qwen3-Embedding-8B","messages":[]}',
            headers={"content-type": "application/json"},
            method="POST",
        )
        with self.assertRaises(HTTPError) as caught:
            urlopen(req, timeout=2)  # noqa: S310 - local test server
        self.assertEqual(caught.exception.code, 404)

    def test_unknown_model_returns_404(self) -> None:
        req = Request(
            f"{self.base_url}/v1/embeddings",
            data=b'{"model":"missing","input":"hello"}',
            headers={"content-type": "application/json"},
            method="POST",
        )
        with self.assertRaises(HTTPError) as caught:
            urlopen(req, timeout=2)  # noqa: S310 - local test server
        self.assertEqual(caught.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
