from __future__ import annotations

import json
import socket
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from vllm_sleeper_proxy.client import HttpResponse
from vllm_sleeper_proxy.config import ModelConfig
from vllm_sleeper_proxy.manager import ModelManager, WakeError
from vllm_sleeper_proxy.server import build_server, require_qwen_admission


class AdmissionWiringTests(unittest.TestCase):
    def test_qwen_requires_status_path(self):
        model = ModelConfig(
            name="qwen38-27b-nvfp4",
            upstream_model="unsloth/Qwen3.8-27B-NVFP4",
            upstream_base_url="http://qwen:8000/v1",
            control_base_url="http://qwen:8000",
        )
        with self.assertRaisesRegex(RuntimeError, "SLEEPER_ADMISSION_STATUS_PATH"):
            require_qwen_admission([model], None)
        require_qwen_admission([model], "/run/status.json")


class FakeStream:
    def __init__(self, chunks: list[bytes]) -> None:
        self.status = 200
        self.headers = {"content-type": "text/event-stream"}
        self.chunks = chunks
        self.closed = False

    def iter_chunks(self, chunk_size=64 * 1024):
        yield from self.chunks

    def close(self):
        self.closed = True


class FailingStream(FakeStream):
    def iter_chunks(self, chunk_size=64 * 1024):
        yield b'data: {"choices":[{"delta":{"content":"e2"}}]}\n\n'
        raise TimeoutError("upstream stream timed out")


class ControlledDisconnectStream(FakeStream):
    def __init__(self) -> None:
        super().__init__([])
        self.started = threading.Event()
        self.resume = threading.Event()
        self.finished = threading.Event()

    def iter_chunks(self, chunk_size=64 * 1024):
        self.started.set()
        yield b"data: first\n\n"
        self.resume.wait(timeout=1)
        for _ in range(16):
            yield b"x" * (1024 * 1024)

    def close(self):
        super().close()
        self.finished.set()


class ProxyHttp:
    def __init__(self) -> None:
        self.requests: list[tuple[str, str, bytes | None]] = []
        self.sleeping = True
        self.buffer_error: Exception | None = None
        self.stream_error: Exception | None = None
        self.lifecycle_error: Exception | None = None
        self.last_stream: FakeStream | None = None
        self.stream_factory = None

    def request(self, method, url, *, headers=None, body=None, timeout=None):
        self.requests.append((method, url, body))
        if self.lifecycle_error and (
            url.endswith("/is_sleeping")
            or url.endswith("/wake_up?tags=weights")
            or url.endswith("/v1/models")
        ):
            raise self.lifecycle_error
        if url.endswith("/wake_up") or url.endswith("/wake_up?tags=weights"):
            self.sleeping = False
            return HttpResponse(200, {}, b"{}")
        if url.endswith("/wake_up?tags=kv_cache"):
            self.sleeping = False
            return HttpResponse(200, {}, b"{}")
        if url.endswith("/collective_rpc"):
            return HttpResponse(200, {}, b"{}")
        if url.endswith("/reset_mm_cache"):
            return HttpResponse(200, {}, b"{}")
        if "/sleep?" in url:
            self.sleeping = True
            return HttpResponse(200, {}, b"{}")
        if url.endswith("/is_sleeping"):
            return HttpResponse(
                200,
                {},
                json.dumps({"is_sleeping": self.sleeping}).encode(),
            )
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
        if url.endswith("/v1/chat/completions"):
            if self.buffer_error:
                raise self.buffer_error
            decoded = json.loads((body or b"{}").decode())
            if decoded["model"] != "Qwen/Qwen3-Embedding-8B":
                return HttpResponse(400, {"content-type": "application/json"}, b'{"bad":"model"}')
            return HttpResponse(
                200,
                {"content-type": "application/json", "x-upstream": "vllm"},
                b'{"choices":[{"message":{"content":"e2e4"}}]}',
            )
        raise AssertionError(f"unexpected request {method} {url}")

    def stream(self, method, url, *, headers=None, body=None, timeout=None):
        self.requests.append((method, url, body))
        if self.stream_error:
            raise self.stream_error
        if not url.endswith("/v1/chat/completions"):
            raise AssertionError(f"unexpected streaming request {method} {url}")
        decoded = json.loads((body or b"{}").decode())
        if decoded["model"] != "Qwen/Qwen3-Embedding-8B":
            raise AssertionError("logical model was not rewritten")
        if self.stream_factory is not None:
            self.last_stream = self.stream_factory()
            return self.last_stream
        self.last_stream = FakeStream(
            [
                b'data: {"choices":[{"delta":{"content":"e2"}}]}\n\n',
                b'data: {"choices":[{"delta":{"content":"e4"}}]}\n\n',
                b"data: [DONE]\n\n",
            ]
        )
        return self.last_stream


class ServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.http = ProxyHttp()
        model = ModelConfig(
            name="Qwen3-Embedding-8B",
            upstream_model="Qwen/Qwen3-Embedding-8B",
            upstream_base_url="http://vllm:8888/v1",
            control_base_url="http://vllm:8888",
        )
        self.manager = ModelManager([model], self.http, poll_interval_s=0, wake_timeout_s=1)
        self.server = build_server("127.0.0.1", 0, self.manager)
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

    def assert_request_ownership_released(self) -> None:
        deadline = time.monotonic() + 1
        while self.manager.inflight_requests != 0 and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(self.manager.inflight_requests, 0)

    def test_v1_models(self) -> None:
        payload = self.get_json("/v1/models")
        self.assertEqual(payload["data"][0]["id"], "Qwen3-Embedding-8B")

    def test_sleep_endpoint_quiesces_active_model(self) -> None:
        self.post_json(
            "/v1/embeddings",
            {"model": "Qwen3-Embedding-8B", "input": "hello"},
        )
        status, payload = self.post_json("/sleep", {})
        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["slept_model"], "Qwen3-Embedding-8B")
        self.assertIsNone(self.manager.active_model_name)

    def test_server_does_not_bind_when_startup_reconciliation_fails(self) -> None:
        class FailedStartupManager(ModelManager):
            def reconcile_startup_state(self) -> None:
                raise WakeError("vision sleep state unavailable")

        model = ModelConfig(
            name="Qwen3-Embedding-8B",
            upstream_model="Qwen/Qwen3-Embedding-8B",
            upstream_base_url="http://vllm:8888/v1",
            control_base_url="http://vllm:8888",
        )
        manager = FailedStartupManager([model], ProxyHttp())
        with self.assertRaisesRegex(WakeError, "vision sleep state unavailable"):
            build_server("127.0.0.1", 0, manager)

    def test_embedding_request_wakes_and_forwards(self) -> None:
        status, payload = self.post_json(
            "/v1/embeddings",
            {"model": "Qwen3-Embedding-8B", "input": "hello"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["data"][0]["embedding"], [1, 2, 3])
        urls = [url for _, url, _ in self.http.requests]
        self.assertIn("http://vllm:8888/wake_up?tags=weights", urls)
        self.assertIn("http://vllm:8888/collective_rpc", urls)
        self.assertIn("http://vllm:8888/wake_up?tags=kv_cache", urls)
        self.assertIn("http://vllm:8888/reset_mm_cache", urls)
        self.assertIn("http://vllm:8888/v1/embeddings", urls)

    def test_embedding_request_accepts_upstream_model_id(self) -> None:
        status, payload = self.post_json(
            "/v1/embeddings",
            {"model": "Qwen/Qwen3-Embedding-8B", "input": "hello"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["data"][0]["embedding"], [1, 2, 3])

    def test_non_streaming_chat_preserves_multimodal_messages(self) -> None:
        payload = {
            "model": "Qwen3-Embedding-8B",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What move occurred?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,AAAA"},
                        },
                    ],
                }
            ],
            "temperature": 0,
        }
        status, response = self.post_json("/v1/chat/completions", payload)
        self.assertEqual(status, 200)
        self.assertEqual(response["choices"][0]["message"]["content"], "e2e4")
        forwarded = json.loads(self.http.requests[-1][2].decode())
        self.assertEqual(forwarded["messages"], payload["messages"])
        self.assertEqual(forwarded["model"], "Qwen/Qwen3-Embedding-8B")

    def test_streaming_chat_preserves_sse_framing(self) -> None:
        req = Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": "Qwen3-Embedding-8B",
                    "messages": [{"role": "user", "content": "move?"}],
                    "stream": True,
                }
            ).encode(),
            headers={"content-type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=2) as response:  # noqa: S310 - local test server
            body = response.read()
            self.assertEqual(response.headers["content-type"], "text/event-stream")
        self.assertEqual(body.count(b"\n\n"), 3)
        self.assertTrue(body.endswith(b"data: [DONE]\n\n"))
        self.assertTrue(self.http.last_stream.closed)
        self.assert_request_ownership_released()

    def test_midstream_timeout_closes_upstream_and_releases(self) -> None:
        self.http.stream_factory = lambda: FailingStream([])
        req = Request(
            f"{self.base_url}/v1/chat/completions",
            data=b'{"model":"Qwen3-Embedding-8B","messages":[],"stream":true}',
            headers={"content-type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=2) as response:  # noqa: S310 - local test server
            self.assertIn(b"data:", response.read())
        self.assertTrue(self.http.last_stream.closed)
        self.assert_request_ownership_released()

    def test_downstream_disconnect_closes_upstream_and_releases(self) -> None:
        controlled = ControlledDisconnectStream()
        self.http.stream_factory = lambda: controlled
        body = b'{"model":"Qwen3-Embedding-8B","messages":[],"stream":true}'
        client = socket.create_connection(
            ("127.0.0.1", self.server.server_address[1]),
            timeout=2,
        )
        client.sendall(
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            b"Host: localhost\r\n"
            b"content-type: application/json\r\n"
            + f"content-length: {len(body)}\r\n".encode()
            + b"connection: close\r\n\r\n"
            + body
        )
        self.assertTrue(controlled.started.wait(timeout=1))
        client.close()
        controlled.resume.set()
        self.assertTrue(controlled.finished.wait(timeout=2))
        self.assertTrue(controlled.closed)
        self.assert_request_ownership_released()

    def test_buffered_upstream_failure_returns_stable_error_and_releases(self) -> None:
        self.http.buffer_error = ConnectionError("vLLM unavailable")
        req = Request(
            f"{self.base_url}/v1/chat/completions",
            data=b'{"model":"Qwen3-Embedding-8B","messages":[]}',
            headers={"content-type": "application/json"},
            method="POST",
        )
        with self.assertRaises(HTTPError) as caught:
            urlopen(req, timeout=2)  # noqa: S310 - local test server
        self.assertEqual(caught.exception.code, 502)
        payload = json.loads(caught.exception.read().decode())
        self.assertEqual(payload["error"]["type"], "upstream_unavailable")
        self.assert_request_ownership_released()

    def test_stream_open_failure_returns_stable_error_and_releases(self) -> None:
        self.http.stream_error = TimeoutError("vLLM timed out")
        req = Request(
            f"{self.base_url}/v1/chat/completions",
            data=b'{"model":"Qwen3-Embedding-8B","messages":[],"stream":true}',
            headers={"content-type": "application/json"},
            method="POST",
        )
        with self.assertRaises(HTTPError) as caught:
            urlopen(req, timeout=2)  # noqa: S310 - local test server
        self.assertEqual(caught.exception.code, 502)
        caught.exception.read()
        self.assert_request_ownership_released()

    def test_lifecycle_transport_failure_returns_wake_error(self) -> None:
        self.http.lifecycle_error = ConnectionError("control endpoint unavailable")
        req = Request(
            f"{self.base_url}/v1/chat/completions",
            data=b'{"model":"Qwen3-Embedding-8B","messages":[]}',
            headers={"content-type": "application/json"},
            method="POST",
        )
        with self.assertRaises(HTTPError) as caught:
            urlopen(req, timeout=2)  # noqa: S310 - local test server
        self.assertEqual(caught.exception.code, 503)
        payload = json.loads(caught.exception.read().decode())
        self.assertEqual(payload["error"]["type"], "wake_failed")

    def test_invalid_json_returns_stable_error(self) -> None:
        req = Request(
            f"{self.base_url}/v1/chat/completions",
            data=b"not json",
            headers={"content-type": "application/json"},
            method="POST",
        )
        with self.assertRaises(HTTPError) as caught:
            urlopen(req, timeout=2)  # noqa: S310 - local test server
        self.assertEqual(caught.exception.code, 400)
        payload = json.loads(caught.exception.read().decode())
        self.assertEqual(payload["error"]["type"], "invalid_request")

    def test_request_body_limit_returns_413(self) -> None:
        self.server.RequestHandlerClass.max_request_body_bytes = 4
        req = Request(
            f"{self.base_url}/v1/chat/completions",
            data=b"12345",
            headers={"content-type": "application/json"},
            method="POST",
        )
        with self.assertRaises(HTTPError) as caught:
            urlopen(req, timeout=2)  # noqa: S310 - local test server
        self.assertEqual(caught.exception.code, 413)

    def test_chunked_request_body_is_rejected_explicitly(self) -> None:
        with socket.create_connection(
            ("127.0.0.1", self.server.server_address[1]),
            timeout=2,
        ) as client:
            client.sendall(
                b"POST /v1/chat/completions HTTP/1.1\r\n"
                b"Host: localhost\r\n"
                b"content-type: application/json\r\n"
                b"transfer-encoding: chunked\r\n"
                b"connection: close\r\n\r\n"
            )
            response = b""
            while chunk := client.recv(4096):
                response += chunk
        self.assertIn(b" 411 ", response.split(b"\r\n", 1)[0])
        self.assertIn(b'"type":"length_required"', response)

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


class BootstrapServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.http = ProxyHttp()
        model = ModelConfig(
            name="Qwen3-Embedding-8B",
            upstream_model="Qwen/Qwen3-Embedding-8B",
            upstream_base_url="http://vllm:8888/v1",
            control_base_url="http://vllm:8888",
        )
        self.manager = ModelManager(
            [model], self.http, poll_interval_s=0, wake_timeout_s=1, bootstrap_mode=True
        )
        self.server = build_server("127.0.0.1", 0, self.manager)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_bootstrap_state_and_explicit_sleep(self) -> None:
        with urlopen(f"{self.base_url}/startup/state?model=Qwen3-Embedding-8B", timeout=2) as response:
            state = json.loads(response.read().decode())
        self.assertFalse(state["finalized"])
        self.assertTrue(state["models"][0]["is_sleeping"])
        req = Request(
            f"{self.base_url}/startup/sleep?model=Qwen3-Embedding-8B",
            data=b"",
            method="POST",
        )
        with urlopen(req, timeout=2) as response:
            payload = json.loads(response.read().decode())
        self.assertEqual(payload["slept_model"], "Qwen3-Embedding-8B")

    def test_inference_is_denied_until_finalize(self) -> None:
        req = Request(
            f"{self.base_url}/v1/embeddings",
            data=b'{"model":"Qwen3-Embedding-8B","input":"hello"}',
            headers={"content-type": "application/json"},
            method="POST",
        )
        with self.assertRaises(HTTPError) as caught:
            urlopen(req, timeout=2)
        self.assertEqual(caught.exception.code, 503)
        self.assertEqual(json.loads(caught.exception.read())["error"]["type"], "startup_not_finalized")

    def test_finalize_unlocks_inference_and_readiness(self) -> None:
        req = Request(f"{self.base_url}/startup/finalize", data=b"", method="POST")
        with urlopen(req, timeout=2) as response:
            self.assertTrue(json.loads(response.read().decode())["finalized"])
        with urlopen(f"{self.base_url}/healthz", timeout=2) as response:
            self.assertTrue(json.loads(response.read().decode())["ready"])
        req = Request(
            f"{self.base_url}/v1/embeddings",
            data=b'{"model":"Qwen3-Embedding-8B","input":"hello"}',
            headers={"content-type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=2) as response:
            self.assertEqual(response.status, 200)


if __name__ == "__main__":
    unittest.main()
