from __future__ import annotations

import json
import socket
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from vllm_sleeper_proxy.admission import FileThermalAdmissionGuard, ThermalCooldownError
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


class ControlledDrainStream(FakeStream):
    def __init__(self) -> None:
        super().__init__([])
        self.started = threading.Event()
        self.resume = threading.Event()

    def iter_chunks(self, chunk_size=64 * 1024):
        self.started.set()
        yield b"data: first\n\n"
        self.resume.wait(timeout=2)
        yield b"data: [DONE]\n\n"


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

    def test_healthz_is_nonblocking_while_lifecycle_holds_condition(self) -> None:
        readiness_started = threading.Event()
        release_readiness = threading.Event()
        original = self.manager._wait_until_model_listed

        def blocked_readiness(model):
            readiness_started.set()
            release_readiness.wait(timeout=2)
            original(model)

        self.manager._wait_until_model_listed = blocked_readiness
        result = []
        request_thread = threading.Thread(
            target=lambda: result.append(self.post_json(
                "/v1/embeddings",
                {"model": "Qwen3-Embedding-8B", "input": "health"},
            ))
        )
        request_thread.start()
        self.assertTrue(readiness_started.wait(timeout=1))
        started = time.monotonic()
        health = self.get_json("/healthz")
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertTrue(health["ok"])
        self.assertFalse(health["ready"])
        self.assertEqual(health["lifecycle_state"], "starting")
        self.assertEqual(health["inflight_requests"], 0)
        release_readiness.set()
        request_thread.join(timeout=2)
        self.assertFalse(request_thread.is_alive())
        self.assertEqual(result[0][0], 200)

    def test_active_thermal_hold_keeps_control_health_and_fences_inference(self) -> None:
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            status_path = root / "admission.json"
            containment_path = root / "containment.json"
            status_path.write_text(json.dumps({
                "schema_version": 1,
                "generated_at_epoch": time.time(),
                "max_age_seconds": 5,
                "allowed": False,
                "state": "recovering",
                "reason": "thermal_cooldown",
                "reason_codes": ["acpi_temperature_requires_sleep"],
                "retry_after_seconds": 60,
                "sequence": 7,
            }))
            containment_path.write_text(json.dumps({
                "schema_version": 1,
                "active": True,
                "state": "sleep",
                "action_id": "thermal-600-accepted",
            }))
            self.manager.pre_admission_check = FileThermalAdmissionGuard(
                status_path,
                containment_path=containment_path,
            )
            self.http.requests.clear()

            health = self.get_json("/healthz")
            self.assertTrue(health["ok"])
            self.assertTrue(health["startup_finalized"])
            self.assertEqual("finalized", health["startup_state"])
            self.assertTrue(health["lifecycle_ready"])
            self.assertFalse(health["ready"])
            self.assertFalse(health["inference_available"])
            self.assertEqual("sleep", health["thermal_phase"])
            self.assertEqual("thermal-600-accepted", health["thermal_action_id"])

            request = Request(f"{self.base_url}/v1/embeddings", method="POST")
            with self.assertRaises(HTTPError) as caught:
                urlopen(request, timeout=2)
            self.assertEqual(503, caught.exception.code)
            self.assertEqual("60", caught.exception.headers["Retry-After"])
            error = json.loads(caught.exception.read())["error"]
            self.assertEqual("thermal_protection_active", error["code"])
            self.assertEqual("sleep", error["thermal_phase"])
            self.assertEqual("thermal-600-accepted", error["action_id"])
            self.assertEqual([], self.http.requests)

            containment_path.write_text(json.dumps({
                "schema_version": 1,
                "active": True,
                "state": [],
                "action_id": "thermal-corrupt-health",
            }))
            corrupt_status = json.loads(status_path.read_text())
            corrupt_status["generated_at_epoch"] = 10**400
            status_path.write_text(json.dumps(corrupt_status))
            health = self.get_json("/healthz")
            self.assertTrue(health["ok"])
            self.assertFalse(health["inference_available"])
            self.assertEqual("recovering", health["thermal_phase"])
            self.assertEqual("thermal-corrupt-health", health["thermal_action_id"])
            with self.assertRaises(HTTPError) as corrupt:
                urlopen(request, timeout=2)
            corrupt_error = json.loads(corrupt.exception.read())["error"]
            self.assertEqual("thermal_protection_active", corrupt_error["code"])
            self.assertEqual("recovering", corrupt_error["thermal_phase"])
            self.assertEqual("thermal-corrupt-health", corrupt_error["action_id"])

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

    def test_bodyless_sleep_endpoint_remains_compatible(self) -> None:
        self.post_json(
            "/v1/embeddings",
            {"model": "Qwen3-Embedding-8B", "input": "hello"},
        )
        request = Request(f"{self.base_url}/sleep", method="POST")
        with urlopen(request, timeout=2) as response:
            self.assertEqual(200, response.status)
            payload = json.loads(response.read())
        self.assertTrue(payload["ok"])
        self.assertEqual("Qwen3-Embedding-8B", payload["slept_model"])

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

    def test_thermal_cooldown_returns_503_retry_after_without_wake(self) -> None:
        self.http.requests.clear()
        def deny(model) -> None:
            raise ThermalCooldownError(
                "thermal_cooldown: temperatures are recovering; state=warning",
                state="warning",
                retry_after_seconds=60,
            )
        self.manager.pre_admission_check = deny
        self.manager.admission_check = deny
        for path, payload in (
            ("/v1/embeddings", {"model": "Qwen3-Embedding-8B", "input": "hello"}),
            ("/v1/chat/completions", {"model": "Qwen3-Embedding-8B", "messages": []}),
        ):
            with self.subTest(path=path):
                req = Request(
                    f"{self.base_url}{path}",
                    data=json.dumps(payload).encode(),
                    headers={"content-type": "application/json"},
                    method="POST",
                )
                with self.assertRaises(HTTPError) as caught:
                    urlopen(req, timeout=2)
                self.assertEqual(503, caught.exception.code)
                self.assertEqual("60", caught.exception.headers["Retry-After"])
                body = json.loads(caught.exception.read().decode())
                self.assertEqual("service_unavailable", body["error"]["type"])
                self.assertEqual("thermal_protection_active", body["error"]["code"])
                self.assertEqual("warning", body["error"]["thermal_phase"])
                self.assertEqual(60, body["error"]["retry_after_seconds"])
                self.assertEqual(
                    "Inference is temporarily paused for thermal protection. Retry shortly.",
                    body["error"]["message"],
                )
        self.assertEqual([], self.http.requests)
        self.assertEqual(0, self.manager.inflight_requests)

    def test_thermal_cooldown_wins_before_request_body_validation(self) -> None:
        def deny(model) -> None:
            raise ThermalCooldownError(
                "thermal_cooldown: status is stale",
                state="stale",
                retry_after_seconds=10,
            )
        self.manager.pre_admission_check = deny
        self.http.requests.clear()
        for path in ("/v1/embeddings", "/v1/chat/completions"):
            with self.subTest(path=path):
                req = Request(f"{self.base_url}{path}", method="POST")
                with self.assertRaises(HTTPError) as caught:
                    urlopen(req, timeout=2)
                self.assertEqual(503, caught.exception.code)
                self.assertEqual("10", caught.exception.headers["Retry-After"])
                error = json.loads(caught.exception.read())["error"]
                self.assertEqual("service_unavailable", error["type"])
                self.assertEqual("thermal_protection_active", error["code"])
        self.assertEqual([], self.http.requests)

    def test_thermal_cooldown_precedes_malformed_oversized_and_chunked_bodies(self) -> None:
        def deny(model) -> None:
            raise ThermalCooldownError(
                "must not be exposed",
                state="urgent_hold",
                retry_after_seconds=7,
            )

        self.manager.pre_admission_check = deny
        self.http.requests.clear()
        for path, body in (
            ("/v1/embeddings", b"{"),
            (
                "/v1/chat/completions",
                b'{"model":"alias","messages":[],"stream":true}',
            ),
        ):
            with self.subTest(path=path):
                request = Request(
                    f"{self.base_url}{path}",
                    data=body,
                    headers={"content-type": "application/json"},
                    method="POST",
                )
                with self.assertRaises(HTTPError) as caught:
                    urlopen(request, timeout=2)
                self.assertEqual(503, caught.exception.code)
                self.assertEqual("7", caught.exception.headers["Retry-After"])
                self.assertEqual(
                    "thermal_protection_active",
                    json.loads(caught.exception.read())["error"]["code"],
                )

        for extra_headers in (
            b"Content-Length: 999999999\r\n",
            b"Transfer-Encoding: chunked\r\n",
        ):
            with self.subTest(headers=extra_headers):
                client = socket.create_connection(
                    ("127.0.0.1", self.server.server_address[1]),
                    timeout=1,
                )
                client.settimeout(1)
                client.sendall(
                    b"POST /v1/embeddings HTTP/1.1\r\n"
                    b"Host: localhost\r\n"
                    + extra_headers
                    + b"Connection: close\r\n\r\n"
                )
                response = b""
                while True:
                    chunk = client.recv(65536)
                    if not chunk:
                        break
                    response += chunk
                client.close()
                self.assertIn(b" 503 ", response.split(b"\r\n", 1)[0])
                self.assertIn(b"thermal_protection_active", response)
        self.assertEqual([], self.http.requests)
        self.assertEqual(0, self.manager.inflight_requests)

    def test_restart_adoption_refences_from_active_v1_latch(self) -> None:
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            status_path = root / "admission.json"
            containment_path = root / "containment.json"
            status_path.write_text(json.dumps({
                "schema_version": 1,
                "generated_at_epoch": time.time(),
                "max_age_seconds": 5,
                "allowed": False,
                "state": "recovering",
                "reason": "thermal_cooldown",
                "reason_codes": ["thermal_recovery_hold_pending"],
                "retry_after_seconds": 60,
                "sequence": 8,
            }))
            containment_path.write_text(json.dumps({
                "schema_version": 1,
                "active": True,
                "state": "sleep",
                "action_id": "thermal-restart-proof",
            }))
            model = ModelConfig(
                name="Qwen3-Embedding-8B",
                upstream_model="Qwen/Qwen3-Embedding-8B",
                upstream_base_url="http://vllm:8888/v1",
                control_base_url="http://vllm:8888",
                required=True,
            )
            restart_http = ProxyHttp()
            guard = FileThermalAdmissionGuard(
                status_path,
                containment_path=containment_path,
            )
            manager = ModelManager(
                [model],
                restart_http,
                bootstrap_mode=True,
                pre_admission_check=guard,
                admission_check=guard,
                poll_interval_s=0,
                wake_timeout_s=1,
            )
            server = build_server("127.0.0.1", 0, manager)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                inference = Request(
                    f"{base_url}/v1/embeddings",
                    data=b"not-json",
                    method="POST",
                )
                with self.assertRaises(HTTPError) as before_finalize:
                    urlopen(inference, timeout=2)
                before = json.loads(before_finalize.exception.read())["error"]
                self.assertEqual("thermal_protection_active", before["code"])
                self.assertEqual("thermal-restart-proof", before["action_id"])

                adopt = Request(
                    f"{base_url}/startup/adopt",
                    data=b"{}",
                    method="POST",
                )
                with urlopen(adopt, timeout=2) as response:
                    self.assertEqual(200, response.status)
                with urlopen(f"{base_url}/healthz", timeout=2) as response:
                    health = json.loads(response.read())
                self.assertTrue(health["ok"])
                self.assertFalse(health["inference_available"])
                self.assertEqual("thermal-restart-proof", health["thermal_action_id"])

                with self.assertRaises(HTTPError) as after_adopt:
                    urlopen(inference, timeout=2)
                self.assertEqual(
                    "thermal_protection_active",
                    json.loads(after_adopt.exception.read())["error"]["code"],
                )
                self.assertFalse(any("/wake_up" in url for _, url, _ in restart_http.requests))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_missing_fast_thermal_status_is_user_facing_cooldown(self) -> None:
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            self.manager.pre_admission_check = FileThermalAdmissionGuard(
                Path(directory) / "missing.json"
            )
            req = Request(
                f"{self.base_url}/v1/embeddings",
                data=json.dumps({"model": "Qwen3-Embedding-8B", "input": "late"}).encode(),
                headers={"content-type": "application/json"},
                method="POST",
            )
            with self.assertRaises(HTTPError) as caught:
                urlopen(req, timeout=2)
            self.assertEqual(503, caught.exception.code)
            self.assertEqual("10", caught.exception.headers["Retry-After"])
            body = json.loads(caught.exception.read())
            self.assertEqual("service_unavailable", body["error"]["type"])
            self.assertEqual("thermal_protection_active", body["error"]["code"])
            self.assertEqual("unavailable", body["error"]["thermal_phase"])
            (Path(directory) / "missing.json").write_text("[]")
            with self.assertRaises(HTTPError) as malformed:
                urlopen(req, timeout=2)
            self.assertEqual(503, malformed.exception.code)
            self.assertEqual(
                "thermal_protection_active",
                json.loads(malformed.exception.read())["error"]["code"],
            )

    def test_thermal_cooldown_denies_same_model_already_awake(self) -> None:
        self.post_json(
            "/v1/embeddings",
            {"model": "Qwen3-Embedding-8B", "input": "warm"},
        )
        self.http.requests.clear()
        def deny(model) -> None:
            raise ThermalCooldownError(
                "thermal_cooldown: recovery hold active",
                state="recovering",
                retry_after_seconds=60,
            )
        self.manager.pre_admission_check = deny
        req = Request(
            f"{self.base_url}/v1/embeddings",
            data=json.dumps({"model": "Qwen3-Embedding-8B", "input": "late"}).encode(),
            headers={"content-type": "application/json"},
            method="POST",
        )
        with self.assertRaises(HTTPError) as caught:
            urlopen(req, timeout=2)
        self.assertEqual(503, caught.exception.code)
        self.assertEqual("thermal_protection_active", json.loads(caught.exception.read())["error"]["code"])
        self.assertEqual([], self.http.requests)

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

    def test_thermal_sleep_drains_active_stream_and_late_request_gets_503(self) -> None:
        controlled = ControlledDrainStream()
        self.http.stream_factory = lambda: controlled
        stream_result = []
        def run_stream() -> None:
            req = Request(
                f"{self.base_url}/v1/chat/completions",
                data=json.dumps({
                    "model": "Qwen3-Embedding-8B", "messages": [], "stream": True,
                }).encode(),
                headers={"content-type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=3) as response:
                stream_result.append((response.status, response.read()))
        stream_thread = threading.Thread(target=run_stream)
        stream_thread.start()
        self.assertTrue(controlled.started.wait(timeout=1))

        def deny(model) -> None:
            raise ThermalCooldownError(
                "thermal_cooldown: sleep threshold active",
                state="sleep",
                retry_after_seconds=60,
            )
        self.manager.pre_admission_check = deny
        sleep_result = []
        sleep_thread = threading.Thread(
            target=lambda: sleep_result.append(self.post_json("/sleep", {}))
        )
        sleep_thread.start()
        deadline = time.monotonic() + 1
        while not self.manager._quiescing and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertTrue(self.manager._quiescing)
        self.assertTrue(sleep_thread.is_alive())

        late = Request(
            f"{self.base_url}/v1/embeddings",
            data=json.dumps({"model": "Qwen3-Embedding-8B", "input": "late"}).encode(),
            headers={"content-type": "application/json"},
            method="POST",
        )
        started = time.monotonic()
        with self.assertRaises(HTTPError) as caught:
            urlopen(late, timeout=2)
        self.assertEqual(503, caught.exception.code)
        self.assertLess(time.monotonic() - started, 0.5)

        controlled.resume.set()
        stream_thread.join(timeout=2)
        sleep_thread.join(timeout=2)
        self.assertFalse(stream_thread.is_alive())
        self.assertFalse(sleep_thread.is_alive())
        self.assertEqual(200, stream_result[0][0])
        self.assertIn(b"[DONE]", stream_result[0][1])
        self.assertEqual(200, sleep_result[0][0])
        self.assertEqual("Qwen3-Embedding-8B", sleep_result[0][1]["slept_model"])
        wake_count = sum("/wake_up" in url for _, url, _ in self.http.requests)
        with self.assertRaises(HTTPError) as after_sleep:
            urlopen(late, timeout=2)
        self.assertEqual(503, after_sleep.exception.code)
        self.assertEqual(
            wake_count,
            sum("/wake_up" in url for _, url, _ in self.http.requests),
        )

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
        self.assertEqual(self.manager.lifecycle_state, "unknown")
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

    def test_buffered_forward_open_failure_returns_lifecycle_503_and_releases(self) -> None:
        self.http.buffer_error = ConnectionError("vLLM unavailable")
        req = Request(
            f"{self.base_url}/v1/chat/completions",
            data=b'{"model":"Qwen3-Embedding-8B","messages":[]}',
            headers={"content-type": "application/json"},
            method="POST",
        )
        with self.assertRaises(HTTPError) as caught:
            urlopen(req, timeout=2)  # noqa: S310 - local test server
        self.assertEqual(caught.exception.code, 503)
        self.assertEqual(caught.exception.headers["retry-after"], "10")
        payload = json.loads(caught.exception.read().decode())
        self.assertEqual(payload["error"]["type"], "lifecycle_unavailable")
        self.assertNotIn("vLLM unavailable", payload["error"]["message"])
        self.assertEqual(self.manager.lifecycle_state, "unknown")
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
        self.assertEqual(caught.exception.code, 503)
        payload = json.loads(caught.exception.read().decode())
        self.assertEqual(payload["error"]["type"], "lifecycle_unavailable")
        self.assertEqual(self.manager.lifecycle_state, "unknown")
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

    def test_stale_same_model_owner_returns_503_before_upstream_forward(self) -> None:
        status, _ = self.post_json(
            "/v1/embeddings",
            {"model": "Qwen3-Embedding-8B", "input": "first"},
        )
        self.assertEqual(status, 200)
        before_failure = len(self.http.requests)
        self.http.lifecycle_error = ConnectionError("owned engine unavailable")

        request = Request(
            f"{self.base_url}/v1/embeddings",
            data=b'{"model":"Qwen3-Embedding-8B","input":"retry"}',
            headers={"content-type": "application/json"},
            method="POST",
        )
        with self.assertRaises(HTTPError) as caught:
            urlopen(request, timeout=2)  # noqa: S310 - local test server
        self.assertEqual(caught.exception.code, 503)
        self.assertEqual(caught.exception.headers["retry-after"], "10")
        payload = json.loads(caught.exception.read().decode())
        self.assertEqual(payload["error"]["type"], "lifecycle_unavailable")
        self.assertEqual(
            [],
            [
                url
                for _, url, _ in self.http.requests[before_failure:]
                if url.endswith("/v1/embeddings")
            ],
        )
        self.assertEqual(self.manager.inflight_requests, 0)

        health = self.get_json("/healthz")
        self.assertTrue(health["ok"])
        self.assertFalse(health["ready"])
        self.assertEqual(health["lifecycle_state"], "unknown")
        self.assertEqual(health["active_model"], "Qwen3-Embedding-8B")
        self.assertEqual(health["inflight_requests"], 0)

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

    def test_bootstrap_liveness_does_not_probe_absent_upstreams(self) -> None:
        with urlopen(f"{self.base_url}/startup/state", timeout=2) as response:
            state = json.loads(response.read().decode())
        self.assertFalse(state["finalized"])
        self.assertEqual(state["models"], [{"model": "Qwen3-Embedding-8B", "is_sleeping": None}])
        self.assertEqual(self.http.requests, [])

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

    def test_adopt_preserves_one_awake_engine_without_sleeping_it(self) -> None:
        self.http.sleeping = False
        before = len(self.http.requests)
        req = Request(f"{self.base_url}/startup/adopt", data=b"", method="POST")
        with urlopen(req, timeout=2) as response:
            payload = json.loads(response.read().decode())
        self.assertTrue(payload["finalized"])
        self.assertEqual("Qwen3-Embedding-8B", payload["active_model"])
        lifecycle = [url for _, url, _ in self.http.requests[before:]]
        self.assertTrue(any(url.endswith("/is_sleeping") for url in lifecycle))
        self.assertFalse(any("/sleep?" in url for url in lifecycle))
        with urlopen(f"{self.base_url}/healthz", timeout=2) as response:
            health = json.loads(response.read().decode())
        self.assertTrue(health["ready"])
        self.assertEqual(health["lifecycle_state"], "active")

    def test_adopt_rejects_ambiguous_awake_state_without_sleeping(self) -> None:
        second = ModelConfig(
            name="vision-vla",
            upstream_model="Qwen/Qwen3-VL-4B-Instruct",
            upstream_base_url="http://vision:8888/v1",
            control_base_url="http://vision:8888",
        )
        manager = ModelManager(
            [self.manager.models[0], second], self.http,
            poll_interval_s=0, wake_timeout_s=1, bootstrap_mode=True,
        )
        with patch.object(manager, "_is_sleeping", return_value=False), \
             patch.object(manager, "_sleep", side_effect=AssertionError("sleep called")):
            with self.assertRaisesRegex(WakeError, "multiple awake"):
                manager.adopt_startup_state()
        self.assertFalse(manager.startup_finalized)

    def test_adopt_rejects_unknown_state_without_sleeping(self) -> None:
        with patch.object(self.manager, "_is_sleeping", return_value=None), \
             patch.object(self.manager, "_sleep", side_effect=AssertionError("sleep called")):
            with self.assertRaisesRegex(WakeError, "cannot verify"):
                self.manager.adopt_startup_state()
        self.assertFalse(self.manager.startup_finalized)

    def test_finalize_unlocks_inference_and_readiness(self) -> None:
        req = Request(f"{self.base_url}/startup/finalize", data=b"", method="POST")
        with urlopen(req, timeout=2) as response:
            self.assertTrue(json.loads(response.read().decode())["finalized"])
        with urlopen(f"{self.base_url}/healthz", timeout=2) as response:
            health = json.loads(response.read().decode())
        self.assertTrue(health["ready"])
        self.assertEqual(health["lifecycle_state"], "sleeping")
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
