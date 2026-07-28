from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import socket
from typing import Iterable
from urllib.parse import urlsplit

from .client import UrllibHttpClient
from .config import load_models_from_env
from .manager import ModelManager, UnknownModelError, WakeError

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}

PROXIED_POST_PATHS = {
    "/v1/chat/completions",
    "/v1/embeddings",
}


class SleeperProxyHandler(BaseHTTPRequestHandler):
    manager: ModelManager
    max_request_body_bytes = 10 * 1024 * 1024

    server_version = "vllm-sleeper-proxy/0.1"

    def log_message(self, fmt: str, *args) -> None:  # pragma: no cover - stdlib hook
        if os.environ.get("SLEEPER_PROXY_QUIET", "0") == "1":
            return
        super().log_message(fmt, *args)

    def do_GET(self) -> None:  # noqa: N802 - stdlib API
        path = urlsplit(self.path).path
        if path == "/healthz":
            self._send_json(200, {"ok": True, "active_model": self.manager.active_model_name})
        elif path == "/v1/models":
            self._send_json(200, self.manager.list_openai_models())
        elif path == "/api/tags":
            self._send_json(200, self.manager.list_ollama_tags())
        else:
            self._send_json(404, {"error": {"message": f"not found: {path}"}})

    def do_POST(self) -> None:  # noqa: N802 - stdlib API
        path = urlsplit(self.path).path
        if path not in PROXIED_POST_PATHS:
            self._send_json(404, {"error": {"message": f"not found: {path}"}})
            return

        if self.headers.get("transfer-encoding") is not None:
            self._send_error(411, "chunked request bodies are not supported", "length_required")
            return
        raw_content_length = self.headers.get("content-length")
        if raw_content_length is None:
            self._send_error(411, "content-length is required", "length_required")
            return
        try:
            content_length = int(raw_content_length)
        except ValueError:
            self._send_error(400, "content-length must be an integer", "invalid_request")
            return
        if content_length < 0:
            self._send_error(400, "content-length must not be negative", "invalid_request")
            return
        if content_length > self.max_request_body_bytes:
            self._send_error(
                413,
                f"request body exceeds {self.max_request_body_bytes} bytes",
                "request_too_large",
            )
            return
        body = self.rfile.read(content_length)
        if len(body) != content_length:
            self._send_error(400, "request body ended before content-length", "truncated_request")
            return
        payload = self._decode_payload(body)
        if payload is None:
            self._send_error(400, "request body must be a JSON object", "invalid_request")
            return
        requested_model = payload.get("model")
        if not isinstance(requested_model, str) or not requested_model:
            self._send_error(400, "request JSON must include a model field", "invalid_request")
            return
        stream = path == "/v1/chat/completions" and payload.get("stream") is True

        try:
            lease = self.manager.acquire(requested_model)
        except UnknownModelError as exc:
            self._send_error(404, str(exc), "unknown_model")
            return
        except WakeError as exc:
            self._send_error(503, str(exc), "wake_failed", {"retry-after": "10"})
            return

        with lease as target:
            upstream_body = self.manager.rewrite_request_body(body, target)
            upstream_url = f"{target.upstream_base_url}{path.removeprefix('/v1')}"
            headers = self._forward_headers(extra_content_length=len(upstream_body))
            if stream:
                self._forward_stream(upstream_url, headers, upstream_body)
            else:
                self._forward_buffered(upstream_url, headers, upstream_body)

    def _forward_buffered(
        self,
        upstream_url: str,
        headers: dict[str, str],
        upstream_body: bytes,
    ) -> None:
        try:
            response = self.manager.http.request(
                "POST",
                upstream_url,
                headers=headers,
                body=upstream_body,
                timeout=self.manager.request_timeout_s,
            )
        except (ConnectionError, TimeoutError, socket.timeout) as exc:
            self._send_error(502, str(exc), "upstream_unavailable")
            return
        self.send_response(response.status)
        for key, value in response.headers.items():
            if key.lower() not in HOP_BY_HOP_HEADERS:
                self.send_header(key, value)
        self.send_header("content-length", str(len(response.body)))
        self.end_headers()
        self.wfile.write(response.body)

    def _forward_stream(
        self,
        upstream_url: str,
        headers: dict[str, str],
        upstream_body: bytes,
    ) -> None:
        try:
            response = self.manager.http.stream(
                "POST",
                upstream_url,
                headers=headers,
                body=upstream_body,
                timeout=self.manager.request_timeout_s,
            )
        except (ConnectionError, TimeoutError, socket.timeout) as exc:
            self._send_error(502, str(exc), "upstream_unavailable")
            return

        try:
            self.send_response(response.status)
            for key, value in response.headers.items():
                if key.lower() not in HOP_BY_HOP_HEADERS:
                    self.send_header(key, value)
            self.send_header("connection", "close")
            self.end_headers()
            for chunk in response.iter_chunks():
                self.wfile.write(chunk)
                self.wfile.flush()
        except (
            BrokenPipeError,
            ConnectionError,
            ConnectionResetError,
            TimeoutError,
            socket.timeout,
        ):
            self.close_connection = True
        finally:
            response.close()

    def _forward_headers(self, *, extra_content_length: int) -> dict[str, str]:
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in HOP_BY_HOP_HEADERS
        }
        headers["content-length"] = str(extra_content_length)
        headers.setdefault("content-type", "application/json")
        return headers

    def _decode_payload(self, body: bytes) -> dict[str, object] | None:
        try:
            decoded = json.loads(body.decode("utf-8"))
        except Exception:
            return None
        return decoded if isinstance(decoded, dict) else None

    def _send_error(
        self,
        status: int,
        message: str,
        error_type: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        encoded = json.dumps(
            {"error": {"message": message, "type": error_type}},
            separators=(",", ":"),
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(self, status: int, payload: object) -> None:
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def build_server(
    host: str,
    port: int,
    manager: ModelManager,
    *,
    max_request_body_bytes: int = 10 * 1024 * 1024,
) -> ThreadingHTTPServer:
    class Handler(SleeperProxyHandler):
        pass

    Handler.manager = manager
    Handler.max_request_body_bytes = max_request_body_bytes
    return ThreadingHTTPServer((host, port), Handler)


def main(argv: Iterable[str] | None = None) -> int:
    host = os.environ.get("SLEEPER_PROXY_HOST", "0.0.0.0")
    port = int(os.environ.get("SLEEPER_PROXY_PORT", "8889"))
    manager = ModelManager(
        load_models_from_env(),
        UrllibHttpClient(),
        sleep_level=int(os.environ.get("SLEEPER_SLEEP_LEVEL", "2")),
        request_timeout_s=float(os.environ.get("SLEEPER_REQUEST_TIMEOUT_SECONDS", "30")),
        wake_timeout_s=float(os.environ.get("SLEEPER_WAKE_TIMEOUT_SECONDS", "300")),
        drain_timeout_s=float(os.environ.get("SLEEPER_DRAIN_TIMEOUT_SECONDS", "300")),
        poll_interval_s=float(os.environ.get("SLEEPER_POLL_INTERVAL_SECONDS", "1")),
        always_wake=os.environ.get("SLEEPER_ALWAYS_WAKE", "1") != "0",
        wake_strategy=os.environ.get("SLEEPER_WAKE_STRATEGY", "level2"),
    )
    httpd = build_server(
        host,
        port,
        manager,
        max_request_body_bytes=int(
            os.environ.get("SLEEPER_MAX_REQUEST_BODY_BYTES", str(10 * 1024 * 1024))
        ),
    )
    print(f"vllm-sleeper-proxy listening on http://{host}:{port}", flush=True)
    httpd.serve_forever()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
