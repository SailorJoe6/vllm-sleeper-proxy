from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
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
    "/v1/embeddings",
}


class SleeperProxyHandler(BaseHTTPRequestHandler):
    manager: ModelManager

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

        body = self.rfile.read(int(self.headers.get("content-length", "0") or "0"))
        requested_model = self._extract_model(body)
        if not requested_model:
            self._send_json(400, {"error": {"message": "request JSON must include a model field"}})
            return

        try:
            target = self.manager.ensure_awake(requested_model)
        except UnknownModelError as exc:
            self._send_json(404, {"error": {"message": str(exc), "type": "unknown_model"}})
            return
        except WakeError as exc:
            self.send_response(503)
            self.send_header("content-type", "application/json")
            self.send_header("retry-after", "10")
            self.end_headers()
            self.wfile.write(json.dumps({"error": {"message": str(exc), "type": "wake_failed"}}).encode())
            return

        upstream_body = self.manager.rewrite_request_body(body, target)
        upstream_url = f"{target.upstream_base_url}{path.removeprefix('/v1')}"
        headers = self._forward_headers(extra_content_length=len(upstream_body))
        response = self.manager.http.request(
            "POST",
            upstream_url,
            headers=headers,
            body=upstream_body,
            timeout=self.manager.request_timeout_s,
        )
        self.send_response(response.status)
        for key, value in response.headers.items():
            if key.lower() not in HOP_BY_HOP_HEADERS:
                self.send_header(key, value)
        self.send_header("content-length", str(len(response.body)))
        self.end_headers()
        self.wfile.write(response.body)

    def _forward_headers(self, *, extra_content_length: int) -> dict[str, str]:
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in HOP_BY_HOP_HEADERS
        }
        headers["content-length"] = str(extra_content_length)
        headers.setdefault("content-type", "application/json")
        return headers

    def _extract_model(self, body: bytes) -> str | None:
        try:
            decoded = json.loads(body.decode("utf-8"))
        except Exception:
            return None
        if isinstance(decoded, dict) and isinstance(decoded.get("model"), str):
            return decoded["model"]
        return None

    def _send_json(self, status: int, payload: object) -> None:
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def build_server(host: str, port: int, manager: ModelManager) -> ThreadingHTTPServer:
    class Handler(SleeperProxyHandler):
        pass

    Handler.manager = manager
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
        poll_interval_s=float(os.environ.get("SLEEPER_POLL_INTERVAL_SECONDS", "1")),
        always_wake=os.environ.get("SLEEPER_ALWAYS_WAKE", "1") != "0",
    )
    httpd = build_server(host, port, manager)
    print(f"vllm-sleeper-proxy listening on http://{host}:{port}", flush=True)
    httpd.serve_forever()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
