from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import socket
from pathlib import Path
from typing import Iterable
from urllib.parse import urlsplit

from .client import UrllibHttpClient
from .admission import (
    AdmissionError,
    FileAdmissionGuard,
    FileThermalAdmissionGuard,
    ThermalCooldownError,
)
from .thermal_control import (
    FileThermalActionAuthority,
    ThermalActionControlError,
)
from .config import load_models_from_env
from .manager import (
    LifecycleUnavailableError,
    ModelManager,
    UnknownModelError,
    WakeError,
)

def require_qwen_admission(models, admission_path: str | None) -> None:
    if any(model.upstream_model == "unsloth/Qwen3.8-27B-NVFP4" for model in models) and not admission_path:
        raise RuntimeError("Qwen3.8 requires SLEEPER_ADMISSION_STATUS_PATH")


def compose_admission_checks(*checks):
    configured = [check for check in checks if check is not None]
    if not configured:
        return None

    def check(model) -> None:
        for item in configured:
            item(model)

    return check


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
    thermal_action_control_enabled = False
    thermal_action_authority: FileThermalActionAuthority | None = None
    thermal_action_max_body_bytes = 2048

    server_version = "vllm-sleeper-proxy/0.1"

    def log_message(self, fmt: str, *args) -> None:  # pragma: no cover - stdlib hook
        if os.environ.get("SLEEPER_PROXY_QUIET", "0") == "1":
            return
        super().log_message(fmt, *args)

    def do_GET(self) -> None:  # noqa: N802 - stdlib API
        path = urlsplit(self.path).path
        if path == "/startup/state" and self.manager.bootstrap_mode:
            try:
                model = urlsplit(self.path).query
                from urllib.parse import parse_qs
                requested = parse_qs(model).get("model", [None])[0]
                self._send_json(200, self.manager.startup_state(requested))
            except (UnknownModelError, WakeError) as exc:
                self._send_error(503, str(exc), "startup_unavailable")
            return
        if path == "/healthz":
            finalized = self.manager.startup_finalized
            lifecycle_ready = self.manager.inference_ready
            thermal = self.manager.thermal_admission_snapshot()
            thermal_fenced = bool(thermal is not None and thermal.fenced)
            inference_available = lifecycle_ready and not thermal_fenced
            self._send_json(200 if finalized else 503, {
                "ok": finalized,
                "control_healthy": finalized,
                "startup_finalized": finalized,
                "startup_state": "finalized" if finalized else "initializing",
                "ready": inference_available,
                "inference_available": inference_available,
                "lifecycle_ready": lifecycle_ready,
                "lifecycle_state": self.manager.lifecycle_state,
                "thermal_phase": thermal.phase if thermal is not None else "unconfigured",
                "thermal_action_id": thermal.action_id if thermal is not None else None,
                "active_model": self.manager.active_model_name,
                "starting_model": self.manager.starting_model_name,
                "inflight_requests": self.manager.inflight_requests,
            })
        elif path == "/v1/models":
            self._send_json(200, self.manager.list_openai_models())
        elif path == "/api/tags":
            self._send_json(200, self.manager.list_ollama_tags())
        else:
            self._send_json(404, {"error": {"message": f"not found: {path}"}})

    def do_POST(self) -> None:  # noqa: N802 - stdlib API
        path = urlsplit(self.path).path
        if path in {"/thermal/actions/hold", "/thermal/actions/release"}:
            self._handle_thermal_action(path.rsplit("/", 1)[-1])
            return
        if self.manager.bootstrap_mode and path == "/startup/sleep":
            from urllib.parse import parse_qs
            requested = parse_qs(urlsplit(self.path).query).get("model", [None])[0]
            if not requested:
                self._send_error(400, "startup sleep requires model", "invalid_request")
                return
            try:
                slept = self.manager.startup_sleep_model(requested)
                self._send_json(200, {"ok": True, "slept_model": slept})
            except (UnknownModelError, WakeError) as exc:
                self._send_error(503, str(exc), "startup_sleep_failed")
            return
        if self.manager.bootstrap_mode and path == "/startup/adopt":
            try:
                self.manager.adopt_startup_state()
                self._send_json(200, {
                    "ok": True,
                    "finalized": True,
                    "active_model": self.manager.active_model_name,
                })
            except WakeError as exc:
                self._send_error(503, str(exc), "startup_adopt_failed")
            return
        if self.manager.bootstrap_mode and path == "/startup/finalize":
            try:
                self.manager.finalize_startup()
                self._send_json(200, {"ok": True, "finalized": True})
            except WakeError as exc:
                self._send_error(503, str(exc), "startup_finalize_failed")
            return
        if path == "/sleep":
            try:
                slept_model = self.manager.sleep_active_model()
            except WakeError as exc:
                self._send_error(503, str(exc), "sleep_failed", {"retry-after": "10"})
                return
            self._send_json(200, {"ok": True, "slept_model": slept_model})
            return
        if path in PROXIED_POST_PATHS:
            try:
                self.manager.check_fast_admission()
            except ThermalCooldownError as exc:
                self._send_thermal_cooldown(exc)
                return
        if self.manager.bootstrap_mode and not self.manager.startup_finalized and path in PROXIED_POST_PATHS:
            self._send_error(503, "required lineup startup is not finalized", "startup_not_finalized")
            return
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
        except LifecycleUnavailableError as exc:
            self._send_error(
                503,
                str(exc),
                "lifecycle_unavailable",
                {"retry-after": "10"},
            )
            return
        except WakeError as exc:
            self._send_error(503, str(exc), "wake_failed", {"retry-after": "10"})
            return
        except ThermalCooldownError as exc:
            self._send_thermal_cooldown(exc)
            return
        except AdmissionError as exc:
            self._send_error(503, str(exc), "admission_denied", {"Retry-After": "10"})
            return

        with lease as target:
            upstream_body = self.manager.rewrite_request_body(body, target)
            upstream_url = f"{target.upstream_base_url}{path.removeprefix('/v1')}"
            headers = self._forward_headers(extra_content_length=len(upstream_body))
            if stream:
                self._forward_stream(target, upstream_url, headers, upstream_body)
            else:
                self._forward_buffered(target, upstream_url, headers, upstream_body)

    def _send_lifecycle_unavailable(self, target) -> None:
        self.manager.mark_owner_unavailable(target)
        self._send_error(
            503,
            f"lifecycle state unavailable for {target.name}; retry later",
            "lifecycle_unavailable",
            {"retry-after": "10"},
        )

    def _forward_buffered(
        self,
        target,
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
        except (ConnectionError, TimeoutError, socket.timeout):
            self._send_lifecycle_unavailable(target)
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
        target,
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
        except (ConnectionError, TimeoutError, socket.timeout):
            self._send_lifecycle_unavailable(target)
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
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except (ConnectionError, TimeoutError, socket.timeout):
            self.manager.mark_owner_unavailable(target)
            self.close_connection = True
        finally:
            response.close()

    def _handle_thermal_action(self, operation: str) -> None:
        authority = self.thermal_action_authority
        if not self.thermal_action_control_enabled or authority is None:
            self._send_json(404, {"error": {"message": "not found"}})
            return
        if self.headers.get("transfer-encoding") is not None:
            self._send_error(411, "content-length is required", "length_required")
            return
        raw_length = self.headers.get("content-length")
        try:
            content_length = int(raw_length) if raw_length is not None else -1
        except ValueError:
            content_length = -1
        if content_length <= 0 or content_length > self.thermal_action_max_body_bytes:
            self._send_error(400, "invalid thermal action body", "invalid_request")
            return
        try:
            body = self.rfile.read(content_length)
            payload = json.loads(body.decode("utf-8"))
        except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
            self._send_error(400, "invalid thermal action body", "invalid_request")
            return
        try:
            action = authority.authorize(operation, payload)
        except ThermalActionControlError as exc:
            if exc.code == "invalid_request":
                self._send_error(
                    400,
                    "invalid thermal action request",
                    "invalid_request",
                    error_fields={"code": "thermal_action_invalid"},
                )
            else:
                self._send_error(
                    409,
                    "thermal action does not match root authority",
                    "conflict",
                    error_fields={"code": "thermal_action_mismatch"},
                )
            return

        def reauthorize() -> None:
            refreshed = authority.authorize(operation, payload)
            if refreshed != action:
                raise ThermalActionControlError("mismatch")

        try:
            if operation == "hold":
                proof = self.manager.thermal_hold(
                    action, reauthorize=reauthorize
                )
                response = {
                    "schema_version": 1,
                    "ok": True,
                    "operation": "hold",
                    "action_id": action.action_id,
                    "root_phase": action.phase,
                    **proof,
                }
            else:
                proof = self.manager.thermal_release_ready(
                    action, reauthorize=reauthorize
                )
                response = {
                    "schema_version": 1,
                    "ok": True,
                    "operation": "release",
                    "action_id": action.action_id,
                    "root_phase": action.phase,
                    **proof,
                }
        except ThermalActionControlError:
            self._send_error(
                409,
                "thermal action changed during lifecycle work",
                "conflict",
                error_fields={"code": "thermal_action_mismatch"},
            )
            return
        except (WakeError, OSError, TimeoutError, socket.timeout):
            self._send_error(
                503,
                "thermal lifecycle proof is unavailable",
                "service_unavailable",
                {"Retry-After": "1"},
                {"code": "thermal_action_unavailable"},
            )
            return
        try:
            # Bind success to the same root action, phase, creation time, and
            # immutable deadlines after all manager cleanup has completed.
            reauthorize()
        except ThermalActionControlError:
            self._send_error(
                409,
                "thermal action changed before proof return",
                "conflict",
                error_fields={"code": "thermal_action_mismatch"},
            )
            return
        self._send_json(200, response)

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

    def _send_thermal_cooldown(self, exc: ThermalCooldownError) -> None:
        fields: dict[str, object] = {
            "code": "thermal_protection_active",
            "thermal_phase": exc.phase,
            "retry_after_seconds": exc.retry_after_seconds,
        }
        if exc.action_id is not None:
            fields["action_id"] = exc.action_id
        self._send_error(
            503,
            "Inference is temporarily paused for thermal protection. Retry shortly.",
            "service_unavailable",
            {"Retry-After": str(exc.retry_after_seconds)},
            fields,
        )

    def _send_error(
        self,
        status: int,
        message: str,
        error_type: str,
        headers: dict[str, str] | None = None,
        error_fields: dict[str, object] | None = None,
    ) -> None:
        error = {"message": message, "type": error_type}
        error.update(error_fields or {})
        encoded = json.dumps(
            {"error": error},
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
    thermal_action_control_enabled: bool = False,
    thermal_action_authority: FileThermalActionAuthority | None = None,
) -> ThreadingHTTPServer:
    # Reconcile before ThreadingHTTPServer binds its listening socket. If any
    # engine cannot be verified asleep, startup fails closed and no request can
    # observe an incorrect active_model=None state.
    if not manager.bootstrap_mode:
        manager.reconcile_startup_state()

    class Handler(SleeperProxyHandler):
        pass

    Handler.manager = manager
    Handler.max_request_body_bytes = max_request_body_bytes
    Handler.thermal_action_control_enabled = thermal_action_control_enabled
    Handler.thermal_action_authority = thermal_action_authority
    return ThreadingHTTPServer((host, port), Handler)


def main(argv: Iterable[str] | None = None) -> int:
    host = os.environ.get("SLEEPER_PROXY_HOST", "0.0.0.0")
    port = int(os.environ.get("SLEEPER_PROXY_PORT", "8889"))
    admission_path = os.environ.get("SLEEPER_ADMISSION_STATUS_PATH")
    thermal_status_path = os.environ.get("SLEEPER_THERMAL_ADMISSION_STATUS_PATH")
    thermal_containment_path = os.environ.get(
        "SLEEPER_THERMAL_CONTAINMENT_STATUS_PATH"
    )
    thermal_action_enabled = (
        os.environ.get("SLEEPER_THERMAL_ACTION_CONTROL_ENABLED", "0") == "1"
    )
    thermal_action_path = os.environ.get("SLEEPER_THERMAL_ACTION_STATUS_PATH")
    if thermal_action_enabled and not thermal_action_path:
        raise RuntimeError(
            "thermal action control requires SLEEPER_THERMAL_ACTION_STATUS_PATH"
        )
    thermal_action_authority = (
        FileThermalActionAuthority(Path(thermal_action_path))
        if thermal_action_enabled and thermal_action_path
        else None
    )
    models = load_models_from_env()
    require_qwen_admission(models, admission_path)
    resource_admission = (
        FileAdmissionGuard(Path(admission_path)) if admission_path else None
    )
    thermal_admission = (
        FileThermalAdmissionGuard(
            Path(thermal_status_path),
            containment_path=(
                Path(thermal_containment_path)
                if thermal_containment_path
                else None
            ),
            maximum_age_seconds=float(
                os.environ.get("SLEEPER_THERMAL_MAX_AGE_SECONDS", "5")
            ),
        )
        if thermal_status_path
        else None
    )
    admission_check = compose_admission_checks(thermal_admission, resource_admission)
    manager = ModelManager(
        models,
        UrllibHttpClient(),
        sleep_level=int(os.environ.get("SLEEPER_SLEEP_LEVEL", "2")),
        request_timeout_s=float(os.environ.get("SLEEPER_REQUEST_TIMEOUT_SECONDS", "30")),
        owner_validation_timeout_s=float(
            os.environ.get("SLEEPER_OWNER_VALIDATION_TIMEOUT_SECONDS", "2")
        ),
        wake_timeout_s=float(os.environ.get("SLEEPER_WAKE_TIMEOUT_SECONDS", "300")),
        drain_timeout_s=float(os.environ.get("SLEEPER_DRAIN_TIMEOUT_SECONDS", "300")),
        poll_interval_s=float(os.environ.get("SLEEPER_POLL_INTERVAL_SECONDS", "1")),
        always_wake=os.environ.get("SLEEPER_ALWAYS_WAKE", "1") != "0",
        wake_strategy=os.environ.get("SLEEPER_WAKE_STRATEGY", "level2"),
        admission_check=admission_check,
        pre_admission_check=thermal_admission,
        startup_lease_path=os.environ.get(
            "SLEEPER_STARTUP_LEASE_PATH", "/tmp/vllm-sleeper-proxy-startup.lock"
        ),
        transition_state_path=os.environ.get("SLEEPER_TRANSITION_STATE_PATH"),
        bootstrap_mode=os.environ.get("SLEEPER_PROXY_BOOTSTRAP", "0") == "1",
    )
    httpd = build_server(
        host,
        port,
        manager,
        max_request_body_bytes=int(
            os.environ.get("SLEEPER_MAX_REQUEST_BODY_BYTES", str(10 * 1024 * 1024))
        ),
        thermal_action_control_enabled=thermal_action_enabled,
        thermal_action_authority=thermal_action_authority,
    )
    print(f"vllm-sleeper-proxy listening on http://{host}:{port}", flush=True)
    httpd.serve_forever()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
