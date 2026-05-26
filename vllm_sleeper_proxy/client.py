from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass
class HttpResponse:
    status: int
    headers: dict[str, str]
    body: bytes

    def json(self) -> object:
        if not self.body:
            return None
        return json.loads(self.body.decode("utf-8"))


class HttpClient(Protocol):
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
        timeout: float | None = None,
    ) -> HttpResponse: ...


class UrllibHttpClient:
    """Tiny stdlib HTTP client so the proxy has no runtime package deps."""

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
        timeout: float | None = None,
    ) -> HttpResponse:
        req = Request(url, data=body, method=method.upper())
        for key, value in (headers or {}).items():
            req.add_header(key, value)
        try:
            with urlopen(req, timeout=timeout) as resp:  # noqa: S310 - operator-configured local endpoints
                return HttpResponse(
                    status=resp.status,
                    headers={k.lower(): v for k, v in resp.headers.items()},
                    body=resp.read(),
                )
        except HTTPError as exc:
            return HttpResponse(
                status=exc.code,
                headers={k.lower(): v for k, v in exc.headers.items()},
                body=exc.read(),
            )
        except URLError as exc:
            raise ConnectionError(f"HTTP request failed for {url}: {exc}") from exc


def sleep_until(predicate, *, timeout_s: float, interval_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() <= deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return predicate()
