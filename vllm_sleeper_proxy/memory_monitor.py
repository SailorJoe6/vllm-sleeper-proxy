from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path


def memory_utilization(meminfo_path: Path = Path("/proc/meminfo")) -> float:
    values: dict[str, int] = {}
    for line in meminfo_path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, remainder = line.split(":", 1)
        fields = remainder.split()
        if fields and fields[0].isdigit():
            values[key] = int(fields[0])
    total = values["MemTotal"]
    available = values["MemAvailable"]
    return 100.0 * (total - available) / total


def request_sleep(control_url: str, timeout_s: float) -> dict[str, object]:
    request = urllib.request.Request(control_url, method="POST", data=b"")
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        payload = json.loads(response.read(4096))
        if response.status >= 400 or not isinstance(payload, dict):
            raise RuntimeError("sleep control returned an invalid response")
        return payload


def monitor(
    *,
    control_url: str,
    critical_percent: float,
    poll_seconds: float,
    timeout_s: float,
) -> None:
    backoff_requested = False
    peak_percent = 0.0
    while True:
        utilization = memory_utilization()
        if utilization > peak_percent:
            peak_percent = utilization
            print(f"total_host_memory_peak_percent={peak_percent:.2f}", flush=True)
        if utilization >= critical_percent and not backoff_requested:
            result = request_sleep(control_url, timeout_s)
            print(
                "resource_backoff="
                f"{result.get('slept_model') or 'no_active_model'} "
                f"utilization_percent={utilization:.2f}",
                flush=True,
            )
            backoff_requested = True
        elif utilization < critical_percent:
            if backoff_requested:
                print(
                    f"resource_recovered utilization_percent={utilization:.2f}",
                    flush=True,
                )
            backoff_requested = False
        time.sleep(poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-url", default="http://sleeper-proxy:8889/sleep")
    parser.add_argument("--critical-percent", type=float, default=95.0)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--timeout-seconds", type=float, default=620.0)
    args = parser.parse_args()
    if not 0 < args.critical_percent <= 95:
        parser.error("--critical-percent must be greater than zero and at most 95")
    if args.poll_seconds <= 0 or args.timeout_seconds <= 0:
        parser.error("poll and timeout values must be greater than zero")
    monitor(
        control_url=args.control_url,
        critical_percent=args.critical_percent,
        poll_seconds=args.poll_seconds,
        timeout_s=args.timeout_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
