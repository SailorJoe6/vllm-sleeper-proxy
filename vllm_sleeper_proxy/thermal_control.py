from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
import time
from pathlib import Path
from typing import Callable

ACTION_ID = re.compile(r"^[A-Za-z0-9_.:@-]{1,128}$")
REASON_CODE = re.compile(r"^[a-z0-9_]{1,128}$")
HOLD_PHASES = {"graceful_hold", "urgent_hold", "held"}
RELEASE_PHASES = {"release_authorized", "releasing"}
REQUEST_KEYS = {"schema_version", "action_id", "phase"}
MAX_ACTION_WINDOW_SECONDS = 300.0
MAX_CLOCK_FUTURE_SKEW_SECONDS = 5.0


class ThermalActionControlError(RuntimeError):
    """A bounded action request is not authorized by the root projection."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class ThermalAction:
    action_id: str
    phase: str
    drain_deadline_epoch: float
    sleep_deadline_epoch: float
    overall_deadline_epoch: float
    created_at_epoch: float | None = None


class FileThermalActionAuthority:
    """Strict, non-mutating reader for one root-published action projection."""

    def __init__(
        self,
        path: Path,
        *,
        now: Callable[[], float] = time.time,
        maximum_bytes: int = 16 * 1024,
    ) -> None:
        self.path = path
        self.now = now
        self.maximum_bytes = max(512, min(64 * 1024, int(maximum_bytes)))

    @staticmethod
    def _allowed_phases(operation: str) -> set[str]:
        if operation == "hold":
            return HOLD_PHASES
        if operation == "release":
            return RELEASE_PHASES
        raise ValueError("unsupported thermal action operation")

    def _load(self) -> dict[str, object]:
        try:
            with self.path.open("rb") as handle:
                raw = handle.read(self.maximum_bytes + 1)
        except OSError as exc:
            raise ThermalActionControlError("unavailable") from exc
        if len(raw) > self.maximum_bytes:
            raise ThermalActionControlError("invalid_authority")
        try:
            value = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ThermalActionControlError("invalid_authority") from exc
        if not isinstance(value, dict):
            raise ThermalActionControlError("invalid_authority")
        if value.get("active") is not True:
            raise ThermalActionControlError("inactive")
        return value

    @staticmethod
    def _finite_number(value: object) -> float:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ThermalActionControlError("invalid_authority")
        try:
            number = float(value)
        except (OverflowError, ValueError) as exc:
            raise ThermalActionControlError("invalid_authority") from exc
        if not math.isfinite(number):
            raise ThermalActionControlError("invalid_authority")
        return number

    def authorize(self, operation: str, request: object) -> ThermalAction:
        allowed_phases = self._allowed_phases(operation)
        if not isinstance(request, dict) or set(request) != REQUEST_KEYS:
            raise ThermalActionControlError("invalid_request")
        if type(request.get("schema_version")) is not int or request["schema_version"] != 1:
            raise ThermalActionControlError("invalid_request")
        requested_id = request.get("action_id")
        requested_phase = request.get("phase")
        if not isinstance(requested_id, str) or not ACTION_ID.fullmatch(requested_id):
            raise ThermalActionControlError("invalid_request")
        if not isinstance(requested_phase, str):
            raise ThermalActionControlError("invalid_request")
        if requested_phase not in allowed_phases:
            raise ThermalActionControlError("phase_not_allowed")

        value = self._load()
        if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
            raise ThermalActionControlError("invalid_authority")
        action_id = value.get("action_id")
        phase = value.get("phase")
        if not isinstance(action_id, str) or not ACTION_ID.fullmatch(action_id):
            raise ThermalActionControlError("invalid_authority")
        if not isinstance(phase, str) or phase not in allowed_phases:
            raise ThermalActionControlError("phase_not_allowed")
        if action_id != requested_id or phase != requested_phase:
            raise ThermalActionControlError("mismatch")
        if value.get("state") not in {"sleep", "recovering"}:
            raise ThermalActionControlError("invalid_authority")
        if not isinstance(value.get("applied"), bool):
            raise ThermalActionControlError("invalid_authority")
        if value.get("requirement") != "REQ-MODEL-AVAIL-001":
            raise ThermalActionControlError("invalid_authority")
        reasons = value.get("reason_codes")
        if (
            not isinstance(reasons, list)
            or len(reasons) > 16
            or any(
                not isinstance(reason, str) or not REASON_CODE.fullmatch(reason)
                for reason in reasons
            )
        ):
            raise ThermalActionControlError("invalid_authority")
        created = self._finite_number(value.get("created_at_epoch"))
        generated = self._finite_number(value.get("generated_at_epoch"))
        now = self.now()
        if (
            created > generated
            or created > now + MAX_CLOCK_FUTURE_SKEW_SECONDS
            or generated > now + MAX_CLOCK_FUTURE_SKEW_SECONDS
        ):
            raise ThermalActionControlError("invalid_authority")

        drain = self._finite_number(value.get("drain_deadline_epoch"))
        sleep = self._finite_number(value.get("sleep_deadline_epoch"))
        overall = self._finite_number(value.get("overall_deadline_epoch"))
        if not created <= drain <= sleep <= overall:
            raise ThermalActionControlError("invalid_authority")
        if overall - created > MAX_ACTION_WINDOW_SECONDS:
            raise ThermalActionControlError("invalid_authority")
        if operation == "hold" and phase != "held" and overall <= now:
            raise ThermalActionControlError("deadline_expired")
        return ThermalAction(
            action_id=action_id,
            phase=phase,
            drain_deadline_epoch=drain,
            sleep_deadline_epoch=sleep,
            overall_deadline_epoch=overall,
            created_at_epoch=created,
        )
