from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
import time
from pathlib import Path
from typing import Callable

SCHEMA_VERSION = 2
ACTION_ID = re.compile(r"^[A-Za-z0-9_.:@-]{1,128}$")
ENGINE_ID = re.compile(r"^[A-Za-z0-9_.:@-]{1,128}$")
REASON_CODE = re.compile(r"^[a-z0-9_]{1,128}$")
HOLD_PHASES = {"graceful_hold", "urgent_hold", "held"}
RELEASE_PHASES = {
    "release_authorized", "cutoff_recovery_authorized", "releasing"
}
TRANSITION_KINDS = {
    "graceful_hold", "urgent_hold", "hard_cutoff", "cutoff_recovery",
    "store_degraded_cutoff_reconstruction",
}
CONTAINMENT_LEVELS = {"graceful", "urgent", "cutoff", "cutoff_recovery"}
REQUEST_KEYS = {
    "schema_version", "operation", "record_revision", "incident_id",
    "action_id", "generation", "predecessor_action_id", "transition_kind",
    "phase", "containment_level", "created_at_epoch", "phase_updated_at_epoch",
    "drain_deadline_epoch", "sleep_deadline_epoch", "overall_deadline_epoch",
    "release_authorized_at_epoch", "repair_deadline_epoch",
    "repair_target_engine_key", "repair_target_status",
    "repair_target_deadline_epoch", "recovery_authorized",
    "engine_keys", "proof_engine_keys",
}
PROJECTION_KEYS = {
    "schema_version", "record_revision", "incident_id", "action_id",
    "generation", "predecessor_action_id", "transition_kind",
    "created_at_epoch", "phase_updated_at_epoch", "generated_at_epoch",
    "active", "state", "phase", "containment_level", "applied", "result",
    "recovery_authorized", "reason_codes", "drain_deadline_epoch",
    "sleep_deadline_epoch", "overall_deadline_epoch",
    "release_authorized_at_epoch", "repair_deadline_epoch",
    "repair_target_engine_key", "repair_target_status",
    "repair_target_deadline_epoch", "engine_keys",
    "sleeping_peer_engine_keys", "authorized_operations", "requirement",
}
BOUND_REQUEST_FIELDS = REQUEST_KEYS - {
    "schema_version", "operation", "proof_engine_keys"
}
MAX_ACTION_WINDOW_SECONDS = 300.0
MAX_SLEEPING_SUBSET_PROOF_SECONDS = 2.0
MAX_URGENT_DRAIN_SECONDS = 0.5
MAX_URGENT_SLEEP_SECONDS = 5.0
MAX_TOTAL_REPAIR_SECONDS = 5_400.0
MAX_CLOCK_FUTURE_SKEW_SECONDS = 5.0
MAX_ENGINES = 16
MAX_INTEGER = (1 << 63) - 1


def _reject_duplicate_json_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


class ThermalActionControlError(RuntimeError):
    """A bounded action request is not authorized by the root projection."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class ThermalAction:
    record_revision: int
    incident_id: str
    action_id: str
    generation: int
    predecessor_action_id: str | None
    transition_kind: str
    phase: str
    containment_level: str
    created_at_epoch: float
    phase_updated_at_epoch: float
    drain_deadline_epoch: float | None
    sleep_deadline_epoch: float | None
    overall_deadline_epoch: float | None
    release_authorized_at_epoch: float | None
    repair_deadline_epoch: float | None
    repair_target_engine_key: str | None
    repair_target_status: str | None
    repair_target_deadline_epoch: float | None
    recovery_authorized: bool
    engine_keys: tuple[str, ...]
    proof_engine_keys: tuple[str, ...]


class FileThermalActionAuthority:
    """Strict, non-mutating reader for one root-published schema-v2 projection."""

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
        if operation == "sleeping_subset_proof":
            return {"held"}
        if operation in {"repair_peer_proof", "repair_converge"}:
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
            value = json.loads(
                raw,
                object_pairs_hook=_reject_duplicate_json_object,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ThermalActionControlError("invalid_authority") from exc
        if not isinstance(value, dict):
            raise ThermalActionControlError("invalid_authority")
        if value.get("active") is not True:
            raise ThermalActionControlError("inactive")
        if set(value) != PROJECTION_KEYS:
            raise ThermalActionControlError("invalid_authority")
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

    @classmethod
    def _nullable_number(cls, value: object) -> float | None:
        return None if value is None else cls._finite_number(value)

    @staticmethod
    def _positive_integer(value: object) -> int:
        if type(value) is not int or not 1 <= value <= MAX_INTEGER:
            raise ThermalActionControlError("invalid_authority")
        return value

    @staticmethod
    def _engine_keys(value: object) -> tuple[str, ...]:
        if (
            not isinstance(value, list)
            or not 1 <= len(value) <= MAX_ENGINES
            or value != sorted(set(value))
            or any(not isinstance(item, str) or not ENGINE_ID.fullmatch(item) for item in value)
        ):
            raise ThermalActionControlError("invalid_authority")
        return tuple(value)

    @staticmethod
    def _engine_subset(
        value: object,
        *,
        engine_keys: tuple[str, ...],
        allow_empty: bool,
        error_code: str,
    ) -> tuple[str, ...]:
        if (
            not isinstance(value, list)
            or len(value) > len(engine_keys)
            or (not allow_empty and not value)
            or value != sorted(set(value))
            or any(
                not isinstance(item, str)
                or not ENGINE_ID.fullmatch(item)
                or item not in engine_keys
                for item in value
            )
        ):
            raise ThermalActionControlError(error_code)
        return tuple(value)

    def authorize(self, operation: str, request: object) -> ThermalAction:
        allowed_phases = self._allowed_phases(operation)
        if not isinstance(request, dict) or set(request) != REQUEST_KEYS:
            raise ThermalActionControlError("invalid_request")
        if type(request.get("schema_version")) is not int or request["schema_version"] != SCHEMA_VERSION:
            raise ThermalActionControlError("invalid_request")
        if request.get("operation") != operation:
            raise ThermalActionControlError("invalid_request")
        requested_id = request.get("action_id")
        requested_phase = request.get("phase")
        if not isinstance(requested_id, str) or not ACTION_ID.fullmatch(requested_id):
            raise ThermalActionControlError("invalid_request")
        if not isinstance(requested_phase, str) or requested_phase not in allowed_phases:
            raise ThermalActionControlError("phase_not_allowed")

        value = self._load()
        if type(value.get("schema_version")) is not int or value["schema_version"] != SCHEMA_VERSION:
            raise ThermalActionControlError("invalid_authority")
        for key in BOUND_REQUEST_FIELDS:
            requested = request.get(key)
            authoritative = value.get(key)
            if (
                type(requested) is not type(authoritative)
                or requested != authoritative
            ):
                raise ThermalActionControlError("mismatch")

        record_revision = self._positive_integer(value.get("record_revision"))
        incident_id = value.get("incident_id")
        action_id = value.get("action_id")
        generation = self._positive_integer(value.get("generation"))
        predecessor = value.get("predecessor_action_id")
        kind = value.get("transition_kind")
        phase = value.get("phase")
        level = value.get("containment_level")
        if (
            not isinstance(incident_id, str) or not ACTION_ID.fullmatch(incident_id)
            or not isinstance(action_id, str) or not ACTION_ID.fullmatch(action_id)
            or action_id != requested_id
            or kind not in TRANSITION_KINDS
            or phase not in allowed_phases
            or phase != requested_phase
            or level not in CONTAINMENT_LEVELS
        ):
            raise ThermalActionControlError("invalid_authority")
        if predecessor is not None and (
            not isinstance(predecessor, str) or not ACTION_ID.fullmatch(predecessor)
        ):
            raise ThermalActionControlError("invalid_authority")
        if (generation == 1) != (predecessor is None):
            raise ThermalActionControlError("invalid_authority")
        if kind in {"graceful_hold", "hard_cutoff", "store_degraded_cutoff_reconstruction"} and generation != 1:
            raise ThermalActionControlError("invalid_authority")
        if kind == "cutoff_recovery" and generation == 1:
            raise ThermalActionControlError("invalid_authority")
        allowed_kind_phases = {
            "graceful_hold": {"graceful_hold", "held", "release_authorized", "releasing"},
            "urgent_hold": {"urgent_hold", "held", "release_authorized", "releasing"},
            "hard_cutoff": set(),
            "cutoff_recovery": {"cutoff_recovery_authorized", "releasing"},
            "store_degraded_cutoff_reconstruction": set(),
        }[kind]
        expected_level = (
            "cutoff_recovery" if kind == "cutoff_recovery"
            else "graceful" if kind == "graceful_hold"
            else "urgent" if kind == "urgent_hold"
            else "cutoff"
        )
        expected_state = (
            "sleep" if operation in {"hold", "sleeping_subset_proof"}
            else "recovering"
        )
        if (
            phase not in allowed_kind_phases
            or level != expected_level
            or value.get("state") != expected_state
        ):
            raise ThermalActionControlError("invalid_authority")
        if operation == "hold" and phase not in HOLD_PHASES:
            raise ThermalActionControlError("invalid_authority")
        if operation == "sleeping_subset_proof" and phase != "held":
            raise ThermalActionControlError("invalid_authority")
        if (
            operation in {"repair_peer_proof", "repair_converge", "release"}
            and phase not in RELEASE_PHASES
        ):
            raise ThermalActionControlError("invalid_authority")
        if value.get("state") not in {"sleep", "recovering"}:
            raise ThermalActionControlError("invalid_authority")
        if not isinstance(value.get("applied"), bool):
            raise ThermalActionControlError("invalid_authority")
        if operation in {"hold", "sleeping_subset_proof"}:
            if phase == "held":
                if value["applied"] is not True or value.get("result") != "all_sleeping_or_stopped":
                    raise ThermalActionControlError("invalid_authority")
            elif value["applied"] is not False or value.get("result") not in {"pending", "failed"}:
                raise ThermalActionControlError("invalid_authority")
        elif kind != "cutoff_recovery" and value["applied"] is not True:
            raise ThermalActionControlError("invalid_authority")
        if not isinstance(value.get("recovery_authorized"), bool):
            raise ThermalActionControlError("invalid_authority")
        if value.get("requirement") != "REQ-MODEL-AVAIL-001":
            raise ThermalActionControlError("invalid_authority")
        reasons = value.get("reason_codes")
        if (
            not isinstance(reasons, list)
            or not 1 <= len(reasons) <= 16
            or reasons != sorted(set(reasons))
            or any(not isinstance(reason, str) or not REASON_CODE.fullmatch(reason) for reason in reasons)
        ):
            raise ThermalActionControlError("invalid_authority")
        engine_keys = self._engine_keys(value.get("engine_keys"))
        sleeping_peer_keys = self._engine_subset(
            value.get("sleeping_peer_engine_keys"),
            engine_keys=engine_keys,
            allow_empty=True,
            error_code="invalid_authority",
        )
        repair_target = value.get("repair_target_engine_key")
        repair_status = value.get("repair_target_status")
        repair_target_deadline = self._nullable_number(
            value.get("repair_target_deadline_epoch")
        )
        if repair_target is None and repair_status is None and repair_target_deadline is None:
            pass
        elif (
            not isinstance(repair_target, str)
            or repair_target not in engine_keys
            or repair_status not in {"starting", "converging"}
            or repair_target_deadline is None
            or repair_target in sleeping_peer_keys
        ):
            raise ThermalActionControlError("invalid_authority")
        requested_proof_keys = self._engine_subset(
            request.get("proof_engine_keys"),
            engine_keys=engine_keys,
            allow_empty=operation == "repair_peer_proof",
            error_code="invalid_request",
        )
        if operation in {"sleeping_subset_proof", "repair_peer_proof"}:
            expected_proof_keys = sleeping_peer_keys
        elif operation == "repair_converge":
            expected_proof_keys = (repair_target,) if repair_target is not None else ()
        else:
            expected_proof_keys = engine_keys
        if phase == "held":
            expected_operations = ["hold"]
            if sleeping_peer_keys:
                expected_operations.append("sleeping_subset_proof")
        elif repair_target is not None:
            expected_operations = [
                "repair_peer_proof"
                if repair_status == "starting"
                else "repair_converge"
            ]
        elif operation == "hold":
            expected_operations = ["hold"]
        elif operation == "release":
            expected_operations = ["release"]
        else:
            expected_operations = []
        if (
            requested_proof_keys != expected_proof_keys
            or value.get("authorized_operations") != expected_operations
            or operation not in expected_operations
        ):
            raise ThermalActionControlError("mismatch")

        created = self._finite_number(value.get("created_at_epoch"))
        phase_updated = self._finite_number(value.get("phase_updated_at_epoch"))
        generated = self._finite_number(value.get("generated_at_epoch"))
        now = self.now()
        if (
            not created <= phase_updated <= generated
            or created > now + MAX_CLOCK_FUTURE_SKEW_SECONDS
            or phase_updated > now + MAX_CLOCK_FUTURE_SKEW_SECONDS
            or generated > now + MAX_CLOCK_FUTURE_SKEW_SECONDS
        ):
            raise ThermalActionControlError("invalid_authority")

        drain = self._nullable_number(value.get("drain_deadline_epoch"))
        sleep = self._nullable_number(value.get("sleep_deadline_epoch"))
        overall = self._nullable_number(value.get("overall_deadline_epoch"))
        release_authorized = self._nullable_number(value.get("release_authorized_at_epoch"))
        repair_deadline = self._nullable_number(value.get("repair_deadline_epoch"))
        if operation in {"hold", "sleeping_subset_proof"}:
            if any(item is None for item in (drain, sleep, overall)):
                raise ThermalActionControlError("invalid_authority")
            assert drain is not None and sleep is not None and overall is not None
            if not created <= drain <= sleep <= overall or overall - created > MAX_ACTION_WINDOW_SECONDS:
                raise ThermalActionControlError("invalid_authority")
            if kind == "urgent_hold" and (
                drain - created > MAX_URGENT_DRAIN_SECONDS
                or sleep - drain > MAX_URGENT_SLEEP_SECONDS
                or overall != sleep
            ):
                raise ThermalActionControlError("invalid_authority")
            if phase != "held" and overall <= now:
                raise ThermalActionControlError("deadline_expired")
            if release_authorized is not None or repair_deadline is not None or value["recovery_authorized"]:
                raise ThermalActionControlError("invalid_authority")
        else:
            if (release_authorized is None) != (repair_deadline is None):
                raise ThermalActionControlError("invalid_authority")
            if (
                release_authorized is None
                or repair_deadline is None
                or not 0 < repair_deadline - release_authorized <= MAX_TOTAL_REPAIR_SECONDS
                or value["recovery_authorized"] is not True
            ):
                raise ThermalActionControlError("invalid_authority")
            if repair_deadline <= now:
                raise ThermalActionControlError("deadline_expired")
            if kind == "cutoff_recovery":
                if (
                    phase not in {"cutoff_recovery_authorized", "releasing"}
                    or level != "cutoff_recovery"
                    or any(item is not None for item in (drain, sleep, overall))
                    or created != release_authorized
                ):
                    raise ThermalActionControlError("invalid_authority")
            elif (
                any(item is None for item in (drain, sleep, overall))
                or level not in {"graceful", "urgent"}
            ):
                raise ThermalActionControlError("invalid_authority")
        if operation in {"hold", "sleeping_subset_proof", "release"}:
            if any(item is not None for item in (
                repair_target, repair_status, repair_target_deadline
            )):
                raise ThermalActionControlError("invalid_authority")
        else:
            expected_status = (
                "starting" if operation == "repair_peer_proof" else "converging"
            )
            if (
                repair_target is None
                or repair_status != expected_status
                or repair_target_deadline is None
                or repair_deadline is None
                or repair_target_deadline <= now
                or repair_target_deadline > repair_deadline
            ):
                raise ThermalActionControlError("invalid_authority")

        return ThermalAction(
            record_revision=record_revision,
            incident_id=incident_id,
            action_id=action_id,
            generation=generation,
            predecessor_action_id=predecessor,
            transition_kind=kind,
            phase=phase,
            containment_level=level,
            created_at_epoch=created,
            phase_updated_at_epoch=phase_updated,
            drain_deadline_epoch=drain,
            sleep_deadline_epoch=sleep,
            overall_deadline_epoch=overall,
            release_authorized_at_epoch=release_authorized,
            repair_deadline_epoch=repair_deadline,
            repair_target_engine_key=repair_target,
            repair_target_status=repair_status,
            repair_target_deadline_epoch=repair_target_deadline,
            recovery_authorized=value["recovery_authorized"],
            engine_keys=engine_keys,
            proof_engine_keys=requested_proof_keys,
        )
