from __future__ import annotations

from dataclasses import dataclass
import errno
import hashlib
import json
import math
import os
import re
import stat
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
PROJECTION_PROTOCOL_VERSION = 1
REQUIREMENT = "REQ-MODEL-AVAIL-001"
DIGEST = re.compile(r"^[0-9a-f]{64}$")
PUBLICATION_STATES = {"quiescing", "active", "inactive"}
TRANSITIONS = {"bootstrap", "advance", "rebind"}
ENVELOPE_KEYS = {
    "projection_protocol_version", "publication_state", "transition",
    "publication_sequence", "predecessor_publication_sequence",
    "publication_id", "predecessor_publication_id", "successor_publication_id",
    "source_store_revision", "source_record_revision",
    "target_store_revision", "target_record_revision",
    "source_authority_sha256", "target_authority_sha256",
    "prepared_at_epoch", "deadline_epoch", "authority", "requirement",
}
CONTROL_REQUEST_KEYS = ENVELOPE_KEYS - {"authority"} | {"operation"}
PUBLICATION_BINDING_KEYS = {
    "publication_sequence", "publication_id", "target_store_revision",
    "target_authority_sha256", "proxy_instance_id", "activation_token",
}

LEGACY_REQUEST_KEYS = {
    "schema_version", "operation", "record_revision", "incident_id",
    "action_id", "generation", "predecessor_action_id", "transition_kind",
    "phase", "containment_level", "created_at_epoch", "phase_updated_at_epoch",
    "drain_deadline_epoch", "sleep_deadline_epoch", "overall_deadline_epoch",
    "release_authorized_at_epoch", "repair_deadline_epoch",
    "repair_target_engine_key", "repair_target_status",
    "repair_target_deadline_epoch", "recovery_authorized",
    "engine_keys", "proof_engine_keys",
}
REQUEST_KEYS = LEGACY_REQUEST_KEYS | PUBLICATION_BINDING_KEYS
INACTIVE_PROJECTION_KEYS = {
    "schema_version", "generated_at_epoch", "active", "state", "applied",
    "action_id", "reason_codes", "requirement",
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
BOUND_REQUEST_FIELDS = LEGACY_REQUEST_KEYS - {
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
class ProjectionEnvelope:
    publication_state: str
    transition: str
    publication_sequence: int
    predecessor_publication_sequence: int | None
    publication_id: str
    predecessor_publication_id: str | None
    successor_publication_id: str | None
    source_store_revision: int
    source_record_revision: int
    target_store_revision: int
    target_record_revision: int
    source_authority_sha256: str | None
    target_authority_sha256: str
    prepared_at_epoch: float
    deadline_epoch: float
    authority: dict[str, object] | None

    def control_fields(self, operation: str) -> dict[str, object]:
        return {
            "projection_protocol_version": PROJECTION_PROTOCOL_VERSION,
            "operation": operation,
            "publication_state": self.publication_state,
            "transition": self.transition,
            "publication_sequence": self.publication_sequence,
            "predecessor_publication_sequence": self.predecessor_publication_sequence,
            "publication_id": self.publication_id,
            "predecessor_publication_id": self.predecessor_publication_id,
            "successor_publication_id": self.successor_publication_id,
            "source_store_revision": self.source_store_revision,
            "source_record_revision": self.source_record_revision,
            "target_store_revision": self.target_store_revision,
            "target_record_revision": self.target_record_revision,
            "source_authority_sha256": self.source_authority_sha256,
            "target_authority_sha256": self.target_authority_sha256,
            "prepared_at_epoch": self.prepared_at_epoch,
            "deadline_epoch": self.deadline_epoch,
            "requirement": REQUIREMENT,
        }

    @property
    def action_binding(self) -> tuple[int, str, int, str]:
        return (
            self.publication_sequence,
            self.publication_id,
            self.target_store_revision,
            self.target_authority_sha256,
        )


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
    publication_sequence: int = 0
    publication_id: str = ""
    target_store_revision: int = 0
    target_authority_sha256: str = ""
    proxy_instance_id: str = ""
    activation_token: str = ""


class FileThermalActionAuthority:
    """Strict, non-mutating reader for one root-published projection envelope."""

    def __init__(
        self,
        path: Path,
        *,
        now: Callable[[], float] = time.time,
        maximum_bytes: int = 16 * 1024,
        expected_uid: int | None = None,
        expected_mode: int | None = None,
        allow_legacy: bool | None = None,
    ) -> None:
        self.path = path
        self.now = now
        self.maximum_bytes = max(512, min(64 * 1024, int(maximum_bytes)))
        self.expected_uid = expected_uid
        self.expected_mode = expected_mode
        self._allow_legacy_explicit = allow_legacy is True
        self.allow_legacy = True if allow_legacy is None else allow_legacy

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

    def _read_json(self) -> dict[str, object]:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            path_before = os.lstat(self.path)
            descriptor = os.open(self.path, flags)
        except OSError as exc:
            code = "invalid_authority" if exc.errno in {errno.ELOOP, errno.EMLINK} else "unavailable"
            raise ThermalActionControlError(code) from exc
        try:
            metadata = os.fstat(descriptor)
            stable_fields = (
                "st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_nlink",
                "st_size", "st_mtime_ns", "st_ctime_ns",
            )
            if any(
                getattr(path_before, field) != getattr(metadata, field)
                for field in stable_fields
            ):
                raise ThermalActionControlError("invalid_authority")
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ThermalActionControlError("invalid_authority")
            mode = stat.S_IMODE(metadata.st_mode)
            if self.expected_uid is not None and metadata.st_uid != self.expected_uid:
                raise ThermalActionControlError("invalid_authority")
            if self.expected_mode is not None and mode != self.expected_mode:
                raise ThermalActionControlError("invalid_authority")
            chunks: list[bytes] = []
            remaining = self.maximum_bytes + 1
            while remaining > 0:
                chunk = os.read(descriptor, remaining)
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            after = os.fstat(descriptor)
            path_after = os.lstat(self.path)
            if (
                any(getattr(metadata, field) != getattr(after, field) for field in stable_fields)
                or any(getattr(after, field) != getattr(path_after, field) for field in stable_fields)
                or len(raw) != metadata.st_size
            ):
                raise ThermalActionControlError("invalid_authority")
        except OSError as exc:
            raise ThermalActionControlError("invalid_authority") from exc
        finally:
            os.close(descriptor)
        if len(raw) > self.maximum_bytes:
            raise ThermalActionControlError("invalid_authority")
        try:
            value = json.loads(raw, object_pairs_hook=_reject_duplicate_json_object)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ThermalActionControlError("invalid_authority") from exc
        if not isinstance(value, dict):
            raise ThermalActionControlError("invalid_authority")
        if "projection_protocol_version" in value:
            if mode & 0o022:
                raise ThermalActionControlError("invalid_authority")
            try:
                canonical = json.dumps(
                    value, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=False, allow_nan=False,
                ).encode("utf-8")
            except (TypeError, ValueError) as exc:
                raise ThermalActionControlError("invalid_authority") from exc
            if raw != canonical:
                raise ThermalActionControlError("invalid_authority")
        return value

    @staticmethod
    def _canonical_digest(value: dict[str, object]) -> str:
        try:
            encoded = json.dumps(
                value, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ThermalActionControlError("invalid_authority") from exc
        return hashlib.sha256(encoded).hexdigest()

    def _parse_envelope(
        self, value: dict[str, object], *, require_live_deadline: bool
    ) -> ProjectionEnvelope:
        if set(value) != ENVELOPE_KEYS:
            raise ThermalActionControlError("invalid_authority")
        if (
            type(value.get("projection_protocol_version")) is not int
            or value.get("projection_protocol_version") != PROJECTION_PROTOCOL_VERSION
        ):
            raise ThermalActionControlError("invalid_authority")
        state = value.get("publication_state")
        transition = value.get("transition")
        if (
            not isinstance(state, str)
            or state not in PUBLICATION_STATES
            or not isinstance(transition, str)
            or transition not in TRANSITIONS
        ):
            raise ThermalActionControlError("invalid_authority")

        def positive(name: str) -> int:
            item = value.get(name)
            if type(item) is not int or not 1 <= item <= MAX_INTEGER:
                raise ThermalActionControlError("invalid_authority")
            return item

        sequence = positive("publication_sequence")
        predecessor_sequence = value.get("predecessor_publication_sequence")
        if predecessor_sequence is not None and (
            type(predecessor_sequence) is not int
            or not 1 <= predecessor_sequence < sequence
        ):
            raise ThermalActionControlError("invalid_authority")
        publication_id = value.get("publication_id")
        predecessor_id = value.get("predecessor_publication_id")
        successor_id = value.get("successor_publication_id")
        source_digest = value.get("source_authority_sha256")
        target_digest = value.get("target_authority_sha256")
        if not isinstance(publication_id, str) or not DIGEST.fullmatch(publication_id):
            raise ThermalActionControlError("invalid_authority")
        for digest in (predecessor_id, successor_id, source_digest, target_digest):
            if digest is not None and (
                not isinstance(digest, str) or not DIGEST.fullmatch(digest)
            ):
                raise ThermalActionControlError("invalid_authority")
        if sequence == 1:
            if predecessor_sequence is not None or predecessor_id is not None:
                raise ThermalActionControlError("invalid_authority")
        elif (
            predecessor_sequence != sequence - 1
            or predecessor_id is None
        ):
            raise ThermalActionControlError("invalid_authority")
        if (state == "quiescing") != (successor_id is not None):
            raise ThermalActionControlError("invalid_authority")
        if (transition == "bootstrap") != (source_digest is None):
            raise ThermalActionControlError("invalid_authority")

        source_store = positive("source_store_revision")
        source_record = positive("source_record_revision")
        target_store = positive("target_store_revision")
        target_record = positive("target_record_revision")
        if transition == "advance":
            if target_store != source_store + 1 or target_record != source_record + 1:
                raise ThermalActionControlError("invalid_authority")
        elif target_store != source_store or target_record != source_record:
            raise ThermalActionControlError("invalid_authority")

        prepared = self._finite_number(value.get("prepared_at_epoch"))
        deadline = self._finite_number(value.get("deadline_epoch"))
        current = self.now()
        if (
            prepared > current + MAX_CLOCK_FUTURE_SKEW_SECONDS
            or deadline <= prepared
            or (require_live_deadline and deadline <= current)
        ):
            raise ThermalActionControlError(
                "deadline_expired" if require_live_deadline and deadline <= current
                else "invalid_authority"
            )
        authority = value.get("authority")
        if state == "quiescing":
            if authority is not None:
                raise ThermalActionControlError("invalid_authority")
        else:
            expected_authority_keys = (
                PROJECTION_KEYS if state == "active" else INACTIVE_PROJECTION_KEYS
            )
            if not isinstance(authority, dict) or set(authority) != expected_authority_keys:
                raise ThermalActionControlError("invalid_authority")
            if authority.get("active") is not (state == "active"):
                raise ThermalActionControlError("invalid_authority")
            if state == "inactive" and (
                type(authority.get("schema_version")) is not int
                or authority.get("schema_version") != SCHEMA_VERSION
                or authority.get("state") != "none"
                or authority.get("applied") is not False
                or authority.get("action_id") is not None
                or authority.get("reason_codes") != []
                or authority.get("requirement") != REQUIREMENT
            ):
                raise ThermalActionControlError("invalid_authority")
            if state == "inactive":
                generated = self._finite_number(authority.get("generated_at_epoch"))
                if generated < 0 or generated > current + MAX_CLOCK_FUTURE_SKEW_SECONDS:
                    raise ThermalActionControlError("invalid_authority")
            if self._canonical_digest(authority) != target_digest:
                raise ThermalActionControlError("invalid_authority")
        if value.get("requirement") != REQUIREMENT:
            raise ThermalActionControlError("invalid_authority")
        semantic = dict(value)
        semantic.pop("publication_id", None)
        semantic.pop("predecessor_publication_id", None)
        semantic.pop("successor_publication_id", None)
        if self._canonical_digest(semantic) != publication_id:
            raise ThermalActionControlError("invalid_authority")
        assert isinstance(state, str) and isinstance(transition, str)
        assert isinstance(publication_id, str) and isinstance(target_digest, str)
        return ProjectionEnvelope(
            publication_state=state,
            transition=transition,
            publication_sequence=sequence,
            predecessor_publication_sequence=predecessor_sequence,
            publication_id=publication_id,
            predecessor_publication_id=predecessor_id,
            successor_publication_id=successor_id,
            source_store_revision=source_store,
            source_record_revision=source_record,
            target_store_revision=target_store,
            target_record_revision=target_record,
            source_authority_sha256=source_digest,
            target_authority_sha256=target_digest,
            prepared_at_epoch=prepared,
            deadline_epoch=deadline,
            authority=authority,
        )

    def legacy_projection_configured(self) -> bool:
        if not self.allow_legacy:
            return False
        if self._allow_legacy_explicit:
            return True
        try:
            return set(self._read_json()) == PROJECTION_KEYS
        except ThermalActionControlError:
            return False

    def load_envelope(self, *, require_live_deadline: bool = False) -> ProjectionEnvelope:
        return self._parse_envelope(
            self._read_json(), require_live_deadline=require_live_deadline
        )

    @staticmethod
    def _request_for_envelope_operation(
        envelope: ProjectionEnvelope, operation: str
    ) -> dict[str, object]:
        authority = envelope.authority
        if authority is None:
            raise ThermalActionControlError("invalid_authority")
        request = {
            "schema_version": SCHEMA_VERSION,
            "operation": operation,
            **{key: authority.get(key) for key in BOUND_REQUEST_FIELDS},
            "publication_sequence": envelope.publication_sequence,
            "publication_id": envelope.publication_id,
            "target_store_revision": envelope.target_store_revision,
            "target_authority_sha256": envelope.target_authority_sha256,
            "proxy_instance_id": "0" * 64,
            "activation_token": "0" * 64,
        }
        if operation in {"sleeping_subset_proof", "repair_peer_proof"}:
            proof = authority.get("sleeping_peer_engine_keys")
        elif operation == "repair_converge":
            target = authority.get("repair_target_engine_key")
            proof = [target] if isinstance(target, str) else []
        else:
            proof = authority.get("engine_keys")
        request["proof_engine_keys"] = proof
        return request

    def _validate_active_envelope_authority(
        self, envelope: ProjectionEnvelope
    ) -> None:
        authority = envelope.authority
        assert authority is not None
        operations = authority.get("authorized_operations")
        allowed = {
            "hold", "release", "sleeping_subset_proof",
            "repair_peer_proof", "repair_converge",
        }
        if (
            not isinstance(operations, list)
            or not operations
            or len(operations) != len(set(operations))
            or any(not isinstance(item, str) or item not in allowed for item in operations)
            or authority.get("record_revision") != envelope.target_record_revision
        ):
            raise ThermalActionControlError("invalid_authority")
        for operation in operations:
            self.authorize(
                operation,
                self._request_for_envelope_operation(envelope, operation),
            )

    def authorize_control(
        self, operation: str, request: object
    ) -> ProjectionEnvelope:
        expected_states = {
            "quiesce": {"quiescing"},
            "activate": {"active", "inactive"},
        }.get(operation)
        if expected_states is None or not isinstance(request, dict) or set(request) != CONTROL_REQUEST_KEYS:
            raise ThermalActionControlError("invalid_request")
        if request.get("operation") != operation:
            raise ThermalActionControlError("invalid_request")
        envelope = self.load_envelope(require_live_deadline=True)
        if envelope.publication_state not in expected_states:
            raise ThermalActionControlError("mismatch")
        if request != envelope.control_fields(operation):
            raise ThermalActionControlError("mismatch")
        if envelope.publication_state == "active":
            self._validate_active_envelope_authority(envelope)
        return envelope

    def _load(self) -> tuple[dict[str, object], ProjectionEnvelope | None]:
        raw = self._read_json()
        if self.allow_legacy and set(raw) == PROJECTION_KEYS:
            if raw.get("active") is not True:
                raise ThermalActionControlError("inactive")
            return raw, None
        if self.allow_legacy and raw.get("active") is False and "publication_state" not in raw:
            raise ThermalActionControlError("inactive")
        envelope = self._parse_envelope(raw, require_live_deadline=True)
        if envelope.publication_state != "active":
            raise ThermalActionControlError("inactive")
        assert envelope.authority is not None
        if envelope.authority.get("active") is not True:
            raise ThermalActionControlError("inactive")
        return envelope.authority, envelope

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
        if (
            not isinstance(request, dict)
            or frozenset(request) not in {frozenset(LEGACY_REQUEST_KEYS), frozenset(REQUEST_KEYS)}
        ):
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

        value, envelope = self._load()
        if envelope is not None and PUBLICATION_BINDING_KEYS <= set(request):
            requested_binding = tuple(request.get(key) for key in (
                "publication_sequence", "publication_id", "target_store_revision",
                "target_authority_sha256",
            ))
            proxy_instance_id = request.get("proxy_instance_id")
            activation_token = request.get("activation_token")
            if (
                requested_binding != envelope.action_binding
                or not isinstance(proxy_instance_id, str)
                or not DIGEST.fullmatch(proxy_instance_id)
                or not isinstance(activation_token, str)
                or not DIGEST.fullmatch(activation_token)
            ):
                raise ThermalActionControlError("mismatch")
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
            publication_sequence=(envelope.publication_sequence if envelope else 0),
            publication_id=(envelope.publication_id if envelope else ""),
            target_store_revision=(envelope.target_store_revision if envelope else 0),
            target_authority_sha256=(envelope.target_authority_sha256 if envelope else ""),
            proxy_instance_id=(
                str(request.get("proxy_instance_id")) if envelope else ""
            ),
            activation_token=(
                str(request.get("activation_token")) if envelope else ""
            ),
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
