from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from vllm_sleeper_proxy.thermal_control import (
    FileThermalActionAuthority,
    ThermalActionControlError,
)


def projection(
    *,
    phase: str = "graceful_hold",
    action_id: str = "thermal-7",
    incident_id: str = "incident-3",
    generation: int = 1,
    predecessor_action_id: str | None = None,
    transition_kind: str | None = None,
    repair_target_engine_key: str | None = None,
    repair_target_status: str | None = None,
    repair_target_deadline_epoch: float | None = None,
) -> dict:
    release = phase in {"release_authorized", "cutoff_recovery_authorized", "releasing"}
    cutoff_recovery = phase == "cutoff_recovery_authorized"
    if transition_kind is None:
        transition_kind = "cutoff_recovery" if cutoff_recovery else (
            "urgent_hold" if phase == "urgent_hold" else "graceful_hold"
        )
    if cutoff_recovery:
        generation = 2
        predecessor_action_id = predecessor_action_id or "thermal-cutoff-6"
    created = 95.0 if cutoff_recovery else (100.0 if transition_kind == "urgent_hold" else 90.0)
    phase_updated = 96.0 if release else created
    return {
        "schema_version": 2,
        "record_revision": 1 if cutoff_recovery else (3 if release else 1),
        "incident_id": incident_id,
        "action_id": action_id,
        "generation": generation,
        "predecessor_action_id": predecessor_action_id,
        "transition_kind": transition_kind,
        "created_at_epoch": created,
        "phase_updated_at_epoch": phase_updated,
        "generated_at_epoch": 100.0 if transition_kind == "urgent_hold" else 99.0,
        "active": True,
        "state": "recovering" if release else "sleep",
        "phase": phase,
        "containment_level": "cutoff_recovery" if cutoff_recovery else (
            "urgent" if transition_kind == "urgent_hold" else "graceful"
        ),
        "applied": not cutoff_recovery and (phase in {"held", "release_authorized", "releasing"}),
        "result": "repair_pending" if cutoff_recovery else (
            "all_sleeping_or_stopped" if release or phase == "held" else "pending"
        ),
        "recovery_authorized": release,
        "reason_codes": ["acpi_temperature_requires_sleep"],
        "drain_deadline_epoch": None if cutoff_recovery else (100.5 if transition_kind == "urgent_hold" else 150.0),
        "sleep_deadline_epoch": None if cutoff_recovery else (105.5 if transition_kind == "urgent_hold" else 180.0),
        "overall_deadline_epoch": None if cutoff_recovery else (105.5 if transition_kind == "urgent_hold" else 200.0),
        "release_authorized_at_epoch": created if release else None,
        "repair_deadline_epoch": 200.0 if release else None,
        "repair_target_engine_key": repair_target_engine_key,
        "repair_target_status": (
            repair_target_status
            if repair_target_engine_key is not None else None
        ),
        "repair_target_deadline_epoch": repair_target_deadline_epoch,
        "engine_keys": ["Qwen3-Embedding-8B", "qwen3.8-flash-next", "vision-vla"],
        "sleeping_peer_engine_keys": [],
        "authorized_operations": [
            (
                "repair_peer_proof"
                if repair_target_status == "starting"
                else "repair_converge"
            ) if repair_target_engine_key is not None
            else "release" if release else "hold"
        ],
        "requirement": "REQ-MODEL-AVAIL-001",
    }


def request(
    *, phase: str = "graceful_hold", action_id: str = "thermal-7", **kwargs
) -> dict:
    value = projection(phase=phase, action_id=action_id, **kwargs)
    operation = value["authorized_operations"][0]
    keys = {
        "record_revision", "incident_id", "action_id", "generation",
        "predecessor_action_id", "transition_kind", "phase",
        "containment_level", "created_at_epoch", "phase_updated_at_epoch",
        "drain_deadline_epoch", "sleep_deadline_epoch", "overall_deadline_epoch",
        "release_authorized_at_epoch", "repair_deadline_epoch",
        "repair_target_engine_key", "repair_target_status",
        "repair_target_deadline_epoch",
        "recovery_authorized", "engine_keys",
    }
    return {
        "schema_version": 2,
        "operation": operation,
        **{key: value[key] for key in keys},
        "proof_engine_keys": (
            [value["repair_target_engine_key"]]
            if operation == "repair_converge"
            else list(value["sleeping_peer_engine_keys"])
            if operation in {"repair_peer_proof", "sleeping_subset_proof"}
            else list(value["engine_keys"])
        ),
    }


def request_from_projection(value: dict, *, operation: str | None = None) -> dict:
    selected = operation or value["authorized_operations"][0]
    keys = {
        "record_revision", "incident_id", "action_id", "generation",
        "predecessor_action_id", "transition_kind", "phase",
        "containment_level", "created_at_epoch", "phase_updated_at_epoch",
        "drain_deadline_epoch", "sleep_deadline_epoch", "overall_deadline_epoch",
        "release_authorized_at_epoch", "repair_deadline_epoch",
        "repair_target_engine_key", "repair_target_status",
        "repair_target_deadline_epoch",
        "recovery_authorized", "engine_keys",
    }
    return {
        "schema_version": 2,
        "operation": selected,
        **{key: value[key] for key in keys},
        "proof_engine_keys": (
            [value["repair_target_engine_key"]]
            if selected == "repair_converge"
            else list(value["sleeping_peer_engine_keys"])
            if selected in {"repair_peer_proof", "sleeping_subset_proof"}
            else list(value["engine_keys"])
        ),
    }


class FileThermalActionAuthorityTests(unittest.TestCase):
    def test_exact_active_hold_contract_is_authorized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "containment.json"
            path.write_text(json.dumps(projection()))
            authority = FileThermalActionAuthority(path, allow_legacy=True, now=lambda: 100.0)
            action = authority.authorize("hold", request())
        self.assertEqual("thermal-7", action.action_id)
        self.assertEqual("graceful_hold", action.phase)
        self.assertEqual(150.0, action.drain_deadline_epoch)
        self.assertEqual(180.0, action.sleep_deadline_epoch)
        self.assertEqual(200.0, action.overall_deadline_epoch)
        self.assertEqual(90.0, action.created_at_epoch)

    def test_inactive_malformed_or_mismatched_authority_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "containment.json"
            authority = FileThermalActionAuthority(path, allow_legacy=True, now=lambda: 100.0)
            nonfinite = {**projection(), "overall_deadline_epoch": float("nan")}
            inverted = {**projection(), "drain_deadline_epoch": 190.0, "sleep_deadline_epoch": 180.0}
            cases = (
                (None, request(), "unavailable"),
                ({"schema_version": 1, "active": False}, request(), "inactive"),
                (projection(), request(action_id="thermal-other"), "mismatch"),
                (projection(), {**request(), "unexpected": True}, "invalid_request"),
                (nonfinite, request_from_projection(nonfinite), "mismatch"),
                (inverted, request_from_projection(inverted), "invalid_authority"),
            )
            for value, body, code in cases:
                with self.subTest(code=code):
                    if value is None:
                        path.unlink(missing_ok=True)
                    else:
                        path.write_text(json.dumps(value))
                    with self.assertRaises(ThermalActionControlError) as observed:
                        authority.authorize("hold", body)
                    self.assertEqual(code, observed.exception.code)

    def test_expired_graceful_hold_cannot_reset_its_deadline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "containment.json"
            path.write_text(json.dumps(projection()))
            authority = FileThermalActionAuthority(path, allow_legacy=True, now=lambda: 201.0)
            with self.assertRaises(ThermalActionControlError) as observed:
                authority.authorize("hold", request())
        self.assertEqual("deadline_expired", observed.exception.code)

    def test_hold_and_release_phases_are_operation_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "containment.json"
            authority = FileThermalActionAuthority(path, allow_legacy=True, now=lambda: 100.0)
            path.write_text(json.dumps(projection(phase="release_authorized")))
            with self.assertRaises(ThermalActionControlError) as observed:
                authority.authorize("hold", request(phase="release_authorized"))
            self.assertEqual("invalid_request", observed.exception.code)
            release = authority.authorize(
                "release", request(phase="release_authorized")
            )
            self.assertEqual("release_authorized", release.phase)

    def test_action_window_is_bounded_from_immutable_creation_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "containment.json"
            value = projection()
            value["overall_deadline_epoch"] = value["created_at_epoch"] + 301
            value["sleep_deadline_epoch"] = value["overall_deadline_epoch"]
            path.write_text(json.dumps(value))
            authority = FileThermalActionAuthority(path, allow_legacy=True, now=lambda: 100.0)
            with self.assertRaises(ThermalActionControlError) as observed:
                authority.authorize("hold", request_from_projection(value))
        self.assertEqual("invalid_authority", observed.exception.code)

    def test_urgent_hold_is_authorized_but_expired_urgent_work_is_not(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "containment.json"
            path.write_text(json.dumps(projection(phase="urgent_hold")))
            authority = FileThermalActionAuthority(path, allow_legacy=True, now=lambda: 100.0)
            action = authority.authorize(
                "hold", request(phase="urgent_hold")
            )
            self.assertEqual("urgent_hold", action.phase)
            expired = FileThermalActionAuthority(path, allow_legacy=True, now=lambda: 201.0)
            with self.assertRaises(ThermalActionControlError) as observed:
                expired.authorize("hold", request(phase="urgent_hold"))
        self.assertEqual("deadline_expired", observed.exception.code)

    def test_reloaded_deadline_change_is_visible_to_exact_reauthorization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "containment.json"
            value = projection()
            path.write_text(json.dumps(value))
            authority = FileThermalActionAuthority(path, allow_legacy=True, now=lambda: 100.0)
            original = authority.authorize("hold", request())
            value["sleep_deadline_epoch"] = 190.0
            path.write_text(json.dumps(value))
            changed = authority.authorize("hold", request_from_projection(value))
        self.assertNotEqual(original, changed)
        self.assertEqual(original.created_at_epoch, changed.created_at_epoch)

    def test_authority_input_is_size_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "containment.json"
            path.write_text("{" + "x" * 20000)
            authority = FileThermalActionAuthority(
                path, now=lambda: 100.0, maximum_bytes=4096
            )
            with self.assertRaises(ThermalActionControlError) as observed:
                authority.authorize("hold", request())
        self.assertEqual("invalid_authority", observed.exception.code)


    def test_held_sleeping_subset_proof_is_exact_and_operation_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "containment.json"
            value = projection(phase="held")
            value["sleeping_peer_engine_keys"] = [
                "qwen3.8-flash-next", "vision-vla"
            ]
            value["authorized_operations"] = [
                "hold", "sleeping_subset_proof"
            ]
            body = request_from_projection(
                value, operation="sleeping_subset_proof"
            )
            body["proof_engine_keys"] = list(
                value["sleeping_peer_engine_keys"]
            )
            path.write_text(json.dumps(value))
            authority = FileThermalActionAuthority(path, allow_legacy=True, now=lambda: 100.0)
            action = authority.authorize("sleeping_subset_proof", body)
            self.assertEqual("held", action.phase)
            self.assertEqual(
                ("qwen3.8-flash-next", "vision-vla"),
                action.proof_engine_keys,
            )
            self.assertEqual(
                (
                    "Qwen3-Embedding-8B", "qwen3.8-flash-next", "vision-vla"
                ),
                action.engine_keys,
            )

            for keys in (
                [],
                ["qwen3.8-flash-next"],
                ["qwen3.8-flash-next", "qwen3.8-flash-next"],
                ["Qwen3-Embedding-8B", "qwen3.8-flash-next", "vision-vla"],
                ["qwen3.8-flash-next", "vision-vla", "unknown"],
                ["vision-vla", "qwen3.8-flash-next"],
            ):
                with self.subTest(keys=keys):
                    changed = dict(body)
                    changed["proof_engine_keys"] = keys
                    with self.assertRaises(ThermalActionControlError):
                        authority.authorize("sleeping_subset_proof", changed)

            for field, malformed in (
                ("record_revision", True),
                ("generation", True),
                ("recovery_authorized", 0),
            ):
                with self.subTest(field=field, malformed=malformed):
                    changed = dict(body)
                    changed[field] = malformed
                    with self.assertRaises(ThermalActionControlError):
                        authority.authorize("sleeping_subset_proof", changed)

            duplicate_authority = json.dumps(value).replace(
                '"record_revision": 1,',
                '"record_revision": 1, "record_revision": 1,',
                1,
            )
            self.assertIn(
                '"record_revision": 1, "record_revision": 1,',
                duplicate_authority,
            )
            path.write_text(duplicate_authority)
            with self.assertRaises(ThermalActionControlError) as duplicate:
                authority.authorize("sleeping_subset_proof", body)
            self.assertEqual("invalid_authority", duplicate.exception.code)

            nonheld = projection()
            nonheld["sleeping_peer_engine_keys"] = list(
                value["sleeping_peer_engine_keys"]
            )
            nonheld["authorized_operations"] = [
                "hold", "sleeping_subset_proof"
            ]
            path.write_text(json.dumps(nonheld))
            changed = request_from_projection(
                nonheld, operation="sleeping_subset_proof"
            )
            changed["proof_engine_keys"] = list(
                nonheld["sleeping_peer_engine_keys"]
            )
            with self.assertRaises(ThermalActionControlError):
                authority.authorize("sleeping_subset_proof", changed)

    def test_v1_request_and_active_authority_are_rejected_without_downgrade(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "containment.json"
            path.write_text(json.dumps(projection()))
            authority = FileThermalActionAuthority(path, allow_legacy=True, now=lambda: 100.0)
            legacy = {"schema_version": 1, "action_id": "thermal-7", "phase": "graceful_hold"}
            with self.assertRaises(ThermalActionControlError) as observed:
                authority.authorize("hold", legacy)
            self.assertEqual("invalid_request", observed.exception.code)
            legacy_projection = projection()
            legacy_projection["schema_version"] = 1
            path.write_text(json.dumps(legacy_projection))
            with self.assertRaises(ThermalActionControlError) as observed:
                authority.authorize("hold", request())
            self.assertEqual("invalid_authority", observed.exception.code)

    def test_every_bound_lineage_revision_deadline_and_engine_field_is_exact(self) -> None:
        mutations = {
            "record_revision": 2,
            "incident_id": "incident-other",
            "action_id": "thermal-other",
            "generation": 2,
            "predecessor_action_id": "thermal-6",
            "transition_kind": "urgent_hold",
            "phase": "held",
            "containment_level": "urgent",
            "created_at_epoch": 91.0,
            "phase_updated_at_epoch": 91.0,
            "drain_deadline_epoch": 151.0,
            "sleep_deadline_epoch": 181.0,
            "overall_deadline_epoch": 201.0,
            "release_authorized_at_epoch": 95.0,
            "repair_deadline_epoch": 195.0,
            "recovery_authorized": True,
            "engine_keys": ["Qwen3-Embedding-8B"],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "containment.json"
            current = projection()
            path.write_text(json.dumps(current))
            authority = FileThermalActionAuthority(path, allow_legacy=True, now=lambda: 100.0)
            for key, changed in mutations.items():
                with self.subTest(key=key):
                    stale = request()
                    stale[key] = changed
                    with self.assertRaises(ThermalActionControlError) as observed:
                        authority.authorize("hold", stale)
                    self.assertEqual("mismatch", observed.exception.code)

    def test_repair_peer_proof_is_exact_non_target_release_subset(self) -> None:
        value = projection(
            phase="release_authorized",
            repair_target_engine_key="Qwen3-Embedding-8B",
            repair_target_status="starting",
            repair_target_deadline_epoch=150.0,
        )
        value["sleeping_peer_engine_keys"] = [
            "qwen3.8-flash-next", "vision-vla"
        ]
        body = request_from_projection(value, operation="repair_peer_proof")
        self.assertEqual(
            ["qwen3.8-flash-next", "vision-vla"],
            body["proof_engine_keys"],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "containment.json"
            path.write_text(json.dumps(value))
            authority = FileThermalActionAuthority(path, allow_legacy=True, now=lambda: 100.0)
            action = authority.authorize("repair_peer_proof", body)
        self.assertEqual("starting", action.repair_target_status)
        self.assertNotIn(
            action.repair_target_engine_key, action.proof_engine_keys
        )
        variants = []
        wrong_target = dict(body)
        wrong_target["repair_target_engine_key"] = "vision-vla"
        variants.append(wrong_target)
        target_included = dict(body)
        target_included["proof_engine_keys"] = [
            "Qwen3-Embedding-8B", "qwen3.8-flash-next", "vision-vla"
        ]
        variants.append(target_included)
        wrong_status = dict(body)
        wrong_status["repair_target_status"] = "converging"
        variants.append(wrong_status)
        bool_deadline = dict(body)
        bool_deadline["repair_target_deadline_epoch"] = True
        variants.append(bool_deadline)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "containment.json"
            path.write_text(json.dumps(value))
            authority = FileThermalActionAuthority(path, allow_legacy=True, now=lambda: 100.0)
            for changed in variants:
                with self.subTest(changed=changed):
                    with self.assertRaises(ThermalActionControlError):
                        authority.authorize("repair_peer_proof", changed)
            expired = dict(value)
            expired["repair_target_deadline_epoch"] = 99.0
            path.write_text(json.dumps(expired))
            with self.assertRaises(ThermalActionControlError):
                authority.authorize(
                    "repair_peer_proof",
                    request_from_projection(
                        expired, operation="repair_peer_proof"
                    ),
                )

    def test_repair_converge_is_exact_single_target_release_phase_only(self) -> None:
        value = projection(
            phase="release_authorized",
            repair_target_engine_key="Qwen3-Embedding-8B",
            repair_target_status="converging",
            repair_target_deadline_epoch=150.0,
        )
        body = request_from_projection(value, operation="repair_converge")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "containment.json"
            path.write_text(json.dumps(value))
            authority = FileThermalActionAuthority(path, allow_legacy=True, now=lambda: 100.0)
            action = authority.authorize("repair_converge", body)
            self.assertEqual(
                "Qwen3-Embedding-8B", action.repair_target_engine_key
            )
            self.assertEqual(150.0, action.repair_target_deadline_epoch)
            self.assertEqual(("Qwen3-Embedding-8B",), action.proof_engine_keys)
            variants = []
            wrong_target = dict(body)
            wrong_target["repair_target_engine_key"] = "vision-vla"
            variants.append(wrong_target)
            wrong_proof = dict(body)
            wrong_proof["proof_engine_keys"] = ["vision-vla"]
            variants.append(wrong_proof)
            extended = dict(body)
            extended["repair_target_deadline_epoch"] = 151.0
            variants.append(extended)
            wrong_status = dict(body)
            wrong_status["repair_target_status"] = "starting"
            variants.append(wrong_status)
            bool_deadline = dict(body)
            bool_deadline["repair_target_deadline_epoch"] = True
            variants.append(bool_deadline)
            for changed in variants:
                with self.subTest(changed=changed):
                    with self.assertRaises(ThermalActionControlError):
                        authority.authorize("repair_converge", changed)

            expired = dict(value)
            expired["repair_target_deadline_epoch"] = 99.0
            path.write_text(json.dumps(expired))
            with self.assertRaises(ThermalActionControlError):
                authority.authorize(
                    "repair_converge",
                    request_from_projection(expired, operation="repair_converge"),
                )

    def test_cutoff_recovery_authorizes_release_proof_only_with_null_containment_deadlines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "containment.json"
            value = projection(phase="cutoff_recovery_authorized")
            body = request(phase="cutoff_recovery_authorized")
            path.write_text(json.dumps(value))
            authority = FileThermalActionAuthority(path, allow_legacy=True, now=lambda: 100.0)
            action = authority.authorize("release", body)
            self.assertEqual("cutoff_recovery", action.transition_kind)
            self.assertIsNone(action.overall_deadline_epoch)
            wrong = dict(body)
            wrong["operation"] = "hold"
            with self.assertRaises(ThermalActionControlError) as observed:
                authority.authorize("hold", wrong)
            self.assertEqual("phase_not_allowed", observed.exception.code)

    def test_cutoff_recovery_and_ordinary_release_deadline_matrices_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "containment.json"
            authority = FileThermalActionAuthority(path, allow_legacy=True, now=lambda: 100.0)
            cases = []
            cutoff_with_containment = projection(phase="cutoff_recovery_authorized")
            cutoff_with_containment["overall_deadline_epoch"] = 150.0
            cases.append(cutoff_with_containment)
            ordinary_without_containment = projection(phase="release_authorized")
            ordinary_without_containment["drain_deadline_epoch"] = None
            cases.append(ordinary_without_containment)
            overlong_repair = projection(phase="release_authorized")
            overlong_repair["repair_deadline_epoch"] = overlong_repair["release_authorized_at_epoch"] + 5400.001
            cases.append(overlong_repair)
            for value in cases:
                with self.subTest(phase=value["phase"]):
                    path.write_text(json.dumps(value))
                    body = request_from_projection(value)
                    with self.assertRaises(ThermalActionControlError) as observed:
                        authority.authorize("release", body)
                    self.assertEqual("invalid_authority", observed.exception.code)


    def test_kind_generation_phase_and_level_matrix_rejects_impossible_authority(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "containment.json"
            authority = FileThermalActionAuthority(path, allow_legacy=True, now=lambda: 100.0)
            cases = []
            bad = projection()
            bad.update({"transition_kind": "hard_cutoff"})
            cases.append(("hold", bad))
            bad = projection(phase="release_authorized")
            bad.update({"transition_kind": "hard_cutoff"})
            cases.append(("release", bad))
            bad = projection(phase="cutoff_recovery_authorized")
            bad.update({
                "generation": 1, "predecessor_action_id": None,
            })
            cases.append(("release", bad))
            bad = projection(phase="cutoff_recovery_authorized")
            bad.update({
                "transition_kind": "graceful_hold",
                "containment_level": "graceful",
                "drain_deadline_epoch": 150.0,
                "sleep_deadline_epoch": 180.0,
                "overall_deadline_epoch": 200.0,
                "created_at_epoch": 90.0,
                "release_authorized_at_epoch": 95.0,
            })
            cases.append(("release", bad))
            bad = projection()
            bad["containment_level"] = "urgent"
            cases.append(("hold", bad))
            for operation, value in cases:
                with self.subTest(operation=operation, kind=value["transition_kind"],
                                  phase=value["phase"], level=value["containment_level"]):
                    value["authorized_operations"] = [operation]
                    path.write_text(json.dumps(value))
                    with self.assertRaises(ThermalActionControlError) as observed:
                        authority.authorize(
                            operation, request_from_projection(value, operation=operation)
                        )
                    self.assertEqual("invalid_authority", observed.exception.code)


if __name__ == "__main__":
    unittest.main()
