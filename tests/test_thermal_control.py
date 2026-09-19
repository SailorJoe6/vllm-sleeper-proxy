from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from vllm_sleeper_proxy.thermal_control import (
    FileThermalActionAuthority,
    ThermalActionControlError,
)


def projection(*, phase: str = "graceful_hold", action_id: str = "thermal-7") -> dict:
    return {
        "schema_version": 1,
        "created_at_epoch": 90.0,
        "generated_at_epoch": 99.0,
        "active": True,
        "state": "sleep",
        "phase": phase,
        "applied": False,
        "action_id": action_id,
        "reason_codes": ["acpi_temperature_requires_sleep"],
        "drain_deadline_epoch": 150.0,
        "sleep_deadline_epoch": 180.0,
        "overall_deadline_epoch": 200.0,
        "requirement": "REQ-MODEL-AVAIL-001",
    }


def request(*, phase: str = "graceful_hold", action_id: str = "thermal-7") -> dict:
    return {"schema_version": 1, "action_id": action_id, "phase": phase}


class FileThermalActionAuthorityTests(unittest.TestCase):
    def test_exact_active_hold_contract_is_authorized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "containment.json"
            path.write_text(json.dumps(projection()))
            authority = FileThermalActionAuthority(path, now=lambda: 100.0)
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
            authority = FileThermalActionAuthority(path, now=lambda: 100.0)
            cases = (
                (None, request(), "unavailable"),
                ({"schema_version": 1, "active": False}, request(), "inactive"),
                (projection(), request(action_id="thermal-other"), "mismatch"),
                (projection(), {**request(), "unexpected": True}, "invalid_request"),
                ({**projection(), "overall_deadline_epoch": float("nan")}, request(), "invalid_authority"),
                ({**projection(), "drain_deadline_epoch": 190.0, "sleep_deadline_epoch": 180.0}, request(), "invalid_authority"),
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
            authority = FileThermalActionAuthority(path, now=lambda: 201.0)
            with self.assertRaises(ThermalActionControlError) as observed:
                authority.authorize("hold", request())
        self.assertEqual("deadline_expired", observed.exception.code)

    def test_hold_and_release_phases_are_operation_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "containment.json"
            authority = FileThermalActionAuthority(path, now=lambda: 100.0)
            path.write_text(json.dumps(projection(phase="release_authorized")))
            with self.assertRaises(ThermalActionControlError) as observed:
                authority.authorize("hold", request(phase="release_authorized"))
            self.assertEqual("phase_not_allowed", observed.exception.code)
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
            authority = FileThermalActionAuthority(path, now=lambda: 100.0)
            with self.assertRaises(ThermalActionControlError) as observed:
                authority.authorize("hold", request())
        self.assertEqual("invalid_authority", observed.exception.code)

    def test_urgent_hold_is_authorized_but_expired_urgent_work_is_not(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "containment.json"
            path.write_text(json.dumps(projection(phase="urgent_hold")))
            authority = FileThermalActionAuthority(path, now=lambda: 100.0)
            action = authority.authorize(
                "hold", request(phase="urgent_hold")
            )
            self.assertEqual("urgent_hold", action.phase)
            expired = FileThermalActionAuthority(path, now=lambda: 201.0)
            with self.assertRaises(ThermalActionControlError) as observed:
                expired.authorize("hold", request(phase="urgent_hold"))
        self.assertEqual("deadline_expired", observed.exception.code)

    def test_reloaded_deadline_change_is_visible_to_exact_reauthorization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "containment.json"
            value = projection()
            path.write_text(json.dumps(value))
            authority = FileThermalActionAuthority(path, now=lambda: 100.0)
            original = authority.authorize("hold", request())
            value["sleep_deadline_epoch"] = 190.0
            path.write_text(json.dumps(value))
            changed = authority.authorize("hold", request())
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


if __name__ == "__main__":
    unittest.main()
