from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from vllm_sleeper_proxy.admission import (
    AdmissionError,
    FileAdmissionGuard,
    FileThermalAdmissionGuard,
    ThermalCooldownError,
)
from vllm_sleeper_proxy.config import ModelConfig, load_models_from_env


MODEL = ModelConfig(
    name="chess-vlm-bootstrap",
    upstream_model="Qwen/Qwen3-VL-4B-Instruct",
    upstream_base_url="http://vision:8000/v1",
    control_base_url="http://vision:8000",
)
REQUIRED_MODEL = ModelConfig(
    name=MODEL.name,
    upstream_model=MODEL.upstream_model,
    upstream_base_url=MODEL.upstream_base_url,
    control_base_url=MODEL.control_base_url,
    required=True,
)


class AdmissionTests(unittest.TestCase):
    def test_required_flag_is_explicitly_parsed_and_type_checked(self) -> None:
        payload = [{
            "name": "required-model",
            "upstream_base_url": "http://engine:8000/v1",
            "control_base_url": "http://engine:8000",
            "required": True,
        }]
        with patch.dict(os.environ, {"SLEEPER_MODELS": json.dumps(payload)}):
            self.assertTrue(load_models_from_env()[0].required)
        payload[0]["required"] = "true"
        with patch.dict(os.environ, {"SLEEPER_MODELS": json.dumps(payload)}):
            with self.assertRaisesRegex(ValueError, "required must be boolean"):
                load_models_from_env()

    def write_status(self, root: Path, payload: dict) -> Path:
        path = root / "status.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def status(self, *, allowed: bool = True, generated_at: float = 100.0) -> dict:
        return {
            "generated_at_epoch": generated_at,
            "max_age_seconds": 30,
            "model_admission": {
                MODEL.name: {"allowed": allowed, "reason": "measured_headroom"}
            },
            "workload_policy": {"model_dependencies": [MODEL.name]},
        }

    def thermal_status(
        self, *, state: str = "healthy", allowed: bool = True,
        generated_at: float = 100.0, retry_after: int = 60,
    ) -> dict:
        return {
            "schema_version": 1,
            "generated_at_epoch": generated_at,
            "max_age_seconds": 5,
            "allowed": allowed,
            "state": state,
            "reason": (
                "observability_degraded"
                if state == "telemetry_failure" and allowed
                else "allowed" if allowed else "thermal_cooldown"
            ),
            "reason_codes": [] if allowed else ["acpi_temperature_denies_new_wake"],
            "retry_after_seconds": retry_after,
            "sequence": 1,
        }

    def test_fast_thermal_guard_allows_only_fresh_consistent_healthy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_status(Path(directory), self.thermal_status())
            FileThermalAdmissionGuard(path, now=lambda: 104.0)(MODEL)
            for state in ("warning", "sleep", "recovering", "cutoff", "telemetry_failure"):
                with self.subTest(state=state):
                    allowed = state in {"warning", "telemetry_failure"}
                    path.write_text(json.dumps(self.thermal_status(state=state, allowed=allowed)))
                    with self.assertRaises(ThermalCooldownError) as caught:
                        FileThermalAdmissionGuard(path, now=lambda: 104.0)(MODEL)
                    self.assertEqual(state, caught.exception.state)
                    self.assertEqual(60, caught.exception.retry_after_seconds)
                    self.assertIn("thermal_cooldown", str(caught.exception))

    def test_fast_thermal_guard_fails_closed_for_missing_stale_future_and_malformed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "missing.json"
            with self.assertRaises(ThermalCooldownError):
                FileThermalAdmissionGuard(missing, now=lambda: 100.0)(MODEL)
            cases = [
                self.thermal_status(generated_at=90.0),
                self.thermal_status(generated_at=103.0),
                {**self.thermal_status(), "schema_version": 2},
                {**self.thermal_status(), "schema_version": True},
                {**self.thermal_status(), "allowed": "true"},
                {**self.thermal_status(), "reason": "thermal_cooldown"},
                {**self.thermal_status(), "state": "healthy", "allowed": False},
                {**self.thermal_status(), "reason_codes": ["unsafe reason"]},
                {**self.thermal_status(), "retry_after_seconds": True},
            ]
            for index, payload in enumerate(cases):
                with self.subTest(index=index):
                    path = self.write_status(root, payload)
                    with self.assertRaises(ThermalCooldownError):
                        FileThermalAdmissionGuard(path, now=lambda: 100.0)(MODEL)
            for malformed in ("not json", "[]", "null", '"string"'):
                path.write_text(malformed)
                with self.assertRaises(ThermalCooldownError):
                    FileThermalAdmissionGuard(path, now=lambda: 100.0)(MODEL)

    def test_required_thermal_guard_allows_monitor_uncertainty_and_warning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "missing.json"
            FileThermalAdmissionGuard(missing, now=lambda: 100.0)(REQUIRED_MODEL)
            containment = root / "containment.json"
            containment.write_text(json.dumps({"active": True, "state": "cutoff"}))
            with self.assertRaises(ThermalCooldownError):
                FileThermalAdmissionGuard(
                    missing,
                    containment_path=containment,
                    now=lambda: 100.0,
                )(REQUIRED_MODEL)
            containment.write_text(json.dumps({"active": False, "state": "none"}))
            path = self.write_status(root, self.thermal_status(generated_at=1.0))
            FileThermalAdmissionGuard(path, now=lambda: 100.0)(REQUIRED_MODEL)
            path.write_text("not json", encoding="utf-8")
            FileThermalAdmissionGuard(path, now=lambda: 100.0)(REQUIRED_MODEL)
            for state, reason_codes in (
                ("warning", ["acpi_temperature_denies_new_wake"]),
                ("telemetry_failure", ["thermal_telemetry_unavailable"]),
            ):
                payload = self.thermal_status(state=state, allowed=True)
                payload["reason_codes"] = reason_codes
                path.write_text(json.dumps(payload), encoding="utf-8")
                FileThermalAdmissionGuard(path, now=lambda: 104.0)(REQUIRED_MODEL)

    def test_required_thermal_guard_still_denies_affirmative_containment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for state in ("sleep", "recovering", "cutoff"):
                path = self.write_status(
                    root, self.thermal_status(state=state, allowed=False)
                )
                with self.assertRaises(ThermalCooldownError):
                    FileThermalAdmissionGuard(path, now=lambda: 104.0)(
                        REQUIRED_MODEL
                    )
            stale_cutoff = self.thermal_status(
                state="cutoff", allowed=False, generated_at=1.0
            )
            path.write_text(json.dumps(stale_cutoff), encoding="utf-8")
            with self.assertRaises(ThermalCooldownError):
                FileThermalAdmissionGuard(path, now=lambda: 104.0)(REQUIRED_MODEL)

    def test_required_resource_guard_allows_uncertainty_but_not_explicit_danger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            FileAdmissionGuard(root / "missing.json", now=lambda: 100.0)(
                REQUIRED_MODEL
            )
            path = self.write_status(root, self.status(generated_at=1.0))
            FileAdmissionGuard(path, now=lambda: 100.0)(REQUIRED_MODEL)
            uncertain = self.status(allowed=False)
            uncertain["model_admission"][MODEL.name]["reason"] = "unknown_model_state"
            path.write_text(json.dumps(uncertain), encoding="utf-8")
            FileAdmissionGuard(path, now=lambda: 110.0)(REQUIRED_MODEL)
            danger = self.status(allowed=False)
            danger["model_admission"][MODEL.name]["reason"] = "host_state_critical"
            path.write_text(json.dumps(danger), encoding="utf-8")
            with self.assertRaisesRegex(AdmissionError, "host_state_critical"):
                FileAdmissionGuard(path, now=lambda: 110.0)(REQUIRED_MODEL)
            danger["generated_at_epoch"] = 1.0
            path.write_text(json.dumps(danger), encoding="utf-8")
            with self.assertRaisesRegex(AdmissionError, "stale"):
                FileAdmissionGuard(path, now=lambda: 110.0)(REQUIRED_MODEL)

    def test_allows_fresh_model_specific_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_status(Path(directory), self.status())
            FileAdmissionGuard(path, now=lambda: 110.0)(MODEL)

    def test_refreshes_transient_unknown_model_state_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            denied = self.status(allowed=False)
            denied["model_admission"][MODEL.name]["reason"] = "unknown_model_state"
            path = self.write_status(root, denied)
            def refresh(_: float) -> None:
                path.write_text(json.dumps(self.status(allowed=True)), encoding="utf-8")
            FileAdmissionGuard(path, now=lambda: 110.0, sleep=refresh)(MODEL)

    def test_unknown_model_state_remains_fail_closed_after_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            denied = self.status(allowed=False)
            denied["model_admission"][MODEL.name]["reason"] = "unknown_model_state"
            path = self.write_status(Path(directory), denied)
            with self.assertRaisesRegex(AdmissionError, "unknown_model_state"):
                FileAdmissionGuard(path, now=lambda: 110.0, sleep=lambda _: None)(MODEL)

    def test_denies_explicit_model_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_status(Path(directory), self.status(allowed=False))
            with self.assertRaisesRegex(AdmissionError, "measured_headroom"):
                FileAdmissionGuard(path, now=lambda: 110.0)(MODEL)

    def test_qwen_fails_closed_without_gpu_telemetry(self) -> None:
        qwen = ModelConfig(
            name="qwen38-27b-nvfp4",
            upstream_model="unsloth/Qwen3.8-27B-NVFP4",
            upstream_base_url="http://qwen:8000/v1",
            control_base_url="http://qwen:8000",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_status(Path(directory), self.status())
            status = json.loads(path.read_text())
            status["workload_policy"]["model_dependencies"].append(qwen.name)
            status["model_admission"][qwen.name] = {"allowed": True}
            path.write_text(json.dumps(status))
            with self.assertRaisesRegex(AdmissionError, "unavailable"):
                FileAdmissionGuard(path, now=lambda: 110.0)(qwen)

    def test_fails_closed_when_dependency_metadata_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            status = self.status()
            del status["workload_policy"]
            path = self.write_status(Path(directory), status)
            with self.assertRaisesRegex(AdmissionError, "unavailable"):
                FileAdmissionGuard(path, now=lambda: 110.0)(MODEL)

    def test_fails_closed_for_stale_missing_or_invalid_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stale = self.write_status(root, self.status(generated_at=1.0))
            with self.assertRaisesRegex(AdmissionError, "stale"):
                FileAdmissionGuard(stale, now=lambda: 100.0)(MODEL)
            stale.write_text("not json", encoding="utf-8")
            with self.assertRaisesRegex(AdmissionError, "unavailable"):
                FileAdmissionGuard(stale, now=lambda: 100.0)(MODEL)
            with self.assertRaisesRegex(AdmissionError, "unavailable"):
                FileAdmissionGuard(root / "missing.json", now=lambda: 100.0)(MODEL)


if __name__ == "__main__":
    unittest.main()
