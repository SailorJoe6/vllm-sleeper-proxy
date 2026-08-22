from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from vllm_sleeper_proxy.admission import AdmissionError, FileAdmissionGuard
from vllm_sleeper_proxy.config import ModelConfig


MODEL = ModelConfig(
    name="chess-vlm-bootstrap",
    upstream_model="Qwen/Qwen3-VL-4B-Instruct",
    upstream_base_url="http://vision:8000/v1",
    control_base_url="http://vision:8000",
)


class AdmissionTests(unittest.TestCase):
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
