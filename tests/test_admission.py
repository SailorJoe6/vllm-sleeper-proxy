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
        }

    def test_allows_fresh_model_specific_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_status(Path(directory), self.status())
            FileAdmissionGuard(path, now=lambda: 110.0)(MODEL)

    def test_denies_explicit_model_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_status(Path(directory), self.status(allowed=False))
            with self.assertRaisesRegex(AdmissionError, "measured_headroom"):
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
