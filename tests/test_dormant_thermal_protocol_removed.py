import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DormantThermalProtocolRemovalTests(unittest.TestCase):
    def test_removed_modules_and_routes_stay_absent(self):
        self.assertFalse((ROOT / "vllm_sleeper_proxy/thermal_control.py").exists())
        source = "\n".join(
            (ROOT / relative).read_text(encoding="utf-8")
            for relative in (
                "README.md",
                "docs/requirements.md",
                "vllm_sleeper_proxy/manager.py",
                "vllm_sleeper_proxy/server.py",
            )
        )
        for forbidden in (
            "/thermal/actions/",
            "/thermal/projection/",
            "SLEEPER_THERMAL_ACTION_CONTROL_ENABLED",
            "SLEEPER_THERMAL_ACTION_STATUS_PATH",
            "thermal_sleeping_subset_proof",
            "thermal_repair_peer_proof",
            "thermal_repair_converge",
            "projection_quiesce",
            "projection_activate",
            "activation_token",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_retained_thermal_and_lifecycle_foundations_stay_present(self):
        admission = (ROOT / "vllm_sleeper_proxy/admission.py").read_text(encoding="utf-8")
        manager = (ROOT / "vllm_sleeper_proxy/manager.py").read_text(encoding="utf-8")
        server = (ROOT / "vllm_sleeper_proxy/server.py").read_text(encoding="utf-8")
        self.assertIn("FileThermalAdmissionGuard", admission)
        self.assertIn("thermal_protection_active", server)
        self.assertIn("SLEEPER_THERMAL_ADMISSION_STATUS_PATH", server)
        self.assertIn("/sleep", server)
        self.assertIn("sleep_level: int = 2", manager)
        self.assertIn("startup_lease", manager)


if __name__ == "__main__":
    unittest.main()
