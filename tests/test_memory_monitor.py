from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from vllm_sleeper_proxy import memory_monitor
from vllm_sleeper_proxy.memory_monitor import memory_utilization


class MemoryMonitorTests(unittest.TestCase):
    def test_uses_memavailable_for_total_host_utilization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meminfo"
            path.write_text(
                "MemTotal:       100000 kB\n"
                "MemAvailable:    15000 kB\n",
                encoding="utf-8",
            )
            self.assertEqual(85.0, memory_utilization(path))

    def test_default_critical_threshold_is_95_percent(self) -> None:
        with patch("sys.argv", ["memory_monitor"]), patch.object(
            memory_monitor, "monitor"
        ) as monitor:
            self.assertEqual(0, memory_monitor.main())
        monitor.assert_called_once()
        self.assertEqual(95.0, monitor.call_args.kwargs["critical_percent"])

    def test_critical_threshold_cannot_exceed_95_percent(self) -> None:
        with patch("sys.argv", ["memory_monitor", "--critical-percent", "96"]):
            with self.assertRaises(SystemExit):
                memory_monitor.main()


if __name__ == "__main__":
    unittest.main()
