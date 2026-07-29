from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
