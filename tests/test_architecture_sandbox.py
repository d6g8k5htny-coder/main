"""Actual runtime controls for the restricted hosted architecture container."""

import errno
import os
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(os.environ.get("ARCHITECTURE_SANDBOX") == "1", "requires hosted restricted container")
class ArchitectureSandbox(unittest.TestCase):
    def test_nonroot_and_no_new_privileges(self):
        self.assertEqual(os.getuid(), 65532)
        status = Path("/proc/self/status").read_text()
        self.assertIn("NoNewPrivs:\t1", status)
        self.assertIn("CapEff:\t0000000000000000", status)

    def test_source_mount_is_read_only_and_environment_has_no_publishing_secret(self):
        source_mounts = [line.split() for line in Path("/proc/self/mountinfo").read_text().splitlines()
                         if line.split()[4] == "/source"]
        self.assertEqual(len(source_mounts), 1)
        self.assertIn("ro", source_mounts[0][5].split(","))
        for key in ("GITHUB_TOKEN", "GH_TOKEN", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LEASE_SIGNING_KEY"):
            self.assertNotIn(key, os.environ)

    def test_network_namespace_has_no_external_interface_or_route(self):
        interfaces = {line.split(":", 1)[0].strip() for line in Path("/proc/net/dev").read_text().splitlines()[2:]}
        self.assertEqual(interfaces, {"lo"})
        self.assertEqual(len(Path("/proc/net/route").read_text().splitlines()), 1)

    def test_kernel_enforces_cpu_memory_and_process_limits(self):
        cgroup = Path("/sys/fs/cgroup")
        self.assertEqual((cgroup / "memory.max").read_text().strip(), str(1024 ** 3))
        self.assertEqual((cgroup / "pids.max").read_text().strip(), "128")
        quota, period = map(int, (cgroup / "cpu.max").read_text().split())
        self.assertEqual(quota, 2 * period)

    def test_output_quota_stops_disk_exhaustion_and_cleanup_recovers_space(self):
        before = os.statvfs("/output")
        self.assertLessEqual(before.f_blocks * before.f_frsize, 64 * 1024 ** 2)
        with tempfile.TemporaryDirectory(dir="/output") as folder:
            with open(Path(folder) / "quota-control", "wb", buffering=0) as stream:
                with self.assertRaises(OSError) as failure:
                    for _ in range(96):
                        stream.write(b"x" * 1024 ** 2)
                self.assertEqual(failure.exception.errno, errno.ENOSPC)
        self.assertGreater(os.statvfs("/output").f_bavail, 0)


if __name__ == "__main__":
    unittest.main()
