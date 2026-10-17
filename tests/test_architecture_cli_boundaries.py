"""External CLI boundary regressions; execute only in hosted isolation.

The sibling package and its private-input canary are trusted, synthetic TEST
instruments. They never refer to an actual private capture or user file. Existing
fixture modules are imported as modules, so their TestCases are not rediscovered
or inherited by these four controls. A passing diagnostic is not the full suite.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import test_architecture_artifact_index as artifact_fixtures
import test_architecture_graph_artifact_readback as graph_fixtures


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_CLI = ROOT / "tools/artifact_index_export.py"
GRAPH_CLI = ROOT / "tools/architecture_graph_artifact_readback.py"
CHILD_DEADLINE_SECONDS = 20
PRIVATE_CANARY = b"TEST_ONLY_SYNTHETIC_PRIVATE_INPUT_CANARY"
# Trusted test instrument: the real stdout descriptor receives a real Python
# buffered stream. No exception, captured value or product result is fabricated.
BUFFERED_STDOUT_LAUNCHER = r'''
import io
import runpy
import sys
capacity, cli = int(sys.argv[1]), sys.argv[2]
sys.stdout = io.TextIOWrapper(
    io.BufferedWriter(io.FileIO(1, mode="wb", closefd=False), buffer_size=capacity),
    encoding="utf-8", newline="\n")
sys.argv = [cli, *sys.argv[3:]]
runpy.run_path(cli, run_name="__main__")
'''


class ArchitectureCliBoundaryContract(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="TEST-cli-boundary-")
        self.addCleanup(temporary.cleanup)
        self.cwd = Path(temporary.name)
        self.environment = dict(os.environ)
        for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONOPTIMIZE", "PYTHONUNBUFFERED"):
            self.environment.pop(name, None)

    def python_flags(self):
        flags = ["-I", "-B", "-S"]
        if sys.flags.optimize:
            flags.append("-" + "O" * sys.flags.optimize)
        return flags

    def invoke(self, command, raw=b"", *, closed_stdout=False):
        """Launch the real child, optionally with a pipe that has no reader.

        The reader is closed before Popen, so no child write or shutdown flush
        can race a parent read/close. Parent and child each close their writer.
        """
        writer = None
        if closed_stdout:
            reader, writer = os.pipe()
            os.close(reader)
        try:
            process = subprocess.Popen(
                command, stdin=subprocess.PIPE,
                stdout=writer if closed_stdout else subprocess.PIPE,
                stderr=subprocess.PIPE, cwd=self.cwd, env=self.environment,
            )
        finally:
            if writer is not None:
                os.close(writer)
        try:
            stdout, stderr = process.communicate(raw, timeout=CHILD_DEADLINE_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            # Killing the child closes its pipes; keep cleanup bounded as well.
            process.communicate(timeout=5)
            self.fail("CLI child exceeded its 20-second deadline")
        return subprocess.CompletedProcess(command, process.returncode, stdout or b"", stderr)

    def artifact_case(self):
        packet = artifact_fixtures.synthetic_packet()
        expected = artifact_fixtures.pins_for(packet)
        arguments = [part for name, flag in artifact_fixtures.PIN_FLAGS.items()
                     for part in (flag, expected[name])]
        return ARTIFACT_CLI, arguments, artifact_fixtures.canonical(packet) + b"\n"

    def graph_case(self):
        # Reuse the existing fixture builder and its source-pin assertions,
        # without inheriting or executing any of its test methods.
        fixture = graph_fixtures.ArchitectureGraphArtifactReadbackContract()
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        raw = graph_fixtures.classic_zip(fixture.entries())
        archive = fixture.cwd / "TEST-cli-boundary-input.zip"
        archive.write_bytes(raw)
        return GRAPH_CLI, fixture.arguments(archive, fixture.expected(raw)), b""

    def command(self, cli, arguments):
        self.assertTrue(cli.is_file(), "required external CLI is absent")
        return [sys.executable, *self.python_flags(), str(cli), *arguments]

    def accepted_baseline(self, case):
        cli, arguments, raw = case
        result = self.invoke(self.command(cli, arguments), raw)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, b"")
        document = json.loads(result.stdout)
        self.assertEqual(document["scientific_effect"], "NONE")
        self.assertIs(document["scientific_status_authority"], False)
        self.assertEqual(document["custody"], "unknown")
        return result.stdout

    def symlink_does_not_select_sibling_package(self, case):
        cli, arguments, raw = case
        baseline = self.accepted_baseline(case)
        attack_root = self.cwd / "TEST-attacker-source-root"
        tools = attack_root / "tools"
        package = attack_root / "architecture"
        tools.mkdir(parents=True)
        package.mkdir()
        secret = self.cwd / "TEST-synthetic-private-input"
        marker = self.cwd / "TEST-sibling-package-read-marker"
        secret.write_bytes(PRIVATE_CANARY)
        observed = PRIVATE_CANARY + b"\nOPTIMIZE=" + str(sys.flags.optimize).encode("ascii")
        package.joinpath("__init__.py").write_text(
            "from pathlib import Path\n"
            "import sys\n"
            "Path(" + repr(str(marker)) + ").write_bytes(\n"
            "    Path(" + repr(str(secret)) + ").read_bytes()\n"
            "    + b'\\nOPTIMIZE=' + str(sys.flags.optimize).encode('ascii'))\n"
            "raise RuntimeError('TEST_SYNTHETIC_SIBLING_PACKAGE_EXECUTED')\n",
            encoding="utf-8",
        )
        linked_cli = tools / cli.name
        linked_cli.symlink_to(cli)

        # Positive calibration deliberately imports this trusted TEST package.
        # It proves the marker detects a synthetic private read in this mode.
        calibration = self.invoke([
            sys.executable, *self.python_flags(), "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); import architecture",
            str(attack_root),
        ])
        self.record_diagnostic_child("synthetic-sibling-package-calibration", calibration)
        self.assertNotEqual(calibration.returncode, 0)
        self.assertIn(b"TEST_SYNTHETIC_SIBLING_PACKAGE_EXECUTED", calibration.stderr)
        self.assertEqual(marker.read_bytes(), observed, "sibling-package observer was not live")
        marker.unlink()

        result = self.invoke(self.command(linked_cli, arguments), raw)
        self.record_diagnostic_child("symlinked-cli-with-synthetic-sibling", result)
        self.assertEqual(secret.read_bytes(), PRIVATE_CANARY)
        self.assertFalse(marker.exists(), "symlink-selected sibling package read the synthetic private input")
        self.assertNotIn(PRIVATE_CANARY, result.stdout + result.stderr)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, b"")
        self.assertEqual(result.stdout, baseline, "symlink invocation changed the validated public unit")

    def record_diagnostic_child(self, variant, result):
        # Only the focused hosted workflow sets this path. Do not print between
        # unittest's progress and result tokens in the required full-suite log.
        report = self.environment.get("ARCHITECTURE_CLI_BOUNDARY_REPORT")
        if report is not None:
            record = {
                "test_id": self.id(), "variant": variant,
                "optimize": sys.flags.optimize, "returncode": result.returncode,
                "stdout_bytes": len(result.stdout),
                "stderr": result.stderr.decode("utf-8", errors="backslashreplace"),
            }
            with open(report, "ab") as stream:
                stream.write((json.dumps(record, sort_keys=True) + "\n").encode("utf-8"))

    def assert_opaque_refusal(self, result, refusal):
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(result.stdout, b"")
        self.assertEqual(result.stderr, refusal)

    def broken_stdout_has_opaque_refusal(self, case, refusal, *, buffered_flush=False):
        # This positive run ensures an unrelated argument/fixture refusal cannot
        # satisfy the closed-reader protocol assertion below.
        baseline = self.accepted_baseline(case)
        cli, arguments, raw = case
        with self.subTest(stdout_boundary="direct CLI"):
            result = self.invoke(self.command(cli, arguments), raw, closed_stdout=True)
            self.record_diagnostic_child("direct-closed-reader", result)
            self.assert_opaque_refusal(result, refusal)
        if buffered_flush:
            # The graph unit can exceed Python's default buffer and fail during
            # direct write, leaving no bytes for shutdown to flush. A larger
            # real buffer exercises flush-time failure instead. Its capacity is
            # derived from the already accepted bounded synthetic unit only.
            capacity = len(baseline) + 1
            self.assertLessEqual(capacity, 16 * 1024 * 1024 + 1)
            command = [sys.executable, *self.python_flags(), "-c", BUFFERED_STDOUT_LAUNCHER,
                       str(capacity), str(cli), *arguments]
            calibration = self.invoke(command, raw)
            self.record_diagnostic_child("buffered-open-reader-calibration", calibration)
            self.assertEqual(calibration.returncode, 0, calibration.stderr)
            self.assertEqual(calibration.stderr, b"")
            self.assertEqual(calibration.stdout, baseline)
            with self.subTest(stdout_boundary="real buffered flush"):
                result = self.invoke(command, raw, closed_stdout=True)
                self.record_diagnostic_child("buffered-closed-reader", result)
                self.assert_opaque_refusal(result, refusal)

    def test_artifact_index_symlink_cannot_read_synthetic_private_input(self):
        self.symlink_does_not_select_sibling_package(self.artifact_case())

    def test_graph_readback_symlink_cannot_read_synthetic_private_input(self):
        self.symlink_does_not_select_sibling_package(self.graph_case())

    def test_artifact_index_broken_stdout_has_one_opaque_refusal_and_exit_one(self):
        self.broken_stdout_has_opaque_refusal(self.artifact_case(), b"ARTIFACT_INDEX_REFUSED\n")

    def test_graph_readback_broken_stdout_has_one_opaque_refusal_and_exit_one(self):
        self.broken_stdout_has_opaque_refusal(
            self.graph_case(), b"ARCHITECTURE_GRAPH_READBACK_REFUSED\n", buffered_flush=True)


if __name__ == "__main__":
    unittest.main()
