"""Adverse controls for the source-bridge browser admission and launch policy."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "atlas_source_bridge_browser_check.py"
TARGET = "tests.test_atlas_source_bridge_runner"


class AtlasSourceBridgeMutationTests(unittest.TestCase):
    def assert_mutation_killed(self, before, after, expected_test):
        source = RUNNER.read_text()
        self.assertEqual(source.count(before), 1)
        mutated = source.replace(before, after)
        with tempfile.TemporaryDirectory() as raw:
            mutant = Path(raw) / RUNNER.name
            mutant.write_text(mutated)
            env = dict(os.environ)
            env["ATLAS_SOURCE_BRIDGE_RUNNER"] = str(mutant)
            command = [sys.executable, "-B"]
            if sys.flags.optimize:
                command.append("-O")
            command.extend(["-S", "-m", "unittest", TARGET, "-v"])
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
        combined = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0, combined)
        self.assertIn(expected_test, combined)

    def test_wrong_default_browser_channel_is_killed(self):
        self.assert_mutation_killed(
            "else:launch['channel']='chrome'",
            "else:launch['channel']='chromium'",
            "test_current_site_reaches_sandboxed_default_chrome_launch",
        )

    def test_disabled_chromium_sandbox_is_killed(self):
        self.assert_mutation_killed(
            "launch={'chromium_sandbox':True,",
            "launch={'chromium_sandbox':False,",
            "test_current_site_reaches_sandboxed_default_chrome_launch",
        )

    def test_missing_exact_site_manifest_profile_is_killed(self):
        self.assert_mutation_killed(
            "'8d43188691c695f66e233727322484da22662a58':{",
            "'f8d43188691c695f66e233727322484da22662a5':{",
            "test_current_site_reaches_sandboxed_default_chrome_launch",
        )


if __name__ == "__main__":
    unittest.main()
