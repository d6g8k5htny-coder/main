"""Adverse controls for the accessible-list browser launch policy."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "atlas_accessible_list_browser_check.py"
TARGET = "tests.test_atlas_accessible_list_runner"


class AtlasAccessibleListMutationTests(unittest.TestCase):
    def assert_mutation_killed(self, before, after, expected_test):
        source = RUNNER.read_text()
        self.assertEqual(source.count(before), 1)
        mutated = source.replace(before, after)
        with tempfile.TemporaryDirectory() as raw:
            mutant = Path(raw) / RUNNER.name
            mutant.write_text(mutated)
            env = dict(os.environ)
            env["ATLAS_ACCESSIBLE_LIST_RUNNER"] = str(mutant)
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
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(expected_test, result.stdout + result.stderr)

    def test_wrong_default_browser_channel_is_killed(self):
        self.assert_mutation_killed(
            "launch['channel'] = 'chrome'",
            "launch['channel'] = 'chromium'",
            "test_default_browser_uses_sandbox_capable_chrome_channel",
        )

    def test_disabled_chromium_sandbox_is_killed(self):
        self.assert_mutation_killed(
            "launch = {'chromium_sandbox': True,",
            "launch = {'chromium_sandbox': False,",
            "test_default_browser_uses_sandbox_capable_chrome_channel",
        )


if __name__ == "__main__":
    unittest.main()
