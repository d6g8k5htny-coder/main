"""Process-level controls for the website-navigation browser runner."""

import os
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "website_navigation_browser_check.py"


class WebsiteNavigationBrowserRunnerTests(unittest.TestCase):
    def run_until_browser_launch(self, *, explicit_browser=False):
        with tempfile.TemporaryDirectory() as raw:
            work = Path(raw)
            binaries = work / "bin"
            package = work / "python" / "playwright"
            binaries.mkdir()
            package.mkdir(parents=True)

            chromium = binaries / "chromium"
            chromium.write_text("#!/bin/sh\nexit 0\n")
            chromium.chmod(0o755)
            git = binaries / "git"
            git.write_text("#!/bin/sh\nprintf '%040d\\n' 0\n")
            git.chmod(0o755)

            capture = work / "launch.json"
            (package / "__init__.py").write_text("")
            (package / "sync_api.py").write_text(
                textwrap.dedent(
                    f"""
                    from pathlib import Path

                    def expect(*_args, **_kwargs):
                        raise AssertionError("expect should not run before launch")

                    class Chromium:
                        def launch(self, **kwargs):
                            import json
                            Path({str(capture)!r}).write_text(json.dumps(kwargs, sort_keys=True))
                            raise SystemExit(73)

                    class Playwright:
                        chromium = Chromium()

                    class Context:
                        def __enter__(self):
                            return Playwright()
                        def __exit__(self, *_args):
                            return False

                    def sync_playwright():
                        return Context()
                    """
                )
            )

            env = dict(os.environ)
            env["PATH"] = str(binaries)
            env["PYTHONPATH"] = str(work / "python")
            command = [sys.executable, str(RUNNER), "--output", str(work / "output")]
            if explicit_browser:
                command.extend(["--browser", str(chromium)])
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

            return result, json.loads(capture.read_text())

    def test_default_browser_uses_sandbox_capable_chrome_channel(self):
        result, launch = self.run_until_browser_launch()
        self.assertEqual(result.returncode, 73, result.stderr)
        self.assertEqual(
            launch,
            {"channel": "chrome", "chromium_sandbox": True},
        )

    def test_explicit_browser_uses_requested_executable(self):
        result, launch = self.run_until_browser_launch(explicit_browser=True)
        self.assertEqual(result.returncode, 73, result.stderr)
        self.assertEqual(launch["chromium_sandbox"], True)
        self.assertTrue(launch["executable_path"].endswith("/bin/chromium"))


if __name__ == "__main__":
    unittest.main()
