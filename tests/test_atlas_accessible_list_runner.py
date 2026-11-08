"""Process-level controls for the Atlas accessible-list browser check."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = Path(
    os.environ.get(
        "ATLAS_ACCESSIBLE_LIST_RUNNER",
        ROOT / "tools" / "atlas_accessible_list_browser_check.py",
    )
)


class AtlasAccessibleListRunnerTests(unittest.TestCase):
    def run_until_browser_launch(self, *, explicit_browser=False):
        with tempfile.TemporaryDirectory() as raw:
            work = Path(raw)
            package = work / "python" / "playwright"
            package.mkdir(parents=True)
            browser = work / "browser"
            browser.write_text("#!/bin/sh\nexit 0\n")
            browser.chmod(0o755)
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
            output = work / "output"
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join(
                [str(work / "python"), str(ROOT / "tools")]
            )
            command = [sys.executable]
            if sys.flags.optimize:
                command.append("-O")
            command.extend(
                [
                    str(RUNNER),
                    "--site",
                    str(ROOT / "docs" / "site"),
                    "--output",
                    str(output),
                ]
            )
            if explicit_browser:
                command.extend(["--browser", str(browser)])
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            launch = json.loads(capture.read_text()) if capture.exists() else None
            report_path = output / "contrast-browser.json"
            report = json.loads(report_path.read_text()) if report_path.exists() else None
            return result, launch, browser, report

    def test_default_browser_uses_sandbox_capable_chrome_channel(self):
        result, launch, _browser, _report = self.run_until_browser_launch()
        self.assertEqual(result.returncode, 73, result.stderr)
        self.assertEqual(
            launch,
            {
                "args": ["--enable-unsafe-swiftshader"],
                "channel": "chrome",
                "chromium_sandbox": True,
            },
        )

    def test_explicit_browser_uses_requested_executable(self):
        result, launch, browser, _report = self.run_until_browser_launch(
            explicit_browser=True
        )
        self.assertEqual(result.returncode, 73, result.stderr)
        self.assertEqual(
            launch,
            {
                "args": ["--enable-unsafe-swiftshader"],
                "chromium_sandbox": True,
                "executable_path": str(browser),
            },
        )

    def test_report_records_default_launch_boundary_before_browser_work(self):
        result, _launch, _browser, report = self.run_until_browser_launch()
        self.assertEqual(result.returncode, 73, result.stderr)
        self.assertEqual(
            report["browser_launch"],
            {
                "args": ["--enable-unsafe-swiftshader"],
                "channel": "chrome",
                "chromium_sandbox": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
