"""Requested local browser identity survives failure before browser launch."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class HostedBrowserFailureIdentityTests(unittest.TestCase):
    def test_explicit_browser_and_source_failure_write_a_truthful_report(self):
        # Removing the browser selector must not make a diagnostic disappear
        # in argparse before the source-unavailable report is written.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'tools').mkdir()
            (root / 'docs/site').mkdir(parents=True)
            script = root / 'tools/current_site_browser_check.py'
            shutil.copyfile(ROOT / 'tools/current_site_browser_check.py', script)
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            subprocess.run(['git', '-C', str(root), '-c', 'user.name=Fixture',
                            '-c', 'user.email=fixture@example.invalid',
                            '-c', 'core.hooksPath=/dev/null', 'commit',
                            '--allow-empty', '-qm', 'Fixture'], check=True)
            output = root / 'evidence'
            result = subprocess.run([sys.executable, '-B', '-S', str(script),
                                     '--browser', '/not-installed/atlas-browser',
                                     '--output', str(output)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue((output / 'report.json').is_file(), result.stderr)
            report = json.loads((output / 'report.json').read_text())
            self.assertEqual(report['browser_request']['kind'], 'executable')
            self.assertEqual(report['browser_request']['value'], '/not-installed/atlas-browser')
            self.assertTrue(report['browser_request']['chromium_sandbox'])
            self.assertTrue(report['error'])
            self.assertIsNone(report['source_payload_sha256'])
            self.assertEqual(report['cases'], [])
            self.assertEqual(report['scientific_effect'], 'NONE')


if __name__ == '__main__':
    unittest.main()
