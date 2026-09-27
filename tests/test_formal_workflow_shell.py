"""Regression controls for the actual formal workflow shell and failure summary."""
import os
from pathlib import Path
import re
import subprocess
import tempfile
import textwrap
import unittest

WORKFLOW = Path(__file__).resolve().parents[1] / '.github/workflows/formal-verification.yml'
PROBE_NAME = 'Verify actual shell rejects a failing producer'


def probe_body(text):
    pattern = r'      - name: ' + re.escape(PROBE_NAME) + r'\n        run: \|\n((?:          [^\n]*\n)+)'
    found = re.search(pattern, text)
    if found is None:
        raise ValueError('missing actual-shell negative control')
    return textwrap.dedent(found.group(1))


class FormalWorkflowShellTests(unittest.TestCase):
    def setUp(self):
        self.text = WORKFLOW.read_text()

    def test_explicit_bash_default_without_overrides(self):
        self.assertIn('defaults:\n  run:\n    shell: bash\n', self.text)
        self.assertEqual(re.findall(r'^\s+shell:\s*(.+)$', self.text, re.MULTILINE), ['bash'])

    def test_workflow_triggers_and_executes_regression(self):
        parent=(WORKFLOW.parent/'workspace-landing.yml').read_text()
        self.assertIn('  workflow_call:', self.text)
        self.assertIn('uses: ./.github/workflows/formal-verification.yml', parent)
        self.assertIn('  pull_request:', parent)
        self.assertIn('  push:', parent)
        self.assertNotIn('paths:', parent)
        self.assertIn('python -B -S -m unittest discover -s tests -p test_formal_workflow_shell.py -v', self.text)

    def run_probe(self, pipefail):
        body = probe_body(self.text)
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp)/'probe.sh'; script.write_text(body)
            cmd = ['bash', '--noprofile', '--norc', '-e']
            if pipefail:
                cmd += ['-o', 'pipefail']
            cmd.append(str(script))
            return subprocess.run(cmd, env={**os.environ, 'RUNNER_TEMP': tmp},
                                  capture_output=True, text=True, timeout=20)

    def test_actual_probe_accepts_fail_closed_shell(self):
        result = self.run_probe(True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertIn('PIPELINE_FAILURE_PROPAGATION_PASS', result.stdout)

    def test_actual_probe_rejects_original_fail_open_shell(self):
        result = self.run_probe(False)
        self.assertNotEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertIn('PIPELINE_FAILURE_PROPAGATION_FAIL', result.stdout)

    def test_summary_does_not_claim_success_unconditionally(self):
        self.assertIn('if [ "${{ job.status }}" = "success" ] && [ -s formal/.lake/formal-evidence/receipt.json ]; then', self.text)
        self.assertIn('No successful kernel-verification claim is made for this run.', self.text)


if __name__ == '__main__':
    unittest.main()
