"""Regression controls for the actual formal workflow shell and failure summary."""
import os
import shutil
import sys
from pathlib import Path
import re
import subprocess
import tempfile
import textwrap
import unittest
from unittest import mock

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
            # Hosted Python setup provides `python`; minimal external runners
            # may expose only `python3`. Test the unchanged shell with the same
            # running interpreter rather than changing its exit-code contract.
            (Path(tmp) / 'python').symlink_to(Path(sys.executable).resolve())
            env = {**os.environ, 'RUNNER_TEMP': tmp,
                   'PATH': tmp + os.pathsep + os.environ.get('PATH', '')}
            return subprocess.run(cmd, env=env, capture_output=True,
                                  text=True, timeout=20)

    def test_actual_probe_accepts_fail_closed_shell(self):
        result = self.run_probe(True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertIn('PIPELINE_FAILURE_PROPAGATION_PASS', result.stdout)

    def test_actual_probe_rejects_original_fail_open_shell(self):
        result = self.run_probe(False)
        self.assertNotEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertIn('PIPELINE_FAILURE_PROPAGATION_FAIL', result.stdout)

    def test_probe_when_parent_path_has_no_python_alias(self):
        # The shell still needs Bash and tee, but not a preinstalled `python`.
        with tempfile.TemporaryDirectory() as tool_dir:
            for command in ('bash', 'tee'):
                source = shutil.which(command)
                self.assertIsNotNone(source, f'{command} unavailable')
                (Path(tool_dir) / command).symlink_to(source)
            with mock.patch.dict(os.environ, {'PATH': tool_dir}):
                fail_closed = self.run_probe(True)
                fail_open = self.run_probe(False)
        self.assertEqual(fail_closed.returncode, 0,
                         fail_closed.stdout + fail_closed.stderr)
        self.assertIn('PIPELINE_FAILURE_PROPAGATION_PASS', fail_closed.stdout)
        self.assertNotEqual(fail_open.returncode, 0,
                            fail_open.stdout + fail_open.stderr)
        self.assertIn('PIPELINE_FAILURE_PROPAGATION_FAIL', fail_open.stdout)

    def test_summary_does_not_claim_success_unconditionally(self):
        self.assertIn('if [ "${{ job.status }}" = "success" ] && [ -s formal/.lake/formal-evidence/receipt.json ]; then', self.text)
        self.assertIn('No successful kernel-verification claim is made for this run.', self.text)


if __name__ == '__main__':
    unittest.main()
