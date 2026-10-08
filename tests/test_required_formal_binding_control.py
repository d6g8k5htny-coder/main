"""Proposed W8b control (Grok Bot agent 2): tools/required_formal_check.py must bind a
formal receipt to the checked-out commit and to the current workflow run, refuse
symlinked receipt/manifest evidence, and report CLI refusals as its own handled
refusal (exit 1 plus one REQUIRED_FORMAL_CHECK_FAILED line), not as an arbitrary
crash. Positive cases: a valid binding and a valid aggregate are accepted, so a
reject-everything implementation fails. Negative cases assert exact messages."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'tools/required_formal_check.py'
SPEC = importlib.util.spec_from_file_location('required_check_control', SCRIPT)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class RequiredFormalBindingControl(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.env = {'GITHUB_SHA': 'a' * 40, 'GITHUB_REPOSITORY': 'd6g8k5htny-coder/main',
                    'GITHUB_RUN_ID': '123', 'GITHUB_RUN_ATTEMPT': '2'}
        self.manifest = self.root / 'manifest.json'
        self.manifest.write_text('{"source":"fixture"}\n')
        self.log = self.root / 'build.log'
        self.log.write_text('fixture build log\n')
        self.receipt = self.root / 'receipt.json'
        self.record = {'checked_commit': 'a' * 40, 'repository': 'd6g8k5htny-coder/main',
                       'workflow_run_id': '123', 'formalization_status': 'kernel-checked',
                       'scientific_effect': 'NONE',
                       'manifest_sha256': hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
                       'logs': {'build.log': hashlib.sha256(self.log.read_bytes()).hexdigest()}}
        self.needs = {'checks': {'result': 'success', 'outputs': {'checked_commit': 'a' * 40}},
                      'formal': {'result': 'success', 'outputs': {
                          'checked_commit': 'a' * 40, 'repository': 'd6g8k5htny-coder/main',
                          'run_id': '123', 'run_attempt': '2', 'receipt_sha256': 'b' * 64}}}

    def bind(self, head='a' * 40, receipt=None, manifest=None):
        self.receipt.write_text(json.dumps(self.record))
        return M.bind_receipt(receipt or self.receipt, manifest or self.manifest, self.env, head)

    def refused(self, message, **kwargs):
        with self.assertRaises(ValueError) as caught:
            self.bind(**kwargs)
        self.assertEqual(str(caught.exception), message)

    def cli(self, needs):
        env = {**os.environ, **self.env, 'REQUIRED_FORMAL_NEEDS': json.dumps(needs)}
        env.pop('PYTHONOPTIMIZE', None)
        return subprocess.run([sys.executable, '-B', *(['-O'] if sys.flags.optimize else []), '-S', str(SCRIPT), 'aggregate'],
                              env=env, capture_output=True, text=True, timeout=30)

    # --- positive -----------------------------------------------------------------
    def test_valid_receipt_binds_to_commit_run_and_bytes(self):
        result = self.bind()
        self.assertEqual((result['checked_commit'], result['repository'], result['run_id'], result['run_attempt']),
                         ('a' * 40, 'd6g8k5htny-coder/main', '123', '2'))
        self.assertEqual(result['receipt_sha256'], hashlib.sha256(self.receipt.read_bytes()).hexdigest())

    def test_cli_accepts_valid_aggregate(self):
        good = self.cli(self.needs)
        self.assertEqual((good.returncode, good.stderr), (0, ''))
        self.assertEqual(json.loads(good.stdout)['conclusion'], 'success')

    # --- negatives: exact handled refusals ------------------------------------------
    def test_checkout_other_than_tested_commit_is_refused(self):
        self.refused('checkout/tested commit mismatch', head='c' * 40)

    def test_receipt_from_another_run_is_refused(self):
        self.record['workflow_run_id'] = '122'
        self.refused('receipt mismatch: workflow_run_id')

    def test_receipt_from_another_repository_is_refused(self):
        self.record['repository'] = 'd6g8k5htny-coder/Math-'
        self.refused('receipt mismatch: repository')

    def test_symlinked_receipt_or_manifest_is_refused(self):
        real = self.root / 'real-receipt.json'
        real.write_text(json.dumps(self.record))
        link = self.root / 'link-receipt.json'
        link.symlink_to(real)
        with self.assertRaises(ValueError) as caught:
            M.bind_receipt(link, self.manifest, self.env, 'a' * 40)
        self.assertEqual(str(caught.exception), 'symlinked evidence')
        manifest_link = self.root / 'link-manifest.json'
        manifest_link.symlink_to(self.manifest)
        self.refused('symlinked evidence', manifest=manifest_link)

    def test_cli_refusal_is_the_handled_refusal_not_a_crash(self):
        self.needs['formal']['result'] = 'failure'
        bad = self.cli(self.needs)
        self.assertEqual(bad.returncode, 1)
        self.assertEqual(bad.stderr, 'REQUIRED_FORMAL_CHECK_FAILED: dependency did not succeed: formal\n')
        self.assertEqual(bad.stdout, '')


if __name__ == '__main__':
    unittest.main()
