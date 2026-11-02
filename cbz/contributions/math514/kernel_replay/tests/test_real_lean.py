"""Actual installed Lean 4.34.1 tests; no project proof replay."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
LEAN = Path(os.environ.get('CBZ_TEST_LEAN', '/home/agent/.elan/toolchains/leanprover--lean4---v4.34.1/bin/lean'))


class RealLeanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('kernel_replay', ROOT / 'kernel_replay.py')
        cls.kr = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.kr)
        cls.base = ROOT / 'evidence/compiled-fixtures'
        cls.base.mkdir(exist_ok=True)
        cls.fixtures = ROOT / 'tests/fixtures'
        for name in ('Good', 'BadPublic', 'BadPrivate', 'UnapprovedAxiom'):
            p = subprocess.run([str(LEAN), '-o', str(cls.base / (name + '.olean')),
                                str(cls.fixtures / (name + '.lean'))], cwd=cls.fixtures,
                               capture_output=True, text=True, timeout=60)
            if p.returncode:
                raise AssertionError(name + ': ' + p.stdout + p.stderr)

    def replay(self, name):
        m = self.kr.prepare_manifest(str(LEAN), [str(self.base)],
                [f'{name}={self.fixtures / (name + ".lean")}'], [])
        r = self.kr.run_manifest(m, 8192, 60)
        (ROOT / 'evidence' / ('real-' + name + '.json')).write_text(json.dumps(r, indent=2) + '\n')
        return r

    def test_good_public_private_inductive_all_parts(self):
        r = self.replay('Good')
        self.assertEqual(len(r['manifest']['modules'][0]['parts']), 3)
        self.assertTrue(r['ok'], r)
        self.assertEqual(r['completed_count'], 1)

    def test_public_self_reference_rejected(self):
        r = self.replay('BadPublic')
        self.assertFalse(r['ok'], r)
        self.assertEqual(r['results'][0]['status'], 'FAIL')
        self.assertIn('malformedPublic', r['results'][0]['stdout'])

    def test_private_self_reference_rejected(self):
        r = self.replay('BadPrivate')
        self.assertFalse(r['ok'], r)
        self.assertIn('malformedPrivate', r['results'][0]['stdout'])

    def test_unapproved_axiom_kernel_valid_but_independent_policy_rejects(self):
        r = self.replay('UnapprovedAxiom')
        self.assertTrue(r['ok'], r)
        self.assertEqual(r['g2_status'], 'NOT_ASSESSED')
        check = self.base / 'FixtureAxiomPolicy.lean'
        check.write_text('import Lean\nimport UnapprovedAxiom\nopen Lean Elab Command\n'
            'run_cmd do\n'
            '  let some (.axiomInfo _) := (← getEnv).find? `unapprovedKernelReplayFixture\n'
            '    | throwError "fixture missing its axiom"\n'
            '  throwError "G2 fixture policy: unapproved axiom unapprovedKernelReplayFixture"\n')
        env = os.environ.copy()
        env['LEAN_PATH'] = str(self.base)
        p = subprocess.run([str(LEAN), str(check)], cwd=self.base, env=env,
                           capture_output=True, text=True, timeout=60)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('G2 fixture policy: unapproved axiom', p.stdout)
        (ROOT / 'evidence' / 'fixture-g2-rejection.json').write_text(json.dumps({
            'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr,
            'scope': 'one explicit unapproved fixture axiom; not a production all-declaration G2 auditor'
        }, indent=2) + '\n')

    def test_exact_module_no_prefix_glob(self):
        # A bad submodule must not be replayed when checking the exact Good module.
        nested = self.base / 'Good'
        nested.mkdir(exist_ok=True)
        (nested / 'Broken.olean').write_bytes(b'invalid olean')
        self.assertTrue(self.replay('Good')['ok'])

    def test_user_lean_path_cannot_shadow_checker(self):
        shadow = self.base / 'LeanChecker.olean'
        shadow.write_bytes(b'untrusted shadow of installed checker')
        try:
            self.assertTrue(self.replay('Good')['ok'])
        finally:
            shadow.unlink()

    def test_real_private_body_cannot_hide_behind_olean_symlink(self):
        linked = self.base / 'symlink-negative'
        linked.mkdir(exist_ok=True)
        alias = linked / 'BadPrivate.olean'
        if not alias.exists():
            alias.symlink_to(self.base / 'BadPrivate.olean')
        try:
            m = self.kr.prepare_manifest(str(LEAN), [str(linked)],
                    [f'BadPrivate={self.fixtures / "BadPrivate.lean"}'], [])
        except ValueError as e:
            self.assertIn('symlink', str(e))
            return
        # Before the fix this bound destination private bytes but upstream
        # selected lexical siblings, accepted the exported-only presentation.
        r = self.kr.run_manifest(m, 8192, 60)
        (ROOT / 'evidence/initial-real-symlink-replay.json').write_text(json.dumps(r, indent=2) + '\n')
        self.assertFalse(r['ok'], r)

    def test_driver_missing_request_and_binding_rejected(self):
        env = os.environ.copy()
        env['LEAN_PATH'] = str(self.base)
        env.pop('CBZ_KERNEL_REPLAY_MODULE', None)
        p = subprocess.run([str(LEAN), str(ROOT / 'ExactReplay.lean')], cwd=ROOT, env=env,
                           capture_output=True, text=True, timeout=60)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('missing exact-module request', p.stdout)
        env['CBZ_KERNEL_REPLAY_MODULE'] = 'Good'
        env['CBZ_KERNEL_REPLAY_OLEAN'] = '/wrong/path.olean'
        p = subprocess.run([str(LEAN), str(ROOT / 'ExactReplay.lean')], cwd=ROOT, env=env,
                           capture_output=True, text=True, timeout=60)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('resolved olean mismatch', p.stdout)

    def test_cli_prepare_and_run(self):
        manifest = self.base / 'cli-manifest.json'
        report = self.base / 'cli-report.json'
        cli = [sys.executable, str(ROOT / 'kernel_replay.py')]
        p = subprocess.run([*cli, 'prepare', '--lean', str(LEAN), '--lean-path', str(self.base),
                           '--module', f'Good={self.fixtures / "Good.lean"}', '--output', str(manifest)],
                          capture_output=True, text=True, timeout=60)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        p = subprocess.run([*cli, 'run', str(manifest), '--memory-mib', '8192',
                           '--timeout-seconds', '60', '--output', str(report)],
                          capture_output=True, text=True, timeout=65)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertTrue(json.loads(report.read_text())['ok'])
        m = json.loads(manifest.read_text())
        m['modules'][0]['source']['sha256'] = '0' * 64
        manifest.write_text(json.dumps(m))
        p = subprocess.run([*cli, 'run', str(manifest), '--memory-mib', '8192',
                           '--timeout-seconds', '60', '--output', str(report)],
                          capture_output=True, text=True, timeout=65)
        self.assertEqual(p.returncode, 1)
        self.assertIn('binding changed', json.loads(report.read_text()).get('error', ''))


if __name__ == '__main__':
    unittest.main(verbosity=2)
