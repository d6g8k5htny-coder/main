"""Reject infrastructure errors masquerading as Lean proof rejections.

Only subprocess results are synthetic. The unchanged gate's execute, log writer,
manifest checks and receipt writer run against the existing temporary fixture.
"""
import subprocess
import unittest
from unittest import mock
from tests import test_formal_gate as fixtures


class NegativeDiagnosticContract(unittest.TestCase):
    def execute_with_result(self, code, stdout, stderr='', label='tight'):
        fixture = fixtures.SyntheticControls('test_valid_source_check_and_no_writes')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        real = subprocess.CompletedProcess
        def outcome(args, returncode, output=None, error=None):
            if args and str(args[-1]).endswith('/' + label + '.lean'):
                text = stdout.replace('{path}', str(args[-1]))
                return real(args, code, text, stderr)
            return real(args, returncode, output, error)
        with mock.patch.object(fixtures.subprocess, 'CompletedProcess', side_effect=outcome):
            return fixture.runtime_probe([fixture.RUNTIME_RELEASE, fixture.RUNTIME_RELEASE])

    def test_unrelated_failures_cannot_mint_successful_negative_receipt(self):
        for code, text in [(17, 'failed to allocate memory\n'),
                           (-9, 'process failed externally\n'),
                           (1, 'dependency download failed\n'),
                           (1, 'unsolved goals\n'),
                           (1, 'is false\n')]:
            with self.subTest(code=code, text=text), self.assertRaises(ValueError):
                self.execute_with_result(code, text)

    def test_real_diagnostic_with_wrong_exit_or_stderr_is_refused(self):
        text = '{path}:6:30: error: Tactic `decide` proved that the proposition\n  77 < 77\nis false\n'
        for code, stderr in [(17, ''), (-9, ''), (1, 'failed resource operation\n')]:
            with self.subTest(code=code, stderr=stderr), self.assertRaises(ValueError):
                self.execute_with_result(code, text, stderr)

    def test_extra_wrong_location_or_wrong_goal_diagnostic_is_refused(self):
        text = '{path}:6:30: error: Tactic `decide` proved that the proposition\n  77 < 77\nis false\n'
        for changed in [text + 'dependency failed\n', text.replace(':6:', ':7:'),
                        text.replace('77 < 77', '78 < 77'), text.replace('{path}', 'other.lean')]:
            with self.subTest(text=changed), self.assertRaises(ValueError):
                self.execute_with_result(1, changed)

    def test_axiom_controls_require_the_exact_complete_forbidden_report(self):
        for label, text in [
            ('custom_imported', "'injected' depends on axioms: [sorryAx]\n"),
            ('native', "'injected' depends on axioms: [hiddenPremise]\n"),
            ('custom_imported', "'other' depends on axioms: [hiddenPremise]\n"),
            ('custom_imported', "'injected' depends on axioms: [hiddenPremise]\n" * 2),
            ('custom_imported', "'injected' depends on axioms: [hiddenPremise]\nerror: failed elsewhere\n"),
            ('custom_imported', "'injected' depends on axioms: [bad name]\n"),
        ]:
            with self.subTest(label=label, text=text), self.assertRaises(ValueError):
                self.execute_with_result(0, text, label=label)

    def test_axiom_control_stderr_is_not_ignored(self):
        with self.assertRaises(ValueError):
            self.execute_with_result(0, "'injected' depends on axioms: [hiddenPremise]\n",
                                     'failed external operation\n', 'custom_imported')

    def test_exact_observed_false_goal_is_accepted(self):
        text = '{path}:6:30: error: Tactic `decide` proved that the proposition\n  77 < 77\nis false\n'
        receipt = self.execute_with_result(1, text)
        self.assertEqual(receipt['negative_controls']['tight'], 'REJECTED_BY_LEAN')

    def test_missing_or_malformed_control_contract_is_refused(self):
        for mutation in ['empty', 'missing', 'missing-diagnostic', 'ambiguous-source']:
            fixture = fixtures.SyntheticControls('test_valid_source_check_and_no_writes')
            fixture.setUp()
            self.addCleanup(fixture.doCleanups)
            if mutation == 'empty':
                fixture.manifest['negative_controls'] = {}
            elif mutation == 'missing':
                del fixture.manifest['negative_controls']
            elif mutation == 'missing-diagnostic':
                del fixture.manifest['negative_controls']['tight']['expected_diagnostic']
            else:
                path = fixture.root / 'pkg/Demo/Basic.lean'
                path.write_text(path.read_text() + '\n-- 77 < 78\n')
            fixture.refresh()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                fixture.check()

    def test_failed_rerun_archives_prior_receipt_and_logs_together(self):
        fixture = fixtures.SyntheticControls('test_valid_source_check_and_no_writes')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        manifest, digest, pkg = fixture.check()
        out = pkg / '.lake/formal-evidence'
        out.mkdir(parents=True)
        originals = {'receipt.json': b'{"historical":true}\n', 'build.log': b'old build\n'}
        for name, data in originals.items():
            (out / name).write_bytes(data)
        with mock.patch.object(fixtures.gate, 'lake_binary', return_value='/synthetic/lake'), \
             mock.patch.object(fixtures.gate.subprocess, 'run',
                               return_value=subprocess.CompletedProcess([], 17, 'failed preflight\n')):
            with self.assertRaises(ValueError):
                fixtures.gate.execute(manifest, digest, pkg, fixture.root)
        self.assertFalse((out / 'receipt.json').exists())
        history = pkg / '.lake/formal-evidence-history'
        archives = list(history.glob('*/formal-evidence'))
        self.assertEqual(len(archives), 1)
        self.assertEqual({p.name:p.read_bytes() for p in archives[0].iterdir()}, originals)

    def test_permissive_axiom_auditor_mutation_is_detected(self):
        fixture = fixtures.SyntheticControls('test_valid_source_check_and_no_writes')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        with mock.patch.object(fixtures.gate, 'audit_axioms', return_value={'injected': []}):
            with self.assertRaises(ValueError):
                fixture.runtime_probe([fixture.RUNTIME_RELEASE, fixture.RUNTIME_RELEASE])

    def test_one_removed_main_control_is_refused(self):
        import json
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = json.loads((fixtures.ROOT / 'formal/manifest.json').read_text())
            for name in [*manifest['files'], *manifest['unbound_files']]:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((fixtures.ROOT / name).read_bytes())
            del manifest['negative_controls']['wrong_image_constant']
            (root / 'formal/manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'incomplete main negative control inventory'):
                fixtures.gate.source_check(root)

    def test_archive_paths_cannot_follow_symlinks(self):
        from pathlib import Path
        import tempfile
        for link in ['.lake', '.lake/formal-evidence', '.lake/formal-evidence-history']:
            fixture = fixtures.SyntheticControls('test_valid_source_check_and_no_writes')
            fixture.setUp()
            self.addCleanup(fixture.doCleanups)
            manifest, digest, pkg = fixture.check()
            with tempfile.TemporaryDirectory() as directory:
                outside = Path(directory)
                path = pkg / link
                path.parent.mkdir(parents=True, exist_ok=True)
                if link.endswith('history'):
                    (pkg / '.lake/formal-evidence').mkdir()
                path.symlink_to(outside, target_is_directory=True)
                with self.subTest(link=link), \
                     mock.patch.object(fixtures.gate, 'lake_binary', return_value='/synthetic/lake'), \
                     self.assertRaisesRegex(ValueError, 'symlink'):
                    fixtures.gate.execute(manifest, digest, pkg, fixture.root)
                self.assertEqual(list(outside.iterdir()), [])

    def test_missing_tool_rerun_does_not_leave_current_success(self):
        fixture = fixtures.SyntheticControls('test_valid_source_check_and_no_writes')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        manifest, digest, pkg = fixture.check()
        out = pkg / '.lake/formal-evidence'
        out.mkdir(parents=True)
        (out / 'receipt.json').write_bytes(b'old success')
        with mock.patch.object(fixtures.gate, 'lake_binary', return_value=None), \
             self.assertRaises(ValueError):
            fixtures.gate.execute(manifest, digest, pkg, fixture.root)
        self.assertFalse((out / 'receipt.json').exists())
        self.assertEqual([p.read_bytes() for p in (pkg / '.lake/formal-evidence-history').glob('*/formal-evidence/receipt.json')], [b'old success'])

    def test_new_diagnostic_controls_cannot_be_unbound(self):
        fixture = fixtures.SyntheticControls('test_valid_source_check_and_no_writes')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.bound.remove('tests/test_formal_negative_diagnostics.py')
        fixture.refresh()
        with self.assertRaisesRegex(ValueError, 'unbound control or scope file'):
            fixture.check()
