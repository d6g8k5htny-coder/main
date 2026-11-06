"""Exact Git-source custody controls; no Lean process is needed here."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from tests import test_formal_gate as fixtures

gate = fixtures.gate


class ExactSourceCommitTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.SyntheticControls('test_valid_source_check_and_no_writes')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        # Custody includes the actual loaded producer and compiled-audit helper.
        for relative, source in [('tools/formal_gate_check.py', gate.__file__),
                                 ('tools/formal_compiled_inventory.py', gate.compiled.__file__)]:
            (self.root / relative).write_bytes(Path(source).read_bytes())
        self.fixture.refresh()
        (self.root / 'unrelated.txt').write_text('unrelated repository data\n')
        self.git('init', '--quiet')
        self.git('add', '--all')
        self.git('commit', '--quiet', '-m', 'Exact source fixture')
        self.head = self.git('rev-parse', 'HEAD').strip()

    def git(self, *args):
        env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
        env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull)
        result = subprocess.run(['git', '-c', 'core.hooksPath=/dev/null', '-c', 'commit.gpgsign=false',
                                 '-c', 'user.name=CBZ test', '-c', 'user.email=cbz-test@invalid', *args],
                                cwd=self.root, env=env, text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def identity(self, *, expected=None):
        manifest, digest, _ = self.fixture.check()
        return gate.checked_source_commit(self.root, manifest, digest,
                                          manifest_rel='pkg/manifest.json', expected_head=expected)

    def test_clean_exact_commit_is_bound(self):
        self.assertEqual(self.identity(), self.head)
        self.assertEqual(self.identity(expected=self.head), self.head)

    def test_dirty_owned_source_is_refused_even_if_manifest_is_refreshed(self):
        source = self.root / 'pkg/Demo/Basic.lean'
        source.write_text(source.read_text() + '\n-- source-byte custody probe\n')
        self.fixture.refresh()
        with self.assertRaisesRegex(ValueError, 'checked HEAD'):
            self.identity()

    def test_dirty_manifest_bytes_are_refused_even_when_parsed_data_is_equal(self):
        path = self.root / 'pkg/manifest.json'
        path.write_bytes(path.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'checked HEAD'):
            self.identity()

    def test_staged_but_uncommitted_source_and_manifest_are_refused(self):
        path = self.root / 'pkg/Demo/Basic.lean'
        path.write_text(path.read_text() + '\n-- staged custody probe\n')
        self.fixture.refresh()
        self.git('add', '--all')
        with self.assertRaisesRegex(ValueError, 'checked HEAD'):
            self.identity()

    def test_assume_unchanged_cannot_hide_changed_owned_bytes(self):
        relative = 'pkg/Demo/Basic.lean'
        self.git('update-index', '--assume-unchanged', relative)
        path = self.root / relative
        path.write_text(path.read_text() + '\n-- hidden dirty bytes\n')
        self.assertEqual(self.git('status', '--porcelain'), '')
        manifest = self.fixture.manifest
        digest = hashlib.sha256((self.root / 'pkg/manifest.json').read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, 'checked HEAD'):
            gate.checked_source_commit(self.root, manifest, digest, manifest_rel='pkg/manifest.json')

    def test_untracked_manifest_bound_input_is_refused(self):
        relative = 'tools/extra-input.txt'
        (self.root / relative).write_text('untracked declared input\n')
        self.fixture.bound.append(relative)
        self.fixture.refresh()
        self.git('add', 'pkg/manifest.json')
        self.git('commit', '--quiet', '-m', 'Declare missing Git input')
        with self.assertRaisesRegex(ValueError, 'source custody inventory'):
            self.identity()

    def test_executable_mode_drift_is_refused(self):
        path = self.root / 'tools/formal_gate_check.py'
        path.chmod(path.stat().st_mode | 0o100)
        with self.assertRaisesRegex(ValueError, 'source custody mode'):
            self.identity()

    def test_core_filemode_false_cannot_hide_owned_mode_drift(self):
        self.git('config', 'core.filemode', 'false')
        path = self.root / 'tools/formal_gate_check.py'
        path.chmod(path.stat().st_mode | 0o100)
        self.assertEqual(self.git('status', '--porcelain'), '')
        with self.assertRaisesRegex(ValueError, 'source custody mode'):
            self.identity()

    def test_subdirectory_cannot_be_presented_as_another_git_root(self):
        manifest, digest, _ = self.fixture.check()
        with self.assertRaisesRegex(ValueError, 'exact Git top-level'):
            gate.checked_source_commit(self.root / 'pkg', manifest, digest, manifest_rel='manifest.json')

    def test_execute_package_argument_cannot_substitute_another_tree(self):
        manifest, digest, _ = self.fixture.check()
        other = self.root / 'other-package'
        other.mkdir()
        with mock.patch.object(gate, 'lake_binary', return_value='/synthetic/lake'), \
             mock.patch.object(gate, 'run', side_effect=AssertionError('Lean must not run')) as lean_run, \
             self.assertRaisesRegex(ValueError, 'package argument'):
            gate.execute(manifest, digest, other, self.root, manifest_rel='pkg/manifest.json')
        lean_run.assert_not_called()
        self.assertFalse((other / '.lake').exists())

    def test_custody_refusal_archives_prior_success_before_any_lean_process(self):
        manifest, digest, pkg = self.fixture.check()
        out = pkg / '.lake/formal-evidence'
        out.mkdir(parents=True)
        prior = {'receipt.json': b'{"prior_success":true}\n', 'build.log': b'prior build\n'}
        for name, raw in prior.items():
            (out / name).write_bytes(raw)
        with mock.patch.object(gate, 'lake_binary', return_value='/synthetic/lake'), \
             mock.patch.object(gate, 'checked_source_commit', side_effect=ValueError('controlled custody refusal')), \
             mock.patch.object(gate, 'run', side_effect=AssertionError('Lean must not run')) as lean_run, \
             self.assertRaisesRegex(ValueError, 'controlled custody refusal'):
            gate.execute(manifest, digest, pkg, self.root, manifest_rel='pkg/manifest.json')
        lean_run.assert_not_called()
        self.assertFalse((out / 'receipt.json').exists())
        archived = list((pkg / '.lake/formal-evidence-history').glob('*/formal-evidence'))
        self.assertEqual(len(archived), 1)
        self.assertEqual({path.name: path.read_bytes() for path in archived[0].iterdir()}, prior)

    def test_symlink_to_identical_bound_bytes_is_refused(self):
        path = self.root / 'pkg/Demo/Basic.lean'
        outside = self.root / 'same-bytes.txt'
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
        with self.assertRaises(ValueError):
            self.identity()

    def test_expected_head_must_remain_stable(self):
        self.git('commit', '--quiet', '--allow-empty', '-m', 'Different source epoch')
        with self.assertRaisesRegex(ValueError, 'HEAD changed'):
            self.identity(expected=self.head)

    def test_ambient_git_overrides_cannot_substitute_another_repository(self):
        with mock.patch.dict(os.environ, {'GIT_DIR': '/nonexistent-cbz-git',
                                         'GIT_WORK_TREE': '/nonexistent-cbz-tree',
                                         'GIT_INDEX_FILE': '/nonexistent-cbz-index'}):
            self.assertEqual(self.identity(), self.head)

    def test_git_replace_cannot_launder_dirty_bytes_as_original_commit(self):
        path = self.root / 'pkg/Demo/Basic.lean'
        path.write_text(path.read_text() + '\n-- replacement-tree probe\n')
        self.fixture.refresh()
        self.git('add', '--all')
        self.git('commit', '--quiet', '-m', 'Replacement source tree')
        replacement = self.git('rev-parse', 'HEAD').strip()
        self.git('update-ref', 'HEAD', self.head, replacement)
        self.git('replace', self.head, replacement)
        with self.assertRaisesRegex(ValueError, 'checked HEAD'):
            self.identity()

    def test_build_outputs_and_unrelated_dirty_data_do_not_invalidate_owned_commit(self):
        path = self.root / 'pkg/.lake/formal-evidence/log.txt'
        path.parent.mkdir(parents=True)
        path.write_text('runtime output\n')
        (self.root / 'unrelated.txt').write_text('unrelated edit\n')
        self.assertEqual(self.identity(), self.head)

    def test_manifest_parsed_data_must_match_argument(self):
        manifest, digest, _ = self.fixture.check()
        with self.assertRaises(ValueError):
            gate.checked_source_commit(self.root, {**manifest, 'package': 'forged'}, digest,
                                       manifest_rel='pkg/manifest.json')

    def test_manifest_digest_must_match_argument(self):
        manifest, _, _ = self.fixture.check()
        with self.assertRaises(ValueError):
            gate.checked_source_commit(self.root, manifest, '0' * 64, manifest_rel='pkg/manifest.json')

    def loaded_profile_refused(self, relative):
        path = self.root / relative
        path.write_bytes(path.read_bytes() + b'\n# different loaded-source profile\n')
        self.fixture.refresh()
        self.git('add', '--all')
        self.git('commit', '--quiet', '-m', 'Different producer profile')
        with self.assertRaisesRegex(ValueError, 'loaded auditor source'):
            self.identity()

    def test_loaded_producer_must_match_bound_source(self):
        self.loaded_profile_refused('tools/formal_gate_check.py')

    def test_loaded_helper_must_match_bound_source(self):
        self.loaded_profile_refused('tools/formal_compiled_inventory.py')


class MainCustodyBindingTests(unittest.TestCase):
    def test_live_consumer_and_custody_controls_cannot_be_unbound(self):
        original = json.loads((fixtures.ROOT / 'formal/manifest.json').read_text())
        for missing in ['tools/required_formal_check.py', 'tests/test_required_formal_check.py',
                        'tests/test_required_formal_binding_control.py', 'tests/test_formal_source_custody.py',
                        '.github/workflows/workspace-landing.yml']:
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                for relative in set(original['files']) | set(original['unbound_files']):
                    path = root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes((fixtures.ROOT / relative).read_bytes())
                manifest = {**original, 'files': dict(original['files'])}
                del manifest['files'][missing]
                (root / 'formal/manifest.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, 'unbound live receipt consumer or source custody control'):
                    gate.source_check(root)


class CustodySourceMutationTests(unittest.TestCase):
    def test_nine_declared_source_mutants_have_reason_bound_kills(self):
        import io
        import types
        from tests import test_formal_compiled_inventory as mutation_contract
        original = gate
        source = Path(original.__file__).read_text()
        # Every replacement is unique and validated before compiling the mutant.
        mutations = {
            'ignore_blob_bytes': ([('require(path.read_bytes() == git("cat-file", "blob", object_id.decode("ascii")),', 'require(True,')],
                {'test_assume_unchanged_cannot_hide_changed_owned_bytes',
                 'test_dirty_manifest_bytes_are_refused_even_when_parsed_data_is_equal',
                 'test_dirty_owned_source_is_refused_even_if_manifest_is_refreshed',
                 'test_staged_but_uncommitted_source_and_manifest_are_refused',
                 'test_git_replace_cannot_launder_dirty_bytes_as_original_commit'}),
            'ignore_modes': ([('stat.S_ISREG(actual_mode) and bool(actual_mode & stat.S_IXUSR) == (mode == b"100755")', 'True')],
                {'test_executable_mode_drift_is_refused', 'test_core_filemode_false_cannot_hide_owned_mode_drift'}),
            'accept_changed_head': ([('expected_head is None or head == expected_head', 'True')],
                {'test_expected_head_must_remain_stable'}),
            'ignore_manifest_arguments': ([('sha(manifest_raw) == digest and load_json(manifest_raw.decode("utf-8")) == m', 'True')],
                {'test_manifest_parsed_data_must_match_argument', 'test_manifest_digest_must_match_argument'}),
            'ignore_loaded_sources': ([('m["files"].get(relative) == loaded_digest\n                and sha(loaded_path.read_bytes()) == loaded_digest', 'True')],
                {'test_loaded_producer_must_match_bound_source', 'test_loaded_helper_must_match_bound_source'}),
            'omit_untracked_inputs': ([('set(entries) == critical', 'True'), ('for relative in sorted(critical):', 'for relative in sorted(entries):')],
                {'test_untracked_manifest_bound_input_is_refused'}),
            'allow_git_replacements': ([('"git", "--no-replace-objects", "--literal-pathspecs"', '"git", "--literal-pathspecs"'),
                                        ('GIT_NO_REPLACE_OBJECTS="1", GIT_OPTIONAL_LOCKS="0"', 'GIT_OPTIONAL_LOCKS="0"')],
                {'test_git_replace_cannot_launder_dirty_bytes_as_original_commit'}),
            'omit_initial_binding': ([('checked_head = checked_source_commit(root, m, digest, manifest_rel=manifest_rel)', 'checked_head = "a" * 40')],
                {'test_custody_refusal_archives_prior_success_before_any_lean_process',
                 'test_valid_runtime_preserves_both_version_observations_and_logs',
                 'test_changed_checked_head_after_execution_refuses_receipt'}),
            'omit_final_binding': ([('head = checked_source_commit(root, m, digest, manifest_rel=manifest_rel, expected_head=checked_head)', 'head = checked_head')],
                {'test_valid_runtime_preserves_both_version_observations_and_logs',
                 'test_changed_checked_head_after_execution_refuses_receipt'}),
        }
        loader = unittest.TestLoader()
        def controls():
            return unittest.TestSuite([
                loader.loadTestsFromTestCase(ExactSourceCommitTests),
                fixtures.SyntheticControls('test_valid_runtime_preserves_both_version_observations_and_logs'),
                fixtures.SyntheticControls('test_changed_checked_head_after_execution_refuses_receipt'),
            ])
        baseline_suite = controls()
        expected_ids = mutation_contract.policy_test_ids(baseline_suite)
        baseline = mutation_contract.run_policy_controls(baseline_suite)
        mutation_contract.require_policy_control(baseline, expected_ids)
        try:
            for label, (replacements, expected_failures) in mutations.items():
                with self.subTest(mutation=label):
                    mutated = source
                    for old, new in replacements:
                        self.assertEqual(mutated.count(old), 1, 'stale custody mutation anchor: ' + label)
                        mutated = mutated.replace(old, new)
                    module = types.ModuleType('custody_source_mutant')
                    module.__file__ = original.__file__
                    exec(compile(mutated, original.__file__, 'exec'), module.__dict__)
                    globals()['gate'] = module
                    with mock.patch.object(fixtures, 'gate', module):
                        result = mutation_contract.run_policy_controls(controls())
                    mutation_contract.require_policy_mutation_kill(result, expected_ids, expected_failures)
        finally:
            globals()['gate'] = original


if __name__ == '__main__':
    unittest.main()
