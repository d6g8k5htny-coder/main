import copy
import hashlib
import importlib.util
import json
import contextlib
import io
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('gate', ROOT / 'gate.py')
gate = importlib.util.module_from_spec(spec) if spec else None
if spec and spec.loader and (ROOT / 'gate.py').exists():
    spec.loader.exec_module(gate)

STD_AXIOMS = ('Classical.choice', 'Quot.sound', 'propext')


def neutral_overlay(root):
    """Write an empty RF312 manifest overlay into a copied formal tree.

    The production overlay rebinds the compatibility files; these fixtures recompute every
    binding from the copied bytes instead, so the effective manifest is exactly the one the
    test edits. Call before hashing the tree, so the overlay file's own hash is bound too.
    """
    (root / 'manifest.overlay.json').write_text(json.dumps({'files': {}, 'schema_version': 1}))


def encode_name(value):
    return gate.encode(tuple(value) if isinstance(value, tuple) else tuple(value.split('.')))


def inventory_transcript(root, manifest, injected=()):
    """A clean compiled-environment inventory for the copied package, as the generated Lean
    program would print it (RF-GATE-01 protocol), with optional injected constants.

    User theorems are placed in the module whose source declares them, exactly as the
    source pre-check does; every other constant comes from the manifest's bound
    `compiled_inventory`. `injected` entries are (kind, origin, module, name, extra_axioms)
    with module/name as component tuples; a registered module's count grows with them.
    """
    package = gate.PACKAGE
    modules = [package + '.' + path[len(package) + 1:-5] for path in manifest['source_modules']]
    constants, counts, definitions = [], {m: 0 for m in modules}, []
    for path, module in zip(manifest['source_modules'], modules):
        for theorem in gate.grammar_check((root / path).read_text(), path, definitions):
            constants.append(('theorem', 'user', encode_name(module), encode_name(package + '.' + theorem)))
            counts[module] += 1
    for kind, origin, module, name in manifest['compiled_inventory']:
        constants.append((kind, origin, encode_name(module), encode_name(name)))
        counts[module] += 1
    axioms = set(STD_AXIOMS)
    for kind, origin, module, name, extra in injected:
        constants.append((kind, origin, gate.encode(module), gate.encode(name)))
        rendered = '.'.join(str(c) for c in module)
        if rendered in counts:
            counts[rendered] += 1
        axioms.update(extra)
    lines = ['MODULE|' + encode_name(m) + '|false|' + str(counts[m]) + '|0' for m in modules]
    lines.append('MODULE|' + encode_name(package) + '|false|0|0')
    lines.extend('INV|' + '|'.join(c) for c in constants)
    lines.append('AXIOMS|' + ','.join(sorted(encode_name(a) for a in axioms)))
    lines.append('CLOSURE|' + str(len(constants) + 1000))
    return '\n'.join(lines) + '\n'


class GateTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(gate, 'audit_axioms'), 'formal evidence gate not implemented')

    def test_standard_axioms(self):
        self.assertEqual(gate.audit_axioms("'X.a' depends on axioms: [propext, Classical.choice, Quot.sound]", ['X.a'])['X.a'], ['Classical.choice', 'Quot.sound', 'propext'])
    def test_axiom_free(self):
        self.assertEqual(gate.audit_axioms("'X.a' does not depend on any axioms", ['X.a']), {'X.a': []})
    def test_sorry(self):
        with self.assertRaises(ValueError): gate.audit_axioms("'X.a' depends on axioms: [sorryAx]", ['X.a'])
    def test_custom(self):
        with self.assertRaises(ValueError): gate.audit_axioms("'X.a' depends on axioms: [Other.hidden]", ['X.a'])
    def test_native(self):
        with self.assertRaises(ValueError): gate.audit_axioms("'X.a' depends on axioms: [X.a._nativeDecide_1]", ['X.a'])
    def test_missing_target(self):
        with self.assertRaises(ValueError): gate.audit_axioms('', ['X.a'])
    def test_duplicate_target_output(self):
        with self.assertRaises(ValueError): gate.audit_axioms("'X.a' does not depend on any axioms\n'X.a' does not depend on any axioms", ['X.a'])
    def test_duplicate_target_request(self):
        with self.assertRaises(ValueError): gate.audit_axioms("'X.a' does not depend on any axioms", ['X.a','X.a'])
    def test_unexpected_target(self):
        with self.assertRaises(ValueError): gate.audit_axioms("'X.b' does not depend on any axioms", ['X.a'])
    def test_multiline(self):
        self.assertIn('X.a', gate.audit_axioms("'X.a' depends on axioms: [propext,\n Classical.choice]", ['X.a']))
    def test_noise_denied(self):
        with self.assertRaises(ValueError): gate.audit_axioms("error: bad\n'X.a' does not depend on any axioms", ['X.a'])
    def test_manifest_content(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); (p/'x').write_bytes(b'abc')
            self.assertEqual(gate.check_files(p, {'x':hashlib.sha256(b'abc').hexdigest()}), None)
    def test_manifest_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); (p/'x').write_bytes(b'bad')
            with self.assertRaises(ValueError): gate.check_files(p, {'x':hashlib.sha256(b'abc').hexdigest()})
    def test_manifest_missing(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError): gate.check_files(Path(d), {'missing':'0'*64})
    def test_manifest_escape(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError): gate.check_files(Path(d), {'../x':'0'*64})
    def test_manifest_absolute(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError): gate.check_files(Path(d), {'/x':'0'*64})
    def test_manifest_symlink(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); (p/'x').write_bytes(b'a'); (p/'y').symlink_to(p/'x')
            with self.assertRaises(ValueError): gate.check_files(p, {'y':hashlib.sha256(b'a').hexdigest()})
    def test_manifest_empty(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError): gate.check_files(Path(d), {})
    def review(self):
        return dict(disposition='ACCEPTED',manifest_sha256='a'*64,targets=['X.a'],scope_sha256='b'*64,
                    author={'provider':'OpenAI','family':'GPT','agent':'A'},reviewer={'provider':'Other','family':'Different','agent':'B'},
                    evidence={'repository':'o/r','commit':'c'*40,'path':'review.md','sha256':'d'*64})
    def test_review_valid(self):
        gate.check_alignment(self.review(), 'a'*64, ['X.a'], 'b'*64)
    def test_review_stale(self):
        with self.assertRaises(ValueError): gate.check_alignment(self.review(), 'e'*64, ['X.a'], 'b'*64)
    def test_review_self(self):
        r=self.review(); r['reviewer']=r['author'].copy()
        with self.assertRaises(ValueError): gate.check_alignment(r, 'a'*64, ['X.a'], 'b'*64)
    def test_review_same_provider(self):
        r=self.review(); r['reviewer']['provider']='OpenAI'
        with self.assertRaises(ValueError): gate.check_alignment(r, 'a'*64, ['X.a'], 'b'*64)
    def test_review_partial(self):
        with self.assertRaises(ValueError): gate.check_alignment(self.review(), 'a'*64, ['X.a','X.b'], 'b'*64)
    def test_review_bad_disposition(self):
        r=self.review(); r['disposition']='AMEND_REQUIRED'
        with self.assertRaises(ValueError): gate.check_alignment(r, 'a'*64, ['X.a'], 'b'*64)
    def test_review_bad_evidence(self):
        r=self.review(); r['evidence']['commit']='main'
        with self.assertRaises(ValueError): gate.check_alignment(r, 'a'*64, ['X.a'], 'b'*64)
    def test_review_changed_scope(self):
        with self.assertRaises(ValueError): gate.check_alignment(self.review(), 'a'*64, ['X.a'], 'c'*64)

class BlueprintTests(unittest.TestCase):
    def test_valid_link(self):
        self.assertEqual(gate.check_blueprint(r'\lean{X.a}', ['X.a']), ['X.a'])
    def test_wrong_link(self):
        with self.assertRaises(ValueError): gate.check_blueprint(r'\lean{X.b}', ['X.a'])
    def test_empty_link(self):
        with self.assertRaises(ValueError): gate.check_blueprint('', ['X.a'])
    def test_unsupported_leanok(self):
        with self.assertRaises(ValueError): gate.check_blueprint(r'\lean{X.a}\leanok', ['X.a'])

class SourceContractTests(unittest.TestCase):
    def fixture(self):
        import shutil
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        p = Path(self.tmp.name) / 'formal'
        shutil.copytree(ROOT, p, ignore=shutil.ignore_patterns('.lake', '__pycache__'))
        neutral_overlay(p)
        self.previous = gate.ROOT; gate.ROOT = p
        self.addCleanup(lambda: setattr(gate, 'ROOT', self.previous))
        self.m = json.loads((p/'manifest.json').read_text())
        self.m['files'] = {q.relative_to(p).as_posix(): hashlib.sha256(q.read_bytes()).hexdigest() for q in p.rglob('*') if q.is_file() and q.name != 'manifest.json'}
        self.p = p
        return p
    def save(self):
        (self.p/'manifest.json').write_text(json.dumps(self.m))
    def test_source_baseline(self):
        self.fixture(); self.save(); gate.source_check()
    def test_unbound_gate_refused(self):
        self.fixture(); del self.m['files']['gate.py']; self.save()
        with self.assertRaises(ValueError): gate.source_check()
    def test_manifest_cannot_claim_kernel_status(self):
        self.fixture(); self.m['formalization_status']='kernel-checked'; self.save()
        with self.assertRaises(ValueError): gate.source_check()
    def test_root_declaration_cannot_evade_audit(self):
        p=self.fixture(); root=p/'ResearchFormalCoreR1.lean'
        root.write_text(root.read_text()+'theorem hidden : False := by sorry\n')
        self.m['files']['ResearchFormalCoreR1.lean']=hashlib.sha256(root.read_bytes()).hexdigest(); self.save()
        with self.assertRaises(ValueError): gate.source_check()
    def test_rebound_root_redirects_source_overlay_and_evidence_reads(self):
        # RF312 regression: rebinding `gate.ROOT` must redirect every read of the layered gate --
        # the helper's manifest and source checks, the overlay, and the execution evidence
        # directory -- not only this wrapper's own reads (dot's failure map, Math-#312).
        p = self.fixture(); self.save()

        def overlay(files):
            path = p / 'manifest.overlay.json'
            path.write_text(json.dumps({'schema_version': 1, 'files': files}))
            self.m['files']['manifest.overlay.json'] = hashlib.sha256(path.read_bytes()).hexdigest()
            self.save()

        _, digest = gate.source_check()
        self.assertEqual(Path(gate._core.ROOT), p, 'helper still reads the original tree')
        self.assertEqual(Path(gate.OVERLAY), p / 'manifest.overlay.json')
        # A wrong binding written only into the copy's overlay is seen, so the overlay read moved.
        overlay({'README.md': '0' * 64})
        with self.assertRaisesRegex(ValueError, 'source hash mismatch: README.md'):
            gate.source_check()
        # The digest binds the overlay bytes that were actually read.
        overlay({'README.md': hashlib.sha256((p / 'README.md').read_bytes()).hexdigest()})
        _, other = gate.source_check()
        self.assertNotEqual(digest, other)
        # A source edit made only in the copy is seen by the helper's own file check.
        (p / 'README.md').write_bytes((p / 'README.md').read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'source hash mismatch: README.md'):
            gate.source_check()

    def test_duplicate_json_rejected(self):
        with self.assertRaises(ValueError): gate.load_json('{"a":1,"a":2}')
    def test_json_object(self):
        self.assertEqual(gate.load_json('{"a":1}'), {'a':1})

class RuntimeVersionTests(unittest.TestCase):
    # These are orchestration tests with synthetic external process results;
    # source validation, log writes, build deletion and receipts are real.
    RELEASE = 'Lean (version 4.34.1, x86_64-unknown-linux-gnu, commit 5045d0056413266e57c625dcd7c365b10e377c52, Release)'

    def fixture(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / 'formal'
        shutil.copytree(ROOT, self.root, ignore=shutil.ignore_patterns('.lake', '__pycache__'))
        neutral_overlay(self.root)
        previous = gate.ROOT
        gate.ROOT = self.root
        self.addCleanup(lambda: setattr(gate, 'ROOT', previous))
        manifest_path = self.root / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['files'] = {p.relative_to(self.root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in self.root.rglob('*') if p.is_file() and p.name != 'manifest.json'}
        manifest_path.write_text(json.dumps(manifest))
        self.manifest, self.digest = gate.source_check()
        self.sentinel = self.root / '.lake/build/preserve-before-preflight'
        self.sentinel.parent.mkdir(parents=True)
        self.sentinel.write_text('old build')
        self.calls = []
        self.version_index = 0

    def execute_synthetic(self, versions, version_exit=0, mutate_source=False, wrong_dependency=False,
                          dirty_dependency=None, drift_dependency=False, env=None):
        # The layered gate observes the running Lean three times: the wrapper's preflight, the
        # reviewed helper's own check after its controls, and the wrapper's final observation.
        # `versions` keeps its two-entry meaning: [preflight, final]; the helper's read sees the
        # preflight release (nothing changes it during the helper's own execution).
        schedule = [versions[0], versions[0], versions[-1]]
        package = gate.PACKAGE
        first_module = self.manifest['source_modules'][0]
        local = gate.LOCAL_MODULE
        admissions = {
            'hidden_sorry': [('theorem', 'user', local, ('_private', 'fixture', 0, package, 'hiddenAdmission'), ('sorryAx',))],
            'hidden_theorem': [('theorem', 'user', local, (package, 'indentedAdmission'), ())],
            'hidden_def': [('def', 'user', local, (package, 'unusedAdmission'), ('sorryAx',))],
            'outside_namespace': [('theorem', 'user', local, ('outsideNamespaceAdmission',), ())],
        }

        def effective_root(kwargs):
            cwd = kwargs.get('cwd')
            if cwd is not None and Path(cwd).resolve().is_relative_to(self.root.resolve()):
                return Path(cwd)
            return self.root

        def compiled_injections(package_root):
            # The compiled-admission copy carries the injected declaration in its first module.
            text = (package_root / first_module).read_text()
            module = tuple((package + '.' + first_module[len(package) + 1:-5]).split('.'))
            found = []
            for label, declaration, (kind, origin, name, private), expected in gate.COMPILED_INJECTIONS:
                if declaration in text:
                    full = ('_private', 'fixture', 0) + name if private else name
                    extra = ('sorryAx',) if 'sorry' in declaration else ()
                    found.append((kind, origin, module, full, extra))
            return found

        def external_run(command, **kwargs):
            self.calls.append(command)
            if command == ['lake', 'env', 'lean', '--version']:
                text = schedule[min(self.version_index, len(schedule) - 1)] + '\n'
                self.version_index += 1
                return subprocess.CompletedProcess(command, version_exit, text)
            if command == ['lake', 'build']:
                # A synthetic build leaves a build directory, as the real one does.
                (effective_root(kwargs) / '.lake' / 'build').mkdir(parents=True, exist_ok=True)
                if mutate_source:
                    path = self.root / 'lake-manifest.json'
                    path.write_bytes(path.read_bytes() + b'\n')
            label = Path(command[-1]).stem
            if label == 'Audit':
                text = '\n'.join("'" + n + "' does not depend on any axioms" for n in self.manifest['targets'])
            elif label in ('false_fold', 'false_power'):
                return subprocess.CompletedProcess(command, 1, 'unsolved goals\n')
            elif label in ('sorry', 'custom_imported', 'native'):
                forbidden = {'sorry': 'sorryAx', 'custom_imported': 'hiddenPremise', 'native': 'injected._nativeDecide_1'}[label]
                text = "'injected' depends on axioms: [" + forbidden + ']'
            elif label == 'Inventory':
                # User theorems come from the fixture's own (uninjected) sources; the compiled-
                # admission copy differs only by the declaration it injected, read from `cwd`.
                text = inventory_transcript(self.root, self.manifest, compiled_injections(effective_root(kwargs)))
            elif label in admissions:
                text = inventory_transcript(self.root, self.manifest, admissions[label])
            else:
                text = ''
            return subprocess.CompletedProcess(command, 0, text)

        def external_git(command, **kwargs):
            if command == ['git', 'rev-parse', 'HEAD']:
                return 'b' * 40 + '\n'
            name = Path(command[2]).name
            final_phase = self.version_index >= 2  # after the final version observation
            if command[3:] == ['status', '--porcelain', '--untracked-files=normal']:
                dirty = dirty_dependency == 'preflight' and not final_phase or dirty_dependency == 'final' and final_phase
                return ' M Mathlib/Injected.lean\n' if dirty else ''
            revision = self.manifest['dependency_revisions'][name]
            return ('c' * 40 if wrong_dependency or drift_dependency and final_phase else revision) + '\n'

        with patch.object(gate.subprocess, 'run', side_effect=external_run), \
             patch.object(gate.subprocess, 'check_output', side_effect=external_git), \
             patch.dict(gate.os.environ, env or {}), \
             contextlib.redirect_stdout(io.StringIO()):
            return gate.execute(self.manifest, self.digest)

    def assert_no_receipt(self):
        self.assertFalse((self.root / '.lake/formal-evidence/receipt.json').exists())

    def test_complete_pinned_release_record(self):
        self.assertTrue(callable(getattr(gate, 'check_lean_version', None)), 'complete release validator missing')
        self.assertEqual(gate.check_lean_version(self.RELEASE + '\n'), self.RELEASE)

    def test_other_versions_and_incomplete_or_extra_records_refused(self):
        self.assertTrue(callable(getattr(gate, 'check_lean_version', None)), 'complete release validator missing')
        invalid = ['', 'unrelated version 4.34.1 text',
                   self.RELEASE.replace('4.34.1', '4.34.10'), self.RELEASE.replace('4.34.1', '4.34.100'),
                   self.RELEASE.replace('4.34.1', '4.34.1-rc1'), self.RELEASE.replace('4.34.1', '4.34.1-nightly'),
                   'Lean (version 4.34.1,', self.RELEASE.replace(', Release)', ', Debug)'),
                   'noise\n' + self.RELEASE, self.RELEASE + '\nnoise', self.RELEASE + '\n' + self.RELEASE]
        for text in invalid:
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, 'unexpected running Lean version'):
                gate.check_lean_version(text)

    def test_short_and_other_release_commits_refused(self):
        # A release label cannot authenticate a truncated or different compiler.
        for commit in ('a', '5045d005', '0' * 40, 'f' * 40):
            text = 'Lean (version 4.34.1, x86_64-unknown-linux-gnu, commit ' + commit + ', Release)'
            with self.subTest(commit=commit), self.assertRaisesRegex(ValueError, 'unexpected running Lean version'):
                gate.check_lean_version(text)

    def test_other_commit_preflight_preserves_build(self):
        self.fixture()
        wrong = 'Lean (version 4.34.1, x86_64-unknown-linux-gnu, commit ' + '0' * 40 + ', Release)'
        with self.assertRaisesRegex(ValueError, 'unexpected running Lean version'):
            self.execute_synthetic([wrong])
        self.assertEqual(self.sentinel.read_bytes(), b'old build')
        self.assertEqual(self.calls, [['lake', 'env', 'lean', '--version']])
        self.assert_no_receipt()

    def test_other_commit_after_execution_refuses_receipt(self):
        self.fixture()
        wrong = 'Lean (version 4.34.1, x86_64-unknown-linux-gnu, commit ' + 'f' * 40 + ', Release)'
        with self.assertRaisesRegex(ValueError, 'unexpected running Lean version'):
            self.execute_synthetic([self.RELEASE, wrong])
        self.assertEqual(self.version_index, 3)  # preflight, the helper's own check, final
        self.assertFalse(self.sentinel.exists())
        self.assert_no_receipt()

    def test_invalid_preflight_preserves_build_and_runs_no_build(self):
        self.fixture()
        with self.assertRaisesRegex(ValueError, 'unexpected running Lean version'):
            self.execute_synthetic([self.RELEASE.replace('4.34.1', '4.34.10')])
        self.assertTrue(self.sentinel.exists(), 'preflight refusal deleted the old build')
        self.assertEqual(self.calls, [['lake', 'env', 'lean', '--version']])
        self.assert_no_receipt()

    def test_preflight_process_failure_preserves_build(self):
        self.fixture()
        with self.assertRaisesRegex(ValueError, 'unexpected process outcome'):
            self.execute_synthetic([self.RELEASE], version_exit=1)
        self.assertTrue(self.sentinel.exists(), 'failed preflight deleted the old build')
        self.assertEqual(self.calls, [['lake', 'env', 'lean', '--version']])
        self.assert_no_receipt()

    def test_success_checks_before_build_and_after_controls_and_binds_both_logs(self):
        self.fixture()
        receipt = self.execute_synthetic([self.RELEASE, self.RELEASE])
        self.assertEqual(self.calls[0], ['lake', 'env', 'lean', '--version'])
        self.assertEqual(self.calls[1], ['lake', 'build'])
        self.assertEqual(self.calls[-1], ['lake', 'env', 'lean', '--version'])
        self.assertEqual(self.version_index, 3)  # preflight, the helper's own check, final
        self.assertFalse(self.sentinel.exists())
        self.assertEqual(receipt['lean_version'], self.RELEASE)
        self.assertEqual(receipt['manifest_sha256'], self.digest)
        self.assertEqual(receipt['checked_commit'], 'b' * 40)  # synthetic Git result
        self.assertEqual(set(receipt['negative_controls']), {'false_fold', 'false_power', 'sorry', 'custom_imported', 'native'})
        out = self.root / '.lake/formal-evidence'
        self.assertEqual(set(receipt['logs']), {p.name for p in out.glob('*.log')})
        for name in ('version-preflight.log', 'version.log'):
            self.assertEqual(receipt['logs'][name], hashlib.sha256((out / name).read_bytes()).hexdigest())

    def test_version_change_after_execution_refuses_receipt(self):
        self.fixture()
        with self.assertRaisesRegex(ValueError, 'unexpected running Lean version'):
            self.execute_synthetic([self.RELEASE, self.RELEASE.replace('4.34.1', '4.34.10')])
        self.assertEqual(self.version_index, 3)  # preflight, the helper's own check, final
        self.assert_no_receipt()

    def test_source_change_during_build_still_refuses_receipt(self):
        self.fixture()
        with self.assertRaisesRegex(ValueError, 'source hash mismatch: lake-manifest.json'):
            self.execute_synthetic([self.RELEASE], mutate_source=True)
        self.assert_no_receipt()

    def test_installed_dependency_change_still_refuses_receipt(self):
        self.fixture()
        with self.assertRaisesRegex(ValueError, 'installed dependency HEAD differs from lock'):
            self.execute_synthetic([self.RELEASE], wrong_dependency=True)
        self.assert_no_receipt()

class RuntimeGuardTests(unittest.TestCase):
    # The uniform runtime guard: the exact release, locked and clean dependencies and the bound
    # sources are observed before the first Lean execution and again after the last one, and the
    # two observations must be equal. These controls fail on a gate that checks only versions.
    # Reuses the synthetic-process harness above without re-running its tests.
    RELEASE = RuntimeVersionTests.RELEASE
    OTHER_PLATFORM = RELEASE.replace('x86_64-unknown-linux-gnu', 'aarch64-unknown-linux-gnu')
    fixture = RuntimeVersionTests.fixture
    execute_synthetic = RuntimeVersionTests.execute_synthetic
    assert_no_receipt = RuntimeVersionTests.assert_no_receipt

    def test_dirty_dependency_before_execution_refuses_and_preserves_build(self):
        self.fixture()
        with self.assertRaisesRegex(ValueError, 'dependency worktree is not clean'):
            self.execute_synthetic([self.RELEASE, self.RELEASE], dirty_dependency='preflight')
        self.assertEqual(self.calls, [['lake', 'env', 'lean', '--version']])
        self.assertTrue(self.sentinel.exists(), 'preflight refusal deleted the old build')
        self.assert_no_receipt()

    def test_wrong_dependency_before_execution_refuses_and_preserves_build(self):
        self.fixture()
        with self.assertRaisesRegex(ValueError, 'installed dependency HEAD differs from lock'):
            self.execute_synthetic([self.RELEASE, self.RELEASE], wrong_dependency=True)
        self.assertEqual(self.calls, [['lake', 'env', 'lean', '--version']])
        self.assertTrue(self.sentinel.exists(), 'preflight refusal deleted the old build')
        self.assert_no_receipt()

    def test_dependency_dirtied_during_execution_refuses_receipt(self):
        self.fixture()
        with self.assertRaisesRegex(ValueError, 'dependency worktree is not clean'):
            self.execute_synthetic([self.RELEASE, self.RELEASE], dirty_dependency='final')
        self.assertEqual(self.version_index, 3)  # the wrapper's final observation sees the dirty tree
        self.assert_no_receipt()

    def test_dependency_moved_during_execution_refuses_receipt(self):
        self.fixture()
        with self.assertRaisesRegex(ValueError, 'installed dependency HEAD differs from lock'):
            self.execute_synthetic([self.RELEASE, self.RELEASE], drift_dependency=True)
        self.assertEqual(self.version_index, 2)
        self.assert_no_receipt()

    def test_runtime_changed_between_observations_refuses_receipt(self):
        # Both lines are well-formed records of the pinned release; only the platform differs.
        self.fixture()
        self.assertEqual(gate.check_lean_version(self.OTHER_PLATFORM), self.OTHER_PLATFORM)
        with self.assertRaisesRegex(ValueError, 'changed during execution'):
            self.execute_synthetic([self.RELEASE, self.OTHER_PLATFORM])
        self.assert_no_receipt()

    def test_receipt_binds_attempt_and_both_equal_observations(self):
        self.fixture()
        receipt = self.execute_synthetic([self.RELEASE, self.RELEASE],
                                         env={'GITHUB_RUN_ID': '12345', 'GITHUB_RUN_ATTEMPT': '2'})
        self.assertEqual(receipt['workflow_run_id'], '12345')
        self.assertEqual(receipt['workflow_run_attempt'], '2')
        self.assertEqual(receipt['dependency_worktrees'], 'CLEAN')
        expected = {'lean_version': self.RELEASE, 'dependency_revisions': self.manifest['dependency_revisions'],
                    'dependency_worktrees': 'CLEAN', 'manifest_sha256': self.digest}
        self.assertEqual(receipt['observations'], {'preflight': expected, 'final': expected})
        written = json.loads((self.root / '.lake/formal-evidence/receipt.json').read_text())
        self.assertEqual(written['observations'], receipt['observations'])

class SuccessorTests(unittest.TestCase):
    def test_only_documented_algebra_repairs(self):
        original=(ROOT/'originals/Algebra.lean.txt').read_text()
        expected=original.replace('def foldPotential', 'noncomputable def foldPotential').replace('  field_simp [hr]\n  ring\n', '  field_simp [hr]\n')
        self.assertEqual((ROOT/'ResearchFormalCoreR1/AlgebraV2.lean').read_text(), expected)
    def test_only_documented_probability_repair(self):
        original=(ROOT/'originals/ProbabilityCompanions.lean.txt').read_text()
        expected=original.replace('  field_simp [hr, hcZ]\n  ring\n', '  field_simp [hr, hcZ]\n')
        self.assertEqual((ROOT/'ResearchFormalCoreR1/ProbabilityCompanionsV2.lean').read_text(), expected)

if __name__ == '__main__': unittest.main()
