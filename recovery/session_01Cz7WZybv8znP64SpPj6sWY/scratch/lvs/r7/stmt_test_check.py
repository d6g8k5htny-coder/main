#!/usr/bin/env python3
"""Positive and negative controls for check.py (standard library only).

Synthetic controls build a disposable repository with a miniature packet (five modules, two
pinned sources, a register) so that every refusal is exercised in isolation. Real-tree controls
check the committed packet's pins and module scans. Lean is required only by the one test that
is guarded by `skipUnless(lake)`; it builds a dependency-free fixture package, never Mathlib.
No test here is evidence about any mathematical claim; scientific effect NONE.
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('lean_spec_check', HERE / 'check.py')
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)

COMMIT = '760340e921ac4ceda296b8118da936f1133e956e'
SOURCE_TEXT = 'Section 1. In the source, two and two make four; a given object is named here.\n'
LANDING_TEXT = json.dumps({'claims': [{'claim_id': 'demo-claim', 'disposition': 'HOLD_WITH_DOMAIN'}]}) + '\n'
LOCK = {'version': '1.2.0', 'packagesDir': '.lake/packages', 'packages': [
    {'name': 'batteries', 'url': 'https://github.com/leanprover-community/batteries', 'type': 'git', 'subDir': None,
     'scope': 'leanprover-community', 'rev': 'f2effa3d803fda822b1f97b806c47cf2adfbcbc2', 'manifestFile': 'lake-manifest.json',
     'inputRev': 'main', 'inherited': True, 'configFile': 'lakefile.toml'},
    {'name': 'mathlib', 'url': 'https://github.com/leanprover-community/mathlib4.git', 'type': 'git', 'subDir': None,
     'scope': '', 'rev': check.MATHLIB_REV, 'manifestFile': 'lake-manifest.json', 'inputRev': check.MATHLIB_REV,
     'inherited': False, 'configFile': 'lakefile.lean'}],
    'name': 'universal_law_formal', 'lakeDir': '.lake', 'fixedToolchain': True}
LAKEFILE = ('name = "universal_law_specifications"\nversion = "0.1.0"\ndefaultTargets = ["Specifications"]\n\n[[require]]\n'
            'name = "mathlib"\ngit = "https://github.com/leanprover-community/mathlib4.git"\nrev = "' + check.MATHLIB_REV
            + '"\n\n[[lean_lib]]\nname = "Specifications"\n' + check.LEAN_OPTIONS_LINE + '\n')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def blob(data):
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def labels(anchor='two and two make four', text=SOURCE_TEXT, path='docs/DEMO.md', interface='none.'):
    return (f'Source: d6g8k5htny-coder/Math- commit {COMMIT} path {path} sha256 {sha(text.encode())}\n'
            f'Anchor: "{anchor}"\nInterface: {interface}\nDoes not claim: anything about the source theorem.')


FIELD = f'''import Mathlib

/-! Demo module: statements only. The word theorem may appear in a module docstring. -/

namespace UniversalLaw.Spec.Field

open MeasureTheory

/-- Ambient space (auxiliary; not registered). -/
abbrev E (d : ℕ) : Type := EuclideanSpace ℝ (Fin d)

/-- A helper constant (auxiliary; not registered). -/
noncomputable def helper (x : ℝ) : ℝ := x + 1

/-- Two and two.

{labels()} -/
def TwoAndTwo (n : ℕ) (h : n = 2 := by rfl) : Prop := n + n = 4

/-- An interface.

{labels('a given object is named here', interface='the object is given.')} -/
structure DemoInterface (d : ℕ) where
  x : E d
  pos : 0 < d

/-- A multi-line header with a Prop-valued binder.

{labels()} -/
def MultiLine (d : ℕ) (I : DemoInterface d)
    (P : (E d → ℝ) → Prop) :
    Prop :=
  P (fun _ => 0) ∧ 0 < d

end UniversalLaw.Spec.Field
'''


def module(stem):
    return (f'import Mathlib\n\nnamespace UniversalLaw.Spec.{stem}\n\n/-- Demo statement.\n\n{labels()} -/\n'
            f'def Demo{stem} : Prop := (2 : ℕ) + 2 = 4\n\nend UniversalLaw.Spec.{stem}\n')


def statement(module_path, declaration, kind='def_prop', package=check.PACKAGE, progress='specified', anchor='two and two make four'):
    return {'package': package, 'module': module_path, 'declaration': declaration, 'kind': kind, 'formal_progress': progress,
            'anchor': anchor, 'does_not_claim': 'the source theorem; review; acceptance'}


class Fixture:
    """A disposable repository with a miniature packet that passes the source check."""

    def __init__(self):
        self.tmp = tempfile.mkdtemp()
        self.repo = Path(self.tmp) / 'repo'
        self.packet = self.repo / check.PACKAGE
        (self.packet / 'Specifications').mkdir(parents=True)
        self.write('formal/lake-manifest.json', json.dumps(LOCK, indent=2) + '\n', repo=True)
        self.write(check.WORKFLOW, 'name: placeholder\n', repo=True)
        self.write('.gitignore', '.lake/\n__pycache__/\n')
        for name in ('README.md', 'SCOPE.md', 'check.py', 'test_check.py', 'replay.sh'):
            self.write(name, '# bound copy\n')
        self.write('lakefile.toml', LAKEFILE)
        self.write('lean-toolchain', check.TOOLCHAIN + '\n')
        self.write('lake-manifest.json', json.dumps(dict(LOCK, name=check.LOCK_NAME), indent=2) + '\n')
        self.write('Specifications.lean', ''.join('import ' + m[:-5].replace('/', '.') + '\n' for m in check.MODULES))
        self.write('Specifications/Field.lean', FIELD)
        for m in check.MODULES[1:]:
            self.write(m, module(Path(m).stem))
        self.write('sources/demo/DEMO.md', SOURCE_TEXT)
        self.write('sources/landing-claims/LANDING_CLAIMS.json', LANDING_TEXT)
        sources = [self.source('demo', 'docs/DEMO.md', SOURCE_TEXT, 'sources/demo/DEMO.md'),
                   self.source('landing-claims', 'claims/LANDING_CLAIMS.json', LANDING_TEXT, 'sources/landing-claims/LANDING_CLAIMS.json')]
        self.write('sources/SOURCES.json', json.dumps({'schema_version': 1, 'scientific_effect': 'NONE', 'meaning': 'pins',
                                                        'sources': sources}, indent=2) + '\n')
        register = {
            'schema_version': 1, 'object': check.REGISTER_OBJECT, 'scientific_effect': 'NONE', 'scientific_status_authority': False,
            'alignment_status': 'PENDING_INDEPENDENT_REVIEW', 'meaning': 'inventory of formal statements; labels only',
            'toolchain': {'lean': check.TOOLCHAIN, 'lean_commit': check.LEAN_COMMIT, 'mathlib': check.MATHLIB_REV},
            'vocabulary': {'formal_progress': list(check.FORMAL_PROGRESS)},
            'sources': [{k: s[k] for k in check.SOURCE_KEYS} for s in sources],
            'claims': [
                {'claim_id': 'demo-claim', 'kind': 'landing_claim', 'title': 'Demo claim', 'layer0': {'source_id': 'demo', 'heading': 'Theorem'},
                 'disposition_transcribed': 'HOLD_WITH_DOMAIN', 'formal_progress': 'specified',
                 'statements': [statement('Specifications/Field.lean', 'UniversalLaw.Spec.Field.TwoAndTwo'),
                                statement('Specifications/Field.lean', 'UniversalLaw.Spec.Field.DemoInterface', 'structure', anchor='a given object is named here'),
                                statement('Specifications/Field.lean', 'UniversalLaw.Spec.Field.MultiLine'),
                                statement('UniversalLaw/Demo.lean', 'UniversalLaw.Demo.two_add', 'theorem', 'formal', 'proved', 'two and two make four')],
                 'kernel_evidence': [{'package': 'formal', 'receipt_or_workflow': 'formal-lean.yml', 'note': 'arithmetic skeleton only'}],
                 'not_established': 'the claim itself', 'next_step': 'independent alignment review'},
                {'claim_id': 'demo-graph-node', 'kind': 'graph_node', 'title': 'Demo node', 'layer0': {'source_id': 'demo', 'heading': None},
                 'disposition_transcribed': None, 'formal_progress': 'specified',
                 'statements': [statement(m, f'UniversalLaw.Spec.{Path(m).stem}.Demo{Path(m).stem}') for m in check.MODULES[1:]],
                 'kernel_evidence': [], 'not_established': 'everything beyond the statement', 'next_step': 'none planned'}]}
        self.write('REGISTER.json', json.dumps(register, indent=2, ensure_ascii=False) + '\n')

    @staticmethod
    def source(sid, path, text, local):
        data = text.encode()
        return {'id': sid, 'repository': 'd6g8k5htny-coder/Math-', 'commit': COMMIT, 'path': path, 'blob': blob(data),
                'bytes': len(data), 'sha256': sha(data), 'local_copy': local}

    def path(self, rel, repo=False):
        return (self.repo if repo else self.packet) / rel

    def write(self, rel, text, repo=False):
        p = self.path(rel, repo)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding='utf-8')

    def read(self, rel, repo=False):
        return self.path(rel, repo).read_text(encoding='utf-8')

    def mutate(self, rel, old, new, repo=False, count=1):
        text = self.read(rel, repo)
        if old not in text:
            raise AssertionError(f'{old!r} not in {rel}')
        self.write(rel, text.replace(old, new, count), repo)

    def inject(self, rel, line):
        """Insert a column-0 line before the closing `end` of a module."""
        text = self.read(rel)
        head, _, tail = text.rpartition('\nend ')
        self.write(rel, head + '\n' + line + '\n\nend ' + tail)

    def register(self):
        return json.loads(self.read('REGISTER.json'))

    def save_register(self, reg):
        self.write('REGISTER.json', json.dumps(reg, indent=2, ensure_ascii=False) + '\n')

    def run(self, allow_dirty=True):
        return check.source_check(self.packet, self.repo, allow_dirty)

    def cleanup(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class Base(unittest.TestCase):
    def setUp(self):
        # Fixture packets are synthetic: their pinned commits exist in no git history and their
        # worktree is not a CI checkout, so the CI-only requirements that check.py applies when
        # GITHUB_ACTIONS is set (GITHUB_SHA == HEAD, repository name, every pinned source verified
        # against git history) must not apply to them. Identity.test_clean_worktree_and_ci re-enables
        # CI mode explicitly inside its own patch to test exactly those refusals.
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        for key in ('GITHUB_ACTIONS', 'GITHUB_SHA', 'GITHUB_REPOSITORY', 'GITHUB_RUN_ID', 'GITHUB_RUN_ATTEMPT'):
            os.environ.pop(key, None)
        self.f = Fixture()
        self.addCleanup(self.f.cleanup)

    def assertRefused(self, fragment, allow_dirty=True):
        with self.assertRaises(ValueError) as ctx:
            self.f.run(allow_dirty)
        self.assertIn(fragment, str(ctx.exception))
        return str(ctx.exception)


class Positive(Base):
    def test_fixture_passes(self):
        info = self.f.run()
        self.assertEqual(info['prop_definitions'], 6)
        self.assertEqual(info['structures'], 1)
        self.assertEqual(info['auxiliary_definitions'], 2)
        self.assertEqual((info['claims'], info['statements'], info['registered_local_statements']), (2, 8, 7))
        self.assertEqual(info['formalization_status'], 'specified')
        self.assertEqual(info['alignment_status'], 'PENDING_INDEPENDENT_REVIEW')
        self.assertEqual((info['checked_commit'], info['worktree']), (None, 'UNVERIFIED'))
        self.assertEqual(info['dependency_revisions']['mathlib'], check.MATHLIB_REV)
        names = {d['name'] for d in info['scans']['Specifications/Field.lean']}
        self.assertEqual(names, {'UniversalLaw.Spec.Field.E', 'UniversalLaw.Spec.Field.helper', 'UniversalLaw.Spec.Field.TwoAndTwo',
                                 'UniversalLaw.Spec.Field.DemoInterface', 'UniversalLaw.Spec.Field.MultiLine'})

    def test_cli(self):
        cmd = [sys.executable, '-B', '-S', str(HERE / 'check.py'), '--root', str(self.f.packet), '--repo', str(self.f.repo), '--allow-dirty']
        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn('SOURCE_CHECK_PASS', proc.stdout)
        self.assertIn('WORKTREE_UNVERIFIED', proc.stderr)
        proc = subprocess.run(cmd + ['source'], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, 'the DESIGN 3b spelling `check.py source` is the default mode: ' + proc.stderr)
        self.assertIn('SOURCE_CHECK_PASS', proc.stdout)
        proc = subprocess.run(cmd + ['execute'], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 2, 'only `source` is a positional mode')
        self.assertNotIn('SOURCE_CHECK_PASS', proc.stdout)
        self.f.inject('Specifications/RN.lean', 'theorem injected : True := trivial')
        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 1)
        self.assertIn('LEAN_SPEC_CHECK_FAIL', proc.stderr)
        self.assertIn('refused token', proc.stderr)
        self.assertNotIn('SOURCE_CHECK_PASS', proc.stdout)

    def test_comments_may_mention_refused_words(self):
        self.f.inject('Specifications/RN.lean', '-- theorem sorry axiom native_decide set_option')
        self.f.mutate('Specifications/RN.lean', 'Demo statement.', 'Demo statement; the source says theorem and the proof uses sorry nowhere.')
        self.f.run()

    def test_unnamed_end_is_refused(self):
        self.f.mutate('Specifications/RN.lean', '\nend UniversalLaw.Spec.RN', '\nend\n\nend UniversalLaw.Spec.RN')
        self.assertRefused('does not close')


class Identity(Base):
    def test_requires_git_without_allow_dirty(self):
        self.assertRefused('not a git repository', allow_dirty=False)

    @unittest.skipUnless(shutil.which('git'), 'git not available')
    def test_clean_and_dirty_worktrees(self):
        def g(*args):
            return subprocess.run(['git', '-C', str(self.f.repo), '-c', 'user.name=t', '-c', 'user.email=t@t', '-c', 'commit.gpgsign=false',
                                   *args], capture_output=True, text=True, check=True).stdout
        g('init', '-q')
        g('add', '-A')
        g('commit', '-q', '-m', 'fixture')
        with mock.patch.dict(os.environ):
            os.environ.pop('GITHUB_ACTIONS', None)
            info = self.f.run(allow_dirty=False)
            self.assertEqual(info['worktree'], 'CLEAN')
            self.assertRegex(info['checked_commit'], '^[0-9a-f]{40}$')
            self.f.mutate('Specifications/RN.lean', '(2 : ℕ) + 2 = 4', '(2 : ℕ) + 2 = 5')
            self.assertRefused('uncommitted or untracked', allow_dirty=False)
            g('checkout', '--', '.')
            self.f.write('sources/demo/EXTRA.md', 'untracked\n')
            self.assertRefused('uncommitted or untracked', allow_dirty=False)
            self.f.path('sources/demo/EXTRA.md').unlink()
            self.f.mutate('formal/lake-manifest.json', '"version": "1.2.0"', '"version": "1.2.1"', repo=True)
            self.assertRefused('uncommitted or untracked', allow_dirty=False)
            g('checkout', '--', '.')
            self.f.run(allow_dirty=False)
            os.environ['GITHUB_ACTIONS'] = 'true'
            os.environ['GITHUB_SHA'] = '0' * 40
            os.environ['GITHUB_REPOSITORY'] = check.REPOSITORY
            self.assertRefused('GITHUB_SHA', allow_dirty=False)
            os.environ['GITHUB_SHA'] = info['checked_commit']
            os.environ['GITHUB_REPOSITORY'] = 'someone/fork'
            self.assertRefused('GITHUB_REPOSITORY', allow_dirty=False)
            os.environ['GITHUB_REPOSITORY'] = check.REPOSITORY
            # The fixture history does not contain the pinned commit, so CI must refuse (fetch-depth 0 is required).
            self.assertRefused('verified against its commit in git history', allow_dirty=False)


class Lexical(Base):
    def test_refused_tokens(self):
        cases = {'theorem': 'theorem injected : True := trivial', 'lemma': 'lemma injected : True := trivial',
                 'example': 'example : True := trivial', 'instance': 'instance : Inhabited ℕ := ⟨0⟩',
                 'axiom': 'axiom hidden : False', 'sorry': 'def Injected : Prop := sorry',
                 'admit': 'def Injected : Prop := by admit', 'native_decide': 'def Injected : Prop := (by native_decide : 2 + 2 = 4)',
                 'unsafe': 'unsafe def Injected : Prop := True', 'opaque': 'opaque Injected : Prop',
                 'partial': 'partial def Injected (n : ℕ) : ℕ := Injected n', 'macro': 'macro "tt" : term => `(True)',
                 'notation': 'notation "⊤⊤" => True', 'set_option': 'set_option autoImplicit true',
                 'syntax': 'syntax "tt" : term', 'elab': 'elab "tt" : term => pure (mkConst ``True)',
                 'implemented_by': '@[implemented_by helper] def Injected : Prop := True', 'extern': '@[extern "x"] def Injected : Prop := True',
                 'deriving': 'deriving instance Repr for Nat', 'attribute': 'attribute [simp] helper',
                 'inductive': 'inductive Injected | a', 'class': 'class Injected where\n  x : ℕ',
                 'local': 'local notation "x" => True', 'include': 'include x', 'mutual': 'mutual\nend'}
        for token, line in cases.items():
            with self.subTest(token=token):
                f = Fixture()
                self.addCleanup(f.cleanup)
                f.inject('Specifications/RN.lean', line)
                with self.assertRaises(ValueError) as ctx:
                    f.run()
                self.assertIn('refused token', str(ctx.exception))
                self.assertIn(token, str(ctx.exception))

    def test_refused_token_detection_unit(self):
        self.assertEqual(check.refused_tokens('theorem t : False := by sorry\n'), ['sorry', 'theorem'])
        self.assertEqual(check.refused_tokens('def partialSups : ℕ := 0\n'), [])
        self.assertEqual(check.refused_tokens('def x := Foo.sorry\n'), ['sorry'])

    def test_column0_commands(self):
        for line, fragment in (('@[simp] def Injected : Prop := True', 'refused character'),
                               ('private def Injected : Prop := True', 'off column 0'),
                               ('protected def Injected : Prop := True', 'off column 0'),
                               ('universe u', 'column-0 command'),
                               ('noncomputable section', 'off column 0'),
                               ('| injected => True', 'column-0 command')):
            with self.subTest(line=line):
                f = Fixture()
                self.addCleanup(f.cleanup)
                f.inject('Specifications/RN.lean', line)
                with self.assertRaises(ValueError) as ctx:
                    f.run()
                self.assertIn(fragment, str(ctx.exception))

    def test_section_and_variable_are_refused(self):
        """`variable (h : False)` mentioned in a body adds a vacuating hypothesis the scan cannot see,
        so both commands are refused until a module needs them (DESIGN 3b lists them; none uses them)."""
        for line in ('section\nend', 'section Named\nend Named', 'variable (h : False)', 'variable (d : ℕ)', 'noncomputable section\nend'):
            with self.subTest(line=line):
                f = Fixture()
                self.addCleanup(f.cleanup)
                f.inject('Specifications/RN.lean', line)
                with self.assertRaises(ValueError) as ctx:
                    f.run()
                self.assertRegex(str(ctx.exception), 'column-0 command not allowed|off column 0')
        self.f.inject('Specifications/RN.lean', 'variable (h : False)\n\n/-- s\n\n' + labels() + ' -/\ndef Vacuous : Prop := h.elim')
        reg = self.f.register()
        reg['claims'][1]['statements'].append(statement('Specifications/RN.lean', 'UniversalLaw.Spec.RN.Vacuous'))
        self.f.save_register(reg)
        self.assertIn("'variable'", self.assertRefused('column-0 command not allowed'))

    def test_imports_must_be_mathlib(self):
        self.f.mutate('Specifications/RN.lean', 'import Mathlib\n', 'import Mathlib.Data.Real.Basic\n')
        self.assertRefused('imports must be exactly')
        self.f.mutate('Specifications/RN.lean', 'import Mathlib.Data.Real.Basic\n', 'import Mathlib\nimport Specifications.Field\n')
        self.assertRefused('imports must be exactly')


class Escapes(Base):
    """The escape families found by the adversarial review of the first checker: constructs that
    Lean 4.34.1 accepts and a regex scanner over a naive comment stripper let through. Each must
    be refused by the source scan; the environment audit of --execute is the second, independent
    refusal point (LeanHelpers.test_fixture_package)."""

    def register_rn(self, *names):
        reg = self.f.register()
        for n in names:
            reg['claims'][1]['statements'].append(statement('Specifications/RN.lean', 'UniversalLaw.Spec.RN.' + n))
        self.f.save_register(reg)

    def test_theorem_axiom_sorry_between_string_literals_with_opener_registered(self):
        self.f.inject('Specifications/RN.lean', f'/-- s\n\n{labels()} -/\ndef Opener : Prop := "/-" = "x"\naxiom hidden : False\n'
                      'theorem smuggled : False := hidden\ndef Sorried : Prop := sorry\ndef Closer : Prop := "-/" = "x"')
        self.register_rn('Opener')
        msg = self.assertRefused('refused token')
        for tok in ('axiom', 'sorry', 'theorem'):
            self.assertIn(tok, msg, 'string literals no longer open a fake comment')

    def test_string_literal_alone_is_refused(self):
        self.f.inject('Specifications/RN.lean', f'/-- s\n\n{labels()} -/\ndef S : Prop := "x" = "x"')
        self.register_rn('S')
        self.assertRefused('refused character')

    def test_line_comment_desync_with_statement_registered(self):
        self.f.inject('Specifications/RN.lean', f'/-- s\n\n{labels()} -/\ndef S2 : Prop := "--" = "" ∧ sorry')
        self.register_rn('S2')
        self.assertIn('sorry', self.assertRefused('refused token'))

    def test_guillemet_desync(self):
        self.f.inject('Specifications/RN.lean', 'def «/-» : ℕ := 0\ntheorem smuggled : True := trivial\ndef «-/» : ℕ := 0')
        self.assertIn('theorem', self.assertRefused('refused token'))
        self.f = Fixture()
        self.addCleanup(self.f.cleanup)
        self.f.inject('Specifications/RN.lean', 'def «x y» : ℕ := 0')
        self.assertRefused('refused character')

    def test_indented_commands(self):
        for line, fragment in ((' def Hidden : Prop := False', 'off column 0'),
                               ('  def proofHidden : (2 : ℕ) + 2 = 4 := rfl', 'off column 0'),
                               (' @[simp] def Hidden : Prop := False', 'refused character'),
                               (' #eval IO.println "escaped"', 'refused character'),
                               (' #check Nat', 'refused character'),
                               (' namespace Other\n structure Hidden where\n   n : ℕ\n end Other', 'off column 0'),
                               (' private def Hidden : Prop := False', 'off column 0'),
                               (' open Classical', 'off column 0'),
                               (' section\n end', 'off column 0'),
                               (' abbrev Hidden : Prop := True', 'off column 0')):
            with self.subTest(line=line):
                f = Fixture()
                self.addCleanup(f.cleanup)
                f.inject('Specifications/RN.lean', line)
                with self.assertRaises(ValueError) as ctx:
                    f.run()
                self.assertIn(fragment, str(ctx.exception))

    def test_in_chains(self):
        for line, fragment in (('open Classical in def Hidden : Prop := True', 'off column 0'),
                               ('variable (n : ℕ) in def Hidden : Prop := n = n', 'off column 0'),
                               ('open Classical in', '`open … in`'),
                               ('open Classical\n  Real in', '`open … in`'),
                               ('variable (n : ℕ) in', 'column-0 command not allowed'),
                               ('variable (n : ℕ)\n  (m : ℕ) in', 'column-0 command not allowed')):
            with self.subTest(line=line):
                f = Fixture()
                self.addCleanup(f.cleanup)
                f.inject('Specifications/RN.lean', line)
                with self.assertRaises(ValueError) as ctx:
                    f.run()
                self.assertIn(fragment, str(ctx.exception))

    def test_in_inside_binders_is_fine(self):
        self.f.inject('Specifications/RN.lean', 'open Finset\n  (sum)\n\ndef inBinders (s : Finset ℕ) : ℕ := ∑ i in s, i')
        info = self.f.run()
        self.assertEqual(info['auxiliary_definitions'], 3)

    def test_two_commands_on_one_line(self):
        self.f.inject('Specifications/RN.lean', f'/-- a\n\n{labels()} -/\ndef A : Prop := True def B : Prop := False')
        self.register_rn('A')
        self.assertRefused('off column 0')

    def test_proof_written_as_definition(self):
        for line in ('def proofAsDef : (2 : ℕ) + 2 = 4 := rfl', 'abbrev proofAsAbbrev : True := trivial',
                     'def proofImp : (1 : ℕ) = 1 → True := fun _ => trivial', 'def proofForall : ∀ n : ℕ, n = n := fun _ => rfl',
                     'noncomputable def proofLe : (1 : ℝ) ≤ 2 := by norm_num', 'def proofMem (s : Set ℕ) (h : 0 ∈ s) : 0 ∈ s := h',
                     'def proofNot : ¬ False := id', 'def proofFalse : False → True := fun h => h.elim'):
            with self.subTest(line=line):
                f = Fixture()
                self.addCleanup(f.cleanup)
                f.inject('Specifications/RN.lean', line)
                with self.assertRaises(ValueError) as ctx:
                    f.run()
                self.assertIn('proof-shaped', str(ctx.exception))

    def test_relations_inside_brackets_are_types_not_proofs(self):
        self.f.inject('Specifications/RN.lean', 'def dec (n : ℕ) : Decidable (n = 2) := inferInstance\n'
                                                'def two : {x : ℕ // x = 2} := ⟨2, rfl⟩\ndef pred : Set (ℕ × ℕ) := {p | p.1 ≤ p.2}')
        info = self.f.run()
        self.assertEqual(info['auxiliary_definitions'], 5)

    def test_untyped_definition(self):
        for line in ('def Untyped := (2 : ℕ) + 2 = 4', 'abbrev U := True', 'noncomputable def c := (1 : ℝ)'):
            with self.subTest(line=line):
                f = Fixture()
                self.addCleanup(f.cleanup)
                f.inject('Specifications/RN.lean', line)
                with self.assertRaises(ValueError) as ctx:
                    f.run()
                self.assertIn('explicit result type', str(ctx.exception))

    def test_sort_and_root(self):
        self.f.inject('Specifications/RN.lean', 'def SortZero : Sort 0 := True')
        self.assertIn('Sort', self.assertRefused('refused token'))
        self.f = Fixture()
        self.addCleanup(self.f.cleanup)
        self.f.inject('Specifications/RN.lean', 'def _root_.Escaped : ℕ := 0')
        self.assertIn('_root_', self.assertRefused('refused token'))

    def test_underscore_name_components(self):
        for name in ('_hidden', 'Inner._hidden'):
            with self.subTest(name=name):
                f = Fixture()
                self.addCleanup(f.cleanup)
                f.inject('Specifications/RN.lean', f'def {name} : ℕ := 0')
                with self.assertRaises(ValueError) as ctx:
                    f.run()
                self.assertIn('may not start with `_`', str(ctx.exception))

    def test_commands_that_add_declarations_under_other_names(self):
        for token, line in (('export', 'export Nat (succ)'), ('alias', 'alias succ := Nat.succ'),
                            ('irreducible_def', 'irreducible_def x : ℕ := 0'), ('universe', 'universe u')):
            with self.subTest(token=token):
                f = Fixture()
                self.addCleanup(f.cleanup)
                f.inject('Specifications/RN.lean', line)
                with self.assertRaises(ValueError) as ctx:
                    f.run()
                self.assertIn('refused token' if token != 'universe' else 'column-0 command', str(ctx.exception))
                self.assertIn(token, str(ctx.exception))


class Root(Base):
    def test_order(self):
        self.f.write('Specifications.lean', 'import Specifications.Lifetime\nimport Specifications.Field\nimport Specifications.Side24\n'
                                            'import Specifications.RN\nimport Specifications.P15\n')
        self.assertRefused('root module must import')

    def test_missing_and_extra(self):
        self.f.write('Specifications.lean', ''.join('import ' + m[:-5].replace('/', '.') + '\n' for m in check.MODULES[:-1]))
        self.assertRefused('root module must import')
        self.f.write('Specifications.lean', ''.join('import ' + m[:-5].replace('/', '.') + '\n' for m in check.MODULES) + 'import Mathlib\n')
        self.assertRefused('root module must import')
        self.f.write('Specifications.lean', ''.join('import ' + m[:-5].replace('/', '.') + '\n' for m in check.MODULES) + 'def x : Prop := True\n')
        self.assertRefused('root module must import')

    def test_comments_in_root_ignored(self):
        self.f.write('Specifications.lean', '/-! root -/\n' + ''.join('import ' + m[:-5].replace('/', '.') + '\n' for m in check.MODULES))
        self.f.run()


class Namespaces(Base):
    def test_wrong_namespace(self):
        self.f.mutate('Specifications/RN.lean', 'UniversalLaw.Spec.RN', 'UniversalLaw.Spec.Other', count=2)
        self.assertRefused('outside namespace UniversalLaw.Spec.RN')
        self.f.write('Specifications/RN.lean', 'import Mathlib\n\nnamespace UniversalLaw.Spec.Other\n\nend UniversalLaw.Spec.Other\n')
        self.assertRefused('exactly one namespace')

    def test_two_namespaces(self):
        self.f.inject('Specifications/RN.lean', 'namespace UniversalLaw.Spec.RN.Inner\nend UniversalLaw.Spec.RN.Inner')
        self.assertRefused('exactly one namespace')

    def test_unclosed(self):
        self.f.mutate('Specifications/RN.lean', '\nend UniversalLaw.Spec.RN\n', '\n')
        self.assertRefused('unclosed namespace')

    def test_mismatched_end(self):
        self.f.mutate('Specifications/RN.lean', '\nend UniversalLaw.Spec.RN\n', '\nend UniversalLaw.Spec.Field\n')
        self.assertRefused('does not close')

    def test_declaration_outside_namespace(self):
        self.f.mutate('Specifications/RN.lean', 'namespace UniversalLaw.Spec.RN\n', 'def Early : Prop := True\n\nnamespace UniversalLaw.Spec.RN\n')
        self.assertRefused('declaration outside namespace')


class Scanner(unittest.TestCase):
    def sources(self):
        return {'demo': {'id': 'demo', 'repository': 'd6g8k5htny-coder/Math-', 'commit': COMMIT, 'path': 'docs/DEMO.md',
                         'sha256': sha(SOURCE_TEXT.encode()), 'text': SOURCE_TEXT}}

    def scan(self, body, stem='Field'):
        return check.scan_module(f'import Mathlib\n\nnamespace UniversalLaw.Spec.{stem}\n\n{body}\n\nend UniversalLaw.Spec.{stem}\n', stem, self.sources())

    def test_fixture_module_shapes(self):
        decls = {d['name']: d for d in check.scan_module(FIELD, 'Field', self.sources())}
        self.assertEqual(decls['UniversalLaw.Spec.Field.E']['kind'], None)
        self.assertEqual(decls['UniversalLaw.Spec.Field.helper'], {'name': 'UniversalLaw.Spec.Field.helper', 'lean': 'def', 'kind': None,
                                                                    'result': 'ℝ', 'line': 13})
        self.assertEqual(decls['UniversalLaw.Spec.Field.TwoAndTwo']['kind'], 'def_prop')
        self.assertEqual(decls['UniversalLaw.Spec.Field.MultiLine']['result'], 'Prop')
        self.assertEqual(decls['UniversalLaw.Spec.Field.DemoInterface']['kind'], 'structure')
        self.assertEqual(decls['UniversalLaw.Spec.Field.DemoInterface']['doc_anchor'], 'a given object is named here')
        self.assertEqual(decls['UniversalLaw.Spec.Field.TwoAndTwo']['source_id'], 'demo')

    def test_where_definition_is_auxiliary(self):
        decls = self.scan('structure Data where\n  n : ℕ\n\ndef sample : Data where\n  n := 2\n'.replace('structure Data', f'/-- s\n\n{labels()} -/\nstructure Data'))
        self.assertEqual([d['kind'] for d in decls], ['structure', None])

    def test_arrow_prop_result_refused(self):
        with self.assertRaises(ValueError) as ctx:
            self.scan(f'/-- p\n\n{labels()} -/\ndef Pred : ℕ → Prop := fun n => n = 0')
        self.assertIn('exactly as `: Prop`', str(ctx.exception))
        with self.assertRaises(ValueError):
            self.scan(f'/-- p\n\n{labels()} -/\ndef Pred (n : ℕ) : (Prop) := n = 0')

    def test_prop_abbrev_is_a_registered_kind(self):
        decls = self.scan(f'/-- p\n\n{labels()} -/\nabbrev Short : Prop := True')
        self.assertEqual(decls[0]['kind'], 'def_prop')
        self.assertEqual(decls[0]['lean'], 'abbrev')

    def test_pattern_arms_end_header(self):
        decls = self.scan('def f : ℕ → ℕ\n  | 0 => 1\n  | n + 1 => n')
        self.assertEqual(decls[0]['result'], 'ℕ → ℕ')

    def test_docstring_requirements(self):
        with self.assertRaises(ValueError) as ctx:
            self.scan('def Bare : Prop := True')
        self.assertIn('has no docstring', str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            self.scan('/-! module doc, not a docstring -/\ndef Bare : Prop := True')
        self.assertIn('has no docstring', str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            self.scan(f'/-- p\n\n{labels()} -/\n\ndef Gap : ℕ := 0\n\ndef Bare : Prop := True')
        self.assertIn('has no docstring', str(ctx.exception))
        for label in check.LABELS:
            with self.subTest(label=label), self.assertRaises(ValueError) as ctx:
                self.scan(f'/-- p\n\n{labels().replace(label, label[:-1] + ".")} -/\ndef P : Prop := True')
            self.assertIn('exactly one', str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            self.scan(f'/-- p\n\n{labels(anchor="two and three make four")} -/\ndef P : Prop := True')
        self.assertIn('docstring anchor is not a verbatim', str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            self.scan(f'/-- p\n\n{labels(anchor="two and two")} -/\ndef P : Prop := True')
        self.assertIn('shorter than 16', str(ctx.exception), 'a short verbatim anchor locates nothing')
        self.scan(f'/-- p\n\n{labels(anchor="two and two make")} -/\ndef P : Prop := True')
        with self.assertRaises(ValueError) as ctx:
            self.scan(f'/-- p\n\n{labels(text="other bytes")} -/\ndef P : Prop := True')
        self.assertIn('names no registered', str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            self.scan(f'/-- p\n\n{labels()} -/\ndef P : Prop := True'.replace('Anchor: "two and two make four"', 'Anchor: two and two make four'))
        self.assertIn('double-quoted', str(ctx.exception))
        self.scan(f'/-- p\n\n{labels()} -/\ndef P : Prop := True'.replace('Does not claim: anything', 'Does not claim:\n  anything'))

    def test_lex(self):
        code, docs = check.lex('/- a /- nested -/ b -/\n-- line /- not opened\n/-- doc -/\ndef x := 1 -- tail\n')
        self.assertEqual([line.strip() for line in code.split('\n')], ['', '', '', 'def x := 1', ''])
        self.assertEqual(len(code.split('\n')), 5, 'newlines are preserved so line numbers stay stable')
        self.assertEqual(docs, [(2, 2, '/-- doc -/')])
        with self.assertRaises(ValueError):
            check.lex('/- open\n')

    def test_lex_strings_and_guillemets_do_not_open_comments(self):
        code, _ = check.lex('def a := "/-" -- c\ntheorem t : True := trivial\ndef b := "-/"\n')
        self.assertIn('theorem t', code, 'a string literal "/-" must not hide the next lines')
        self.assertIn('"/-"', code, 'the literal stays visible so the scan can refuse it')
        self.assertNotIn('-- c', code)
        code, _ = check.lex('def c := "--" ∧ sorry\n')
        self.assertIn('sorry', code)
        code, _ = check.lex('def «/-» := 0\ntheorem t : True := trivial\n')
        self.assertIn('theorem t', code)
        code, _ = check.lex('def d := "a\\"/-" -- x\ndef e := 1\n')
        self.assertIn('def e', code, 'escaped quotes stay inside the literal')
        code, _ = check.lex('def f := "two\nlines"\ndef g := 1\n')
        self.assertEqual(len(code.split('\n')), 4, 'newlines inside literals keep line numbers stable')
        for bad in ('def x := "open\n', 'def «open := 0\n'):
            with self.assertRaises(ValueError):
                check.lex(bad)
        self.assertEqual(check.lex('/- "unterminated inside a comment is fine -/\ndef x := 1\n')[0].split('\n')[1], 'def x := 1')

    def test_depth_zero(self):
        self.assertEqual(check.depth_zero('Decidable (n = 2)'), 'Decidable' + ' ' * 8)
        self.assertEqual(check.depth_zero('a = b'), 'a = b')
        self.assertEqual(check.depth_zero('{x : ℕ // x = 2} → ℕ'), ' ' * 17 + '→ ℕ')
        self.assertEqual(check.depth_zero('∀ n, (n : ℕ) = n'), '∀ n,         = n')

    def test_header_and_result_type(self):
        self.assertEqual(check.result_type(' (x : ℕ) (h : x = 1 := rfl) : Prop '), 'Prop')
        self.assertEqual(check.result_type(' {X : Type} [Fintype X] (F : Set X) :\n    Set (X × X) '), 'Set (X × X)')
        self.assertIsNone(check.result_type(' (x : ℕ) '))
        self.assertEqual(check.header(['def f (x : ℕ) : ℕ := x', 'next'], 0, ' (x : ℕ) : ℕ := x'), ' (x : ℕ) : ℕ ')
        self.assertEqual(check.header(['structure S (d : ℕ) where', '  x : ℕ'], 0, ' (d : ℕ) where'), ' (d : ℕ) ')


class Inventory(Base):
    def test_short_anchors_are_refused(self):
        reg = self.f.register()
        reg['claims'][0]['statements'][0]['anchor'] = 'two and two'
        self.f.save_register(reg)
        self.assertIn('TwoAndTwo', self.assertRefused('shorter than 16'))
        reg = self.f.register()
        reg['claims'][0]['statements'][0]['anchor'] = 'two and two make four'
        reg['claims'][0]['statements'][3]['anchor'] = 'four;'
        self.f.save_register(reg)
        self.assertIn('two_add', self.assertRefused('shorter than 16'))
        reg = self.f.register()
        reg['claims'][0]['statements'][3]['anchor'] = 'two and two make four'
        self.f.save_register(reg)
        self.f.run()

    def test_unregistered_declaration(self):
        self.f.inject('Specifications/RN.lean', f'/-- extra\n\n{labels()} -/\ndef Extra : Prop := True')
        msg = self.assertRefused('unregistered declaration')
        self.assertIn('UniversalLaw.Spec.RN.Extra', msg)

    def test_unregistered_structure(self):
        self.f.inject('Specifications/RN.lean', f'/-- extra\n\n{labels()} -/\nstructure Extra where\n  n : ℕ')
        self.assertRefused('unregistered declaration')

    def test_missing_declaration(self):
        reg = self.f.register()
        reg['claims'][0]['statements'][0]['declaration'] = 'UniversalLaw.Spec.Field.DoesNotExist'
        self.f.save_register(reg)
        self.assertRefused('is not declared')
        reg['claims'][0]['statements'][0]['declaration'] = 'UniversalLaw.Spec.Field.helper'
        self.f.save_register(reg)
        self.assertRefused('is not declared')

    def test_kind_and_module_mismatch(self):
        reg = self.f.register()
        reg['claims'][0]['statements'][0]['kind'] = 'structure'
        self.f.save_register(reg)
        self.assertRefused('registered as structure')
        reg = self.f.register()
        reg['claims'][0]['statements'][0]['module'] = 'Specifications/RN.lean'
        self.f.save_register(reg)
        self.assertRefused('is not declared')

    def test_control_row_helper(self):
        reg = self.f.register()
        reg['claims'].append(check.control_row('Specifications/Field.lean', 'UniversalLaw.Spec.Field.DoesNotExistControl', 'two and two make four'))
        self.f.save_register(reg)
        self.assertRefused('is not declared')
        with self.assertRaises(ValueError):
            check.control_row('Specifications/Field.lean', 'UniversalLaw.Spec.Field.DoesNotExistControl', 'x')

    def test_source_side_negative_controls(self):
        """The four source-side controls that --execute runs, exercised on the fixture without Lean."""
        info = self.f.run()
        reg = self.f.register()
        sources, _ = check.load_sources(self.f.packet, None, reg)
        ctrl = Path(self.f.tmp) / 'controls'
        ctrl.mkdir()
        outcomes, missing, injected = check.source_controls(self.f.packet, reg, info['scans'], sources, check.MODULES, ctrl)
        self.assertEqual(outcomes, {k: 'REFUSED_BY_SOURCE_CHECK' for k in
                                    ('injected_theorem', 'missing_declaration', 'unregistered_declaration', 'proved_label')})
        self.assertEqual(missing, 'UniversalLaw.Spec.Field.DoesNotExistControl')
        self.assertIn('theorem injected : True := trivial', injected)
        self.assertIn('def UnregisteredControl : Prop := True', (ctrl / 'Unregistered.lean.txt').read_text())
        mutations = json.loads((ctrl / 'register_mutations.json').read_text())
        self.assertEqual(mutations['proved_label']['formal_progress'], 'proved')
        self.assertEqual(self.f.read('Specifications/Field.lean'), FIELD, 'the packet module is never modified by a control')
        with self.assertRaises(ValueError) as ctx:
            check.expect_refusal(lambda: None, 'anything', 'escape')
        self.assertIn('escaped', str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            check.expect_refusal(lambda: check.require(False, 'other reason'), 'expected reason', 'wrong')
        self.assertIn('wrong reason', str(ctx.exception))


class Vocabulary(Base):
    def set_statement(self, **changes):
        reg = self.f.register()
        reg['claims'][0]['statements'][0].update(changes)
        self.f.save_register(reg)

    def set_top(self, key, value):
        reg = self.f.register()
        reg[key] = value
        self.f.save_register(reg)

    def set_row(self, index, **changes):
        reg = self.f.register()
        reg['claims'][index].update(changes)
        self.f.save_register(reg)

    def test_local_statement_never_proved(self):
        self.set_statement(formal_progress='proved')
        self.assertRefused('never proved')

    def test_local_statement_never_kernel_checked_or_none(self):
        for value in ('kernel-checked', 'none', 'verified'):
            with self.subTest(value=value):
                self.set_statement(formal_progress=value)
                self.assertRefused('kernel-checked appears only in a run receipt')

    def test_local_statement_never_theorem(self):
        self.set_statement(kind='theorem')
        self.assertRefused('never theorem')

    def test_external_theorem_label(self):
        reg = self.f.register()
        reg['claims'][0]['statements'][3]['formal_progress'] = 'specified'
        self.f.save_register(reg)
        self.assertRefused('outside this package a theorem is proved')

    def test_claim_level_progress(self):
        for value in ('proved', 'kernel-checked', 'verified'):
            with self.subTest(value=value):
                self.set_row(0, formal_progress=value)
                self.assertRefused('claim-level formal_progress')

    def test_specified_requires_def_prop(self):
        reg = self.f.register()
        reg['claims'][0]['statements'] = [reg['claims'][0]['statements'][1], reg['claims'][0]['statements'][3]]
        reg['claims'].append({**reg['claims'][1], 'claim_id': 'rest', 'statements': [
            statement('Specifications/Field.lean', 'UniversalLaw.Spec.Field.TwoAndTwo'),
            statement('Specifications/Field.lean', 'UniversalLaw.Spec.Field.MultiLine')]})
        self.f.save_register(reg)
        self.assertRefused('specified requires a def_prop statement')

    def test_top_level_vocabulary(self):
        for key, value, fragment in (('alignment_status', 'ACCEPTED', 'alignment_status'), ('alignment_status', 'ALIGNED', 'alignment_status'),
                                     ('scientific_effect', 'LOW', 'scientific_effect'), ('scientific_status_authority', True, 'scientific_status_authority'),
                                     ('vocabulary', {'formal_progress': ['none', 'specified', 'proved', 'kernel-checked', 'verified']}, 'vocabulary'),
                                     ('object', 'LEAN-STATEMENT-REGISTER-20261010-v2', 'object must be'), ('schema_version', 2, 'schema_version'),
                                     ('toolchain', {'lean': check.TOOLCHAIN, 'lean_commit': check.LEAN_COMMIT, 'mathlib': 'f' * 40}, 'toolchain'),
                                     ('meaning', '', 'meaning')):
            with self.subTest(key=key, value=value):
                self.set_top(key, value)
                self.assertRefused(fragment)
                self.f = Fixture()
                self.addCleanup(self.f.cleanup)

    def test_unknown_keys(self):
        self.set_top('formalization_status', 'kernel-checked')
        self.assertRefused('top-level keys')
        self.f = Fixture()
        self.addCleanup(self.f.cleanup)
        self.set_row(0, review_verdict='ALIGNED')
        self.assertRefused('fixed schema')
        self.f = Fixture()
        self.addCleanup(self.f.cleanup)
        self.set_statement(reviewed=True)
        self.assertRefused('fixed schema')

    def test_row_fields(self):
        for index, changes, fragment in ((0, {'kind': 'theorem_row'}, 'kind must be one of'), (1, {'claim_id': 'demo-claim'}, 'unique'),
                                         (0, {'not_established': ''}, 'not_established'), (0, {'next_step': ' '}, 'next_step'),
                                         (0, {'layer0': {'source_id': 'nope', 'heading': None}}, 'layer0.source_id'),
                                         (0, {'layer0': {'source_id': 'demo'}}, 'layer0 must be'),
                                         (0, {'kernel_evidence': [{'package': 'formal'}]}, 'kernel_evidence rows'),
                                         (0, {'title': ''}, 'title')):
            with self.subTest(changes=changes):
                self.f = Fixture()
                self.addCleanup(self.f.cleanup)
                self.set_row(index, **changes)
                self.assertRefused(fragment)

    def test_dispositions_are_transcriptions(self):
        self.set_row(0, disposition_transcribed='REVIEWED_SCOPED')
        self.assertRefused('verbatim')
        self.set_row(0, disposition_transcribed='ACCEPT')
        self.assertRefused('LANDING_CLAIMS disposition')
        self.set_row(0, disposition_transcribed='HOLD_WITH_DOMAIN', claim_id='not-a-landing-claim')
        self.assertRefused('landing_claim rows must carry')
        self.f = Fixture()
        self.addCleanup(self.f.cleanup)
        self.set_row(1, disposition_transcribed='HOLD_WITH_DOMAIN')
        self.assertRefused('must be null or the verbatim')
        self.set_row(1, disposition_transcribed=None)
        self.f.run()


class Anchors(Base):
    def test_statement_anchor_verbatim(self):
        reg = self.f.register()
        reg['claims'][0]['statements'][0]['anchor'] = 'two and two make five'
        self.f.save_register(reg)
        self.assertRefused('verbatim substring')

    def test_external_anchor_verbatim(self):
        reg = self.f.register()
        reg['claims'][0]['statements'][3]['anchor'] = 'absent words of sufficient length'
        self.f.save_register(reg)
        self.assertRefused('verbatim substring')
        reg['claims'][0]['statements'][3]['anchor'] = 'two and two make five'
        reg['claims'][0]['layer0']['source_id'] = None
        self.f.save_register(reg)
        self.assertRefused('need a layer0 source')

    def test_docstring_anchor_verbatim(self):
        self.f.mutate('Specifications/RN.lean', 'Anchor: "two and two make four"', 'Anchor: "two and two make  four"')
        self.assertRefused('docstring anchor is not a verbatim')


class Sources(Base):
    def test_local_copy_bytes(self):
        self.f.write('sources/demo/DEMO.md', SOURCE_TEXT + '\n')
        self.assertRefused('local copy differs')

    def test_register_differs_from_sources_json(self):
        def mutate_bytes(reg):
            reg['sources'][0]['bytes'] += 1

        def mutate_blob(reg):
            reg['sources'][0]['blob'] = 'x'
        for mutation, fragment in ((mutate_bytes, 'differs from sources/SOURCES.json'),
                                   (lambda reg: reg['sources'].reverse(), 'exactly the sources/SOURCES.json ids'),
                                   (mutate_blob, 'fixed schema')):
            with self.subTest(fragment=fragment):
                self.f = Fixture()
                self.addCleanup(self.f.cleanup)
                reg = self.f.register()
                mutation(reg)
                self.f.save_register(reg)
                self.assertRefused(fragment)

    def test_sources_json_blob(self):
        self.f.mutate('sources/SOURCES.json', blob(SOURCE_TEXT.encode()), '0' * 40)
        self.assertRefused('local copy differs')

    def test_nonpublic_repository(self):
        self.f.mutate('sources/SOURCES.json', 'd6g8k5htny-coder/Math-', 'someone/private', count=1)
        reg = self.f.register()
        reg['sources'][0]['repository'] = 'someone/private'
        self.f.save_register(reg)
        self.assertRefused('nonpublic')

    def test_local_copy_location(self):
        self.f.mutate('sources/SOURCES.json', '"local_copy": "sources/demo/DEMO.md"', '"local_copy": "DEMO.md"')
        reg = self.f.register()
        reg['sources'][0]['local_copy'] = 'DEMO.md'
        self.f.save_register(reg)
        shutil.move(self.f.path('sources/demo/DEMO.md'), self.f.path('DEMO.md'))
        self.assertRefused('live under sources/')


class Pins(Base):
    def test_lock_name(self):
        self.f.mutate('lake-manifest.json', check.LOCK_NAME, 'other_name')
        self.assertRefused('name must be')

    def test_lock_drift(self):
        self.f.mutate('lake-manifest.json', 'f2effa3d803fda822b1f97b806c47cf2adfbcbc2', 'f2effa3d803fda822b1f97b806c47cf2adfbcbc3')
        self.assertRefused('differs from formal/lake-manifest.json')

    def test_mathlib_rev_pinned_even_if_reference_moves(self):
        other = 'a' * 40
        self.f.mutate('lake-manifest.json', check.MATHLIB_REV, other, count=2)
        self.f.mutate('formal/lake-manifest.json', check.MATHLIB_REV, other, repo=True, count=2)
        self.f.mutate('lakefile.toml', check.MATHLIB_REV, other)
        self.assertRefused('lakefile.toml lacks')

    def test_toolchain(self):
        self.f.write('lean-toolchain', 'leanprover/lean4:v4.34.2\n')
        self.assertRefused('lean-toolchain must read')
        self.f.write('lean-toolchain', check.TOOLCHAIN)
        self.assertRefused('lean-toolchain must read')

    def test_lakefile(self):
        self.f.mutate('lakefile.toml', 'name = "universal_law_specifications"', 'name = "other"')
        self.assertRefused('lakefile.toml lacks')
        self.f = Fixture()
        self.addCleanup(self.f.cleanup)
        self.f.write('lakefile.toml', LAKEFILE + '\n[[require]]\nname = "extra"\ngit = "https://example.invalid/x.git"\nrev = "' + 'b' * 40 + '"\n')
        self.assertRefused('exactly one dependency')

    def test_auto_implicit_must_be_off(self):
        for replacement in ('', 'leanOptions = { autoImplicit = true }\n', 'leanOptions = { relaxedAutoImplicit = false }\n'):
            with self.subTest(replacement=replacement):
                self.f = Fixture()
                self.addCleanup(self.f.cleanup)
                self.f.mutate('lakefile.toml', check.LEAN_OPTIONS_LINE + '\n', replacement)
                self.assertRefused('lakefile.toml lacks')


class Docs(Base):
    def test_placeholders_refused(self):
        for name in ('README.md', 'SCOPE.md'):
            with self.subTest(name=name):
                self.f = Fixture()
                self.addCleanup(self.f.cleanup)
                self.f.write(name, '# bound copy\n\n## Declarations\n\n<<DECLARATION TABLE>>\n')
                self.assertRefused('placeholder')

    def test_placeholder_check_runs_after_the_inventory(self):
        self.f.write('README.md', '<<CLAIM TABLE>>\n')
        self.f.inject('Specifications/RN.lean', f'/-- extra\n\n{labels()} -/\ndef Extra : Prop := True')
        self.assertRefused('unregistered declaration')


class Tree(Base):
    def test_extra_file(self):
        self.f.write('NOTES.md', 'stray\n')
        self.assertRefused('packet tree differs')

    def test_extra_lean_module(self):
        self.f.write('Specifications/Extra.lean', 'import Mathlib\n')
        self.assertRefused('packet tree differs')

    def test_missing_fixed_file(self):
        self.f.path('SCOPE.md').unlink()
        self.assertRefused('packet tree differs')

    def test_gitignore(self):
        self.f.write('.gitignore', '__pycache__/\n')
        self.assertRefused('.gitignore must ignore')

    def test_lake_directory_ignored(self):
        (self.f.packet / '.lake' / 'build').mkdir(parents=True)
        self.f.write('.lake/build/junk.olean', 'x')
        self.f.run()


class Audit(unittest.TestCase):
    def test_allowed(self):
        text = "'A.x' depends on axioms: [propext, Classical.choice, Quot.sound]\n'A.y' does not depend on any axioms\n"
        self.assertEqual(check.audit_axioms(text, ['A.x', 'A.y']), {'A.x': ['Classical.choice', 'Quot.sound', 'propext'], 'A.y': []})

    def test_forbidden(self):
        for bad in ('sorryAx', 'Lean.ofReduceBool', 'hiddenPremise'):
            with self.assertRaises(ValueError) as ctx:
                check.audit_axioms("'A.x' depends on axioms: [propext, " + bad + "]\n", ['A.x'])
            self.assertIn('forbidden transitive axiom', str(ctx.exception))

    def test_missing_duplicate_unrecognized(self):
        with self.assertRaises(ValueError):
            check.audit_axioms("'A.x' does not depend on any axioms\n", ['A.x', 'A.y'])
        with self.assertRaises(ValueError):
            check.audit_axioms("'A.x' does not depend on any axioms\n" * 2, ['A.x'])
        with self.assertRaises(ValueError):
            check.audit_axioms("'A.z' does not depend on any axioms\n", ['A.x'])
        with self.assertRaises(ValueError):
            check.audit_axioms("warning: x\n'A.x' does not depend on any axioms\n", ['A.x'])

    def test_lean_version(self):
        good = f'Lean (version 4.34.1, x86_64-unknown-linux-gnu, commit {check.LEAN_COMMIT}, Release)\n'
        self.assertTrue(check.check_lean_version(good).startswith('Lean (version 4.34.1,'))
        self.assertEqual(check.check_lean_version('info: toolchain not updated; already up-to-date\n' + good), good.strip())
        for bad in (good.replace('4.34.1', '4.34.2'), good.replace(check.LEAN_COMMIT, check.LEAN_COMMIT[:12]),
                    good.replace(check.LEAN_COMMIT, 'f' * 40), good.replace('Release', 'Debug'), '',
                    'warning: something\n' + good, good + 'trailing line\n'):
            with self.assertRaises(ValueError):
                check.check_lean_version(bad)

    def test_strict_json(self):
        self.assertEqual(check.load_json('{"a": 1}'), {'a': 1})
        for raw in ('{"a": 1, "a": 2}', '{"a": 1.5}', '{"a": NaN}', '[Infinity]'):
            with self.assertRaises(ValueError):
                check.load_json(raw)

    def test_signatures(self):
        log = 'A.x (n : ℕ) : Prop\nA.y (long : ℕ) :\n  Prop\nA.z : Type\n'
        self.assertEqual(check.signatures(log, ['A.x', 'A.y']), {'A.x': 'A.x (n : ℕ) : Prop', 'A.y': 'A.y (long : ℕ) :\n  Prop'})
        with self.assertRaises(ValueError):
            check.signatures(log, ['A.w'])
        with self.assertRaises(ValueError):
            check.signatures(log + 'A.x : Prop\n', ['A.x'])
        with self.assertRaises(ValueError):
            check.signatures('A.xy : Prop\n', ['A.x'])

    def test_parse_audit(self):
        log = ('Environment.lean:9:0: info: AUDIT statement A.x\nEnvironment.lean:9:0: info: AUDIT structure A.S\n'
               'Environment.lean:9:0: info: AUDIT generated A.S.mk\nEnvironment.lean:9:0: info: AUDIT auxiliary A.h\n'
               'Environment.lean:9:0: info: AUDIT END 4 0\n')
        self.assertEqual(check.parse_audit(log), {'statement': {'A.x'}, 'structure': {'A.S'}, 'generated': {'A.S.mk'}, 'auxiliary': {'A.h'}})
        refused = log.replace('AUDIT auxiliary A.h', 'AUDIT REFUSED:theorem A.h').replace('END 4 0', 'END 4 1')
        self.assertEqual(check.parse_audit(refused)['REFUSED:theorem'], {'A.h'})
        for bad in (log.replace('END 4 0', 'END 5 0'), log.replace('END 4 0', 'END 4 1'), log.replace('Environment.lean:9:0: info: AUDIT END 4 0\n', ''),
                    log + 'Environment.lean:9:0: info: AUDIT END 4 0\n', log.replace('AUDIT auxiliary A.h', 'AUDIT auxiliary A.x').replace('END 4', 'END 3')):
            with self.assertRaises(ValueError):
                check.parse_audit(bad)

    def test_audit_source(self):
        src = check.audit_source('Specifications', ['Specifications.Field', 'Specifications.RN'])
        self.assertIn('import Specifications\n', src)
        self.assertIn('def auditModules : Array Name := #[`Specifications.Field, `Specifications.RN]', src)
        self.assertIn('run_cmd do', src)
        self.assertIn('"REFUSED:theorem"', src)
        self.assertIn('forallTelescope ci.type fun _ body => pure body.isProp', src)
        self.assertNotIn('isPropFormerType', src, 'the reducing classification called Set-valued definitions statements')
        ctrl = check.audit_source('Specifications', [], check.HIDDEN_CONTROL)
        self.assertIn('def auditModules : Array Name := #[]', ctrl)
        self.assertLess(ctrl.index('theorem smuggledBetweenStrings'), ctrl.index('run_cmd do'))
        self.assertIn('def auditPrefix : Name := `UniversalLaw.Spec.Control', ctrl)


LAKE = check.lake_binary()


FIXTURE_FIELD = '''namespace UniversalLaw.Spec.Field

/-- two and two, parametrized -/
def TwoAndTwo (n : Nat) : Prop := n + n = 4

/-- interface with a Prop field: its projection is a generated theorem -/
structure Iface (d : Nat) where
  x : Nat
  pos : 0 < x

/-- extending interface: `toIface` is a generated projection -/
structure Ext (d : Nat) extends Iface d where
  y : Nat
  hy : y = x

/-- auxiliary definition with a nested proof: `_proof_1` is a generated, range-less theorem -/
def toFin3 (x : Nat) : Fin 3 := ⟨x % 3, Nat.mod_lt _ (by decide)⟩

/-- auxiliary abbreviation -/
abbrev E (d : Nat) : Type := Fin d → Nat

/-- stand-in for Mathlib's `def Set (α : Type u) := α → Prop` (Mathlib/Data/Set/Defs.lean): a type
former whose unfolding is a Prop former -/
def Fam (α : Type) : Type := α → Prop

/-- auxiliary `Fam`-valued definition (the real modules have eleven `Set`-valued ones): its type
`Fam (Nat × Nat)` unfolds to `Nat × Nat → Prop`, so a reducing classification (`isPropFormerType`)
calls it a statement while the scan's `: Prop` rule has it auxiliary -/
def pred : Fam (Nat × Nat) := fun p => p.1 ≤ p.2

end UniversalLaw.Spec.Field
'''
FIXTURE_EXPECTED = {'statement': ['UniversalLaw.Spec.Field.TwoAndTwo'], 'structure': ['UniversalLaw.Spec.Field.Iface', 'UniversalLaw.Spec.Field.Ext'],
                    'auxiliary': ['UniversalLaw.Spec.Field.toFin3', 'UniversalLaw.Spec.Field.E', 'UniversalLaw.Spec.Field.Fam', 'UniversalLaw.Spec.Field.pred']}


@unittest.skipUnless(LAKE, 'lake not available; the Lean-driving helpers are exercised only with the toolchain')
class LeanHelpers(unittest.TestCase):
    """A dependency-free fixture package: real `lake build`, `leanchecker`, `#check`, `#print axioms`,
    the environment audit, its two controls, the Lean-side negative controls, and module copies
    carrying each escape family that the environment audit must refuse. No Mathlib; nothing here
    is about the real modules."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        (self.tmp / 'Specifications').mkdir()
        (self.tmp / 'lakefile.toml').write_text('name = "spec_fixture"\ndefaultTargets = ["Specifications"]\n\n[[lean_lib]]\nname = "Specifications"\n'
                                                + check.LEAN_OPTIONS_LINE + '\n')
        (self.tmp / 'lean-toolchain').write_text(check.TOOLCHAIN + '\n')
        (self.tmp / 'Specifications.lean').write_text('import Specifications.Field\n')
        (self.tmp / 'Specifications/Field.lean').write_text(FIXTURE_FIELD)
        self.out = self.tmp / '.lake' / 'evidence'
        (self.out / 'controls').mkdir(parents=True)
        self.env = check.lean_env(LAKE)

    def build(self, module_text=None, expect_success=True):
        if module_text is not None:
            (self.tmp / 'Specifications/Field.lean').write_text(module_text)
        if (self.tmp / '.lake' / 'build').exists():
            shutil.rmtree(self.tmp / '.lake' / 'build')
        return check.run([LAKE, 'build'], 'build', self.out, self.tmp, self.env, expect_success=expect_success)

    def audit(self, expected=FIXTURE_EXPECTED):
        return check.environment_audit(LAKE, self.tmp, self.env, self.out, ['Specifications.Field'], expected)

    def test_fixture_package(self):
        names = ['UniversalLaw.Spec.Field.TwoAndTwo', 'UniversalLaw.Spec.Field.Iface', 'UniversalLaw.Spec.Field.Ext']
        with contextlib.redirect_stdout(io.StringIO()):
            check.check_lean_version(check.run([LAKE, 'env', 'lean', '--version'], 'version', self.out, self.tmp, self.env))
            log = self.build()
            self.assertNotIn('is a proposition; use', log)
            check.run([LAKE, 'env', 'leanchecker', 'Specifications'], 'leanchecker', self.out, self.tmp, self.env)
            axioms, sigs = check.kernel_audit(LAKE, self.tmp, self.env, self.out, names)
            self.assertEqual(axioms, {n: [] for n in names})
            self.assertEqual(sigs['UniversalLaw.Spec.Field.TwoAndTwo'], 'UniversalLaw.Spec.Field.TwoAndTwo (n : Nat) : Prop')
            self.assertEqual(sigs['UniversalLaw.Spec.Field.Ext'], 'UniversalLaw.Spec.Field.Ext (d : Nat) : Type')
            counts = self.audit()
            self.assertEqual((counts['statement'], counts['structure'], counts['auxiliary']), (1, 2, 4))
            self.assertGreater(counts['generated'], 20, 'projections, recursors, injEq/inj/sizeOf_spec, _proof_1 are tolerated as generated')
            env_log = (self.out / 'environment.log').read_text()
            self.assertIn('AUDIT generated UniversalLaw.Spec.Field.Iface.pos', env_log, 'a Prop-field projection is a generated theorem')
            self.assertIn('AUDIT generated UniversalLaw.Spec.Field.toFin3._proof_1', env_log)
            self.assertIn('AUDIT auxiliary UniversalLaw.Spec.Field.pred', env_log,
                          'a definition whose type merely unfolds to `… → Prop` is auxiliary, as in the scan (statement = syntactically `∀ …, Prop`)')
            self.assertIn('AUDIT auxiliary UniversalLaw.Spec.Field.Fam', env_log)
            self.assertNotIn('REFUSED', env_log)
            with self.assertRaises(ValueError) as ctx:
                self.audit({**FIXTURE_EXPECTED, 'auxiliary': ['UniversalLaw.Spec.Field.E']})
            self.assertIn('unexpected', str(ctx.exception))
            with self.assertRaises(ValueError) as ctx:
                self.audit({**FIXTURE_EXPECTED, 'statement': FIXTURE_EXPECTED['statement'] + ['UniversalLaw.Spec.Field.Ghost']})
            self.assertIn('missing', str(ctx.exception))
            with self.assertRaises(ValueError):
                check.environment_audit(LAKE, self.tmp, self.env, self.out, ['Specifications.Missing'], FIXTURE_EXPECTED)
            self.assertEqual(check.lean_control_type_error(LAKE, self.tmp, self.env, self.out), 'REJECTED_BY_LEAN')
            self.assertEqual(check.lean_control_missing(LAKE, self.tmp, self.env, self.out, 'UniversalLaw.Spec.Field.Missing'), 'REJECTED_BY_LEAN')
            with self.assertRaises(ValueError):
                check.kernel_audit(LAKE, self.tmp, self.env, self.out, names + ['UniversalLaw.Spec.Field.Missing'])
            self.assertIn('REFUSED_BY_ENVIRONMENT_AUDIT', check.lean_control_environment(LAKE, self.tmp, self.env, self.out))
            hidden_log = (self.out / 'control_environment_hidden.log').read_text()
            for verdict in ('REFUSED:theorem UniversalLaw.Spec.Control.smuggledBetweenStrings', 'REFUSED:proof UniversalLaw.Spec.Control.proofAsDef',
                            'REFUSED:axioms:sorryAx UniversalLaw.Spec.Control.usesSorry', 'REFUSED:axiom UniversalLaw.Spec.Control.hidden',
                            'REFUSED:instance UniversalLaw.Spec.Control.controlInstance', 'statement UniversalLaw.Spec.Control.HiddenIndented'):
                self.assertIn('AUDIT ' + verdict, hidden_log)
            injected = self.out / 'Injected.lean'
            injected.write_text('import Specifications\ntheorem injected : True := trivial\n#print axioms injected\n')
            log = check.run([LAKE, 'env', 'lean', str(injected)], 'injected', self.out, self.tmp, self.env)
        self.assertIn("'injected' does not depend on any axioms", log)

    def test_environment_audit_refuses_escapes_lean_accepts(self):
        """Module copies Lean builds without error (the source scan never saw them) must be refused
        by the environment audit, each for its own reason; the build log check catches the proof."""
        cases = {
            'indented Prop definition': (FIXTURE_FIELD.replace('end UniversalLaw', ' def HiddenIndented : Prop := False\nend UniversalLaw'), 'unexpected'),
            'theorem between string literals': (FIXTURE_FIELD.replace('end UniversalLaw', 'def Opener : Prop := "/-" = "x"\ntheorem smuggled : True := trivial\n'
                                                                       'def Closer : Prop := "-/" = "x"\nend UniversalLaw'), 'REFUSED:theorem'),
            'axiom': (FIXTURE_FIELD.replace('end UniversalLaw', 'axiom hidden : False\nend UniversalLaw'), 'REFUSED:axiom'),
            'instance': (FIXTURE_FIELD.replace('end UniversalLaw', 'instance hiddenInst : Inhabited Nat := ⟨0⟩\nend UniversalLaw'), 'REFUSED:instance'),
            'untyped Prop definition': (FIXTURE_FIELD.replace('end UniversalLaw', 'def Untyped := (2 : Nat) + 2 = 4\nend UniversalLaw'), 'unexpected'),
            'Sort 0': (FIXTURE_FIELD.replace('end UniversalLaw', 'def SortZero : Sort 0 := True\nend UniversalLaw'), 'unexpected'),
            'opaque': (FIXTURE_FIELD.replace('end UniversalLaw', 'opaque opq : Nat\nend UniversalLaw'), 'REFUSED:opaque'),
            'inductive': (FIXTURE_FIELD.replace('end UniversalLaw', 'inductive Two | a | b\nend UniversalLaw'), 'REFUSED:inductive'),
        }
        with contextlib.redirect_stdout(io.StringIO()):
            for label, (text, fragment) in cases.items():
                with self.subTest(label=label):
                    self.build(text)
                    with self.assertRaises(ValueError) as ctx:
                        self.audit()
                    message = str(ctx.exception) + (self.out / 'environment.log').read_text()
                    self.assertIn(fragment, message)
            text = FIXTURE_FIELD.replace('end UniversalLaw', 'def proofAsDef : (2 : Nat) + 2 = 4 := rfl\nabbrev proofAsAbbrev : True := trivial\nend UniversalLaw')
            log = self.build(text)
            self.assertIn('is a proposition; use', log, 'the build-log check of execute() refuses the defProp linter warning')
            with self.assertRaises(ValueError):
                self.audit()
            self.assertIn('AUDIT REFUSED:proof UniversalLaw.Spec.Field.proofAsDef', (self.out / 'environment.log').read_text())
            self.assertIn('AUDIT REFUSED:proof UniversalLaw.Spec.Field.proofAsAbbrev', (self.out / 'environment.log').read_text())
            log = self.build(FIXTURE_FIELD.replace('end UniversalLaw', 'def usesSorry : Prop := sorry\nend UniversalLaw'))
            self.assertIn('declaration uses', log)
            with self.assertRaises(ValueError):
                self.audit()
            self.assertIn('AUDIT REFUSED:axioms:sorryAx UniversalLaw.Spec.Field.usesSorry', (self.out / 'environment.log').read_text())

    def test_auto_implicit_is_off(self):
        """With `autoImplicit = false` an unbound name in a header is an error instead of a silent
        implicit parameter (under the default, `def X (f : Fin d → Nat) : Prop` gets `{d : Nat}`)."""
        with contextlib.redirect_stdout(io.StringIO()):
            log = self.build(FIXTURE_FIELD.replace('end UniversalLaw', 'def X (f : Fin d → Nat) : Prop := f = f\nend UniversalLaw'), expect_success=False)
        self.assertIn('Unknown identifier `d`', log)


REAL_REPO = HERE.parents[1]


@unittest.skipUnless((REAL_REPO / check.REFERENCE_LOCK).is_file() and (HERE / 'REGISTER.json').is_file(), 'real tree not present')
class RealPacket(unittest.TestCase):
    """The committed packet's pins and module scans, independent of the register inventory."""

    def test_pins(self):
        revs = check.check_lock(HERE, REAL_REPO)
        self.assertEqual(revs['mathlib'], check.MATHLIB_REV)
        self.assertEqual(len(revs), 9)
        check.check_root(HERE, check.MODULES)
        lock = json.loads((HERE / 'lake-manifest.json').read_text())
        ref = json.loads((REAL_REPO / check.REFERENCE_LOCK).read_text())
        self.assertEqual(lock['packages'], ref['packages'])
        self.assertEqual(lock['name'], check.LOCK_NAME)
        self.assertIn('.lake/', (HERE / '.gitignore').read_text().splitlines())
        self.assertIn(check.LEAN_OPTIONS_LINE, (HERE / 'lakefile.toml').read_text().splitlines())

    def test_register_skeleton_and_sources(self):
        reg = check.load_json((HERE / 'REGISTER.json').read_text(encoding='utf-8'))
        self.assertEqual(set(reg), check.TOP_KEYS)
        self.assertEqual(reg['vocabulary'], {'formal_progress': list(check.FORMAL_PROGRESS)})
        self.assertEqual(reg['alignment_status'], 'PENDING_INDEPENDENT_REVIEW')
        self.assertIs(reg['scientific_status_authority'], False)
        sources, _ = check.load_sources(HERE, None, reg)
        self.assertGreaterEqual(len(sources), 20)
        self.assertIn(check.LANDING_SOURCE, sources)
        landing = check.landing_dispositions(sources)
        self.assertTrue(set(landing.values()) <= set(check.DISPOSITIONS))

    def test_modules_scan(self):
        reg = check.load_json((HERE / 'REGISTER.json').read_text(encoding='utf-8'))
        sources, _ = check.load_sources(HERE, None, reg)
        for m in check.MODULES:
            with self.subTest(module=m):
                if not (HERE / m).is_file():
                    self.skipTest(m + ' not written yet')
                decls = check.scan_module((HERE / m).read_text(encoding='utf-8'), Path(m).stem, sources)
                self.assertTrue(any(d['kind'] == 'def_prop' for d in decls), 'a statement module declares at least one Prop')

    @staticmethod
    def chunks(module):
        """Top-level declaration chunks of a module: {short name: (docstring, declaration text)}, each
        the comment-stripped code from the declaration line to the next column-0 line."""
        text = (HERE / module).read_text(encoding='utf-8')
        code, docs = check.lex(text)
        lines = code.split('\n')
        doc_end = {end: raw for _, end, raw in docs}
        out = {}
        for i, line in enumerate(lines):
            m = re.match(r'^(?:noncomputable )?(?:def|abbrev|structure)\s+(\S+)', line)
            if not m:
                continue
            j = i + 1
            while j < len(lines) and (not lines[j] or lines[j][0].isspace()):
                j += 1
            k = i - 1
            while k >= 0 and k not in doc_end and not lines[k].strip():
                k -= 1
            out[m.group(1)] = (doc_end.get(k), '\n'.join(lines[i:j]).rstrip())
        return out

    def test_shared_declarations_are_byte_identical(self):
        """Field.lean is duplicated into Lifetime.lean (26 declarations, docstrings and bodies) and
        RN.lean (9 declarations; bodies identical, the `pinM`/`pinS` docstrings differ as the RN header
        declares) because the scan forces `import Mathlib` exactly. The copies must not drift."""
        field = self.chunks('Specifications/Field.lean')
        lifetime = self.chunks('Specifications/Lifetime.lean')
        rn = self.chunks('Specifications/RN.lean')
        shared_l = sorted(set(field) & set(lifetime))
        shared_r = sorted(set(field) & set(rn))
        self.assertEqual(len(shared_l), 26, shared_l)
        self.assertEqual(len(shared_r), 9, shared_r)
        for name in shared_l:
            with self.subTest(module='Lifetime', name=name):
                self.assertEqual(field[name], lifetime[name])
        for name in shared_r:
            with self.subTest(module='RN', name=name):
                self.assertEqual(field[name][1], rn[name][1], 'body drifted')
                if name in ('pinM', 'pinS'):
                    self.assertNotEqual(field[name][0], rn[name][0], 'the RN header declares these two docstrings differ')
                else:
                    self.assertEqual(field[name][0], rn[name][0], 'docstring drifted')

    def test_source_check(self):
        reg = check.load_json((HERE / 'REGISTER.json').read_text(encoding='utf-8'))
        if not reg['claims']:
            self.skipTest('REGISTER.json has no claim rows yet; the register compiler fills them, after which this test runs the full source check')
        info = check.source_check(HERE, REAL_REPO, allow_dirty=True)
        self.assertEqual(info['formalization_status'], 'specified')
        self.assertGreater(info['prop_definitions'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
