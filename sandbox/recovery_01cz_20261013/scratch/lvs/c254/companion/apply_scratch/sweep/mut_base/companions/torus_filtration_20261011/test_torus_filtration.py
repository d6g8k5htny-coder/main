#!/usr/bin/env python3
"""Protocol tests and negative controls for the torus-filtration companion driver (stdlib only).

Nothing here executes Lean. The fixture tests exercise check.py's `source` and `finish`
commands against a temporary Git repository whose Lean modules are inventory placeholders,
so every weakening of a check is caught now, before the hosted replay runs the real modules.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

SIDE = Path(__file__).resolve().parent
ROOT = SIDE.parents[1]
INVENTORY = {
    'TorusTopology': {
        'theorem': '''torus_field_latticeVec_formula torus_basis_sum_eq_latticeVec torus_lattice_mem_iff
            torus_proj_surjective torus_proj_continuous torus_proj_eq_iff torus_control_lattice_translate
            torus_proj_isCoveringMap torus_compact torus_pathConnectedSpace torus_field_periodic_formula
            torus_periodic_respects torus_descend_proj torus_descend_continuous torus_path_lift_from
            torus_control_constant_path_endpoint torus_control_no_arbitrary_endpoint torus_higher_path_iff
            torus_strict_higher_path_iff torus_closed_higher_path_iff torus_global_max_iff'''.split(),
        'abbrev': ['torus_E', 'torus_quotient'],
        'def': ['torus_latticeVec', 'torus_periodBasis', 'torus_periodLattice', 'torus_proj', 'torus_isPeriodic',
                'torus_descend'],
        'instance': ['torus_lattice_discrete', 'torus_lattice_addSubgroup_discrete', 'torus_lattice_isZLattice']},
    'ClosedSuperlevel': {
        'theorem': '''superlevel_control_closed_endpoint superlevel_control_strict_endpoint
            superlevel_control_parameter_sign superlevel_control_inclusion_direction superlevel_control_essential
            superlevel_mono superlevel_isClosed superlevel_inclusion_id superlevel_inclusion_comp
            superlevel_mem_at_own_height superlevel_neg_height superlevel_control_strict_thresholds
            superlevel_componentMap_mk superlevel_componentMap_id superlevel_componentMap_comp
            superlevel_componentInclusion_id superlevel_componentInclusion_comp superlevel_component_eq_iff_joinedIn
            superlevel_h0Filtration_map_id superlevel_h0Filtration_map_comp'''.split(),
        'abbrev': [],
        'def': ['superlevel_set', 'superlevel_inclusion', 'superlevel_componentMap', 'superlevel_componentInclusion',
                'superlevel_filtration', 'superlevel_h0Filtration'],
        'instance': []}}
HANDOFF_COMMENTS = ('6111056103', '6111059166', '6111059498', '6111059848', '6111255007', '6111093545', '6111139060')
LEAN_VERSION_LINE = 'Lean (version 4.34.1, x86_64-unknown-linux-gnu, commit 5045d0056413266e57c625dcd7c365b10e377c52, Release)\n'
FALSE_LOG = '/tmp/evidence/{0}.lean:4:2: error: unsolved goals\n⊢ False\n'
SORRY_WARNING = "/tmp/evidence/RejectSorry.lean:2:8: warning: declaration uses `sorry`\n"
INJECTED = "'injected' depends on axioms: [{0}]\n"
LEANCHECKER_MODULES = 'leanchecker TorusTopology ClosedSuperlevel TorusFiltration'
LEANCHECKER_FRESH = 'leanchecker --fresh TorusFiltration'
EMITTED = ('Audit.lean', 'Types.lean', 'Positive.lean', 'RejectSuperlevelSignFlip.lean', 'RejectSuperlevelBelowThreshold.lean',
           'RejectConstantNotPeriodic.lean', 'RejectSorry.lean', 'RejectCustomAxiom.lean', 'RejectNativeDecide.lean')


def checker():
    spec = importlib.util.spec_from_file_location('torus_filtration_checker', SIDE / 'check.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def placeholder_module(module):
    """Inventory-exact Lean text that is never compiled; only the whitelist scan reads it."""
    lines = ['import Mathlib', '', '/-! fixture module; not the companion source -/', '', 'namespace TorusFiltration', '']
    lines += ['theorem %s : True := trivial' % n for n in INVENTORY[module]['theorem']]
    lines += ['abbrev %s : Type := Nat' % n for n in INVENTORY[module]['abbrev']]
    lines += ['noncomputable def %s : Nat := 0' % n for n in INVENTORY[module]['def']]
    lines += ['instance %s : Inhabited Nat := ⟨0⟩' % n for n in INVENTORY[module]['instance']]
    lines += ['', 'end TorusFiltration', '']
    return '\n'.join(lines)


def good_injection_log(c, control):
    _, forbidden, _ = c.INJECTIONS[control]
    return (SORRY_WARNING if control == 'RejectSorry' else '') + INJECTED.format(forbidden)


class Fixture:
    """A committed temporary repository carrying the companion layout and a matching manifest."""

    def __init__(self, c, tmp):
        self.c = c
        self.root = Path(tmp)
        self.side = self.root / c.PACKAGE
        self.side.mkdir(parents=True)
        (self.root / c.WORKFLOW).parent.mkdir(parents=True)
        (self.root / 'formal').mkdir()
        for module in c.MODULES:
            (self.side / (module + '.lean')).write_text(placeholder_module(module))
        (self.side / 'TorusFiltration.lean').write_text(c.ROOT_SOURCE)
        shutil.copy(SIDE / 'check.py', self.side / 'check.py')
        for name in ('replay.sh', 'test_torus_filtration.py', 'README.md', 'SCOPE.md'):
            (self.side / name).write_text('# fixture ' + name + '\n')
        (self.root / c.WORKFLOW).write_text('name: fixture\n')
        (self.root / 'formal/lean-toolchain').write_text('leanprover/lean4:v4.34.1\n')
        (self.root / 'formal/lakefile.toml').write_text('name = "fixture"\n')
        self.revisions = {'mathlib': 'd1' * 20, 'batteries': 'f2' * 20}
        self.write_manifest([{'name': k, 'rev': v} for k, v in self.revisions.items()])
        self.regenerate()
        self.git('init', '-q', '-b', 'main')
        self.commit()

    def git(self, *args):
        return subprocess.run(['git', '-c', 'user.name=fixture', '-c', 'user.email=fixture@example.invalid',
                               '-c', 'commit.gpgsign=false', *args], cwd=self.root, check=True,
                              capture_output=True, text=True).stdout

    def commit(self):
        self.git('add', '-A')
        self.git('commit', '-q', '-m', 'fixture', '--allow-empty')

    def write_manifest(self, packages):
        (self.root / 'formal/lake-manifest.json').write_text(json.dumps({'packages': packages}))

    def regenerate(self):
        meta = self.c.metadata()
        meta['files'] = {p: self.c.identity((self.root / p).read_bytes()) for p in self.c.REQUIRED_FILES
                         if (self.root / p).is_file()}
        (self.side / 'SOURCE_FILES.json').write_text(json.dumps(meta, indent=2, sort_keys=True) + '\n')

    def manifest(self, mutate):
        """Commit a SOURCE_FILES.json with one field rewritten (mutate edits the dict, or returns raw text)."""
        meta = json.loads((self.side / 'SOURCE_FILES.json').read_text())
        raw = mutate(meta)
        text = raw if isinstance(raw, str) else json.dumps(meta, indent=2, sort_keys=True) + '\n'
        (self.side / 'SOURCE_FILES.json').write_text(text)
        self.commit()

    def restore(self):
        self.regenerate()
        self.commit()

    def run(self, *args, env=None):
        environment = {k: v for k, v in os.environ.items() if not k.startswith('GITHUB_')}
        environment.update(env or {})
        return subprocess.run([sys.executable, '-B', '-S', str(self.side / 'check.py'), *args], cwd=self.root,
                              capture_output=True, text=True, env=environment)

    def evidence(self):
        out = self.root / 'evidence'
        out.mkdir()
        source = self.run('source')
        if source.returncode != 0:
            raise AssertionError(source.stderr)
        (out / 'source.json').write_text(source.stdout)
        emitted = self.run('emit', str(out))
        if emitted.returncode != 0:
            raise AssertionError(emitted.stderr)
        (out / 'toolchain.log').write_text(LEAN_VERSION_LINE)
        (out / 'packages.log').write_text(''.join('%s %s\n' % kv for kv in self.revisions.items()))
        for label in self.c.BUILDS:
            (out / (label + '.status')).write_text('0\n')
            (out / (label + '.log')).write_text('')
        (out / 'leanchecker.cmd').write_text(LEANCHECKER_MODULES + '\n')
        meta = self.c.metadata()
        targets = self.c.targets(meta)
        lines = ["'%s' depends on axioms: [propext, Classical.choice, Quot.sound]" % n for n in targets[:-1]]
        lines.append("'%s' does not depend on any axioms" % targets[-1])
        (out / 'axioms.log').write_text('\n'.join(lines) + '\n')
        (out / 'types.log').write_text(''.join('%s (d : ℕ) : True\n' % n for n in self.c.theorem_targets(meta)))
        for control in self.c.FALSE_CONTROLS:
            (out / (control + '.log')).write_text(FALSE_LOG.format(control))
            (out / (control + '.status')).write_text('1\n')
        for control in self.c.INJECTIONS:
            (out / (control + '.log')).write_text(good_injection_log(self.c, control))
            (out / (control + '.status')).write_text('0\n')
        return out


class FixtureCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = checker()

    def assertFails(self, run, message):
        self.assertEqual(run.returncode, 1, run.stdout + run.stderr)
        self.assertIn('TORUS_FILTRATION_FAIL: ' + message, run.stderr)


class InventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = checker()

    def test_inventory_constants(self):
        names = [n for m in INVENTORY.values() for kind in m.values() for n in kind]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(sum(len(m['theorem']) for m in INVENTORY.values()), 41)
        self.assertEqual([len(INVENTORY[m]['theorem']) for m in ('TorusTopology', 'ClosedSuperlevel')], [21, 20])
        self.assertEqual(sum(len(m['def']) for m in INVENTORY.values()), 12)
        self.assertEqual(sum(len(m['abbrev']) + len(m['instance']) for m in INVENTORY.values()), 5)
        self.assertTrue(all(re.fullmatch(self.c.IDENT, n) for n in names))
        self.assertEqual(tuple(INVENTORY), self.c.MODULES)

    def test_source_files_inventory_provenance_and_identities(self):
        meta = self.c.metadata()
        for module in self.c.MODULES:
            self.assertEqual(meta['theorems'][module], INVENTORY[module]['theorem'])
            for kind in ('abbrev', 'def', 'instance'):
                self.assertEqual(meta['definitions'][module][kind], INVENTORY[module][kind])
        self.assertEqual(len(self.c.targets(meta)), 58)
        self.assertEqual(self.c.theorem_targets(meta)[0], 'TorusFiltration.torus_field_latticeVec_formula')
        prov = meta['provenance']
        self.assertEqual(set(HANDOFF_COMMENTS), set(prov['handoff_comments'].values()))
        self.assertEqual(prov['handoff_pr'], 'd6g8k5htny-coder/Math-#419')
        self.assertIn('OpenAI/Codex', prov['source_author'])
        self.assertIn('Anthropic/Claude', prov['compile_fixes_and_companion'])
        modules = {self.c.PACKAGE + '/' + m + '.lean' for m in self.c.MODULES}
        for path in self.c.REQUIRED_FILES:
            target = ROOT / path
            if target.is_file() and (path in meta['files'] or path not in modules):
                self.assertIn(path, meta['files'], 'present companion file without identity: ' + path)
                self.assertEqual(meta['files'][path], self.c.identity(target.read_bytes()), 'stale identity: ' + path)
            elif not target.is_file():
                self.assertNotIn(path, meta['files'], 'identity listed for an absent file: ' + path)

    def test_lean_modules_match_inventory(self):
        self.assertEqual((SIDE / 'TorusFiltration.lean').read_text(), self.c.ROOT_SOURCE)
        for module in self.c.MODULES:
            path = SIDE / (module + '.lean')
            if not path.is_file():
                self.skipTest('module not yet placed by the integrator: ' + module)
            scan = self.c.scan_module(path.read_text(encoding='utf-8'), module)
            self.assertEqual(scan, INVENTORY[module], module)

    def test_no_admission_tokens_in_comment_stripped_code(self):
        self.assertEqual(self.c.strip_comments('a /- sorry\n axiom -/ b -- native_decide\nc').split(), ['a', 'b', 'c'])
        for path in sorted(SIDE.glob('*.lean')):
            code = self.c.strip_comments(path.read_text(encoding='utf-8'))
            for token in self.c.ADMISSION:
                self.assertIsNone(re.search(r'(?<![A-Za-z0-9_])' + token + r'(?![A-Za-z0-9_])', code), path.name + ': ' + token)

    def test_comment_lexer_follows_lean_comment_rules(self):
        c = self.c
        self.assertEqual(c.strip_comments('x /- a /- nested -/ still comment -/ y').split(), ['x', 'y'])
        self.assertEqual(c.strip_comments('/-- doc -- not a closer -/ x').split(), ['x'])
        self.assertEqual(c.strip_comments('a -- /- not an opener\nb').split(), ['a', 'b'])
        self.assertEqual(c.strip_comments('/-! module\n# heading\ntheorem phantom : False := sorry\n-/\nz').split(), ['z'])
        self.assertEqual(c.strip_comments('/-- one -/\ntheorem a : True := trivial').split(), ['theorem', 'a', ':', 'True', ':=', 'trivial'])
        self.assertEqual(len(c.code_lines('a\n/- b\nc -/\nd')), 4)
        with self.assertRaisesRegex(ValueError, 'comment closer outside a comment'):
            c.strip_comments('a -/ b')
        with self.assertRaisesRegex(ValueError, 'unterminated block comment'):
            c.strip_comments('a /- b')

    def test_module_scan_accepts_only_the_whitelisted_shapes(self):
        good = placeholder_module('ClosedSuperlevel')
        inventory = INVENTORY['ClosedSuperlevel']
        prelude = ('import Mathlib\nimport Mathlib.Topology.Basic\n\nnoncomputable section\n\nopen CategoryTheory Topology\n\n'
                   'universe w v u\n\nnamespace TorusFiltration\n/-- Control: a docstring\nwith a column-0 continuation line '
                   'mentioning theorem and sorry -/\nvariable {C : Type u} [Category.{v} C]\n    [Preadditive C] [CategoryWithHomology C]\n')
        for text in (good, good + '-- sorry in a comment is text\n',
                     good.replace('import Mathlib\n', 'import Mathlib\n/-! theorem phantom : False := by native_decide -/\n'),
                     good.replace('namespace TorusFiltration', prelude.split('\n\nnamespace TorusFiltration\n')[0] + '\n\nnamespace TorusFiltration\n'
                                  + prelude.split('\n\nnamespace TorusFiltration\n')[1]),
                     good.replace(':= trivial', ':= by\n  exact trivial\n', 1),
                     good.replace('noncomputable def superlevel_set : Nat := 0', 'def superlevel_set : Nat :=\n  0 -- indented body'),
                     good.replace('theorem superlevel_mono : True := trivial', 'theorem superlevel_mono\n    (h : True) : True := h')):
            with self.subTest(text=text[-80:]):
                self.assertEqual(self.c.scan_module(text, 'ClosedSuperlevel'), inventory)

    def test_module_scan_refuses_admissions_and_unexpected_kinds(self):
        good = placeholder_module('ClosedSuperlevel')

        def inject(line):
            return good.replace('\nend TorusFiltration', '\n' + line + '\nend TorusFiltration')
        unexpected = 'unexpected column-0 line in ClosedSuperlevel:'
        command = 'command token outside a column-0 declaration in ClosedSuperlevel:'
        cases = [
            (good.replace(':= trivial', ':= by sorry', 1), 'admission token in ClosedSuperlevel: sorry'),
            (good.replace(':= trivial', ':= sorryAx _ false', 1), 'admission token in ClosedSuperlevel: sorryAx'),
            (good.replace(':= trivial', ':= by native_decide', 1), 'admission token in ClosedSuperlevel: native_decide'),
            (good.replace(':= trivial', ':= by\n  skipKernelTC', 1), 'admission token in ClosedSuperlevel: skipKernelTC'),
            (good.replace(':= trivial', ':= Lean.ofReduceBool _ _ rfl', 1), 'admission token in ClosedSuperlevel: ofReduceBool'),
            (inject('axiom hidden : False'), 'admission token in ClosedSuperlevel: axiom'),
            (inject('opaque hidden : Nat := 0'), 'admission token in ClosedSuperlevel: opaque'),
            (inject('unsafe def hidden : Nat := 0'), 'admission token in ClosedSuperlevel: unsafe'),
            (inject('partial def hidden : Nat := 0'), 'admission token in ClosedSuperlevel: partial'),
            (inject('set_option autoImplicit true'), 'admission token in ClosedSuperlevel: autoImplicit'),
            (inject('@[simp] theorem extra_hidden : True := trivial'), unexpected),
            (inject('private theorem hidden : True := trivial'), unexpected),
            (inject('protected def hidden : Nat := 0'), unexpected),
            (inject('noncomputable instance hidden : Inhabited Nat := ⟨0⟩'), unexpected),
            (inject('scoped macro_rules | `(tactic| foo) => `(tactic| trivial)'), unexpected),
            (inject('macro_rules | `(tactic| foo) => `(tactic| trivial)'), unexpected),
            (inject('local instance : Inhabited Nat := ⟨0⟩'), unexpected),
            (inject('attribute [simp] superlevel_set'), unexpected),
            (inject('set_option maxHeartbeats 0'), unexpected),
            (inject('set_option maxHeartbeats 0 in'), unexpected),
            (inject('section\n  theorem indented_hidden : True := trivial\nend'), unexpected),
            (inject('  theorem indented_hidden : True := trivial'), command),
            (inject('theorem a_hidden : True := trivial theorem b_hidden : True := trivial'), command),
            (inject('theorem a_hidden : True := by\n  open Classical in exact trivial'), command),
            (inject('variable (h : False) in theorem hidden : True := trivial'), command),
            (inject('universe z in'), command),
            (inject('open Classical in'), unexpected),
            (inject('macro foo : tactic => `(tactic| trivial)'), unexpected),
            (inject('elab foo : term => pure (mkConst ``True)'), unexpected),
            (inject('syntax foo : tactic'), unexpected),
            (inject('notation x ⊕ y => x'), unexpected),
            (inject('lemma extra : True := trivial'), unexpected),
            (inject('example : True := trivial'), unexpected),
            (inject('structure Hidden where\n  x : Nat'), unexpected),
            (inject('inductive Hidden | a'), unexpected),
            (inject('class Hidden (α : Type) where\n  k : α'), unexpected),
            (inject('instance : Inhabited Nat := ⟨0⟩'), unexpected),
            (inject('mutual\ntheorem a_hidden : True := trivial\nend'), unexpected),
            (inject('deriving instance Repr for Nat'), unexpected),
            (inject('#eval 1'), unexpected),
            (inject('#print axioms superlevel_set'), unexpected),
            (inject('export Nat (succ)'), unexpected),
            (inject('include h'), unexpected),
            (inject('run_cmd pure ()'), unexpected),
            (inject('initialize hidden : Nat ← pure 0'), unexpected),
            (inject('import Hidden'), unexpected),
            (inject('import Mathlib'), 'import inside the namespace: ClosedSuperlevel:'),
            (inject("theorem hidden' : True := trivial"), unexpected),
            (inject('theorem hidden.name : True := trivial'), unexpected),
            (inject('theorem'), unexpected),
            (inject('end'), unexpected),
            (inject('noncomputable section extra'), unexpected),
            (inject('def s : String := "x"'), 'string literal in code: ClosedSuperlevel:'),
            (inject('def s : String := "/-"\ndef t : String := "-/"'), 'string literal in code: ClosedSuperlevel:'),
            (inject('theorem a_hidden : True := trivial -- fine\n  @[simp] theorem b_hidden : True := trivial'), command),
            (inject('theorem a_hidden : True := by\n  simp #[1]'), 'attribute or #-command in ClosedSuperlevel:'),
            (good.replace('namespace TorusFiltration', 'namespace ResearchFormalCoreR1'), 'handoff namespace left in place: ClosedSuperlevel'),
            (good.replace('namespace TorusFiltration', ''), 'declaration outside namespace TorusFiltration: ClosedSuperlevel:'),
            (good.replace('namespace TorusFiltration\n', 'namespace TorusFiltration\nnamespace TorusFiltration\n'), 'namespace declaration: ClosedSuperlevel:'),
            (good.replace('end TorusFiltration', ''), 'namespace TorusFiltration not opened and closed exactly once: ClosedSuperlevel'),
            (good + 'theorem after_end : True := trivial\n', 'code after end TorusFiltration: ClosedSuperlevel:'),
            (good + '-/\n', 'comment closer outside a comment'),
            (good + '/- open forever\n', 'unterminated block comment'),
            (good.replace('import Mathlib', 'theorem early : True := trivial\nimport Mathlib'), 'declaration outside namespace TorusFiltration'),
        ]
        self.assertEqual(self.c.scan_module(good, 'ClosedSuperlevel'), INVENTORY['ClosedSuperlevel'])
        for text, message in cases:
            with self.subTest(message=message, text=text[-70:]), self.assertRaisesRegex(ValueError, re.escape(message)):
                self.c.scan_module(text, 'ClosedSuperlevel')
        # Every command keyword the scanner knows is refused somewhere other than the start of a whitelisted line.
        for keyword in self.c.COMMANDS:
            with self.subTest(keyword=keyword), self.assertRaises(ValueError):
                self.c.scan_module(inject('theorem k_hidden : True := trivial ' + keyword), 'ClosedSuperlevel')


class ParserControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = checker()

    def test_negative_false_controls_accept_only_the_intended_diagnostic(self):
        for control in self.c.FALSE_CONTROLS:
            good = FALSE_LOG.format(control)
            self.assertEqual(self.c.negative(good, control, 1), 'REJECTED_BY_LEAN')
            for bad, status in [(good, 0), (good, 2), (good, True), (good, '1'), (good.replace('False', 'True'), 1),
                                (good.replace(control, 'Other'), 1), (good + 'error: unknown identifier\n', 1),
                                ('Traceback: bad runtime\n', 1), ('error: unsolved goals\n', 1), ('', 1),
                                ('warning: declaration uses `sorry`\n' + good, 1)]:
                with self.subTest(control=control, bad=bad, status=status), self.assertRaises(ValueError):
                    self.c.negative(bad, control, status)
        with self.assertRaises(ValueError):
            self.c.negative(FALSE_LOG.format('Unknown'), 'Unknown', 1)

    def test_negative_injection_controls_require_the_audit_to_refuse(self):
        for control, (_, forbidden, _) in self.c.INJECTIONS.items():
            good = good_injection_log(self.c, control)
            self.assertEqual(self.c.negative(good, control, 0), 'REJECTED_BY_AXIOM_AUDIT')
            escaped = good.replace(forbidden, 'propext')
            for bad, status in [(good, 1), (good, 2), (escaped, 0), (good.replace('injected', 'other'), 0),
                                (good + 'extra\n', 0), ('', 0), (good.replace('[' + forbidden + ']', '[]'), 0),
                                (good.replace("'injected' depends", "'injected' does not depend"), 0)]:
                with self.subTest(control=control, bad=bad, status=status), self.assertRaises(ValueError):
                    self.c.negative(bad, control, status)
        sorry_log = good_injection_log(self.c, 'RejectSorry')
        with self.assertRaises(ValueError):
            self.c.negative(sorry_log.replace('`sorry`', "'sorry'"), 'RejectSorry', 0)
        with self.assertRaises(ValueError):
            self.c.negative(INJECTED.format('sorryAx'), 'RejectSorry', 0)
        with self.assertRaises(ValueError):
            self.c.negative(SORRY_WARNING + INJECTED.format('hiddenPremise'), 'RejectCustomAxiom', 0)

    def test_audit_axioms_rules(self):
        names = ['TorusFiltration.a', 'TorusFiltration.b', 'TorusFiltration.c']
        good = ("'TorusFiltration.a' depends on axioms: [propext, Classical.choice, Quot.sound]\n"
                "'TorusFiltration.b' depends on axioms: [propext]\n'TorusFiltration.c' does not depend on any axioms\n")
        self.assertEqual(self.c.audit_axioms(good, names),
                         {'TorusFiltration.a': ['Classical.choice', 'Quot.sound', 'propext'],
                          'TorusFiltration.b': ['propext'], 'TorusFiltration.c': []})
        for bad, message in ((good.replace('[propext]', '[propext, sorryAx]'), 'forbidden transitive axiom'),
                             (good.replace('[propext]', '[Lean.ofReduceBool]'), 'forbidden transitive axiom'),
                             (good.replace('TorusFiltration.b', 'TorusFiltration.unknown'), 'unexpected or duplicate axiom target'),
                             (good.replace("'TorusFiltration.b' depends on axioms: [propext]\n", ''), 'missing target axiom report'),
                             (good + "'TorusFiltration.a' depends on axioms: [propext]\n", 'unexpected or duplicate axiom target'),
                             ('warning: something\n' + good, 'unrecognized audit output'),
                             (good.replace('[propext]', '[prop ext]'), 'malformed axiom name'),
                             (good.replace('[propext]', '[propext,]'), 'malformed axiom name'),
                             (good.replace('[propext]', '[hiddenPremise]'), 'forbidden transitive axiom')):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, message):
                self.c.audit_axioms(bad, names)
        for targets in ([], ['TorusFiltration.a', 'TorusFiltration.a'], 'TorusFiltration.a'):
            with self.subTest(targets=targets), self.assertRaisesRegex(ValueError, 'empty or duplicate target list'):
                self.c.audit_axioms(good, targets)
        duplicated = {'theorems': {'TorusTopology': ['a'], 'ClosedSuperlevel': ['a']},
                      'definitions': {m: {'abbrev': [], 'def': [], 'instance': []} for m in self.c.MODULES}}
        with self.assertRaisesRegex(ValueError, 'duplicate or empty target list'):
            self.c.targets(duplicated)
        duplicated['theorems'] = {m: [] for m in self.c.MODULES}
        with self.assertRaisesRegex(ValueError, 'duplicate or empty target list'):
            self.c.targets(duplicated)

    def test_types_rules(self):
        names = ['TorusFiltration.a', 'TorusFiltration.b']
        good = 'TorusFiltration.a : True\nTorusFiltration.b (d : ℕ) :\n  ∀ x, x = x\n'
        self.assertEqual(self.c.types(good, names), names)
        for bad, message in (('TorusFiltration.a : True\n', 'type inventory mismatch'),
                             (good + 'TorusFiltration.c : True\n', 'type inventory mismatch'),
                             ('TorusFiltration.a : True\nTorusFiltration.a : True\n', 'type inventory mismatch'),
                             ('TorusFiltration.b : True\nTorusFiltration.a : True\n', 'type inventory mismatch'),
                             ('error: crash\n' + good, 'type diagnostic'), (good + 'error: crash\n', 'type diagnostic'),
                             ('warning: lint\n' + good, 'type diagnostic'), (good + 'warning: lint\n', 'type diagnostic'),
                             ('preamble\n' + good, 'unrecognized type preamble')):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, message):
                self.c.types(bad, names)

    def test_lean_version_and_packages(self):
        self.assertEqual(self.c.lean_version(LEAN_VERSION_LINE), LEAN_VERSION_LINE.strip())
        for bad in (LEAN_VERSION_LINE.replace('4.34.1', '4.34.0'), LEAN_VERSION_LINE.replace('5045d0056', '5045d005'),
                    LEAN_VERSION_LINE.replace('Release', 'Debug'), LEAN_VERSION_LINE + 'extra\n', ''):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.c.lean_version(bad)
        expected = {'mathlib': 'd1' * 20, 'batteries': 'f2' * 20}
        good = 'batteries %s\nmathlib %s\n' % ('f2' * 20, 'd1' * 20)
        self.assertEqual(self.c.packages(good, expected), expected)
        for bad in ('mathlib %s\n' % ('d1' * 20), good + 'aesop %s\n' % ('a3' * 20), good.replace('d1' * 20, 'e1' * 20),
                    good + 'mathlib %s\n' % ('d1' * 20), good.replace('\nmathlib', '\n mathlib'), 'mathlib short\n'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.c.packages(bad, expected)

    def test_leanchecker_mode_accepts_exactly_the_two_recorded_invocations(self):
        self.assertEqual(self.c.leanchecker_mode(LEANCHECKER_MODULES + '\n'), 'modules')
        self.assertEqual(self.c.leanchecker_mode(LEANCHECKER_FRESH + '\n'), 'fresh')
        self.assertEqual(self.c.LEANCHECKER, {LEANCHECKER_MODULES: 'modules', LEANCHECKER_FRESH: 'fresh'})
        for bad in ('', 'leanchecker TorusFiltration', 'leanchecker --fresh TorusTopology', 'leanchecker TorusTopology ClosedSuperlevel',
                    LEANCHECKER_MODULES + ' Extra', 'leanchecker --verbose ' + LEANCHECKER_MODULES[12:],
                    'lean ' + LEANCHECKER_MODULES[12:], LEANCHECKER_MODULES + '\n' + LEANCHECKER_FRESH):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, 'unrecognized leanchecker invocation'):
                self.c.leanchecker_mode(bad)

    def test_emit_writes_exactly_the_audit_types_positive_and_controls(self):
        meta = self.c.metadata()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.c.emit(out)
            self.assertEqual({p.name for p in out.iterdir()}, set(EMITTED))
            self.assertEqual({p.name for p in out.iterdir()},
                             {'Audit.lean', 'Types.lean', 'Positive.lean'} | {n + '.lean' for n in self.c.CONTROLS})
            audit = (out / 'Audit.lean').read_text().splitlines()
            self.assertEqual(audit[0], 'import TorusFiltration')
            self.assertEqual(audit[1:], ['#print axioms ' + n for n in self.c.targets(meta)])
            self.assertEqual(len(audit[1:]), 58)
            types = (out / 'Types.lean').read_text().splitlines()
            self.assertEqual(types[:2], ['import TorusFiltration', 'set_option pp.explicit true'])
            self.assertEqual(types[2:], ['#check ' + n for n in self.c.theorem_targets(meta)])
            positive = (out / 'Positive.lean').read_text()
            self.assertTrue(positive.startswith('import TorusFiltration\n'))
            self.assertTrue(3 <= len(re.findall(r'^example', positive, re.M)) <= 6)
            for token in ('sorry', 'admit', 'axiom', 'native_decide'):
                self.assertIsNone(re.search(r'\b' + token + r'\b', positive))
            for control in self.c.CONTROLS:
                text = (out / (control + '.lean')).read_text()
                self.assertTrue(text.startswith('import TorusFiltration\n'), control)
                self.assertEqual('sorry' in text, control == 'RejectSorry')
                self.assertEqual('axiom hiddenPremise' in text, control == 'RejectCustomAxiom')
                self.assertEqual('native_decide' in text, control == 'RejectNativeDecide')
                if control in self.c.FALSE_CONTROLS:
                    self.assertIn('example : ' + self.c.FALSE_CONTROLS[control][0] + ' := by', text)
                else:
                    self.assertIn('#print axioms injected', text)
        self.assertEqual(set(self.c.FALSE_CONTROLS) | set(self.c.INJECTIONS), set(self.c.CONTROLS))
        self.assertGreaterEqual(len(self.c.FALSE_CONTROLS), 1)
        self.assertEqual({v[1] for v in self.c.INJECTIONS.values()},
                         {'sorryAx', 'hiddenPremise', 'injected._native.native_decide.ax_1_1'})
        self.assertTrue(all(re.fullmatch(self.c.NAME, v[1]) and v[1] not in self.c.ALLOWED for v in self.c.INJECTIONS.values()))


class WorkflowAndReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = checker()
        cls.workflow = (ROOT / cls.c.WORKFLOW).read_text()
        cls.replay = (SIDE / 'replay.sh').read_text()
        cls.d2 = (ROOT / '.github/workflows/d2-moment-bridge.yml').read_text()

    def paths(self, event):
        marker = '  ' + event + ':\n'
        self.assertEqual(self.workflow.count(marker), 1, event)
        rest = self.workflow.split(marker, 1)[1]
        ends = [i for i in (rest.find('\n  push:\n'), rest.find('\n  workflow_dispatch:\n')) if i >= 0]
        block = rest[:min(ends)]
        self.assertEqual(block.count('    paths:\n'), 1, event)
        return re.findall(r"^      - '([^']+)'$", block, re.M)

    def test_path_triggers_exclude_the_core_tree(self):
        expected = ['companions/torus_filtration_20261011/**', '.github/workflows/torus-filtration.yml',
                    'formal/lakefile.toml', 'formal/lake-manifest.json', 'formal/lean-toolchain']
        self.assertEqual(self.paths('pull_request'), expected)
        self.assertEqual(self.paths('push'), expected)
        self.assertNotIn("'formal/**'", self.workflow)
        self.assertIn('    branches: [main]\n', self.workflow)
        self.assertIn('  workflow_dispatch:\n', self.workflow)
        self.assertIn('permissions:\n  contents: read\n', self.workflow)

    def test_action_pins_match_the_d2_companion_workflow(self):
        for action in ('actions/checkout@', 'leanprover/lean-action@', 'actions/upload-artifact@'):
            mine = re.findall(re.escape(action) + r'([0-9a-f]{40})', self.workflow)
            theirs = re.findall(re.escape(action) + r'([0-9a-f]{40})', self.d2)
            self.assertEqual(len(mine), 1, action)
            self.assertEqual(mine, theirs, action)
        for required in ('lake-package-directory: formal', "use-mathlib-cache: 'true'", "build: 'false'",
                         'persist-credentials: false', 'run: bash companions/torus_filtration_20261011/replay.sh',
                         'name: torus-filtration-${{ github.run_id }}-${{ github.run_attempt }}',
                         'path: .lake/torus-filtration-evidence/', 'retention-days: 30', 'if-no-files-found: error',
                         'if: always()', 'LEAN_NUM_THREADS: 2'):
            self.assertIn(required, self.workflow, required)
        self.assertRegex(self.workflow, r'\n    timeout-minutes: [0-9]+\n', 'job timeout must be explicit')
        self.assertNotIn('continue-on-error', self.workflow)
        self.assertNotIn('formal/.lake/formal-evidence', self.workflow)
        # The hosted job must run the modules-mode leanchecker: no --fresh and no fresh switch in the workflow.
        self.assertNotIn('--fresh', self.workflow)
        self.assertNotIn('TORUS_LEANCHECKER_FRESH', self.workflow)

    def test_replay_contract(self):
        for required in ('set -euo pipefail', 'test ! -e "$OUT"', 'OUT="$ROOT/.lake/torus-filtration-evidence"',
                         'python3 -B -S "$SIDE/check.py" source > "$OUT/source.json"',
                         'python3 -B -S "$SIDE/test_torus_filtration.py"', 'python3 -B -O -S "$SIDE/test_torus_filtration.py"',
                         'python3 -B -S "$SIDE/check.py" emit "$OUT"', 'lake env bash', 'lean --version > "$TORUS_OUT/toolchain.log"',
                         'rev-parse HEAD)" >> "$TORUS_OUT/packages.log"',
                         'printf \'%s\\n\' "$LEANCHECKER_COMMAND" > "$TORUS_OUT/leanchecker.cmd"',
                         'run_lean leanchecker $LEANCHECKER_COMMAND',
                         'run_lean positive lean -DwarningAsError=true', 'run_lean axioms lean -DwarningAsError=true',
                         'run_lean types lean -DwarningAsError=true'):
            self.assertIn(required, self.replay, required)
        for module in ('TorusTopology', 'ClosedSuperlevel', 'TorusFiltration'):
            self.assertIn('run_lean build-%s lean -DwarningAsError=true --root="$TORUS_SIDE" -o "$TORUS_SIDE/%s.olean" "$TORUS_SIDE/%s.lean"'
                          % (module, module, module), self.replay)
        commands = '\n'.join(line for line in self.replay.splitlines() if not line.lstrip().startswith('#'))
        for forbidden in ('gate.py', 'formal/tests', 'lake build', 'lake update', 'lake exe cache', 'formal-evidence',
                          'ResearchFormalCoreR1'):
            self.assertNotIn(forbidden, commands, forbidden)
        # leanchecker: the modules-mode command is the default; --fresh exists only behind the operator switch.
        switch = re.search(r'^if \[ "\$\{TORUS_LEANCHECKER_FRESH:-0\}" = 1 \]; then\n  LEANCHECKER_COMMAND=\'([^\']+)\'\nelse\n'
                           r'  LEANCHECKER_COMMAND=\'([^\']+)\'\nfi\n', commands, re.M)
        self.assertIsNotNone(switch, 'leanchecker mode switch')
        self.assertEqual(switch.group(1), LEANCHECKER_FRESH)
        self.assertEqual(switch.group(2), LEANCHECKER_MODULES)
        self.assertEqual(self.c.LEANCHECKER[switch.group(2)], 'modules')
        self.assertEqual(commands.count('--fresh'), 1)
        self.assertEqual(commands.count('leanchecker.cmd'), 1)
        loops = re.findall(r'^for control in ([^;]+); do\n  code=0\n  lean ([^\n]*)', self.replay, re.M)
        self.assertEqual(len(loops), 2)
        self.assertEqual(loops[0][0].split(), list(self.c.FALSE_CONTROLS))
        self.assertIn('-DwarningAsError=true', loops[0][1])
        self.assertEqual(loops[1][0].split(), list(self.c.INJECTIONS))
        self.assertNotIn('warningAsError', loops[1][1])
        finish = [line for line in self.replay.splitlines() if 'check.py" finish' in line]
        self.assertEqual(finish, ['python3 -B -S "$SIDE/check.py" finish "$OUT"'])
        self.assertTrue(self.replay.strip().splitlines()[-1].startswith("echo 'PASS:"))

    def test_documents_keep_the_status_vocabulary(self):
        for name in ('README.md', 'SCOPE.md'):
            text = (SIDE / name).read_text(encoding='utf-8')
            for required in ('Scientific effect', 'NONE', 'credit', 'PENDING_INDEPENDENT_REVIEW', 'OpenAI/Codex',
                             'Anthropic/Claude', '8a5c7a24-784b-4aaa-b951-cd9f37b0dc64', 'efa80d28', '1fed3eda',
                             'ResearchFormalCoreR1.<n>', 'TorusFiltration.<n>', 'PairLawInterface.p',
                             'critical-level attainment', 'once-counting', LEANCHECKER_MODULES):
                self.assertIn(required, text, name + ' lacks ' + required)
            self.assertNotIn('`leanchecker --fresh` re-checked', text, name + ' describes the withdrawn --fresh step as run')
            self.assertNotIn('runs `leanchecker --fresh TorusFiltration`', text)
            prose = re.sub(r'`[^`\n]*`', '', text)
            for forbidden in ('verified', 'accepted', 'kernel-checked'):
                hits = [m.start() for m in re.finditer(r'(?i)\b' + forbidden + r'\b', prose)]
                if forbidden == 'kernel-checked':
                    # Allowed only when naming the receipt field it belongs to.
                    hits = [h for h in hits if 'receipt' not in prose[max(0, h - 160):h + 160]]
                self.assertEqual(hits, [], name + ' uses the status word ' + forbidden + ' outside code spans')
        readme = (SIDE / 'README.md').read_text(encoding='utf-8')
        self.assertIn('## Compile fixes at the pin', readme)
        self.assertNotIn('formal/**', readme.replace('`formal/`', ''))
        # The control table quotes each emitted false statement verbatim, and the .olean hazard is stated.
        for statement, _ in self.c.FALSE_CONTROLS.values():
            self.assertIn('`' + statement + '`', readme, 'control statement not quoted verbatim: ' + statement)
        for required in ('.olean', 'never be committed', 'NOT_RUN', 'TORUS_LEANCHECKER_FRESH', 'leanchecker.cmd'):
            self.assertIn(required, readme, 'README lacks ' + required)


class SourceFixture(FixtureCase):
    def test_clean_fixture_passes_and_records_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(self.c, tmp)
            run = f.run('source')
            self.assertEqual(run.returncode, 0, run.stderr)
            result = json.loads(run.stdout)
            self.assertEqual(result['checked_commit'], f.git('rev-parse', 'HEAD').strip())
            self.assertEqual(set(result['sources']), set(self.c.REQUIRED_FILES) | {self.c.PACKAGE + '/SOURCE_FILES.json'})
            self.assertEqual(len(result['declarations']), 58)
            self.assertEqual(result['environment'], {'lean_toolchain': 'leanprover/lean4:v4.34.1',
                                                     'dependency_revisions': f.revisions})
            self.assertEqual(result['run_id'], 'local')
            hosted = {'GITHUB_ACTIONS': 'true', 'GITHUB_SHA': result['checked_commit'], 'GITHUB_REPOSITORY': self.c.REPO,
                      'GITHUB_RUN_ID': '77', 'GITHUB_RUN_ATTEMPT': '2'}
            run = f.run('source', env=hosted)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads(run.stdout)['run_id'], '77')
            self.assertFails(f.run('source', env={**hosted, 'GITHUB_SHA': '0' * 40}), 'host/source mismatch')
            self.assertFails(f.run('source', env={**hosted, 'GITHUB_REPOSITORY': 'someone-else/Math-'}), 'host/source mismatch')
            self.assertFails(f.run('source', env={**hosted, 'GITHUB_RUN_ID': ''}), 'run identity absent')
            self.assertFails(f.run('source', env={**hosted, 'GITHUB_RUN_ATTEMPT': ''}), 'run identity absent')
            (f.side / 'stray.olean').write_bytes(b'untracked build artefact')
            self.assertEqual(f.run('source').returncode, 0, 'untracked artefacts are not source')

    def test_membership_and_byte_mutations_are_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(self.c, tmp)
            (f.side / 'README.md').write_text('# edited without a commit\n')
            self.assertFails(f.run('source'), 'dirty execution/source bytes: ' + self.c.PACKAGE + '/README.md')
            f.git('checkout', '--', '.')
            (f.root / 'formal/lean-toolchain').write_text('leanprover/lean4:v4.34.0\n')
            self.assertFails(f.run('source'), 'uncommitted modification in bound paths')
            f.git('checkout', '--', '.')
            (f.side / 'EXTRA.md').write_text('tracked but not in the manifest\n')
            f.commit()
            self.assertFails(f.run('source'), 'tracked companion membership differs')
            f.git('rm', '-q', '--', str(f.side / 'EXTRA.md'))
            f.commit()
            self.assertEqual(f.run('source').returncode, 0)
            (f.side / 'TorusTopology.olean').write_bytes(b'committed build artefact')
            f.commit()
            self.assertFails(f.run('source'), "tracked companion membership differs: ['" + self.c.PACKAGE + "/TorusTopology.olean']")
            f.git('rm', '-q', '--', str(f.side / 'TorusTopology.olean'))
            f.commit()
            (f.side / 'SCOPE.md').write_text('# changed and committed, manifest stale\n')
            f.commit()
            self.assertFails(f.run('source'), 'manifest source mismatch: ' + self.c.PACKAGE + '/SCOPE.md')
            f.restore()
            self.assertEqual(f.run('source').returncode, 0)
            (f.side / 'README.md').unlink()
            os.symlink('SCOPE.md', f.side / 'README.md')
            f.restore()
            self.assertFails(f.run('source'), 'symlink source: ' + self.c.PACKAGE + '/README.md')

    def test_module_mutations_are_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(self.c, tmp)
            module = f.side / 'ClosedSuperlevel.lean'
            original = module.read_text()
            cases = [(original.replace('theorem superlevel_mono ', 'theorem superlevel_monotone '),
                      'theorem inventory differs from source: ClosedSuperlevel'),
                     (original.replace('def superlevel_set ', 'def superlevel_family '),
                      'def inventory differs from source: ClosedSuperlevel'),
                     (original.replace(':= trivial', ':= by sorry', 1), 'admission token in ClosedSuperlevel: sorry'),
                     (original + 'lemma injected : True := trivial\n', 'code after end TorusFiltration: ClosedSuperlevel:'),
                     (original.replace('\nend TorusFiltration', '\nlemma injected : True := trivial\nend TorusFiltration'),
                      'unexpected column-0 line in ClosedSuperlevel:'),
                     (original.replace('\nend TorusFiltration', '\nnoncomputable instance hidden : Inhabited Nat := ⟨0⟩\nend TorusFiltration'),
                      'unexpected column-0 line in ClosedSuperlevel:'),
                     (original.replace('\nend TorusFiltration', '\n  theorem hidden : True := trivial\nend TorusFiltration'),
                      'command token outside a column-0 declaration in ClosedSuperlevel:'),
                     (original.replace('namespace TorusFiltration', 'namespace ResearchFormalCoreR1'),
                      'handoff namespace left in place: ClosedSuperlevel')]
            for text, message in cases:
                with self.subTest(message=message):
                    module.write_text(text)
                    f.restore()
                    self.assertFails(f.run('source'), message)
            module.write_text(original + '-- sorry inside a comment is not an admission\n')
            f.restore()
            self.assertEqual(f.run('source').returncode, 0)
            (f.side / 'TorusFiltration.lean').write_text('import TorusTopology\n')
            f.restore()
            self.assertFails(f.run('source'), 'root module must import exactly the two modules')
            (f.side / 'TorusFiltration.lean').write_text(self.c.ROOT_SOURCE)
            f.regenerate()
            meta = json.loads((f.side / 'SOURCE_FILES.json').read_text())
            del meta['files'][self.c.PACKAGE + '/TorusTopology.lean']
            (f.side / 'SOURCE_FILES.json').write_text(json.dumps(meta, indent=2, sort_keys=True) + '\n')
            f.commit()
            self.assertFails(f.run('source'), 'files inventory must list every required companion file')

    def test_environment_pins_are_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(self.c, tmp)
            (f.root / 'formal/lean-toolchain').write_text('leanprover/lean4:v4.34.2\n')
            f.commit()
            self.assertFails(f.run('source'), "wrong Lean toolchain: 'leanprover/lean4:v4.34.2'")
            (f.root / 'formal/lean-toolchain').write_text('leanprover/lean4:v4.34.1\n')
            f.write_manifest([{'name': 'mathlib', 'rev': 'main'}, {'name': 'batteries', 'rev': 'f2' * 20}])
            f.commit()
            self.assertFails(f.run('source'), 'unpinned dependency')
            f.write_manifest([{'name': 'mathlib', 'rev': 'd1' * 20}, {'name': 'mathlib', 'rev': 'd1' * 20}])
            f.commit()
            self.assertFails(f.run('source'), 'duplicate or empty dependency list')
            f.write_manifest([])
            f.commit()
            self.assertFails(f.run('source'), 'duplicate or empty dependency list')
            f.write_manifest([{'name': k, 'rev': v} for k, v in f.revisions.items()])
            f.commit()
            self.assertEqual(f.run('source').returncode, 0)


class MetadataFixture(FixtureCase):
    """Every honesty label and schema rule of SOURCE_FILES.json is enforced through the CLI, one field at a time."""

    def test_each_label_and_schema_rule_withholds_source(self):
        c = self.c
        workflow = c.WORKFLOW
        original = c.ORIGINAL_SHA256

        def setter(path, value):
            def mutate(m):
                node = m
                for key in path[:-1]:
                    node = node[key]
                node[path[-1]] = value
            return mutate

        def deleter(path):
            def mutate(m):
                node = m
                for key in path[:-1]:
                    node = node[key]
                del node[path[-1]]
            return mutate
        cases = [
            (setter(('scientific_effect',), 'LOW'), 'scientific effect must be NONE'),
            (setter(('alignment_status',), 'ACCEPTED'), 'alignment status cannot self-award review'),
            (setter(('provenance', 'organizational_independence_credit'), 1), 'credit must stay zero'),
            (setter(('provenance', 'human_review'), 'self'), 'credit must stay zero'),
            (setter(('provenance', 'claim'), c.CLAIM[:-1] + '5'), 'provenance identity'),
            (setter(('provenance', 'original_sha256', next(iter(original))), '0' * 64), 'provenance identity'),
            (setter(('provenance', 'original_status'), 'COMPILED'), 'provenance labels'),
            (setter(('provenance', 'namespace_mapping'), 'TorusFiltration.<n> -> TorusFiltration.<n>'), 'provenance labels'),
            (deleter(('provenance', 'human_review')), 'provenance key missing: human_review'),
            (setter(('provenance',), 'none'), 'provenance object'),
            (setter(('extra',), 1), 'SOURCE_FILES schema keys'),
            (deleter(('provenance',)), 'SOURCE_FILES schema keys'),
            (setter(('schema',), 2), 'SOURCE_FILES schema version'),
            (setter(('schema',), '1'), 'SOURCE_FILES schema version'),
            (setter(('schema',), True), 'SOURCE_FILES schema version'),
            (lambda m: m['files'].__setitem__('formal/gate.py', m['files'][workflow]), 'files inventory outside the companion'),
            (setter(('files',), {}), 'files inventory outside the companion'),
            (setter(('theorems', 'Third'), ['x']), 'per-module inventory keys'),
            (deleter(('definitions', 'TorusTopology')), 'per-module inventory keys'),
            (lambda m: m['theorems']['ClosedSuperlevel'].append('torus_compact'), 'duplicate declaration name'),
            (lambda m: m['definitions']['ClosedSuperlevel']['def'].append('superlevel_set'), 'duplicate declaration name'),
            (setter(('namespace',), 'ResearchFormalCoreR1'), 'namespace or root module'),
            (setter(('root_module',), 'TorusTopology'), 'namespace or root module'),
            (setter(('modules',), ['ClosedSuperlevel', 'TorusTopology']), 'module list'),
            (deleter(('definitions', 'TorusTopology', 'instance')), 'definition kinds: TorusTopology'),
            (setter(('theorems', 'TorusTopology', 0), 'bad-name'), 'invalid declaration name in TorusTopology'),
            (setter(('definitions', 'TorusTopology', 'def', 0), 7), 'invalid declaration name in TorusTopology'),
            (setter(('theorems', 'ClosedSuperlevel'), []), 'empty theorem list: ClosedSuperlevel'),
            (deleter(('files', workflow, 'bytes')), 'file entry keys: ' + workflow),
            (setter(('files', workflow, 'bytes'), -1), 'file bytes: ' + workflow),
            (setter(('files', workflow, 'bytes'), '1776'), 'file bytes: ' + workflow),
            (setter(('files', workflow, 'sha256'), 'zz' * 32), 'file sha256: ' + workflow),
            (setter(('files', workflow, 'git_blob'), 'ab' * 19), 'file git blob: ' + workflow),
            (lambda m: json.dumps(m, indent=2, sort_keys=True).replace('{\n  "alignment_status"', '{\n  "schema": 1,\n  "alignment_status"', 1) + '\n',
             'duplicate JSON key: schema'),
            (lambda m: json.dumps(m, indent=2, sort_keys=True).replace('"schema": 1', '"schema": 1.0', 1) + '\n',
             'nonfinite or noninteger JSON number'),
            (lambda m: json.dumps(m, indent=2, sort_keys=True).replace('"schema": 1', '"schema": NaN', 1) + '\n',
             'nonfinite or noninteger JSON number'),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(c, tmp)
            self.assertEqual(f.run('source').returncode, 0)
            for mutate, message in cases:
                with self.subTest(message=message):
                    f.manifest(mutate)
                    self.assertFails(f.run('source'), message)
                    f.restore()
            self.assertEqual(f.run('source').returncode, 0)


class FinishFixture(FixtureCase):
    def test_complete_evidence_yields_a_bounded_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(self.c, tmp)
            out = f.evidence()
            self.assertEqual({p.name for p in out.iterdir() if p.name.endswith('.lean')}, set(EMITTED))
            run = f.run('finish', str(out))
            self.assertEqual(run.returncode, 0, run.stderr)
            receipt = json.loads((out / 'receipt.json').read_text())
            self.assertEqual(receipt['schema'], 1)
            self.assertEqual(receipt['scientific_effect'], 'NONE')
            self.assertEqual(receipt['formalization_status'], 'kernel-checked')
            self.assertEqual(receipt['alignment_status'], 'PENDING_INDEPENDENT_REVIEW')
            self.assertIs(receipt['scientific_acceptance'], False)
            self.assertEqual(receipt['lean_version'], LEAN_VERSION_LINE.strip())
            self.assertEqual(receipt['dependency_revisions'], f.revisions)
            self.assertEqual(len(receipt['declarations']), 58)
            self.assertEqual(len(receipt['axioms']), 58)
            self.assertEqual(receipt['theorem_count'], 41)
            self.assertEqual(receipt['definition_count'], {'abbrev': 2, 'def': 12, 'instance': 3})
            self.assertEqual(receipt['negative_controls'],
                             {**{k: 'REJECTED_BY_LEAN' for k in self.c.FALSE_CONTROLS},
                              **{k: 'REJECTED_BY_AXIOM_AUDIT' for k in self.c.INJECTIONS}})
            self.assertEqual(receipt['provenance']['claim'], self.c.CLAIM)
            self.assertEqual(receipt['provenance']['original_sha256'], self.c.ORIGINAL_SHA256)
            self.assertEqual(receipt['leanchecker'], {'command': LEANCHECKER_MODULES, 'mode': 'modules'})
            self.assertIn('own declarations against the cached', receipt['scope_note'])
            self.assertNotIn('--fresh', receipt['scope_note'])
            for name, digest in receipt['log_sha256'].items():
                self.assertEqual(hashlib.sha256((out / name).read_bytes()).hexdigest(), digest, name)
            self.assertEqual(set(receipt['log_sha256']), {p.name for p in out.iterdir() if p.is_file()} - {'receipt.json'})
            self.assertIn('axioms.log', receipt['log_sha256'])
            self.assertIn('leanchecker.cmd', receipt['log_sha256'])
            self.assertIn('Positive.lean', receipt['log_sha256'])
            # An operator's --fresh run is recorded as such, with leanchecker's own progress line tolerated.
            (out / 'receipt.json').unlink()
            (out / 'leanchecker.cmd').write_text(LEANCHECKER_FRESH + '\n')
            (out / 'leanchecker.log').write_text('replaying TorusFiltration with --fresh\n')
            run = f.run('finish', str(out))
            self.assertEqual(run.returncode, 0, run.stderr)
            receipt = json.loads((out / 'receipt.json').read_text())
            self.assertEqual(receipt['leanchecker'], {'command': LEANCHECKER_FRESH, 'mode': 'fresh'})
            self.assertIn('--fresh', receipt['scope_note'])

    def test_every_weakening_withholds_the_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(self.c, tmp)
            out = f.evidence()
            meta = self.c.metadata()
            first = self.c.targets(meta)[0]
            axioms = (out / 'axioms.log').read_text()
            types_log = (out / 'types.log').read_text()
            audit = (out / 'Audit.lean').read_text()
            cases = [
                ('build-TorusTopology.status', '1\n', 'missing successful process: build-TorusTopology'),
                ('leanchecker.status', '', 'missing successful process: leanchecker'),
                ('leanchecker.cmd', 'leanchecker TorusFiltration\n', "unrecognized leanchecker invocation: 'leanchecker TorusFiltration'"),
                ('leanchecker.cmd', 'leanchecker --fresh TorusTopology\n', 'unrecognized leanchecker invocation'),
                ('leanchecker.log', 'uncaught exception: failed to read module data\n', 'leanchecker reported diagnostics'),
                ('toolchain.log', LEAN_VERSION_LINE.replace('4.34.1', '4.34.0'), 'unexpected running Lean version'),
                ('packages.log', 'mathlib %s\n' % ('d1' * 20), 'installed dependency HEADs differ'),
                ('axioms.log', axioms.replace('[propext, Classical.choice, Quot.sound]', '[propext, sorryAx]', 1),
                 'forbidden transitive axiom'),
                ('axioms.log', axioms.replace("'%s' depends on axioms: [propext, Classical.choice, Quot.sound]\n" % first, ''),
                 'missing target axiom report'),
                ('axioms.log', 'warning: noise\n' + axioms, 'unrecognized audit output'),
                ('types.log', ''.join(types_log.splitlines(True)[1:]), 'type inventory mismatch'),
                ('types.log', types_log + 'error: crash\n', 'type diagnostic'),
                ('RejectSuperlevelSignFlip.log', FALSE_LOG.format('RejectSuperlevelSignFlip').replace('False', 'True'),
                 'not the intended False-goal diagnostic'),
                ('RejectSuperlevelSignFlip.status', '0\n', 'false-statement control must exit exactly one'),
                ('RejectSorry.status', '1\n', 'injection control must be accepted by Lean'),
                ('RejectSorry.log', SORRY_WARNING + INJECTED.format('propext'), 'not the intended injected axiom report'),
                ('RejectNativeDecide.log', INJECTED.format('sorryAx'), 'not the intended injected axiom report'),
                ('source.json', json.dumps({**json.loads((out / 'source.json').read_text()), 'run_id': '1'}),
                 'source changed across execution'),
                ('Positive.lean', 'import TorusFiltration\n', 'emitted Lean file changed across execution: Positive.lean'),
                ('Audit.lean', ''.join(audit.splitlines(True)[:-1]), 'emitted Lean file changed across execution: Audit.lean'),
                ('Types.lean', (out / 'Types.lean').read_text().replace('set_option pp.explicit true\n', ''),
                 'emitted Lean file changed across execution: Types.lean'),
                ('RejectSorry.lean', (out / 'RejectSorry.lean').read_text().replace('by sorry', 'trivial'),
                 'emitted Lean file changed across execution: RejectSorry.lean'),
                ('RejectConstantNotPeriodic.lean', 'import TorusFiltration\n',
                 'emitted Lean file changed across execution: RejectConstantNotPeriodic.lean')]
            for name, content, message in cases:
                with self.subTest(name=name, message=message):
                    saved = (out / name).read_text()
                    (out / name).write_text(content)
                    try:
                        self.assertFails(f.run('finish', str(out)), message)
                        self.assertFalse((out / 'receipt.json').exists(), 'receipt written despite refusal')
                    finally:
                        (out / name).write_text(saved)
            for name, message in (('leanchecker.cmd', 'leanchecker invocation record absent'),
                                  ('Positive.lean', 'emitted Lean file changed across execution: Positive.lean'),
                                  ('RejectSuperlevelBelowThreshold.status', '[Errno 2] No such file or directory')):
                with self.subTest(name=name, message=message):
                    saved = (out / name).read_text()
                    (out / name).unlink()
                    try:
                        self.assertFails(f.run('finish', str(out)), message)
                        self.assertFalse((out / 'receipt.json').exists(), 'receipt written despite refusal')
                    finally:
                        (out / name).write_text(saved)
            self.assertEqual(f.run('finish', str(out)).returncode, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
