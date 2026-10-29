"""Compiled inventory tests adapted from Math-@8151f592 formal/tests/test_inventory_gate.py.
Synthetic protocol fixtures do not claim real Lean/kernel execution.
"""
import unittest
from tools import formal_compiled_inventory as gate

P = gate.PACKAGE
ALPHA, BETA = P + '.Alpha', P + '.Beta'
MODULES = {ALPHA, BETA}
TARGETS = [P + '.first', P + '.second']
STD = ('Classical.choice', 'Quot.sound', 'propext')
LOCAL = ('_local',)
# The private name Lean gave the dec3f646 hidden_sorry control (file .lake/formal-evidence/hidden_sorry.lean).
QUOTED_PRIVATE = ('_private', '.lake', 'formal-evidence', 'hidden_sorry', 0, P, 'hiddenAdmission')


def name(value):
    return tuple(value.split('.')) if isinstance(value, str) else tuple(value)


def enc(value):
    return gate.encode(name(value))


def inv(kind, origin, module, constant):
    return 'INV|' + kind + '|' + origin + '|' + enc(module) + '|' + enc(constant)


def table(alpha=5, beta=4, root='0', is_module='false', extra='0'):
    return [
        'MODULE|' + enc(ALPHA) + '|' + is_module + '|' + str(alpha) + '|' + extra,
        'MODULE|' + enc(BETA) + '|' + is_module + '|' + str(beta) + '|' + extra,
        'MODULE|' + enc(P) + '|false|' + root + '|' + root,
    ]


def axioms_line(axioms=STD):
    return 'AXIOMS|' + ','.join(sorted(enc(a) for a in axioms))


def clean(axioms=STD, closure='500', **kw):
    return '\n'.join(table(**kw) + [
        inv('theorem', 'user', ALPHA, P + '.first'),
        inv('theorem', 'user', BETA, P + '.second'),
        inv('def', 'user', ALPHA, P + '.helper'),
        inv('theorem', 'aux', ALPHA, P + '.first._proof_1'),
        inv('theorem', 'aux', ALPHA, P + '.helper._simp_1_2'),
        inv('theorem', 'aux', ALPHA, P + '.helper.eq_1'),
        inv('inductive', 'user', BETA, P + '.Shape'),
        inv('ctor', 'user', BETA, P + '.Shape.circle'),
        inv('rec', 'aux', BETA, P + '.Shape.rec'),
        axioms_line(axioms),
        'CLOSURE|' + closure,
    ])


BOUND_RECORDS = [['def', 'user', ALPHA, P + '.helper'], ['theorem', 'aux', ALPHA, P + '.first._proof_1'],
                 ['theorem', 'aux', ALPHA, P + '.helper._simp_1_2'], ['theorem', 'aux', ALPHA, P + '.helper.eq_1'],
                 ['inductive', 'user', BETA, P + '.Shape'], ['ctor', 'user', BETA, P + '.Shape.circle'],
                 ['rec', 'aux', BETA, P + '.Shape.rec']]
BOUND = gate.bound_inventory(BOUND_RECORDS, TARGETS, MODULES)


def bound_with(*records):
    return BOUND | gate.bound_inventory([list(r) for r in records], TARGETS, MODULES)


def audit(text, targets=TARGETS, modules=MODULES, bound=None, **kw):
    return gate.audit_inventory(text, targets, modules, bound=BOUND if bound is None else bound, **kw)


def plus(*lines, **kw):
    return clean(**kw) + '\n' + '\n'.join(lines)


def codes(text, targets=TARGETS, modules=MODULES, bound=None, **kw):
    """The exact (code, rendered name) findings of a well-formed but rejected inventory."""
    try:
        audit(text, targets, modules, bound, **kw)
    except gate.InventoryRejected as e:
        return {(code, gate.render(n)) for code, n, note in e.violations}
    raise AssertionError('inventory unexpectedly passed')


def wrap(body):
    return 'namespace ' + P + '\n' + body + 'end ' + P + '\n'


class NameEncodingTests(unittest.TestCase):
    def test_round_trip(self):
        for n in (('a',), (P, 'first'), QUOTED_PRIVATE, ('', 'x'), ('été', 7), ('a.b',)):
            self.assertEqual(gate.decode(gate.encode(n)), n)

    def test_lossless_where_dotted_rendering_collides(self):
        pairs = ((('a.b',), ('a', 'b')), (('0',), (0,)), (('x', '1'), ('x', 1)))
        for left, right in pairs:
            self.assertNotEqual(gate.encode(left), gate.encode(right))

    def test_malformed_encodings_refused(self):
        for bad in ('s6', 's6G', 'sC3', 'n01', 'n', 'sff', 's61/n1x', '', 'x1', 's61/'):
            with self.assertRaises(gate.InventoryProtocolError, msg=bad):
                gate.decode(bad)

    def test_render_marks_quoted_components(self):
        self.assertEqual(gate.render(QUOTED_PRIVATE), '_private.«.lake».«formal-evidence».hidden_sorry.0.' + P + '.hiddenAdmission')

    def test_private_user_name(self):
        self.assertEqual(gate.user_name(QUOTED_PRIVATE), ((P, 'hiddenAdmission'), True))
        self.assertEqual(gate.user_name((P, 'first')), ((P, 'first'), False))

    def test_simple_name_refuses_unrepresentable_targets(self):
        for bad in ('a..b', 'a.', "a'", '«x»'):
            with self.assertRaises(ValueError):
                gate.simple_name(bad)


class InventorySourceTests(unittest.TestCase):
    def test_package_import_and_module_table_enumeration(self):
        src = gate.inventory_source()
        self.assertTrue(src.startswith('import ' + P + '\nimport Lean\n'))
        for piece in ('header.moduleData[idx]!', 'pkg.isPrefixOf m', 'data.constNames', 'e.getUsedConstants.forM',
                      'findDeclarationRanges? c', '_inv.enc m', '_inv.enc c', 's.toUTF8.foldl', '| .num p k =>'):
            self.assertIn(piece, src, piece)
        for rule in ('.axiomInfo v', '.defnInfo v', '.thmInfo v', '.opaqueInfo v', '.quotInfo _', '.ctorInfo v', '.recInfo v', '.inductInfo v', '| none'):
            self.assertIn(rule, src, rule)
        self.assertNotIn('toString c', src)
        self.assertTrue(src.rstrip().endswith('`' + P + ' false'))

    def test_local_mode_adds_current_file_constants(self):
        src = gate.inventory_source(include_local=True)
        self.assertIn('map₂', src)
        self.assertIn('`_local', src)
        self.assertEqual(src.count('import Lean\n'), 1)
        self.assertTrue(src.rstrip().endswith('`' + P + ' true'))

    def test_missing_constant_is_fatal_in_lean(self):
        self.assertIn('throwError "constant', gate.inventory_source())


class AuditInventoryTests(unittest.TestCase):
    def test_clean_passes(self):
        report = audit(clean())
        self.assertEqual(report['public_theorems'], sorted(TARGETS))
        self.assertEqual(report['declarations'], 9)
        self.assertEqual(report['auxiliary_theorems'], 3)
        self.assertEqual(report['axioms'], ['Classical.choice', 'Quot.sound', 'propext'])
        self.assertEqual(report['closure'], 500)
        self.assertEqual(len(report['sha256']), 64)

    def test_axiom_free_closure_passes(self):
        self.assertEqual(audit(clean(axioms=()))['axioms'], [])

    def test_report_is_deterministic(self):
        self.assertEqual(audit(clean()), audit(clean()))

    def test_noise_lines_refused(self):
        with self.assertRaises(gate.InventoryProtocolError):
            audit('warning: something\n' + clean() + '\ntrailing')
        with self.assertRaises(gate.InventoryProtocolError):
            audit('warning: only\n')

    def test_quoted_private_sorry_rejected_for_both_reasons(self):
        # The dec3f646 helper stopped at a malformed-line error on this name; it now parses.
        text = plus(inv('theorem', 'user', ALPHA, QUOTED_PRIVATE), axioms=STD + ('sorryAx',), alpha=6)
        self.assertEqual(codes(text), {('forbidden_axiom', 'sorryAx'), ('extra_theorem', gate.render(QUOTED_PRIVATE))})

    def test_custom_axiom_in_closure_rejected(self):
        self.assertEqual(codes(clean(axioms=STD + ('Other.hidden',))), {('forbidden_axiom', 'Other.hidden')})

    def test_extra_user_theorem_rejected_whatever_its_name(self):
        # Astra's synthetic records (#312, 6009106591) and an underscore name: shape is irrelevant.
        for extra in (P + '.indentedExtra', P + '._unregistered', P + '.proof_999', P + '.first.eq_99',
                      P + '.helper._simp_999_999', P + '.first._proof_7'):
            self.assertEqual(codes(plus(inv('theorem', 'user', ALPHA, extra), alpha=6)), {('extra_theorem', extra)}, extra)

    def test_auxiliary_theorem_needs_user_parent_in_same_module(self):
        ok = (P + '.first.match_1.eq_1', P + '.helper.eq_def', P + '.first._proof_1._simp_2_3')
        for aux in ok:
            audit(plus(inv('theorem', 'aux', ALPHA, aux), alpha=6), bound=bound_with(('theorem', 'aux', ALPHA, aux)))
        bad = ((ALPHA, P + '.orphan._proof_1'), (ALPHA, P + '._proof_1'), (BETA, P + '.first._proof_9'),
               (ALPHA, P + '.first.helperLemma'), (ALPHA, P + '.first.proof_1'), (ALPHA, P + '.first.eq_1.extra'))
        for module, aux in bad:
            counts = dict(alpha=6) if module == ALPHA else dict(beta=5)
            # Bound in the manifest, so only the parent rule speaks.
            self.assertEqual(codes(plus(inv('theorem', 'aux', module, aux), **counts), bound=bound_with(('theorem', 'aux', module, aux))),
                             {('unbound_auxiliary', aux)}, aux)

    def test_auxiliary_parent_must_be_user_written(self):
        text = plus(inv('theorem', 'aux', ALPHA, P + '.first._proof_1.eq_1'), alpha=6)
        audit(text, bound=bound_with(('theorem', 'aux', ALPHA, P + '.first._proof_1.eq_1')))  # user parent, compiler suffixes
        text = plus(inv('def', 'aux', ALPHA, P + '.gen'), inv('theorem', 'aux', ALPHA, P + '.gen.eq_1'), alpha=7)
        bound = bound_with(('def', 'aux', ALPHA, P + '.gen'), ('theorem', 'aux', ALPHA, P + '.gen.eq_1'))
        self.assertEqual(codes(text, bound=bound), {('unbound_auxiliary', P + '.gen.eq_1')})

    def test_non_theorem_auxiliaries_are_closure_only(self):
        records = (('def', 'aux', ALPHA, P + '.helper._unsafe_rec'), ('opaque', 'aux', ALPHA, P + '.x'))
        audit(plus(*(inv(*r) for r in records), alpha=7), bound=bound_with(*records))

    def test_axiom_kind_rejected_even_if_unused(self):
        self.assertEqual(codes(plus(inv('axiom', 'user', ALPHA, P + '.hiddenPremise'), alpha=6)),
                         {('package_axiom', P + '.hiddenPremise'), ('unregistered_constant', P + '.hiddenPremise')})

    def test_module_table_findings(self):
        self.assertEqual(codes(plus('MODULE|' + enc(P + '.Gamma') + '|false|1|0', inv('def', 'user', P + '.Gamma', P + '.stray'))),
                         {('unregistered_module', P + '.Gamma'), ('outside_module', P + '.stray')})
        self.assertEqual(codes(plus(inv('def', 'user', P + '.Gamma', P + '.stray'))), {('outside_module', P + '.stray')})
        text = '\n'.join(line for line in clean().splitlines() if not line.startswith('MODULE|' + enc(BETA) + '|'))
        self.assertEqual(codes(text), {('missing_module', BETA)})
        self.assertEqual(codes(clean(is_module='true')), {('module_system', ALPHA), ('module_system', BETA)})
        self.assertEqual(codes(clean(root='1')), {('root_not_empty', P)})
        self.assertEqual(codes(clean(extra='1')), {('codegen_extra', ALPHA), ('codegen_extra', BETA)})
        self.assertEqual(codes(clean(alpha=4)), {('count_mismatch', ALPHA)})
        self.assertEqual(codes(plus('MODULE|' + enc(ALPHA) + '|false|5|0')), {('duplicate_module', ALPHA)})

    def test_local_module_only_when_allowed(self):
        text = plus(inv('def', 'user', LOCAL, '_eval'))
        self.assertEqual(codes(text), {('outside_module', '_eval')})
        audit(text, allow_local=True)

    def test_local_user_theorem_rejected_even_when_local_allowed(self):
        text = plus(inv('theorem', 'user', LOCAL, P + '.indentedAdmission'))
        self.assertEqual(codes(text, allow_local=True), {('extra_theorem', P + '.indentedAdmission')})

    def test_missing_target_rejected(self):
        text = '\n'.join(line for line in clean(beta=3).splitlines() if enc(P + '.second') not in line)
        self.assertEqual(codes(text), {('missing_theorem', P + '.second')})

    def test_target_only_as_auxiliary_or_private_is_missing(self):
        text = '\n'.join(line for line in clean().splitlines() if enc(P + '.second') not in line)
        self.assertIn(('missing_theorem', P + '.second'), codes(text + '\n' + inv('theorem', 'aux', BETA, P + '.second')))
        private = ('_private', P, 'Beta', 0, P, 'second')
        self.assertEqual(codes(text + '\n' + inv('theorem', 'user', BETA, private)),
                         {('missing_theorem', P + '.second'), ('extra_theorem', gate.render(private))})

    def test_duplicate_name_rejected(self):
        self.assertEqual(codes(plus(inv('theorem', 'user', ALPHA, P + '.first'), alpha=6)), {('duplicate_name', P + '.first')})

    def test_axiom_closure_line_required_exactly_once(self):
        with self.assertRaises(gate.InventoryProtocolError):
            audit(plus('AXIOMS|'))
        for prefix in ('AXIOMS|', 'CLOSURE|'):
            with self.assertRaises(gate.InventoryProtocolError):
                audit('\n'.join(l for l in clean().splitlines() if not l.startswith(prefix)))

    def test_closure_smaller_than_inventory_rejected(self):
        self.assertEqual(codes(clean(closure='3')), {('impossible_closure', 'CLOSURE')})

    def test_per_target_audit_cross_check(self):
        audit(clean(), reported_axioms={TARGETS[0]: ['propext'], TARGETS[1]: []})
        audit(clean(), reported_axioms={})
        self.assertEqual(codes(clean(axioms=('propext',)), reported_axioms={TARGETS[0]: ['propext', 'Quot.sound']}),
                         {('reported_axiom_omitted', 'Quot.sound')})

    def test_malformed_lines_are_protocol_errors(self):
        for bad in ('INV|theorem|user|' + enc(ALPHA), 'INV|macro|user|' + enc(ALPHA) + '|' + enc('x'),
                    'INV|theorem|' + enc(ALPHA) + '|' + enc(P + '.first'), 'INV|theorem|maybe|' + enc(ALPHA) + '|' + enc('x'),
                    'INV|theorem|user|' + ALPHA + '|' + P + '.x', 'INV|theorem|user|' + enc(ALPHA) + '|s6',
                    'MODULE|' + enc(ALPHA) + '|maybe|1|0', 'AXIOMS|[propext]', 'CLOSURE|x'):
            with self.assertRaises(gate.InventoryProtocolError, msg=bad):
                audit(plus(bad))

    def test_every_finding_is_reported(self):
        text = plus(inv('theorem', 'user', ALPHA, P + '.extra'), inv('axiom', 'user', BETA, P + '.ax'), axioms=STD + ('sorryAx',), alpha=6, beta=5)
        self.assertEqual(codes(text), {('extra_theorem', P + '.extra'), ('package_axiom', P + '.ax'), ('unregistered_constant', P + '.ax'),
                                       ('forbidden_axiom', 'sorryAx')})

    def test_unbound_constants_rejected_whatever_their_provenance(self):
        # The compiled RF312-SUCCESSOR-COMMAND-004 probe: a range-less theorem added by a wrapped
        # elaborator under a real user parent with a compiler-shaped suffix (#312, 6009707611).
        probe = P + '.first._proof_999'
        self.assertEqual(codes(plus(inv('theorem', 'aux', ALPHA, probe), alpha=6)), {('unregistered_constant', probe)})
        for record in (('def', 'user', ALPHA, P + '.extraDef'), ('def', 'aux', BETA, P + '.second._unsafe_rec'),
                       ('opaque', 'user', BETA, P + '.hiddenOpaque'), ('inductive', 'user', ALPHA, P + '.T')):
            counts = dict(alpha=6) if record[2] == ALPHA else dict(beta=5)
            self.assertEqual(codes(plus(inv(*record), **counts)), {('unregistered_constant', record[3])}, record)

    def test_bound_record_must_match_kind_and_provenance(self):
        text = '\n'.join(l for l in clean().splitlines() if enc(P + '.helper.eq_1') not in l)
        changed = text + '\n' + inv('theorem', 'user', ALPHA, P + '.helper.eq_1')
        self.assertEqual(codes(changed), {('extra_theorem', P + '.helper.eq_1'), ('missing_constant', P + '.helper.eq_1')})
        relabeled = clean().replace(inv('def', 'user', ALPHA, P + '.helper'), inv('def', 'aux', ALPHA, P + '.helper'))
        self.assertEqual(codes(relabeled), {('unregistered_constant', P + '.helper'), ('missing_constant', P + '.helper'),
                                            ('unbound_auxiliary', P + '.helper._simp_1_2'), ('unbound_auxiliary', P + '.helper.eq_1')})

    def test_missing_bound_constant_rejected(self):
        text = '\n'.join(l for l in clean(alpha=4).splitlines() if enc(P + '.helper.eq_1') not in l)
        self.assertEqual(codes(text), {('missing_constant', P + '.helper.eq_1')})

    def test_default_bound_is_empty(self):
        with self.assertRaises(gate.InventoryRejected):
            gate.audit_inventory(clean(), TARGETS, MODULES)


class BoundInventoryTests(unittest.TestCase):
    def test_valid_records(self):
        self.assertEqual(len(BOUND), 7)
        self.assertIn(('theorem', 'aux', name(ALPHA), name(P + '.first._proof_1')), BOUND)

    def test_invalid_records_refused(self):
        for records in (None, {}, [['def', 'user', ALPHA]], [['axiom', 'user', ALPHA, P + '.x']], [['theorem', 'user', ALPHA, P + '.x']],
                        [['def', 'maybe', ALPHA, P + '.x']], [['def', 'user', P + '.Gamma', P + '.x']], [['def', 'user', ALPHA, P + '.first']],
                        [['def', 'user', ALPHA, P + '.x'], ['def', 'user', ALPHA, P + '.x']],
                        [['def', 'user', ALPHA, P + '.x'], ['opaque', 'user', ALPHA, P + '.x']], [['def', 'user', ALPHA, "x'"]], [[1, 2, 3, 4]]):
            with self.assertRaises(ValueError, msg=repr(records)):
                gate.bound_inventory(records, TARGETS, MODULES)

    def test_empty_targets_or_modules_rejected(self):
        for targets, modules in (([], MODULES), (TARGETS + [TARGETS[0]], MODULES), (TARGETS, set())):
            with self.assertRaises(ValueError):
                gate.audit_inventory(clean(), targets, modules)




class IntegrationTests(unittest.TestCase):
    def test_source_requires_compiled_binding(self):
        from tests.test_formal_gate import SyntheticControls
        fixture = SyntheticControls('test_valid_source_check_and_no_writes')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.manifest.pop('compiled_inventory', None)
        fixture.save()
        with self.assertRaisesRegex(ValueError, 'compiled inventory binding'):
            fixture.check()

class AuditorMutationTests(unittest.TestCase):
    def test_policy_mutants_are_killed(self):
        import types
        from pathlib import Path
        import io
        source = Path(gate.__file__).read_text()
        mutations = {
            'ignore_noise': ('lines = text.splitlines()', "lines = [line for line in text.splitlines() if line.startswith(('INV|', 'MODULE|', 'AXIOMS|', 'CLOSURE|'))]"),
            'allow_forbidden': ("flag('forbidden_axiom', name)", 'pass'),
            'allow_extra_constant': ("flag('unregistered_constant', name, kind + '/' + origin)", 'pass'),
            'allow_missing_module': ("flag('missing_module', name)", 'pass'),
            'allow_duplicate_constant': ("flag('duplicate_name', name)", 'pass'),
            'allow_extra_theorem': ("flag('extra_theorem', name)", 'pass'),
            'allow_bad_count': ("flag('count_mismatch', name, str(per_module[name]) + '/' + str(counts[name]))", 'pass'),
        }
        original = globals()['gate']
        try:
            for label, (old, new) in mutations.items():
                with self.subTest(mutation=label):
                    self.assertEqual(source.count(old), 1)
                    mutant = types.ModuleType('mutant')
                    exec(compile(source.replace(old, new), '<auditor-mutant>', 'exec'), mutant.__dict__)
                    globals()['gate'] = mutant
                    suite = unittest.defaultTestLoader.loadTestsFromTestCase(AuditInventoryTests)
                    result = unittest.TextTestRunner(stream=io.StringIO()).run(suite)
                    self.assertFalse(result.wasSuccessful(), 'surviving auditor mutation: ' + label)
        finally:
            globals()['gate'] = original

class PinnedLeanInventoryTests(unittest.TestCase):
    def test_actual_clean_and_omitted_constants(self):
        import json, os, shutil, subprocess, tempfile
        from pathlib import Path
        lake = shutil.which('lake')
        if not lake:
            self.skipTest('pinned Lean unavailable')
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root / 'formal/manifest.json').read_text())
        targets = [t['name'] for t in manifest['targets']]
        modules = [p[:-5].replace('/', '.') for p in manifest['source_modules']]
        bound = gate.bound_inventory(manifest['compiled_inventory'], targets, modules)
        specimens = {
            'clean': ('', set()),
            'unused_admit': ('\ndef UniversalLaw.Side24.unusedFalse : False := by admit\n', {'forbidden_axiom','unregistered_constant'}),
            'hidden_theorem': ('\nnamespace UniversalLaw.Side24\n  private theorem hiddenFalse : False := by admit\nend UniversalLaw.Side24\n', {'forbidden_axiom','extra_theorem'}),
            'opaque_admit': ('\nopaque UniversalLaw.Side24.unusedOpaque : False := by admit\n', {'forbidden_axiom','unregistered_constant'}),
            'custom_axiom': ('\naxiom hiddenPremise : False\ndef UniversalLaw.Side24.fromHidden : False := hiddenPremise\n', {'forbidden_axiom','package_axiom','unregistered_constant'}),
            'native': ('\ndef UniversalLaw.Side24.nativeUnused : 1 = 1 := by native_decide\n', {'forbidden_axiom','unregistered_constant'}),
        }
        with tempfile.TemporaryDirectory() as temp:
            pkg = Path(temp) / 'formal'
            shutil.copytree(root / 'formal', pkg, ignore=shutil.ignore_patterns('.lake'))
            module = pkg / 'UniversalLaw/Side24/Endpoints.lean'
            original = module.read_text()
            audit = pkg / '.lake/Inventory.lean'
            def run(args):
                p = subprocess.run([lake, *args], cwd=pkg, text=True, capture_output=True, timeout=180)
                self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
                return p.stdout
            self.assertIn('4.34.1', run(['env','lean','--version']))
            for label, (injection, expected) in specimens.items():
                with self.subTest(specimen=label):
                    module.write_text(original + injection)
                    run(['build'])
                    audit.parent.mkdir(exist_ok=True)
                    audit.write_text(gate.inventory_source())
                    transcript = run(['env','lean',str(audit)])
                    if not expected:
                        report = gate.audit_inventory(transcript, targets, modules, bound=bound, codegen=gate.bound_codegen(manifest['codegen_inventory'], modules))
                        self.assertEqual(report['declarations'],45)
                        self.assertEqual(len(report['public_theorems']),31)
                        self.assertEqual(report['axioms'],[])
                    else:
                        with self.assertRaises(gate.InventoryRejected) as caught:
                            gate.audit_inventory(transcript, targets, modules, bound=bound, codegen=gate.bound_codegen(manifest['codegen_inventory'], modules))
                        codes = {v[0] for v in caught.exception.violations}
                        self.assertTrue(expected <= codes, (label,codes))

class CodegenInventoryTests(unittest.TestCase):
    def test_codegen_names_are_bound_separately(self):
        extra = ALPHA + "._boxed"
        text = clean().replace('|false|5|0', '|false|5|1')
        text += '\nEXTRA|' + enc(ALPHA) + '|' + enc(extra)
        with self.assertRaises((gate.InventoryProtocolError, gate.InventoryRejected)):
            audit(text)
        self.assertIn('EXTRA|', gate.inventory_source())
        binding = gate.bound_codegen([[ALPHA,extra]], MODULES)
        report = gate.audit_inventory(text, TARGETS, MODULES, bound=BOUND, codegen=binding)
        self.assertEqual(report['codegen_entries'],1)
        for changed in (text + '\nEXTRA|' + enc(ALPHA) + '|' + enc(extra), text.replace('|false|5|1','|false|5|2')):
            with self.assertRaises(gate.InventoryRejected):
                gate.audit_inventory(changed, TARGETS, MODULES, bound=BOUND, codegen=binding)

    def test_closure_missing_reference_is_fatal(self):
        self.assertIn('axiom closure references missing constants', gate.inventory_source())

if __name__ == '__main__':
    unittest.main()
