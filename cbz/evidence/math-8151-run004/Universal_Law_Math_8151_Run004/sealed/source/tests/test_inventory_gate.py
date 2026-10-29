"""RF-GATE-01: the environment inventory, grammar pre-check and comment stripper.

Synthetic fixtures only; nothing here compiles Lean. The authoritative inventory is produced
by `inventory_source()` inside `gate.py --execute` and audited by `audit_inventory`. The
source-gate regression cases first published by Sol on PR #312 are replayed against a copied
formal tree in `SourceGateRegressionTests`.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('inventory_gate_under_test', ROOT / 'gate.py')
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)

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


class StripCommentsTests(unittest.TestCase):
    def test_line_comment(self):
        self.assertEqual(gate.strip_lean_comments('theorem a -- theorem hidden\n:= rfl'), 'theorem a \n:= rfl')

    def test_block_comment_nested(self):
        self.assertEqual(gate.strip_lean_comments('x /- a /- theorem b -/ sorry -/ y'), 'x   y')

    def test_doc_comment(self):
        self.assertEqual(gate.strip_lean_comments('/-- doc with sorry -/\ntheorem a : True := trivial'), ' \ntheorem a : True := trivial')

    def test_line_comment_inside_block_does_not_close(self):
        self.assertEqual(gate.strip_lean_comments('/- -- still -/ z'), '  z')

    def test_newline_positions_preserved(self):
        text = 'a -- c\n/- b\n c -/ d\n/- x /- y\n -/ -/ e\n'
        stripped = gate.strip_lean_comments(text)
        self.assertEqual(stripped.count('\n'), text.count('\n'))
        self.assertEqual(stripped.splitlines()[2], '  d')

    def test_block_comment_separates_tokens(self):
        # RF312-SUCCESSOR-COMMAND-004: `opaque/-x-/hidden` must not become `opaquehidden`.
        self.assertEqual(gate.strip_lean_comments('opaque/-x-/hidden'), 'opaque hidden')

    def test_unterminated(self):
        with self.assertRaises(ValueError):
            gate.strip_lean_comments('/- open forever')

    def test_no_comments(self):
        text = 'theorem a : True := trivial\n'
        self.assertEqual(gate.strip_lean_comments(text), text)

    def test_comment_markers_inside_string_are_not_comments(self):
        # The old stripper cut the rest of this line, hiding the code after the string.
        self.assertEqual(gate.strip_lean_comments('def s := "a -- b" ++ sorry'), 'def s := "a -- b" ++ sorry')
        self.assertEqual(gate.strip_lean_comments('def s := "/-" ++ x -- "-/"'), 'def s := "/-" ++ x ')

    def test_escaped_quote_does_not_end_string(self):
        self.assertEqual(gate.strip_lean_comments('def s := "a \\" -- b" -- c'), 'def s := "a \\" -- b" ')

    def test_character_literals(self):
        self.assertEqual(gate.strip_lean_comments("def c := '\"' -- \"\nx"), "def c := '\"' \nx")
        self.assertEqual(gate.strip_lean_comments("def c := '\\'' -- q"), "def c := '\\'' ")

    def test_apostrophe_in_identifier_is_not_a_literal(self):
        self.assertEqual(gate.strip_lean_comments("theorem h' : x' = x' := rfl -- c"), "theorem h' : x' = x' := rfl ")

    def test_raw_strings(self):
        self.assertEqual(gate.strip_lean_comments('def s := r"--" -- c'), 'def s := r"--" ')
        self.assertEqual(gate.strip_lean_comments('def s := r#"a "-- b"# -- c'), 'def s := r#"a "-- b"# ')

    def test_comment_contents_never_open_strings(self):
        self.assertEqual(gate.strip_lean_comments('-- an "unterminated\n/- also " -/ ok'), '\n  ok')

    def test_unsupported_literals_refused(self):
        for text in ('def s := s!"x"', 'def s := "{x}"', 'def s := "open', "def c := f ' x", 'def s := r#"open', "def c := 'ab'"):
            with self.assertRaises(ValueError, msg=text):
                gate.strip_lean_comments(text)


class GrammarCheckTests(unittest.TestCase):
    def test_canonical_names(self):
        text = wrap('theorem first : True := trivial\nlemma second : True := trivial\nnoncomputable def third : Nat := 1\n')
        self.assertEqual(gate.grammar_check(text, 'm.lean'), ['first', 'second'])

    def test_refused_spellings(self):
        for body in ('  theorem hidden : True := trivial\n', 'private theorem hidden : True := trivial\n',
                     'protected theorem hidden : True := trivial\n', '@[simp] theorem hidden : True := trivial\n',
                     '  def hidden : Nat := 1\n', 'private def hidden : Nat := 1\n',
                     'def a : Nat := 1 theorem b : True := trivial\n',
                     'def s : String := "a theorem walks in"\n',
                     'private\ttheorem hidden : True := trivial\n', 'public theorem hidden : True := trivial\n',
                     'theorem\thidden : True := trivial\n'):
            with self.assertRaises(ValueError, msg=body):
                gate.grammar_check(wrap(body), 'm.lean')

    def test_unsupported_commands_refused(self):
        for body in ('opaque hidden : Nat\n', 'abbrev hidden : Nat := 1\n', 'instance : Inhabited Nat := ⟨0⟩\n',
                     'example : True := trivial\n', 'structure S where\n  x : Nat\n', 'inductive T where\n  | a\n',
                     'mutual\nend\n', 'macro "m" : term => `(1)\n', 'elab "e" : term => pure default\n',
                     'attribute [simp] first\n', 'notation "n" => 1\n', 'run_cmd pure ()\n', '#eval 1\n',
                     '  #print axioms first\n', 'initialize pure ()\n', 'local notation "n" => 1\n',
                     'section\nend\n', 'class C where\n'):
            with self.assertRaises(ValueError, msg=body):
                gate.grammar_check(wrap(body), 'm.lean')

    def test_attribute_previous_line_allowed(self):
        self.assertEqual(gate.grammar_check(wrap('@[simp]\ntheorem ok : True := trivial\n'), 'm.lean'), ['ok'])

    def test_keyword_inside_comment_ignored(self):
        text = wrap('-- theorem commented : False\n/- lemma also : False\nprivate theorem x -/\ntheorem ok : True := trivial\n')
        self.assertEqual(gate.grammar_check(text, 'm.lean'), ['ok'])

    def test_keyword_as_identifier_part_ignored(self):
        self.assertEqual(gate.grammar_check(wrap('def my_theorem_count : Nat := 1\ntheorem ok : True := Function.eq_def\n'), 'm.lean'), ['ok'])

    def test_comment_marker_in_string_cannot_hide_code(self):
        with self.assertRaises(ValueError):
            gate.grammar_check(wrap('def s : String := "--" ++ by sorry\n'), 'm.lean')

    def test_apostrophe_and_unicode_names_refused(self):
        for bad in ("foo'", '«hidden»', 'foo✓'):
            with self.assertRaises(ValueError):
                gate.grammar_check(wrap('theorem ' + bad + ' : True := trivial\n'), 'm.lean')

    def test_namespace_structure(self):
        gate.grammar_check(wrap(''), 'm.lean')
        for text in ('theorem a : True := trivial\n', 'namespace Other\ntheorem a : True := trivial\nend Other\n',
                     wrap('namespace Nested\ntheorem a : True := trivial\nend Nested\n'), wrap('') + wrap(''),
                     'namespace ' + P + '\n', '  namespace ' + P + '\nend ' + P + '\n'):
            with self.assertRaises(ValueError, msg=text):
                gate.grammar_check(text, 'm.lean')

    def test_duplicate_declaration_refused(self):
        with self.assertRaises(ValueError):
            gate.grammar_check(wrap('theorem a : True := trivial\ntheorem a : True := trivial\n'), 'm.lean')

    def test_refusal_names_original_line_number(self):
        text = '/- a\n b -/\n-- c\nnamespace ' + P + '\n  theorem hidden : True := trivial\nend ' + P + '\n'
        with self.assertRaises(ValueError) as cm:
            gate.grammar_check(text, 'm.lean')
        self.assertIn('m.lean:5', str(cm.exception))

    def test_forbidden_tokens(self):
        for body in ('theorem bad : False := by sorry\n', 'axiom hidden : False\n', 'theorem t : 2 + 2 = 4 := by native_decide\n',
                     '@[implemented_by f]\ndef g : Nat := 0\n', '@[extern "c"]\ndef g : Nat := 0\n', 'unsafe def g : Nat := 0\n'):
            with self.assertRaises(ValueError, msg=body):
                gate.grammar_check(wrap(body), 'm.lean')

    def test_forbidden_token_inside_string_refused(self):
        with self.assertRaises(ValueError):
            gate.grammar_check(wrap('def s : String := "sorry"\n'), 'm.lean')

    def test_tokens_in_comments_and_identifiers_allowed(self):
        self.assertEqual(gate.grammar_check(wrap('-- no sorry here\ntheorem ok : True := trivial -- sorryAx\n'), 'm.lean'), ['ok'])

    def test_unterminated_comment_refused(self):
        with self.assertRaises(ValueError):
            gate.grammar_check(wrap('/- theorem a : True := trivial\n'), 'm.lean')

    def test_wrapped_and_term_level_metaprograms_refused(self):
        # RF312-SUCCESSOR-COMMAND-004 reproduction (#312, 6009593567) plus nested and term forms.
        for body in ('set_option maxRecDepth 1000 in run_elab pure ()\n', 'open Lean in run_elab pure ()\n',
                     'set_option maxRecDepth 1000 in #check True\n', 'opaque/-separator-/hidden : Nat := 0\n',
                     'theorem t : True := by_elab pure default\n', 'set_option a 1 in\nset_option b 2 in\n  run_elab pure ()\n',
                     'theorem t : True := by\n  run_tac pure ()\n', 'theorem t : True := by\n  #check True\n  trivial\n',
                     'set_option maxRecDepth 1000 in\n  Lean.addDecl x\n', 'def m : MetaM Unit := pure ()\n',
                     'set_option x 1 in opaque y : Nat\n', 'theorem t : True := (by exact trivial : True) -- ok\nelab_rules : term | _ => default\n'):
            with self.assertRaises(ValueError, msg=body):
                gate.grammar_check(wrap(body), 'm.lean')

    def test_set_option_in_before_theorem_allowed(self):
        self.assertEqual(gate.grammar_check(wrap('set_option maxHeartbeats 400000 in\ntheorem ok : True := trivial\n'), 'm.lean'), ['ok'])

    def test_definitions_collected(self):
        definitions = []
        gate.grammar_check(wrap('noncomputable def f : Nat := 1\ndef g : Nat := 2\ntheorem ok : True := trivial\n'), 'm.lean', definitions)
        self.assertEqual(definitions, ['f', 'g'])


class NameEncodingTests(unittest.TestCase):
    def test_round_trip(self):
        for n in (('a',), (P, 'first'), QUOTED_PRIVATE, ('', 'x'), ('été', 7), ('a.b',)):
            self.assertEqual(gate.decode(gate.encode(n)), n)

    def test_lossless_where_dotted_rendering_collides(self):
        pairs = ((('a.b',), ('a', 'b')), (('0',), (0,)), (('x', '1'), ('x', 1)))
        for left, right in pairs:
            self.assertNotEqual(gate.encode(left), gate.encode(right))

    def test_malformed_encodings_refused(self):
        for bad in ('s6', 's6G', 'sC3', 'n01', 'n', 'sff', 's61/n1x'):
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

    def test_noise_lines_ignored_but_inventory_required(self):
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

    def test_live_manifest_binding(self):
        m = gate.load_json((ROOT / 'manifest.json').read_text())
        modules = {P + '.' + path[len(P) + 1:-5] for path in m['source_modules']}
        bound = gate.bound_inventory(m['compiled_inventory'], m['targets'], modules)
        kinds = sorted({(e[0], e[1]) for e in bound})
        self.assertEqual(kinds, [('def', 'user'), ('theorem', 'aux')])
        self.assertEqual(len(bound), 53)
        definitions = []
        for path in m['source_modules']:
            gate.grammar_check((ROOT / path).read_text(), path, definitions)
        self.assertEqual({gate.render(e[3]) for e in bound if e[0] == 'def'}, {P + '.' + d for d in definitions})

    def test_empty_targets_or_modules_rejected(self):
        for targets, modules in (([], MODULES), (TARGETS + [TARGETS[0]], MODULES), (TARGETS, set())):
            with self.assertRaises(ValueError):
                gate.audit_inventory(clean(), targets, modules)


class AdmissionCheckTests(unittest.TestCase):
    INJECTED = ('theorem', 'user', (P, 'hiddenAdmission'), True, LOCAL)
    EXPECTED = {('forbidden_axiom', ('sorryAx',), False), ('extra_theorem', (P, 'hiddenAdmission'), True)}
    PRIVATE = ('_private', '.lake', 'formal-evidence', 'hidden_sorry', 0, P, 'hiddenAdmission')

    def check(self, text, injected=None, expected=None):
        return gate.check_admission(text, 'hidden_sorry', injected or self.INJECTED, self.EXPECTED if expected is None else expected, TARGETS, MODULES, True, BOUND)

    def test_expected_reason_counts(self):
        result = self.check(plus(inv('theorem', 'user', LOCAL, self.PRIVATE), axioms=STD + ('sorryAx',)))
        self.assertEqual(result['outcome'], 'REJECTED_FOR_EXPECTED_REASON')
        self.assertEqual(result['injected'], gate.render(self.PRIVATE))

    def test_passing_inventory_fails_the_experiment(self):
        with self.assertRaises(ValueError) as cm:
            self.check(plus(inv('theorem', 'aux', LOCAL, self.PRIVATE)), injected=('theorem', 'aux', (P, 'hiddenAdmission'), True, LOCAL))
        self.assertIn('unexpected reason', str(cm.exception))

    def test_other_reason_fails_the_experiment(self):
        # sorryAx missing from the closure: rejected, but not for the control's reason.
        with self.assertRaises(ValueError) as cm:
            self.check(plus(inv('theorem', 'user', LOCAL, self.PRIVATE)))
        self.assertIn('unexpected reason', str(cm.exception))
        with self.assertRaises(ValueError):
            self.check(plus(inv('theorem', 'user', LOCAL, self.PRIVATE), inv('axiom', 'user', LOCAL, P + '.extraAx'), axioms=STD + ('sorryAx',)))

    def test_protocol_error_fails_the_experiment(self):
        # The historical dec3f646 transcript shape: the private name was printed with
        # Name.toString and failed the old pattern. Such output is never a rejection.
        old = plus('INV|theorem|_local|_private.«.lake».«formal-evidence».hidden_sorry.0.' + P + '.hiddenAdmission', axioms=STD + ('sorryAx',))
        with self.assertRaises(gate.InventoryProtocolError):
            self.check(old)

    def test_missing_injection_fails_the_experiment(self):
        with self.assertRaises(ValueError) as cm:
            self.check(plus(inv('theorem', 'user', LOCAL, (P, 'otherName')), axioms=STD + ('sorryAx',)),
                       expected={('forbidden_axiom', ('sorryAx',), False), ('extra_theorem', (P, 'otherName'), False)})
        self.assertIn('did not inject', str(cm.exception))

    def test_same_short_private_name_elsewhere_fails_the_experiment(self):
        # RF312-CONTROL-PRIVATE-IDENTITY-006 (#312, 6014639706): an unrelated private theorem with
        # the same short name in another module must not collapse into the expected finding.
        injected = ('theorem', 'user', (P, 'compiledAdmission'), True, name(ALPHA))
        expected = {('forbidden_axiom', ('sorryAx',), False), ('extra_theorem', (P, 'compiledAdmission'), True)}
        mine = ('_private',) + name(ALPHA) + (0, P, 'compiledAdmission')
        other = ('_private',) + name(BETA) + (0, P, 'compiledAdmission')
        alone = plus(inv('theorem', 'user', ALPHA, mine), axioms=STD + ('sorryAx',), alpha=6)
        result = gate.check_admission(alone, 'compiled_admission', injected, expected, TARGETS, MODULES, False, BOUND)
        self.assertEqual(result['injected'], gate.render(mine))
        both = plus(inv('theorem', 'user', ALPHA, mine), inv('theorem', 'user', BETA, other), axioms=STD + ('sorryAx',), alpha=6, beta=5)
        with self.assertRaises(ValueError) as cm:
            gate.check_admission(both, 'compiled_admission', injected, expected, TARGETS, MODULES, False, BOUND)
        self.assertIn('unexpected reason', str(cm.exception))
        self.assertIn(gate.render(other), str(cm.exception))

    def test_unidentifiable_private_expectation_refused(self):
        injected = ('theorem', 'user', (P, 'hiddenAdmission'), True, LOCAL)
        expected = {('extra_theorem', (P, 'hiddenAdmission'), True), ('extra_theorem', (P, 'otherPrivate'), True)}
        with self.assertRaises(ValueError):
            gate.check_admission(plus(inv('theorem', 'user', LOCAL, self.PRIVATE)), 'x', injected, expected, TARGETS, MODULES, True, BOUND)

    def test_injection_in_wrong_module_fails_the_experiment(self):
        with self.assertRaises(ValueError):
            self.check(plus(inv('theorem', 'user', ALPHA, self.PRIVATE), axioms=STD + ('sorryAx',), alpha=6))


class SourceCheckIntegrationTests(unittest.TestCase):
    def test_live_modules_pass_grammar_and_match_manifest(self):
        m = gate.load_json((ROOT / 'manifest.json').read_text())
        names = []
        for path in m['source_modules']:
            names.extend(P + '.' + n for n in gate.grammar_check((ROOT / path).read_text(), path))
        self.assertEqual(names, m['targets'])

    def test_required_files_bound(self):
        m = gate.load_json((ROOT / 'manifest.json').read_text())
        for required in ('tests/test_inventory_gate.py', 'GATE_HARDENING.md', 'gate.py'):
            self.assertIn(required, m['files'])


class SourceGateRegressionTests(unittest.TestCase):
    """Sol's #312 source-gate regression cases, replayed against a copied formal tree."""
    MODULE = 'ResearchFormalCoreR1/AlgebraV2.lean'

    def fixture(self, extra, targets=()):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        root = Path(td.name) / 'formal'
        shutil.copytree(ROOT, root, ignore=shutil.ignore_patterns('.lake', '__pycache__', '*.pyc'))
        path = root / self.MODULE
        marker = '\nend ' + P + '\n'
        text = path.read_text()
        self.assertEqual(text.count(marker), 1)
        path.write_text(text.replace(marker, '\n' + extra + '\n' + marker))
        manifest = json.loads((root / 'manifest.json').read_text())
        manifest['files'][self.MODULE] = hashlib.sha256(path.read_bytes()).hexdigest()
        at = manifest['targets'].index(P + '.ec014_contact_power') + 1
        manifest['targets'][at:at] = list(targets)
        (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        return subprocess.run([sys.executable, 'gate.py'], cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)

    def test_hidden_declaration_forms_are_rejected(self):
        for extra in ('  theorem rf_gate_indented : False := by sorry', 'private theorem rf_gate_private : False := by sorry',
                      '@[simp] theorem rf_gate_attributed : False := by sorry', 'def rf_gate_unused : False := by sorry',
                      'opaque rf_gate_opaque : False', 'axiom rf_gate_axiom : False',
                      'def rf_gate_string : String := "--" ++ "x"\nprivate theorem rf_gate_after : True := trivial'):
            with self.subTest(extra=extra):
                self.assertNotEqual(self.fixture(extra).returncode, 0)

    def test_comment_text_is_not_inventoried_as_a_declaration(self):
        result = self.fixture('/-\ntheorem rf_gate_comment_ghost : False := by sorry\n-/')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_duplicate_target_list_is_rejected_source_side(self):
        result = self.fixture('theorem ec005_fold_gap (s : ℝ) : True := by trivial', [P + '.ec005_fold_gap'])
        self.assertNotEqual(result.returncode, 0)

    def test_nested_namespace_cannot_be_misprefixed(self):
        result = self.fixture('namespace Nested\ntheorem rf_gate_nested : True := by trivial\nend Nested', [P + '.rf_gate_nested'])
        self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
