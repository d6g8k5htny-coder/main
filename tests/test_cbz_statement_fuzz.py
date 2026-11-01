"""Tests for tools/cbz/statement_fuzz.py — the CBZ-0 G12 empirical statement check.

The harness evaluates core-Lean arithmetic *statements* independently of Lean, in exact arithmetic
(Python integers and fractions, never floats), with Lean's Nat/Int semantics. Run:

    python3 -B -S -m unittest tests.test_cbz_statement_fuzz -v     (and with -O)
"""
import json
import os
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools' / 'cbz'))

import statement_fuzz as sf  # noqa: E402


LEDGER = '''
namespace Demo

/-- doc comment with a tempting 1 + 1 = 3 inside, which is text, not a statement -/
def fact : Nat → Nat
  | 0 => 1
  | n + 1 => (n + 1) * fact n

def pairingSum (q : Nat) : Nat :=
  (List.range (q / 2 + 1)).foldl
    (fun acc j => acc + fact q / (2 ^ j * fact j * fact (q - 2 * j))) 0

def sixA (d : Nat) : Nat := 6 * (d - 1) + 4 + 3 * ((d - 1) * d / 2)
def sixB (d : Nat) : Nat := 6 * d + 3 * ((d - 1) * d / 2)
def imageConstant : Nat := 1458 * (76 * 24 ^ 6 + 15)

theorem pairing_sum_six : pairingSum 6 = 76 := by decide

theorem pairing_sum_le_six : ∀ q, q ∈ [0, 1, 2, 3, 4, 5, 6] → pairingSum q ≤ 76 := by
  decide

theorem odd_block_eigenvalues :
    (1 : Int) + 15 = 2 * 8 ∧ (1 : Int) * 15 - (-3) * (-3) = 6 ∧ (8 : Int) ^ 2 - 58 = 6 := by
  decide

theorem exponent_bounds :
    ∀ d, d ∈ [2, 3] → sixA d + 2 * sixB d < 84 ∧ 2 * sixA d + sixB d < 78 := by
  decide

theorem image_constant_value : imageConstant = 21175738586478 := by decide

end Demo
'''


class TokenizerTests(unittest.TestCase):
    def test_tokens_cover_unicode_operators_and_ascriptions(self):
        toks = [t.value for t in sf.tokenize('(15 : Int) - (-3) * (-3) / 1 = 6 ∧ ∀ q, q ∈ [0, 1] → q ≤ 1')]
        self.assertEqual(toks[:9], ['(', '15', ':', 'Int', ')', '-', '(', '-', '3'])
        self.assertIn('∧', toks)
        self.assertIn('∀', toks)
        self.assertIn('∈', toks)
        self.assertIn('→', toks)
        self.assertIn('≤', toks)

    def test_comments_are_stripped(self):
        toks = [t.value for t in sf.tokenize('1 -- trailing\n+ /- block 2 -/ 3')]
        self.assertEqual(toks, ['1', '+', '3'])


class EvaluationTests(unittest.TestCase):
    def ev(self, text, defs=''):
        module = sf.parse_module(defs + f'\ntheorem t : {text} := by decide\n')
        return sf.evaluate_theorem(module, module.theorems[-1])   # the appended theorem

    def test_true_and_false_closed_identities(self):
        self.assertEqual(self.ev('1 * 3 * 5 = 15').status, 'PASS')
        self.assertEqual(self.ev('1 * 3 * 5 = 16').status, 'FAIL')

    def test_a_bound_gives_a_finite_domain_only_for_a_nat_binder(self):
        # x : Int with only an upper bound ranges over all negative integers; x = -4 falsifies these
        for text in ('∀ x : Int, x ≤ 3 → x * x ≤ 9', '∀ x : ℤ, x < 3 → x + 10 > 0',
                     '∀ x, x ≤ 3 → x * x ≤ (9 : Int)'):
            with self.subTest(text=text):
                self.assertEqual(self.ev(text).status, 'UNSUPPORTED')
        self.assertEqual(self.ev('∀ x : Nat, x ≤ 3 → x * x ≤ 9').status, 'PASS')
        self.assertEqual(self.ev('∀ x : ℕ, x < 4 → x * x ≤ 8').status, 'FAIL')

    def test_nat_subtraction_truncates_and_int_does_not(self):
        self.assertEqual(self.ev('2 - 5 + 5 = 5').status, 'PASS')        # Nat: (2 - 5) = 0
        self.assertEqual(self.ev('(2 : Int) - 5 + 5 = 2').status, 'PASS')
        self.assertEqual(self.ev('(2 : Int) - 5 + 5 = 5').status, 'FAIL')

    def test_nat_division_floors_and_exponent_binds_tightest(self):
        self.assertEqual(self.ev('24 ^ 2 / 2 = 288').status, 'PASS')
        self.assertEqual(self.ev('7 / 2 = 3 ∧ 7 % 2 = 1').status, 'PASS')
        self.assertEqual(self.ev('2 ^ 3 * 2 = 16').status, 'PASS')

    def test_int_division_and_modulus_follow_lean_core_euclidean_convention(self):
        # Lean 4 core: `/` on Int is Int.ediv and `%` is Int.emod (Euclidean: 0 ≤ r < |b|), and both are
        # totalized at b = 0 (q = 0, r = a). Golden controls from main#684/6511137449; the two cases with a
        # negative divisor are exactly where Python's floor division disagrees.
        goldens = [((5, -2), (-2, 1)), ((-5, -2), (3, 1)), ((-5, 2), (-3, 1)),
                   ((5, 2), (2, 1)), ((5, 0), (0, 5)), ((-5, 0), (0, -5))]
        for (a, b), (q, r) in goldens:
            with self.subTest(a=a, b=b):
                self.assertEqual(self.ev(f'({a} : Int) / {b} = {q} ∧ ({a} : Int) % {b} = {r}').status, 'PASS')
                self.assertEqual(self.ev(f'({a} : Int) / {b} = {q + 1}').status, 'FAIL')
        # Python semantics would read (5, -2) as (-3, -1): that reading must be reported FAIL, not PASS
        self.assertEqual(self.ev('(5 : Int) / -2 = -3').status, 'FAIL')
        self.assertEqual(self.ev('(5 : Int) % -2 = -1').status, 'FAIL')
        # T-division (toward zero) would read (-5, 2) as (-2, -1): also FAIL
        self.assertEqual(self.ev('(-5 : Int) / 2 = -2').status, 'FAIL')

    def test_nat_division_and_modulus_are_totalized_at_zero(self):
        self.assertEqual(self.ev('7 / 0 = 0 ∧ 7 % 0 = 7').status, 'PASS')

    def test_recursive_def_with_patterns(self):
        self.assertEqual(self.ev('fact 6 = 720', LEDGER).status, 'PASS')
        self.assertEqual(self.ev('fact 6 = 721', LEDGER).status, 'FAIL')

    def test_list_range_foldl_lambda(self):
        self.assertEqual(self.ev('pairingSum 6 = 76', LEDGER).status, 'PASS')
        self.assertEqual(self.ev('imageConstant = 21175738586478', LEDGER).status, 'PASS')

    def test_finite_quantifier_is_enumerated_exhaustively(self):
        r = self.ev('∀ q, q ∈ [0, 1, 2, 3, 4, 5, 6] → pairingSum q ≤ 76', LEDGER)
        self.assertEqual(r.status, 'PASS')
        self.assertEqual(r.instances, 7)
        self.assertFalse(r.vacuous)
        r2 = self.ev('∀ d, d ∈ [2, 3] → sixA d + 2 * sixB d < 84 ∧ 2 * sixA d + sixB d < 78', LEDGER)
        self.assertEqual(r2.status, 'PASS')
        self.assertEqual(r2.instances, 2)

    def test_quantifier_failure_names_the_witness(self):
        r = self.ev('∀ q, q ∈ [0, 1, 2, 3, 4, 5, 6] → pairingSum q ≤ 75', LEDGER)
        self.assertEqual(r.status, 'FAIL')
        self.assertIn('q = 6', r.detail)

    def test_empty_domain_is_flagged_vacuous(self):
        r = self.ev('∀ q, q ∈ [] → 1 = 2', LEDGER)
        self.assertEqual(r.status, 'PASS')
        self.assertTrue(r.vacuous)

    def test_boundary_probe_reports_tightness_outside_the_domain(self):
        # 2a+b < 78 is 77 < 78 at d = 3 and fails at d = 4: the probe records that, as information only
        r = self.ev('∀ d, d ∈ [2, 3] → 2 * sixA d + sixB d < 78', LEDGER)
        self.assertEqual(r.status, 'PASS')
        self.assertTrue(any(p['holds'] is False for p in r.probes))

    def test_unsupported_constructs_fail_closed(self):
        self.assertEqual(self.ev('Real.exp 1 < 3').status, 'UNSUPPORTED')
        self.assertEqual(self.ev('∀ x : Nat, x + 0 = x').status, 'UNSUPPORTED')   # unbounded domain
        self.assertEqual(self.ev('undefinedName 3 = 4').status, 'UNSUPPORTED')

    def test_ascription_inside_conjunction(self):
        r = self.ev('(1 : Int) + 15 = 2 * 8 ∧ (1 : Int) * 15 - (-3) * (-3) = 6 ∧ (8 : Int) ^ 2 - 58 = 6')
        self.assertEqual(r.status, 'PASS')

    def test_no_float_is_ever_produced(self):
        r = self.ev('10 * (125 ^ 20 * 2432902008176640000) < 10 ^ 62')
        self.assertEqual(r.status, 'PASS')
        for v in r.values:
            self.assertNotIsInstance(v, float)


P15 = '''
namespace P
def subsets2 : List (Bool × Bool) := [(false, false), (true, false), (false, true), (true, true)]
def good (U : Bool × Bool) : Bool := !(U.1 && U.2)
def weightNum (U : Bool × Bool) : Nat := (if U.1 then 1 else 2 - 1) * (if U.2 then 1 else 2 - 1)
def priceNum (U : Bool × Bool) : Nat := (if U.1 then 2 else 3) * (if U.2 then 2 else 3)
def totalMassNum : Nat := subsets2.foldl (fun acc U => acc + weightNum U) 0
def goodMassNum : Nat := (subsets2.filter good).foldl (fun acc U => acc + weightNum U) 0
def choose (n : Nat) : Nat → Nat
  | 0 => 1
  | k + 1 => choose n k * (n - k) / (k + 1)
def ceilDiv (a b : Nat) : Nat := (a + b - 1) / b
end P
'''


class P15ConstructTests(unittest.TestCase):
    def ev(self, text):
        module = sf.parse_module(P15 + f'\ntheorem t : {text} := by decide\n')
        return sf.evaluate_theorem(module, module.theorems[-1])

    def test_bool_pairs_filter_and_list_equality(self):
        self.assertEqual(self.ev('subsets2.filter good = [(false, false), (true, false), (false, true)]').status, 'PASS')
        self.assertEqual(self.ev('subsets2.filter (fun U => !good U) = [(true, true)]').status, 'PASS')
        self.assertEqual(self.ev('subsets2.filter good = [(false, false)]').status, 'FAIL')

    def test_if_then_else_and_folds_over_pairs(self):
        self.assertEqual(self.ev('totalMassNum = 2 ^ 2 ∧ goodMassNum * 4 = 3 * 2 ^ 2').status, 'PASS')
        self.assertEqual(self.ev('3 * 3 = 9 ∧ subsets2.map priceNum = [9, 6, 6, 4]').status, 'PASS')

    def test_builtin_function_as_higher_order_argument(self):
        self.assertEqual(self.ev('(subsets2.map priceNum).foldl min 9 = 4 ∧ 1 * 9 < 4 * 3').status, 'PASS')
        self.assertEqual(self.ev('(subsets2.map priceNum).foldl max 0 = 9').status, 'PASS')
        self.assertEqual(self.ev('(subsets2.map priceNum).foldl Nat.add 0 = 25').status, 'PASS')

    def test_pattern_def_with_leading_parameters(self):
        self.assertEqual(self.ev('choose 6 4 = 15').status, 'PASS')
        self.assertEqual(self.ev('6 * choose 409 2 = 500616 ∧ 15 * 409 ^ 4 = 419743994415').status, 'PASS')

    def test_tuple_domains_with_projections(self):
        r = self.ev('∀ p, p ∈ [(1, 1), (1, 408), (2, 3), (3, 5), (5, 2)] → ceilDiv (p.1 * p.2 + 1) p.1 = p.2 + 1')
        self.assertEqual(r.status, 'PASS')
        self.assertEqual(r.instances, 5)

    def test_quantifiers_inside_a_conjunction(self):
        r = self.ev('(∀ p, p ∈ [(1, 3), (1, 4), (2, 5)] → 4 ^ (p.1 + 1) * 27 ≤ 16 * 3 ^ p.2) ∧ '
                    '(∀ a, a ∈ [1, 2, 3, 4, 5, 6] → 4 * 4 ^ a * 27 ≤ 16 * (3 * 9 ^ a))')
        self.assertEqual(r.status, 'PASS')
        self.assertEqual(r.instances, 9)
        r2 = self.ev('1 * 2 + 1 = 3 ∧ ∀ p, p ∈ [(1, 2), (1, 3), (2, 2), (3, 5), (4, 2)] → 2 * p.1 + 1 ≤ p.1 * p.2 + 1')
        self.assertEqual(r2.status, 'PASS')
        r3 = self.ev('(∀ a, a ∈ [1, 2, 3] → a < 3)')
        self.assertEqual(r3.status, 'FAIL')
        self.assertIn('a = 3', r3.detail)


class ModuleTests(unittest.TestCase):
    def test_module_lists_defs_and_theorems_in_order_and_ignores_doc_text(self):
        m = sf.parse_module(LEDGER)
        self.assertEqual([t.name for t in m.theorems],
                         ['Demo.pairing_sum_six', 'Demo.pairing_sum_le_six', 'Demo.odd_block_eigenvalues',
                          'Demo.exponent_bounds', 'Demo.image_constant_value'])
        self.assertEqual(sorted(m.defs), ['fact', 'imageConstant', 'pairingSum', 'sixA', 'sixB'])
        self.assertEqual(m.namespace, 'Demo')

    def test_check_module_evaluates_every_theorem(self):
        results = sf.check_module(LEDGER)
        self.assertEqual({r.status for r in results}, {'PASS'})
        self.assertEqual(len(results), 5)


class MutationSelfTests(unittest.TestCase):
    def test_every_mutant_of_a_true_statement_is_detected(self):
        m = sf.parse_module(LEDGER)
        report = sf.mutation_selftest(m)
        self.assertGreaterEqual(report['mutants'], 10)
        self.assertEqual(report['survivors'], [])
        self.assertTrue(report['control_passed'])

    def test_truth_preserving_weakenings_must_pass_not_fail(self):
        # Codex finding 684/6510829073: 522 < 529 is true and so is 522 ≤ 529; a harness that rejects the
        # weakening has an evaluator defect. Weakenings are a separate class that must PASS.
        m = sf.parse_module('theorem t : 58 * 9 < 23 ^ 2 := by decide\n')
        report = sf.mutation_selftest(m)
        self.assertEqual(report['weakenings'], 1)
        self.assertEqual(report['weakenings_rejected'], [])
        self.assertEqual(report['mutants'], 1)          # the one guaranteed-false mutant: 58 * 9 ≥ 23 ^ 2
        self.assertTrue(report['harness_ok'])

    def test_declared_manifest_controls_must_fail(self):
        src = 'namespace D\ntheorem s : 58 * 9 < 23 ^ 2 := by decide\ntheorem u : 1 + 1 = 2 := by decide\nend D\n'
        controls = {
            'false_eigenvalue_third': {'module': 'D.lean', 'replace': '23 ^ 2', 'with': '22 ^ 2'},
            'sorry_injection': {'module': 'D.lean', 'replace': ':= by decide\ntheorem u', 'with': ':= by sorry\ntheorem u'},
            'not_applicable': {'module': 'E.lean', 'replace': 'x', 'with': 'y'},
        }
        report = sf.declared_controls(src, 'D.lean', controls)
        self.assertEqual(report['false_eigenvalue_third']['class'], 'statement')
        self.assertEqual(report['false_eigenvalue_third']['status'], 'FAIL')
        self.assertEqual(report['false_eigenvalue_third']['theorem'], 'D.s')
        self.assertEqual(report['sorry_injection']['class'], 'non-statement')   # proof text, not a statement edit
        self.assertNotIn('not_applicable', report)                              # another module
        self.assertTrue(report['_summary']['all_statement_controls_fail'])

    def test_unsupported_mutant_is_a_survivor_not_a_kill(self):
        m = sf.parse_module('theorem t : 1 + 1 = 2 := by decide\n')
        def evaluate(module, thm):
            if thm.name == 't' and isinstance(thm.statement, sf.BinOp) and thm.statement.op == '=':
                return sf.EvalResult('PASS', 'baseline', 1, False, [], [])
            return sf.EvalResult('UNSUPPORTED', 'cannot read the mutant', 0, False, [], [])
        report = sf.mutation_selftest(m, evaluate=evaluate)
        self.assertEqual(report['killed'], 0)
        self.assertIn('UNSUPPORTED', [x['status'] for x in report['survivors']])   # recorded, never a kill
        self.assertFalse(report['harness_ok'])

    def test_a_broken_evaluator_is_caught_by_the_control(self):
        m = sf.parse_module(LEDGER)
        report = sf.mutation_selftest(m, evaluate=lambda module, thm: sf.EvalResult('PASS', 'stub', 0, False, [], []))
        self.assertFalse(report['harness_ok'])


class PackageTests(unittest.TestCase):
    def test_main_formal_package_declared_controls_all_fail(self):
        report = sf.check_package(ROOT / 'formal', selftest=True)
        dc = report['declared_controls']
        statement_controls = {k: v for k, v in dc.items() if v['class'] == 'statement'}
        self.assertGreaterEqual(len(statement_controls), 4)
        self.assertEqual({v['status'] for v in statement_controls.values()}, {'FAIL'})
        self.assertEqual(report['exit_code'], 0)

    def test_main_formal_package_targets_all_pass(self):
        report = sf.check_package(ROOT / 'formal')
        registered = len(json.loads((ROOT / 'formal' / 'manifest.json').read_text())['targets'])
        self.assertGreaterEqual(registered, 31)
        self.assertEqual(report['targets'], registered)
        self.assertEqual(report['status_counts'].get('PASS'), registered)
        self.assertEqual(report['status_counts'].get('FAIL', 0), 0)
        self.assertEqual(report['status_counts'].get('UNSUPPORTED', 0), 0)
        self.assertEqual(sorted(report['per_target']), sorted(t['name'] for t in report['manifest_targets']))

    def test_mathlib_syntax_anywhere_in_a_module_does_not_crash_the_package_check(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            (root / 'Y').mkdir()
            (root / 'Y' / 'Core.lean').write_text(
                'import Mathlib\nnamespace Y\n'
                'theorem a : ∑ i ∈ Finset.range 3, (i : ℝ) = 3 := by\n  rw [← Finset.sum_range_succ]\n  norm_num\n'
                'theorem b : (2 : Nat) + 2 = 4 := by decide\n'
                'lemma c (x : ℝ) : x ^ 2 ≥ 0 := by positivity\nend Y\n')
            (root / 'manifest.json').write_text(json.dumps({'source_modules': ['Y/Core.lean'], 'targets': [
                {'name': 'Y.a', 'module': 'Y/Core.lean'}, {'name': 'Y.b', 'module': 'Y/Core.lean'},
                {'name': 'Y.c', 'module': 'Y/Core.lean'}]}))
            report = sf.check_package(root)
            self.assertEqual(report['per_target']['Y.a']['status'], 'UNSUPPORTED')
            self.assertEqual(report['per_target']['Y.b']['status'], 'PASS')
            # `(x : ℝ)` before the colon is sampled, never PASS: the package still fails on Y.a
            self.assertEqual(report['per_target']['Y.c']['status'], 'CORROBORATED')
            self.assertEqual(report['exit_code'], 1)

    def test_manifest_targets_may_be_bare_names(self):
        # Math-'s formal/manifest.json lists targets as qualified names without a module field
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            (root / 'Z').mkdir()
            (root / 'Z' / 'A.lean').write_text('namespace Z\ntheorem t : 2 + 2 = 4 := by decide\nend Z\n')
            (root / 'manifest.json').write_text(json.dumps({'source_modules': ['Z/A.lean'], 'targets': ['Z.t', 'Z.missing']}))
            report = sf.check_package(root)
            self.assertEqual(report['per_target']['Z.t']['status'], 'PASS')
            self.assertEqual(report['per_target']['Z.t']['module'], 'Z/A.lean')
            self.assertEqual(report['per_target']['Z.missing']['status'], 'UNSUPPORTED')

    def test_mathlib_statement_is_reported_unsupported_not_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            (root / 'X').mkdir()
            (root / 'X' / 'Real.lean').write_text('import Mathlib\nnamespace X\ntheorem r : ∀ x : ℝ, x ^ 2 ≥ 0 := by positivity\nend X\n')
            (root / 'manifest.json').write_text('{"source_modules": ["X/Real.lean"], "targets": [{"name": "X.r", "module": "X/Real.lean"}]}')
            report = sf.check_package(root)
            self.assertEqual(report['status_counts'], {'UNSUPPORTED': 1})
            self.assertEqual(report['exit_code'], 1)



D2 = '''
namespace Demo
def d2DetV (m2 m4 q : ℝ) : ℝ := (1 - 4 * q) * (m2 ^ 2 * m4) + 4 * q * ((m4 - m2 ^ 2) * (m4 + 3 * m2 ^ 2) / 4)
def d2Numerator (m2 m4 : ℝ) : ℝ := m2 ^ 2 * (m4 - m2 ^ 2)
def d2Schur (m2 m4 q : ℝ) : ℝ := d2Numerator m2 m4 / d2DetV m2 m4 q
end Demo
'''


class RealBinderTests(unittest.TestCase):
    """Sampled falsification of statements with real binders: exact rationals, Lean's totalized
    division and sqrt, hypothesis filtering, pathological boundary values, interval-decided sqrt."""

    def ev(self, binders, text, defs='', seed=1, samples=64):
        module = sf.parse_module(defs + f'\ntheorem t {binders} : {text} := by sorry\n')
        return sf.evaluate_theorem(module, module.theorems[-1], seed=seed, samples=samples)

    def test_true_real_identity_is_corroborated_with_sample_counts(self):
        r = self.ev('(a b : ℝ)', '(a + b) ^ 2 = a ^ 2 + 2 * a * b + b ^ 2')
        self.assertEqual(r.status, 'CORROBORATED')
        self.assertGreaterEqual(r.instances, 64)
        self.assertIn('admitted', r.detail)

    def test_false_real_identity_is_refuted_with_an_exact_witness(self):
        r = self.ev('(a b : ℝ)', '(a + b) ^ 2 = a ^ 2 + b ^ 2')
        self.assertEqual(r.status, 'FAIL')
        self.assertIn('a =', r.detail)
        self.assertIn('b =', r.detail)
        self.assertTrue(all(not isinstance(v, float) for v in r.values))

    def test_hypotheses_filter_samples(self):
        self.assertEqual(self.ev('(r : ℝ) (hr0 : 0 ≤ r) (hr1 : r ≤ 1)', 'r ^ 4 ≤ r ^ 3').status, 'CORROBORATED')
        self.assertEqual(self.ev('(r : ℝ) (hr0 : 0 ≤ r)', 'r ^ 4 ≤ r ^ 3').status, 'FAIL')     # r = 2

    def test_lean_real_division_is_totalized(self):
        self.assertEqual(self.ev('(x : ℝ)', 'x / 0 = 0').status, 'CORROBORATED')
        self.assertEqual(self.ev('(x : ℝ) (hx : x ≠ 0)', 'x / x = 1').status, 'CORROBORATED')
        r = self.ev('(x : ℝ)', 'x / x = 1')
        self.assertEqual(r.status, 'FAIL')
        self.assertIn('x = 0', r.detail)                                           # 0 / 0 = 0 ≠ 1

    def test_unsatisfiable_hypotheses_are_not_a_pass(self):
        r = self.ev('(x : ℝ) (h1 : x < 0) (h2 : 0 < x)', 'x = 1')
        self.assertEqual(r.status, 'UNSUPPORTED')
        self.assertIn('no sample', r.detail)

    def test_sqrt_is_exact_on_squares_and_totalized_on_negatives(self):
        self.assertEqual(self.ev('', '(1 : ℝ) = Real.sqrt 4 * Real.sqrt (1 / 4 : ℝ)').status, 'PASS')
        self.assertEqual(self.ev('', 'Real.sqrt (-1 : ℝ) = 0').status, 'PASS')
        self.assertEqual(self.ev('', '¬ (1 : ℝ) ≤ Real.sqrt 4 * (1 / 4 : ℝ)').status, 'PASS')

    def test_sqrt_of_a_non_square_is_decided_by_outward_rounded_intervals(self):
        self.assertEqual(self.ev('', 'Real.sqrt 2 < 3 / 2').status, 'PASS')
        self.assertEqual(self.ev('', 'Real.sqrt 2 = 3 / 2').status, 'FAIL')
        self.assertEqual(self.ev('', '(7 : ℝ) / 5 < Real.sqrt 2 ∧ Real.sqrt 2 < 141422 / 100000').status, 'PASS')

    def test_sqrt_identity_with_sampled_arguments_is_corroborated(self):
        r = self.ev('(r cW cZ q : ℝ) (hr : r ≠ 0) (hcZ : cZ ≠ 0)',
                    '(Real.sqrt cW * r ^ 2 * Real.sqrt q) / (cZ * r ^ 2) = (Real.sqrt cW / cZ) * Real.sqrt q')
        self.assertEqual(r.status, 'CORROBORATED')
        r2 = self.ev('(cW : ℝ)', 'Real.sqrt cW * Real.sqrt cW = cW')
        self.assertEqual(r2.status, 'FAIL')                                         # cW < 0: 0 ≠ cW

    def test_abs_max_and_ne(self):
        self.assertEqual(self.ev('', '|(-3 : ℝ)| = 3 ∧ max (1 : ℝ) 2 = 2').status, 'PASS')
        self.assertEqual(self.ev('(a b z t δ : ℝ) (hz : 0 < z) (ht : 0 < t) (ha : 0 ≤ a) (haz : a ≤ z) (hab : |a - b| ≤ δ) (hzt : |z - t| ≤ δ)',
                                 '|a / z - b / t| ≤ 2 * δ / t').status, 'CORROBORATED')

    def test_local_real_defs_with_parameters(self):
        self.assertEqual(self.ev('(m2 m4 : ℝ)', 'd2DetV m2 m4 0 = m2 ^ 2 * m4', D2).status, 'CORROBORATED')
        self.assertEqual(self.ev('', 'd2DetV 1 1 (1 / 4) = 0 ∧ d2Schur 1 1 (1 / 4) = 0 ∧ d2Numerator 1 1 / d2DetV 1 1 (1 / 4) = 0', D2).status, 'PASS')
        self.assertEqual(self.ev('(m2 m4 q : ℝ) (hm2 : 0 < m2) (hm4 : m2 ^ 2 < m4) (hq0 : 0 ≤ q) (hq1 : q ≤ 1 / 4)',
                                 '0 < d2DetV m2 m4 q', D2).status, 'CORROBORATED')

    def test_non_numeric_binders_stay_unsupported(self):
        for b in ('(μ : Measure Ω)', '(f : Ω → ℝ)', '[IsProbabilityMeasure μ] (μ : Measure Ω)', '(A : Set Ω)'):
            with self.subTest(b=b):
                self.assertEqual(self.ev(b, '(1 : ℝ) = 1').status, 'UNSUPPORTED')

    def test_boundary_values_from_hypotheses_are_sampled(self):
        # true only strictly inside (0, 1): fails at the boundary r = 1 exactly, which the sampler must try
        r = self.ev('(r : ℝ) (hr0 : 0 ≤ r) (hr1 : r ≤ 1)', 'r ^ 4 < r ^ 3')
        self.assertEqual(r.status, 'FAIL')
        self.assertTrue('r = 1' in r.detail or 'r = 0' in r.detail)

    def test_sampling_is_deterministic_for_a_seed(self):
        a = self.ev('(a b : ℝ)', '(a + b) ^ 2 = a ^ 2 + 2 * a * b + b ^ 2', seed=7)
        b = self.ev('(a b : ℝ)', '(a + b) ^ 2 = a ^ 2 + 2 * a * b + b ^ 2', seed=7)
        self.assertEqual((a.status, a.instances, a.detail), (b.status, b.instances, b.detail))

    def test_mutation_contract_applies_to_sampled_statements(self):
        module = sf.parse_module(D2 + '\ntheorem t (m2 m4 : ℝ) : d2DetV m2 m4 0 = m2 ^ 2 * m4 := by ring\n'
                                 '\ntheorem u (r : ℝ) (hr0 : 0 ≤ r) (hr1 : r ≤ 1) : r ^ 4 ≤ r ^ 3 := by nlinarith\n')
        report = sf.mutation_selftest(module)
        self.assertTrue(report['control_passed'])
        self.assertEqual(report['killed'], report['mutants'])
        self.assertGreaterEqual(report['mutants'], 2)
        self.assertEqual(report['weakenings_rejected'], [])
        self.assertTrue(report['harness_ok'])


class ExactRealAlgebraTests(unittest.TestCase):
    """Closed statements over ℝ are decided exactly in ℚ(√d₁, …, √dₖ): never a float, never an approximation."""

    def ev(self, text, defs=''):
        module = sf.parse_module(defs + f'\ntheorem t : {text} := by norm_num\n')
        return sf.evaluate_theorem(module, module.theorems[-1])

    def test_quadratic_identities_and_inverses_are_exact(self):
        for text in ('Real.sqrt 2 * Real.sqrt 2 = 2', 'Real.sqrt 2 + Real.sqrt 8 = 3 * Real.sqrt 2',
                     '1 / (Real.sqrt 2 + Real.sqrt 3) = Real.sqrt 3 - Real.sqrt 2',
                     '(Real.sqrt 2 + Real.sqrt 3) ^ 2 = 5 + 2 * Real.sqrt 6',
                     '(Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 30)⁻¹ * (Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 30) = 1',
                     'Real.sqrt (Real.sqrt 2 * Real.sqrt 2) = Real.sqrt 2', '√2 * √2 = 2 ∧ √(2 : ℝ) ^ 2 = 2',
                     'Real.sqrt (1 / 2) = Real.sqrt 2 / 2', 'Real.sqrt 12 * Real.sqrt 3 = 6'):
            with self.subTest(text=text):
                r = self.ev(text)
                self.assertEqual(r.status, 'PASS', r.detail)
                self.assertFalse(any(isinstance(v, float) for v in r.values))

    def test_orderings_near_zero_are_decided_by_refinement_not_floats(self):
        self.assertEqual(self.ev('Real.sqrt 2 + Real.sqrt 3 < Real.sqrt 10').status, 'PASS')     # 3.1462… < 3.1622…
        self.assertEqual(self.ev('Real.sqrt 2 + Real.sqrt 3 > Real.sqrt 10').status, 'FAIL')
        self.assertEqual(self.ev('Real.sqrt 2 > 1.41421356 ∧ Real.sqrt 2 < 1.41421357').status, 'PASS')
        self.assertEqual(self.ev('Real.sqrt 1000001 - 1000 > 1 / 2001 ∧ Real.sqrt 1000001 - 1000 < 1 / 2000').status, 'PASS')

    def test_scientific_literals_need_a_field_context(self):
        self.assertEqual(self.ev('(0.5 : ℝ) = 1 / 2').status, 'PASS')
        self.assertEqual(self.ev('(1 : ℝ) / 3 = 0.3333').status, 'FAIL')
        self.assertEqual(self.ev('0.5 = 1 / 2').status, 'UNSUPPORTED')    # Lean would default 0.5 to Float

    def test_exponent_typing_follows_lean(self):
        self.assertEqual(self.ev('(2 : ℝ) ^ (-2 : ℤ) = 1 / 4 ∧ (0 : ℝ) ^ (-2 : ℤ) = 0').status, 'PASS')   # zpow, 0⁻¹ = 0
        self.assertEqual(self.ev('(9 : ℝ) ^ (1 / 2) = 1').status, 'PASS')          # the exponent 1/2 is a Nat: 0
        self.assertEqual(self.ev('(9 : ℝ) ^ (1 / 2 : ℝ) = 3').status, 'UNSUPPORTED')   # rpow

    def test_transcendental_and_nested_radicals_stay_unsupported(self):
        self.assertEqual(self.ev('Real.exp 0 = 1').status, 'UNSUPPORTED')
        self.assertEqual(self.ev('Real.sqrt (Real.sqrt 2) < 2').status, 'UNSUPPORTED')
        self.assertIn('nested radical', self.ev('Real.sqrt (Real.sqrt 2) < 2').detail)

    def test_roots_of_radicands_beyond_exact_factoring_stay_exact_but_never_guess_zero(self):
        # 10^47 + 3 cannot be split into square and squarefree parts within the deterministic primality range;
        # its root is kept exactly, orderings are decided by enclosures, and an exact zero is refused
        big = '(10 ^ 47 + 3 : ℝ)'
        self.assertEqual(self.ev(f'Real.sqrt {big} > 316227766016837933199889 ∧ Real.sqrt {big} < 316227766016837933199890').status, 'PASS')
        self.assertEqual(self.ev(f'Real.sqrt {big} = Real.sqrt {big}').status, 'PASS')
        self.assertEqual(self.ev(f'Real.sqrt (2 * {big}) = Real.sqrt 2 * Real.sqrt {big}').status, 'PASS')
        self.assertEqual(self.ev(f'Real.sqrt {big} = 316227766016837933199889').status, 'FAIL')
        r = self.ev(f'Real.sqrt ((10 ^ 6 + 3) ^ 2 * {big}) = (10 ^ 6 + 3) * Real.sqrt {big}')   # true, but not recognisable
        self.assertEqual(r.status, 'UNSUPPORTED')
        self.assertIn('unsplit radicand', r.detail)

    def test_ascription_scopes_the_elaboration_type(self):
        # binop% pushes the cast to the leaves: `((7 : ℕ) / 2 : ℝ)` is `↑7 / 2 = 7/2`; only an inner ascription
        # on the quotient keeps Nat division
        self.assertEqual(self.ev('((7 : ℕ) / 2 : ℝ) = 7 / 2 ∧ (7 / 2 : ℝ) = 7 / 2 ∧ ((7 / 2 : ℕ) : ℝ) = 3').status, 'PASS')
        self.assertEqual(self.ev('((7 : ℕ) / 2 : ℝ) = 3').status, 'FAIL')


class SampledBinderSemanticsTests(unittest.TestCase):
    def ev(self, binders, text, defs='', seed=1, samples=64):
        module = sf.parse_module(defs + f'\ntheorem t {binders} : {text} := by sorry\n')
        return sf.evaluate_theorem(module, module.theorems[-1], seed=seed, samples=samples)

    def test_nat_and_int_binders_are_sampled_with_lean_semantics(self):
        self.assertEqual(self.ev('(n : ℕ)', 'n * (n + 1) % 2 = 0').status, 'CORROBORATED')
        self.assertEqual(self.ev('(n : ℤ)', 'n ^ 2 ≥ n').status, 'CORROBORATED')
        r = self.ev('(n : ℤ)', 'n ^ 3 ≥ n')
        self.assertEqual(r.status, 'FAIL')
        self.assertIn('n = -', r.detail)
        self.assertEqual(self.ev('(a b : ℤ) (hb : b ≠ 0)', 'a = b * (a / b) + a % b ∧ 0 ≤ a % b ∧ a % b < |b|').status, 'CORROBORATED')
        self.assertEqual(self.ev('(n : ℕ)', '((n - 1 : ℕ) : ℝ) = (n : ℝ) - 1').status, 'FAIL')       # truncation at n = 0
        self.assertEqual(self.ev('(n : ℕ)', '(n - 1 : ℝ) = (n : ℝ) - 1').status, 'CORROBORATED')

    def test_infeasible_instances_are_skipped_and_counted_not_approximated(self):
        r = self.ev('(n : ℕ)', '2 ^ n > n')
        self.assertEqual(r.status, 'CORROBORATED')
        self.assertGreater(r.sampling['infeasible'], 0)       # n = 2^63 and friends: no 2^(2^63) is attempted
        self.assertEqual(self.ev('(n : ℕ)', 'Nat.factorial n ≥ 1').status, 'CORROBORATED')
        self.assertEqual(self.ev('', '2 ^ (2 ^ 40) > 0').status, 'UNSUPPORTED')    # closed: infeasible is fail-closed

    def test_interval_membership_hypotheses_desugar(self):
        self.assertEqual(self.ev('(x : ℝ) (hx : x ∈ Set.Icc (0 : ℝ) 1)', 'x ^ 2 ≤ x').status, 'CORROBORATED')
        self.assertEqual(self.ev('(x : ℝ) (hx : x ∈ Set.Icc (0 : ℝ) 1)', 'x ^ 2 < x').status, 'FAIL')
        self.assertEqual(self.ev('(x : ℝ) (hx : x ∈ Set.Ioo (0 : ℝ) 1)', 'x ^ 2 < x').status, 'CORROBORATED')

    def test_hypothesis_boundaries_between_variables_are_found(self):
        self.assertEqual(self.ev('(x y : ℝ) (h : x < y)', 'x < (x + y) / 2 ∧ (x + y) / 2 < y').status, 'CORROBORATED')
        r = self.ev('(x y : ℝ) (h : x ≤ y)', 'x < (x + y) / 2')
        self.assertEqual(r.status, 'FAIL')                     # x = y

    def test_inner_finite_quantifier_under_a_sampled_binder(self):
        self.assertEqual(self.ev('(n : ℕ)', '∀ k, k ∈ [0, 1, 2] → k ≤ n + 2').status, 'CORROBORATED')
        self.assertEqual(self.ev('(n : ℕ)', '∀ k : ℕ, k ≤ 3 → k ≤ n + 3').status, 'CORROBORATED')
        r = self.ev('(n : ℕ)', '∀ k : ℕ, k ≤ 3 → k ≤ n + 2')
        self.assertEqual(r.status, 'FAIL')                     # n = 0, k = 3
        self.assertIn('n = 0', r.detail)
        self.assertEqual(self.ev('(ε : ℝ) (hε : 0 < ε)', '∃ n, 1 / n < ε').status, 'UNSUPPORTED')

    def test_undefined_predicates_in_hypotheses_name_the_binder(self):
        r = self.ev('(p : ℕ) (hp : Nat.Prime p)', 'p ≥ 2')
        self.assertEqual(r.status, 'UNSUPPORTED')
        self.assertIn('hp', r.detail)
        r = self.ev('(x : ℝ) (hx : Real.exp x > 0)', 'x = x')
        self.assertEqual(r.status, 'UNSUPPORTED')
        self.assertIn('Real.exp', r.detail)

    def test_type_tokens_inside_a_binder_type_are_types_not_hypotheses(self):
        for b in ('(x : ℝ≥0)', '(n : ℕ+)', '(v : Fin 3 → ℝ)'):
            with self.subTest(b=b):
                r = self.ev(b, '(1 : ℝ) = 1')
                self.assertEqual(r.status, 'UNSUPPORTED')
                self.assertIn('neither a sampled numeric type', r.detail)

    def test_a_comparison_of_scientific_literals_alone_has_no_exact_type(self):
        # `1.5 = 1.5` elaborates at Float in Lean; the interpreter refuses rather than reading it as ℝ
        r = self.ev('', '1.5 = 1.5')
        self.assertEqual(r.status, 'UNSUPPORTED')
        self.assertIn('Float', r.detail)
        self.assertEqual(self.ev('(x : ℝ)', 'x * 1.5 = 3 * x / 2').status, 'CORROBORATED')

    def test_abs_is_an_application_argument_only_without_whitespace(self):
        self.assertEqual(self.ev('(x : ℝ)', 'max |x| 0 = |x|').status, 'CORROBORATED')
        self.assertEqual(self.ev('(x y : ℝ)', '|x * y| = |x| * |y| ∧ |x + y| ≤ |x| + |y|').status, 'CORROBORATED')

    def test_sampled_witness_is_exact_and_hypotheses_hold_at_it(self):
        r = self.ev('(x : ℝ) (hx : 0 < x)', 'x ^ 2 ≤ x')
        self.assertEqual(r.status, 'FAIL')
        self.assertEqual(r.sampling['witness']['x'], '2')
        self.assertFalse(any(isinstance(v, float) for v in r.values))


class AdmitCorroboratedPackageTests(unittest.TestCase):
    def test_exit_code_is_strict_by_default_and_admits_sampled_verdicts_on_request(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            (root / 'W').mkdir()
            (root / 'W' / 'R.lean').write_text('import Mathlib\nnamespace W\n'
                                              'theorem a (x : ℝ) : x ^ 2 ≥ 0 := by positivity\n'
                                              'theorem b : (2 : ℝ) + 2 = 4 := by norm_num\nend W\n')
            (root / 'manifest.json').write_text(json.dumps({'source_modules': ['W/R.lean'], 'targets': ['W.a', 'W.b']}))
            strict = sf.check_package(root)
            self.assertEqual(strict['status_counts'], {'CORROBORATED': 1, 'PASS': 1})
            self.assertEqual(strict['exit_code'], 1)                    # the registered contract: PASS only
            self.assertEqual(strict['per_target']['W.a']['sampling']['accepted'], 64)
            admitting = sf.check_package(root, admit_corroborated=True)
            self.assertEqual(admitting['exit_code'], 0)
            self.assertTrue(admitting['sampling']['admit_corroborated'])
            totals = admitting['sampling_totals']          # a8's #742 nit: infeasible draws are reported, per package
            self.assertEqual(totals['sampled_targets'], 1)
            self.assertEqual(totals['accepted_total'], 64)
            self.assertEqual(totals['infeasible_total'], 0)
            self.assertEqual(totals['targets_with_infeasible_draws'], [])
            self.assertEqual(sf.main(['--package', str(root)]), 1)
            self.assertEqual(sf.main(['--package', str(root), '--admit-corroborated', '--seed', '3', '--samples', '16']), 0)


FM = '''
import Mathlib
namespace FM
open MeasureTheory
open scoped ENNReal
variable {Ω : Type*} [MeasurableSpace Ω]

noncomputable def weightedLaw (μ : Measure Ω) (W : Ω → ℝ) : Measure Ω :=
  (ENNReal.ofReal (∫ x, W x ∂μ))⁻¹ •
    μ.withDensity (fun x => ENNReal.ofReal (W x))

def badJetEvent (R : Ω → ℝ) (r ε : ℝ) : Set Ω :=
  {x | ε < r * R x ^ 5}
end FM
'''


class FiniteModelTests(unittest.TestCase):
    """Measure-theoretic statements are tested on finite discrete models: Ω = Fin n, nonnegative rational
    weights, functions as tables, sets as subsets. Integrals are finite sums; a.e. relations hold off the
    null points; measurability and integrability are automatic; measure equality and ≪ are pointwise."""

    def ev(self, binders, text, seed=1, samples=64):
        module = sf.parse_module(FM + f'\ntheorem t {binders} : {text} := by sorry\n')
        return sf.evaluate_theorem(module, module.theorems[-1], seed=seed, samples=samples)

    MARKOV = ('(μ : Measure Ω) [IsFiniteMeasure μ] (Y : Ω → ℝ) (A : Set Ω) (hY : Integrable Y μ) (hY0 : 0 ≤ᵐ[μ] Y) '
              '(a : ℝ) (ha : 0 < a) (hsub : A ⊆ {x | a ≤ Y x})')

    def test_markov_event_is_corroborated_and_its_misstatements_refuted(self):
        r = self.ev(self.MARKOV, '(μ A).toReal ≤ (∫ x, Y x ∂μ) / a')
        self.assertEqual(r.status, 'CORROBORATED', r.detail)
        self.assertGreaterEqual(r.instances, 64)
        self.assertEqual(self.ev(self.MARKOV, '(μ A).toReal ≤ (∫ x, Y x ∂μ) / (2 * a)').status, 'FAIL')
        r = self.ev('(μ : Measure Ω) [IsFiniteMeasure μ] (Y : Ω → ℝ) (A : Set Ω) (hY0 : 0 ≤ᵐ[μ] Y) (a : ℝ) (ha : 0 < a)',
                    '(μ A).toReal ≤ (∫ x, Y x ∂μ) / a')          # without A ⊆ {a ≤ Y}: false
        self.assertEqual(r.status, 'FAIL')
        self.assertIn('μ =', r.detail)
        self.assertIn('A =', r.detail)
        self.assertIn('Fin', r.detail)

    def test_integral_cauchy_schwarz_with_exact_square_roots(self):
        cs = '(μ : Measure Ω) (f g : Ω → ℝ) (hf : MemLp f 2 μ) (hg : MemLp g 2 μ) (hf0 : 0 ≤ᵐ[μ] f) (hg0 : 0 ≤ᵐ[μ] g)'
        self.assertEqual(self.ev(cs, '(∫ x, f x * g x ∂μ) ≤ Real.sqrt (∫ x, f x ^ 2 ∂μ) * Real.sqrt (∫ x, g x ^ 2 ∂μ)').status,
                         'CORROBORATED')
        self.assertEqual(self.ev(cs, '(∫ x, f x * g x ∂μ) < Real.sqrt (∫ x, f x ^ 2 ∂μ) * Real.sqrt (∫ x, g x ^ 2 ∂μ)').status,
                         'FAIL')                                   # equality at f = g, or on one point

    def test_set_integral_and_event_mass(self):
        ev_cs = ('(μ : Measure Ω) [IsProbabilityMeasure μ] (W : Ω → ℝ) (A : Set Ω) (hW : MemLp W 2 μ) (hW0 : 0 ≤ᵐ[μ] W) '
                 '(hA : MeasurableSet A)')
        self.assertEqual(self.ev(ev_cs, '(∫ x in A, W x ∂μ) ≤ Real.sqrt (∫ x, W x ^ 2 ∂μ) * Real.sqrt ((μ A).toReal)').status,
                         'CORROBORATED')
        self.assertEqual(self.ev(ev_cs, '0 ≤ (∫ x in A, W x ∂μ) ∧ (∫ x in A, W x ∂μ) ≤ ∫ x, W x ∂μ').status, 'CORROBORATED')
        self.assertEqual(self.ev('(μ : Measure Ω) [IsProbabilityMeasure μ] (W : Ω → ℝ) (A : Set Ω)',
                                 '(∫ x in A, W x ∂μ) ≤ ∫ x, W x ∂μ').status, 'FAIL')     # W may be negative off A

    def test_weighted_law_apply_needs_the_ae_nonnegativity_and_ennreal_semantics(self):
        apply = 'weightedLaw μ W A = (ENNReal.ofReal (∫ x, W x ∂μ))⁻¹ * ENNReal.ofReal (∫ x in A, W x ∂μ)'
        self.assertEqual(self.ev('(μ : Measure Ω) (W : Ω → ℝ) (A : Set Ω) (hW : Integrable W μ) (hW0 : 0 ≤ᵐ[μ] W) (hA : MeasurableSet A)',
                                 apply).status, 'CORROBORATED')
        # `ofReal` clips negative values: with a signed W the two sides differ (∞ · 0 = 0 on one side only)
        self.assertEqual(self.ev('(μ : Measure Ω) (W : Ω → ℝ) (A : Set Ω) (hW : Integrable W μ) (hA : MeasurableSet A)',
                                 apply).status, 'FAIL')

    def test_weighted_law_is_a_probability_measure_zero_one_and_absolute_continuity(self):
        self.assertEqual(self.ev('(μ : Measure Ω) (W : Ω → ℝ) (hW : Integrable W μ) (hW0 : 0 ≤ᵐ[μ] W) (hz : 0 < ∫ x, W x ∂μ)',
                                 'IsProbabilityMeasure (weightedLaw μ W)').status, 'CORROBORATED')
        self.assertEqual(self.ev('(μ : Measure Ω) (W : Ω → ℝ) (hW : Integrable W μ) (hW0 : 0 ≤ᵐ[μ] W)',
                                 'IsProbabilityMeasure (weightedLaw μ W)').status, 'FAIL')      # ∫ W = 0
        self.assertEqual(self.ev('(μ : Measure Ω)', 'weightedLaw μ (fun _ : Ω => (0 : ℝ)) = 0').status, 'CORROBORATED')
        self.assertEqual(self.ev('(μ : Measure Ω) [IsProbabilityMeasure μ]', 'weightedLaw μ (fun _ : Ω => (1 : ℝ)) = μ').status,
                         'CORROBORATED')
        self.assertEqual(self.ev('(μ : Measure Ω)', 'weightedLaw μ (fun _ : Ω => (1 : ℝ)) = μ').status, 'FAIL')
        self.assertEqual(self.ev('(μ : Measure Ω) (W : Ω → ℝ)', 'weightedLaw μ W ≪ μ').status, 'CORROBORATED')
        self.assertEqual(self.ev('(μ : Measure Ω) (W V : Ω → ℝ) (h : W =ᵐ[μ] V)', 'weightedLaw μ W = weightedLaw μ V').status,
                         'CORROBORATED')
        self.assertEqual(self.ev('(μ : Measure Ω) (W V : Ω → ℝ)', 'weightedLaw μ W = weightedLaw μ V').status, 'FAIL')

    def test_set_builder_events_and_subsets(self):
        self.assertEqual(self.ev('(R : Ω → ℝ) (r ε : ℝ) (hr : 0 < r) (hε : 0 < ε)',
                                 'badJetEvent R r ε ⊆ {x | (ε / r) ^ 8 ≤ R x ^ 40}').status, 'CORROBORATED')
        self.assertEqual(self.ev('(R : Ω → ℝ) (r ε : ℝ) (hr : 0 < r) (hε : 0 < ε)',
                                 'badJetEvent R r ε ⊆ {x | 1 ≤ R x}').status, 'FAIL')
        tail = ('(μ : Measure Ω) [IsProbabilityMeasure μ] (R : Ω → ℝ) (r ε M : ℝ) (hr : 0 < r) (hε : 0 < ε) '
                '(hRi : Integrable (fun x => R x ^ 40) μ) (hMbound : (∫ x, R x ^ 40 ∂μ) ≤ M)')
        self.assertEqual(self.ev(tail, '(μ (badJetEvent R r ε)).toReal ≤ (M / ε ^ 8) * r ^ 8').status, 'CORROBORATED')
        # without the moment bound on M the tail bound is false (M = 0 under a nonempty event)
        no_bound = ('(μ : Measure Ω) [IsProbabilityMeasure μ] (R : Ω → ℝ) (r ε M : ℝ) (hr : 0 < r) (hε : 0 < ε) '
                    '(hRi : Integrable (fun x => R x ^ 40) μ)')
        self.assertEqual(self.ev(no_bound, '(μ (badJetEvent R r ε)).toReal ≤ (M / ε ^ 8) * r ^ 8').status, 'FAIL')

    def test_structural_hypotheses_are_not_assumed_on_infinite_mass_measures(self):
        # `(ENNReal.ofReal 0)⁻¹ • μ` has infinite mass wherever μ does not vanish: finiteness is decided false and
        # integrability is not taken for granted (a8, 743/5729046258 N2)
        self.assertEqual(self.ev('(μ : Measure Ω) [IsProbabilityMeasure μ]',
                                 'IsFiniteMeasure ((ENNReal.ofReal 0)⁻¹ • μ)').status, 'FAIL')
        self.assertEqual(self.ev('(μ : Measure Ω) [IsProbabilityMeasure μ]', 'IsFiniteMeasure ((2 : ENNReal) • μ)').status,
                         'CORROBORATED')
        r = self.ev('(μ : Measure Ω) [IsProbabilityMeasure μ] (W : Ω → ℝ) (h : Integrable W ((ENNReal.ofReal 0)⁻¹ • μ))',
                    '0 ≤ ∫ x, W x ∂μ')
        self.assertEqual(r.status, 'UNSUPPORTED')
        self.assertIn('infinite mass', r.detail)

    def test_quantifier_over_all_subsets_is_enumerated(self):
        r = self.ev('(μ : Measure Ω)', '∀ A : Set Ω, MeasurableSet A → (μ A).toReal ≤ (μ Set.univ).toReal')
        self.assertEqual(r.status, 'CORROBORATED')
        self.assertEqual(self.ev('(μ : Measure Ω)', '∀ A : Set Ω, MeasurableSet A → (μ A).toReal < (μ Set.univ).toReal').status, 'FAIL')

    def test_structural_conclusions_and_real_indexed_families_stay_unsupported(self):
        r = self.ev('(R : Ω → ℝ) (hR : Measurable R) (r ε : ℝ)', 'MeasurableSet (badJetEvent R r ε)')
        self.assertEqual(r.status, 'UNSUPPORTED')
        self.assertIn('trivial on finite discrete models', r.detail)
        r = self.ev('(μ : ℝ → Measure Ω) [∀ r, IsProbabilityMeasure (μ r)] (W : ℝ → Ω → ℝ) (M : ℝ)', '0 ≤ M')
        self.assertEqual(r.status, 'UNSUPPORTED')
        self.assertIn('μ : ℝ → Measure Ω', r.detail)

    def test_model_witness_is_exact_and_sampling_is_seeded(self):
        a = self.ev('(μ : Measure Ω) (W : Ω → ℝ)', '0 ≤ ∫ x, W x ∂μ', seed=5)
        b = self.ev('(μ : Measure Ω) (W : Ω → ℝ)', '0 ≤ ∫ x, W x ∂μ', seed=5)
        self.assertEqual(a.status, 'FAIL')
        self.assertEqual((a.detail, a.instances), (b.detail, b.instances))
        self.assertIn('W = [', a.detail)
        self.assertFalse(any(isinstance(v, float) for v in a.values))

    def test_mutation_contract_holds_on_finite_model_statements(self):
        module = sf.parse_module(FM + '\ntheorem t ' + self.MARKOV + ' : (μ A).toReal ≤ (∫ x, Y x ∂μ) / a := by sorry\n')
        report = sf.mutation_selftest(module)
        self.assertTrue(report['control_passed'])
        self.assertEqual(report['killed'], report['mutants'])
        self.assertGreaterEqual(report['mutants'], 1)
        self.assertTrue(report['harness_ok'])


if __name__ == '__main__':
    unittest.main()
