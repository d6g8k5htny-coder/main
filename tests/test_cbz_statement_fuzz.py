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
        self.assertEqual(report['targets'], 31)
        self.assertEqual(report['status_counts'].get('PASS'), 31)
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
            self.assertEqual(report['per_target']['Y.c']['status'], 'UNSUPPORTED')
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


if __name__ == '__main__':
    unittest.main()
