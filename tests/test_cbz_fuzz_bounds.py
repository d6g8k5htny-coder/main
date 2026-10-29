"""Tests for the CBZ-0 empirical fuzzing harness (gate G6, fuzz part). Scientific effect: NONE.

CBZ_FUZZ_BOUNDS may name an alternative harness file; the mutation harness
(gate G0) uses it to run this suite against each mutant.
"""
import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = Path(os.environ.get('CBZ_FUZZ_BOUNDS', ROOT / 'tools/cbz/fuzz_bounds.py'))
sys.path.insert(0, str(TARGET.parent))
spec = importlib.util.spec_from_file_location('cbz_fuzz_bounds_under_test', TARGET)
fuzz = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fuzz)


def statuses(text, samples=300):
    return {name: record['status'] for name, record in fuzz.fuzz_text(text, samples=samples).items()}


def one(statement, samples=300, prelude=''):
    record = fuzz.fuzz_text(prelude + 'theorem t ' + statement + ' := by decide\n', samples=samples)['t']
    return record


class LeanSemantics(unittest.TestCase):
    """Exact evaluation follows Lean, not Python."""

    def check(self, cases):
        for statement, expected in cases:
            with self.subTest(statement=statement):
                self.assertEqual(one(statement)['status'], expected)

    def test_nat_subtraction_truncates(self):
        self.check([(': (3 : Nat) - 5 = 0', 'PASS-exact'), (': (3 : Int) - 5 = -2', 'PASS-exact'),
                    (': (3 : Int) - 5 = 0', 'FAIL')])

    def test_division_conventions(self):
        self.check([(': (7 : Nat) / 2 = 3', 'PASS-exact'), (': (-7 : Int) / 2 = -4', 'PASS-exact'),
                    (': (-7 : Int) / 2 = -3', 'FAIL'), (': (7 : Int) / (-2) = -3', 'PASS-exact'),
                    (': (5 : Nat) / 0 = 0', 'PASS-exact'), (': (1 : ℝ) / 0 = 0', 'PASS-exact'),
                    (': (1 : ℚ) / 3 + 1 / 3 + 1 / 3 = 1', 'PASS-exact'), (': (1 : Nat) / 3 + 1 / 3 + 1 / 3 = 1', 'FAIL')])

    def test_modulo_conventions(self):
        self.check([(': (7 : Int) % (-2) = 1', 'PASS-exact'), (': (-7 : Int) % 2 = 1', 'PASS-exact'),
                    (': (7 : Nat) % 0 = 7', 'PASS-exact')])

    def test_precedence_and_associativity(self):
        self.check([(': 2 + 3 * 4 = 14', 'PASS-exact'), (': 2 ^ 3 ^ 2 = 512', 'PASS-exact'),
                    (': (2 ^ 3) ^ 2 = 64', 'PASS-exact'), (': (-2 ^ 2 : Int) = -4', 'PASS-exact'),
                    (': ((-2) ^ 2 : Int) = 4', 'PASS-exact'), (': 10 - 3 - 2 = 5', 'PASS-exact'),
                    (': 2 * 3 ^ 2 = 18', 'PASS-exact'), (': 100 / 10 / 5 = 2', 'PASS-exact')])

    def test_mixed_leaf_types_elaborate_at_the_maximal_type(self):
        self.check([(': (2 : Nat) - 3 + (1 : Int) = 0', 'PASS-exact'), (': (2 : Nat) - 3 + 1 = 1', 'PASS-exact')])

    def test_logic(self):
        self.check([(': 1 < 2 ∧ 2 < 3', 'PASS-exact'), (': 2 < 1 ∨ 1 = 1', 'PASS-exact'),
                    (': ¬ (1 = 2)', 'PASS-exact'), (': (1 = 2) → False', 'PASS-exact'),
                    (': (1 = 1) ↔ (2 = 2)', 'PASS-exact'), (': 1 ≠ 1', 'FAIL'), (': 3 ∣ 12 ∧ ¬ (5 ∣ 12)', 'PASS-exact')])

    def test_finite_quantifiers_are_exhaustive(self):
        self.check([(': ∀ q, q ∈ [0, 1, 2] → q * q ≤ 4', 'PASS-exact'), (': ∀ q, q ∈ [0, 1, 3] → q * q ≤ 4', 'FAIL'),
                    (': ∀ n < 10, n * n < 100', 'PASS-exact'), (': ∀ n ≤ 10, n * n < 100', 'FAIL'),
                    (': ∀ n ∈ Finset.range 5, n < 5', 'PASS-exact'), (': ∃ n ∈ Finset.range 5, n = 4', 'PASS-exact'),
                    (': ∃ n ∈ Finset.range 4, n = 4', 'FAIL')])

    def test_big_operators_and_builtins(self):
        self.check([(': ∑ i ∈ Finset.range 5, i = 10', 'PASS-exact'), (': ∏ i ∈ Finset.range 4, (i + 1) = 24', 'PASS-exact'),
                    (': ∑ i in Finset.range 5, i = 11', 'FAIL'), (': 5 ! = 120', 'PASS-exact'),
                    (': Nat.choose 5 2 = 10', 'PASS-exact'), (': |(-3 : Int)| = 3', 'PASS-exact'),
                    (': Nat.gcd 12 18 = 6', 'PASS-exact'), (': min 3 5 = 3 ∧ max 3 5 = 5', 'PASS-exact')])

    def test_set_intervals(self):
        self.check([(': (1 / 2 : ℝ) ∈ Set.Icc 0 1', 'PASS-exact'), (': (2 : ℝ) ∈ Set.Ioo 0 2', 'FAIL'),
                    (': (2 : ℝ) ∈ Set.Ioc 0 2', 'PASS-exact'), (': (0 : ℝ) ∉ Set.Ioi 0', 'PASS-exact')])

    def test_decimal_literals_are_exact(self):
        self.check([(': (0.1 : ℝ) + 0.2 = 0.3', 'PASS-exact'), (': (0.1 : ℚ) * 3 = 0.3', 'PASS-exact')])


class Sampling(unittest.TestCase):
    """Infinite domains are searched with pathological edges first."""

    def test_true_universal_statements_pass(self):
        self.assertEqual(one('(x : ℝ) : x ^ 2 ≥ 0')['status'], 'PASS-sampled')
        record = one('(x y : ℝ) : (x + y) ^ 2 = x ^ 2 + 2 * x * y + y ^ 2')
        self.assertEqual(record['status'], 'PASS-sampled')
        self.assertEqual(record['evaluated'], 300)

    def test_edge_counterexamples_are_found(self):
        record = one('(n : Nat) : n ≠ 2 ^ 32 + 7')
        self.assertEqual(record['status'], 'FAIL')
        self.assertEqual(record['counterexample'], {'n': str(2 ** 32 + 7)})
        self.assertEqual(one('(x : ℝ) : x ≠ 1 / 3')['status'], 'FAIL')
        self.assertEqual(one('(n : Nat) : n < 10 ^ 30')['status'], 'FAIL')
        self.assertEqual(one('(z : Int) : z ≥ -1')['status'], 'FAIL')
        self.assertEqual(one('(x : ℝ) : x < 10 ^ 12')['status'], 'FAIL')
        self.assertEqual(one('(x : ℝ) : x ≥ 0 → x ≥ 1 / 10 ^ 6')['status'], 'FAIL')

    def test_counterexample_is_exact(self):
        record = one('(x y : ℝ) : x * y ≠ 1 / 4')
        self.assertEqual(record['status'], 'FAIL')
        x, y = (fuzz.Fraction(record['counterexample'][k]) for k in ('x', 'y'))
        self.assertEqual(x * y, fuzz.Fraction(1, 4))

    def test_hypotheses_are_respected(self):
        record = one('(x : ℝ) (h : x ≠ 0) : x * (1 / x) = 1')
        self.assertEqual(record['status'], 'PASS-sampled')
        self.assertGreater(record['hypothesis_hits'], 0)
        self.assertEqual(one('(n : Nat) (h : 3 ≤ n) : n * n ≥ 9')['status'], 'PASS-sampled')

    def test_vacuous_hypotheses_are_flagged(self):
        self.assertEqual(one('(n : Nat) (h : n < 0) : n = 5')['status'], 'VACUITY_SUSPECT')
        self.assertEqual(one('(x : ℝ) (h : x < x) : x = 5')['status'], 'VACUITY_SUSPECT')

    def test_equality_hypotheses_are_repaired_soundly(self):
        record = one('(c s : ℝ) (h : c ^ 2 + s ^ 2 = 1) : c ^ 2 ≤ 1')
        self.assertEqual(record['status'], 'PASS-sampled')
        self.assertGreater(record['hypothesis_hits'], 0)
        self.assertEqual(one('(c s : ℝ) (h : c ^ 2 + s ^ 2 = 1) : c ^ 2 < 1')['status'], 'FAIL')
        self.assertEqual(one('(a b : ℝ) (h : 3 * a + b = 7) : b = 7 - 3 * a')['status'], 'PASS-sampled')

    def test_interval_binders_are_sampled_inside_the_interval(self):
        record = one(': ∀ x ∈ Set.Icc (0 : ℝ) 1, x * x ≤ x')
        self.assertEqual(record['status'], 'PASS-sampled')
        self.assertGreater(record['hypothesis_hits'], 0)
        self.assertEqual(one(': ∀ x ∈ Set.Icc (0 : ℝ) 2, x * x ≤ x')['status'], 'FAIL')

    def test_runs_are_deterministic(self):
        a = fuzz.fuzz_text('theorem t (x y : ℝ) : (x - y) ^ 2 ≥ 0 := by positivity\n', samples=200)
        b = fuzz.fuzz_text('theorem t (x y : ℝ) : (x - y) ^ 2 ≥ 0 := by positivity\n', samples=200)
        self.assertEqual(a, b)


class Definitions(unittest.TestCase):
    PRELUDE = ('noncomputable def f (x : ℝ) : ℝ := x ^ 2 + 1\n'
               'def Good (x : ℝ) : Prop := 0 < x\n'
               'def c : Nat := 7340691930603427103\n'
               'def g : Nat → Nat\n  | 0 => 1\n  | n + 1 => g n\n'
               'def loop (n : Nat) : Nat := loop n\n')

    def test_definitions_expand(self):
        self.assertEqual(one('(x : ℝ) : f x > 0', prelude=self.PRELUDE)['status'], 'PASS-sampled')
        self.assertEqual(one('(x : ℝ) : f x > 1', prelude=self.PRELUDE)['status'], 'FAIL')
        self.assertEqual(one(': c + 1 = 7340691930603427104', prelude=self.PRELUDE)['status'], 'PASS-exact')

    def test_prop_definitions_expand(self):
        self.assertEqual(one('(x : ℝ) (h : Good x) : x ≠ 0', prelude=self.PRELUDE)['status'], 'PASS-sampled')
        self.assertEqual(one('(x : ℝ) (h : Good x) : x > 1', prelude=self.PRELUDE)['status'], 'FAIL')

    def test_unsupported_definitions_make_theorems_na(self):
        record = one(': g 3 = 1', prelude=self.PRELUDE)
        self.assertEqual(record['status'], 'NA')
        self.assertIn('pattern-matching', record['reason'])
        self.assertIn('recursive', one(': loop 3 = 1', prelude=self.PRELUDE)['reason'])


class OutsideTheFragment(unittest.TestCase):
    def test_na_reasons(self):
        cases = {
            '(f : ℝ → ℝ) : f = f': 'unsupported binder',
            '[Fact (1 < 2)] : True': 'instance binder',
            ': ∫ x in (0 : ℝ)..1, x = 1 / 2': 'unsupported symbol',
            ': ∃ x : ℝ, x = 1': 'infinite domain',
            ': Real.sqrt 4 = 2': 'unknown function',
            ': 2 ^ (2 ^ 40) > 0': 'size limit',
        }
        for statement, reason in cases.items():
            with self.subTest(statement=statement):
                record = one(statement)
                self.assertEqual(record['status'], 'NA')
                self.assertIn(reason, record['reason'])

    def test_unknown_names_are_na_not_errors(self):
        record = one('(x : ℝ) : x ≤ undefinedThing x')
        self.assertEqual(record['status'], 'NA')


def git(path, *args):
    subprocess.run(['git', '-C', str(path), *args], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class Gate(unittest.TestCase):
    """The CLI gate over a real repository with two packages."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name) / 'r'
        (self.repo / 'A').mkdir(parents=True)
        (self.repo / 'B').mkdir()
        git(self.repo, 'init', '-q')
        git(self.repo, 'config', 'user.email', 'cbz@example.invalid')
        git(self.repo, 'config', 'user.name', 'cbz')
        (self.repo / 'A/lakefile.toml').write_text('name = "A"\n', encoding='utf-8')
        (self.repo / 'B/lakefile.toml').write_text('name = "B"\n', encoding='utf-8')
        (self.repo / 'A/Defs.lean').write_text('namespace A\nnoncomputable def sq (x : ℝ) : ℝ := x * x\nend A\n',
                                               encoding='utf-8')
        (self.repo / 'B/Use.lean').write_text(
            'namespace A\ntheorem sq_nonneg (x : ℝ) : sq x ≥ 0 := by positivity\n'
            'theorem empty (n : Nat) (h : n < 0) : n = 1 := by omega\nend A\n', encoding='utf-8')
        git(self.repo, 'add', '-A')
        git(self.repo, 'commit', '-q', '-m', 'init')
        self.ack = Path(self.tmp.name) / 'ack.json'

    def tearDown(self):
        self.tmp.cleanup()

    def cli(self):
        out = Path(self.tmp.name) / 'report.json'
        with contextlib.redirect_stdout(io.StringIO()):
            code = fuzz.main(['--repo', f'R={self.repo}@HEAD', '--samples', '200', '--acknowledged', str(self.ack),
                              '--json', str(out)])
        return code, json.loads(out.read_text(encoding='utf-8'))

    def test_cross_package_definitions_and_unacknowledged_vacuity(self):
        code, report = self.cli()
        by_name = {r['theorem']: r for r in report['records']}
        self.assertEqual(by_name['A.sq_nonneg']['status'], 'PASS-sampled')
        self.assertEqual(by_name['A.empty']['status'], 'VACUITY_SUSPECT')
        self.assertEqual(code, 1)
        self.assertFalse(report['certifies'])

    def test_acknowledgement_must_match_the_exact_statement(self):
        _, report = self.cli()
        sha = next(r['signature_sha256'] for r in report['records'] if r['theorem'] == 'A.empty')
        row = {'repository': 'R', 'theorem': 'A.empty', 'signature_sha256': sha,
               'reason': 'deliberately vacuous negative control, indexed as vacuous'}
        self.ack.write_text(json.dumps({'schema': 1, 'acknowledged': [row]}), encoding='utf-8')
        code, _ = self.cli()
        self.assertEqual(code, 0)
        row['signature_sha256'] = '0' * 64
        self.ack.write_text(json.dumps({'schema': 1, 'acknowledged': [row]}), encoding='utf-8')
        code, _ = self.cli()
        self.assertEqual(code, 1)

    def acknowledge_empty(self):
        _, report = self.cli()
        sha = next(r['signature_sha256'] for r in report['records'] if r['theorem'] == 'A.empty')
        row = {'repository': 'R', 'theorem': 'A.empty', 'signature_sha256': sha,
               'reason': 'deliberately vacuous negative control, indexed as vacuous'}
        self.ack.write_text(json.dumps({'schema': 1, 'acknowledged': [row]}), encoding='utf-8')

    def test_counterexample_alone_fails_the_gate(self):
        self.acknowledge_empty()
        self.assertEqual(self.cli()[0], 0)
        (self.repo / 'B/Bad.lean').write_text('theorem bad (x : ℝ) : x * x > 0 := by sorry\n', encoding='utf-8')
        git(self.repo, 'add', '-A')
        git(self.repo, 'commit', '-q', '-m', 'bad')
        code, report = self.cli()
        bad = next(r for r in report['records'] if r['theorem'] == 'bad')
        self.assertEqual((code, bad['status'], bad['counterexample']), (1, 'FAIL', {'x': '0'}))
        self.assertEqual(report['summary']['blocking'], 1)

    def test_internal_fault_is_a_blocking_error(self):
        self.acknowledge_empty()
        original = fuzz.prop
        def broken(*args, **kwargs):
            raise RuntimeError('planted fault')
        fuzz.prop = broken
        try:
            code, report = self.cli()
        finally:
            fuzz.prop = original
        statuses = {r['theorem']: r['status'] for r in report['records']}
        self.assertEqual(statuses['A.sq_nonneg'], 'ERROR')
        self.assertEqual(code, 1)


if __name__ == '__main__':
    unittest.main()
