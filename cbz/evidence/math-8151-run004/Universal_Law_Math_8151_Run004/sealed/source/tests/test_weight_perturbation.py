"""Exact finite-model controls and source wiring; not a Lean kernel replay."""
from fractions import Fraction as F
import hashlib
import itertools
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULE = 'ResearchFormalCoreR1/WeightPerturbation.lean'
NAMES = [
    'weightPerturbation_integral',
    'weightPerturbation_setIntegral',
    'weightPerturbation_normalizer_lower',
    'weightPerturbation_event_integral_bounds',
    'weightPerturbation_quotient_bound',
    'weightedLaw_event_perturbation',
    'weightedLaw_event_perturbation_r2',
]


def integral(masses, values):
    if len(masses) != len(values) or any(m < 0 for m in masses):
        raise ValueError('invalid finite nonnegative measure')
    return sum((m * x for m, x in zip(masses, values)), F(0))


def l1_error(masses, w, v):
    if len(w) != len(v):
        raise ValueError('weights have different domains')
    return integral(masses, tuple(abs(x-y) for x, y in zip(w, v)))


def event_integral(masses, values, event):
    if len(event) != len(values):
        raise ValueError('event has a different domain')
    return integral(masses, tuple(x if yes else F(0) for x, yes in zip(values, event)))


class WeightPerturbationSourceTests(unittest.TestCase):
    def test_module_and_exact_declarations(self):
        self.assertTrue((ROOT/MODULE).is_file(), 'weight perturbation module absent')
        text = (ROOT/MODULE).read_text()
        self.assertEqual(re.findall(r'^theorem\s+([A-Za-z_][A-Za-z0-9_]*)', text, re.M), NAMES)

    def test_exact_extension_inventory(self):
        m = json.loads((ROOT/'manifest.json').read_text())
        self.assertEqual(m['targets'][40:47], ['ResearchFormalCoreR1.'+n for n in NAMES])
        self.assertGreaterEqual(len(m['targets']), 47)
        self.assertIn(MODULE, m['source_modules'])

    def test_source_scope_and_tests_are_bound(self):
        m = json.loads((ROOT/'manifest.json').read_text())
        for path in (MODULE, 'WEIGHT_PERTURBATION.md', 'tests/test_weight_perturbation.py'):
            self.assertIn(path, m['files'])
            self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(), m['files'][path])

    def test_root_import(self):
        self.assertIn('import ResearchFormalCoreR1.WeightPerturbation\n',
                      (ROOT/'ResearchFormalCoreR1.lean').read_text())

    def test_predecessor_and_gate_identity_contract(self):
        m = json.loads((ROOT/'manifest.json').read_text())
        pins = {
            'ResearchFormalCoreR1/WeightedLaw.lean': 'f31dfd2cd048cc0c0446c54c46d89c9cdd5d7bb9d76e9d8d16b7c034f351d736',
            'ResearchFormalCoreR1/MomentGenerality.lean': 'fdec55f1602938cd8fdb5b71535962e76a9671bf230dea53868b14147fb3b509',
            'gate.py': 'ff8a5374597772b2d37054fd722c2243c4b552a63802c69f7c605519832be6c4',
        }
        for path, digest in pins.items():
            self.assertEqual(m['files'][path], digest)
        self.assertEqual(m['scientific_effect'], 'NONE')
        self.assertEqual(m['alignment_status'], 'PENDING_INDEPENDENT_REVIEW')


class WeightPerturbationFiniteTests(unittest.TestCase):
    def test_generic_integral_bound_including_signed_weights(self):
        m = (F(0), F(1, 4), F(3, 4))
        vectors = list(itertools.product((F(-2), F(0), F(2)), repeat=3))
        for w, v in itertools.product(vectors, repeat=2):
            delta = l1_error(m, w, v)
            z, t = integral(m, w), integral(m, v)
            self.assertLessEqual(abs(z-t), delta)
            self.assertLessEqual(t-delta, z)

    def test_set_integral_domination_and_nonnegative_bounds(self):
        m = (F(1, 4), F(1, 4), F(1, 2))
        for w in itertools.product((F(0), F(1), F(2)), repeat=3):
            v = (F(2), F(0), F(1))
            delta, z = l1_error(m, w, v), integral(m, w)
            for event in itertools.product((False, True), repeat=3):
                a, b = event_integral(m, w, event), event_integral(m, v, event)
                self.assertGreaterEqual(a, 0)
                self.assertLessEqual(a, z)
                self.assertLessEqual(abs(a-b), delta)

    def test_quotient_bound_does_not_need_b_nonnegative(self):
        for z, t in itertools.product((F(1, 2), F(1), F(2)), repeat=2):
            for a in (F(0), z/2, z):
                for b in (F(-1), F(0), t/2, t, 2*t):
                    delta = max(abs(a-b), abs(z-t))
                    self.assertLessEqual(abs(a/z-b/t), 2*delta/t)

    def test_normalized_laws_uniform_over_all_events(self):
        tested = 0
        vectors = list(itertools.product((F(0), F(1), F(2)), repeat=3))
        for total in (F(0), F(1, 2), F(1), F(2)):
            m = tuple(total*x for x in (F(1, 4), F(1, 4), F(1, 2)))
            for w, v in itertools.product(vectors, repeat=2):
                z, t, delta = integral(m, w), integral(m, v), l1_error(m, w, v)
                if t <= 0 or delta > t/2:
                    continue
                self.assertGreaterEqual(z, t/2)
                self.assertEqual(sum(x*y/z for x, y in zip(m, w)), 1)
                for event in itertools.product((False, True), repeat=3):
                    a, b = event_integral(m, w, event), event_integral(m, v, event)
                    self.assertLessEqual(abs(a/z-b/t), 2*delta/t)
                    tested += 1
        self.assertGreater(tested, 0)

    def test_zero_error_and_null_set_changes(self):
        m = (F(0), F(1, 2), F(1, 2))
        w, v = (F(-100), F(2), F(1)), (F(100), F(2), F(1))
        self.assertEqual(l1_error(m, w, v), 0)
        self.assertEqual(integral(m, w), integral(m, v))
        for event in itertools.product((False, True), repeat=3):
            self.assertEqual(event_integral(m, w, event), event_integral(m, v, event))

    def test_half_budget_endpoint_still_normalizes(self):
        m, w, v = (F(1),), (F(1, 2),), (F(1),)
        c, delta = F(1), l1_error(m, w, v)
        self.assertEqual(delta, c/2)
        self.assertEqual(integral(m, w), c/2)
        self.assertGreater(integral(m, w), 0)

    def test_r2_scale_cancels_without_r_le_one(self):
        m, w0, v0 = (F(1, 2), F(1, 2)), (F(3, 4), F(1)), (F(1), F(1))
        eta = l1_error(m, w0, v0)
        for r in (F(1, 100), F(1, 2), F(1), F(4)):
            w, v = tuple(r*r*x for x in w0), tuple(r*r*x for x in v0)
            self.assertEqual(l1_error(m, w, v), eta*r*r)
            self.assertEqual(2*l1_error(m, w, v)/integral(m, v), 2*eta)

    def test_small_absolute_error_without_normalizer_scale_is_insufficient(self):
        m, event = (F(1, 2), F(1, 2)), (True, False)
        for eps in (F(1, 100), F(1, 10000)):
            w, v = (eps, F(0)), (F(0), eps)
            self.assertEqual(l1_error(m, w, v), eps)
            difference = abs(event_integral(m, w, event)/integral(m, w)
                             - event_integral(m, v, event)/integral(m, v))
            self.assertEqual(difference, 1)
            self.assertGreater(difference, 2*eps)  # omitting the denominator is false

    def test_absolute_integral_cannot_replace_integral_absolute(self):
        m, w, v = (F(1, 2), F(1, 2)), (F(1), F(3)), (F(3), F(1))
        self.assertEqual(abs(integral(m, w)-integral(m, v)), 0)
        self.assertEqual(l1_error(m, w, v), 2)
        self.assertNotEqual(event_integral(m, w, (True, False)),
                            event_integral(m, v, (True, False)))

    def test_same_weight_under_different_laws_is_not_zero_error_transfer(self):
        w = (F(1), F(1))
        mu, nu = (F(1), F(0)), (F(0), F(1))
        self.assertEqual(l1_error(mu, w, w), 0)
        self.assertNotEqual(event_integral(mu, w, (True, False)),
                            event_integral(nu, w, (True, False)))

    def test_signed_clipping_and_zero_normalizer_are_not_probability_cases(self):
        m, w = (F(1, 2), F(1, 2)), (F(-1), F(3))
        z = integral(m, w)
        self.assertEqual(z, 1)
        self.assertEqual(integral(m, tuple(max(x, F(0)) for x in w))/z, F(3, 2))
        v, zero = (F(1), F(1)), (F(0), F(0))
        self.assertEqual(integral(m, zero), 0)
        self.assertGreater(l1_error(m, zero, v), integral(m, v)/2)


if __name__ == '__main__':
    unittest.main()
