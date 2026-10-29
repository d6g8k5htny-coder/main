"""Exact finite-model falsification and source wiring, not Lean kernel execution."""
from fractions import Fraction as F
import hashlib
import itertools
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULE = 'ResearchFormalCoreR1/MomentGenerality.lean'
NAMES = [
    'p02_lm009_moment40_tail_finite',
    'p02_lm009_moment40_tail_of_finite',
    'p02_lm009_moment40_family_r3_interval',
    'p02_lm009_moment40_family_r3_of_interval',
]


def finite_tail(masses, values, r, eps):
    if len(masses) != len(values) or any(m < 0 for m in masses):
        raise ValueError('invalid finite nonnegative measure')
    if r <= 0 or eps <= 0:
        raise ValueError('threshold parameters must be positive')
    event = tuple(eps < r*x**5 for x in values)
    q = sum((m for m, inside in zip(masses, event) if inside), F(0))
    moment = sum((m*x**40 for m, x in zip(masses, values)), F(0))
    return event, q, moment, moment/eps**8*r**8


class MomentGeneralitySourceTests(unittest.TestCase):
    def test_module_and_exact_declarations(self):
        self.assertTrue((ROOT/MODULE).is_file(), 'generality module absent')
        source = (ROOT/MODULE).read_text()
        self.assertEqual(re.findall(r'^theorem\s+([A-Za-z_][A-Za-z0-9_]*)', source, re.M), NAMES)

    def test_exact_inventory(self):
        m = json.loads((ROOT/'manifest.json').read_text())
        self.assertEqual(m['targets'][36:40], ['ResearchFormalCoreR1.'+n for n in NAMES])
        self.assertGreaterEqual(len(m['targets']), 40)
        self.assertIn(MODULE, m['source_modules'])

    def test_source_scope_and_tests_bound(self):
        m = json.loads((ROOT/'manifest.json').read_text())
        for name in (MODULE, 'MOMENT_GENERALITY.md', 'tests/test_moment_generality.py'):
            self.assertIn(name, m['files'])
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(), m['files'][name])

    def test_root_import(self):
        self.assertIn('import ResearchFormalCoreR1.MomentGenerality\n',
                      (ROOT/'ResearchFormalCoreR1.lean').read_text())

    def test_predecessor_and_gate_pins_preserved(self):
        m = json.loads((ROOT/'manifest.json').read_text())
        pins = {
            'ResearchFormalCoreR1/MomentTail.lean': '87fe3ee90afc52d8c2652881f1ae195d1b0b15a2ffc18c84d6166d1a4d0de2aa',
            'ResearchFormalCoreR1/WeightedLaw.lean': 'f31dfd2cd048cc0c0446c54c46d89c9cdd5d7bb9d76e9d8d16b7c034f351d736',
            'gate.py': 'ff8a5374597772b2d37054fd722c2243c4b552a63802c69f7c605519832be6c4',
        }
        for path, digest in pins.items():
            self.assertEqual(m['files'][path], digest)
        self.assertEqual(m['scientific_effect'], 'NONE')
        self.assertEqual(m['alignment_status'], 'PENDING_INDEPENDENT_REVIEW')


class MomentGeneralityFiniteModelTests(unittest.TestCase):
    def test_nonunit_mass_tail_grid(self):
        count = 0
        for total in (F(0), F(1, 2), F(2)):
            masses = tuple(total*x for x in (F(1, 4), F(1, 4), F(1, 2)))
            for values in itertools.product((F(-2), F(0), F(2)), repeat=3):
                for r, eps in itertools.product((F(1, 16), F(1), F(2)), (F(1, 4), F(1), F(30))):
                    event, q, moment, bound = finite_tail(masses, values, r, eps)
                    self.assertLessEqual(q, bound)
                    for x, inside in zip(values, event):
                        if inside:
                            self.assertLessEqual((eps/r)**8, x**40)
                    count += 1
        self.assertEqual(count, 729)

    def test_no_extra_total_mass_normalization(self):
        _, q, moment, bound = finite_tail((F(2),), (F(2),), F(1), F(30))
        self.assertEqual(q, 2)  # Event MASS may exceed one.
        self.assertLessEqual(q, bound)
        self.assertGreater(q, bound/2)  # An extra division by total mass is false.

    def test_mass_scaling(self):
        masses, values = (F(1, 3), F(2, 3)), (F(2), F(-1))
        _, q, moment, bound = finite_tail(masses, values, F(1, 2), F(1))
        for scale in (F(0), F(1, 2), F(2), F(7)):
            _, sq, sm, sb = finite_tail(tuple(scale*x for x in masses), values, F(1, 2), F(1))
            self.assertEqual((sq, sm, sb), (scale*q, scale*moment, scale*bound))

    def test_strict_signed_and_null_edges(self):
        event, q, _, _ = finite_tail((F(2), F(1), F(0)), (F(1), F(-1), F(100)), F(1), F(1))
        self.assertEqual(event, (False, False, True))
        self.assertEqual(q, 0)
        _, q, moment, bound = finite_tail((F(0), F(0)), (F(-10), F(100)), F(2), F(1, 2))
        self.assertEqual((q, moment, bound), (0, 0, 0))

    def test_interval_family_and_outside_zero_measure(self):
        # Uniform cW=cZ=1, M=3/4, eps=1/64, W_r=r^2 on the interval.
        r0, eps, moment_bound = F(1), F(1, 64), F(3, 4)
        coef_squared = moment_bound/eps**8
        for r in (F(-1), F(0), F(1, 128), F(1, 64), F(1, 2), F(1), F(2)):
            inside = 0 < r <= r0
            masses = (F(1, 4), F(3, 4)) if inside else (F(0), F(0))
            self.assertEqual(sum(masses), 1 if inside else 0)
            if not inside:
                continue  # No probability instance or conclusion is required here.
            values, weights = (F(0), F(1)), (r*r, r*r)
            event, q, moment, _ = finite_tail(masses, values, r, eps)
            z = sum(m*w for m, w in zip(masses, weights))
            second = sum(m*w*w for m, w in zip(masses, weights))
            n = sum((m*w for m, w, flag in zip(masses, weights, event) if flag), F(0))
            self.assertEqual(z, r*r)
            self.assertEqual(second, r**4)
            self.assertLessEqual(moment, moment_bound)
            self.assertLessEqual((n/z)**2, coef_squared*r**6)

    def test_r_dependent_coefficient_is_not_uniform(self):
        # p=1, q=r^8, D(r)=r^-4 satisfy p=D(r)*sqrt(q), but do not decay.
        ratios = []
        for r in (F(1, 2), F(1, 4), F(1, 8), F(1, 16)):
            self.assertEqual(r**(-4) * r**4, 1)
            ratios.append(1/r**3)
        self.assertEqual(ratios, [8, 64, 512, 4096])

    def test_invalid_threshold_parameters_rejected(self):
        for r, eps in ((F(0), F(1)), (F(1), F(0)), (F(-1), F(1))):
            with self.assertRaisesRegex(ValueError, 'positive'):
                finite_tail((F(2),), (F(2),), r, eps)


if __name__ == '__main__':
    unittest.main()
