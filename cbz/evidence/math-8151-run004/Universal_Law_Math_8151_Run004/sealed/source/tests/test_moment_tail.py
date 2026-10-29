"""Exact finite-model falsification and source wiring; not a Lean kernel replay."""
from fractions import Fraction as F
import hashlib
import itertools
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULE = 'ResearchFormalCoreR1/MomentTail.lean'
NAMES = [
    'p02_lm009_markov_event',
    'p02_lm009_badJet_measurable',
    'p02_lm009_badJet_subset',
    'p02_lm009_moment40_tail',
    'p02_lm009_moment40_weighted_r4',
    'p02_lm009_moment40_weighted_r3',
    'p02_lm009_moment40_family_r3',
]


def statistics(p, values, weights, r, eps):
    event = [eps < r*x**5 for x in values]
    q = sum((a for a, yes in zip(p, event) if yes), F(0))
    moment = sum(a*x**40 for a, x in zip(p, values))
    z = sum(a*w for a, w in zip(p, weights))
    second = sum(a*w*w for a, w in zip(p, weights))
    n = sum((a*w for a, w, yes in zip(p, weights, event) if yes), F(0))
    return event, q, moment, z, second, n


class MomentTailSourceTests(unittest.TestCase):
    def test_new_source_and_exact_declarations(self):
        self.assertTrue((ROOT/MODULE).is_file(), 'moment-tail formalization absent')
        source = (ROOT/MODULE).read_text()
        self.assertEqual(re.findall(r'^theorem\s+([A-Za-z_][A-Za-z0-9_]*)', source, re.M), NAMES)

    def test_exact_inventory(self):
        m = json.loads((ROOT/'manifest.json').read_text())
        self.assertEqual(m['targets'][29:36], ['ResearchFormalCoreR1.'+n for n in NAMES])
        self.assertGreaterEqual(len(m['targets']), 36)
        self.assertIn(MODULE, m['source_modules'])

    def test_source_scope_and_tests_bound(self):
        m = json.loads((ROOT/'manifest.json').read_text())
        for name in (MODULE, 'MOMENT_TAIL.md', 'tests/test_moment_tail.py'):
            self.assertIn(name, m['files'])
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(), m['files'][name])

    def test_root_import(self):
        self.assertIn('import ResearchFormalCoreR1.MomentTail\n',
                      (ROOT/'ResearchFormalCoreR1.lean').read_text())

    def test_retained_control_and_proof_pins(self):
        m = json.loads((ROOT/'manifest.json').read_text())
        pins = {
            'ResearchFormalCoreR1/MeasureBridge.lean': '28ac130db490c3c3e2fbfabf3ac5be43c854ab1468bf573bf5a8c0dd2c50955f',
            'ResearchFormalCoreR1/WeightedLaw.lean': 'f31dfd2cd048cc0c0446c54c46d89c9cdd5d7bb9d76e9d8d16b7c034f351d736',
            'gate.py': 'ff8a5374597772b2d37054fd722c2243c4b552a63802c69f7c605519832be6c4',
        }
        for name, digest in pins.items():
            self.assertEqual(m['files'][name], digest)
        self.assertEqual(m['scientific_effect'], 'NONE')
        self.assertEqual(m['alignment_status'], 'PENDING_INDEPENDENT_REVIEW')


class MomentTailFiniteModelTests(unittest.TestCase):
    def test_markov_tail_and_threshold_inclusion(self):
        p = (F(1,4), F(1,4), F(1,2))
        count = 0
        for values in itertools.product((F(-2), F(0), F(2)), repeat=3):
            for r, eps in itertools.product((F(1), F(1,2), F(1,16)), (F(1), F(1,4), F(16))):
                event, q, m, _, _, _ = statistics(p, values, (F(1),)*3, r, eps)
                for value, inside in zip(values, event):
                    if inside:
                        self.assertLessEqual((eps/r)**8, value**40)
                self.assertLessEqual(q, (m/eps**8)*r**8)
                count += 1
        self.assertEqual(count, 243)

    def test_actual_weighted_probabilities_and_cubic_composition(self):
        p = (F(1,4), F(1,4), F(1,2))
        choices = ((F(1),)*3, (F(0),F(4),F(0)), (F(1,2),F(2),F(0)), (F(2),F(0),F(3)))
        count = 0
        for values in itertools.product((F(-2), F(0), F(2)), repeat=3):
            for weights in choices:
                for r, eps in itertools.product((F(1), F(1,2), F(1,16)), (F(1), F(1,4), F(16))):
                    _, q, m, z, second, n = statistics(p, values, weights, r, eps)
                    self.assertGreater(z, 0)
                    pw = n/z
                    coef_squared = second/z**2 * (m/eps**8)
                    self.assertLessEqual(pw**2, coef_squared*r**8)
                    self.assertLessEqual(pw**2, coef_squared*r**6)
                    count += 1
        self.assertEqual(count, 972)

    def test_epsilon_factor_cannot_be_omitted(self):
        r, eps = F(1,2), F(1,4)
        _, q, m, _, _, _ = statistics((F(1),), (F(1),), (F(1),), r, eps)
        self.assertGreater(q, m*r**8)
        self.assertLessEqual(q, m/eps**8*r**8)

    def test_twentieth_moment_cannot_replace_fortieth(self):
        r, eps, value = F(1), F(16), F(2)
        self.assertLess(eps, r*value**5)
        self.assertGreater(F(1), value**20/eps**8*r**8)
        self.assertLessEqual(F(1), value**40/eps**8*r**8)

    def test_wrong_r_sixteen_power_fails(self):
        r, value = F(1,16), F(2)
        self.assertLess(F(1), r*value**5)
        self.assertGreater(F(1), value**40*r**16)

    def test_strict_boundary_and_signed_jet(self):
        event, q, _, _, _, _ = statistics((F(1,2),F(1,2)), (F(1),F(-1)), (F(1),F(1)), F(1), F(1))
        self.assertEqual(event, [False,False])
        self.assertEqual(q, 0)

    def test_zero_moment_null_atom_and_dependent_weight(self):
        _, q, m, z, _, n = statistics((F(0),F(1)), (F(100),F(0)), (F(100),F(1)), F(1), F(1))
        self.assertEqual((q,m,z,n), (0,0,1,0))


if __name__ == '__main__':
    unittest.main()
