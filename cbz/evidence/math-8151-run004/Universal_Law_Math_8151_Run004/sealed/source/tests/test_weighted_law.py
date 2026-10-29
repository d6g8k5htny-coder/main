"""Exact finite-model falsification and source wiring, not Lean execution."""
from fractions import Fraction as F
import hashlib
import itertools
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULE = 'ResearchFormalCoreR1/WeightedLaw.lean'
NAMES = [
    'weightedLaw_apply', 'weightedLaw_isProbabilityMeasure',
    'weightedLaw_event_real', 'weightedLaw_absolutelyContinuous',
    'weightedLaw_congr_ae', 'weightedLaw_zero', 'weightedLaw_one',
    'p02_lm008_probability_transfer', 'p02_lm009_probability_transfer_r8_to_r3',
]


def normalized_atoms(probabilities, weights):
    """Independent rational finite model; nonnegativity is almost everywhere."""
    if len(probabilities) != len(weights) or sum(probabilities, F(0)) != 1:
        raise ValueError('not a probability vector')
    if any(p < 0 for p in probabilities):
        raise ValueError('negative probability')
    if any(p > 0 and w < 0 for p, w in zip(probabilities, weights)):
        raise ValueError('weight negative on positive mass')
    z = sum((p*w for p, w in zip(probabilities, weights)), F(0))
    if z <= 0:
        raise ValueError('normalizer must be positive')
    return tuple(p*max(w, F(0))/z for p, w in zip(probabilities, weights))


class WeightedLawSourceTests(unittest.TestCase):
    def test_module_and_declarations(self):
        self.assertTrue((ROOT/MODULE).is_file(), 'weighted-law module not supplied')
        source = (ROOT/MODULE).read_text()
        self.assertEqual(re.findall(r'^theorem\s+([A-Za-z_][A-Za-z0-9_]*)', source, re.M), NAMES)
        self.assertIn('noncomputable def weightedLaw', source)

    def test_exact_target_inventory(self):
        m = json.loads((ROOT/'manifest.json').read_text())
        self.assertEqual(m['targets'][20:29], ['ResearchFormalCoreR1.'+n for n in NAMES])
        self.assertGreaterEqual(len(m['targets']), 29)

    def test_bound_source_scope_and_tests(self):
        m = json.loads((ROOT/'manifest.json').read_text())
        for path in (MODULE, 'WEIGHTED_LAW.md', 'tests/test_weighted_law.py'):
            self.assertIn(path, m['files'])
            self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(), m['files'][path])

    def test_root_import(self):
        self.assertIn('import ResearchFormalCoreR1.WeightedLaw\n',
                      (ROOT/'ResearchFormalCoreR1.lean').read_text())

    def test_parent_proofs_and_execution_controls_retained(self):
        m = json.loads((ROOT/'manifest.json').read_text())
        pins = {
            'ResearchFormalCoreR1/MeasureBridge.lean': '28ac130db490c3c3e2fbfabf3ac5be43c854ab1468bf573bf5a8c0dd2c50955f',
            'gate.py': 'ff8a5374597772b2d37054fd722c2243c4b552a63802c69f7c605519832be6c4',
            'lean-toolchain': 'd5edba4e4b8faad9c1baeadb265716d20d03be4d1a2647dc5e35b0c0325bea7b',
        }
        for path, digest in pins.items():
            self.assertEqual(m['files'][path], digest)
        self.assertEqual(m['alignment_status'], 'PENDING_INDEPENDENT_REVIEW')
        self.assertEqual(m['scientific_effect'], 'NONE')


class WeightedLawFiniteModelTests(unittest.TestCase):
    def test_all_event_ratios_and_transfer(self):
        probabilities = (F(1, 4), F(1, 4), F(1, 2))
        cases = 0
        for weights in itertools.product((F(0), F(1, 2), F(3)), repeat=3):
            z = sum(p*w for p, w in zip(probabilities, weights))
            if z == 0:
                continue
            atoms = normalized_atoms(probabilities, weights)
            self.assertEqual(sum(atoms), 1)
            second = sum(p*w*w for p, w in zip(probabilities, weights))
            for event in itertools.product((False, True), repeat=3):
                q = sum(p for p, a in zip(probabilities, event) if a)
                n = sum(p*w for p, w, a in zip(probabilities, weights, event) if a)
                actual = sum(p for p, a in zip(atoms, event) if a)
                self.assertEqual(actual, n/z)
                self.assertGreaterEqual(actual, 0)
                self.assertLessEqual(actual, 1)
                self.assertLessEqual(actual**2, second*q/z**2)
                cases += 1
        self.assertEqual(cases, 208)

    def test_constant_positive_weights(self):
        p = (F(1, 4), F(3, 4))
        for a in (F(1, 7), F(1), F(5)):
            self.assertEqual(normalized_atoms(p, (a, a)), p)

    def test_zero_normalizer_rejected(self):
        with self.assertRaisesRegex(ValueError, 'normalizer'):
            normalized_atoms((F(1, 2), F(1, 2)), (F(0), F(0)))

    def test_signed_weight_rejected_and_clipping_is_not_normalization(self):
        p, w = (F(1, 2), F(1, 2)), (F(-1), F(3))
        with self.assertRaisesRegex(ValueError, 'negative'):
            normalized_atoms(p, w)
        z = sum(a*b for a, b in zip(p, w))
        self.assertEqual(z, 1)
        self.assertEqual(sum(a*max(b, F(0))/z for a, b in zip(p, w)), F(3, 2))

    def test_ae_representatives_and_null_events(self):
        p = (F(0), F(1, 4), F(3, 4))
        first = normalized_atoms(p, (F(-100), F(4), F(0)))
        second = normalized_atoms(p, (F(100), F(4), F(0)))
        self.assertEqual(first, second)
        self.assertEqual(first, (F(0), F(1), F(0)))

    def test_positive_rescaling(self):
        p, w = (F(1, 3), F(2, 3)), (F(2), F(5))
        for a in (F(1, 7), F(9)):
            self.assertEqual(normalized_atoms(p, w), normalized_atoms(p, tuple(a*x for x in w)))

    def test_omitting_normalizer_changes_total_mass(self):
        p, w = (F(1, 2), F(1, 2)), (F(2), F(2))
        self.assertEqual(sum(a*b for a, b in zip(p, w)), 2)
        self.assertEqual(sum(normalized_atoms(p, w)), 1)

    def test_countable_additivity_finite_partition_witness(self):
        p, w = (F(1, 4), F(1, 4), F(1, 2)), (F(1), F(2), F(3))
        atoms = normalized_atoms(p, w)
        self.assertEqual(sum(atoms[:2]) + atoms[2], 1)
        self.assertEqual(sum(atoms[1:]), 1-atoms[0])


if __name__ == '__main__':
    unittest.main()
