"""Source wiring and exact finite-model controls; these do not run the Lean kernel."""
from fractions import Fraction as F
import hashlib
import itertools
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULE = 'ResearchFormalCoreR1/MeasureBridge.lean'
NAMES = [
    'p02_lm008_integral_cs',
    'p02_lm008_event_cs',
    'p02_lm008_event_numerator',
    'p02_lm008_measure_transfer',
    'p02_lm009_measure_transfer_r8_to_r3',
    'p02_lm008_sqrt_counterexample',
    'p02_lm008_upper_normalizer_counterexample',
]
ORIGINALS = {
    'ResearchFormalCoreR1/AlgebraV2.lean': '4480708263c40a2f7f03f12ef7e15f0eb4f8193c2dccbb253921a7ec251863c3',
    'ResearchFormalCoreR1/ProbabilityCompanionsV2.lean': '338452a0c2ba24d016059354d543cc995b8cb29c9d4a41d98bb93499f5f8cb8f',
}


def moments(probabilities, weights, event):
    q = sum((p for p, inside in zip(probabilities, event) if inside), F(0))
    n = sum((p*w for p, w, inside in zip(probabilities, weights, event) if inside), F(0))
    z = sum((p*w for p, w in zip(probabilities, weights)), F(0))
    second = sum((p*w*w for p, w in zip(probabilities, weights)), F(0))
    return q, n, z, second


class MeasureBridgeSourceTests(unittest.TestCase):
    def test_bridge_module_present(self):
        self.assertTrue((ROOT/MODULE).is_file(), 'measure-theoretic bridge not supplied')

    def test_exact_new_inventory(self):
        manifest = json.loads((ROOT/'manifest.json').read_text())
        targets = ['ResearchFormalCoreR1.'+name for name in NAMES]
        self.assertEqual(manifest['targets'][13:20], targets)
        self.assertGreaterEqual(len(manifest['targets']), 20)
        self.assertIn(MODULE, manifest['source_modules'])

    def test_module_and_scope_bound(self):
        manifest = json.loads((ROOT/'manifest.json').read_text())
        for path in (MODULE, 'MEASURE_BRIDGE.md', 'tests/test_measure_bridge.py'):
            self.assertIn(path, manifest['files'])
            self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),
                             manifest['files'][path])

    def test_registered_declarations_and_import(self):
        self.assertTrue((ROOT/MODULE).exists(), 'measure-theoretic bridge not supplied')
        source = (ROOT/MODULE).read_text()
        self.assertEqual(re.findall(r'^(?:theorem|lemma)\s+([A-Za-z_][A-Za-z0-9_]*)', source, re.M), NAMES)
        self.assertIn('import ResearchFormalCoreR1.MeasureBridge\n',
                      (ROOT/'ResearchFormalCoreR1.lean').read_text())

    def test_original_companion_identities_retained(self):
        manifest = json.loads((ROOT/'manifest.json').read_text())
        for path, digest in ORIGINALS.items():
            self.assertEqual(manifest['files'][path], digest)
        self.assertEqual(manifest['scientific_effect'], 'NONE')
        self.assertEqual(manifest['alignment_status'], 'PENDING_INDEPENDENT_REVIEW')


class MeasureBridgeFiniteModelTests(unittest.TestCase):
    def test_dependent_events_and_zero_edges(self):
        # Includes null/full events, zero weights, and events depending on W.
        probabilities = [F(1, 4), F(1, 4), F(1, 2)]
        for weights in itertools.product([F(0), F(1, 2), F(2)], repeat=3):
            for event in itertools.product([False, True], repeat=3):
                q, n, z, second = moments(probabilities, weights, event)
                self.assertLessEqual(n*n, second*q)
                if z > 0:
                    # Square both nonnegative sides of the exact-normalizer bound.
                    self.assertLessEqual((n/z)**2, second*q/(z*z))
                if q == 0:
                    self.assertEqual(n, 0)

    def test_omitted_square_root_is_false_and_cs_is_sharp(self):
        q, n, z, second = moments([F(1, 4), F(3, 4)], [F(4), F(0)], [True, False])
        self.assertEqual((q, n, z, second), (F(1, 4), F(1), F(1), F(4)))
        self.assertEqual(n*n, second*q)
        self.assertGreater(n*n, second*q*q)

    def test_upper_normalizer_bound_does_not_replace_lower(self):
        q, n, z, second = moments([F(1)], [F(1)], [True])
        r, cW, cZ = F(1), F(1), F(2)
        self.assertLessEqual(second, cW*r**4)
        self.assertLessEqual(z, cZ*r**2)  # deliberately wrong direction
        self.assertGreater((n/z)**2, cW*q/cZ**2)

    def test_second_moment_cannot_be_replaced_by_first(self):
        q, n, z, second = moments([F(1, 16), F(15, 16)], [F(16), F(0)], [True, False])
        self.assertEqual(z, 1)
        self.assertGreater(n*n, z*q)
        self.assertEqual(n*n, second*q)

    def test_cross_law_substitution_fails(self):
        _, n, _, second = moments([F(1, 4), F(3, 4)], [F(4), F(0)], [True, False])
        wrong_q = F(1, 16)
        self.assertGreater(n*n, second*wrong_q)

    def test_eighth_order_tail_ledger(self):
        for r in [F(1), F(1, 2), F(1, 4), F(1, 8)]:
            self.assertEqual((r**4)**2, r**8)
            self.assertLessEqual(r**4, r**3)


if __name__ == '__main__':
    unittest.main()
