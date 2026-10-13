"""Independent re-derivation of the amendment's three EnclosureTests; argv[1] = repo root."""
import sys, pathlib, unittest
from fractions import Fraction as Q
from unittest import mock
root = pathlib.Path(sys.argv.pop(1)).resolve()
sys.path.insert(0, str(root / 'experiments/universality/two_parameter_cusp_population'))
import population_certificate as pc

class EnclosureTests(unittest.TestCase):
    def test_1_outward_polarity_on_irrational_radicands(self):
        for v in (Q(2), Q(3), Q(1, 3), Q(5, 7)):
            for name, fn, k in (('_sqrt', lambda x: pc._sqrt((x, x)), 2),
                                ('quarter', pc.quarter_power_bounds, 4),
                                ('cube', pc.cube_root_bounds, 3)):
                with self.subTest(v=v, fn=name):
                    lo, hi = fn(v)
                    self.assertLess(lo, hi)
                    self.assertLessEqual(lo ** k, v)
                    self.assertLessEqual(v, hi ** k)
    def test_2_direct_row_composition_replay(self):
        row = pc.direct_mass(1)
        core, tail, sl = row['core_interval'], row['tail_interval'], row['sliver_width']
        self.assertEqual(row['normalized_interval'], (core[0] + tail[0], core[1] + tail[1] + sl))
        self.assertGreater(sl, Q(0))
        self.assertGreater(row['discretization_components']['core'], Q(0))
        self.assertGreater(row['discretization_components']['tail'], Q(0))
        self.assertEqual(row['interval'], tuple(x / row['n'] for x in row['normalized_interval']))
        self.assertEqual((row['tau'], row['n']), (Q(1, 8), 4))
    def test_3_failed_gate_is_not_certified(self):
        with mock.patch.object(pc, 'CORE_PANELS', 64):
            row = pc.direct_mass(1)
        self.assertIs(row['gates']['normalized_width'], False)
        self.assertEqual(row['status'], 'INCONCLUSIVE_PRECISION')

if __name__ == '__main__':
    unittest.main(verbosity=1)
