"""Verifier's reconstruction of the T343-4 proposed EnclosureTests."""
from fractions import Fraction as Q
from unittest import mock
import unittest

import population_certificate as pc


class EnclosureTests(unittest.TestCase):
    def test_power_enclosures_bracket_non_perfect_powers(self):
        for v in (Q(2), Q(3), Q(1, 3), Q(5, 7)):
            with self.subTest(v=v):
                lo, hi = pc._sqrt((v, v))
                self.assertLess(lo, hi)
                self.assertTrue(lo ** 2 <= v <= hi ** 2)
                lo, hi = pc.quarter_power_bounds(v)
                self.assertLess(lo, hi)
                self.assertTrue(lo ** 4 <= v <= hi ** 4)
                lo, hi = pc.cube_root_bounds(v)
                self.assertLess(lo, hi)
                self.assertTrue(lo ** 3 <= v <= hi ** 3)

    def test_direct_row_composition_is_replayable(self):
        row = pc.direct_mass(1)
        core, tail, sliver = row['core_interval'], row['tail_interval'], row['sliver_width']
        self.assertEqual(row['normalized_interval'], (core[0] + tail[0], core[1] + tail[1] + sliver))
        self.assertGreater(sliver, 0)
        self.assertGreater(row['discretization_components']['core'], 0)
        self.assertGreater(row['discretization_components']['tail'], 0)
        self.assertEqual(row['interval'], tuple(x / row['n'] for x in row['normalized_interval']))
        self.assertEqual((row['tau'], row['n']), (Q(1, 8), 4))

    def test_failed_width_gate_withholds_certification(self):
        with mock.patch.object(pc, 'CORE_PANELS', 64):
            row = pc.direct_mass(1)
        self.assertFalse(row['gates']['normalized_width'])
        self.assertEqual(row['status'], 'INCONCLUSIVE_PRECISION')


if __name__ == '__main__':
    unittest.main()
