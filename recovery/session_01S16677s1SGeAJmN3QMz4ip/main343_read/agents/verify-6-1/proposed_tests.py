"""Reviewer T343-5 proposed amendment, transcribed verbatim from the finding."""
from fractions import Fraction as Q
import unittest
import population_certificate as pc


class PredicateFixtureTests(unittest.TestCase):
    def test_strict_interior_coefficient_fixtures(self):
        b, e = (Q(1, 5), Q(3, 10)), (Q(1, 10), Q(1, 10))
        self.assertEqual(pc.classify_coefficient((Q(12, 100), Q(15, 100)), b, e), 'INCONCLUSIVE_PRECISION')
        self.assertEqual(pc.classify_coefficient((Q(35, 100), Q(38, 100)), b, e), 'INCONCLUSIVE_PRECISION')
        self.assertEqual(pc.classify_coefficient((Q(9, 100), Q(99, 1000)), b, e), 'FALSIFIED')

    def test_void_width_limit_scale(self):
        target = (Q(1, 2), Q(1, 2))
        self.assertEqual(pc.classify_void((Q(1, 2), Q(1, 2) + Q(1, 1 << 16)), target), 'PASS')
        self.assertEqual(pc.classify_void((Q(1, 2), Q(1, 2) + Q(1, 1 << 16) + Q(1, 1 << 40)), target),
                         'INCONCLUSIVE_PRECISION')


if __name__ == '__main__':
    unittest.main()
