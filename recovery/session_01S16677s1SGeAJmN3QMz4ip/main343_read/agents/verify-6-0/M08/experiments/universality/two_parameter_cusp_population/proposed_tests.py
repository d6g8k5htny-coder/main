from fractions import Fraction as Q
import unittest
import population_certificate as pc


class PredicateFixtureTests(unittest.TestCase):
    def test_falsified_threshold_is_outer_not_inner(self):
        b, e = (Q(1, 5), Q(3, 10)), (Q(1, 10), Q(1, 10))
        # strictly between outer[0]=1/10 and inner[0]=1/5 on the low side
        self.assertEqual(pc.classify_coefficient((Q(12, 100), Q(15, 100)), b, e), 'INCONCLUSIVE_PRECISION')
        # strictly between inner[1]=3/10 and outer[1]=2/5 on the high side
        self.assertEqual(pc.classify_coefficient((Q(35, 100), Q(38, 100)), b, e), 'INCONCLUSIVE_PRECISION')
        # strictly below outer[0]
        self.assertEqual(pc.classify_coefficient((Q(9, 100), Q(99, 1000)), b, e), 'FALSIFIED')

    def test_width_limit_scale(self):
        target = (Q(1, 2), Q(1, 2))
        self.assertEqual(pc.classify_void((Q(1, 2), Q(1, 2) + Q(1, 1 << 16)), target), 'PASS')
        self.assertEqual(pc.classify_void((Q(1, 2), Q(1, 2) + Q(1, 1 << 16) + Q(1, 1 << 40)), target),
                         'INCONCLUSIVE_PRECISION')


if __name__ == '__main__':
    unittest.main()
