"""Exact uniform-cell oracles for exposed finite dependence controls."""
from fractions import Fraction as Q
import importlib
import importlib.util
from itertools import product
from math import lcm
import unittest


def enumerated_law(probabilities, shared):
    """Integrate indicators by equiprobable uniform cells, without recursion."""
    denominator = lcm(*(p.denominator for p in probabilities)) if probabilities else 1
    cells = [Q(2*i+1, 2*denominator) for i in range(denominator)]
    outcomes = ((u,)*len(probabilities) for u in cells) if shared else product(cells, repeat=len(probabilities))
    counts = [0]*(len(probabilities)+1)
    total = 0
    for outcome in outcomes:
        counts[sum(u < p for u, p in zip(outcome, probabilities))] += 1
        total += 1
    return tuple(Q(count, total) for count in counts)


class AmplitudeProcessControls(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('amplitude_process_controls'),
                             'Exact dependence-control implementation is missing')
        self.m = importlib.import_module('amplitude_process_controls')

    def test_exposed_fixture_has_equal_marginals_and_distinct_joint_laws(self):
        probabilities = self.m.short_bar_probabilities([Q(1), Q(8)], Q(1, 4))
        self.assertEqual(probabilities, (Q(1, 4), Q(1, 16)))
        shared = self.m.count_distribution(probabilities, coupling='shared')
        independent = self.m.count_distribution(probabilities, coupling='independent')
        self.assertEqual(shared, (Q(3, 4), Q(3, 16), Q(1, 16)))
        self.assertEqual(independent, (Q(45, 64), Q(9, 32), Q(1, 64)))
        self.assertEqual(self.m.expected_count(shared), Q(5, 16))
        self.assertEqual(self.m.expected_count(independent), Q(5, 16))
        self.assertEqual(self.m.pgf(shared, Q(1, 2)), Q(55, 64))
        self.assertEqual(self.m.pgf(independent, Q(1, 2)), Q(217, 256))

    def test_complete_laws_agree_with_independent_uniform_cell_enumeration(self):
        cases = [(), (Q(1, 2),), (Q(1, 3), Q(1, 2), Q(2, 3)),
                 (Q(1, 3),)*3, (Q(0), Q(1), Q(1, 2)),
                 (Q(0),)*3, (Q(1),)*3, (Q(1, 4), Q(1, 16))]
        for probabilities in cases:
            for coupling in ('shared', 'independent'):
                with self.subTest(probabilities=probabilities, coupling=coupling):
                    actual = self.m.count_distribution(probabilities, coupling=coupling)
                    expected = enumerated_law(probabilities, coupling == 'shared')
                    self.assertEqual(actual, expected)
                    self.assertEqual(sum(actual), 1)
                    self.assertTrue(all(type(p) is Q and p >= 0 for p in actual))
                    self.assertEqual(self.m.expected_count(actual), sum(probabilities, Q(0)))
                    for z in (Q(0), Q(1, 3), Q(1)):
                        direct = sum((p*z**i for i, p in enumerate(expected)), Q(0))
                        self.assertEqual(self.m.pgf(actual, z), direct)

    def test_shared_law_is_invariant_to_order_and_ties(self):
        for coupling in ('shared', 'independent'):
            self.assertEqual(self.m.count_distribution((Q(2, 3), Q(1, 3), Q(1, 3)), coupling=coupling),
                             self.m.count_distribution((Q(1, 3), Q(1, 3), Q(2, 3)), coupling=coupling))

    def test_zero_empty_and_support_clipping_precede_exact_root_evaluation(self):
        self.assertEqual(self.m.short_bar_probabilities([], Q(1, 4)), ())
        self.assertEqual(self.m.short_bar_probabilities([Q(1), Q(8)], Q(0)), (Q(0), Q(0)))
        self.assertEqual(self.m.short_bar_probabilities([Q(1), Q(8)], Q(2)), (Q(1), Q(1, 4)))
        self.assertEqual(self.m.short_bar_probabilities([Q(1), Q(8)], Q(16)), (Q(1), Q(1)))
        # A saturated term is exactly one even when its untruncated root is irrational.
        self.assertEqual(self.m.short_bar_probabilities([Q(1)], Q(3)), (Q(1),))
        # ratio=8/27 has cube root 2/3; numerator roots must not be dropped.
        self.assertEqual(self.m.short_bar_probabilities([Q(27, 16)], Q(1)), (Q(4, 9),))

    def test_nonrational_root_refuses_the_complete_request(self):
        for lifetimes in ([Q(1)], [Q(1, 2), Q(1)]):
            with self.assertRaisesRegex(ValueError, 'root'):
                self.m.short_bar_probabilities(lifetimes, Q(1))

    def test_invalid_lifetimes_are_not_hidden_by_zero_or_saturation(self):
        for lifetimes in ([Q(0)], [Q(-1)], [True], [1], [1.0], [Q(1), Q(0)]):
            for threshold in (Q(0), Q(100)):
                with self.subTest(lifetimes=lifetimes, threshold=threshold):
                    with self.assertRaises(ValueError):
                        self.m.short_bar_probabilities(lifetimes, threshold)
        for lifetimes in (None, '1', {Q(1)}):
            with self.assertRaises(ValueError):
                self.m.short_bar_probabilities(lifetimes, Q(1))
        for threshold in (True, 1, 0.25, Q(-1)):
            with self.assertRaises(ValueError):
                self.m.short_bar_probabilities([], threshold)

    def test_invalid_marginals_and_coupling_are_rejected(self):
        for probabilities in ([Q(-1)], [Q(2)], [True], [1], [0.5], None, '1'):
            with self.assertRaises(ValueError):
                self.m.count_distribution(probabilities, coupling='shared')
        for coupling in ('unknown', '', None, True, []):
            with self.assertRaises(ValueError):
                self.m.count_distribution((), coupling=coupling)

    def test_pgf_and_mean_require_a_complete_exact_probability_law(self):
        for law in ((), (Q(1, 2),), (Q(-1), Q(2)), (True,), (1,), None):
            with self.assertRaises(ValueError):
                self.m.expected_count(law)
            with self.assertRaises(ValueError):
                self.m.pgf(law, Q(1, 2))
        for z in (True, 0, 0.5, Q(-1), Q(2)):
            with self.assertRaises(ValueError):
                self.m.pgf((Q(1),), z)

    def test_report_preserves_abstract_and_exposed_scope(self):
        report = self.m.report()
        self.assertEqual(report['scope'], 'EXACT_FINITE_BARCODE_DEPENDENCE_CONTROL')
        self.assertEqual(report['independent_world'], 'ABSTRACT_BARCODE_SURROGATE')
        self.assertTrue(report['reference_answers_exposed'])
        self.assertFalse(report['field_realization_established'])
        self.assertFalse(report['theorem_conclusion_consumed'])
        self.assertFalse(report['scientific_acceptance'])
        self.assertEqual(report['scientific_effect'], 'NONE')
        self.assertEqual(report['shared']['pgf_at_half'], '55/64')
        self.assertEqual(report['independent']['pgf_at_half'], '217/256')


if __name__ == '__main__':
    unittest.main()
