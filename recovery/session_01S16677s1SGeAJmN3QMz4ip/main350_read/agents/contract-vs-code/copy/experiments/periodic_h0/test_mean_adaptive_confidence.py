"""Mean-adaptive inference controls, using abstract laws rather than field data."""
import importlib
import importlib.util
import unittest
from fractions import Fraction as Q
from math import factorial

from gaussian_tail import exp_neg_bounds


class MeanAdaptiveConfidenceTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('mean_adaptive_confidence'),
                             'Mean-adaptive confidence implementation is missing')
        self.m = importlib.import_module('mean_adaptive_confidence')

    def test_log_enclosures_pay_for_range_reduction_and_reciprocal_sign(self):
        # Detect a missing remainder, wrong reciprocal sign or missing log(2).
        known_lo = Q(693147180559945309417232121458, 10**30)
        known_hi = known_lo + Q(1, 10**30)
        lo, hi = self.m.log_bounds(Q(2), bits=80)
        self.assertLessEqual(lo, known_hi)
        self.assertGreaterEqual(hi, known_lo)
        self.assertLessEqual(hi-lo, Q(1, 1 << 80))
        self.assertEqual(self.m.log_bounds(Q(1), bits=80), (Q(0), Q(0)))
        for value in (Q(3, 2), Q(1, 1000), Q(1 << 200), Q(1, 1 << 200)):
            a, b = self.m.log_bounds(value, bits=40)
            c, d = self.m.log_bounds(1/value, bits=40)
            self.assertLessEqual(b-a, Q(1, 1 << 40))
            self.assertEqual((a, b), (-d, -c))
        a, b = self.m.log_bounds(Q(1 << 200), bits=40)
        self.assertLessEqual(a, 200*known_hi)
        self.assertGreaterEqual(b, 200*known_lo)

    def test_kl_has_correct_orientation_and_boundary_limits(self):
        self.assertEqual(self.m.kl_bounds(Q(1, 3), Q(1, 3)), (Q(0), Q(0)))
        for s, q, lower, upper in [
                (Q(0), Q(1, 2), Q(693147, 10**6), Q(693148, 10**6)),
                (Q(1), Q(1, 4), Q(1386294, 10**6), Q(1386295, 10**6)),
                (Q(1, 4), Q(1, 2), Q(130812, 10**6), Q(130813, 10**6))]:
            lo, hi = self.m.kl_bounds(s, q, bits=40)
            self.assertGreater(lo, lower)
            self.assertLess(hi, upper)

    @staticmethod
    def _chernoff_power(total, n, cap, q):
        """Independent rational oracle: [exp(-n kl(T/nC,q))]^C."""
        s = Q(total, n*cap)
        if s == 0:
            return (1-q)**(n*cap)
        if s == 1:
            return q**(n*cap)
        return (q/s)**total * ((1-q)/(1-s))**(n*cap-total)

    def test_inversion_endpoints_are_outward_by_rational_power_oracle(self):
        # Reversing either bisection update would put an endpoint inside its root.
        n, cap, beta = 6, 16, Q(1)
        threshold_lo = exp_neg_bounds(beta*cap)[0]
        for total in (1, 19, 48, 77, 95):
            s = Q(total, n*cap)
            lo, hi = self.m.confidence_bounds(s, n, beta, bits=16)
            self.assertTrue(0 <= lo < s < hi <= 1)
            if lo:
                self.assertLessEqual(self._chernoff_power(total, n, cap, lo), threshold_lo)
            if hi < 1:
                self.assertLessEqual(self._chernoff_power(total, n, cap, hi), threshold_lo)
            # Adjacent interior probes must cross the root at the requested scale.
            step = Q(1, 1 << 14)
            if lo+step < s:
                self.assertGreater(self._chernoff_power(total, n, cap, lo+step), threshold_lo)
            if hi-step > s:
                self.assertGreater(self._chernoff_power(total, n, cap, hi-step), threshold_lo)

    def test_zero_and_full_rows_use_correct_cap_and_complement(self):
        cap, n, beta = 16384, 131072, Q(8)
        lo, hi = self.m.confidence_bounds(Q(0), n, beta)
        self.assertEqual(lo, Q(0))
        self.assertLessEqual(cap*hi, Q(1))
        self.assertGreater(cap*hi, Q(999, 1000))
        full_lo, full_hi = self.m.confidence_bounds(Q(1), n, beta)
        self.assertEqual(full_hi, Q(1))
        self.assertEqual(full_lo, 1-hi)
        self.assertEqual(self.m.confidence_bounds(Q(0), 1, Q(5000)), (Q(0), Q(1)))
        self.assertEqual(self.m.confidence_bounds(Q(1), 1, Q(5000)), (Q(0), Q(1)))

    def test_tiny_zero_case_retains_analytic_bound_below_exponential_precision(self):
        # Outward exp rounding alone loses the useful bound at extremely small x.
        n = 1 << 300
        lo, hi = self.m.confidence_bounds(Q(0), n, Q(1))
        self.assertEqual(lo, Q(0))
        self.assertLessEqual(hi, Q(1, n))
        self.assertEqual(self.m.confidence_bounds(Q(1), n, Q(1))[0], 1-hi)

    def test_ambiguous_low_precision_comparison_keeps_enclosing_bracket(self):
        # The exact KL at (1/2,1/4) is about .143841036; 18-bit log
        # enclosures cannot decide this nearby rational candidate.
        self.assertEqual(self.m.confidence_bounds(Q(1, 2), 1, Q(143841, 10**6), bits=1),
                         (Q(0), Q(1)))

    def test_family_risk_includes_both_sides_and_all_bins(self):
        risk = self.m.risk_upper(8, Q(8))
        self.assertGreater(risk, Q(5367402046440189, 10**18))
        self.assertLess(risk, Q(5367402046440190, 10**18))
        self.assertEqual(risk, 8*self.m.risk_upper(1, Q(8)))
        self.assertEqual(self.m.risk_upper(8, Q(1, 100)), Q(1))
        self.assertGreater(self.m.risk_upper(1, Q(5000)), Q(0))
        self.assertLess(self.m.risk_upper(1, Q(5000)), Q(1, 10**30))

    def test_retained_rows_and_clipping_loss_keep_the_original_denominator(self):
        rows = [(4, 5), None, (2, 3), (6, 8)]
        lo, hi = self.m.mean_interval(1, Q(0), rows, Q(1), 4, bits=16)
        with self.assertRaises(ValueError):
            self.m.mean_interval(1, Q(0), rows[:3], Q(1), 4)
        self.assertEqual(self.m.mean_interval(1, Q(0), [None]*4, Q(1), 4),
                         (Q(0), Q(16)))
        clipped = self.m.mean_interval(1, Q(1, 100), rows, Q(1), 4, bits=16)
        self.assertEqual(clipped, (max(Q(0), lo-Q(4, 25)), min(Q(16), hi+Q(4, 25))))

    def test_exact_nonbernoulli_law_and_adaptive_widening_keep_coverage(self):
        # Three-valued counts prevent an accidental Bernoulli-only argument.
        n, beta, mean = 6, Q(4), Q(19, 3)
        misses = widened_misses = Q(0)
        for a in range(n+1):
            for b in range(n-a+1):
                c = n-a-b
                values = [0]*a+[3]*b+[16]*c
                probability = Q(factorial(n)//(factorial(a)*factorial(b)*factorial(c)), 3**n)
                rows = [(v, v) for v in values]
                lo, hi = self.m.mean_interval(1, Q(0), rows, beta, n, bits=14)
                misses += probability * (not lo <= mean <= hi)
                # Selection depends on the entire outcome; every row stays retained.
                widened = [None if (sum(values)+i)%3 == 0 else row
                           for i, row in enumerate(rows)]
                wlo, whi = self.m.mean_interval(1, Q(0), widened, beta, n, bits=14)
                widened_misses += probability * (not wlo <= mean <= whi)
        self.assertEqual(misses, Q(1, 729))
        self.assertLessEqual(widened_misses, misses)
        self.assertLessEqual(misses, self.m.risk_upper(1, beta))

    def test_normalization_and_dependence_negative_controls(self):
        # Wrongly dropping C gives an upper bound beta/n=1/2 although mu=1.
        zero_probability = Q(15, 16)**16
        self.assertGreater(zero_probability, Q(1, 4))
        self.assertGreater(zero_probability, self.m.risk_upper(1, Q(8)))
        lo, hi = self.m.mean_interval(1, Q(0), [(0, 0)]*16, Q(8), 16)
        self.assertGreater(hi, Q(1))
        # Eight repetitions of one bit have miss probability1; they are not IID.
        for value in (0, 16):
            lo, hi = self.m.mean_interval(1, Q(0), [(value, value)]*8, Q(2), 8)
            self.assertFalse(lo <= 8 <= hi)
        self.assertLess(self.m.risk_upper(1, Q(2)), Q(1))

    def test_optional_stopping_and_selected_bins_need_different_theorems(self):
        # Stop at the first exclusion of mu=1/2, through time8. Exact lattice
        # recursion pays for every ordered Bernoulli sequence without sampling.
        states, failed = {0: Q(1)}, Q(0)
        for n in range(1, 9):
            following = {}
            for successes, probability in states.items():
                for step in (0, 1):
                    k = successes+step
                    lo, hi = self.m.confidence_bounds(Q(k, n), n, Q(2), bits=12)
                    if not lo <= Q(1, 2) <= hi:
                        failed += probability/2
                    else:
                        following[k] = following.get(k, Q(0))+probability/2
            states = following
        self.assertEqual(failed, Q(35, 128))
        self.assertGreater(failed, self.m.risk_upper(1, Q(2)))
        # Choose uniformly which of four bins has count16. The selected bin
        # always excludes its mean4 if it is falsely reported as a fixed bin.
        selected = self.m.mean_interval(1, Q(0), [(16, 16)], Q(1), 1)
        self.assertGreater(selected[0], Q(4))
        self.assertLess(self.m.risk_upper(1, Q(1)), Q(1))
        self.assertEqual(self.m.risk_upper(4, Q(1)), Q(1))

    def test_invalid_types_and_domains_do_not_become_inference(self):
        for value in (True, 1, 0.5, float('nan'), float('inf'), Q(0), Q(-1)):
            with self.assertRaises(ValueError):
                self.m.log_bounds(value)
        for bits in (True, 0, -1, 257, 12.0):
            with self.assertRaises(ValueError):
                self.m.log_bounds(Q(2), bits=bits)
        for s, q in ((True, Q(1, 2)), (Q(-1), Q(1, 2)), (Q(2), Q(1, 2)),
                     (Q(1, 2), Q(0)), (Q(1, 2), Q(1)), (Q(1, 2), 0.5)):
            with self.assertRaises(ValueError):
                self.m.kl_bounds(s, q)
        for n, beta in ((True, Q(1)), (0, Q(1)), (2.0, Q(1)),
                        (2, True), (2, 1.0), (2, Q(0)), (2, Q(-1))):
            with self.assertRaises(ValueError):
                self.m.confidence_bounds(Q(1, 2), n, beta)
        for m, beta in ((True, Q(1)), (0, Q(1)), (1, Q(0)), (1, float('inf'))):
            with self.assertRaises(ValueError):
                self.m.risk_upper(m, beta)
        for k in (True, 0, 65, 1.0):
            with self.assertRaises(ValueError):
                self.m.mean_interval(k, Q(0), [(0, 1)], Q(1), 1)
        for rows in ([], [(False, 2)], [(3, 2)], [(0, 17)], [(0.0, 2)], [(0, 2, 3)]):
            with self.assertRaises(ValueError):
                self.m.mean_interval(1, Q(0), rows, Q(1), 1)
        for p in (True, 0.1, float('nan'), Q(-1), Q(2)):
            with self.assertRaises(ValueError):
                self.m.mean_interval(1, p, [(0, 1)], Q(1), 1)


if __name__ == '__main__':
    unittest.main()
