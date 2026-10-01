"""Independent exact series, shell and scope controls for the Gaussian tail."""
import importlib
import importlib.util
import unittest
from fractions import Fraction as Q
from math import factorial


class GaussianTailTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('gaussian_tail'),
                             'Gaussian spectral-tail module is not implemented')
        self.mod = importlib.import_module('gaussian_tail')

    def test_exponential_direct_series_enclosures(self):
        # Independent, unreduced alternating series for inputs <=1.
        for x in (Q(0), Q(1,1000), Q(1,8), Q(1,2), Q(1)):
            lo, hi = self.mod.exp_neg_bounds(x)
            lower = sum(((-x)**j/Q(factorial(j)) for j in range(120)), Q(0))
            upper = lower + x**120/Q(factorial(120))
            self.assertLessEqual(lo, lower)
            self.assertGreaterEqual(hi, upper)
            self.assertLess(hi-lo, Q(1,10**70))
        self.assertEqual(self.mod.exp_neg_bounds(Q(0)), (Q(1),Q(1)))

    def test_reduction_semigroup_and_tiny_values(self):
        for x in (Q(3,2), Q(9), Q(70), Q(145)):
            lo, hi = self.mod.exp_neg_bounds(x)
            a,b = self.mod.exp_neg_bounds(x/2)
            self.assertLessEqual(lo,b*b)
            self.assertGreaterEqual(hi,a*a)
            self.assertGreater(lo, 0)
            self.assertLessEqual(hi,1/(1+x))
        lo,hi = self.mod.exp_neg_bounds(Q(145))
        self.assertLess(hi,Q(1,10**62))
        self.assertGreater(lo,Q(1,10**64))

    def test_exponential_invalid_inputs(self):
        for x in (True, 1, .5, '1/2', Q(-1)):
            with self.subTest(x=x), self.assertRaises(ValueError):
                self.mod.exp_neg_bounds(x)

    def test_half_lattice_shell_identity_and_probability_multiplicity(self):
        # Exact toy weights q^(x²+y²): counts and identity do not depend on exp.
        q = Q(1,2)
        for m in range(1,9):
            half = [(x,y) for x in range(m+1) for y in range(-m,m+1)
                    if (x>0 or y>0) and max(x,abs(y))==m]
            self.assertEqual(len(half),4*m)
            s = lambda n: sum((q**(j*j) for j in range(-n,n+1)),Q(0))
            self.assertEqual(sum((q**(x*x+y*y) for x,y in half),Q(0)),
                             q**(m*m)*(s(m)+s(m-1)))
        p = self.mod.tail_bound(24,Q(8))
        e = self.mod.exp_neg_bounds(Q(32))[0]
        # Even the first shell already contains 100 Rayleigh events.
        self.assertGreaterEqual(Q(p['failure_probability_upper']),100*e)
        self.assertLess(Q(p['failure_probability_upper']),Q(13,10**13))
        self.assertGreater(Q(p['tail_supremum_upper']),Q(0))
        self.assertGreater(Q(p['infinite_remainder_upper']),Q(0))

    def test_theta_tail_excludes_no_infinite_modes(self):
        for c in (Q(1,32),Q(1,3),Q(1)):
            for m in (0,3,24):
                b = self.mod.theta_tail_upper(c,m)
                # Finite partial tail is a strict lower bound on the true tail.
                lower = 2*sum((self.mod.exp_neg_bounds(c*j*j)[0]
                               for j in range(m+1,m+12)),Q(0))
                self.assertGreaterEqual(b,lower)
        with self.assertRaises(ValueError): self.mod.tail_bound(True,Q(8))
        with self.assertRaises(ValueError): self.mod.tail_bound(24,8)
        with self.assertRaises(ValueError): self.mod.tail_bound(24,Q(0))

    def test_missing_coupling_never_becomes_zero_error(self):
        self.assertIsNone(self.mod.total_error(Q(1),Q(2),None,Q(3),Q(4)))
        self.assertEqual(self.mod.total_error(Q(1),Q(2),Q(5),Q(3),Q(4)),Q(20))
        for invalid in (True,'0',0.,Q(-1)):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                self.mod.total_error(Q(1),Q(2),invalid,Q(3),Q(4))


if __name__ == '__main__': unittest.main()
