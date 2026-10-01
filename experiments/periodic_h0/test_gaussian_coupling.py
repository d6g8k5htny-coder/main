"""Independent contracts for a prospective quantile coupling, not an RNG test."""
import importlib
import importlib.util
import unittest
from fractions import Fraction as Q
from math import factorial
import finite_certificate as fc
import gaussian_tail as gt


class CouplingTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('gaussian_coupling'),
                             'Finite-bit coupling has not been implemented')
        self.m=importlib.import_module('gaussian_coupling')

    def test_cdf_encloses_independent_alternating_integral(self):
        pl,ph=fc.pi_bounds()
        sl=fc.sqrt_bounds(2*pl)[0]; sh=fc.sqrt_bounds(2*ph)[1]
        for x in (Q(0),Q(1,8),Q(1,2),Q(1),Q(2)):
            low=sum(((-1)**n*x**(2*n+1)/Q(2**n*factorial(n)*(2*n+1))
                     for n in range(100)),Q(0))
            high=low+x**201/Q(2**100*factorial(100)*201)
            a,b=self.m.cdf_bounds(x)
            self.assertLessEqual(a,Q(1,2)+low/sh)
            self.assertGreaterEqual(b,Q(1,2)+high/sl)
            self.assertLess(b-a,Q(1,10**36))
            self.assertEqual(self.m.cdf_bounds(-x),(1-b,1-a))

    def test_quantiles_clipping_and_probability_residual(self):
        n=1<<128; eps=Q(1,1<<64)
        for j in (0,1,n//100,n//4,n//2-1,n//2,3*n//4,n-2,n-1):
            q=self.m.quantile(j)
            self.assertLessEqual(abs(q),8)
            self.assertEqual(q.denominator&(q.denominator-1),0)
            self.assertLessEqual(q.denominator,1<<64)
            u=Q(2*j+1,2*n)
            if q-eps>-8:self.assertLessEqual(self.m.cdf_bounds(q-eps)[0],u)
            if q+eps<8:self.assertGreaterEqual(self.m.cdf_bounds(q+eps)[1],u)
        self.assertEqual(self.m.quantile(0),Q(-8))
        self.assertEqual(self.m.quantile(n-1),Q(8))

    def test_clipping_failure_counts_all_real_gaussians(self):
        for k in (1,24,32):
            r=self.m.budget(k)
            self.assertEqual(r['real_gaussians'],(2*k+1)**2)
            lower=2*(2*k+1)**2*gt.exp_neg_bounds(Q(32))[0]/(8*fc.sqrt_bounds(2*fc.pi_bounds()[1])[1])
            self.assertGreaterEqual(Q(r['clipping_failure_upper']),lower)
            self.assertGreater(Q(r['coefficient_error_upper']),0)
            self.assertLess(Q(r['coefficient_error_upper']),Q(1,10**15))
            self.assertLess(Q(r['clipping_failure_upper']),Q(1,10**10))

    def test_uniform_precision_guarantees_bisection_termination(self):
        self.assertTrue(hasattr(self.m,'precision_contract'),'Uniform precision proof is missing')
        r=self.m.precision_contract()
        self.assertLess(Q(r['cdf_width_upper']),Q(1,1<<123))
        self.assertLess(2*Q(r['ambiguous_radius_upper']),Q(1,1<<65))
        self.assertEqual(r['maximum_ordinary_halvings'],69)

    def test_fourier_weight_multiplicities_and_errors(self):
        r=self.m.weights(2)
        self.assertEqual(len(r['modes']),12)
        errors=r['dc_error']+2*sum((row[3] for row in r['modes']),Q(0))
        self.assertEqual(errors,r['aggregate_error'])
        self.assertGreater(errors,0)
        for _,_,w,error in r['modes']:
            self.assertGreater(w,0);self.assertGreater(error,0)
            self.assertEqual(w.denominator&(w.denominator-1),0)

    def test_fixture_is_a_complete_deterministic_polynomial(self):
        words=[i*((1<<128)-1)//8 for i in range(9)]
        r=self.m.polynomial(1,words)
        self.assertEqual(len(r['modes']),4)
        self.assertEqual([(a,b) for a,b,*_ in r['modes']],[(0,1),(1,-1),(1,0),(1,1)])
        for q in [r['dc'],*[q for row in r['modes'] for q in row[2:]]]:
            self.assertEqual(q.denominator&(q.denominator-1),0)
        self.assertLessEqual(abs(r['dc'])+sum((abs(a)+abs(b) for _,_,a,b in r['modes']),Q(0)),Q(self.m.budget(1)['polynomial_norm_upper']))

    def test_invalid_types_do_not_hit_cached_valid_results(self):
        self.m.cdf_bounds(Q(1));self.m.quantile(1);self.m.weights(1)
        for x in (True,1,1.0,'1',Q(9)):
            with self.subTest(cdf=x),self.assertRaises(ValueError):self.m.cdf_bounds(x)
        for x in (True,1.,'1',-1,1<<128):
            with self.subTest(word=x),self.assertRaises(ValueError):self.m.quantile(x)
        for x in (True,1.,'1',0,65):
            with self.subTest(cutoff=x),self.assertRaises(ValueError):self.m.weights(x)
        with self.assertRaises(ValueError):self.m.polynomial(1,[0]*8)
        with self.assertRaises(ValueError):self.m.polynomial(1,[False]*9)


if __name__=='__main__':unittest.main()
