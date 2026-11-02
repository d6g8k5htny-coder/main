#!/usr/bin/env python3
"""Independent scalar diagnostic for S8; no repository code or random field is run.

The mathematical argument is in REPORT.md. Finite checks corroborate it, not
replace the general proof. --variant selects a deliberately false diagnostic
formula; it does not mutate the manuscript or any project file.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import itertools
import json
from pathlib import Path
import platform
import sys
import unittest

VARIANT = 'baseline'
VALUES = (F(-2), F(-1, 3), F(0), F(1, 2), F(3))


def observed_weight(w: F, h: F, u: F, v: F) -> F:
    product = (w - h + u) * (w + h + v)
    return abs(product) if VARIANT == 'drop_sign_selection' else max(F(0), -product)


def square_identity(w: F, h: F, u: F, v: F) -> F:
    center = w + (u + v) / 2
    difference = h + (u - v if VARIANT == 'reverse_residual_difference' else v - u) / 2
    return center * center - difference * difference


def sharp_bound(h: F, epsilon: F) -> F:
    return h * h if VARIANT == 'omit_residual_budget' else (abs(h) + epsilon / 2) ** 2


def tail(ell: F, radius: F) -> F:
    power = 6 if VARIANT == 'wrong_tail_power' else 7
    return ell * ell / (F(7) * radius ** power)


def cap(ell: F, radius: F) -> F:
    power = 4 if VARIANT == 'wrong_cap_power' else 5
    return radius ** power / (F(5) * ell)


class ScalarChecks(unittest.TestCase):
    def test_difference_of_squares_identity(self):
        for w, h, u, v in itertools.product(VALUES, repeat=4):
            self.assertEqual((w-h+u)*(w+h+v), square_identity(w,h,u,v))

    def test_uniform_sharp_and_quadratic_bounds(self):
        for w, h, u, v in itertools.product(VALUES, repeat=4):
            eps = abs(u) + abs(v)
            weight = observed_weight(w,h,u,v)
            self.assertLessEqual(weight, sharp_bound(h,eps))
            self.assertLessEqual(sharp_bound(h,eps), 2*h*h+eps*eps/2)

    def test_sharpness_and_boundary(self):
        for h in VALUES:
            sign = F(-1) if h < 0 else F(1)
            for eps in (F(0), F(1,7), F(1), F(5)):
                u, v = -sign*eps/2, sign*eps/2
                self.assertEqual(observed_weight(F(0),h,u,v), sharp_bound(h,eps))
                for w in (h-u, -h-v):
                    self.assertEqual(observed_weight(w,h,u,v), 0)

    def test_zero_gap_r4_and_no_residual_division(self):
        for r in (F(1,2), F(1,4), F(1,16), F(1,64)):
            eps = r*r
            self.assertEqual(observed_weight(F(0),F(0),-eps/2,eps/2), r**4/4)
            for w in VALUES:
                self.assertLessEqual(observed_weight(w,F(0),-eps/2,eps/2), eps**2/4)

    def test_missing_common_term_is_an_actual_counterexample(self):
        # These scalar values can be determinants of diag(-r,-1) and diag(r,-1).
        # No assertion of realizability by the pinned Gaussian field is made.
        for r in (F(1,2),F(1,4),F(1,8),F(1,16)):
            independent_product = r*(-r)
            self.assertGreater(-independent_product, sharp_bound(F(0),F(0)))
            self.assertEqual((-independent_product)/(r**4), 1/(r*r))

    def test_exact_integrals_and_both_cutoff_regions(self):
        for t in (F(1),F(1,2),F(1,3),F(2,5)):
            ell, a = t**12, t**3
            self.assertEqual(tail(ell,a), a/7)
            self.assertEqual(cap(ell,a), a/5)
            for rho in (a/2,a,(a+t)/2,t):
                near_cap = min(a,rho)
                self.assertLessEqual(cap(ell,near_cap), a/5)
                if rho > a:
                    middle = ell**2*(a**-7-rho**-7)/7 + (rho**3-a**3)/3
                    self.assertGreaterEqual(middle,0)
                    self.assertLessEqual(middle,a/7+rho**3/3)
                    # Split positive tail exactly at rho.
                    self.assertEqual(tail(ell,a), ell**2*(a**-7-rho**-7)/7+tail(ell,rho))
                else:
                    middle=F(0)
                self.assertLessEqual(near_cap+middle+tail(ell,rho),
                                     F(8,7)*min(a,rho)+rho**3/3+tail(ell,rho))
                # Direct antiderivative of the equal-height r^2 envelope.
                self.assertEqual(rho**3/3, F(1,3)*rho**3)

    def test_eight_term_rate_and_relative_exponent(self):
        alpha=F(1,8) if VARIANT=='wrong_cutoff_exponent' else F(1,12)
        exponents=[max(F(1,4),alpha),3*alpha,2-7*alpha,F(1,4),3*alpha,
                   F(1,3)-alpha,F(4,3)-5*alpha,F(1,3)]
        self.assertEqual(exponents,[F(1,4),F(1,4),F(17,12),F(1,4),F(1,4),
                                    F(1,4),F(11,12),F(1,3)])
        self.assertEqual(exponents.count(F(1,4)),5)
        self.assertEqual(F(1,4)+F(1,3),F(7,12))
        for t in (F(1),F(1,2),F(1,3),F(2,5),F(1,8)):
            rhs=5*t**3+t**17+t**11+t**4
            self.assertLessEqual(rhs,8*t**3)

    def test_power_law_balance_is_unique_for_these_envelopes(self):
        # Analytic proof: 3a >= 1/4 and 1/3-a >=1/4 force a=1/12.
        # The grid only corroborates this exact pair of inequalities.
        for alpha in (F(j,120) for j in range(1,60)):
            exponents=[F(1,4),3*alpha,2-7*alpha,F(1,3)-alpha,F(4,3)-5*alpha,F(1,3)]
            self.assertEqual(min(exponents)>=F(1,4),alpha==F(1,12))


def main() -> int:
    global VARIANT
    choices=['baseline','drop_sign_selection','reverse_residual_difference',
             'omit_residual_budget','wrong_tail_power','wrong_cap_power','wrong_cutoff_exponent']
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=choices,default='baseline')
    args=parser.parse_args()
    VARIANT=args.variant
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ScalarChecks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report={
        'variant':VARIANT,'python':platform.python_version(),'optimize':sys.flags.optimize,
        'no_site':sys.flags.no_site,'dont_write_bytecode':sys.dont_write_bytecode,
        'tests_run':result.testsRun,'failures':[x.id() for x,_ in result.failures],
        'errors':[x.id() for x,_ in result.errors],'skips':len(result.skipped),
        'successful':result.wasSuccessful(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'Independent exact scalar diagnostic; no project code, field simulation, or theorem acceptance.',
    }
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0 if result.wasSuccessful() else 1

if __name__=='__main__':
    raise SystemExit(main())
