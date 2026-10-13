#!/usr/bin/env python3
"""Exact rational tests for the separately proposed two-atom D2 floors."""
from fractions import Fraction as F
import importlib.util
import itertools
import json
from pathlib import Path
import sys
import unittest

SOURCE = Path(__file__).with_name("two_atom_bounds.py")
COUNTS = {}

def counted(label, n=1):
    COUNTS[label] = COUNTS.get(label, 0) + n

class TwoAtomTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SOURCE.is_file(), "quantitative implementation must exist")
        if not hasattr(type(self), "m"):
            spec = importlib.util.spec_from_file_location("two_atom_bounds", SOURCE)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            type(self).m = module

    def test_weighted_square_identity_and_optimum(self):
        for A, B, u, v, c in itertools.product(
            (F(1, 5), F(1), F(7, 3)), (F(1, 4), F(2)),
            (F(-2), F(0), F(3)), (F(-1), F(1), F(5)),
            (F(-3), F(0), F(2, 3), F(4))):
            lhs = (A+B)*(A*(u-c)**2+B*(v-c)**2)-A*B*(u-v)**2
            rhs = (A*(u-c)+B*(v-c))**2
            self.assertEqual(lhs, rhs)
            floor = self.m.two_square_floor(A, B, u, v)
            self.assertGreaterEqual(A*(u-c)**2+B*(v-c)**2, floor)
            mean = (A*u+B*v)/(A+B)
            self.assertEqual(A*(u-mean)**2+B*(v-mean)**2, floor)
            counted("two_square")

    def test_finite_probability_laws_and_all_d2_floors(self):
        for a, b in ((F(1), F(2)), (F(-1), F(3)),
                     (F(1, 2), F(-3, 2)), (F(-2), F(-3))):
            for p, q in ((F(1, 8), F(1, 4)), (F(1, 3), F(1, 3)),
                         (F(1, 5), F(4, 5)), (F(2, 5), F(1, 10))):
                for extra in (F(0), F(-4), F(2, 3), F(5)):
                    law = [(a,p), (b,q), (extra,1-p-q)]
                    moments = self.m.even_moments(law)
                    m2, m4, m6 = moments
                    floors = self.m.atom_floors(p, q, a, b)
                    delta = m6-m4*m4/m2
                    gap = m4-m2*m2
                    self.assertGreaterEqual(m2, floors["m2"])
                    self.assertGreaterEqual(gap, floors["gap"])
                    self.assertGreaterEqual(delta, floors["delta"])
                    c = m4/m2
                    residual = sum(w*(x**3-c*x)**2 for x,w in law)
                    self.assertEqual(residual, delta)
                    for direction in (F(0), F(1,100), F(1,16), F(1,8), F(3,16), F(1,4)):
                        D,S,T = self.m.d2_values(m2,m4,m6,direction)
                        self.assertGreaterEqual(D, floors["det"])
                        self.assertGreaterEqual(S, floors["schur"])
                        self.assertGreaterEqual(T, floors["tau"])
                        self.assertGreater(floors["det"],0)
                        self.assertGreater(floors["schur"],0)
                        self.assertGreater(floors["tau"],0)
                        counted("directional_floor")
                    counted("probability_law")

    def test_mass_lower_bounds_not_exact_masses(self):
        for p, q, ap, aq in ((F(1,10),F(1,10),F(1,4),F(1,2)),
                              (F(1,6),F(1,5),F(1,3),F(1,3))):
            lower = self.m.atom_floors(p,q,F(1),F(3))
            actual = self.m.atom_floors(ap,aq,F(1),F(3))
            m2,m4,m6 = self.m.even_moments([(F(1),ap),(F(3),aq),(F(0),1-ap-aq)])
            for label in lower:
                self.assertLessEqual(lower[label], actual[label])
            self.assertLessEqual(lower["delta"], m6-m4*m4/m2)
            counted("mass_lower_bound")

    def test_delta_floor_attains_with_zero_remainder(self):
        for a,b,p,q in itertools.product(
            (F(1),F(-1,2)), (F(2),F(-3)),
            (F(1,10),F(1,3)), (F(1,4),F(1,2))):
            m2,m4,m6 = self.m.even_moments([(a,p),(b,q),(F(0),1-p-q)])
            bounds = self.m.atom_floors(p,q,a,b)
            self.assertEqual(m2,bounds["m2"])
            self.assertEqual(m6-m4*m4/m2,bounds["delta"])
            counted("delta_attainment")

    def test_gap_floor_attains_at_intermediate_squared_radius(self):
        # u=1, v=9; weights in ratio 5:3 give intermediate squared radius 4.
        for s in (F(1,4),F(1,2),F(1)):
            p,q = s*F(5,8),s*F(3,8)
            m2,m4,_ = self.m.even_moments([(F(1),p),(F(3),q),(F(2),1-s)])
            bounds = self.m.atom_floors(p,q,F(1),F(3))
            self.assertEqual(m2,F(4))
            self.assertEqual(m4-m2*m2,bounds["gap"])
            counted("gap_attainment")

    def test_scaling_has_correct_degrees(self):
        base=self.m.atom_floors(F(1,5),F(1,3),F(1),F(-2))
        degrees={"m2":2,"gap":4,"delta":6,"det":8,"schur":4,"tau":6}
        for c in (F(1,3),F(2),F(-3)):
            scaled=self.m.atom_floors(F(1,5),F(1,3),c,-2*c)
            for key,power in degrees.items():
                self.assertEqual(scaled[key],c**power*base[key])
                counted("scale_identity")

    def test_probability_normalization_is_essential(self):
        law=[(F(1),F(1)),(F(2),F(1))]
        m2,m4,m6=self.m.even_moments(law,probability=False)
        total=F(2)
        c=m2/total
        centered=sum(w*(x*x-c)**2 for x,w in law)
        self.assertEqual(centered,m4-m2*m2/total)
        self.assertLess(m4-m2*m2,0)
        self.assertGreater(centered,0)
        self.assertEqual(m6-m4*m4/m2,
                         sum(w*(x**3-(m4/m2)*x)**2 for x,w in law))
        with self.assertRaisesRegex(ValueError,"probability mass"):
            self.m.even_moments(law)
        counted("nonunit_mass")

    def test_zero_and_same_radius_are_not_strict_cases(self):
        m2,m4,m6=self.m.even_moments([(F(0),F(1,2)),(F(1),F(1,4)),(F(-1),F(1,4))])
        self.assertGreater(m4-m2*m2,0)
        self.assertEqual(m6-m4*m4/m2,0)
        for a,b in ((F(0),F(1)),(F(1),F(-1)),(F(2),F(2))):
            with self.assertRaisesRegex(ValueError,"nonzero distinct squared"):
                self.m.atom_floors(F(1,4),F(1,4),a,b)
            counted("excluded_atom_case")

    def test_invalid_domains_and_float_inputs_refused(self):
        for p,q in ((F(0),F(1,3)),(F(-1,3),F(1,3)),(F(1),F(1,2))):
            with self.assertRaises(ValueError):
                self.m.atom_floors(p,q,F(1),F(2))
            counted("invalid_domain")
        with self.assertRaises(TypeError):
            self.m.atom_floors(0.25,F(1,4),F(1),F(2))
        with self.assertRaises(ValueError):
            self.m.d2_values(F(1),F(3),F(15),F(1,3))
        with self.assertRaises(ValueError):
            self.m.d2_values(F(1),F(1),F(1),F(1,4))
        with self.assertRaises(ValueError):
            self.m.even_moments([(F(1),F(-1))])
        counted("invalid_domain",4)

    def test_wrong_missing_factor_and_normalizer_fail_on_exact_witnesses(self):
        p=q=F(1,4); a=F(1,10); b=F(1,5)
        m2,m4,m6=self.m.even_moments([(a,p),(b,q),(F(0),1-p-q)])
        correct=self.m.atom_floors(p,q,a,b)["delta"]
        actual=m6-m4*m4/m2
        wrong_missing_ab = p*q*(a*a-b*b)**2/(p*a*a+q*b*b)
        wrong_missing_mass_denominator = correct/(p+q)
        self.assertEqual(actual,correct)
        self.assertGreater(wrong_missing_ab,actual)
        self.assertGreater(wrong_missing_mass_denominator,actual)
        counted("wrong_formula_witness",2)

    def test_endpoint_max_cannot_replace_lower_min(self):
        p=q=F(1,4);a=F(1);b=F(2)
        m2,m4,m6=self.m.even_moments([(a,p),(b,q),(F(0),F(1,2))])
        lower=self.m.atom_floors(p,q,a,b)
        # Actual tau endpoints; using their maximum as an all-angle lower bound fails.
        T0=self.m.d2_values(m2,m4,m6,F(0))[2]
        T1=self.m.d2_values(m2,m4,m6,F(1,4))[2]
        self.assertNotEqual(T0,T1)
        self.assertGreater(max(T0,T1),min(T0,T1))
        self.assertLessEqual(lower["tau"],min(T0,T1))
        counted("wrong_extremum_witness")

    def test_scalar_formula_matches_source_subtractive_schur(self):
        for a,b in ((F(1),F(2)),(F(-1,2),F(3,2))):
            m2,m4,m6=self.m.even_moments([(a,F(1,4)),(b,F(1,3)),(F(0),F(5,12))])
            for t in (F(0),F(1,16),F(1,8),F(1,4)):
                k4=m4-3*m2*m2
                alpha=m4-2*k4*t
                gamma=m2*m2+2*k4*t
                D=m2*m2*m4+k4*(m4+m2*m2)*t
                subtractive=alpha-(gamma**3+(2*gamma+alpha)*k4*k4*t*(1-4*t))/D
                self.assertEqual(self.m.d2_values(m2,m4,m6,t)[1],subtractive)
                counted("subtractive_schur")

if __name__=="__main__":
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(TwoAtomTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({"tests":result.testsRun,"failures":len(result.failures),
        "errors":len(result.errors),"exact_cases":COUNTS},sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)
