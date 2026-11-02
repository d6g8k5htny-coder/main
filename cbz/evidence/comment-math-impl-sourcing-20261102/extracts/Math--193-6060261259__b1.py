import math
import unittest
from free_alpha import fit, power_integral

EDGES = [.001,.002,.004,.008,.016,.032,.064,.128,.256]

class FitControls(unittest.TestCase):
    def synthetic(self, alpha, A, B):
        exposure=8*24**2
        def independent_integral(a,b):
            return math.log(b/a) if alpha==1 else (b**(1-alpha)-a**(1-alpha))/(1-alpha)
        counts=[exposure*(A*independent_integral(a,b)+B*(b-a)) for a,b in zip(EDGES,EDGES[1:])]
        result=fit(EDGES,counts,exposure)
        self.assertLess(abs(result['alpha']-alpha),1e-5)
        for actual,expected in zip(result['predicted_counts'],counts):
            self.assertLess(abs(actual-expected),1e-7*max(1,abs(expected)))

    def test_target_exponent_is_free(self):
        self.synthetic(1/3,.015,.004)

    def test_non_target_and_negative_remainder(self):
        self.synthetic(.8,.001,-.001)

    def test_negative_free_exponent(self):
        self.synthetic(-.4,.02,.004)

    def test_logarithmic_bin_integral(self):
        self.assertAlmostEqual(power_integral(.01,.02,1),math.log(2),places=14)

    def test_constant_is_unidentifiable(self):
        counts=[100*.07*(b-a) for a,b in zip(EDGES,EDGES[1:])]
        result=fit(EDGES,counts,100)
        self.assertEqual(result['A'],0)
        self.assertTrue(result['constant_branch'])

    def test_invalid_counts_rejected(self):
        with self.assertRaises(ValueError):
            fit(EDGES,[-1]*8,100)

    def test_boundary_flag_is_real(self):
        counts=[100*(b**(-1.5)-a**(-1.5))/(-1.5) for a,b in zip(EDGES,EDGES[1:])]
        self.assertTrue(fit(EDGES,counts,100)['boundary_minimum'])

    def test_zero_and_malformed_inputs(self):
        self.assertTrue(fit(EDGES,[0]*8,100)['constant_branch'])
        for edges,counts,exposure in [(EDGES,[0]*7,100),(EDGES,[0]*8,0),
                ([.001,.002,.002,.004],[1]*3,100),(EDGES,[math.inf]*8,100)]:
            with self.assertRaises(ValueError):
                fit(edges,counts,exposure)

if __name__=='__main__':
    unittest.main()

