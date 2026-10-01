"""Exact controls for the real-basis sampler's deterministic grid bridge."""
import copy
import unittest
from fractions import Fraction as Q
import nodal_core as nc
try:
    import word_polynomial as wp
except ModuleNotFoundError:
    wp = None


def polynomial():
    return {'dc': Q(3), 'modes': [(0,1,Q(2),Q(4)), (1,-1,Q(0),Q(0)),
                                  (1,0,Q(0),Q(0)), (1,1,Q(0),Q(0))]}


class WordPolynomialTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(wp, 'Exact word-polynomial bridge is missing')

    def test_real_complex_identity_at_quarter_turns(self):
        record, lift = wp.lift_record(polynomial(), 1)
        self.assertEqual(lift, 1)
        centers, _, error = nc.evaluate(record, 4)
        # 3 + 2*cos(theta) + 4*sin(theta), not twice this or the wrong sine.
        for i, expected in enumerate([5, 7, 1, -1]):
            self.assertLessEqual(abs(Q(centers[i][0], nc.SCALE)-expected), error)
        self.assertEqual(record['dc'], ['3', '0'])

    def test_smallest_exact_lift_and_no_coefficient_rounding(self):
        p = polynomial(); p['dc'] = Q(1, 1<<192)
        p['modes'][0] = (0,1,Q(1,1<<192),Q(3,1<<192))
        record, lift = wp.lift_record(p, 1)
        self.assertEqual(lift, 1<<97)
        self.assertEqual(Q(record['modes'][0][2])/lift, Q(1,1<<193))
        self.assertEqual(Q(record['modes'][0][3])/lift, -Q(3,1<<193))
        nc._input(record, 4)
        too_small = copy.deepcopy(record)
        too_small['modes'][0][2] = str(Q(record['modes'][0][2])/2)
        with self.assertRaises(ValueError): nc._input(too_small, 4)

    def test_rejects_incomplete_nondyadic_and_wrong_types(self):
        bad=[]
        p=polynomial();p['modes'].pop();bad.append(p)
        p=polynomial();p['dc']=Q(1,3);bad.append(p)
        p=polynomial();p['dc']=3;bad.append(p)
        p=polynomial();p['modes'][0]=(0,1,Q(1,1<<193),Q(0));bad.append(p)
        p=polynomial();p['modes'][0]=(True,1,Q(1),Q(0));bad.append(p)
        p=polynomial();p['modes'].reverse();bad.append(p)
        for p in bad:
            with self.subTest(p=p), self.assertRaises(ValueError):wp.lift_record(p,1)
        for k in (True,0,65,1.0):
            with self.assertRaises(ValueError):wp.lift_record(polynomial(),k)

    def test_grid_and_bound_units_survive_lifting(self):
        p=polynomial();p['dc']+=Q(1,1<<192)
        result=wp.analyze(p,1,8,8,[Q(1,10),Q(1),Q(10)])
        lift=int(result['lift']);scale=int(result['sample_scale'])
        self.assertEqual(scale,lift*nc.SCALE)
        self.assertEqual(Q(result['nodal_error']),Q(result['lifted_nodal_error'])/lift)
        self.assertEqual(Q(result['spatial_error']),Q(24**2,8**2)*Q(result['lifted_hessian']['spatial_coefficient'])/lift)
        self.assertEqual(Q(result['finite_polynomial_diagram_bound']),Q(result['nodal_error'])+Q(result['spatial_error']))
        self.assertTrue(result['independent_connectivity_check'])
        self.assertFalse(result['gaussian_law_certified'])
        with self.assertRaises(ValueError):wp.analyze(p,1,2,8,[Q(1),Q(2)])


if __name__=='__main__':unittest.main()
