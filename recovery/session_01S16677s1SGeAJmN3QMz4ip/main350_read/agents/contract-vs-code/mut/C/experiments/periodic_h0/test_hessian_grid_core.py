"""Closed-form fields falsify missing factors, signs, covers and validation."""
import copy
import importlib.util
import unittest
from fractions import Fraction as Q
import finite_certificate as fc
import nodal_core as nc

if importlib.util.find_spec('hessian_grid_core'):
    import hessian_grid_core as hg
else:
    hg = None


def field(modes=None, dc='0'):
    return {'seed': 0, 'dc': [dc, '0'], 'modes': modes or []}


class HessianGridTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(hg, 'Certified Hessian grid bound is not implemented')
        self.qlo, self.qhi = (p/12 for p in fc.pi_bounds())

    def assertEnclosesTightly(self, actual, lower, upper, tolerance=Q(1, 10**20)):
        self.assertGreaterEqual(Q(actual), lower)
        self.assertLessEqual(Q(actual), upper+tolerance)

    def test_pure_axis_modes_preserve_scale_and_quadratic_cover(self):
        # Missing conjugate factor, q^2, the y axis, or h^2/8 breaks this.
        for mode, component in (([1, 0, '1/2', '0'], 'xx'),
                                ([0, 1, '1/2', '0'], 'yy')):
            for n in (4, 8):
                with self.subTest(mode=mode, grid=n):
                    result = hg.bound_record(field([mode]), 24, n)
                    part = result['components'][component]
                    self.assertEqual(Q(part['grid_center_abs_max']), 1)
                    self.assertEnclosesTightly(part['grid_abs_bound'], self.qlo**2, self.qhi**2)
                    self.assertEqual(Q(part['interpolation_cover']), Q(24**2, 8*n*n)*self.qhi**4)
                    self.assertEnclosesTightly(result['grid_hessian_norm_bound'], self.qlo**2, self.qhi**2)
                    cover = Q(24**2, 8*n*n)*self.qhi**4
                    self.assertEnclosesTightly(result['operator_interpolation_cover'], cover, cover)
                    for other in {'xx', 'xy', 'yy'}-{component}:
                        self.assertEqual(Q(result['components'][other]['grid_center_abs_max']), 0)
                        self.assertLess(Q(result['components'][other]['global_bound']), Q(1, 10**20))

    def test_negative_cross_frequency_keeps_signed_psd_matrix(self):
        # |kx*ky| in the PSD matrix would erase the declared cross sign.
        result = hg.bound_record(field([[1, -1, '1/2', '0']]), 24, 4)
        matrix = result['fourth_matrix']
        self.assertEqual(Q(matrix['xx']), 4*self.qhi**4)
        self.assertEqual(Q(matrix['xy']), -4*self.qhi**4)
        self.assertEqual(Q(matrix['yy']), 4*self.qhi**4)
        self.assertEnclosesTightly(matrix['operator_bound'], 8*self.qhi**4, 8*self.qhi**4)
        self.assertEnclosesTightly(result['grid_hessian_norm_bound'], 2*self.qlo**2, 2*self.qhi**2)
        self.assertEqual(Q(result['components']['xy']['interpolation_cover']), 18*self.qhi**4)

    def test_mixed_n4_closed_form_hessian_and_full_center_identity(self):
        # F=cos(xq)+2cos(yq)+3cos((x-y)q); at zero Hess/q²=[[-4,3],[3,-5]].
        record = field([[0, 1, '1', '0'], [1, -1, '3/2', '0'], [1, 0, '1/2', '0']])
        result = hg.bound_record(record, 24, 4)
        cosine = [1, 0, -1, 0]
        expected = {'xx': [], 'xy': [], 'yy': []}
        for x in range(4):
            for y in range(4):
                expected['xx'].append(-cosine[x]-3*cosine[(x-y)%4])
                expected['xy'].append(3*cosine[(x-y)%4])
                expected['yy'].append(-2*cosine[y]-3*cosine[(x-y)%4])
        for key, values in expected.items():
            digest = fc.digest([[str(v*nc.SCALE), '0'] for v in values])
            self.assertEqual(result['components'][key]['center_object_sha256'], digest)
        for key, maximum in (('xx', 4), ('xy', 3), ('yy', 5)):
            self.assertEqual(Q(result['components'][key]['grid_center_abs_max']), maximum)
        lo, hi = fc.sqrt_bounds(Q(37), bits=160)
        self.assertEnclosesTightly(result['grid_hessian_center_norm'], (9+lo)/2, (9+hi)/2)
        self.assertEnclosesTightly(result['grid_hessian_norm_bound'], self.qlo**2*(9+lo)/2, self.qhi**2*(9+hi)/2)
        self.assertEqual(Q(result['fourth_matrix']['xx']), 13*self.qhi**4)
        self.assertEqual(Q(result['fourth_matrix']['xy']), -12*self.qhi**4)
        self.assertEqual(Q(result['fourth_matrix']['yy']), 14*self.qhi**4)

    def test_n8_irrational_extremum_is_not_rounded_down(self):
        # cos(qx)+sin(qx) attains sqrt(2) at the eighth-grid node x=L/8.
        result = hg.bound_record(field([[1, 0, '1/2', '-1/2']]), 24, 8)
        lo, hi = fc.sqrt_bounds(Q(2), bits=160)
        self.assertEnclosesTightly(result['components']['xx']['grid_abs_bound'], self.qlo**2*lo, self.qhi**2*hi)
        self.assertEnclosesTightly(result['grid_hessian_norm_bound'], self.qlo**2*lo, self.qhi**2*hi)
        self.assertGreater(Q(result['grid_hessian_error_bound']), 0)

    def test_constant_field_has_only_tiny_arithmetic_allowance(self):
        # DC must not enter any derivative; zero mixed components remain covered.
        result = hg.bound_record(field(dc='123/8'), 24, 4)
        self.assertEqual(Q(result['operator_interpolation_cover']), 0)
        self.assertEqual(Q(result['grid_hessian_center_norm']), 0)
        self.assertLess(Q(result['H']), Q(1, 10**20))
        self.assertGreater(Q(result['H']), 0)

    def test_spatial_coefficient_selects_a_valid_smaller_bound(self):
        result = hg.bound_record(field([[1, 0, '1/2', '0']]), 24, 8)
        # A pure x wave's component estimate has coefficient close to H/8, not H/4.
        self.assertLess(Q(result['spatial_coefficient']), Q(result['H'])/4)
        self.assertEqual(Q(result['spatial_coefficient']),
                         (Q(result['Mxx'])+Q(result['Myy']))/8+Q(result['Mxy'])/4)
        self.assertGreaterEqual(Q(result['spatial_coefficient']), self.qlo**2/8)

    def test_original_input_validation_cannot_be_hidden_by_derivative_zeros(self):
        valid = field([[0, 1, '1/2', '0']])
        for side in (True, 0, -24, 24.0, '24'):
            with self.subTest(side=side), self.assertRaises(ValueError): hg.bound_record(valid, side, 4)
        for grid in (True, 0, 1, 3, 6, 2048, 4.0):
            with self.subTest(grid=grid), self.assertRaises(ValueError): hg.bound_record(valid, 24, grid)
        mutants = []
        for value in ('1/3', '2/4', 'NaN', True, '1/'+str(2**97)):
            item = copy.deepcopy(valid); item['modes'][0][2] = value; mutants.append(item)
        item = copy.deepcopy(valid); item['dc'][1] = '1'; mutants.append(item)
        item = copy.deepcopy(valid); item['modes'][0][1] = 2; mutants.append(item)
        item = copy.deepcopy(valid); item['modes'][0][0] = True; mutants.append(item)
        item = copy.deepcopy(valid); item['extra'] = 'unbound'; mutants.append(item)
        for item in mutants:
            with self.subTest(record=item), self.assertRaises(ValueError): hg.bound_record(item, 24, 4)


if __name__ == '__main__':
    unittest.main()
