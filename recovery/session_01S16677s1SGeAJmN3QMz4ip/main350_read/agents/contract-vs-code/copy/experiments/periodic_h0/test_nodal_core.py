"""Independent Fourier identities and error controls for integer nodal replay."""
import copy
import importlib.util
import unittest
from fractions import Fraction as Q
import finite_certificate as fc

if importlib.util.find_spec('nodal_core'):
    import nodal_core as nc
else:
    nc = None


def field(modes=None, dc='0'):
    return {'seed': 0, 'dc': [dc, '0'], 'modes': [] if modes is None else modes}


class NodalCoreTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(nc, 'Certified integer nodal core is not implemented')

    def test_nearest_integer_including_negative_halfway_cases(self):
        for numerator in range(-101, 102):
            for denominator in (1, 2, 3, 7, 17):
                q = Q(numerator, denominator)
                rounded = nc.nearest_integer(q)
                self.assertIs(type(rounded), int)
                self.assertLessEqual(abs(q-rounded), Q(1, 2))
        self.assertEqual(nc.nearest_integer(Q(1, 2)), 1)
        self.assertEqual(nc.nearest_integer(Q(-1, 2)), 0)

    def test_quadrant_twiddles_are_exact(self):
        s = nc.SCALE
        self.assertEqual([nc.twiddle_center(k, 4) for k in range(4)],
                         [(s, 0), (0, s), (-s, 0), (0, -s)])
        self.assertEqual(nc.twiddle_center(-1, 4), (0, -s))

    def test_irrational_roots_enclose_independent_square_root_formulas(self):
        rho = nc.twiddle_component_error()
        self.assertEqual(rho, Q(1, nc.SCALE))
        sqrt2lo, sqrt2hi = fc.sqrt_bounds(Q(2), bits=160)
        c8lo = fc.sqrt_bounds((2+sqrt2lo)/4, bits=160)[0]
        c8hi = fc.sqrt_bounds((2+sqrt2hi)/4, bits=160)[1]
        s8lo = fc.sqrt_bounds((2-sqrt2hi)/4, bits=160)[0]
        s8hi = fc.sqrt_bounds((2-sqrt2lo)/4, bits=160)[1]
        cases = [(1, 8, ((sqrt2lo/2, sqrt2hi/2), (sqrt2lo/2, sqrt2hi/2))),
                 (1, 16, ((c8lo, c8hi), (s8lo, s8hi))),
                 (3, 16, ((s8lo, s8hi), (c8lo, c8hi)))]
        for k, n, expected in cases:
            for center, (lo, hi) in zip(nc.twiddle_center(k, n), expected):
                self.assertLessEqual(Q(center, nc.SCALE)-rho, lo)
                self.assertGreaterEqual(Q(center, nc.SCALE)+rho, hi)

    def test_unnormalized_dc_for_two_axes(self):
        for n in (2, 4, 8):
            centers, stages, error = nc.evaluate(field(dc='3/8'), n)
            self.assertEqual(centers, [(3*nc.SCALE//8, 0)]*(n*n))
            self.assertEqual(len(stages), 2*(n.bit_length()-1))
            self.assertIs(type(error), Q)
            self.assertGreaterEqual(error, 0)

    def test_n4_mixed_sine_cosine_and_axes_against_closed_formula(self):
        record = field([[0, 1, '0', '-1/2'], [1, 0, '1/2', '0'],
                        [1, 1, '3/8', '1/2']], dc='3/8')
        centers, stages, error = nc.evaluate(record, 4)
        cos = [1, 0, -1, 0]; sin = [0, 1, 0, -1]
        for x in range(4):
            for y in range(4):
                expected = Q(3, 8)+cos[x]+sin[y]+Q(3, 4)*cos[(x+y)%4]-sin[(x+y)%4]
                real, imag = centers[x*4+y]
                self.assertEqual(Q(real, nc.SCALE), expected)
                self.assertEqual(imag, 0)
        self.assertEqual([s['axis'] for s in stages], ['rows', 'rows', 'columns', 'columns'])

    def test_n8_cosine_with_independent_irrational_values(self):
        centers, _, error = nc.evaluate(field([[1, 0, '1/2', '0']]), 8)
        lower, upper = fc.sqrt_bounds(Q(1, 2), bits=160)
        positive = {1, 7}; negative = {3, 5}
        for x in range(8):
            for y in range(8):
                real, imag = centers[x*8+y]
                if x in positive: lo, hi = lower, upper
                elif x in negative: lo, hi = -upper, -lower
                else: lo = hi = Q({0:1, 2:0, 4:-1, 6:0}[x])
                self.assertLessEqual(Q(real, nc.SCALE)-error, lo)
                self.assertGreaterEqual(Q(real, nc.SCALE)+error, hi)
                self.assertLessEqual(abs(Q(imag, nc.SCALE)), error)

    def test_stage_majorants_follow_the_proved_rational_recurrence(self):
        centers, stages, final_error = nc.evaluate(field([[0, 1, '1/8', '-1/4'], [1, -1, '3/8', '1/2']]), 8)
        error = Q(0)
        for stage in stages:
            self.assertIs(type(stage['input_l1_integer_max']), int)
            self.assertGreaterEqual(stage['input_l1_integer_max'], 0)
            error = Q(5, 2)*error+Q(stage['input_l1_integer_max'], nc.SCALE**2)+Q(1, 2*nc.SCALE)
            self.assertEqual(Q(stage['error_bound']), error)
        self.assertEqual(error, final_error)
        self.assertEqual(len(centers), 64)

    def test_n8_full_mixed_field_against_rational_plus_sqrt_half_formula(self):
        record = field([[0, 1, '1/8', '-1/4'], [1, -1, '3/8', '1/2'],
                        [2, 1, '-1/2', '1/8'], [3, -3, '1/16', '-3/16']], dc='-3/8')
        centers, _, error = nc.evaluate(record, 8)
        # Each entry is A+B/sqrt(2); this independent table covers all octants.
        cosine = [(1,0),(0,1),(0,0),(0,-1),(-1,0),(0,-1),(0,0),(0,1)]
        sine = [(0,0),(0,1),(1,0),(0,1),(0,0),(0,-1),(-1,0),(0,-1)]
        lower, upper = fc.sqrt_bounds(Q(1, 2), bits=160)
        for x in range(8):
            for y in range(8):
                a, b = Q(record['dc'][0]), Q(0)
                for kx, ky, real, imag in record['modes']:
                    phase = (kx*x+ky*y)%8
                    a += 2*(Q(real)*cosine[phase][0]-Q(imag)*sine[phase][0])
                    b += 2*(Q(real)*cosine[phase][1]-Q(imag)*sine[phase][1])
                lo, hi = (a+b*lower, a+b*upper) if b >= 0 else (a+b*upper, a+b*lower)
                real, imag = centers[x*8+y]
                self.assertLessEqual(Q(real, nc.SCALE)-error, lo)
                self.assertGreaterEqual(Q(real, nc.SCALE)+error, hi)
                self.assertLessEqual(abs(Q(imag, nc.SCALE)), error)

    def test_input_mutants_fail_before_transform(self):
        valid = field([[0, 1, '0', '1/2'], [1, 0, '1/2', '0']])
        for n in (True, 4.0, 0, 1, 3, 6, 2048):
            with self.subTest(n=n), self.assertRaises(ValueError): nc.evaluate(valid, n)
        mutants = []
        for value in ('NaN', '1/3', '2/4', True, '1/'+str(2**97)):
            data = copy.deepcopy(valid); data['modes'][0][2] = value; mutants.append(data)
        data = copy.deepcopy(valid); data['dc'][1] = '1'; mutants.append(data)
        data = copy.deepcopy(valid); data['modes'].append(data['modes'][0]); mutants.append(data)
        data = copy.deepcopy(valid); data['modes'].reverse(); mutants.append(data)
        data = copy.deepcopy(valid); data['modes'][0][0] = True; mutants.append(data)
        data = copy.deepcopy(valid); data['modes'][0][1] = -1; mutants.append(data)
        data = copy.deepcopy(valid); data['modes'][1][0] = -1; mutants.append(data)
        data = copy.deepcopy(valid); data['modes'][1][0] = 2; mutants.append(data)
        data = copy.deepcopy(valid); data['seed'] = True; mutants.append(data)
        data = copy.deepcopy(valid); data['extra'] = 'unbound'; mutants.append(data)
        for data in mutants:
            with self.subTest(data=data), self.assertRaises(ValueError): nc.evaluate(data, 4)


if __name__ == '__main__':
    unittest.main()
