"""Independent area/moment oracles and actual periodic-field H0 witnesses."""
import importlib
import importlib.util
import math
import unittest
from collections import Counter
from fractions import Fraction as Q


def triangle_oracle(t):
    # Inside [0,t]^2: the lower triangle u+v<=t, and the upper
    # corner u+v>=1 when 2t>1. These disjoint triangles have known area.
    triple = (t*t + max(2*t-1, Q(0))**2)/2
    # Three uniform marginals and three independent pairs determine
    # the remaining masses from this independently computed triple area.
    return (1-3*t+3*t*t-triple, 3*t-6*t*t+3*triple,
            3*t*t-3*triple, triple)


def binomial_oracle(t):
    return tuple(Q(math.comb(3, k))*t**k*(1-t)**(3-k) for k in range(4))


class PairwiseFieldControls(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('pairwise_field_controls'),
                             'Exact pairwise finite-field control is not implemented')
        self.c = importlib.import_module('pairwise_field_controls')

    def test_quarter_laws_distinguish_the_actual_field_copulas(self):
        self.assertEqual(self.c.population_law(Q(1, 4), world='dependent'),
                         tuple(Q(n, 32) for n in (13, 15, 3, 1)))
        self.assertEqual(self.c.population_law(Q(1, 4), world='iid'),
                         tuple(Q(n, 64) for n in (27, 27, 9, 1)))
        self.assertEqual(self.c.dependent_peaks(Q(1, 8), Q(1, 8)),
                         (Q(1, 8), Q(1, 8), Q(1, 4)))
        self.assertEqual(self.c.dependent_peaks(Q(3, 4), Q(3, 4)),
                         (Q(3, 4), Q(3, 4), Q(1, 2)))
        # Off-diagonal order, the inclusive lower boundary and the exact wrap.
        self.assertEqual(self.c.dependent_peaks(Q(1, 3), Q(1, 5)),
                         (Q(1, 3), Q(1, 5), Q(8, 15)))
        self.assertEqual(self.c.dependent_peaks(Q(0), Q(1, 2)),
                         (Q(0), Q(1, 2), Q(1, 2)))
        self.assertEqual(self.c.dependent_peaks(Q(1, 2), Q(1, 2)),
                         (Q(1, 2), Q(1, 2), Q(0)))

    def test_independent_triangle_and_binomial_oracles(self):
        for t in map(Q, (0, 1)):
            for world in ('dependent', 'iid'):
                self.assertEqual(self.c.population_law(t, world=world), binomial_oracle(t))
        for t in (Q(1, 7), Q(1, 4), Q(1, 3), Q(1, 2), Q(2, 3), Q(3, 4), Q(6, 7)):
            dependent = self.c.population_law(t, world='dependent')
            independent = self.c.population_law(t, world='iid')
            self.assertEqual(dependent, triangle_oracle(t), t)
            self.assertEqual(independent, binomial_oracle(t), t)
            self.assertTrue(all(type(p) is Q and p >= 0 for p in dependent+independent))
            self.assertEqual(sum(dependent), 1)
            self.assertEqual(sum(independent), 1)

    def test_moments_and_median_positive_control(self):
        for t in (Q(0), Q(1, 4), Q(1, 2), Q(2, 3), Q(1)):
            for world in ('dependent', 'iid'):
                law = self.c.population_law(t, world=world)
                self.assertEqual(sum(k*p for k, p in enumerate(law)), 3*t)
                self.assertEqual(sum(math.comb(k, 2)*p for k, p in enumerate(law)), 3*t*t)
                self.assertEqual(sum(k*(k-1)*p for k, p in enumerate(law)), 6*t*t)
        expected = tuple(Q(n, 8) for n in (1, 3, 3, 1))
        self.assertEqual(self.c.population_law(Q(1, 2), world='dependent'), expected)
        self.assertEqual(self.c.population_law(Q(1, 2), world='iid'), expected)

    def test_polygon_areas_cover_the_square_and_representatives_are_interior(self):
        for t in (Q(0), Q(1, 4), Q(1, 2), Q(2, 3), Q(1)):
            cells = self.c.dependent_cells(t)
            self.assertTrue(cells)
            self.assertEqual(sum(self.c.polygon_area(cell) for cell in cells), 1)
            patterns = set()
            for cell in cells:
                self.assertGreater(self.c.polygon_area(cell), 0)
                self.assertEqual(self.c.polygon_area(tuple(reversed(cell))),
                                 self.c.polygon_area(cell))
                u = sum(p[0] for p in cell)/len(cell)
                v = sum(p[1] for p in cell)/len(cell)
                self.assertTrue(0 < u < 1 and 0 < v < 1)
                signs = (u-t, v-t, u+v-t, u+v-1, u+v-1-t)
                self.assertTrue(all(s != 0 for s in signs))
                pattern = tuple(s > 0 for s in signs)
                self.assertNotIn(pattern, patterns)
                patterns.add(pattern)
        self.assertEqual(self.c.polygon_area(((Q(0), Q(0)), (Q(1), Q(0)),
                                               (Q(0), Q(1)))), Q(1, 2))
        self.assertEqual(self.c.polygon_area(()), 0)

    def test_all_sixteen_translation_permutations_are_realized(self):
        peaks = (Q(1, 5), Q(2, 5), Q(4, 5))
        unshifted = {(0, 0): Q(27), (0, 2): peaks[0],
                     (2, 0): peaks[1], (2, 2): peaks[2]}
        samples, essential_positions = [], set()
        for sx in range(4):
            for sy in range(4):
                values, scale = self.c.build_field(peaks, shift=(sx, sy))
                actual = tuple(Q(n, scale) for n in values)
                expected = tuple(unshifted.get(((x-sx) % 4, (y-sy) % 4), Q(0))
                                 for x in range(4) for y in range(4))
                self.assertEqual(actual, expected)
                essential_positions.add(actual.index(Q(27)))
                samples.append(actual)
        self.assertEqual(essential_positions, set(range(16)))
        for site in range(16):
            self.assertEqual(Counter(row[site] for row in samples),
                             Counter({Q(0): 12, Q(27): 1, **{p: 1 for p in peaks}}))

    def test_actual_complete_global_barcodes_and_integer_units(self):
        for peaks in ((Q(1, 3), Q(2, 5), Q(7, 11)),
                      (Q(1, 2), Q(1, 2), Q(1)), (Q(0), Q(1, 2), Q(1))):
            for sx in range(4):
                for sy in range(4):
                    values, scale = self.c.build_field(peaks, shift=(sx, sy))
                    self.assertTrue(type(scale) is int and scale > 0)
                    self.assertTrue(all(type(n) is int for n in values))
                    # Pin the exact rescaling independently of the implementation's own scale.
                    self.assertEqual(sorted(Q(n, scale) for n in values),
                                     sorted([Q(0)]*12 + [Q(27), *peaks]))
                    barcode = self.c.exact_h0.compute(values, 4)
                    self.assertTrue(self.c.exact_h0.verify_by_connectivity(values, 4, barcode))
                    expected = sorted([[int(p*scale), 0] for p in peaks if p > 0])
                    self.assertEqual(barcode['intervals'], expected)
                    self.assertEqual(barcode['essential'], [27*scale])
                    self.assertEqual(barcode['essential_vertices'], [sx*4+sy])
                    self.assertEqual(barcode['zero_count'], 15-len(expected))

    def test_strict_domains_survive_optimization(self):
        for t in (0, 0.25, True, '1/4', Q(-1, 4), Q(5, 4), None):
            for name in ('dependent_cells', 'population_law'):
                with self.assertRaises(ValueError):
                    getattr(self.c, name)(t)
            with self.assertRaises(ValueError):
                self.c.population_law(t, world='iid')
        for world in (None, True, 'independent', 'gaussian', 0):
            with self.assertRaises(ValueError):
                self.c.population_law(Q(1, 4), world=world)
        for peaks in ((Q(1, 2),)*2, [0, Q(1, 2), Q(1)],
                      (Q(-1), Q(1, 2), Q(1)), (Q(2), Q(1, 2), Q(1)), None,
                      (Q(1, 2), Q(-1), Q(1)), (Q(1, 2), 0, Q(1)), (Q(1, 2), Q(1, 2), Q(2))):
            with self.assertRaises(ValueError):
                self.c.build_field(peaks)
        for shift in ((True, 0), (0, 4), (-1, 0), (0,), [0, 0], None):
            with self.assertRaises(ValueError):
                self.c.build_field((Q(1, 2),)*3, shift=shift)
        for u in (0, True, Q(-1), Q(1)):
            with self.assertRaises(ValueError):
                self.c.dependent_peaks(u, Q(1, 2))
            with self.assertRaises(ValueError):
                self.c.dependent_peaks(Q(1, 2), u)
        with self.assertRaises(ValueError):
            self.c.polygon_area(((0, Q(0)), (Q(1), Q(0)), (Q(0), Q(1))))

    def test_report_preserves_population_and_scientific_scope(self):
        report = self.c.report()
        self.assertEqual(len(report), 12)
        self.assertEqual(report['model'], 'stationary discrete 4x4 periodic vertex-cubical fields')
        self.assertEqual(report['field_pair_law_identity'],
                         'analytic Haar shear and conditional translation')
        self.assertEqual(report['dependent_quarter'], ['13/32', '15/32', '3/32', '1/32'])
        self.assertEqual(report['iid_quarter'], ['27/64', '27/64', '9/64', '1/64'])
        self.assertEqual(report['mean_count_quarter'], '3/4')
        self.assertEqual(report['expected_unordered_pairs_quarter'], '3/16')
        self.assertEqual(report['second_factorial_moment_quarter'], '3/8')
        self.assertEqual(report['translation_group_size'], 16)
        self.assertFalse(report['representatives_are_field_law_quadrature'])
        self.assertFalse(report['gaussian_or_continuum_admission'])
        self.assertFalse(report['blind_or_novel_discovery'])
        self.assertEqual(report['scientific_effect'], 'NONE')


if __name__ == '__main__':
    unittest.main()
