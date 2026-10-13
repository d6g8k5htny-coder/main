"""Exact graph controls, including an independent all-threshold rank oracle."""
import copy
import importlib
import importlib.util
import itertools
import random
import unittest
from fractions import Fraction as Q


def axis_graph(n):
    """Build undirected edges independently of the production neighbor code."""
    graph = [set() for _ in range(n*n)]
    for x in range(n):
        for y in range(n):
            u = x*n+y
            for v in (((x+1) % n)*n+y, x*n+(y+1) % n):
                graph[u].add(v)
                graph[v].add(u)
    return graph


def component_labels(values, graph, threshold):
    labels = [-1]*len(values)
    next_label = 0
    for vertex, value in enumerate(values):
        if value < threshold or labels[vertex] >= 0:
            continue
        labels[vertex] = next_label
        stack = [vertex]
        while stack:
            u = stack.pop()
            for v in graph[u]:
                if values[v] >= threshold and labels[v] < 0:
                    labels[v] = next_label
                    stack.append(v)
        next_label += 1
    return labels


class ExactH0Tests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('exact_h0'),
                             'Exact integer H0 engine is not implemented')
        self.h0 = importlib.import_module('exact_h0')

    def check_ranks(self, values, n):
        result = self.h0.compute(values, n)
        graph = axis_graph(n)
        thresholds = sorted({max(values)+1, min(values)-1, *values}, reverse=True)
        labels = {t: component_labels(values, graph, t) for t in thresholds}
        for i, high in enumerate(thresholds):
            for low in thresholds[i:]:
                # Rank is the number of low components meeting a high vertex.
                rank = len({labels[low][v] for v, a in enumerate(values) if a >= high})
                barcode_rank = sum(b >= high and d < low for b, d in result['intervals'])
                barcode_rank += sum(b >= high for b in result['essential'])
                self.assertEqual(barcode_rank, rank, (values, high, low, result))
        self.assertTrue(self.h0.verify_by_connectivity(values, n, result))

    def test_unequal_peaks_preserve_elder_and_longest_finite_bar(self):
        values = [5, 1, 4, 2, 3, 0]*6
        result = self.h0.compute(values, 6)
        self.assertEqual(result, {'intervals': [[3, 2], [4, 1]],
                                 'birth_vertices': [4, 2], 'essential': [5],
                                 'essential_vertices': [0], 'zero_count': 33})
        self.check_ranks(values, 6)

    def test_seams_and_diagonals_change_the_observable(self):
        # The seam joins the peaks at birth; forgetting it creates [4,0].
        self.assertEqual(self.h0.compute([5, 0, 1, 4]*4, 4)['intervals'], [])
        # These opposite square corners are not joined by an axis edge.
        values = [0]*16
        values[0] = values[5] = 5
        self.assertEqual(self.h0.compute(values, 4)['intervals'], [[5, 0]])
        self.check_ranks(values, 4)

    def test_plateaus_ties_and_constant_essential(self):
        result = self.h0.compute([5, 1, 5, 1]*4, 4)
        self.assertEqual(result['intervals'], [[5, 1]])
        self.assertEqual(result['birth_vertices'], [2])
        self.assertEqual(result['essential_vertices'], [0])
        constant = self.h0.compute([-7]*9, 3)
        self.assertEqual(constant, {'intervals': [], 'birth_vertices': [],
                                   'essential': [-7], 'essential_vertices': [0],
                                   'zero_count': 8})
        self.assertTrue(self.h0.verify_by_connectivity([-7]*9, 3, constant))

    def test_integer_precision_translation_and_positive_scaling(self):
        offset = 1 << 96
        values = [offset+v for v in [5, 1, 4, 2, 3, 0]*6]
        result = self.h0.compute(values, 6)
        self.assertEqual(result['intervals'], [[offset+3, offset+2], [offset+4, offset+1]])
        self.assertEqual([b-d for b, d in result['intervals']], [1, 3])
        scaled = self.h0.compute([11*v for v in values], 6)
        self.assertEqual([b-d for b, d in scaled['intervals']], [11, 33])
        self.assertTrue(self.h0.verify_by_connectivity(values, 6, result))

    def test_exhaustive_binary_and_seeded_tied_ranks(self):
        for values in itertools.product((0, 1), repeat=9):
            self.check_ranks(values, 3)
        rng = random.Random(3934000)
        for n in (3, 4, 5):
            for _ in range(25):
                self.check_ranks([rng.randrange(-3, 4) for _ in range(n*n)], n)

    def test_connectivity_rejects_missing_shifted_or_misattributed_bars(self):
        values = [5, 1, 4, 2, 3, 0]*6
        result = self.h0.compute(values, 6)
        mutations = []
        for death in (0, 2):
            other = copy.deepcopy(result)
            other['intervals'][1][1] = death
            mutations.append(other)
        other = copy.deepcopy(result)
        other['intervals'].pop(); other['birth_vertices'].pop(); other['zero_count'] += 1
        mutations.append(other)
        for key, value in [('birth_vertices', [2, 4]), ('essential', [4]),
                           ('essential_vertices', [2]), ('zero_count', 0),
                           ('zero_count', True)]:
            other = copy.deepcopy(result); other[key] = value; mutations.append(other)
        other = copy.deepcopy(result); other['intervals'][0][0] = 3.0; mutations.append(other)
        other = copy.deepcopy(result); other['extra'] = True; mutations.append(other)
        for other in mutations:
            with self.subTest(result=other), self.assertRaises(ValueError):
                self.h0.verify_by_connectivity(values, 6, other)

    def test_strict_grid_and_value_types_reject_bool_float_and_bad_shape(self):
        cases = [([0]*9, True), ([0]*9, 3.0), ([0]*4, 2), ([0]*8, 3),
                 ([0]*8+[True], 3), ([0]*8+[Q(1)], 3), ([0]*8+[1.0], 3),
                 ('0'*9, 3), (iter([0]*9), 3)]
        for values, n in cases:
            with self.subTest(n=n), self.assertRaises(ValueError):
                self.h0.compute(values, n)


class BinTransferTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('exact_h0'),
                             'Exact integer H0 engine is not implemented')
        self.h0 = importlib.import_module('exact_h0')

    def test_exact_half_open_boundaries_and_diagonal_equalities(self):
        # The diagram is synthetic: no grid provenance is asserted by this utility.
        diagram = {'intervals': [[2, 0], [3, 0], [4, 0], [5, 0], [6, 0]],
                   'birth_vertices': [0, 1, 2, 3, 4], 'essential': [7],
                   'essential_vertices': [5], 'zero_count': 0}
        rows = self.h0.bin_transfer(diagram, 10, Q(1, 20), [Q(1, 10), Q(1, 5), Q(2, 5), Q(3, 5)])
        self.assertEqual([r['sample_count'] for r in rows], [0, 2, 2])
        self.assertEqual([r['target_lower_count'] for r in rows], [0, 0, 0])
        self.assertEqual([r['target_upper_count'] for r in rows], [None, 3, 4])
        self.assertEqual([r['clean_upper'] for r in rows], [False, True, True])
        self.assertEqual([r['nonempty_contraction'] for r in rows], [False, False, False])
        self.assertEqual(rows[0]['lower'], Q(1, 10))
        exact = self.h0.bin_transfer(diagram, 10, Q(0), [Q(1, 5), Q(2, 5), Q(3, 5)])
        self.assertEqual([(r['target_lower_count'], r['target_upper_count']) for r in exact], [(2, 2), (2, 2)])

    def test_transfer_rejects_inexact_inputs_and_zero_or_inverted_bars(self):
        result = self.h0.compute([0]*9, 3)
        for scale, error, edges in [(True, Q(0), [Q(1), Q(2)]),
                                    (0, Q(0), [Q(1), Q(2)]),
                                    (1, 0.0, [Q(1), Q(2)]),
                                    (1, Q(-1), [Q(1), Q(2)]),
                                    (1, Q(0), [1, 2]),
                                    (1, Q(0), [Q(0), Q(2)]),
                                    (1, Q(0), [Q(2), Q(1)])]:
            with self.subTest(scale=scale, error=error, edges=edges), self.assertRaises(ValueError):
                self.h0.bin_transfer(result, scale, error, edges)
        for pair in ([0, 0], [0, 1], [True, 0]):
            other = copy.deepcopy(result)
            other['intervals'] = [pair]; other['birth_vertices'] = [1]
            with self.assertRaises(ValueError):
                self.h0.bin_transfer(other, 1, Q(0), [Q(1), Q(2)])


if __name__ == '__main__':
    unittest.main()
