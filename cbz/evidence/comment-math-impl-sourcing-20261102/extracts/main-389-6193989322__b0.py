#!/usr/bin/env python3
"""Exact finite probes for the accompanying general matrix-stability proof.

Standard library only. No repository program, random field, network, or
manuscript mutation is involved. Deliberately false variants test sensitivity;
finite probes do not establish the general theorem.
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

Matrix = tuple[tuple[F, ...], ...]
VARIANT = 'baseline'


def matrix(rows) -> Matrix:
    a = tuple(tuple(F(x) for x in row) for row in rows)
    n = len(a)
    if n == 0 or any(len(row) != n for row in a):
        raise ValueError('a nonempty square matrix is required')
    if any(a[i][j] != a[j][i] for i in range(n) for j in range(n)):
        raise ValueError('a symmetric matrix is required')
    return a


def diagonal(values) -> Matrix:
    return matrix([[x if i == j else 0 for j in range(len(values))]
                   for i, x in enumerate(values)])


def add(a: Matrix, b: Matrix, scale=F(1)) -> Matrix:
    return matrix([[a[i][j] + scale*b[i][j] for j in range(len(a))]
                   for i in range(len(a))])


def row_bound(a: Matrix) -> F:
    # For symmetric matrices this is a rational upper bound on the operator norm.
    return max(sum(abs(x) for x in row) for row in a)


def determinant(a: Matrix) -> F:
    # Independent Leibniz oracle, only used in dimensions <= 4 here.
    n = len(a)
    out = F(0)
    for p in itertools.permutations(range(n)):
        inversions = sum(p[i] > p[j] for i in range(n) for j in range(i+1, n))
        term = F((-1)**inversions)
        for i in range(n):
            term *= a[i][p[i]]
        out += term
    return out


def inertia(a: Matrix) -> tuple[int, int, int]:
    """Exact symmetric congruence elimination; return negative, zero, positive."""
    q = [list(row) for row in a]
    negative = positive = zero = 0
    while q:
        n = len(q)
        pivot = next((i for i in range(n) if q[i][i]), None)
        if pivot is not None:
            order = [pivot] + [i for i in range(n) if i != pivot]
            q = [[q[i][j] for j in order] for i in order]
            p = q[0][0]
            negative += p < 0
            positive += p > 0
            q = [[q[i][j] - q[i][0]*q[0][j]/p
                  for j in range(1, n)] for i in range(1, n)]
            continue
        pair = next(((i, j) for i in range(n) for j in range(i+1, n)
                     if q[i][j]), None)
        if pair is None:
            zero += n
            break
        i, j = pair
        order = [i, j] + [k for k in range(n) if k not in pair]
        q = [[q[i][j] for j in order] for i in order]
        b = q[0][1]
        # The pivot [[0,b],[b,0]] has one positive and one negative eigenvalue.
        negative += 1
        positive += 1
        q = [[q[i][j] - (q[i][0]*q[1][j] + q[i][1]*q[0][j])/b
              for j in range(2, n)] for i in range(2, n)]
    return negative, zero, positive


def congruence(a: Matrix) -> Matrix:
    n = len(a)
    p = [[F(1) if i == j else (F(i+j+1, 3) if i > j else F(0))
          for j in range(n)] for i in range(n)]
    return matrix([[sum(p[k][i]*a[k][l]*p[l][j]
                        for k in range(n) for l in range(n))
                    for j in range(n)] for i in range(n)])


def weight(a: Matrix, j: int) -> F:
    d = determinant(a)
    neg, zeros, _ = inertia(a)
    if VARIANT == 'drop_index':
        return abs(d)
    if zeros or neg != j:
        return F(0)
    if VARIANT == 'drop_determinant':
        return F(1)
    return d if VARIANT == 'signed_determinant' else abs(d)


def single_bound(a: Matrix, b: Matrix) -> F:
    n = len(a)
    r = max(row_bound(a), row_bound(b))
    delta = row_bound(add(a, b, -1))
    coefficient = 1 if VARIANT == 'omit_dimension' else n
    exponent = 0 if VARIANT == 'omit_growth' else n-1
    return coefficient*r**exponent*delta


def pair_bound(a: Matrix, b: Matrix, c: Matrix, d: Matrix) -> F:
    n = len(a)
    r = max(map(row_bound, (a, b, c, d)))
    delta = row_bound(add(a, c, -1)) + row_bound(add(b, d, -1))
    exponent = n-1 if VARIANT == 'wrong_pair_growth' else 2*n-1
    return n*r**exponent*delta


class WeightChecks(unittest.TestCase):
    def test_exact_inertia_oracle(self):
        for n in range(1, 5):
            for values in itertools.product((-1, 0, 2), repeat=n):
                expected = (values.count(-1), values.count(0), values.count(2))
                a = diagonal(values)
                for candidate in (a, congruence(a)):
                    self.assertEqual(inertia(candidate), expected)
                    self.assertEqual(determinant(candidate), determinant(a))
        self.assertEqual(inertia(matrix([[0, 2], [2, 0]])), (1, 0, 1))
        self.assertEqual(inertia(matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]])), (1, 1, 1))

    def test_definition_and_partition(self):
        for values in ((-2, -1), (-2, 1), (2, 1), (0, 1), (-1, 0), (0, 0)):
            a = diagonal(values)
            expected = abs(F(values[0])*values[1])
            self.assertEqual(sum(weight(a, j) for j in range(3)), expected)
            if 0 not in values:
                j = sum(x < 0 for x in values)
                self.assertEqual(weight(a, j), expected)
                self.assertEqual(weight(a, (j+1) % 3), 0)

    def test_crossing_and_repeated_zero_eigenvalues(self):
        for n in range(1, 5):
            for t in (F(1, 2), F(1, 8), F(1, 64)):
                zero = diagonal([0]*n)
                a = diagonal([-t]*n)
                b = diagonal([t]*n)
                self.assertEqual(weight(a, n), t**n)
                self.assertEqual(weight(b, 0), t**n)
                self.assertEqual(weight(zero, 0), 0)
                for j in range(n+1):
                    self.assertLessEqual(abs(weight(a, j)-weight(b, j)), single_bound(a, b))
                    self.assertLessEqual(abs(weight(a, j)-weight(zero, j)), single_bound(a, zero))

    def test_noncommuting_matrix_pairs(self):
        matrices = [matrix([[a, b], [b, c]])
                    for a, b, c in itertools.product((-2, 0, 3), repeat=3)]
        saw_noncommuting = False
        for a, b in itertools.product(matrices, repeat=2):
            ab = [[sum(a[i][k]*b[k][j] for k in range(2)) for j in range(2)]
                  for i in range(2)]
            ba = [[sum(b[i][k]*a[k][j] for k in range(2)) for j in range(2)]
                  for i in range(2)]
            saw_noncommuting |= ab != ba
            for j in range(3):
                self.assertLessEqual(abs(weight(a, j)-weight(b, j)), single_bound(a, b))
        self.assertTrue(saw_noncommuting)

    def test_dimension_and_growth_near_sharpness(self):
        for n in range(1, 5):
            for j in range(n+1):
                signs = [-1]*j+[1]*(n-j)
                for r in (F(1), F(5)):
                    h = r/F(100)
                    a = diagonal([s*r for s in signs])
                    b = diagonal([s*(r-h) for s in signs])
                    difference = weight(a, j)-weight(b, j)
                    self.assertEqual(difference, r**n-(r-h)**n)
                    self.assertLessEqual(difference, single_bound(a, b))
                    if n >= 2:
                        self.assertGreater(difference/h, (n-F(1, 10))*r**(n-1))

    def test_two_endpoint_product_bound(self):
        for n in range(2, 5):
            for r in (F(1), F(3), F(10)):
                a = diagonal([-r]*n)
                b = diagonal([-r]*(n-1)+[r])
                h = r/F(100)
                c = diagonal([-(r-h)]*n)
                d = diagonal([-(r-h)]*(n-1)+[r-h])
                diff = abs(weight(a, n)*weight(b, n-1)-weight(c, n)*weight(d, n-1))
                self.assertEqual(diff, r**(2*n)-(r-h)**(2*n))
                self.assertLessEqual(diff, pair_bound(a, b, c, d))
                # Cross a type boundary at one endpoint, including a singular limit.
                for x in (-r, F(0), r):
                    c = diagonal([-r]*(n-1)+[x])
                    self.assertLessEqual(abs(weight(a, n)*weight(b, n-1)-
                                             weight(c, n)*weight(b, n-1)),
                                         pair_bound(a, b, c, b))

    def test_determinant_free_indicator_really_discontinuous(self):
        for n in range(1, 5):
            self.assertEqual(inertia(diagonal([0]*n))[0], 0)
            for t in (F(1, 2), F(1, 16), F(1, 256)):
                self.assertEqual(inertia(diagonal([-t]*n))[0], n)
                # The indicator stays 1; the weighted value tends to zero exactly.
                self.assertEqual(weight(diagonal([-t]*n), n), t**n)

    def test_probability_only_does_not_pass_expectations(self):
        for n in (2, 3, 4):
            for scale in (2, 4, 8, 16):
                probability = F(1, scale**(2*n))
                a = diagonal([-scale]*n)
                b = diagonal([-scale]*(n-1)+[scale])
                expected = probability*weight(a, n)*weight(b, n-1)
                self.assertEqual(expected, 1)
                self.assertLess(probability, F(1, scale))
                self.assertEqual(probability*F(scale)**(2*n), 1)

    def test_invalid_domain_inputs_are_not_silently_admitted(self):
        for rows in ([], [[1, 2]], [[1, 2], [0, 1]]):
            with self.assertRaises(ValueError):
                matrix(rows)


def main() -> int:
    global VARIANT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', default='baseline', choices=(
        'baseline', 'drop_index', 'drop_determinant', 'signed_determinant',
        'omit_dimension', 'omit_growth', 'wrong_pair_growth'))
    VARIANT = parser.parse_args().variant
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(WeightChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        'variant': VARIANT, 'python': platform.python_version(),
        'optimize': sys.flags.optimize, 'no_site': sys.flags.no_site,
        'dont_write_bytecode': sys.dont_write_bytecode,
        'tests_run': result.testsRun, 'failures': [t.id() for t, _ in result.failures],
        'errors': [t.id() for t, _ in result.errors], 'skips': len(result.skipped),
        'successful': result.wasSuccessful(),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'Exact finite matrix probes; no project code or random-field execution.',
        'norm_limit': 'Rational row-sum operator-norm upper bounds, exact on diagonal probes.',
    }
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
