#!/usr/bin/env python3
"""Exact-arithmetic post-exposure comparison check; no blind credit."""
from fractions import Fraction as F
from itertools import permutations


def determinant(a):
    n = len(a)
    total = F(0)
    for p in permutations(range(n)):
        inversions = sum(p[i] > p[j] for i in range(n) for j in range(i + 1, n))
        value = F(-1 if inversions % 2 else 1)
        for i in range(n):
            value *= a[i][p[i]]
        total += value
    return total


for r in (F(2, 3), F(1), F(7, 5)):
    # Original observations: (f_minus, g_minus, f_plus, g_plus).
    height = [F(1, 2), F(0), F(1, 2), F(0)]
    average_gradient = [F(0), F(1, 2), F(0), F(1, 2)]
    secant = [-1 / r, F(0), 1 / r, F(0)]
    column = [F(0), -1 / r, F(0), 1 / r]
    third = [12 / r ** 3, 6 / r ** 2, -12 / r ** 3, 6 / r ** 2]
    assert secant == [p - r ** 2 * w / 12 for p, w in zip(average_gradient, third)]
    blind = [height, average_gradient, column, third]
    prior = [height, secant, column, third]
    assert determinant(prior) == determinant(blind)
    assert abs(determinant(prior)) == 12 / r ** 4
    b, k = F(3, 7), F(5, 11)
    pin = [b, F(0), b - k * r ** 3, F(0)]
    observe = lambda row: sum(x * y for x, y in zip(row, pin))
    assert observe(average_gradient) == 0
    assert observe(secant) == -k * r ** 2
    assert observe(third) == 12 * k
    assert observe(height) == b - k * r ** 3 / 2

# Cubed scalar conversion prefactor: 144 k^(4/3) dk, u=12k.
assert F(144, 12) ** 3 / F(12) ** 4 == F(1, 12)
print("PASS exact finite-radius secant/average-gradient shear, contact targets, and axial Jacobians")
print("PASS exact cubed prefactor for the u=12k conversion")
print("Post-exposure comparison only; no new blind credit or global-bridge proof.")
