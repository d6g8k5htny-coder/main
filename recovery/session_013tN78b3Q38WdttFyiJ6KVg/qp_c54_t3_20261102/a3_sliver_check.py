#!/usr/bin/env python3
"""Independent check of the BLOCKING finding on QP-C54-T2-029 (frontiers/cusp_coefficient_certified_20261001/d3.py:40-41).

d3._sub(a, b, k) returns panels (a + (b-a)*j/k, a + (b-a)*(j+1)/k). This script does not import or run the packet's code; it
evaluates the same binary64 expression for the last panel end with the cross-verifier's example values (tb = 13 - ax/4,
ax = 2.0333333333353667; a = 0.3/tb, b = 0.7/tb) and reports whether the last panel ends at b. Run with python3 -I.
"""
from fractions import Fraction as F
import random
ax = 2.0333333333353667
tb = 13 - ax / 4
a, b = 0.3 / tb, 0.7 / tb
short = [(k, F(b) - F(a + (b - a) * k / k)) for k in range(1, 49) if a + (b - a) * k / k != b]
print(f'tb={tb!r} a={a!r} b={b!r}')
print(f'k in 1..48 whose last panel ends short of b: {len(short)}; shortfalls equal to 2^-57: {sum(d == F(1, 2**57) for _, d in short)}')
random.seed(1)
N = 200000
miss = sum(1 for _ in range(N) if (lambda x, y: x + (y - x) != y)(random.uniform(2**-6, 2**-5), random.uniform(2**-5, 2**-4)))
print(f'random pairs (a in [2^-6,2^-5), b in [2^-5,2^-4)) with fl(a + fl(b - a)) != b: {miss} of {N}')
