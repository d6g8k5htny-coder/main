from fractions import Fraction as Q
from itertools import product
import hashlib
import json


def require(condition):
    if not condition:
        raise AssertionError('exact RG countercheck failed')


def mul(a, b):
    c = [Q(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            c[i+j] += x*y
    return c


def add(a, b):
    return [(a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0)
            for i in range(max(len(a), len(b)))]


def at(a, x, d=0):
    for _ in range(d):
        a = [i*a[i] for i in range(1, len(a))]
    return sum((v*x**i for i, v in enumerate(a)), Q(0))


counts = {}
counts['pinned_quintics'] = 0
for r, k, c0, c1 in product([Q(1, 2), Q(1, 4), Q(1, 8)],
                            [Q(0), Q(1, 128), Q(1, 64)],
                            map(Q, [-2, -1, 0, 1, 2]),
                            map(Q, [-2, -1, 0, 1, 2])):
    if k > r*r:
        continue
    a = r/2
    h = add([-k*r**3/2, -3*k*r*r/2, 0, 2*k],
            mul(mul([-a*a, 0, 1], [-a*a, 0, 1]), [k*c0, k*c1]))
    require(at(h, -a) == 0 and at(h, a) == -k*r**3)
    require(at(h, -a, 1) == at(h, a, 1) == 0)
    for sign in [-1, 1]:
        require(at(h, sign*a, 2)/r == sign*6*k + r*at(h, 0, 4)/12 + sign*r*r*at(h, 0, 5)/120)
    counts['pinned_quintics'] += 1

counts['sign_window_cases'] = 0
vals = [Q(i, 4) for i in range(-8, 9)]
for w, a, rm, rs in product(vals, repeat=4):
    p, q = w-a+rm, w+a+rs
    if p*q >= 0:
        continue
    R = max(abs(rm), abs(rs))
    require(abs(w) <= abs(a)+R)
    require(max(abs(p), abs(q)) <= 2*(abs(a)+R))
    counts['sign_window_cases'] += 1


def vand(v):
    out = Q(1)
    for i in range(len(v)):
        for j in range(i+1, len(v)):
            out *= abs(v[i]-v[j])
    return out


counts['vandermonde_cancellation_cases'] = 0
for m in range(1, 6):
    for v in product([Q(-2), Q(-1), Q(-1, 2), Q(0), Q(1, 2), Q(1), Q(2)], repeat=m):
        i = min(range(m), key=lambda j: abs(v[j]))
        rest = v[:i]+v[i+1:]
        D = Q(1)
        for x in rest:
            D *= abs(x)
        require(vand(v) <= 2**(m-1)*D*vand(rest))
        counts['vandermonde_cancellation_cases'] += 1

require(1-Q(1,5) == Q(4,5))
require(Q(4,5)-Q(3,4) == Q(1,20))
require(3-11*Q(1,5) == Q(4,5))
require((-2)+(3) == 1)  # radial r^-2 times RG6 r^3
counts['rational_exponent_identities'] = 4

# Finite checks only. The Gaussian integration, conditional moments and
# continuum estimates are proved analytically in the accompanying review.
print(json.dumps({'status': 'PASS', 'counts': counts}, sort_keys=True))
