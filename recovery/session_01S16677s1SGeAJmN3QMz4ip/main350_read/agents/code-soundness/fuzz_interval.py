"""Enclosure fuzz for surgery_field scalar interval ops (stdlib only)."""
import random, sys, time
from fractions import Fraction as Q
sys.path.insert(0, 'experiments/universality/two_parameter_cusp_population')
import surgery_field as sf

random.seed(12345)
def rq(mag=8):
    # random rational with mixed signs and non-dyadic denominators
    n = random.randint(-10**mag, 10**mag); d = random.randint(1, 10**mag)
    return Q(n, d)
def rint():
    a, b = rq(), rq()
    if a > b: a, b = b, a
    if random.random() < 0.2: b = a  # degenerate
    return (a, b)
def pts(iv, k=5):
    lo, hi = iv
    out = [lo, hi]
    for _ in range(k):
        t = Q(random.randint(0, 1000), 1000)
        out.append(lo + (hi-lo)*t)
    return out

bad = 0
N = 4000
for i in range(N):
    a, b = rint(), rint()
    # add/sub/mul/square
    for name, fn, op in (('add', sf._add, lambda x,y: x+y), ('sub', sf._sub, lambda x,y: x-y), ('mul', sf._mul, lambda x,y: x*y)):
        r = fn(a, b)
        sf._interval(r)
        for x in pts(a):
            for y in pts(b):
                v = op(x, y)
                if not (r[0] <= v <= r[1]):
                    bad += 1; print('FAIL', name, a, b, x, y, v, r)
    r = sf._square(a)
    for x in pts(a):
        if not (r[0] <= x*x <= r[1]): bad += 1; print('FAIL square', a, x, r)
    if b[0] > 0:
        r = sf._div(a, b)
        for x in pts(a):
            for y in pts(b):
                if not (r[0] <= x/y <= r[1]): bad += 1; print('FAIL div', a, b, x, y, r)
    else:
        try:
            sf._div(a, b); bad += 1; print('FAIL div accepted nonpositive denominator', b)
        except ValueError:
            pass
print('interval ops fuzz: cases', N, 'failures', bad)

# outward rounding direction
for i in range(2000):
    a = rq(); b = rq()
    if a > b: a, b = b, a
    lo, hi = sf._outward(a, b)
    assert lo <= a and hi >= b and lo.denominator & (lo.denominator-1) == 0 and hi.denominator & (hi.denominator-1) == 0
print('outward direction ok')
