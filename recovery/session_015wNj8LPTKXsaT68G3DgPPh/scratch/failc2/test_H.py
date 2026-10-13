import math, random, sys
import numpy as np
import fcore2 as F
from ia import isqrt
from test_boxes import bigroot, sa2, sb2, sc2, Ma, Mb, Mc
def refH(box, n=24):
    xs, ws = np.polynomial.legendre.leggauss(n)
    pts = []; wts = []
    for (l, h), s2 in zip(box, (sa2, sb2, sc2)):
        p = 0.5 * (h - l) * xs + 0.5 * (h + l)
        wt = 0.5 * (h - l) * ws * np.exp(-p * p / (2 * s2)) / math.sqrt(2 * math.pi * s2)
        pts.append(p); wts.append(wt)
    a, b, c = np.meshgrid(*pts, indexing='ij')
    W = np.einsum('i,j,k->ijk', *wts)
    B = b - a * a / 12; u = c - a * b / 4 + a ** 3 / 72; T = 3 * u * u
    x = bigroot(B, T)
    return float((W * (np.abs(B) ** 3 + 4 * x ** 3 - 3 * B * B * x)).sum())
random.seed(11); bad = 0; better = 0
for t in range(400):
    sz = 10 ** random.uniform(-2.5, -0.3)
    a0 = random.uniform(-3, 3) * math.sqrt(sa2); b0 = random.uniform(-3, 3) * math.sqrt(sb2); c0 = random.uniform(-3, 3) * math.sqrt(sc2)
    if t % 4 == 0: b0 = random.uniform(-0.3, 0.1)       # straddle B = 0 often
    box = ((a0, a0 + sz * math.sqrt(sa2)), (b0, b0 + sz * math.sqrt(sb2)), (c0, c0 + sz * math.sqrt(sc2)))
    I = F.H_box(box, Ma, Mb, Mc); I1 = F.H_box(box, Ma, Mb, Mc, second=False)
    r = refH(box)
    if not (I[0] - 1e-9 * abs(r) <= r <= I[1] + 1e-9 * abs(r)):
        r2 = refH(box, 60)
        if not (I[0] - 1e-7 * abs(r2) <= r2 <= I[1] + 1e-7 * abs(r2)):
            bad += 1; print('FAIL', box, I, r, r2)
    if I[1] - I[0] < I1[1] - I1[0]: better += 1
print('H boxes bad', bad, 'second-order used', better, 'of 400')
