import math, random, sys
from math import erf, sqrt, exp
from proto import conditional, mc, chol, z0

Phi = lambda x: 0.5 * (1 + erf(x / sqrt(2)))

def inner_prob(mu, C, lo, hi, order):
    """Rigorous (in exact arithmetic) lower bound for P(H in box) via a sequential inner z-box in the permuted Cholesky frame."""
    n = len(mu)
    P = [[C[i][j] for j in order] for i in order]
    L = chol([[P[i][j] + (1e-18 if i == j else 0) for j in range(n)] for i in range(n)])
    m = [mu[i] for i in order]; l = [lo[i] for i in order]; h = [hi[i] for i in order]
    za = [0.0] * n; zc = [0.0] * n; p = 1.0
    for i in range(n):
        # H_i = m_i + sum_{j<i} L_ij z_j + L_ii z_i ; bound the partial sum over the z-box so far
        smin = smax = 0.0
        for j in range(i):
            a, c = L[i][j] * za[j], L[i][j] * zc[j]
            smin += min(a, c); smax += max(a, c)
        za[i] = (l[i] - m[i] - smax) / L[i][i]
        zc[i] = (h[i] - m[i] - smin) / L[i][i]
        if zc[i] <= za[i]: return 0.0
        p *= Phi(zc[i]) - Phi(za[i])
    return p

def det_bounds(lo, hi):
    # coordinates: 0 xxM 1 xyM 2 yyM 3 xxS 4 xyS 5 yyS ; conditions xxM<0, yyM<0, detM>0, detS<0
    if hi[0] >= 0 or hi[2] >= 0 or lo[3] <= 0 or hi[5] >= 0: return None
    detM = hi[0] * hi[2] - max(abs(lo[1]), abs(hi[1])) ** 2
    if detM <= 0: return None
    detS = lo[3] * (-hi[5])
    return detM * detS

ORDERS = [(2, 5, 0, 3, 1, 4), (0, 3, 2, 5, 1, 4), (2, 0, 1, 5, 3, 4), (5, 2, 3, 0, 4, 1)]

def slab_bound(mu, C, y0, y1, c1, c2, dl):
    sd = [sqrt(C[i][i]) for i in range(6)]
    lo = [mu[0] - c1 * sd[0], -c2 * sd[1], y0, mu[3] - c1 * sd[3], -c2 * sd[4], y0 - dl]
    hi = [mu[0] + c1 * sd[0], c2 * sd[1], y1, mu[3] + c1 * sd[3], c2 * sd[4], y1 + dl]
    d = det_bounds(lo, hi)
    if d is None: return 0.0
    return d * max(inner_prob(mu, C, lo, hi, o) for o in ORDERS)

def total_bound(mu, C, b, nslab):
    ymin = -b - 4.5 * sqrt(2); edges = [ymin + (0 - ymin) * i / nslab for i in range(nslab + 1)]
    tot = 0.0; params = []
    for i in range(nslab):
        y0, y1 = edges[i], edges[i + 1]
        best = 0.0; bp = None
        for c1 in (1.0, 1.5, 2.0, 2.5, 3.0):
            for c2 in (0.5, 1.0, 1.5, 2.0):
                for dl in (0.05, 0.1, 0.2, 0.4):
                    v = slab_bound(mu, C, y0, y1, c1, c2, dl)
                    if v > best: best, bp = v, (c1, c2, dl)
        tot += best; params.append(bp)
    return tot, params

for r in (0.2, 0.3, 0.5):
    for b in (0.0, 1.0):
        for k in (1.0,):
            mu, C = conditional(r, b, k)
            m, se = mc(mu, C, 100000, 1)
            for ns in (8, 24, 48):
                tot, params = total_bound(mu, C, b, ns)
                print('r=%.1f b=%.0f k=%.1f nslab=%d  MC %.4f  bound %.4f  ratio %.3f' % (r, b, k, ns, m / r ** 2, tot / r ** 2, tot / m))
            print('   params sample', params[len(params)//2], params[-1])
