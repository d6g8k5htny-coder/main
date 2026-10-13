import math, random, sys
from math import erf, sqrt, exp
from proto import conditional, mc, chol, z0
Phi = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
ORDER = (2, 5, 0, 3, 1, 4)   # yyM, yyS, xxM, xxS, xyM, xyS

def inner_prob(mu, C, lo, hi, order=ORDER):
    n = len(mu)
    P = [[C[i][j] for j in order] for i in order]
    L = chol([[P[i][j] + (1e-18 if i == j else 0) for j in range(n)] for i in range(n)])
    m = [mu[i] for i in order]; l = [lo[i] for i in order]; h = [hi[i] for i in order]
    za = [0.0] * n; zc = [0.0] * n; p = 1.0
    for i in range(n):
        smin = smax = 0.0
        for j in range(i):
            a, c = L[i][j] * za[j], L[i][j] * zc[j]
            smin += min(a, c); smax += max(a, c)
        za[i] = (l[i] - m[i] - smin) / L[i][i]
        zc[i] = (h[i] - m[i] - smax) / L[i][i]
        if zc[i] <= za[i]: return 0.0
        p *= Phi(zc[i]) - Phi(za[i])
    return p

def det_bounds(lo, hi):
    if hi[0] >= 0 or hi[2] >= 0 or lo[3] <= 0 or hi[5] >= 0: return None
    detM = hi[0] * hi[2] - max(abs(lo[1]), abs(hi[1])) ** 2
    if detM <= 0: return None
    return detM * (lo[3] * (-hi[5]))

def cell_bound(mu, C, nM, nS, cxx, cxy, span):
    sd = [sqrt(C[i][i]) for i in range(6)]
    yminM = mu[2] - span * sd[2]; yminS = mu[5] - span * sd[5]
    eM = [yminM + (0 - yminM) * i / nM for i in range(nM + 1)]
    eS = [yminS + (0 - yminS) * i / nS for i in range(nS + 1)]
    tot = 0.0
    for i in range(nM):
        for j in range(nS):
            lo = [mu[0] - cxx * sd[0], -cxy * sd[1], eM[i], mu[3] - cxx * sd[3], -cxy * sd[4], eS[j]]
            hi = [mu[0] + cxx * sd[0], cxy * sd[1], eM[i + 1], mu[3] + cxx * sd[3], cxy * sd[4], eS[j + 1]]
            d = det_bounds(lo, hi)
            if d is None: continue
            tot += d * inner_prob(mu, C, lo, hi)
    return tot

if __name__ == '__main__':
    for r in (0.125, 0.3, 0.5):
        for b in (0.0, 1.0):
            for k in (0.5, 2.0):
                mu, C = conditional(r, b, k)
                m, se = mc(mu, C, 100000, 1)
                for (nM, nS, cxx, cxy, span) in ((12, 12, 3, 2.5, 4.5), (24, 24, 3, 2.5, 4.5), (24, 24, 3.5, 3, 5), (40, 40, 3.5, 3, 5)):
                    tot = cell_bound(mu, C, nM, nS, cxx, cxy, span)
                    print('r=%.3f b=%.0f k=%.1f grid=%dx%d c=(%.1f,%.1f,%.1f)  MC %.4f (%.4f)  bound %.4f  ratio %.3f' % (r, b, k, nM, nS, cxx, cxy, span, m / r ** 2, se / r ** 2, tot / r ** 2, tot / m))
