# Brute-force union-find sanity check of Section 3's component claim on the core model (13):
# F = b + P(s) - (1+s^2/4) w^2, P(s) = -s^4/4 + u s^2/2 + v s, d=2, in a box with |s|,|w| <= 2.5 (scaled u=1).
import numpy as np
def comps(mask):
    n, m = mask.shape
    parent = {}
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    def union(a,b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb
    for i in range(n):
        for j in range(m):
            if mask[i,j]: parent[(i,j)] = (i,j)
    for i in range(n):
        for j in range(m):
            if not mask[i,j]: continue
            if i+1<n and mask[i+1,j]: union((i,j),(i+1,j))
            if j+1<m and mask[i,j+1]: union((i,j),(i,j+1))
    return len({find(a) for a in parent})
eta_c = 2/(3*np.sqrt(3))
for eta in [0.1, 0.3, 0.38]:
    for sgn in [1, -1]:
        u, v = 1.0, sgn*eta
        rs = np.sort(np.roots([-1, 0, u, v]).real)
        P = lambda s: -s**4/4 + u*s**2/2 + v*s
        vals = sorted(P(rs))  # saddle, lower max, higher max
        saddle, lower, higher = vals
        N = 1201
        s = np.linspace(-2.5, 2.5, N); w = np.linspace(-2.5, 2.5, N)
        S, W = np.meshgrid(s, w, indexing='ij')
        F = P(S) - (1+S**2/4)*W**2
        res = []
        for frac in [0.1, 0.5, 0.9]:
            lev = saddle + frac*(lower - saddle)
            res.append(comps(F >= lev))
        below = comps(F >= saddle - 0.02*(lower-saddle))
        above_lower = comps(F >= lower + 0.5*(higher-lower))
        print("eta=%.2f sgn=%+d roots=%s  heights(saddle,lower,higher)=(%.4f,%.4f,%.4f)  comps between=%s  just below saddle=%d  above lower max=%d  kappa=%.5f" % (eta, sgn, np.round(rs,4), saddle, lower, higher, res, below, above_lower, lower - saddle))
