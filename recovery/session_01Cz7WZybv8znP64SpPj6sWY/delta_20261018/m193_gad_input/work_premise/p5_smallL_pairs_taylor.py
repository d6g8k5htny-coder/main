"""Premise audit, part 5.
(i)  Re-check Cov(O) positive definiteness for coarse lattices (small L), where double
     precision cannot resolve it (p3 showed -1e-15 artefacts), using mpmath dps=80.
     Includes half-lattice points y=(L/2)e_1, (L/2)(e_1+e_2) where cos/sin symmetries bite.
(ii) Exact Taylor identity F(M)-F(S)+r e.P_r = r^3 f'''(0)/12 + O(r^5) on a generic degree-9
     polynomial along the line (so W_r -> T_e).
"""
import itertools
import mpmath as mp
import sympy as sp

mp.mp.dps = 80
def mp_lattice(L, d, dig=80):
    h = 2*mp.pi/L
    K = int(mp.sqrt(2*(dig*mp.log(10)+30))/h) + 2
    pts = []; tot = mp.mpf(0)
    for m in itertools.product(range(-K, K+1), repeat=d):
        p = [h*k for k in m]
        wv = mp.e**(-mp.fsum(x*x for x in p)/2)
        pts.append((p, wv)); tot += wv
    return [(p, wv/tot) for p, wv in pts]

def covO(pts, y, d):
    k = 2 + 2*d
    C = mp.matrix(k, k)
    for p, wv in pts:
        py = mp.fsum(a*b for a, b in zip(p, y))
        ph = mp.expjpi(py/mp.pi)  # e^{i p.y}
        syms = [mp.mpc(1)] + [mp.mpc(0, x) for x in p] + [ph] + [mp.mpc(0, x)*ph for x in p]
        for i in range(k):
            for j in range(i, k):
                C[i, j] += wv*mp.re(syms[i]*mp.conj(syms[j]))
    for i in range(k):
        for j in range(i):
            C[i, j] = C[j, i]
    s = [mp.sqrt(C[i, i]) for i in range(k)]
    R = mp.matrix(k, k)
    for i in range(k):
        for j in range(k):
            R[i, j] = C[i, j]/(s[i]*s[j])
    return min(mp.eigsy(R, eigvals_only=True))

for d, L, n in [(2, mp.mpf('0.7'), 9), (3, mp.mpf(1), 5)]:
    pts = mp_lattice(L, d)
    worst = (mp.inf, None)
    grid = [-L/2 + L*i/(n-1) for i in range(n)]
    special = [[L/2] + [0]*(d-1), [L/2, L/2] + [0]*(d-2), [L/2]*d, [L/4] + [0]*(d-1), [L*mp.mpf('0.05')] + [0]*(d-1)]
    cand = [list(y) for y in itertools.product(grid, repeat=d)] + special
    for y in cand:
        if mp.sqrt(mp.fsum(x*x for x in y)) < mp.mpf('0.04')*L:
            continue
        me = covO(pts, y, d)
        if me < worst[0]:
            worst = (me, y)
    print('(i) d=%d L=%s: min eig(corr O) = %s at y=%s  (all >0: %s)' % (d, mp.nstr(L, 3), mp.nstr(worst[0], 5),
          [mp.nstr(x, 3) for x in worst[1]], worst[0] > 0))

r, t = sp.symbols('r t')
cs = sp.symbols('c0:10')
f = sum(cs[k]*t**k for k in range(10))
fp = sp.diff(f, t)
expr = f.subs(t, -r/2) - f.subs(t, r/2) + r*(fp.subs(t, -r/2) + fp.subs(t, r/2))/2
ser = sp.expand(expr)
f3 = sp.diff(f, t, 3).subs(t, 0)
print('(ii) [F(M)-F(S)+r e.P_r] - r^3 f\'\'\'(0)/12 =', sp.factor(sp.expand(ser - r**3*f3/12)).as_ordered_terms()[:1],
      '; lowest r-power:', min(sp.Poly(sp.expand(ser - r**3*f3/12), r).monoms())[0] if sp.expand(ser - r**3*f3/12) != 0 else None)
