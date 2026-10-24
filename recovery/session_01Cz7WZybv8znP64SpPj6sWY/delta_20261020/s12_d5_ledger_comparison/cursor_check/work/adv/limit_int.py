# Independent r->0 limit of LS-DER-024 (4), angle-integrated, planar BF kernel exp(-|h|^2/2), b=6/5, kappa=1.
# Covariances by symbolic differentiation of the kernel (not the Hermite closed form).
# Limit frame: (f, f_x, f_y, f_xx, f_xy, f_xxx)(0) = (b,0,0,0,0,2kappa); third point y=t(cos th, sin th): f=b, grad=0.
# Limit endpoint weight W/r^2 -> kappa^2 S0^2 1{S0<0}, S0=f_yy(0); Z/r^2 -> kappa^2 E[S0^2;S0<0] (six pins only).
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np, sympy as sp, mpmath as mp
mp.mp.dps = 50
x1, y1, x2, y2 = sp.symbols('x1 y1 x2 y2', real=True)
K = sp.exp(-((x1 - x2)**2 + (y1 - y2)**2) / 2)
_cache = {}
def kfun(a, b_):
    key = (a, b_)
    if key not in _cache:
        e = K
        if a[0]: e = sp.diff(e, x1, a[0])
        if a[1]: e = sp.diff(e, y1, a[1])
        if b_[0]: e = sp.diff(e, x2, b_[0])
        if b_[1]: e = sp.diff(e, y2, b_[1])
        _cache[key] = sp.lambdify((x1, y1, x2, y2), e, 'mpmath')
    return _cache[key]
def cv(P, a, Q, b_):
    return kfun(a, b_)(P[0], P[1], Q[0], Q[1])
def cond(obs, tgt, qry):
    n, m = len(obs), len(qry)
    Soo = mp.matrix(n, n); Sqo = mp.matrix(m, n); Sqq = mp.matrix(m, m)
    for i, (P, a) in enumerate(obs):
        for j, (Q, b_) in enumerate(obs): Soo[i, j] = cv(P, a, Q, b_)
    for i, (P, a) in enumerate(qry):
        for j, (Q, b_) in enumerate(obs): Sqo[i, j] = cv(P, a, Q, b_)
        for j, (Q, b_) in enumerate(qry): Sqq[i, j] = cv(P, a, Q, b_)
    Si = mp.inverse(Soo)
    return Sqo * Si * mp.matrix(tgt), Sqq - Sqo * Si * Sqo.T
b = mp.mpf(6) / 5; kap = mp.mpf(1)
O = (mp.mpf(0), mp.mpf(0))
jet = [(O, (0, 0)), (O, (1, 0)), (O, (0, 1)), (O, (2, 0)), (O, (1, 1)), (O, (3, 0))]
jtg = [b, 0, 0, 0, 0, 2 * kap]
m0, c0 = cond(jet, jtg, [(O, (0, 2))])
mu0, v0 = m0[0], c0[0, 0]
def m2neg(mu, sd):
    a = -mu / sd
    return (mu**2 + sd**2) * mp.ncdf(a) - mu * sd * mp.npdf(a)
Zl = kap**2 * m2neg(mu0, mp.sqrt(v0))
print('S0 law mean', mp.nstr(mu0, 10), 'var', mp.nstr(v0, 10), 'Z/r^2 limit', mp.nstr(Zl, 10), flush=True)
from math import erf
verf = np.vectorize(erf)
def lam(t, th, N, seed):
    y = (t * mp.cos(th), t * mp.sin(th))
    third = [(y, (0, 0)), (y, (1, 0)), (y, (0, 1))]
    m3, c3 = cond(jet, jtg, third)
    d = mp.matrix([b - m3[0], -m3[1], -m3[2]])
    p = mp.exp(-(d.T * mp.lu_solve(c3, d))[0] / 2) / mp.sqrt((2 * mp.pi)**3 * mp.det(c3))
    mean, C = cond(jet + third, jtg + [b, 0, 0], [(y, (2, 0)), (y, (1, 1)), (y, (0, 2)), (O, (0, 2))])
    m = np.array([float(v) for v in mean]); Cf = np.array([[float(C[i, j]) for j in range(4)] for i in range(4)]); Cf = (Cf + Cf.T) / 2
    CHH = Cf[:3, :3]; cSH = Cf[3, :3]; g = np.linalg.solve(CHH, cSH); sc = np.sqrt(max(Cf[3, 3] - cSH @ g, 0.0))
    L = np.linalg.cholesky(CHH)
    rng = np.random.default_rng(seed)
    tot = 0.0; tot2 = 0.0; n = 0
    B = 250000
    for _ in range(max(1, N // B)):
        z = rng.standard_normal((B, 3)); h = m[:3] + z @ L.T
        mu = m[3] + (h - m[:3]) @ g
        a = -mu / sc
        e2 = (mu**2 + sc**2) * 0.5 * (1 + verf(a / np.sqrt(2))) - mu * sc * np.exp(-a * a / 2) / np.sqrt(2 * np.pi)
        w = np.abs(h[:, 0] * h[:, 2] - h[:, 1]**2) * e2
        tot += w.sum(); tot2 += (w * w).sum(); n += B
    E = tot / n; se = np.sqrt(max(tot2 / n - E * E, 0) / n)
    pf = float(p); Zf = float(Zl)
    return pf * E / Zf, pf * se / Zf, pf
if __name__ == '__main__':
    t = mp.mpf(sys.argv[2]); N = int(sys.argv[3]); ns = [int(v) for v in sys.argv[4].split(',')]
    for n in ns:
        vals = []; ses = []
        for j in range(n):
            th = mp.pi * (j + mp.mpf('0.5')) / n
            l, se, p = lam(t, th, N, 100003 * n + j)
            vals.append(l); ses.append(se)
        Lam = 2 * np.pi / n * sum(vals); SE = 2 * np.pi / n * np.sqrt(sum(s * s for s in ses))
        print(f't={float(t)} midpoints on (0,pi)={n:3d}  Lambda={Lam:.5f} +- {SE:.5f}', flush=True)
