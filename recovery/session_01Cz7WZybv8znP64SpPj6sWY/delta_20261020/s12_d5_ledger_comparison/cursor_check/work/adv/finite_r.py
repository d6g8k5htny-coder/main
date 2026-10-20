# Finite-r typed intensity (LS-DER-024 (4)) on planar BF kernel, angle profile and angle integral, numpy MC.
# Usage: finite_r.py PYLIB r tmode N nang     tmode in {r23, fixed:<t>}
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np, sympy as sp, mpmath as mp
mp.mp.dps = 90
x1, y1, x2, y2 = sp.symbols('x1 y1 x2 y2', real=True)
Kk = sp.exp(-((x1 - x2)**2 + (y1 - y2)**2) / 2)
_c = {}
def kf(a, b_):
    if (a, b_) not in _c:
        e = Kk
        for v, n in ((x1, a[0]), (y1, a[1]), (x2, b_[0]), (y2, b_[1])):
            if n: e = sp.diff(e, v, n)
        _c[(a, b_)] = sp.lambdify((x1, y1, x2, y2), e, 'mpmath')
    return _c[(a, b_)]
def cond(obs, tgt, qry):
    n, m = len(obs), len(qry)
    Soo = mp.matrix(n, n); Sqo = mp.matrix(m, n); Sqq = mp.matrix(m, m)
    for i, (P, a) in enumerate(obs):
        for j, (Q, b_) in enumerate(obs): Soo[i, j] = kf(a, b_)(P[0], P[1], Q[0], Q[1])
    for i, (P, a) in enumerate(qry):
        for j, (Q, b_) in enumerate(obs): Sqo[i, j] = kf(a, b_)(P[0], P[1], Q[0], Q[1])
        for j, (Q, b_) in enumerate(qry): Sqq[i, j] = kf(a, b_)(P[0], P[1], Q[0], Q[1])
    Si = mp.inverse(Soo)
    return Sqo * Si * mp.matrix(tgt), Sqq - Sqo * Si * Sqo.T
def factor(C):
    C = (C + C.T) / 2
    E, Q = mp.eigsy(C)
    n = C.rows
    return np.array([[float(Q[i, j] * mp.sqrt(max(E[j], 0))) for j in range(n)] for i in range(n)])
HS = [(2, 0), (1, 1), (0, 2)]
b = mp.mpf(6) / 5; kap = mp.mpf(1); al = mp.mpf(1) / 2
def typed(v):
    a, bb, c = v[:, 0], v[:, 1], v[:, 2]; dM = a * c - bb * bb
    a2, b2, c2 = v[:, 3], v[:, 4], v[:, 5]; dS = a2 * c2 - b2 * b2
    ok = (dM > 0) & (a < 0) & (dS < 0)
    return np.where(ok, np.abs(dM * dS), 0.0)
def main():
    r = mp.mpf(sys.argv[2]); tmode = sys.argv[3]; N = int(sys.argv[4]); nang = int(sys.argv[5])
    t = r**(mp.mpf(2) / 3) if tmode == 'r23' else mp.mpf(tmode.split(':')[1])
    M = (-r / 2, mp.mpf(0)); S = (r / 2, mp.mpf(0))
    pins = [(M, (0, 0)), (M, (1, 0)), (M, (0, 1)), (S, (0, 0)), (S, (1, 0)), (S, (0, 1))]
    tg = [b, 0, 0, b - kap * r**3 / 6, 0, 0]
    u = b - al * kap * r**3 / 6
    mZ, cZ = cond(pins, tg, [(M, a) for a in HS] + [(S, a) for a in HS])
    LZ = factor(cZ); mz = np.array([float(v) for v in mZ])
    rng = np.random.default_rng(12345)
    Zs = []
    for _ in range(8):
        g = rng.standard_normal((250000, 6)); Zs.append(typed(mz + g @ LZ.T))
    Z = np.concatenate(Zs).mean()
    rf = float(r); tf = float(t)
    print(f'r={rf} t={tf:.5f} Z/r^2={Z / rf**2:.4f}', flush=True)
    acc = 0.0; prof = []
    for j in range(nang):
        th = mp.pi * (j + mp.mpf('0.5')) / nang
        y = (t * mp.cos(th), t * mp.sin(th))
        third = [(y, (0, 0)), (y, (1, 0)), (y, (0, 1))]
        m3, c3 = cond(pins, tg, third)
        d = mp.matrix([u - m3[0], -m3[1], -m3[2]])
        p = mp.exp(-(d.T * mp.lu_solve(c3, d))[0] / 2) / mp.sqrt((2 * mp.pi)**3 * mp.det(c3))
        mH, cH = cond(pins + third, tg + [u, 0, 0], [(M, a) for a in HS] + [(S, a) for a in HS] + [(y, a) for a in HS])
        LH = factor(cH); mh = np.array([float(v) for v in mH])
        tot = 0.0; n = 0
        for _ in range(max(1, N // 200000)):
            g = rng.standard_normal((200000, 9)); v = mh + g @ LH.T
            dY = v[:, 6] * v[:, 8] - v[:, 7]**2
            tot += (typed(v) * np.abs(dY)).sum(); n += 200000
        lam = float(p) * (tot / n) / Z
        prof.append((float(th) * 180 / float(mp.pi), lam, float(p) * tf**6))
        acc += lam
    Lam = 2 * np.pi / nang * acc
    env = rf**4 * tf**-6
    for th, l, pt in prof:
        print(f'  theta={th:7.2f}  lambda={l:.4e}  lambda/(r^4 t^-6)={l / env:.4e}  p t^6={pt:.4f}')
    print(f'r={rf} t={tf:.5f} nang={nang}  Lambda={Lam:.4e}  Lambda/(r^4 t^-6)={Lam / env:.4e}', flush=True)
main()
