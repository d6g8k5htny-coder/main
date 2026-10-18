"""Independent numerical check of the normalization chain (proof.md sections 1,7,8).

For an explicit finite symmetric PRODUCT spectral measure (random rational weights/
frequencies, or the actual lattice law 2 pi m / L truncated where the omitted mass is
< 1e-20), compute
  (a) C_alg via formula (1) of the proof, with a, B(e), V(e), Gamma_e and F(e)
      obtained DIRECTLY from the random trigonometric series coefficients (no use of the
      proof's tensor formulas (8)-(13)), and full-sphere AREA quadrature;
  (b) the finite-r ordered (max, saddle) pair Kac-Rice gap density
        nu(ell) = int_{S^{d-1}} d omega int_0^inf r^{d-1} p_{(grad f(M), grad f(S), f(M)-f(S))}(0,0,ell)
                   E[|det H_M||det H_S| 1{H_M<0} 1{H_S has exactly one positive eigenvalue} | ...] dr
      with M=-re/2, S=+re/2, evaluated at small ell, and report nu(ell) ell^(1/3).
The raw density is evaluated through the exact coordinate change (P,D_r,W) whose Jacobian
12 r^-(d+3) was verified symbolically in sym_norm.py and is re-verified here numerically
(mpmath, 60 digits) against the raw (D_M,D_S,Delta) Gaussian density at sample nodes.

Usage: venv/bin/python -I kr_num.py <d> <measure: rand|L=<float>> <seed>
"""
import sys
import math
from fractions import Fraction
import numpy as np
import itertools
import mpmath as mp

d = int(sys.argv[1])
meas = sys.argv[2]
seed = int(sys.argv[3])
rng_py = __import__('random').Random(seed)

# ------------------------------------------------------------------ spectral measure
if meas == 'rand':
    K = 3
    vals = sorted(Fraction(rng_py.randint(30, 250), 100) for _ in range(K))
    wts = [Fraction(rng_py.randint(1, 40), 1) for _ in range(K + 1)]   # w0, w1..wK (each +/-)
    tot = wts[0] + 2 * sum(wts[1:])
    coord = [(Fraction(0), wts[0] / tot)] + [(s * v, wts[k + 1] / tot) for k, v in enumerate(vals) for s in (1, -1)]
    desc = 'rand: values %s weights %s (sum %s)' % ([str(v) for v in vals], [str(x) for x in wts], tot)
else:
    L = float(meas.split('=')[1])
    h = 2 * math.pi / L
    Kmax = 1
    while math.exp(-(h * Kmax) ** 2 / 2) > 1e-22:
        Kmax += 1
    coord_f = [(h * k, math.exp(-(h * k) ** 2 / 2)) for k in range(-Kmax, Kmax + 1)]
    z = sum(p for _, p in coord_f)
    coord = [(v, p / z) for v, p in coord_f]
    desc = 'lattice L=%g, |m|<=%d, omitted coord mass < %.1e' % (L, Kmax, 2 * math.exp(-(h * (Kmax)) ** 2 / 2) / z)
print('d=%d measure %s' % (d, desc))
cv = np.array([float(v) for v, _ in coord])
cp = np.array([float(p) for _, p in coord])
assert abs(cp.sum() - 1) < 1e-14

# full product support
Q = np.array(list(itertools.product(cv, repeat=d)))
Wq = np.prod(np.array(list(itertools.product(cp, repeat=d))), axis=1)
# half set: first nonzero coordinate positive (zero mode contributes nothing to any variable used)
def first_nonzero_positive(q):
    for x in q:
        if x > 0:
            return True
        if x < 0:
            return False
    return False
hmask = np.array([first_nonzero_positive(q) for q in Q])
QH = Q[hmask]
c = np.sqrt(2 * Wq[hmask])
print('support size %d, half-set %d, total mass of half-set*2 + w0 = %.16f' % (len(Q), len(QH), 2 * Wq[hmask].sum() + Wq[~hmask].sum()))

iu = [(i, j) for i in range(d) for j in range(i, d)]          # upper-triangle entries of a symmetric matrix

# moments a,b,c
a_mom = float(np.sum(cp * cv ** 2)); b_mom = float(np.sum(cp * cv ** 4)); c_mom = float(np.sum(cp * cv ** 6))
print('coordinate moments a=%.15g b=%.15g c=%.15g  (continuum limit 1,3,15)' % (a_mom, b_mom, c_mom))

# ------------------------------------------------------------------ (a) C_alg via formula (1), jets from coefficients
def jets_at_zero(e):
    """Coefficient rows (over 2*|H| Gaussian coordinates [X_q..., Y_q...]) of D, H(upper), T_e at x=0."""
    qe = QH @ e
    n = len(QH)
    Drow = np.zeros((d, 2 * n)); Drow[:, n:] = (c * QH.T)                      # sin: grad = q
    Hrow = np.zeros((len(iu), 2 * n))
    for k, (i, j) in enumerate(iu):
        Hrow[k, :n] = -c * QH[:, i] * QH[:, j]                                 # cos: Hess = -q q^T
    Trow = np.zeros((1, 2 * n)); Trow[0, n:] = -c * qe ** 3                    # sin: d^3/ds^3 sin = -(q.e)^3 cos
    return Drow, Hrow, Trow


def sym_from_upper(vec):
    m = np.zeros(vec.shape[:-1] + (d, d))
    for k, (i, j) in enumerate(iu):
        m[..., i, j] = vec[..., k]
        m[..., j, i] = vec[..., k]
    return m


def frame(e):
    # orthonormal basis of e-perp (any; F is frame independent)
    M = np.eye(d)
    M[:, 0] = e
    q, _ = np.linalg.qr(M)
    U = q[:, 1:]
    return U


def calg_parts(e, rngF, nF):
    Drow, Hrow, Trow = jets_at_zero(e)
    covD = Drow @ Drow.T
    # He as linear functional of upper entries
    He_rows = np.zeros((d, 2 * len(QH)))
    Hfull = lambda i, j: Hrow[iu.index((min(i, j), max(i, j)))]
    for i in range(d):
        He_rows[i] = sum(Hfull(i, j) * e[j] for j in range(d))
    B = He_rows @ He_rows.T
    U = frame(e)
    iuA = [(i, j) for i in range(d - 1) for j in range(i, d - 1)]
    A_rows = np.zeros((len(iuA), 2 * len(QH)))
    for k, (i, j) in enumerate(iuA):
        A_rows[k] = sum(Hfull(p, q) * U[p, i] * U[q, j] for p in range(d) for q in range(d))
    SAA = A_rows @ A_rows.T; SAB = A_rows @ He_rows.T
    Gam = SAA - SAB @ np.linalg.solve(B, SAB.T)
    cTD = Trow @ Drow.T
    V = float((Trow @ Trow.T - cTD @ np.linalg.solve(covD, cTD.T))[0, 0])
    if d == 2:
        F = Gam[0, 0] / 2.0          # E[A^2 1{A<0}] for A ~ N(0, gamma)
        Fse = 0.0
    else:
        Lg = np.linalg.cholesky(Gam)
        Z = rngF.standard_normal((nF, len(iuA)))
        Av = Z @ Lg.T
        Am = np.zeros((nF, d - 1, d - 1))
        for k, (i, j) in enumerate(iuA):
            Am[:, i, j] = Av[:, k]; Am[:, j, i] = Av[:, k]
        dets = np.linalg.det(Am)
        ev = np.linalg.eigvalsh(Am)
        val = dets ** 2 * (ev.max(axis=1) < 0)
        F = val.mean(); Fse = val.std() / math.sqrt(nF)
    return covD, B, V, F, Fse, Gam


def sphere_rule(nth, nph):
    if d == 2:
        th = 2 * math.pi * np.arange(nth) / nth
        E = np.stack([np.cos(th), np.sin(th)], 1)
        Wt = np.full(nth, 2 * math.pi / nth)
        return E, Wt
    if d == 3:
        x, wx = np.polynomial.legendre.leggauss(nth)
        ph = 2 * math.pi * np.arange(nph) / nph
        E = []; Wt = []
        for xi, wi in zip(x, wx):
            s = math.sqrt(1 - xi * xi)
            for p in ph:
                E.append([s * math.cos(p), s * math.sin(p), xi]); Wt.append(wi * 2 * math.pi / nph)
        return np.array(E), np.array(Wt)
    raise SystemExit('d must be 2 or 3')


nth, nph = (24, 0) if d == 2 else (10, 20)
E_dirs, W_dirs = sphere_rule(nth, nph)
print('sphere rule: %d nodes, total weight %.15f (area |S^%d| = %.15f)' % (len(E_dirs), W_dirs.sum(), d - 1,
      2 * math.pi ** (d / 2) / math.gamma(d / 2)))

rngF = np.random.default_rng(seed + 1000)
pref = math.gamma(7 / 6) / (24 ** (1 / 3) * math.sqrt(math.pi) * (2 * math.pi) ** d)
C_alg = 0.0; C_alg_var = 0.0; C_fp = 0.0
gh_x, gh_w = np.polynomial.hermite_e.hermegauss(200)   # for the one-sided moment by quadrature
parts = []
for e, wt in zip(E_dirs, W_dirs):
    covD, B, V, F, Fse, Gam = calg_parts(e, rngF, 400000)
    a_here = covD[0, 0]
    assert np.allclose(covD, a_here * np.eye(d), rtol=1e-12, atol=1e-14)
    integrand = F * V ** (2 / 3) / math.sqrt(np.linalg.det(B))
    C_alg += wt * pref * a_here ** (-d / 2) * integrand
    C_alg_var += (wt * pref * a_here ** (-d / 2) * V ** (2 / 3) / math.sqrt(np.linalg.det(B)) * Fse) ** 2
    # first-principles constant: 12^(-1/3) p_D(0) p_He(0) F E[(T)_+^(4/3)]  with the moment by quadrature
    pD0 = (2 * math.pi) ** (-d / 2) / math.sqrt(np.linalg.det(covD))
    pHe0 = (2 * math.pi) ** (-d / 2) / math.sqrt(np.linalg.det(B))
    # E[(T)_+^(4/3)], T ~ N(0,V): adaptive mpmath quadrature (independent of the Gamma formula)
    mom = float(mp.quad(lambda t: t ** (mp.mpf(4) / 3) * mp.exp(-t * t / (2 * V)) / mp.sqrt(2 * mp.pi * V), [0, mp.inf]))
    C_fp += wt * 12 ** (-1 / 3) * pD0 * pHe0 * F * mom
    parts.append((e, wt, V, F, B))
print('a (from coefficient covariance of grad) = %.15g' % a_here)
print('C_alg (formula (1), area measure)          = %.10g  +- %.2g (MC in F for d=3)' % (C_alg, math.sqrt(C_alg_var)))
print('C_first_principles (12^-1/3 pD pHe F E T+^(4/3), moment by quadrature) = %.10g ; ratio to C_alg = %.12f' % (C_fp, C_fp / C_alg))
# proof's closed forms for B, V vs direct (spot-check consistency, not used above)
kap4 = b_mom - 3 * a_mom ** 2; kap6 = c_mom - 15 * a_mom * b_mom + 30 * a_mom ** 3
mx = 0.0
for e, wt, V, F, B in parts:
    s4 = np.sum(e ** 4); s6 = np.sum(e ** 6)
    Vp = 6 * a_mom ** 3 + 9 * a_mom * kap4 * s4 + (kap6 - kap4 ** 2 / a_mom) * s6
    Bp = a_mom ** 2 * (np.eye(d) + 2 * np.outer(e, e)) + kap4 * np.diag(e ** 2)
    mx = max(mx, abs(Vp / V - 1), np.max(np.abs(Bp - B)) / np.max(np.abs(B)))
print('max rel. diff of proof closed forms (10),(11) vs direct coefficient computation: %.2e' % mx)

# ------------------------------------------------------------------ (b) finite-r Kac-Rice
def gfun(th):
    """(th cos th - sin th)/th^3, stable."""
    out = np.empty_like(th)
    small = np.abs(th) < 0.2
    t = th[small]
    s = np.zeros_like(t); term_pow = np.ones_like(t)
    for k in range(1, 12):
        s += (-1) ** k * (2 * k) / math.factorial(2 * k + 1) * term_pow
        term_pow = term_pow * t * t
    out[small] = s
    t = th[~small]
    out[~small] = (t * np.cos(t) - np.sin(t)) / t ** 3
    return out


def sinc(th):
    return np.sinc(th / math.pi)


def kr_blocks(e, r):
    qe = QH @ e
    th = r * qe / 2
    n = len(QH)
    # even block (cos coefficients): D_r, Hbar
    Dr = (c * (-2) * QH.T * (qe / 2) * sinc(th))                       # -2 q sin(th)/r
    Hb = np.array([-c * QH[:, i] * QH[:, j] * np.cos(th) for (i, j) in iu])
    # odd block (sin coefficients): P, W, G
    P = c * QH.T * np.cos(th)
    W = c * 24 * (qe / 2) ** 3 * gfun(th)                               # (24/r^3)(th cos th - sin th)
    G = np.array([-2 * c * QH[:, i] * QH[:, j] * (qe / 2) * sinc(th) for (i, j) in iu])
    return Dr, Hb, P, W, G


def psd_sqrt(S):
    lam, vec = np.linalg.eigh((S + S.T) / 2)
    lam = np.clip(lam, 0, None)
    return vec * np.sqrt(lam)


Nmc, nbat = 48000, 8
rng = np.random.default_rng(seed)
Z1 = rng.standard_normal((Nmc, len(iu)))
Z2 = rng.standard_normal((Nmc, len(iu)))


def is_negdef(Hm):
    # Sylvester on -H
    if d == 2:
        return (Hm[:, 0, 0] < 0) & (np.linalg.det(Hm) > 0)
    return (Hm[:, 0, 0] < 0) & (Hm[:, 0, 0] * Hm[:, 1, 1] - Hm[:, 0, 1] ** 2 > 0) & (np.linalg.det(Hm) < 0)


def is_posdef(Hm):
    if d == 2:
        return (Hm[:, 0, 0] > 0) & (np.linalg.det(Hm) > 0)
    return (Hm[:, 0, 0] > 0) & (Hm[:, 0, 0] * Hm[:, 1, 1] - Hm[:, 0, 1] ** 2 > 0) & (np.linalg.det(Hm) > 0)


def one_positive(Hm):
    dt = np.linalg.det(Hm)
    if d == 2:
        return dt < 0
    return (dt > 0) & ~is_posdef(Hm)          # signs (+,-,-)


def kr_node(e, r, ell):
    Dr, Hb, P, W, G = kr_blocks(e, r)
    SDD = Dr @ Dr.T; SHD = Hb @ Dr.T; SHH = Hb @ Hb.T
    CH = SHH - SHD @ np.linalg.solve(SDD, SHD.T)
    O = np.vstack([P, W[None, :]])
    SOO = O @ O.T; SGO = G @ O.T; SGG = G @ G.T
    w = 12 * ell / r ** 3
    o = np.zeros(d + 1); o[-1] = w
    sol = np.linalg.solve(SOO, o)
    mG = SGO @ sol
    CG = SGG - SGO @ np.linalg.solve(SOO, SGO.T)
    pDr = (2 * math.pi) ** (-d / 2) / math.sqrt(np.linalg.det(SDD))
    pPW = (2 * math.pi) ** (-(d + 1) / 2) / math.sqrt(np.linalg.det(SOO)) * math.exp(-0.5 * o @ sol)
    Hbar = sym_from_upper(Z1 @ psd_sqrt(CH).T)
    Gm = sym_from_upper(mG + Z2 @ psd_sqrt(CG).T)
    HM = Hbar - (r / 2) * Gm
    HS = Hbar + (r / 2) * Gm
    val = np.abs(np.linalg.det(HM) * np.linalg.det(HS)) * is_negdef(HM) * one_positive(HS)
    bat = val.reshape(nbat, -1).mean(axis=1)
    raw_density = 12 * r ** (-(d + 3)) * pDr * pPW
    return raw_density, bat, (SDD, SOO, w)


def jac_check(e, r, ell):
    """mpmath 60-digit raw density of (grad f(M), grad f(S), f(M)-f(S)) at (0,0,ell) vs 12 r^-(d+3) p_(P,D_r,W)."""
    mp.mp.dps = 60
    rows = []
    n = len(QH)
    em = [mp.mpf(float(x)) for x in e]
    rr = mp.mpf(r)
    for (xs, comp) in [(-1, 'g'), (1, 'g'), (0, 'D')]:
        pass
    # coefficient rows over [X_q, Y_q]
    def val_f(q, x, kind):
        arg = sum(mp.mpf(float(qi)) * xi for qi, xi in zip(q, x))
        return mp.cos(arg) if kind == 0 else mp.sin(arg)
    def grad_f(q, x, kind, i):
        arg = sum(mp.mpf(float(qi)) * xi for qi, xi in zip(q, x))
        return (-mp.mpf(float(q[i])) * mp.sin(arg)) if kind == 0 else (mp.mpf(float(q[i])) * mp.cos(arg))
    Mx = [-rr / 2 * ei for ei in em]; Sx = [rr / 2 * ei for ei in em]
    cs = [mp.sqrt(2 * mp.mpf(float(wq))) for wq in Wq[hmask]]
    R = []
    for i in range(d):
        R.append([cs[k] * grad_f(QH[k], Mx, kind, i) for kind in (0, 1) for k in range(n)])
    for i in range(d):
        R.append([cs[k] * grad_f(QH[k], Sx, kind, i) for kind in (0, 1) for k in range(n)])
    R.append([cs[k] * (val_f(QH[k], Mx, kind) - val_f(QH[k], Sx, kind)) for kind in (0, 1) for k in range(n)])
    Rm = mp.matrix(R)
    S = Rm * Rm.T
    x = mp.matrix([0] * (2 * d) + [mp.mpf(ell)])
    quad = (x.T * mp.lu_solve(S, x))[0]
    dens = (2 * mp.pi) ** (-mp.mpf(2 * d + 1) / 2) / mp.sqrt(mp.det(S)) * mp.exp(-quad / 2)
    return float(dens)


# s-grid (r = s ell^(1/3)) : log-trapezoid
us = np.linspace(math.log(0.35), math.log(30.0), 90)
du = us[1] - us[0]
results = {}
for ell in (1e-5, 1e-7, 1e-9):
    tot_bat = np.zeros(nbat)
    lead_ratio = []
    for idx, (e, wt) in enumerate(zip(E_dirs, W_dirs)):
        for u in us:
            s = math.exp(u)
            r = s * ell ** (1 / 3)
            dens, bat, aux = kr_node(e, r, ell)
            jac = r ** (d - 1) * s * du * ell ** (1 / 3)        # r^{d-1} dr, dr = ell^(1/3) s du
            tot_bat += wt * jac * dens * bat
    nu = tot_bat.mean(); se = tot_bat.std(ddof=1) / math.sqrt(nbat)
    results[ell] = (nu * ell ** (1 / 3), se * ell ** (1 / 3))
    print('ell=%.0e : nu(ell) ell^(1/3) = %.8g +- %.2g   ratio to C_alg = %.5f +- %.5f' % (
        ell, nu * ell ** (1 / 3), se * ell ** (1 / 3), nu * ell ** (1 / 3) / C_alg, se * ell ** (1 / 3) / C_alg))

# Jacobian spot check
e0 = E_dirs[min(3, len(E_dirs) - 1)]
for ell, s in ((1e-5, 1.3), (1e-7, 2.0)):
    r = s * ell ** (1 / 3)
    dens, _, _ = kr_node(e0, r, ell)
    raw = jac_check(e0, r, ell)
    print('Jacobian spot check ell=%.0e s=%.1f: 12 r^-(d+3) p_(P,Dr,W)= %.12g ; raw mpmath p_(DM,DS,Delta) = %.12g ; rel diff %.2e' % (
        ell, s, dens, raw, abs(dens / raw - 1)))
