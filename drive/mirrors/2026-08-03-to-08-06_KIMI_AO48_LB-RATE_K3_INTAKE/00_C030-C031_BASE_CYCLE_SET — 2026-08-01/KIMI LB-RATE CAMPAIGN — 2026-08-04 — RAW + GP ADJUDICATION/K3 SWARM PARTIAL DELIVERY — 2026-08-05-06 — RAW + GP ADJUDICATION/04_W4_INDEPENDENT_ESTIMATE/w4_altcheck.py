#!/usr/bin/env python3
"""
W4 INDEPENDENT - alternative decomposition cross-check for E_win and
P(det<0 & window): integrate (f, X=H11, Y=H22) outer, with the
Z=H12 integral in CLOSED FORM (different conditioning order from
w4_rho.py, which integrates (f, Z) outer and (X|Z,w) inner).

For Z ~ N(m, s^2), q = XY:
  G(q)  = E[(Z^2 - q) 1{Z^2 > q}]
  Pr(q) = P(Z^2 > q)
closed forms via Phi/phi. Then
  E_win = int_{b-ell}^b phi(u) E_{(X,Y)|f=u}[ G(XY) ] du,
  P_win = int_{b-ell}^b phi(u) E_{(X,Y)|f=u}[ Pr(XY) ] du.

Float64 Gauss-Legendre (truncated, certified tails). Deterministic.
Usage: python3 w4_altcheck.py y1 y2
"""
import sys
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import ndtr
from mpmath import mp, mpf

mp.dps = 50
import w4_condlaw as CL

SQRT2PI = np.sqrt(2 * np.pi)


def phi64(x):
    return np.exp(-0.5 * x * x) / SQRT2PI


def Gfun(q, m, s):
    """E[(Z^2 - q) 1{Z^2 > q}] and P(Z^2 > q), Z ~ N(m, s^2).
    Vectorized in q and m (same shape); s scalar."""
    q = np.asarray(q, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    neg = q < 0
    G = np.empty_like(q)
    P = np.empty_like(q)
    G[neg] = s * s + m[neg] ** 2 - q[neg]
    P[neg] = 1.0
    if np.any(~neg):
        r = np.sqrt(q[~neg])
        mm = m[~neg]
        tp = (r - mm) / s
        tm = (-r - mm) / s
        # E[Z^2 1{Z>c}] = (m^2+s^2)(1-Phi(t)) + s(2m + s t) phi(t)
        EZ2 = ((mm * mm + s * s) * (1 - ndtr(tp)) + s * (2 * mm + s * tp) * phi64(tp)
               + (mm * mm + s * s) * ndtr(tm) + s * (2 * mm + s * tm) * phi64(tm))
        Pp = (1 - ndtr(tp)) + ndtr(tm)
        G[~neg] = EZ2 - q[~neg] * Pp
        P[~neg] = Pp
    return G, P


def alt_Ewin(y1, y2, Nu=32, Nw=64):
    y = (mpf(y1), mpf(y2))
    m4, S4 = CL.cond_fH_given_grad0(y)
    m4 = np.array([float(v) for v in m4])
    S4 = np.array([[float(S4[i, j]) for j in range(4)] for i in range(4)])
    mf, vf = m4[0], S4[0, 0]
    sf = np.sqrt(vf)
    beta = S4[1:4, 0] / vf
    muH = m4[1:4]
    SH = S4[1:4, 1:4] - np.outer(beta, beta) * vf
    SH = 0.5 * (SH + SH.T)
    # order in SH: (X=H11, Z=H12, Y=H22) -> rearrange to (X,Y,Z)
    # (X,Y) block
    Sxy = np.array([[SH[0, 0], SH[0, 2]], [SH[0, 2], SH[2, 2]]])
    Sz_xy = np.array([SH[0, 1], SH[2, 1]])       # Cov(Z, (X,Y))
    Sxy_inv = np.linalg.inv(Sxy)
    sZ2 = SH[1, 1] - Sz_xy @ Sxy_inv @ Sz_xy
    assert_free = sZ2 > 0
    if not assert_free:
        sys.stderr.write("CK-FAIL: sZ2 <= 0\n")
        raise SystemExit(2)
    sZ = np.sqrt(sZ2)
    cvec = Sz_xy @ Sxy_inv     # Z | X,Y: mZ = muZ + cvec . ((x,y) - muXY)

    xu, wu = leggauss(Nu)
    b, ell = 1.2, float(CL.ELL)
    u = b - ell / 2 + ell / 2 * xu
    wt_u = ell / 2 * wu
    xw, ww = leggauss(Nw)
    W = 10.0
    w1 = W * xw
    wt = W * ww
    ph1 = phi64(w1)

    E_win = 0.0
    P_win = 0.0
    for iu in range(Nu):
        du = u[iu] - mf
        mu = muH + beta * du
        muX, muZ, muY = mu
        muXY = np.array([muX, muY])
        L = np.linalg.cholesky(Sxy)
        # whiten: (x,y) = muXY + L @ (w1, w2)
        W1, W2 = np.meshgrid(w1, w1, indexing="ij")
        XY1 = muX + L[0, 0] * W1 + L[0, 1] * W2
        XY2 = muY + L[1, 0] * W1 + L[1, 1] * W2
        mZ = muZ + cvec[0] * (XY1 - muX) + cvec[1] * (XY2 - muY)
        q = XY1 * XY2
        G, P = Gfun(q, mZ, sZ)
        w2d = np.outer(wt * ph1, wt * ph1)
        E_win += wt_u[iu] * phi64(du / sf) / sf * (w2d * G).sum()
        P_win += wt_u[iu] * phi64(du / sf) / sf * (w2d * P).sum()
    return E_win, P_win


if __name__ == "__main__":
    y1, y2 = sys.argv[1], sys.argv[2]
    ew, pw = alt_Ewin(mpf(y1), mpf(y2))
    print("ALTCHECK-XY-outer y=(%s,%s)" % (y1, y2))
    print("E_win_alt = %.15e" % ew)
    print("P_detneg_win_alt = %.15e" % pw)
