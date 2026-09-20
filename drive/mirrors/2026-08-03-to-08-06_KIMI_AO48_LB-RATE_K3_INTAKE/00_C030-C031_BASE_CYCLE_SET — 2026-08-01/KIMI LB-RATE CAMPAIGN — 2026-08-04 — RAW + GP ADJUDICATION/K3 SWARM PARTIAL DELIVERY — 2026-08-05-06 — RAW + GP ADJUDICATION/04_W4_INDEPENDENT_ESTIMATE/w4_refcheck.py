#!/usr/bin/env python3
"""
W4 INDEPENDENT - independent mpmath reference for E_win / P(det<0 & window)
at probe points. Fixed-node Tanh-Sinh quadrature at DPS digits (different
mechanism from the float64 Gauss-Legendre nested quadrature in w4_rho.py).
Infinite integrals truncated at |t|<=10 with certified Gaussian tails.

Usage: python3 w4_refcheck.py y1 y2 [N]
"""
import sys
import os
from mpmath import mp, mpf, tanh, sinh, cosh, exp as mexp, sqrt as msqrt, pi as mpi

import w4_condlaw as CL

mp.dps = int(os.environ.get("W4_DPS", "50"))

B = mpf(6) / 5
ELL = CL.ELL


def ck(cond, msg):
    if not cond:
        sys.stderr.write("CK-FAIL: %s\n" % msg)
        raise SystemExit(2)
    return True


def ts_nodes(h=mpf("0.09"), tmax=mpf("3.2")):
    """Tanh-Sinh nodes/weights on [-1,1], step h, |t| <= tmax."""
    xs = [mpf(0)]
    ws = [mpi / 2 * h]
    t = h
    while t <= tmax:
        u = mpi / 2 * sinh(t)
        x = tanh(u)
        w = mpi / 2 * cosh(t) / (cosh(u) ** 2) * h
        xs.append(x)
        ws.append(w)
        t += h
    nodes = [-x for x in reversed(xs[1:])] + xs
    weights = [w for w in reversed(ws[1:])] + ws
    return nodes, weights


def mphin(x):
    return mexp(-x * x / 2) / msqrt(2 * mpi)


def mPhi(x):
    from mpmath import erf
    return (1 + erf(x / msqrt(2))) / 2


def mK(v):
    if v < -30:
        inv = 1 / (v * v)
        return mphin(v) * inv * (1 + inv * (-3 + inv * (15 + inv * (-105 + inv * 945))))
    return v * mPhi(v) + mphin(v)


def reference(y, h_inner="0.035", h_u="0.12"):
    m4, S4 = CL.cond_fH_given_grad0(y)
    mf = m4[0]
    vf = S4[0, 0]
    sf = msqrt(vf)
    muH = [m4[1], m4[2], m4[3]]
    beta = [S4[1, 0] / vf, S4[2, 0] / vf, S4[3, 0] / vf]
    SH = [[S4[1 + i, 1 + j] - beta[i] * beta[j] * vf for j in range(3)] for i in range(3)]
    vZ = SH[1][1]
    sZ = msqrt(vZ)
    cX = SH[0][1] / vZ
    cY = SH[2][1] / vZ
    s1 = msqrt(SH[0][0] - vZ * cX * cX)
    s2 = msqrt(SH[2][2] - vZ * cY * cY)
    rho = (SH[0][2] - vZ * cX * cY) / (s1 * s2)
    ck(abs(rho) < 1, "corr < 1")
    sigy = s2 * msqrt(1 - rho ** 2)

    xn, xw = ts_nodes(mpf(h_inner), mpf("4.2"))
    xnu, xwu = ts_nodes(mpf(h_u), mpf("2.6"))
    WMAX = mpf(10)

    def F(w, z, u, want):
        du = u - mf
        muXu = muH[0] + beta[0] * du
        muYu = muH[2] + beta[2] * du
        muZu = muH[1] + beta[1] * du
        m1 = muXu + cX * (z - muZu)
        m2 = muYu + cY * (z - muZu)
        a = m1 + s1 * w
        muy = m2 + rho * s2 * w
        c = z * z
        if a == 0:
            return c if want == "win" else mpf(1)
        t = (c / a - muy) / sigy
        st = t if a > 0 else -t
        if want == "win":
            return abs(a) * sigy * mK(st)
        return mPhi(st)

    # precompute w-grid values
    wnodes = [WMAX * x for x in xn]
    wweights = [WMAX * w for w in xw]
    phiw = [mphin(w) for w in wnodes]

    def h(z, u, want):
        return sum(wweights[i] * phiw[i] * F(wnodes[i], z, u, want) for i in range(len(wnodes)))

    def g(u, want):
        du = u - mf
        muZu = muH[1] + beta[1] * du
        znodes = [muZu + sZ * WMAX * x for x in xn]
        return sum((WMAX * xw[i]) * mphin(WMAX * xn[i]) * h(znodes[i], u, want)
                   for i in range(len(znodes)))

    def Ewin(want):
        mid = B - ELL / 2
        tot = mpf(0)
        for i in range(len(xnu)):
            u = mid + ELL / 2 * xnu[i]
            tot += (ELL / 2 * xwu[i]) * mphin((u - mf) / sf) / sf * g(u, want)
        return tot

    ew = Ewin("win")
    pw = Ewin("prob")
    return ew, pw


if __name__ == "__main__":
    y = (mpf(sys.argv[1]), mpf(sys.argv[2]))
    hi = sys.argv[3] if len(sys.argv) > 3 else "0.035"
    ew, pw = reference(y, hi)
    print("REFCHECK-TS y=(%s,%s) h_inner=%s dps=%d" % (sys.argv[1], sys.argv[2], hi, mp.dps))
    print("E_win_mpmath =", mp.nstr(ew, 20))
    print("P_detneg_win_mpmath =", mp.nstr(pw, 20))
