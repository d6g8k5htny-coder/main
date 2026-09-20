#!/usr/bin/env python3
"""
W4 INDEPENDENT - Monte Carlo cross-check of E_win / P(det<0 & window) at
probe points, with importance sampling on f (tilt into the window) and exact
conditional Gaussian sampling of H | f = u. Deterministic seed.
Reports estimate +/- 2 sigma (CLT) and a Hoeffding-style bound note.

Usage: python3 w4_mc.py y1 y2 N
"""
import sys
import numpy as np
from mpmath import mp, mpf

mp.dps = 50
import w4_condlaw as CL


def mc_probe(y1, y2, N):
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
    LH = np.linalg.cholesky(SH)
    b, ell = 1.2, float(CL.ELL)
    mw = b - ell / 2

    rng = np.random.default_rng(20260805)
    Ews = []
    Pws = []
    M = 20000  # batch
    done = 0
    while done < N:
        m = min(M, N - done)
        u = mw + sf * rng.standard_normal(m)
        wgt = np.exp(-((u - mf) ** 2 - (u - mw) ** 2) / (2 * vf))
        inw = (u > b - ell) & (u < b)
        Z = rng.standard_normal((m, 3))
        H = muH + np.outer(u - mf, beta) + Z @ LH.T
        det = H[:, 0] * H[:, 2] - H[:, 1] ** 2
        neg = det < 0
        Ews.append((wgt * inw * np.abs(det) * neg))
        Pws.append((wgt * inw * neg).astype(np.float64))
        done += m
    X = np.concatenate(Ews)
    P = np.concatenate(Pws)

    def rep(V):
        mu = V.mean()
        sd = V.std(ddof=1) / np.sqrt(len(V))
        return mu, sd

    e_mu, e_sd = rep(X)
    p_mu, p_sd = rep(P)
    print("MC y=(%s,%s) N=%d seed=20260805" % (y1, y2, N))
    print("E_win_MC = %.10e +/- %.2e (2sd)" % (e_mu, 2 * e_sd))
    print("P_win_MC = %.10e +/- %.2e (2sd)" % (p_mu, 2 * p_sd))
    return e_mu, e_sd, p_mu, p_sd


if __name__ == "__main__":
    mc_probe(sys.argv[1], sys.argv[2], int(sys.argv[3]))
