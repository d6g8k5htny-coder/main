#!/usr/bin/env python3
"""Numerical evaluation of the remote contact kernel Lambda_j(x; b, k, u) of [RM] (13) for the
parent kernel K(z) = exp(-|z|^2/2) (and its L-periodization) in the plane d = 2, together with
its far-field value, the correlation-hole constant and the torus integral k * integral_X Lambda dx
that enters nu(1) = nu_near(1) + k integral_X Lambda ([CL] (1.4)).

Standard library only.  Deterministic nested Gauss quadrature; no random numbers except in the
optional Monte Carlo cross-check (fixed seed).  Nothing here is a proof or an enclosure.

    python -B -S remote.py            # full run, writes RESULTS.json (long: tens of minutes)
    python -B -S remote.py --check    # exact controls + replay of a small subset against RESULTS.json
    python -B -S remote.py --check --mutant NAME   # must exit 1

Conventions.  Coordinates x = (x1, z): x1 along the pin axis u, z transverse.  Jets are indexed by
derivative orders (a, c) = d^a/dx1^a d^c/dz^c.  Cov(d^alpha f(s), d^beta f(t)) = (-1)^|alpha|
(d^(alpha+beta) K)(t - s).  Contact observation U_0 = (f, f_x, f_z, f_xx, f_xz, f_xxx)(0) =
(b, 0, 0, 0, 0, 12k); A_0 = f_zz(0); Y_x = (f_x, f_z, f)(x) = (0, 0, b); H_x = (f_xx, f_xz, f_zz)(x).
w_0 = 36 k^2 A_0^2 1{A_0 < 0}; z_0 = E[w_0 | U_0] = 36 k^2 m_(2,b).
Lambda_j(x) = p_(Y_x | U_0)(0, 0, b) * E[w_0 F_j(H_x) | U_0, Y_x] / z_0,  F_j(H) = |det H| 1{index j}.
"""
import argparse
import json
import math
import os
import random
import sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
MUTANTS = ('hermite-sign', 'det-abs', 'weight-indicator')
MUT = None

SITE0 = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0)]   # f, f_x, f_z, f_xx, f_xz, f_zz, f_xxx at 0
SITEX = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]           # f, f_x, f_z, f_xx, f_xz, f_zz at x
OBS0 = [0, 1, 2, 3, 4, 6]        # U_0 within the 13-vector
OBSX = [8, 9, 7]                 # Y_x = (f_x, f_z, f)(x)
TARGETS = [5, 10, 11, 12]        # A_0 = f_zz(0), h1 = f_xx(x), h2 = f_xz(x), h3 = f_zz(x)


def require(cond, msg):
    if not cond:
        print('FAIL: ' + msg)
        sys.exit(1)


# ----------------------------------------------------------------------------- Hermite / kernel
def hermite(n, t):
    """Probabilists' Hermite polynomial He_n(t):  (d/dt)^n exp(-t^2/2) = (-1)^n He_n(t) exp(-t^2/2)."""
    h0, h1 = 1, t
    if n == 0:
        return h0
    for m in range(1, n):
        h0, h1 = h1, t * h1 - m * h0
    return h1


def kernel_derivative(a, c, x1, z, L=None, images=2):
    """(d^a/dx1^a d^c/dz^c) K_L(x1, z) for K(z) = exp(-|z|^2/2), K_L its normalized L-periodization.
    L = None: continuum kernel (single image)."""
    if L is None:
        return (-1) ** (a + c) * hermite(a, x1) * hermite(c, z) * math.exp(-(x1 * x1 + z * z) / 2)
    tot = 0.0
    norm = 0.0
    for m1 in range(-images, images + 1):
        for m2 in range(-images, images + 1):
            s1, s2 = x1 + L * m1, z + L * m2
            tot += (-1) ** (a + c) * hermite(a, s1) * hermite(c, s2) * math.exp(-(s1 * s1 + s2 * s2) / 2)
            norm += math.exp(-(L * L) * (m1 * m1 + m2 * m2) / 2)
    return tot / norm


def cov_entry(alpha, beta, dx1, dz, L=None):
    """Cov(d^alpha f(s), d^beta f(t)) with t - s = (dx1, dz)."""
    sign = (-1) ** (alpha[0] + alpha[1])
    if MUT == 'hermite-sign':
        sign = (-1) ** (beta[0] + beta[1])
    return sign * kernel_derivative(alpha[0] + beta[0], alpha[1] + beta[1], dx1, dz, L)


def joint_covariance(x1, z, L=None):
    jets = [(0, a) for a in SITE0] + [(1, a) for a in SITEX]
    n = len(jets)
    C = [[0.0] * n for _ in range(n)]
    for i in range(n):
        si, ai = jets[i]
        for j in range(n):
            sj, aj = jets[j]
            d1 = (x1 if sj == 1 else 0.0) - (x1 if si == 1 else 0.0)
            d2 = (z if sj == 1 else 0.0) - (z if si == 1 else 0.0)
            C[i][j] = cov_entry(ai, aj, d1, d2, L)
    return C


# ----------------------------------------------------------------------------- exact one-site table
def hermite_zero_exact(n):
    """He_n(0) exactly: 0 for odd n, (-1)^(n/2) (n-1)!! for even n."""
    if n % 2:
        return Fraction(0)
    v = Fraction(1)
    for m in range(1, n, 2):
        v *= m
    return v * (-1) ** (n // 2)


def exact_one_site_cov(alpha, beta):
    return (-1) ** (alpha[0] + alpha[1]) * hermite_zero_exact(alpha[0] + beta[0]) * hermite_zero_exact(alpha[1] + beta[1])


# ----------------------------------------------------------------------------- linear algebra
def cholesky(A):
    n = len(A)
    Lm = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j] - sum(Lm[i][k] * Lm[j][k] for k in range(j))
            if i == j:
                if s <= 0.0:
                    raise ValueError('matrix not positive definite (pivot %r at %d)' % (s, i))
                Lm[i][j] = math.sqrt(s)
            else:
                Lm[i][j] = s / Lm[j][j]
    return Lm


def chol_solve(Lm, b):
    n = len(Lm)
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - sum(Lm[i][k] * y[k] for k in range(i))) / Lm[i][i]
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (y[i] - sum(Lm[k][i] * x[k] for k in range(i + 1, n))) / Lm[i][i]
    return x


def submatrix(C, rows, cols):
    return [[C[i][j] for j in cols] for i in rows]


def condition(C, obs, values, targets):
    """Gaussian regression: law of C[targets] given C[obs] = values (zero prior means).
    Returns (mean, cov, obs_logdensity_at_values)."""
    Coo = submatrix(C, obs, obs)
    Lm = cholesky(Coo)
    alpha = chol_solve(Lm, values)
    logdet = 2.0 * sum(math.log(Lm[i][i]) for i in range(len(obs)))
    quad = sum(values[i] * alpha[i] for i in range(len(obs)))
    logdens = -0.5 * quad - 0.5 * logdet - 0.5 * len(obs) * math.log(2 * math.pi)
    mean = []
    cov = []
    Cto = submatrix(C, targets, obs)
    for r, t in enumerate(targets):
        mean.append(sum(Cto[r][i] * alpha[i] for i in range(len(obs))))
    solved = [chol_solve(Lm, Cto[r]) for r in range(len(targets))]
    for r, t in enumerate(targets):
        row = []
        for s, u in enumerate(targets):
            row.append(C[t][u] - sum(Cto[r][i] * solved[s][i] for i in range(len(obs))))
        cov.append(row)
    return mean, cov, logdens


# ----------------------------------------------------------------------------- quadrature rules
def gauss_legendre(n):
    nodes, weights = [], []
    for i in range(1, n + 1):
        x = math.cos(math.pi * (i - 0.25) / (n + 0.5))
        for _ in range(100):
            p0, p1 = 1.0, x
            for m in range(2, n + 1):
                p0, p1 = p1, ((2 * m - 1) * x * p1 - (m - 1) * p0) / m
            dp = n * (x * p1 - p0) / (x * x - 1.0)
            dx = p1 / dp
            x -= dx
            if abs(dx) < 1e-15:
                break
        p0, p1 = 1.0, x
        for m in range(2, n + 1):
            p0, p1 = p1, ((2 * m - 1) * x * p1 - (m - 1) * p0) / m
        dp = n * (x * p1 - p0) / (x * x - 1.0)
        nodes.append(x)
        weights.append(2.0 / ((1.0 - x * x) * dp * dp))
    return nodes, weights


def gauss_hermite(n):
    """Probabilists' Gauss-Hermite: sum w_i g(t_i) ~ E g(T), T ~ N(0, 1)."""
    nodes, weights = [], []
    for i in range(1, n + 1):
        if i == 1:
            x = math.sqrt(2 * n + 1) - 1.85575 * (2 * n + 1) ** (-1.0 / 6.0)
        elif i == 2:
            x = nodes[-1] - 1.14 * n ** 0.426 / nodes[-1]
        elif i == 3:
            x = 1.86 * nodes[-1] - 0.86 * nodes[0]
        elif i == 4:
            x = 1.91 * nodes[-1] - 0.91 * nodes[-2]
        else:
            x = 2.0 * nodes[-1] - nodes[-2]
        for _ in range(200):
            h0, h1 = 1.0, x
            for m in range(1, n):
                h0, h1 = h1, x * h1 - m * h0
            dh = n * h0
            dx = h1 / dh
            x -= dx
            if abs(dx) < 1e-14:
                break
        h0, h1 = 1.0, x
        for m in range(1, n):
            h0, h1 = h1, x * h1 - m * h0
        nodes.append(x)
        weights.append(math.factorial(n) / (n * n * h0 * h0))
    return nodes, weights


SQRT2PI = math.sqrt(2 * math.pi)


def phi(t):
    return math.exp(-t * t / 2) / SQRT2PI


def Phi(t):
    return 0.5 * math.erfc(-t / math.sqrt(2))


def m2_negative(m, s):
    """E[A^2 1{A < 0}] for A ~ N(m, s^2)."""
    if MUT == 'weight-indicator':
        return m * m + s * s
    t = m / s
    return (m * m + s * s) * Phi(-t) - m * s * phi(t)


class Rules:
    def __init__(self, n1=24, n2=16, n3=24, cut=9.0):
        self.gl1 = gauss_legendre(n1)
        self.gh2 = gauss_hermite(n2)
        self.gl3 = gauss_legendre(n3)
        self.cut = cut


def gaussian_segment(rule, lo, hi, func):
    """integral_lo^hi phi(t) func(t) dt by Gauss-Legendre on [lo, hi] (both finite)."""
    if hi <= lo:
        return 0.0
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    acc = 0.0
    for t, w in zip(*rule):
        u = mid + half * t
        acc += w * phi(u) * func(u)
    return acc * half


def weighted_determinant_integrals(mu, S, rules):
    """(A, h1, h2, h3) ~ N(mu, S).  Returns dict with
       G[j]  = E[ m2(A | H) * |det H| 1{index H = j} ]  = E[ A^2 1{A<0} |det H| 1{index j} ],
       F[j]  = E[ |det H| 1{index j} ],
       D     = E[ det H ]   (control)."""
    # A | H regression
    SHH = [[S[i][j] for j in (1, 2, 3)] for i in (1, 2, 3)]
    SAH = [S[0][j] for j in (1, 2, 3)]
    LH = cholesky(SHH)
    wA = chol_solve(LH, SAH)
    sA2 = S[0][0] - sum(SAH[i] * wA[i] for i in range(3))
    sA = math.sqrt(max(sA2, 0.0))
    muA, mu1, mu2, mu3 = mu
    # h2 | h1 and h3 | (h1, h2)
    s11, s12, s13, s22, s23, s33 = S[1][1], S[1][2], S[1][3], S[2][2], S[2][3], S[3][3]
    s1 = math.sqrt(s11)
    b21 = s12 / s11
    s2 = math.sqrt(max(s22 - s12 * s12 / s11, 0.0))
    L2 = cholesky([[s11, s12], [s12, s22]])
    c3 = chol_solve(L2, [s13, s23])
    s3 = math.sqrt(max(s33 - (c3[0] * s13 + c3[1] * s23), 0.0))
    cut = rules.cut
    G = [0.0, 0.0, 0.0]
    F = [0.0, 0.0, 0.0]
    D = 0.0
    t0 = (0.0 - mu1) / s1
    for side in (0, 1):
        lo, hi = (-cut, min(t0, cut)) if side == 0 else (max(t0, -cut), cut)
        if hi <= lo:
            continue
        mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
        for t1, w1 in zip(*rules.gl1):
            u1 = mid + half * t1
            h1 = mu1 + s1 * u1
            wt1 = w1 * half * phi(u1)
            m2c = mu2 + b21 * (h1 - mu1)
            for g2, w2 in zip(*rules.gh2):
                h2 = m2c + s2 * g2
                wt12 = wt1 * w2
                m3c = mu3 + c3[0] * (h1 - mu1) + c3[1] * (h2 - mu2)
                # kink of det = h1 h3 - h2^2 in h3
                hstar = h2 * h2 / h1
                tstar = (hstar - m3c) / s3
                # det > 0 side: h1 > 0 -> h3 > hstar ; h1 < 0 -> h3 < hstar
                if h1 > 0:
                    segs = ((max(tstar, -cut), cut, 0), (-cut, min(tstar, cut), 1))   # (lo, hi, index j)
                else:
                    segs = ((-cut, min(tstar, cut), 2), (max(tstar, -cut), cut, 1))
                for lo3, hi3, j in segs:
                    if hi3 <= lo3:
                        continue
                    mid3, half3 = 0.5 * (lo3 + hi3), 0.5 * (hi3 - lo3)
                    accG = accF = 0.0
                    for t3, w3 in zip(*rules.gl3):
                        u3 = mid3 + half3 * t3
                        h3 = m3c + s3 * u3
                        det = h1 * h3 - h2 * h2
                        adet = det if MUT == 'det-abs' else abs(det)
                        mA = muA + wA[0] * (h1 - mu1) + wA[1] * (h2 - mu2) + wA[2] * (h3 - mu3)
                        ww = w3 * phi(u3)
                        accF += ww * adet
                        accG += ww * adet * m2_negative(mA, sA)
                    G[j] += wt12 * half3 * accG
                    F[j] += wt12 * half3 * accF
    D = mu1 * mu3 + S[1][3] - (mu2 * mu2 + S[2][2])
    return {'G': G, 'F': F, 'D': D}


# ----------------------------------------------------------------------------- Lambda
def m2b(b):
    return m2_negative(-b, math.sqrt(2.0))


def lambda_at(x1, z, b, k, rules, L=None):
    """Returns dict: Lambda_j (j = 0, 1, 2), Lambda (sum), the unweighted remote density rho_j =
    p * E[F_j | obs] (what the same point would carry without the pin weight), logp."""
    C = joint_covariance(x1, z, L)
    u0 = [b, 0.0, 0.0, 0.0, 0.0, 12.0 * k]
    # density of Y_x at (0, 0, b) given U_0
    meanY, covY, _ = condition(C, OBS0, u0, OBSX)
    LY = cholesky(covY)
    dev = [0.0 - meanY[0], 0.0 - meanY[1], b - meanY[2]]
    a = chol_solve(LY, dev)
    logp = -0.5 * sum(dev[i] * a[i] for i in range(3)) - sum(math.log(LY[i][i]) for i in range(3)) - 1.5 * math.log(2 * math.pi)
    p = math.exp(logp)
    obs = OBS0 + OBSX
    vals = u0 + [0.0, 0.0, b]
    mu, S, _ = condition(C, obs, vals, TARGETS)
    W = weighted_determinant_integrals(mu, S, rules)
    mm = m2b(b)
    lam = [p * W['G'][j] / mm for j in range(3)]
    rho = [p * W['F'][j] for j in range(3)]
    return {'lambda_j': lam, 'lambda': sum(lam), 'rho_j': rho, 'rho': sum(rho), 'logp': logp,
            'coupling': (W['G'][0] + W['G'][1] + W['G'][2]) / (mm * (W['F'][0] + W['F'][1] + W['F'][2]))}


def lambda_far(b, rules):
    """Far-field value: p = phi(b)/(2 pi), H | (f = b, grad f = 0): h1, h3 ~ N(-b, 2) independent,
    h2 ~ N(0, 1); A_0 independent of H.  Lambda_inf,j = phi(b)/(2 pi) E[F_j(H)]."""
    mu = [-b, -b, 0.0, -b]
    S = [[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 2.0]]
    W = weighted_determinant_integrals(mu, S, rules)
    p = phi(b) / (2 * math.pi)
    lam = [p * W['G'][j] / m2b(b) for j in range(3)]
    return {'lambda_j': lam, 'lambda': sum(lam), 'rho_j': [p * W['F'][j] for j in range(3)], 'E_F_j': W['F']}


def unconditional_hessian_controls(rules):
    """H unconditional: Var f_xx = Var f_zz = 3, Cov = 1, Var f_xz = 1.  E|det H| = sqrt 3 (Longuet-Higgins:
    total critical density sqrt3/(2 pi)), E F_2 = E F_0 = sqrt3/4, E F_1 = sqrt3/2, E det H = 0."""
    mu = [0.0, 0.0, 0.0, 0.0]
    S = [[2.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 3.0]]
    W = weighted_determinant_integrals(mu, S, rules)
    return {'E_absdet': sum(W['F']), 'E_F_j': W['F'], 'E_det': W['D']}


def mc_absdet(n, seed=2026):
    """Monte Carlo cross-check of E|det H| for the unconditional Hessian (fixed seed)."""
    rng = random.Random(seed)
    acc = acc2 = 0.0
    for _ in range(n):
        g1, g2, g3 = rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1)
        h1 = math.sqrt(2.0) * g1 + g3          # Var 3
        h3 = math.sqrt(2.0) * g2 + g3          # Var 3, Cov(h1, h3) = 1
        h2 = rng.gauss(0, 1)
        v = abs(h1 * h3 - h2 * h2)
        acc += v
        acc2 += v * v
    mean = acc / n
    var = acc2 / n - mean * mean
    return mean, math.sqrt(var / n)


# ----------------------------------------------------------------------------- grids
def hole_integral(b, k, rules, Rc=5.0, h=0.2, lam_inf=None):
    """2 * sum over the midpoint grid on [-Rc, Rc] x (0, Rc] of (Lambda(x) - Lambda_inf) h^2, by index.
    Continuum kernel."""
    n1 = int(round(2 * Rc / h))
    n2 = int(round(Rc / h))
    tot = [0.0, 0.0, 0.0]
    tot_rho = [0.0, 0.0, 0.0]
    minlam, maxlam = float('inf'), float('-inf')
    for i in range(n1):
        x1 = -Rc + (i + 0.5) * h
        for j in range(n2):
            z = (j + 0.5) * h
            r = lambda_at(x1, z, b, k, rules)
            for q in range(3):
                tot[q] += r['lambda_j'][q] - lam_inf['lambda_j'][q]
                tot_rho[q] += r['rho_j'][q] - lam_inf['rho_j'][q]
            minlam = min(minlam, r['lambda'])
            maxlam = max(maxlam, r['lambda'])
    scale = 2.0 * h * h
    return {'hole_j': [scale * t for t in tot], 'hole': scale * sum(tot),
            'hole_unweighted_j': [scale * t for t in tot_rho], 'hole_unweighted': scale * sum(tot_rho),
            'grid': {'Rc': Rc, 'h': h, 'points': n1 * n2, 'symmetry': 'z -> -z'},
            'lambda_min_on_grid': minlam, 'lambda_max_on_grid': maxlam}


def torus_integral(b, k, rules, L, h=0.2):
    """integral over the fundamental domain [-L/2, L/2]^2 of Lambda_L(x) dx (periodized kernel), by index."""
    n1 = int(round(L / h))
    n2 = int(round(L / (2 * h)))
    tot = [0.0, 0.0, 0.0]
    for i in range(n1):
        x1 = -L / 2 + (i + 0.5) * h
        for j in range(n2):
            z = (j + 0.5) * h
            r = lambda_at(x1, z, b, k, rules, L)
            for q in range(3):
                tot[q] += r['lambda_j'][q]
    scale = 2.0 * h * h
    return {'integral_j': [scale * t for t in tot], 'integral': scale * sum(tot),
            'grid': {'L': L, 'h': h, 'points': n1 * n2, 'symmetry': 'z -> -z'}}


# ----------------------------------------------------------------------------- controls
SAMPLE_POINTS = [(0.15, 0.15), (0.5, 0.3), (-0.7, 0.9), (1.0, 0.0), (0.0, 1.0), (2.0, 1.5), (-1.5, 0.5), (3.0, 3.0)]


def controls(rules):
    out = {}
    # (a) exact one-site Hermite table versus the floating kernel derivatives at 0 (continuum and L = 24, 12)
    table = {}
    maxdev = {'continuum': 0.0, 'L24': 0.0, 'L12': 0.0, 'L6': 0.0}
    for a in SITE0:
        for c in SITE0:
            ex = exact_one_site_cov(a, c)
            table['%d%d,%d%d' % (a + c)] = str(ex)
            fl = cov_entry(a, c, 0.0, 0.0)
            maxdev['continuum'] = max(maxdev['continuum'], abs(fl - float(ex)))
            for L, key in ((24.0, 'L24'), (12.0, 'L12'), (6.0, 'L6')):
                maxdev[key] = max(maxdev[key], abs(cov_entry(a, c, 0.0, 0.0, L) - float(ex)))
    require(table['20,20'] == '3' and table['30,30'] == '15' and table['10,30'] == '-3' and table['00,20'] == '-1'
            and table['20,02'] == '1' and table['11,11'] == '1' and table['00,00'] == '1', 'exact Hermite one-site table')
    require(maxdev['continuum'] < 1e-12 and maxdev['L24'] < 1e-9 and maxdev['L12'] < 1e-9, 'lattice one-site covariances at L = 24, 12')
    require(1e-4 < maxdev['L6'] < 2e-3, 'L = 6 one-site deviation is of the expected size')
    out['one_site_exact'] = {'Var_fxx': table['20,20'], 'Var_fxxx': table['30,30'], 'Cov_fx_fxxx': table['10,30'],
                             'Cov_f_fxx': table['00,20'], 'Cov_fxx_fzz': table['20,02'], 'Var_fxz': table['11,11']}
    out['one_site_lattice_max_dev'] = maxdev
    # (b) unconditional Hessian: E|det H| = sqrt3, E F_2 = sqrt3/4, E F_1 = sqrt3/2, E det = 0
    U = unconditional_hessian_controls(rules)
    require(abs(U['E_absdet'] - math.sqrt(3)) < 1e-7, 'E|det H| = sqrt(3) (Longuet-Higgins total critical density sqrt3/(2 pi))')
    require(abs(U['E_F_j'][2] - math.sqrt(3) / 4) < 1e-7 and abs(U['E_F_j'][0] - math.sqrt(3) / 4) < 1e-7
            and abs(U['E_F_j'][1] - math.sqrt(3) / 2) < 1e-7, 'index split sqrt3/4, sqrt3/2, sqrt3/4')
    require(abs(U['E_det']) < 1e-12, 'E det H = 0 (Euler characteristic of the torus)')
    out['unconditional_hessian'] = U
    # (c) far field: E[w_0 | U_0] / z_0 = 1 and Lambda_inf = phi(b) E|det H_b| / (2 pi)
    far = {}
    for b in (0.0, 1.0):
        r = lambda_far(b, rules)
        far[str(b)] = r
        rr = lambda_at(8.0, 0.0, b, 1.0, rules)
        require(abs(rr['lambda'] - r['lambda']) < 1e-9 * r['lambda'], 'Lambda(8, 0) equals the far-field value, b = %r' % b)
        require(abs(rr['coupling'] - 1.0) < 1e-9, 'pin weight decouples at |x| = 8')
    out['far_field'] = far
    # (d) z -> -z symmetry and a few sample values
    sym = 0.0
    for (x1, z) in SAMPLE_POINTS[:4]:
        r1 = lambda_at(x1, z, 0.0, 1.0, rules)
        r2 = lambda_at(x1, -z, 0.0, 1.0, rules)
        sym = max(sym, abs(r1['lambda'] - r2['lambda']) / r1['lambda'])
    require(sym < 1e-10, 'reflection symmetry z -> -z')
    out['reflection_symmetry_max_rel_dev'] = sym
    # (e) k-dependence enters only through the f_xxx = 12k conditioning: at |x| = 8 Lambda is k-free
    a1 = lambda_at(8.0, 0.0, 0.0, 0.5, rules)['lambda']
    a2 = lambda_at(8.0, 0.0, 0.0, 2.0, rules)['lambda']
    require(abs(a1 - a2) < 1e-9 * a1, 'far field is k-free')
    return out


def sample_table(b, k, rules, L=None):
    rows = []
    for (x1, z) in SAMPLE_POINTS:
        r = lambda_at(x1, z, b, k, rules, L)
        rows.append({'x': [x1, z], 'lambda': r['lambda'], 'lambda_j': r['lambda_j'], 'rho': r['rho'], 'coupling': r['coupling']})
    return rows


def near_zero_profile(b, k, rules):
    rows = []
    for rad in (0.05, 0.1, 0.2, 0.4):
        for ang_deg in (0, 30, 60, 90, 120, 150, 180):
            th = math.radians(ang_deg)
            x1, z = rad * math.cos(th), rad * math.sin(th)
            if ang_deg in (0, 180):
                z = 0.0
            try:
                r = lambda_at(x1, z, b, k, rules)
                rows.append({'r': rad, 'angle_deg': ang_deg, 'lambda': r['lambda'], 'lambda_j': r['lambda_j']})
            except ValueError as exc:
                rows.append({'r': rad, 'angle_deg': ang_deg, 'lambda': None, 'error': str(exc)[:60]})
    return rows


# ----------------------------------------------------------------------------- main
def full_run(fast=False):
    rules = Rules()
    hi_rules = Rules(32, 24, 32)
    res = {'object': 'CL-C6-REMOTE-COEFF-NUMERICS-20260930-v1', 'scientific_effect': 'NONE', 'certified': False,
           'quadrature': {'h1_gauss_legendre_per_side': 24, 'h2_gauss_hermite': 16, 'h3_gauss_legendre_per_side': 24, 'cut_sd': 9.0,
                          'high_order_check': [32, 24, 32]}}
    res['controls'] = controls(rules)
    mc = mc_absdet(200000)
    res['controls']['mc_absdet_unconditional'] = {'mean': mc[0], 'se': mc[1], 'n': 200000, 'seed': 2026}
    # quadrature-order check at the sample points
    qdev = 0.0
    for (x1, z) in SAMPLE_POINTS:
        a = lambda_at(x1, z, 0.0, 1.0, rules)['lambda']
        c = lambda_at(x1, z, 0.0, 1.0, hi_rules)['lambda']
        qdev = max(qdev, abs(a - c) / c)
    res['quadrature']['max_rel_dev_vs_high_order_at_samples'] = qdev
    res['samples'] = {}
    res['near_zero_profile'] = {}
    res['far_field'] = {}
    res['hole'] = {}
    res['torus'] = {}
    res['remote_total'] = {}
    ks = (0.5, 1.0, 2.0)
    bs = (0.0, 1.0)
    grid_h = 0.4 if fast else 0.2
    for b in bs:
        far = lambda_far(b, rules)
        res['far_field'][str(b)] = {'lambda_inf': far['lambda'], 'lambda_inf_j': far['lambda_j'], 'E_F_j_given_level': far['E_F_j'],
                                    'p_inf': phi(b) / (2 * math.pi)}
        for k in ks:
            key = 'k=%s,b=%s' % (k, b)
            res['samples'][key] = sample_table(b, k, rules)
            res['near_zero_profile'][key] = near_zero_profile(b, k, rules)
            H = hole_integral(b, k, rules, Rc=5.0, h=grid_h, lam_inf=far)
            res['hole'][key] = H
            T6 = torus_integral(b, k, rules, 6.0, h=grid_h)
            res['torus'][key] = {'L=6': T6}
            rt = {}
            for L in (6.0, 12.0, 24.0):
                approx = L * L * far['lambda'] + H['hole']
                rt['L=%g' % L] = {'integral_X_Lambda_approx': approx, 'k_integral': k * approx}
            rt['L=6']['integral_X_Lambda_periodized_direct'] = T6['integral']
            rt['L=6']['k_integral_direct'] = k * T6['integral']
            res['remote_total'][key] = rt
            print('done', key, 'hole', H['hole'], 'L6 direct', T6['integral'], 'approx', 36 * far['lambda'] + H['hole'], flush=True)
    # grid-convergence check for one parameter set
    far0 = lambda_far(0.0, rules)
    Hc = hole_integral(0.0, 1.0, rules, Rc=5.0, h=0.4, lam_inf=far0)
    Ht = hole_integral(0.0, 1.0, rules, Rc=6.0, h=0.4, lam_inf=far0)
    res['grid_convergence'] = {'k=1,b=0': {'h=0.4,Rc=5': Hc['hole'], 'h=0.4,Rc=6': Ht['hole'], 'h=%s,Rc=5' % grid_h: res['hole']['k=1.0,b=0.0']['hole']}}
    return res


def check_run():
    rules = Rules()
    ctl = controls(rules)
    path = os.path.join(HERE, 'RESULTS.json')
    require(os.path.exists(path), 'RESULTS.json present')
    with open(path) as fh:
        ref = json.load(fh)
    for b in (0.0, 1.0):
        far = lambda_far(b, rules)
        require(abs(far['lambda'] - ref['far_field'][str(b)]['lambda_inf']) < 1e-10, 'far-field replay b = %r' % b)
    for key, k in (('k=1.0,b=0.0', 1.0), ('k=0.5,b=1.0', 0.5)):
        b = 0.0 if key.endswith('0.0') else 1.0
        rows = sample_table(b, k, rules)
        for r, rr in zip(rows, ref['samples'][key]):
            require(abs(r['lambda'] - rr['lambda']) <= 1e-9 * max(1.0, abs(rr['lambda'])), 'sample replay %s at %r' % (key, r['x']))
    # periodized kernel replay at two points
    for (x1, z) in ((0.5, 0.3), (2.0, 1.5)):
        r = lambda_at(x1, z, 0.0, 1.0, rules, 6.0)
        # the torus values are not stored point-wise; check they differ from the continuum by less than 5e-2 relative
        c = lambda_at(x1, z, 0.0, 1.0, rules)
        require(abs(r['lambda'] - c['lambda']) < 5e-2 * c['lambda'], 'L = 6 periodization is a small correction at %r' % ((x1, z),))
    print(json.dumps({'check': 'ok', 'controls': list(ctl.keys()), 'mutant': MUT}))


def main():
    global MUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--fast', action='store_true', help='coarser grid (development only)')
    ap.add_argument('--mutant', choices=MUTANTS)
    args = ap.parse_args()
    MUT = args.mutant
    if args.check:
        check_run()
        return
    res = full_run(fast=args.fast)
    with open(os.path.join(HERE, 'RESULTS.json'), 'w') as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
        fh.write('\n')
    print('wrote RESULTS.json')


if __name__ == '__main__':
    main()
