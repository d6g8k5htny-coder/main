"""Prototype: two-sided ratio bounds A_r / A_0^ref on (b, k) sub-boxes, d = 2 and d = 3."""
import importlib.util
import math
import os
from fractions import Fraction as Fr

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


E2 = load('engine_d2', 'theorem_a.py')
E3 = load('engine_d3', 'theorem_a3.py')
R0C = 12            # Loewner coupling radius (d = 3)


def exp_minus(M, t):
    """Enclosure of exp(-t) for rational t of either sign."""
    t = Fr(t)
    if t >= 0:
        return M.exp_neg_point(t)
    return M.exp_neg_point(-t).inv()


def chol_det(M, C):
    """Interval Cholesky of an n x n interval SPD matrix (module M's IV); returns det = prod L_ii^2."""
    n = len(C)
    L = [[M.IV(0)] * n for _ in range(n)]
    det = M.IV(1)
    for i in range(n):
        for j in range(i + 1):
            s = C[i][j] - sum((L[i][l] * L[j][l] for l in range(j)), M.IV(0))
            if i == j:
                if s.lo <= 0:
                    raise ValueError('Cholesky pivot')
                L[i][j] = M.isqrt_iv(s)
                det = det * s
            else:
                L[i][j] = s / L[j][j]
    return det


def pin_det(M, r0, r1, L=10):
    pins, _ = M.functionals()
    P = [pins[n] for n in M.PIN_NAMES]
    M1 = M.torus_M1(L)
    n = len(P)
    S = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            S[i][j] = S[j][i] = M.cov_band(P[i], P[j], ('p', i, j), r0, r1, M1)
    return chol_det(M, S), n


class Ctx:
    pass


def context(M, r0, r1, d):
    ctx = Ctx()
    ctx.M, ctx.d, ctx.r0, ctx.r1 = M, d, r0, r1
    ctx.law = M.band_law(r0, r1)
    det, n = pin_det(M, r0, r1)
    ctx.detS, ctx.n = det, n
    th = M.IV(1, 1 + ctx.law.th1)
    thn = M.IV(1)
    for _ in range(n):
        thn = thn * th
    # (det Sigma_0^ref / det Sigma_r)^(1/2) = (12 Theta^n / det S)^(1/2)
    q = thn * 12 / det
    ctx.detfac = M.isqrt_iv(q)
    ctx.sqrt2 = M.isqrt_iv(M.IV(2))
    return ctx


def pi_ratio(ctx, bb, kb):
    """Enclosure of pi_r / pi_0^ref over the sub-box (every r in the band, frame, L >= 10)."""
    M, law = ctx.M, ctx.law
    b = M.IV(bb[0], bb[1])
    k = M.IV(kb[0], kb[1])
    dE = b * b * (law.e_bb - Fr(3, 2)) + b * k * law.e_bk * 2 + k * k * (law.e_kk - 24)
    ex = M.IV(exp_minus(M, dE.hi / 2).lo, exp_minus(M, dE.lo / 2).hi)
    return ctx.detfac * ex


# ----------------------------------------------------------------------------------------------- d = 2
def F2(M, m, s):
    return M.elam2_pos(Fr(m), Fr(s))


def F1(M, m, s):
    return M.elam_pos(Fr(m), Fr(s))


def zeta_ratio_d2(ctx, bb, kb):
    """Enclosure of [Z_r / (36 k^2 r^2)] / F(b, sqrt2) over the sub-box, F(m, s) = E[(m + s z)_+^2] (the reference
    contact moment E[A^2 1{A < 0}] at f = b, V = 0 is F(b, sqrt2))."""
    M, law, r1 = ctx.M, ctx.law, ctx.r1
    b0, b1 = bb
    klo, khi = kb
    b = M.IV(b0, b1)
    k = M.IV(klo, khi)
    mu = {n: b * law.alpha[n] + k * law.beta[n] for n in M.TARGET_NAMES}
    mabs = {n: M.absup(mu[n]) for n in M.TARGET_NAMES}
    D = b * (-law.alpha['w'] - 1) + k * (-law.beta['w'])         # m_lambda - b
    s_lo, s_hi = law.sw.lo, law.sw.hi
    r2lo, r2hi = ctx.sqrt2.lo, ctx.sqrt2.hi
    sa, sb = min(s_lo, r2lo), max(s_hi, r2hi)
    m_lo, m_hi = b0 + min(D.lo, 0), b1 + max(D.hi, 0)
    Fmin = F2(M, m_lo, sa).lo
    Lm = 2 * F1(M, m_hi, sb).hi / Fmin
    Ls = 2 * sb / Fmin
    up = max(D.hi, 0) * Lm + max(s_hi - r2lo, 0) * Ls
    dn = max(-D.lo, 0) * Lm + max(r2hi - s_lo, 0) * Ls
    rat_hi = exp_minus(M, -up).hi
    rat_lo = exp_minus(M, dn).lo
    Fref_lo = F2(M, b0, r2lo).lo                                  # <= F(b, sqrt2) on the box
    # corrections (upper)
    gn = lambda n, p: M.gauss_norm(mabs[n], law.sdQ[n].hi, p)
    lam4 = M.gauss_norm(mabs['w'], s_hi, 4)
    six = 6 * klo
    eps2 = (2 * r1 * gn('T', 2) + r1 * r1 * gn('tau', 2)) / six + r1 * gn('T', 4) * (r1 * gn('T', 4) + r1 * r1 * gn('tau', 4)) / six ** 2
    corr_up = (eps2 * lam4 ** 2 + r1 * (1 + eps2) * lam4 * gn('om', 4)
               + r1 * (1 / six + r1 * gn('T', 4) / six ** 2) * lam4 * (r1 * gn('nu', 4) + gn('v', 4)) ** 2)
    # corrections (lower): [E] section 6, as in the engine
    m_lam = -mu['w']
    Elp = M.elam_pos(m_lam.hi, s_hi).hi
    muw_abs = mabs['w']
    lam_n = lambda p: muw_abs + s_hi * M.znorm(M.pow2(p))
    sdT, sdtau = law.sdQ['T'].hi, law.sdQ['tau'].hi
    q_bad = M.gauss_tail2(mabs['T'], sdT, 3 * klo / (2 * r1)) + M.gauss_tail2(mabs['tau'], sdtau, 3 * klo / (2 * r1 * r1))
    bw = {n: M.absup(law.bw[n]) for n in law.bw}
    Eom_g = mabs['om'] + law.sdg['om'].hi * Fr(4, 5)
    Evg2 = mabs['v'] ** 2 + law.sdg['v'].hi ** 2
    E_om_lam = Eom_g * Elp + bw['om'] * (muw_abs * lam_n(1) + lam_n(2) ** 2)
    E_v2_lam = 2 * Evg2 * Elp + 2 * bw['v'] ** 2 * (s_hi * M.znorm(4)) ** 2 * lam_n(2)
    corr_lo = ((lam_n(4) ** 2 * M.iroot(q_bad, 2) if q_bad > 0 else Fr(0))
               + (2 * r1 * M.gauss_norm(mabs['T'], sdT, 2) + r1 * r1 * M.gauss_norm(mabs['tau'], sdtau, 2)) * lam_n(4) ** 2 / six
               + r1 * (E_v2_lam / (3 * klo) + E_om_lam))
    hi = rat_hi + corr_up / Fref_lo
    lo = rat_lo - corr_lo / Fref_lo
    return lo, hi, {'D': D, 'Lm': Lm, 'rat': (rat_lo, rat_hi), 'corr_up': corr_up / Fref_lo, 'corr_lo': corr_lo / Fref_lo}


# ----------------------------------------------------------------------------------------------- d = 3
def grid_up(x, g=2 ** 40):
    return Fr(math.ceil(Fr(x) * g), g)


def grid_dn(x, g=2 ** 40):
    return Fr(math.floor(Fr(x) * g), g)


def zeta_ratio_d3(ctx, bb, kb):
    """Enclosure of [Z_r / (36 k^2 r^2)] / m_3(b) over the sub-box (m_3(b) = E[det(B)^2 1{B > 0}], B = b I + GOE_2)."""
    M, law, r1 = ctx.M, ctx.law, ctx.r1
    b0, b1 = bb
    klo, khi = kb
    b = M.IV(b0, b1)
    k = M.IV(klo, khi)
    mu = {n: b * law.alpha[n] + k * law.beta[n] for n in M.TARGET_NAMES}
    mabs = {n: M.absup(mu[n]) for n in M.TARGET_NAMES}
    dmu = [b * (-law.alpha['a11'] - 1) + k * (-law.beta['a11']), b * (-law.alpha['a22'] - 1) + k * (-law.beta['a22']),
           (b * law.alpha['a12'] + k * law.beta['a12']) * M.SQRT2]
    eps = M.sqrt_up(sum(M.absup(x) ** 2 for x in dmu))
    muB = M.rup_rel(b1 * M.sqrt_up(Fr(2)) + eps)
    sig = M.sqrt_up(law.sB_hi2)
    Lb = M.cholesky([[M.IV(x.lo, x.hi) for x in row] for row in law.Cb])
    eta = M.sqrt_up(sum(M.absup(Lb[i][j] - (M.SQRT2 if i == j else 0)) ** 2 for i in range(3) for j in range(i + 1)))
    R0 = Fr(R0C)
    Dl = M.rup_rel(eps + eta * R0)
    cm = M.contact_moment
    pR0 = M.chi_tail(3, R0)
    # main term, upper: m3(b + Dl) + outside-ball;  m3(b + Dl)/m3(b) <= exp(Dl L3)
    bA = grid_dn(b0 - Dl)
    bB = grid_up(b1 + Dl)
    m3_b0 = cm(2, 0, b0).lo
    L3up = 2 * cm(1, 1, bB).hi / m3_b0
    L3dn = 2 * cm(1, 1, b1).hi / cm(2, 0, bA).lo
    rat_hi = exp_minus(M, -(Dl * L3up)).hi
    rat_lo = exp_minus(M, Dl * L3dn).lo
    U = M.chi_upper_moments(3, R0, 4)
    out_D2 = sum((U[i] * (M.binom(4, i) * muB ** (4 - i) * sig ** i) for i in range(5)), M.IV(0)).hi / 4
    tail_c = M.sqrt_up(cm(4, 0, b1).hi) * M.sqrt_up(pR0)          # E[F(B') 1{||z|| > R0}], B' at b - Dl <= b1
    # Loewner-monotone moments at b'' = b1 + Dl, plus outside-ball parts (as in the engine)
    nBF16 = M.rup_rel(muB + sig * M.chi_norm(3, 16))
    out_ball = M.sqrt_up(pR0)

    def up_moment(a, c):
        crude = M.rup_rel(nBF16 ** (2 * a + c))
        return M.rup_rel(cm(a, c, bB).hi + 2 ** c * crude * out_ball)
    ED2, ED4, ED2T2 = up_moment(2, 0), up_moment(4, 0), up_moment(2, 2)
    DT, TDT, TTDT = up_moment(1, 1), up_moment(1, 2), up_moment(1, 3)
    gn = lambda n, p: M.gauss_norm(mabs[n], law.sdQ[n], p)
    six = 6 * klo

    def epsn(p):
        return (2 * r1 * gn('T', p) + r1 * r1 * gn('tau', p)) / six + r1 * gn('T', 2 * p) * (r1 * gn('T', 2 * p) + r1 * r1 * gn('tau', 2 * p)) / six ** 2
    mu_o = M.sqrt_up(mabs['o11'] ** 2 + mabs['o22'] ** 2 + 2 * mabs['o12'] ** 2)
    trOQ = sum(m * law.C[(n, n)].hi for n, m in zip(M.ONAMES, (1, 1, 2)))
    nOm = lambda p: M.frob_norm(mu_o, trOQ, p)
    mu_v = M.sqrt_up(mabs['v1'] ** 2 + mabs['v2'] ** 2)
    trVQ = law.C[('v1', 'v1')].hi + law.C[('v2', 'v2')].hi
    mu_nu = M.sqrt_up(mabs['nu1'] ** 2 + mabs['nu2'] ** 2)
    trNQ = law.C[('nu1', 'nu1')].hi + law.C[('nu2', 'nu2')].hi
    nv4, nnu4 = M.frob_norm(mu_v, trVQ, 4), M.frob_norm(mu_nu, trNQ, 4)
    mtr = M.absup(mu['a11'] + mu['a22'])
    sdtr = M.sqrt_up(law.Cb[0][0].hi) + M.sqrt_up(law.Cb[1][1].hi)
    ntr8 = M.gauss_norm(mtr, sdtr, 8)
    e2, e4 = epsn(2), epsn(4)
    corr_up = (e2 * M.sqrt_up(ED4) + r1 * (1 + e4) * M.sqrt_up(ED2T2) * nOm(4) + r1 * r1 * (1 + e4) * M.sqrt_up(ED2) * nOm(8) ** 2
               + r1 * (1 / six + r1 * gn('T', 8) / six ** 2) * M.iroot(ED4, 4) * (ntr8 + r1 * nOm(8)) * (r1 * nnu4 + nv4) ** 2)
    # lower corrections (engine section 6)
    sdT, sdtau = law.sdQ['T'], law.sdQ['tau']
    q_bad = M.gauss_tail2(mabs['T'], sdT, 3 * klo / (2 * r1)) + M.gauss_tail2(mabs['tau'], sdtau, 3 * klo / (2 * r1 * r1))
    nF = M.sqrt_up(ED4)
    corr_G0 = nF * M.sqrt_up(q_bad) if q_bad > 0 else Fr(0)
    corr_eps = (2 * r1 * M.gauss_norm(mabs['T'], sdT, 2) + r1 * r1 * M.gauss_norm(mabs['tau'], sdtau, 2)) / six * nF
    trO = sum(m * max(law.Cg[(n, n)].hi, Fr(0)) for n, m in zip(M.ONAMES, (1, 1, 2)))
    trV = law.Cg[('v1', 'v1')].hi + law.Cg[('v2', 'v2')].hi
    bn = law.bnorm
    bv = M.sqrt_up(bn['v1'] ** 2 + bn['v2'] ** 2)
    bo = M.sqrt_up(bn['o11'] ** 2 + bn['o22'] ** 2 + 2 * bn['o12'] ** 2)
    Eog = M.rup_rel(mu_o + M.sqrt_up(trO))
    corr_om = Eog * DT + bo * (TDT + muB * DT)
    Evg2 = mu_v ** 2 + trV
    corr_v = (Fr(17, 16) * Evg2 * DT + 17 * bv ** 2 * (TTDT + 2 * muB * TDT + muB * muB * DT)) / (4 * klo)
    corr_lo = tail_c + corr_G0 + corr_eps + r1 * (corr_om + corr_v)
    hi = rat_hi + (out_D2 + corr_up) / m3_b0
    lo = rat_lo - corr_lo / m3_b0
    return lo, hi, {'eps': eps, 'eta': eta, 'Dl': Dl, 'rat': (rat_lo, rat_hi), 'corr_up': (out_D2 + corr_up) / m3_b0,
                    'corr_lo': corr_lo / m3_b0}


def rho_box(ctx, bb, kb):
    M = ctx.M
    pr = pi_ratio(ctx, bb, kb)
    if ctx.d == 2:
        lo, hi, info = zeta_ratio_d2(ctx, bb, kb)
    else:
        lo, hi, info = zeta_ratio_d3(ctx, bb, kb)
    rho_hi = pr.hi * hi
    rho_lo = pr.lo * lo
    return rho_lo, rho_hi, pr, info
