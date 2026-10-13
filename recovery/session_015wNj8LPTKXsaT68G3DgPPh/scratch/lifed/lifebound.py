"""Certified bounds |A_r / A_0^ref - 1| <= a r^2 + eps on (b, k) sub-boxes, for 0 < r <= R (prototype)."""
import math
from fractions import Fraction as Fr

import lifecore as LC
import tmfx

E2, E3 = LC.E2, LC.E3
T_BAD = Fr(16)                 # threshold of the bad-set split
R0C = Fr(12)                   # Loewner coupling radius (d = 3)
TINY = Fr(1, 10 ** 60)         # polynomial coefficients below this are rounding residue


# ------------------------------------------------------------------------------------------- (a, eps) forms
def ae_add(*xs):
    return (sum(x[0] for x in xs), sum(x[1] for x in xs))


def ae_mul(x, c):
    return (x[0] * c, x[1] * c)


def ae_at(x, R):
    return x[0] * R * R + x[1]


def exp_up(M, x, R):
    """(a, eps) with exp(a0 r^2 + e0) - 1 <= a r^2 + eps on (0, R] (x = (a0, e0) >= 0): e^y - 1 <= y e^y."""
    f = LC.exp_minus(M, -ae_at(x, R)).hi
    return (x[0] * f, x[1] * f)


# ------------------------------------------------------------------------------------------- Taylor-model readers
def tm_dev(t, ref, R):
    """(a, eps): |t(r) - ref| <= a r^2 + eps for 0 <= r <= R."""
    c = t.frac_coeffs()
    eps = abs(c[0] - ref) + abs(c[1]) * R + t.frac_e()
    a, Rp = Fr(0), Fr(1)
    for x in c[2:]:
        a += abs(x) * Rp
        Rp *= R
    return (a, eps)


def tm_lin(t, R):
    """(s, eps): |t(r)| <= s r + eps for 0 <= r <= R."""
    c = t.frac_coeffs()
    eps = abs(c[0]) + t.frac_e()
    s, Rp = Fr(0), Fr(1)
    for x in c[1:]:
        s += abs(x) * Rp
        Rp *= R
    return (s, eps)


def tm_shift(t, ref, R):
    """(lo, hi, eps): t(r) - ref in [lo r^2 - eps, hi r^2 + eps] for 0 <= r <= R."""
    c = t.frac_coeffs()
    eps = abs(c[0] - ref) + abs(c[1]) * R + t.frac_e()
    lo = hi = c[2] if len(c) > 2 else Fr(0)
    Rp = R
    for x in c[3:]:
        v = x * Rp
        lo += min(v, 0)
        hi += max(v, 0)
        Rp *= R
    return (lo, hi, eps)


def assert_flat(t, ref):
    """The reference-kernel polynomial part is exactly ref (only the remainder carries the torus and the truncation)."""
    c = t.frac_coeffs()
    if abs(c[0] - ref) > TINY or any(abs(x) > TINY for x in c[1:]):
        raise ValueError('law of the transverse Hessian is not flat in r: %r' % [float(x) for x in c])


def gersh_min(C):
    """Gershgorin lower bound of lambda_min of a symmetric interval matrix."""
    n = len(C)
    return min(C[i][i].lo - sum(max(abs(C[i][j].lo), abs(C[i][j].hi)) for j in range(n) if j != i) for i in range(n))


def wq_bound(M, Cg, rows, vecs):
    """Upper bound of || V' Cg^-1 V || (V the rows `vecs`, given as lists of magnitudes) through the diagonal scaling
    Cg = D^(1/2) (I + E) D^(1/2): Cg^-1 <= D^-1 / (1 - ||E||_F)."""
    n = len(rows)
    D = [Cg[(rows[i], rows[i])].lo for i in range(n)]
    Ef2 = Fr(0)
    for i in range(n):
        for j in range(n):
            if i != j:
                c = max(abs(Cg[(rows[i], rows[j])].lo), abs(Cg[(rows[i], rows[j])].hi))
                Ef2 += c * c / (D[i] * D[j])
    Ef = M.sqrt_up(Ef2)
    if Ef >= Fr(1, 2):
        raise ValueError('diagonal scaling')
    return sum(sum(x * x for x in vecs[i]) / D[i] for i in range(n)) / (1 - Ef)


def half_moments(M, c):
    """H_j = int_0^inf u^j exp(-u^2/(2c)) du for j = 0..3 (rational upper bounds): sqrt(pi c/2), c, c sqrt(pi c/2), 2 c^2."""
    s = (M.isqrt_iv(M.PI * c / 2)).hi
    return [s, Fr(c), c * s, 2 * Fr(c) ** 2]


# ------------------------------------------------------------------------------------------- the pin-density ratio
def pi_ratio(M, tml, n, Q, R):
    """log(pi_r / pi_0^ref) in [-(dn), up] with dn, up in (a, eps) form."""
    b0, b1, k0, k1 = Q
    th1 = tml.th1
    dlo, dhi, de = tm_shift(tml.det, 12, R)
    xm = (max(abs(dlo), abs(dhi)) * R * R + de) / 12
    if xm >= Fr(1, 2):
        raise ValueError('det deviation')
    half_log_up = (max(Fr(0), -dlo) / (24 * (1 - xm)), de / (24 * (1 - xm)))
    half_log_dn = (max(Fr(0), dhi) / 24, de / 24)
    # Delta E = sum (coefficient deviations) + (Theta - 1) E_u;  b, k >= 0 on the box
    parts = ((tml.e_bb, Fr(3, 2), (b0 * b0, b1 * b1)), (tml.e_bk, Fr(0), (2 * b0 * k0, 2 * b1 * k1)),
             (tml.e_kk, Fr(24), (k0 * k0, k1 * k1)))
    Elo = Ehi = Ee = Fr(0)
    Eu = Fr(0)
    for t, ref, (wlo, whi) in parts:
        lo, hi, e = tm_shift(t, ref, R)
        Elo += min(wlo * lo, whi * lo)
        Ehi += max(wlo * hi, whi * hi)
        Ee += whi * e
        Eu += whi * (abs(ref) + max(abs(lo), abs(hi)) * R * R + e)
    up = ae_add(half_log_up, (max(Fr(0), -Elo / 2), Ee / 2 + n * th1 / 2))
    dn = ae_add(half_log_dn, (max(Fr(0), Ehi / 2), (Ee + th1 * Eu) / 2))
    return up, dn


# ------------------------------------------------------------------------------------------- d = 2
def F2(m, s):
    return E2.elam2_pos(Fr(m), Fr(s))


def F1(m, s):
    return E2.elam_pos(Fr(m), Fr(s))


def bounds_d2(Q, R, tml, il):
    M = E2
    b0, b1, k0, k1 = Q
    th1 = tml.th1
    assert_flat(tml.alpha['w'], -1)
    assert_flat(tml.beta['w'], 0)
    assert_flat(tml.Cu[('w', 'w')], 2)
    # ---- the law of lambda = -w: m_lambda = b + D, sigma^2 = 2 + S
    aA = tm_dev(tml.alpha['w'], -1, R)
    aB = tm_dev(tml.beta['w'], 0, R)
    Dae = ae_add(ae_mul(aA, b1), ae_mul(aB, k1))
    aC = tm_dev(tml.Cu[('w', 'w')], 2, R)
    Cmax = 2 + ae_at(aC, R)
    Sae = (aC[0], aC[1] + Cmax * th1)
    Dm, Sm = ae_at(Dae, R), ae_at(Sae, R)
    s2lo, s2hi = M.isqrt_iv(M.IV(2)).lo, M.isqrt_iv(M.IV(2)).hi
    s_lo = M.isqrt_iv(M.IV(2 - Sm)).lo
    s_hi = M.isqrt_iv(M.IV(2 + Sm)).hi
    m_lo, m_hi = b0 - Dm, b1 + Dm
    mmax = max(abs(m_lo), abs(m_hi))
    Fmin = F2(m_lo, min(s_lo, s2lo)).lo
    Fmax = F2(m_hi, max(s_hi, s2hi)).hi
    F1max = F1(m_hi, max(s_hi, s2hi)).hi
    Lm = 2 * F1max / Fmin
    Ls = 2 * max(s_hi, s2hi) / Fmin
    Fae = ae_add(ae_mul(Dae, Lm), ae_mul(Sae, Ls * Fr(3, 4)))      # |log F(m, s) - log F(b, sqrt2)|
    Fref = F2(b0, s2lo).lo                                          # <= F(b, sqrt2) on the box
    # ---- secondary law: the engine's interval law over [0, R]
    b, k = M.IV(b0, b1), M.IV(k0, k1)
    mu = {nm: b * il.alpha[nm] + k * il.beta[nm] for nm in M.TARGET_NAMES}
    ma = {nm: M.absup(mu[nm]) for nm in M.TARGET_NAMES}
    gn = lambda nm, p: M.gauss_norm(ma[nm], il.sdQ[nm].hi, p)
    lam = lambda p: M.gauss_norm(mmax, s_hi, p)
    T, tau, om, v, nu = (lambda p, x=x: gn(x, p) for x in ('T', 'tau', 'om', 'v', 'nu'))
    kap = lambda p: 6 * k1 * tau(p) + T(2 * p) ** 2 + R * T(2 * p) * tau(2 * p)
    norm36 = 36 * k0 * k0 * Fref
    # T2: r E[lambda_+ om] = r [mu_om F_1 - bw_om (F - m F_1)]
    sa, ea = tm_lin(tml.alpha['om'], R)
    sb, eb = tm_lin(tml.beta['om'], R)
    sc, ec = tm_lin(tml.Cu[('om', 'w')], R)
    cwlo = 2 - ae_at(aC, R)
    T2 = ((b1 * sa + k1 * sb) * F1max + sc / cwlo * (Fmax + mmax * F1max),
          R * ((b1 * ea + k1 * eb) * F1max + ec / cwlo * (Fmax + mmax * F1max)))
    T2 = (T2[0] / Fref, T2[1] / Fref)                                 # 36 k^2 cancels
    # T3 - T6 (all with an explicit r^2)
    T3 = kap(2) * lam(4) ** 2
    T4 = R * kap(4) * lam(4) * om(2)
    xi2 = 6 * k1 * (R * nu(4) ** 2 + 2 * nu(4) * v(4)) + T(4) * (R * R * nu(8) ** 2 + 2 * R * nu(8) * v(8) + 2 * v(8) ** 2) \
        + R * tau(4) * v(8) ** 2
    T5 = lam(2) * xi2
    c2n4 = 6 * k1 + R * T(4) + R * R * tau(4)
    T6 = v(8) ** 2 * c2n4 * om(4) + v(4) ** 2 * (R * nu(4) + v(4)) ** 2
    # bad set: {0 < lambda <= r h} on G_0, h <= 2 v^2/(9k) + |om|; given Y = (v, om, nu) the density of lambda is at
    # most 1/(sigma_c sqrt(2 pi)), 1/sigma_c^2 = 1/sigma_w^2 + bw_Y' Cg_Y^-1 bw_Y (Schur complement)
    Yn = ('v', 'om', 'nu')
    prec = 1 / (2 - Sm) + wq_bound(M, il.Cg, Yn, [[M.absup(il.bw[n])] for n in Yn])
    pbar = Fr(2, 5) * M.sqrt_up(prec)                               # 1/sqrt(2 pi) < 2/5
    h = lambda p: Fr(2) / (9 * k0) * v(2 * p) ** 2 + om(p)
    EGh = h(2) * (Fr(15, 2) * k1 * h(4) + v(8) ** 2) * (9 * k1 * (h(4) + om(4)) + (R * nu(8) + v(8)) ** 2)
    bad = R * pbar * EGh
    # G_0^c: ||P1 P2||_2 P(G_0^c)^(1/2), not r^2-scaled
    q = M.gauss_tail2(ma['T'], il.sdQ['T'].hi, 3 * k0 / (2 * R)) + M.gauss_tail2(ma['tau'], il.sdQ['tau'].hi, 3 * k0 / (2 * R * R))
    P1n = (6 * k1 + R * T(8)) * lam(8) + R * v(8) ** 2
    P2n = (6 * k1 + R * T(8) + R * R * tau(8)) * (lam(8) + R * om(8)) + R * (R * nu(8) + v(8)) ** 2
    g0c = P1n * P2n * M.sqrt_up(q) if q > 0 else Fr(0)
    X = ae_add(T2, ((T3 + T4 + T5 + T6 + bad) / norm36, g0c / norm36))
    z_up = ae_add(exp_up(M, Fae, R), X)
    z_dn = ae_add(Fae, X)
    p_up, p_dn = pi_ratio(M, tml, 6, Q, R)
    return combine(M, p_up, p_dn, z_up, z_dn, R), {'Fae': Fae, 'T2': T2, 'T3456bad': (T3 + T4 + T5 + T6 + bad) / norm36,
                                                     'bad': bad / norm36, 'g0c': g0c / norm36, 'pi': (p_up, p_dn)}


def combine(M, p_up, p_dn, z_up, z_dn, R):
    """rho = (pi ratio)(Z ratio): returns (up, dn) in (a, eps) form, rho - 1 <= up, 1 - rho <= dn."""
    pu = exp_up(M, p_up, R)                     # pi_r/pi_0 - 1 <= pu
    pd = p_dn                                    # 1 - pi_r/pi_0 <= pd (1 - e^-y <= y)
    zR = ae_at(z_up, R)
    up = (pu[0] + z_up[0] + pu[0] * zR, pu[1] + z_up[1] + pu[1] * zR)
    dn = ae_add(pd, z_dn)
    return up, dn


# ------------------------------------------------------------------------------------------- d = 3
def bounds_d3(Q, R, tml, il):
    M = E3
    b0, b1, k0, k1 = Q
    th1 = tml.th1
    FR = (Fr(1), Fr(1), None)
    for nm, ref in (('a11', -1), ('a22', -1), ('a12', 0)):
        assert_flat(tml.alpha[nm], ref)
        assert_flat(tml.beta[nm], 0)
    for (x, y), ref in ((('a11', 'a11'), 2), (('a22', 'a22'), 2), (('a12', 'a12'), 1), (('a11', 'a22'), 0),
                        (('a11', 'a12'), 0), (('a22', 'a12'), 0)):
        assert_flat(tml.Cu[(x, y)], ref)
    # ---- the transverse Hessian: eps = sup ||mu_B - b I||_F, eta = ||chol(Cb) - sqrt2 I|| (both remainder-only)
    comps = []
    for nm, ref, f in (('a11', -1, 1), ('a22', -1, 1), ('a12', 0, 2)):
        a1 = tm_dev(tml.alpha[nm], ref, R)
        a2 = tm_dev(tml.beta[nm], 0, R)
        comps.append(ae_at(ae_add(ae_mul(a1, b1), ae_mul(a2, k1)), R) * f)    # f = 2 > sqrt2 for the off-diagonal entry
    eps = M.sqrt_up(sum(x * x for x in comps))
    Fw = (Fr(1), Fr(1), M.SQRT2)
    names = ('a11', 'a22', 'a12')
    Cb = []
    for i in range(3):
        row = []
        for j in range(3):
            t = tml.Cu[(names[i], names[j])]
            lo, hi = t.enclose(0, R)
            iv = M.IV(lo, hi) * Fw[i] * Fw[j] * M.IV(1 / (1 + th1), 1)
            row.append(iv)
        Cb.append(row)
    Lb = M.cholesky(Cb)
    eta = M.sqrt_up(sum(M.absup(Lb[i][j] - (M.SQRT2 if i == j else 0)) ** 2 for i in range(3) for j in range(i + 1)))
    Dl = M.rup_rel(eps + eta * R0C)
    cm = M.contact_moment
    bA = LC.grid_dn(b0 - Dl)
    bB = LC.grid_up(b1 + Dl)
    m3ref = cm(2, 0, b0).lo                                          # <= m_3(b) on the box
    L3up = 2 * cm(1, 1, bB).hi / m3ref
    L3dn = 2 * cm(1, 1, b1).hi / cm(2, 0, bA).lo
    pR0 = M.chi_tail(3, R0C)
    muB = M.rup_rel(b1 * M.sqrt_up(Fr(2)) + eps)
    sig = M.sqrt_up(max(x.hi for x in (Cb[0][0], Cb[1][1], Cb[2][2])) + 2 * max(M.absup(Cb[i][j]) for i in range(3) for j in range(3) if i != j))
    U = M.chi_upper_moments(3, R0C, 4)
    out_D2 = sum((U[i] * (M.binom(4, i) * muB ** (4 - i) * sig ** i) for i in range(5)), M.IV(0)).hi / 4
    tail_c = M.sqrt_up(cm(4, 0, b1).hi) * M.sqrt_up(pR0)
    main_up = (Fr(0), Dl * L3up)                                     # log(m3(b + Dl)/m3(b)) <= Dl L3up
    main_dn = (Fr(0), Dl * L3dn)
    nBF16 = M.rup_rel(muB + sig * M.chi_norm(3, 16))
    out_ball = M.sqrt_up(pR0)

    def up_moment(a, c):
        crude = M.rup_rel(nBF16 ** (2 * a + c))
        return M.rup_rel(cm(a, c, bB).hi + 2 ** c * crude * out_ball)
    ED2, ED4, ED2T2, ET4, DT = up_moment(2, 0), up_moment(4, 0), up_moment(2, 2), up_moment(0, 4), up_moment(1, 1)
    trCb = sum(x.hi for x in (Cb[0][0], Cb[1][1], Cb[2][2]))
    BF = lambda p: M.frob_norm(muB, trCb, p)
    # ---- secondary law (engine interval law over [0, R])
    b, k = M.IV(b0, b1), M.IV(k0, k1)
    mu = {nm: b * il.alpha[nm] + k * il.beta[nm] for nm in M.TARGET_NAMES}
    ma = {nm: M.absup(mu[nm]) for nm in M.TARGET_NAMES}
    gn = lambda nm, p: M.gauss_norm(ma[nm], il.sdQ[nm], p)
    T = lambda p: gn('T', p)
    tau = lambda p: gn('tau', p)
    mu_o = M.sqrt_up(ma['o11'] ** 2 + ma['o22'] ** 2 + 2 * ma['o12'] ** 2)
    trO = sum(m * il.C[(nm, nm)].hi for nm, m in zip(M.ONAMES, (1, 1, 2)))
    Om = lambda p: M.frob_norm(mu_o, trO, p)
    mu_v = M.sqrt_up(ma['v1'] ** 2 + ma['v2'] ** 2)
    trV = il.C[('v1', 'v1')].hi + il.C[('v2', 'v2')].hi
    v = lambda p: M.frob_norm(mu_v, trV, p)
    mu_nu = M.sqrt_up(ma['nu1'] ** 2 + ma['nu2'] ** 2)
    trN = il.C[('nu1', 'nu1')].hi + il.C[('nu2', 'nu2')].hi
    nu = lambda p: M.frob_norm(mu_nu, trN, p)
    e = lambda p: R * nu(p) + v(p)
    kap = lambda p: 6 * k1 * tau(p) + T(2 * p) ** 2 + R * T(2 * p) * tau(2 * p)
    c1 = lambda p: 6 * k1 + R * T(p)
    c2 = lambda p: 6 * k1 + R * T(p) + R * R * tau(p)
    m3b0 = m3ref
    norm36 = 36 * k0 * k0 * m3b0
    # U3 main: 36 k^2 r [ ||mu_Omega||_F DT + bo sqrt(E D^2 tr^2) sqrt(tr Cb) ]  (relative: divide by 36 k^2 m3)
    s_mu2 = e_mu2 = Fr(0)
    for nm, mult in zip(M.ONAMES, (1, 1, 2)):
        sa, ea = tm_lin(tml.alpha[nm], R)
        sb, eb = tm_lin(tml.beta[nm], R)
        s_mu2 += mult * (b1 * sa + k1 * sb) ** 2
        e_mu2 += mult * (b1 * ea + k1 * eb) ** 2
    s_mu, e_mu = M.sqrt_up(s_mu2), M.sqrt_up(e_mu2)
    s_c2 = e_c2 = Fr(0)
    for nm, mult in zip(M.ONAMES, (1, 1, 2)):
        for an, f2 in zip(names, (1, 1, 2)):
            s, ee = tm_lin(tml.Cu[(nm, an)], R)
            s_c2 += mult * f2 * s * s
            e_c2 += mult * f2 * ee * ee
    lmin = Fr(2) - M.sqrt_up(sum(M.absup(Cb[i][j] - (2 if i == j else 0)) ** 2 for i in range(3) for j in range(3)))
    bo_s, bo_e = M.sqrt_up(s_c2) / lmin, M.sqrt_up(e_c2) / lmin
    U3m = ((s_mu * DT + bo_s * M.sqrt_up(ED2T2) * M.sqrt_up(trCb)) / m3b0,
           R * (e_mu * DT + bo_e * M.sqrt_up(ED2T2) * M.sqrt_up(trCb)) / m3b0)
    # second-order terms (explicit r^2)
    U2 = kap(2) * M.sqrt_up(ED4)
    U3r = R * kap(4) * M.sqrt_up(ED2T2) * Om(4)
    U4 = (36 * k1 * k1 + R * R * kap(4)) * M.sqrt_up(ED2) * Om(8) ** 2 / 2
    U5 = M.sqrt_up(ED2T2) * ((2 * T(4) + R * tau(4)) * v(8) ** 2 + 2 * c1(4) * nu(8) * v(8) + R * c1(4) * nu(8) ** 2)
    U6 = c1(8) * M.sqrt_up(ED2) * Om(8) * e(8) ** 2
    U7 = c2(8) * M.sqrt_up(ET4) * Om(8) * v(8) ** 2
    U8 = R * c2(8) * Om(16) ** 2 / 2 * M.iroot(ET4, 4) * v(16) ** 2
    U9 = M.sqrt_up(ET4) * v(8) ** 2 * e(8) ** 2
    U10 = R * M.iroot(ET4, 4) * v(8) ** 2 * Om(4) * e(8) ** 2
    # bad set {0 < lambda_1 <= r h} on G_0, h <= 2 ||v||^2/(9k) + Omega_F: given Y = (v, Omega', nu), B is Gaussian with
    # covariance C' (lambda_min(C') >= cl by the Schur complement, lambda_max <= c_hi) and mean mu'; in eigenvalue coordinates
    # E[lambda_2 (lambda_2 + r Omega) 1{0 < lambda_1 <= s} | Y] <= s KB' J(||mu'||, r Omega)
    Yn = tuple(M.VNAMES) + tuple(M.ONAMES) + ('nu1', 'nu2')
    lmB = gersh_min(Cb)
    rows = []
    for nm in Yn:
        if nm in M.ONAMES:
            cov = [ae_at((Fr(0), tm_lin(tml.Cu[(nm, an)], R)[0] * R + tm_lin(tml.Cu[(nm, an)], R)[1]), R) * f
                   for an, f in zip(names, (1, 1, 2))]                       # |Cov(o, b~_m)| (f = 2 > sqrt2)
            n2 = M.sqrt_up(sum(x * x for x in cov))
            rows.append([n2 / lmB])                                          # ||Bt_o|| <= ||Cov(o, b~)|| / lambda_min(Cb)
        else:
            rows.append([M.absup(x) for x in il.bt[nm]])
    Wt = wq_bound(M, il.Cg, Yn, rows)                                        # || Bt' Cg^-1 Bt ||
    c_hi = sig * sig
    cl = 1 / (1 / lmB + Wt)                                                  # lambda_min(C_B|Y)
    KBp = M.rup_rel((M.PI * M.SQRT2 / (2 * M.PI * M.SQRT2PI)).hi / (cl * M.sqrt_down(cl)))
    Kn = c_hi * M.sqrt_up(Wt)                                                # ||C_BY C_YY^-1/2||
    # ||mu'|| <= ||mu_B|| + kappa_Y, kappa_Y = ||C_BY C_YY^-1|| ||Y - mu_Y||; J is a polynomial in ||mu'|| with nonnegative
    # coefficients, so ||J_a(||mu'||)||_2 <= J_a(||mu_B|| + ||kappa_Y||_8) (Minkowski on each power, degree <= 4)
    m1 = muB + Kn * M.chi_norm(7, 8)                                         # ||mu' - mu_B|| <= Kn chi_7
    Hs = half_moments(M, c_hi)
    Ja = m1 ** 4 / 4 + sum(M.binom(3, i) * m1 ** (3 - i) * Hs[i] for i in range(4))
    Jb = m1 ** 3 / 3 + sum(M.binom(2, i) * m1 ** (2 - i) * Hs[i] for i in range(3))
    h = lambda p: Fr(2) / (9 * k0) * v(2 * p) ** 2 + Om(p)
    nhg2 = h(4) * (Fr(15, 2) * k1 * h(8) + v(16) ** 2) * (9 * k1 * (h(8) + Om(8)) + e(16) ** 2)
    nhgO2 = h(8) * (Fr(15, 2) * k1 * h(16) + v(32) ** 2) * (9 * k1 * (h(16) + Om(16)) + e(32) ** 2) * Om(8)
    bad = R * KBp * (Ja * nhg2 + R * Jb * nhgO2)
    q = M.gauss_tail2(ma['T'], il.sdQ['T'], 3 * k0 / (2 * R)) + M.gauss_tail2(ma['tau'], il.sdQ['tau'], 3 * k0 / (2 * R * R))
    P1n = c1(8) * BF(16) ** 2 + R * BF(8) * v(16) ** 2
    P2n = c2(8) * (BF(16) + R * Om(16)) ** 2 + R * (BF(8) + R * Om(8)) * e(16) ** 2
    g0c = P1n * P2n * M.sqrt_up(q) if q > 0 else Fr(0)
    sec = (U2 + U3r + U4 + U5 + U6 + U7 + U8 + U9 + U10 + bad) / norm36
    X = ae_add(U3m, (sec, g0c / norm36))
    z_up = ae_add(exp_up(M, main_up, R), (Fr(0), out_D2 / m3b0), X)
    z_dn = ae_add(main_dn, (Fr(0), tail_c / m3b0), X)
    p_up, p_dn = pi_ratio(M, tml, 8, Q, R)
    return combine(M, p_up, p_dn, z_up, z_dn, R), {'Dl': Dl, 'U3m': U3m, 'sec': sec, 'bad': bad / norm36,
                                                     'g0c': g0c / norm36, 'pi': (p_up, p_dn),
                                                     'terms': [x / norm36 for x in (U2, U3r, U4, U5, U6, U7, U8, U9, U10)]}
