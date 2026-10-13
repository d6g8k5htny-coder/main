

# ----------------------------------------------------------------------------- moment calculus (normalized by K = 12 k)
def binom(n, k):
    return math.comb(n, k)


def pos_part_moments_chi(s, c, nu, nmax, extra_max):
    """P[(n, e)] = upper bound of E[u^e (u^n - c^n)_+], u = s chi_nu, for 1 <= n <= nmax, 0 <= e <= extra_max."""
    x = c / s
    M = chi_upper_moments(nu, x, nmax + extra_max)
    out = {}
    for n in range(1, nmax + 1):
        for e in range(extra_max + 1):
            out[(n, e)] = rup_rel((M[n + e] * s ** (n + e) - M[e] * c ** n * s ** e).hi)
    return out


NU = {2: 2, 3: 3, 4: 4}                      # chi degrees of freedom of the free blocks v (2), omega (3), Y~ (4)


def normalized_moments(q, nmax, theta):
    """Component moments, normalized by K = 12 k (K >= Klo), for n <= nmax, under the product of the g-law and nu:
      E1[n], E1o[n], E1v[n] : upper bounds of E[y1^n], E[o0 y1^n], E[v0^2 y1^n], y1 = 1 + cmt1 + (r1/Klo) D1;
      Tn[n], To[n], Tv[n]   : upper bounds of sum_{i=2,3,4} E[(y_i^n - c^n)_+] (and with the factors o0, v0^2),
                              y_i <= u_i + delta_i (u_i = s_i chi_nu_i independent), threshold c = q['c'].
    o0 = Klo u_3 and v0^2 = Klo^2 u_2^2 / 4 are unnormalized."""
    th = theta
    cp = q['c'] * (1 - th)                  # <= c (1 + th)^{-(n-1)/n} for every n (Bernoulli)
    PP = {i: pos_part_moments_chi(q['s'][i], cp, NU[i], nmax, 2) for i in (2, 3, 4)}
    Eo0 = (chi_moment(3, 1) * q['s3']).hi
    Ev0sq = (chi_moment(2, 2) * q['s2'] * q['s2']).hi / 4
    n_o0_2 = q['s3'] * chi_norm(3, 2)
    n_v0sq_2 = q['s2'] * q['s2'] * chi_norm(2, 4) ** 2 / 4
    Klo = q['Klo']
    E1, E1o, E1v, Tn, To, Tv = {}, {}, {}, {}, {}, {}
    base = rup_rel(1 + q['cmt1'])
    bp = [Fr(1)]
    for _ in range(nmax):
        bp.append(rup_rel(bp[-1] * base))
    # (r/K) D1 terms: ||(rK D1)^j||_1 <= (rK ||D1||_pow2(j))^j and ||(rK D1)^j||_2 <= (rK ||D1||_pow2(2j))^j
    d1 = [Fr(1)] + [rup_rel((q['rK'] * q['nD1'](pow2(j))) ** j) for j in range(1, nmax + 1)]
    d2 = [Fr(1)] + [rup_rel((q['rK'] * q['nD1'](pow2(2 * j))) ** j) for j in range(1, nmax + 1)]
    for n in range(0, nmax + 1):
        ey = bp[n]
        eo = bp[n] * Eo0
        ev = bp[n] * Ev0sq
        for j in range(1, n + 1):
            coef = binom(n, j) * bp[n - j]
            ey += coef * d1[j]
            eo += coef * n_o0_2 * d2[j]
            ev += coef * n_v0sq_2 * d2[j]
        E1[n], E1o[n], E1v[n] = rup_rel(ey), rup_rel(eo), rup_rel(ev)
        if n == 0:
            continue
        f1 = (1 + th) ** (n - 1)
        f2 = (1 + 1 / th) ** (n - 1)
        tn = to = tv = Fr(0)
        for i in (2, 3, 4):
            P = PP[i]
            dn = rup_rel(q['nd'](i, pow2(n)) ** n)
            dn2 = rup_rel(q['nd'](i, pow2(2 * n)) ** n)
            tn += f1 * P[(n, 0)] + f2 * dn
            to += (f1 * Klo * P[(n, 1)] if i == 3 else f1 * Eo0 * P[(n, 0)]) + f2 * n_o0_2 * dn2
            tv += (f1 * Klo * Klo / 4 * P[(n, 2)] if i == 2 else f1 * Ev0sq * P[(n, 0)]) + f2 * n_v0sq_2 * dn2
        Tn[n], To[n], Tv[n] = rup_rel(tn), rup_rel(to), rup_rel(tv)
    return E1, E1o, E1v, Tn, To, Tv


def half_gauss_moments(m, sig, J):
    """[J_j(m, sig) for j = 0..J], J_j = integral_0^inf l^j exp(-(l - m)^2/(2 sig^2)) dl, rational m >= 0, sig > 0 (intervals):
    J_0 = sig sqrt(2 pi) Phi(m/sig), J_1 = m J_0 + sig^2 exp(-m^2/(2 sig^2)), J_(j+1) = m J_j + j sig^2 J_(j-1)
    (integration by parts; every term is positive for m >= 0)."""
    m, sig = Fr(m), Fr(sig)
    require(m >= 0 and sig > 0, 'half-line Gaussian moments')
    s2 = sig * sig
    out = [SQRT2PI * sig * (1 - Phibar_pt(m / sig))]
    out.append(out[0] * m + exp_neg_point(m * m / (2 * s2)) * s2)
    for j in range(1, J):
        nxt = out[j] * m + out[j - 1] * (j * s2)
        out.append(IV(rdown_rel(nxt.lo), rup_rel(nxt.hi)))
    return out


def lam2_moments(b_lo, b_hi, eps, sig):
    """Bounds on the lambda_2-integrals of the density majorant g(l) = exp(-(l - b)^2/(2 sig^2) + eps |l - b|/sig^2),
    uniformly over b in [b_lo, b_hi] (b >= 0): I_hi[j] >= integral_0^inf l^j g(l) dl (j = 0..JM), I3_lo <= integral l^3 g.
    For l >= 0, eps |l - b| <= eps (l + b), so g(l) <= exp((4 b eps + eps^2)/(2 sig^2)) exp(-(l - b - eps)^2/(2 sig^2));
    each J_j(m, sig) increases with m; and g(l) >= exp(-(l - b)^2/(2 sig^2))."""
    JM = 3 + 64
    shift = (4 * b_hi * eps + eps * eps) / (2 * sig * sig)
    fac = exp_neg_point(shift).inv().hi
    Jh = half_gauss_moments(b_hi + eps, sig, JM)
    I_hi = [rup_rel(x.hi * fac) for x in Jh]
    I3_lo = rdown_rel(half_gauss_moments(b_lo, sig, 3)[3].lo)
    return I_hi, I3_lo


DETN = {}


def det_norm(p, bpp, muB, sig):
    """Upper bound of ||det(B) 1{B > 0}||_p (p a power of two) for B = mu_B + Gaussian with Loewner coupling
    B <= (b'' + eta ||z||) I + GOE (b'' rounded up to a 1/1024 grid): the contact moment E[det^p 1{B > 0}] at b'' on
    {||z|| <= R0}, plus E[(||B||_F^2/2)^p 1{||z|| > R0}] <= 2^-p E[(||mu_B|| + sig chi_3)^(2p) 1{chi_3 > R0}]."""
    bg = Fr(math.ceil(bpp * 1024), 1024)
    key = (p, bg)
    if key not in DETN:
        DETN[key] = contact_moment(p, 0, bg).hi
    cm = DETN[key]
    R0 = Fr(R0_COUPLING)
    U = chi_upper_moments(3, R0, 2 * p)
    out = sum((U[i] * (binom(2 * p, i) * muB ** (2 * p - i) * sig ** i) for i in range(2 * p + 1)), IV(0)).hi / 2 ** p
    return iroot(rup_rel(cm + out), p)


def box_bound(law, bb, kb, r0, r1, last_band=False, x0=None, theta=None):
    """Certified upper bound of sup_{r in [r0, r1]} Q^W(G_r^c) / r^3 over b in bb, k in kb, with its pieces, for the near-branch
    threshold x0 (X_0 of NOTE section 4) and the split parameter theta (section 5); every choice gives a valid bound."""
    x0 = X0 if x0 is None else Fr(x0)
    theta = THETA if theta is None else Fr(theta)
    b = IV(bb[0], bb[1])
    k = IV(kb[0], kb[1])
    blo, bhi = bb
    klo, khi = kb[0], kb[1]
    require(blo >= 0, 'b >= 0 assumed by the density majorant')
    Klo = 12 * klo
    mu = {n: b * law.alpha[n] + k * law.beta[n] for n in TARGET_NAMES}
    mabs = {n: absup(mu[n]) for n in TARGET_NAMES}
    E = b * b * law.e_bb + b * k * law.e_bk * 2 + k * k * law.e_kk
    require(E.hi > 0, 'energy')
    sqE = sqrt_up(E.hi)
    nl2 = MUT != 'no-lambda2'
    # ---- the transverse Hessian B = -A: mean in Frobenius coordinates, its distance to b I, the density majorant
    dmu = [b * (-law.alpha['a11'] - 1) + k * (-law.beta['a11']), b * (-law.alpha['a22'] - 1) + k * (-law.beta['a22']),
           (b * law.alpha['a12'] + k * law.beta['a12']) * SQRT2]
    eps = sqrt_up(sum(absup(x) ** 2 for x in dmu))                 # sup over the box of ||mu_B - b I||_F
    muB = rup_rel(bhi * sqrt_up(Fr(2)) + eps)                       # sup ||mu_B||_F
    sig = sqrt_up(law.sB_hi2)                                       # rational, >= sqrt(lambda_max(Cov b~))
    sB_lo = law.sB_lo                                               # <= sqrt(lambda_min(Cov b~))
    KB = rup_rel((PI * SQRT2 / (2 * PI * SQRT2PI) / isqrt_iv(law.detCb)).hi)   # pi sqrt2 (2 pi)^(-3/2) det^(-1/2)
    s2g = sig * sig
    ex0 = blo * blo / (2 * s2g) - eps * bhi / s2g          # g(0; b) <= exp(-ex0) over the box
    g0 = exp_neg_point(ex0).hi if ex0 >= 0 else exp_neg_point(-ex0).inv().hi
    gbar = exp_neg_point(eps * eps / (2 * s2g)).inv().hi
    # slope of the linear density bound g(l) <= g0 + l Lg on l >= 0: the global |g'| <= exp(-1/2) gbar / sig, or the chord
    # of min(gbar, g0 exp(c l)) with c = (b_hi + eps)/sig^2 (g(l; b) <= g(0; b) exp((b + eps) l / sig^2) for l >= 0):
    # g0 c (x - 1)/log x with log x = log(gbar/g0) = y (rational), (x - 1)/log x increasing in x
    Lg = rup_rel(exp_neg_point(Fr(1, 2)).hi * gbar / sig)
    y = (eps * eps + blo * blo - 2 * eps * bhi) / (2 * s2g)
    if y > 0:
        cc = (bhi + eps) / s2g
        Lg = min(Lg, rup_rel(g0 * cc * (exp_neg_point(y).inv().hi - 1) / y))
    else:
        Lg = Fr(0)                                    # gbar <= g0: the bound g <= gbar alone is the smaller one
    I_hi, I3_lo = lam2_moments(blo, bhi, eps, sig)
    require(I3_lo > 0, 'lambda_2 moment')

    def l2norm(p):                                  # ||lambda_2||_p under nu (and under nu'), p a power of two
        if not nl2:
            return Fr(0)
        return iroot(rup_rel(I_hi[3 + p] / I3_lo), p)
    # ---- regression on B and the coefficients bar-beta_i (NOTE section 4)
    bn = law.bnorm
    bv = sqrt_up(bn['v1'] ** 2 + bn['v2'] ** 2)
    bo = sqrt_up(bn['o11'] ** 2 + bn['o22'] ** 2 + 2 * bn['o12'] ** 2)
    by = sqrt_up(bn['y111'] ** 2 + 3 * bn['y112'] ** 2 + 3 * bn['y122'] ** 2 + bn['y222'] ** 2)
    bnu = sqrt_up(bn['nu1'] ** 2 + bn['nu2'] ** 2)
    a_fac = Fr(1, 2) if MUT == 'cap-half' else Fr(1)      # the cap constant 4/(3k) (mutant: halved)

    def Bblk(blk):
        return rup_rel(sqrt_up(sum(m * bn[n] ** 2 for n, m in block_members(*blk))) + law.S[blk] / sB_lo)
    Bb = {blk: Bblk(blk) for blk in BLOCKS}
    bb1 = rup_rel(Fr(5, 2) * Bb[(4, 0)] + 2 * Bb[(3, 1)])           # bar-beta_1 / r
    bbars = {1: rup_rel(r1 * bb1),
             2: rup_rel(2 * bv + r1 * (Fr(5, 2) * Bb[(3, 1)] + 2 * Bb[(2, 2)])),
             3: rup_rel(bo + r1 * (Fr(5, 2) * Bb[(2, 2)] + 2 * Bb[(1, 3)])),
             4: rup_rel(by + r1 * (2 * Bb[(1, 3)] + 2 * Bb[(0, 4)]))}
    bbar = max(bbars.values())
    # ---- free blocks: Cholesky factors in Frobenius coordinates (v | omega | Y~)
    Lf = law.Lfree
    Iv, Io, Iy = (0, 1), (2, 3, 4), (5, 6, 7, 8)
    Lvv, Low, Loo = opnorm_lower_tri(Lf, Iv, Iv), opnorm_lower_tri(Lf, Io, Iv), opnorm_lower_tri(Lf, Io, Io)
    Lyv, Lyo, Lyy = opnorm_lower_tri(Lf, Iy, Iv), opnorm_lower_tri(Lf, Iy, Io), opnorm_lower_tri(Lf, Iy, Iy)
    mu_v = sqrt_up(mabs['v1'] ** 2 + mabs['v2'] ** 2)
    mu_o = sqrt_up(mabs['o11'] ** 2 + mabs['o22'] ** 2 + 2 * mabs['o12'] ** 2)
    mu_y = sqrt_up(mabs['y111'] ** 2 + 3 * mabs['y112'] ** 2 + 3 * mabs['y122'] ** 2 + mabs['y222'] ** 2)
    mu_nu = sqrt_up(mabs['nu1'] ** 2 + mabs['nu2'] ** 2)
    s2, s3, s4 = 2 * Lvv, Loo, Lyy

    memo = {}

    def blk_g_norm(blk, p):                         # || ||X_blk,g||_F ||_p
        mem = block_members(*blk)
        m_ = sqrt_up(sum(m * mabs[n] ** 2 for n, m in mem))
        tr = sum(m * max(law.Cg[(n, n)].hi, Fr(0)) for n, m in mem)
        return frob_norm(m_, tr, p)

    def nA(blk, p):                                 # ||A^g_blk||_p, A^g = ||X_blk,g||_F + T^g_blk
        key = ('A', blk, p)
        if key not in memo:
            memo[key] = rup_rel(blk_g_norm(blk, p) + law.S[blk] * (sqE + znorm1(pow2(p))))
        return memo[key]

    def nDelta(i, p):
        key = ('D', i, p)
        if key not in memo:
            memo[key] = rup_rel(nDelta0(i, p))
        return memo[key]

    def nDelta0(i, p):
        if MUT == 'no-delta':
            return Fr(0)
        if i == 1:
            return Fr(5, 2) * nA((4, 0), p) + 2 * nA((3, 1), p)
        if i == 2:
            return Fr(5, 2) * nA((3, 1), p) + 2 * nA((2, 2), p)
        if i == 3:
            return Fr(5, 2) * nA((2, 2), p) + 2 * nA((1, 3), p)
        return 2 * nA((1, 3), p) + 2 * nA((0, 4), p)

    def nd(i, p, lam=True):
        key = ('d', i, p, lam)
        if key not in memo:
            memo[key] = rup_rel(nd0(i, p, lam))
        return memo[key]

    def nd0(i, p, lam):     # ||delta_i||_p / Klo, delta_i including bar-beta_i (||mu_B|| + lambda_2)
        cm = bbars[i] * (muB + (l2norm(p) if lam else 0))
        if i == 2:
            v = 2 * mu_v + cm + r1 * nDelta(2, p)
        elif i == 3:
            v = mu_o + Low * chi_norm(2, pow2(p)) + cm + r1 * nDelta(3, p)
        else:
            v = mu_y + Lyv * chi_norm(2, pow2(p)) + Lyo * chi_norm(3, pow2(p)) + cm + r1 * nDelta(4, p)
        return v / Klo

    kappa = 3 / (1 - 3 * x0)
    Fk = 1 + kappa * x0                      # (1 + kappa x_i) <= Fk on the event {x_i <= x0}
    cth = 1 - kappa * x0 / 2                 # <= Fk^{-1/2}
    NMAX = 24
    q = {'cmt1': bbars[1] * muB / Klo, 'rK': r1 / Klo,
         'nD1': lambda p: rup_rel(nDelta(1, p) + bb1 * l2norm(pow2(p))),
         's': {2: s2 / Klo, 3: s3 / Klo, 4: s4 / Klo}, 'nd': nd, 's3': s3, 's2': s2, 'c': cth, 'Klo': Klo}
    E1, E1o, E1v, Tn, To, Tv = normalized_moments(q, NMAX, theta)
    et1 = rup_rel(16 * kappa * r1 * bbars[1] * a_fac)     # eta_1 K, k-free
    aK2_hi = 192 * khi * a_fac                    # a K^2 = 192 k

    def EUt(p):                                   # upper bound of E[Ut^p], Ut = U/(a K^2) <= max(Z1, Fk y_i^2)
        key = ('U', p)
        if key not in memo:
            memo[key] = rup_rel(sum(binom(p, l) * et1 ** l * E1[2 * p + l] for l in range(p + 1)) + Fk ** p * Tn[2 * p])
        return memo[key]

    def EoUt(p):
        return sum(binom(p, l) * et1 ** l * E1o[2 * p + l] for l in range(p + 1)) + Fk ** p * To[2 * p]

    def EvUt(p):
        return sum(binom(p, l) * et1 ** l * E1v[2 * p + l] for l in range(p + 1)) + Fk ** p * Tv[2 * p]

    def sq(x):
        return sqrt_up(x)

    def nUt(m, p):                                # ||Ut^m||_p <= Fk^m ||max_i y_i||_2mp^2m (on the event x_i <= X0)
        require(2 * m * p <= NMAX and pow2(p) == p, 'norm order')
        key = ('nUt', m, p)
        if key not in memo:
            memo[key] = rup_rel(Fk ** m * iroot(E1[2 * m * p] + Tn[2 * m * p], p))
        return memo[key]

    def nU(p):
        return rup_rel(aK2_hi * nUt(1, p))

    def ndel(p):                                   # || ||B - mu_B||_F ||_p <= ||mu_B|| + lambda_1 + lambda_2 <= ||mu_B|| + 2 lambda_2
        return rup_rel(muB + 2 * l2norm(p))

    def nbar(n, p):                                # ||F_bar||_p, F_bar = |F_g| + ||beta_F|| ||B - mu_B||_F
        return rup_rel(gauss_norm(mabs[n], law.sdg[n], p) + bn[n] * ndel(p))

    six_klo = 6 * klo
    nT2, nT4, nt2, nt4 = nbar('T', 2), nbar('T', 4), nbar('tau', 2), nbar('tau', 4)
    epsc2 = (2 * r1 * nT2 + r1 * r1 * nt2) / six_klo + (r1 * nT4) * (r1 * nT4 + r1 * r1 * nt4) / six_klo ** 2
    nT8, nt8 = nbar('T', 8), nbar('tau', 8)
    epsc4 = (2 * r1 * nT4 + r1 * r1 * nt4) / six_klo + (r1 * nT8) * (r1 * nT8 + r1 * r1 * nt8) / six_klo ** 2
    U2_2, U3_2, U4_2 = sq(EUt(4)), sq(EUt(6)), sq(EUt(8))
    U2_4, U3_4 = nUt(2, 4), nUt(3, 4)
    # the Omega majorant: Omega_bar = o0 + d_o, o0 = s3 chi_3 (shares u_3), d_o = rest
    ndo = lambda p: rup_rel(mu_o + Low * chi_norm(2, pow2(p)) + bo * ndel(p))
    no4 = rup_rel(s3 * chi_norm(3, 4) + ndo(4))
    # the ||r nu - v||^2 majorant (Term C)
    th = THETA_E
    dv = lambda p: rup_rel(mu_v + bv * ndel(p))
    nnu = lambda p: rup_rel(frob_norm(mu_nu, law.Cg[('nu1', 'nu1')].hi + law.Cg[('nu2', 'nu2')].hi, p)
                            + bnu * ndel(p))
    ebar4 = rup_rel((r1 * nnu(8) + Lvv * chi_norm(2, 8) + dv(8)) ** 2)        # ||e_bar||_4
    ebar8 = rup_rel((r1 * nnu(16) + Lvv * chi_norm(2, 16) + dv(16)) ** 2)     # ||e_bar||_8
    cC = (1 + th) * (1 + 1 / th) * dv(4) ** 2 + (1 + 1 / th) * r1 * r1 * nnu(4) ** 2
    # ---- Term A, B, C (one power of U per lambda_1 integration), normalized by 36 k^2 r^5 and by the lambda_2 weight
    termA = aK2_hi ** 3 / 3 * (EUt(3) + epsc2 * U3_2)
    termB = aK2_hi ** 2 / 2 * (EoUt(2) + (ndo(2) + epsc4 * no4) * U2_2)
    termC = aK2_hi ** 2 / (12 * klo) * ((1 + th) ** 2 * EvUt(2) + cC * U2_2 + r1 * nT4 / six_klo * ebar4 * U2_2)
    termAp = aK2_hi ** 4 / 4 * (EUt(4) + epsc2 * U4_2)
    termBp = aK2_hi ** 3 / 3 * (EoUt(3) + (ndo(2) + epsc4 * no4) * U3_2)
    termCp = aK2_hi ** 3 / (18 * klo) * ((1 + th) ** 2 * EvUt(3) + cC * U3_2 + r1 * nT4 / six_klo * ebar4 * U3_2)
    if MUT == 'no-om-term':
        termB = termBp = Fr(0)
    # ---- the Jacobian-correction terms: the part r Omega_bar of (lambda_2 + r Omega_bar), integrated against nu'
    # D  = E[Omega_bar (A + B + C integrands)],  D' = E[Omega_bar (A' + B' + C' integrands)]
    nT16 = nbar('T', 16)
    termD = (aK2_hi ** 3 / 3 * (EoUt(3) + (ndo(2) + epsc4 * no4) * U3_2)
             + aK2_hi ** 2 / 2 * no4 * no4 * (U2_2 + epsc4 * U2_4)
             + aK2_hi ** 2 / (12 * klo) * no4 * (1 + r1 * nT16 / six_klo) * ebar8 * U2_4)
    termDp = (aK2_hi ** 4 / 4 * (EoUt(4) + (ndo(2) + epsc4 * no4) * U4_2)
              + aK2_hi ** 3 / 3 * no4 * no4 * (U3_2 + epsc4 * U3_4)
              + aK2_hi ** 3 / (18 * klo) * no4 * (1 + r1 * nT16 / six_klo) * ebar8 * U3_4)
    if MUT == 'no-om-term':
        termD = termDp = Fr(0)
    I2, I3 = I_hi[2], I_hi[3]
    lead = I3 * (termA + termB + termC) + r1 * I2 * termD
    leadp = I3 * (termAp + termBp + termCp) + r1 * I2 * termDp
    # ---- normalizer floor Z_r / (36 k^2 r^2) >= m3(b'_lo) - corrections (NOTE section 6)
    Lb = cholesky([[IV(x.lo, x.hi) for x in row] for row in law.Cb])
    eta = sqrt_up(sum(absup(Lb[i][j] - (SQRT2 if i == j else 0)) ** 2 for i in range(3) for j in range(i + 1)))
    R0 = Fr(R0_COUPLING)
    bp_lo = blo - eps - eta * R0
    bp_hi = bhi + eps + eta * R0
    bp_lo = Fr(math.floor(bp_lo * 2 ** 40), 2 ** 40)
    bp_hi = Fr(math.ceil(bp_hi * 2 ** 40), 2 ** 40)
    pR0 = chi_tail(3, R0)
    m3lo = m3(bp_lo).lo
    tail_c = sqrt_up(contact_moment(4, 0, bp_lo).hi) * sqrt_up(pR0)
    # upper bounds of E[h(B)] for Loewner-monotone h through b'' = bp_hi, plus the outside-ball part
    nBF16 = rup_rel(muB + sig * chi_norm(3, 16))                 # || ||B||_F ||_16
    out_ball = sqrt_up(pR0)

    def up_moment(a, c):                    # E[det^a tr^c 1{B > 0}] <= contact value at b'' + outside-ball part
        crude = rup_rel(nBF16 ** (2 * a + c))      # det <= ||B||^2, tr <= sqrt2 ||B|| <= 2 ||B||: crude, times P^(1/2)
        return rup_rel(contact_moment(a, c, bp_hi).hi + 2 ** c * crude * out_ball)
    F2 = up_moment(4, 0)                    # E[(det B)^4 1{B > 0}]
    DT, TDT, TTDT = up_moment(1, 1), up_moment(1, 2), up_moment(1, 3)
    sdT, sdtau = law.sdQ['T'], law.sdQ['tau']
    q_bad = gauss_tail2(mabs['T'], sdT, 3 * klo / (2 * r1)) + gauss_tail2(mabs['tau'], sdtau, 3 * klo / (2 * r1 * r1))
    nF = sqrt_up(F2)
    corr_G0 = nF * sqrt_up(q_bad) if q_bad > 0 else Fr(0)
    corr_eps = (2 * r1 * gauss_norm(mabs['T'], sdT, 2) + r1 * r1 * gauss_norm(mabs['tau'], sdtau, 2)) / six_klo * nF
    trO = sum(m * max(law.Cg[(n, n)].hi, Fr(0)) for n, m in zip(ONAMES, (1, 1, 2)))
    trV = law.Cg[('v1', 'v1')].hi + law.Cg[('v2', 'v2')].hi
    Eog = rup_rel(mu_o + sqrt_up(trO))
    corr_om = Eog * DT + bo * (TDT + muB * DT)
    Evg2 = mu_v ** 2 + trV
    corr_v = (Fr(17, 16) * Evg2 * DT + 17 * bv ** 2 * (TTDT + 2 * muB * TDT + muB * muB * DT)) / (4 * klo)
    corr = tail_c + corr_G0 + corr_eps + r1 * (corr_om + corr_v)
    if MUT == 'no-zfloor-correction':
        corr = Fr(0)
    zt = m3lo - corr
    require(zt > 0, 'normalizer floor not positive')
    main = KB * min(gbar * lead, g0 * lead + r1 * Lg * leadp) / zt
    # ---- rare branches at the band top: E[W 1_E] <= ||W 1_typed||_q P(E)^(1 - 1/q), ||W 1_typed||_q / r^2 <= g1 g2
    six_khi = 6 * khi
    trB = sum(x.hi for x in (law.Cb[0][0], law.Cb[1][1], law.Cb[2][2]))
    trOQ = sum(m * law.C[(n, n)].hi for n, m in zip(ONAMES, (1, 1, 2)))
    trVQ = law.C[('v1', 'v1')].hi + law.C[('v2', 'v2')].hi
    trNQ = law.C[('nu1', 'nu1')].hi + law.C[('nu2', 'nu2')].hi

    # W_r / r^2 <= |c1| D [|c2| (D + r Om tr + r^2 Om^2) + r ||e||^2 (tr + r Om)] on the typed support, D = det B 1{B > 0},
    # tr = tr B, Om = ||Omega'||_F (NOTE section 7)
    sd_tr = sqrt_up(law.C[('a11', 'a11')].hi + law.C[('a22', 'a22')].hi + 2 * absup(law.C[('a11', 'a22')]))
    m_tr = absup(mu['a11'] + mu['a22'])

    def wnorm(qq):
        p4, p8 = 4 * qq, 8 * qq
        nD = lambda p: det_norm(p, bp_hi, muB, sig)
        ntr = lambda p: gauss_norm(m_tr, sd_tr, p)
        no = lambda p: frob_norm(mu_o, trOQ, p)
        nT = lambda p: gauss_norm(mabs['T'], sdT, p)
        nc2 = lambda p: six_khi + r1 * nT(p) + r1 * r1 * gauss_norm(mabs['tau'], sdtau, p)
        ne = lambda p: r1 * frob_norm(mu_nu, trNQ, p) + frob_norm(mu_v, trVQ, p)
        g1 = (six_khi + r1 * nT(p4)) * nD(p4)
        g2 = (nc2(p4) * (nD(p4) + r1 * no(p8) * ntr(p8) + r1 * r1 * no(p8) ** 2)
              + r1 * ne(p8) ** 2 * (ntr(p4) + r1 * no(p4)))
        return g1 * g2
    thr = 3 * klo / (10 * r1)
    qA2 = Fr(0)
    args = []
    for blk in BLOCKS:
        a_, c_ = blk
        mem = block_members(*blk)
        m_ = sqrt_up(sum(m * mabs[n] ** 2 for n, m in mem))       # |mean of d_x^a d_u^c f(0)| <= ||mean||_F for unit u
        best = None
        for frac in (Fr(7, 8), Fr(15, 16)):                        # split t = frac t + (1 - frac) t between jet and Taylor tail
            if c_ == 0:                                            # scalar block
                x_ = (thr * frac - m_) / law.sdQ[mem[0][0]]
                pj = gauss_tail2(m_, law.sdQ[mem[0][0]], thr * frac)
            elif c_ == 1:                                          # vector block: ||X - mean|| <= sqrt(lambda_max) chi_2
                lmax = max(sum(absup(law.C[(ni, nj)]) for nj, mj in mem) for ni, mi in mem)
                x_ = (thr * frac - m_) / sqrt_up(lmax)
                pj = chi_tail(2, x_) if x_ > 0 else Fr(1)
            else:                                                  # ||X||_op = max_theta |Q(theta)|, Q a trigonometric
                rho = 1 - Fr(c_, NET_M)                            # polynomial of degree c: Bernstein |Q'| <= c max|Q| and
                sdn = sqrt_up(law.net_var[blk])                    # net gaps <= 2/NET_M give max|Q| <= max_j |Q_j| / rho
                x_ = (thr * frac * rho - m_) / sdn
                pj = min(Fr(1), len(NET) * gauss_tail2(m_, sdn, thr * frac * rho))
            pj += (law.S[blk] * (sqE + znorm(32)) / (thr * (1 - frac))) ** 32
            if best is None or pj < best[0]:
                best = (pj, x_)
        args.append(best[1])
        qA2 += best[0]
    a_hi = Fr(4, 3) / klo * a_fac
    qfar = Fr(0)
    qx = Fr(0)
    sdB11 = sqrt_up(law.C[('a11', 'a11')].hi)
    if bbar > 0:
        lam_far = 1 / (4 * a_hi * r1 * bbar * bbar)
        args.append((lam_far - absup(mu['a11'])) / sdB11)
        qfar = gauss_tail1(absup(mu['a11']), sdB11, lam_far)

    def lam2_tail(s):                       # P(lambda_2 > s) <= P(sig chi_3 > s - ||mu_B||)
        x_ = (s - muB) / sig
        args.append(x_)
        return chi_tail(3, x_) if x_ > 0 else Fr(1)
    for i in (1, 2, 3, 4):                         # x_i > X0  <=>  y_i' + bar-beta_i lambda_2 > X0 / (a r1 bar-beta_i)
        if bbars[i] == 0:
            continue
        tx = x0 / (a_hi * r1 * bbars[i])
        if i == 1:
            cb = 12 * khi + bbars[1] * muB
            if tx <= cb:
                qx += Fr(1)
            else:
                qx += (r1 * nDelta(1, 32) / ((tx - cb) / 2)) ** 32
                qx += lam2_tail((tx - cb) / (2 * bbars[1]))
        else:
            s_i = {2: s2, 3: s3, 4: s4}[i]
            x_ = tx / (3 * s_i)
            args.append(x_)
            qx += chi_tail(NU[i], x_) + (nd(i, 32, lam=False) * Klo / (tx / 3)) ** 32 + lam2_tail(tx / (3 * bbars[i]))
    qs = [min(Fr(1), x) for x in (qA2, qfar, qx)]
    tail_num = min(wnorm(qq) * sum(pow_tail(x, qq) for x in qs) for qq in QS)
    if last_band:
        # (0, R] is the union of [R 2^-(j+1), R 2^-j]; every Gaussian or chi tail argument scales at least like 1/r and is
        # >= 3 at R, so each such piece drops by more than 64 when r halves; every Markov piece is a power r^m with m >= 32;
        # after the power 1 - 1/q (q >= 2) each piece still drops by more than 8, so for the q chosen at R the supremum of
        # tail / r^3 over (0, R] is attained on [R/2, R]
        require(min(args) >= 3, 'last-band tail arguments')
        require(max(qA2, qfar, qx) < 1, 'last-band tail pieces')
        rdiv = r1 / 2
    else:
        rdiv = r0
    tail = tail_num / (36 * klo * klo * zt * rdiv ** 3)
    total = main + tail
    return {'total': total, 'main': main, 'tail': tail, 'zt': zt, 'tail_num': tail_num, 'KB': KB, 'g0': g0, 'gbar': gbar, 'I3': I3,
            'termA': termA, 'termB': termB, 'termC': termC, 'termD': termD, 'qA2': qA2, 'qfar': qfar, 'qx': qx,
            'm3lo': m3lo, 'corr': corr, 'eps': eps, 'eta': eta, 'min_arg': min(args)}
