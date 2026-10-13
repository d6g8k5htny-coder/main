

# ----------------------------------------------------------------------------- moment calculus (normalized by K = 12 k)
def binom(n, k):
    return math.comb(n, k)


def gauss_norm(mu_abs, sd, p):
    """Upper bound of ||X||_p for X Gaussian with |mean| <= mu_abs and sd <= sd."""
    return rup_rel(mu_abs + sd * znorm(pow2(p)))


def pos_part_moments(s, c, nmax, extra_max):
    """P[n][e] = upper bound of E[u^e (u^n - c^n)_+], u = s |zeta|, for 1 <= n <= nmax, 0 <= e <= extra_max."""
    x = c / s
    M = upper_moments(x, nmax + extra_max)
    out = {}
    for n in range(1, nmax + 1):
        for e in range(extra_max + 1):
            out[(n, e)] = rup_rel(((M[n + e] * s ** (n + e) - M[e] * c ** n * s ** e) * 2).hi)
    return out


def normalized_moments(q, nmax):
    """Component moments, normalized by K = 12 k (K >= Klo), for n <= nmax:
      E1[n], E1o[n], E1v[n] : upper bounds of E[y1^n], E[o0 y1^n], E[v0^2 y1^n], y1 = 1 + cmu1/K + (r1/K) Delta1;
      Tn[n], To[n], Tv[n]   : upper bounds of sum_{i=2,3,4} E[(y_i^n - c^n)_+] (and with the factors o0, v0^2),
                              y_i <= u_i + delta_i (u_i = s_i |zeta| independent half-normals), threshold c = q['c'].
    o0 = s3 |z2| and v0^2 = s2^2 z1^2 / 4 are unnormalized."""
    th = THETA
    cp = q['c'] * (1 - th)                  # <= c (1 + th)^{-(n-1)/n} for every n (Bernoulli)
    PP = {i: pos_part_moments(q['s'][i], cp, nmax, 2) for i in (2, 3, 4)}
    Eu3 = (habs_moment(1) * q['s3']).hi
    Eu2sq = (habs_moment(2) * q['s2'] * q['s2']).hi
    n_o0_2 = q['s3'] * znorm(2)
    n_u2sq_2 = q['s2'] * q['s2'] * znorm(4) ** 2
    r3 = q['s3'] / q['s'][3]
    r2sq = (q['s2'] / q['s'][2]) ** 2
    E1, E1o, E1v, Tn, To, Tv = {}, {}, {}, {}, {}, {}
    base = 1 + q['cmt1']
    for n in range(0, nmax + 1):
        ey = base ** n
        eo = base ** n * Eu3
        ev = base ** n * Eu2sq
        for j in range(1, n + 1):
            coef = binom(n, j) * base ** (n - j) * q['rK'] ** j
            ey += coef * q['nD1'](pow2(j)) ** j
            eo += coef * n_o0_2 * q['nD1'](pow2(2 * j)) ** j
            ev += coef * n_u2sq_2 * q['nD1'](pow2(2 * j)) ** j
        E1[n], E1o[n], E1v[n] = rup_rel(ey), rup_rel(eo), rup_rel(ev / 4)
        if n == 0:
            continue
        f1 = (1 + th) ** (n - 1)
        f2 = (1 + 1 / th) ** (n - 1)
        tn = to = tv = Fr(0)
        for i in (2, 3, 4):
            P = PP[i]
            dn = q['nd'](i, pow2(n)) ** n
            dn2 = q['nd'](i, pow2(2 * n)) ** n
            tn += f1 * P[(n, 0)] + f2 * dn
            to += (f1 * r3 * P[(n, 1)] if i == 3 else f1 * Eu3 * P[(n, 0)]) + f2 * n_o0_2 * dn2
            tv += (f1 * r2sq * P[(n, 2)] if i == 2 else f1 * Eu2sq * P[(n, 0)]) + f2 * n_u2sq_2 * dn2
        Tn[n], To[n], Tv[n] = rup_rel(tn), rup_rel(to), rup_rel(tv / 4)
    return E1, E1o, E1v, Tn, To, Tv


def elam_pos(m, s):
    """E[(s z + m)_+] at a point (rational m, s > 0)."""
    x = Fr(m) / s
    return m * Phi(IV(x)) + phi_pt(x) * s


def elam2_pos(m, s):
    """E[(s z + m)_+^2] at a point."""
    x = Fr(m) / s
    return (m * m + s * s) * Phi(IV(x)) + phi_pt(x) * (m * s)


def box_bound(law, bb, kb, r0, r1, last_band=False):
    """Certified upper bound of sup_{r in [r0, r1]} Q^W(G_r^c) / r^3 over b in bb, k in kb, with its pieces."""
    b = IV(bb[0], bb[1])
    k = IV(kb[0], kb[1])
    klo, khi = kb[0], kb[1]
    Klo = 12 * klo
    mu = {n: b * law.alpha[n] + k * law.beta[n] for n in TARGET_NAMES}
    mabs = {n: absup(mu[n]) for n in TARGET_NAMES}
    E = b * b * law.e_bb + b * k * law.e_bk * 2 + k * k * law.e_kk
    require(E.hi > 0, 'energy')
    sqE = isqrt_iv(IV(E.hi)).hi
    sw_lo, sw_hi = law.sw.lo, law.sw.hi
    muw_abs = mabs['w']
    bw = {n: absup(law.bw[n]) for n in law.bw}
    t = {n: law.S[n] / sw_lo for n in JET4}
    a_fac = Fr(1, 2) if MUT == 'cap-half' else Fr(1)      # the cap constant 4/(3k) (mutant: halved)
    B = lambda n: bw[n] + t[n]
    bbars = {1: r1 * (Fr(5, 2) * B('X40') + 2 * B('X31')),
             2: 2 * bw['v'] + r1 * (Fr(5, 2) * B('X31') + 2 * B('X22')),
             3: bw['om'] + r1 * (Fr(5, 2) * B('X22') + 2 * B('X13')),
             4: bw['Y'] + r1 * (2 * B('X13') + 2 * B('X04'))}
    bbars = {i: rup_rel(v) for i, v in bbars.items()}
    bbar = max(bbars.values())
    L3 = law.L3
    s2, s3, s4 = 2 * L3[0][0].hi, L3[1][1].hi, L3[2][2].hi
    l21, l31, l32 = absup(L3[1][0]), absup(L3[2][0]), absup(L3[2][1])

    memo = {}

    def nA(n, p):     # ||A_alpha^g||_p, A_alpha^g = |X_alpha,g| + T_alpha^g
        key = ('A', n, p)
        if key not in memo:
            memo[key] = rup_rel(gauss_norm(mabs[n], law.sdg[n].hi, p) + law.S[n] * (sqE + znorm(pow2(p))))
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
            return Fr(5, 2) * nA('X40', p) + 2 * nA('X31', p)
        if i == 2:
            return Fr(5, 2) * nA('X31', p) + 2 * nA('X22', p)
        if i == 3:
            return Fr(5, 2) * nA('X22', p) + 2 * nA('X13', p)
        return 2 * nA('X13', p) + 2 * nA('X04', p)

    def nd(i, p):
        key = ('d', i, p)
        if key not in memo:
            memo[key] = rup_rel(nd0(i, p))
        return memo[key]

    def nd0(i, p):     # ||delta_i||_p / Klo, delta_i including beta_i |mu_w|
        zp = znorm(pow2(p))
        cm = bbars[i] * muw_abs
        if i == 2:
            v = 2 * mabs['v'] + cm + r1 * nDelta(2, p)
        elif i == 3:
            v = mabs['om'] + l21 * zp + cm + r1 * nDelta(3, p)
        else:
            v = mabs['Y'] + (l31 + l32) * zp + cm + r1 * nDelta(4, p)
        return v / Klo

    kappa = 3 / (1 - 3 * X0)
    Fk = 1 + kappa * X0                      # (1 + kappa x_i) <= Fk on the event {x_i <= X0}
    cth = 1 - kappa * X0 / 2                 # <= Fk^{-1/2}
    NMAX = 18
    q = {'cmt1': bbars[1] * muw_abs / Klo, 'rK': r1 / Klo, 'nD1': lambda p: nDelta(1, p),
         's': {2: s2 / Klo, 3: s3 / Klo, 4: s4 / Klo}, 'nd': nd, 's3': s3, 's2': s2, 'c': cth}
    E1, E1o, E1v, Tn, To, Tv = normalized_moments(q, NMAX)
    et1 = rup_rel(16 * kappa * r1 * bbars[1] * a_fac)     # eta_1 K, k-free
    aK2_hi = 192 * khi * a_fac                    # a K^2 = 192 k

    # moments of Ut = U / (a K^2) <= max(Z1, Fk y_i^2), Z1 = y1^2 (1 + et1 y1)
    def EUt(p):                                   # upper bound of E[Ut^p]
        key = ('U', p)
        if key not in memo:
            memo[key] = rup_rel(sum(binom(p, l) * et1 ** l * E1[2 * p + l] for l in range(p + 1)) + Fk ** p * Tn[2 * p])
        return memo[key]

    def EoUt2():
        return sum(binom(2, l) * et1 ** l * E1o[4 + l] for l in range(3)) + Fk ** 2 * To[4]

    def EvUt2():
        return sum(binom(2, l) * et1 ** l * E1v[4 + l] for l in range(3)) + Fk ** 2 * Tv[4]

    def sq(x):
        return isqrt_iv(IV(x)).hi

    U3_2 = sq(EUt(6))                              # ||Ut^3||_2
    U2_2 = sq(EUt(4))                              # ||Ut^2||_2

    def nU(p):                                     # ||U||_p <= a K^2 Fk ||max_i y_i||_2p^2 (on the event x_i <= X0)
        require(2 * p <= NMAX and pow2(p) == p, 'norm order')
        key = ('nU', p)
        if key not in memo:
            memo[key] = rup_rel(aK2_hi * Fk * iroot(E1[2 * p] + Tn[2 * p], p))
        return memo[key]

    def nbar(n, p):                                # ||F_bar||_p, F_bar = |F_g| + |beta_F| (|mu_w| + r1 U)
        return gauss_norm(mabs[n], law.sdg[n].hi, p) + bw[n] * (muw_abs + r1 * nU(p))

    six_klo = 6 * klo
    nT2, nT4, nt2, nt4 = nbar('T', 2), nbar('T', 4), nbar('tau', 2), nbar('tau', 4)
    epsc2 = (2 * r1 * nT2 + r1 * r1 * nt2) / six_klo + (r1 * nT4) * (r1 * nT4 + r1 * r1 * nt4) / six_klo ** 2
    nT8, nt8 = nbar('T', 8), nbar('tau', 8)
    epsc4 = (2 * r1 * nT4 + r1 * r1 * nt4) / six_klo + (r1 * nT8) * (r1 * nT8 + r1 * r1 * nt8) / six_klo ** 2
    # Term A / (36 k^2)
    A_n = EUt(3) + epsc2 * U3_2
    termA = aK2_hi ** 3 / 3 * A_n
    # Term B / (36 k^2)
    ndo = lambda p: l21 * znorm(pow2(p)) + mabs['om'] + bw['om'] * (muw_abs + r1 * nU(p))
    no4 = s3 * znorm(4) + ndo(4)
    B_n = EoUt2() + (ndo(2) + epsc4 * no4) * U2_2
    termB = aK2_hi ** 2 / 2 * B_n
    if MUT == 'no-om-term':
        termB = Fr(0)
    # Term C / (36 k^2) = (a K^2)^2 / (12 k) * E[(1 + r T/(6k)) e Ut^2]
    th = THETA_E
    dv = lambda p: mabs['v'] + bw['v'] * (muw_abs + r1 * nU(p))
    ebar4 = (r1 * nbar('nu', 8) + L3[0][0].hi * znorm(8) + dv(8)) ** 2
    C_n = ((1 + th) ** 2 * EvUt2()
           + ((1 + th) * (1 + 1 / th) * dv(4) ** 2 + (1 + 1 / th) * r1 * r1 * nbar('nu', 4) ** 2) * U2_2
           + r1 * nT4 / six_klo * ebar4 * U2_2)
    termC = aK2_hi ** 2 / (12 * klo) * C_n
    # density of lambda = -w on [0, inf)
    mlo = max(Fr(0), mu['w'].lo)
    pbar = (phi_pt(mlo / sw_hi) / sw_lo).hi
    # ---- normalizer floor Z_r / (36 k^2 r^2) >= El2 - corr
    m_lam = -mu['w']
    El2 = elam2_pos(m_lam.lo, sw_lo).lo
    Elp = elam_pos(m_lam.hi, sw_hi).hi
    lam_n = lambda p: muw_abs + sw_hi * znorm(pow2(p))
    sdT, sdtau = law.sdQ['T'].hi, law.sdQ['tau'].hi
    q_bad = gauss_tail2(mabs['T'], sdT, 3 * klo / (2 * r1)) + gauss_tail2(mabs['tau'], sdtau, 3 * klo / (2 * r1 * r1))
    Eom_g = mabs['om'] + law.sdg['om'].hi * Fr(4, 5)
    Evg2 = mabs['v'] ** 2 + law.sdg['v'].hi ** 2
    E_om_lam = Eom_g * Elp + bw['om'] * (muw_abs * lam_n(1) + lam_n(2) ** 2)
    E_v2_lam = 2 * Evg2 * Elp + 2 * bw['v'] ** 2 * (sw_hi * znorm(4)) ** 2 * lam_n(2)
    corr = (lam_n(4) ** 2 * iroot(q_bad, 2) if q_bad > 0 else Fr(0)) \
        + (2 * r1 * gauss_norm(mabs['T'], sdT, 2) + r1 * r1 * gauss_norm(mabs['tau'], sdtau, 2)) * lam_n(4) ** 2 / six_klo \
        + r1 * (E_v2_lam / (3 * klo) + E_om_lam)
    if MUT == 'no-zfloor-correction':
        corr = Fr(0)
    zt = El2 - corr
    require(zt > 0, 'normalizer floor not positive')
    main = pbar * (termA + termB + termC) / zt
    # ---- rare branches at the band top: ||W||_2 / r^2 and probabilities
    six_khi = 6 * khi
    nT8Q = gauss_norm(mabs['T'], sdT, 8)
    ntau8Q = gauss_norm(mabs['tau'], sdtau, 8)
    g1 = (six_khi + r1 * nT8Q) * lam_n(8)
    g2 = (six_khi + r1 * nT8Q + r1 * r1 * ntau8Q) * (lam_n(8) + r1 * gauss_norm(mabs['om'], law.sdQ['om'].hi, 8)) \
        + r1 * (r1 * gauss_norm(mabs['nu'], law.sdQ['nu'].hi, 8) + gauss_norm(mabs['v'], law.sdQ['v'].hi, 8)) ** 2
    w2 = g1 * g2
    thr = 3 * klo / (10 * r1)
    qA2 = Fr(0)
    args = []
    for n in JET4:
        args.append((thr * Fr(7, 8) - mabs[n]) / law.sdQ[n].hi)
        qA2 += gauss_tail2(mabs[n], law.sdQ[n].hi, thr * Fr(7, 8))
        qA2 += (law.S[n] * (sqE + znorm(32)) / (thr / 8)) ** 32
    a_hi = Fr(4, 3) / klo * a_fac
    qfar = Fr(0)
    qx = Fr(0)
    if bbar > 0:
        lam_far = 1 / (4 * a_hi * r1 * bbar * bbar)
        args.append((lam_far - muw_abs) / sw_hi)
        qfar = gauss_tail1(muw_abs, sw_hi, lam_far)
    for i in (1, 2, 3, 4):                         # x_i > X0  <=>  y_i' > X0 / (a r1 beta_i)
        if bbars[i] == 0:
            continue
        tx = X0 / (a_hi * r1 * bbars[i])
        if i == 1:
            cb = 12 * khi + bbars[1] * muw_abs
            qx += Fr(1) if tx <= cb else (r1 * nDelta(1, 32) / (tx - cb)) ** 32
        else:
            s_i = {2: s2, 3: s3, 4: s4}[i]
            args.append(tx / (2 * s_i))
            qx += gauss_tail2(0, s_i, tx / 2) + (nd(i, 32) * Klo / (tx / 2)) ** 32
    qs = [min(Fr(1), x) for x in (qA2, qfar, qx)]
    tail_num = w2 * sum((iroot(x, 2) if x > 0 else Fr(0)) for x in qs)
    if last_band:
        # (0, R] is the union of [R 2^-(j+1), R 2^-j]; every Gaussian tail argument scales at least like 1/r and is >= 3 at R,
        # every Markov piece is a power r^q with q >= 16 after the square root, so each halving of r divides every tail
        # piece by more than 8 and the supremum of tail / r^3 over (0, R] is attained on [R/2, R]
        require(min(args) >= 3, 'last-band tail arguments')
        # no piece is capped at 1, so the factor-64 decrease of every piece per halving carries over to each sum
        require(max(qA2, qfar, qx) < 1, 'last-band tail pieces')
        rdiv = r1 / 2
    else:
        rdiv = r0
    tail = tail_num / (36 * klo * klo * zt * rdiv ** 3)
    total = main + tail
    return {'total': total, 'main': main, 'tail': tail, 'zt': zt, 'pbar': pbar, 'termA': termA, 'termB': termB,
            'termC': termC, 'qA2': qA2, 'qfar': qfar, 'qx': qx, 'El2': El2, 'corr': corr, 'min_arg': min(args)}


def phibar_upper(x):
    """Rational upper bound of 1 - Phi(x) for rational x >= 0: the series value for x <= 12, and for x > 12 the Mills bound
    phi(x)/x with exp(-y) <= m!/y^m (m = 40), which stays exact far below the 2^-160 rounding floor."""
    x = Fr(x)
    if x <= 12:
        return Phibar_pt(x).hi
    y = x * x / 2
    m = 40
    return Fr(math.factorial(m)) / y ** m / (x * Fr(5, 2))      # sqrt(2 pi) > 5/2


def gauss_tail2(mu_abs, sd, t):
    """Upper bound of P(|X| > t) for X Gaussian with |mean| <= mu_abs, sd <= sd."""
    if t <= mu_abs:
        return Fr(1)
    return min(Fr(1), 2 * phibar_upper((t - mu_abs) / sd))


def gauss_tail1(mu_abs, sd, t):
    if t <= mu_abs:
        return Fr(1)
    return min(Fr(1), phibar_upper((t - mu_abs) / sd))
