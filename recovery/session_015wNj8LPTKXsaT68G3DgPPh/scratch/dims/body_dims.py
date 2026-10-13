# ----------------------------------------------------------------------------- exact contact law in d = 3 (reference kernel)
def _h(n):
    return 0 if n % 2 else (-1) ** (n // 2) * math.prod(range(n - 1, 0, -2))


def _cov(a, b):
    """E[d^a f(0) d^b f(0)] = (-1)^|b| d^(a+b) exp(-|z|^2/2) at 0."""
    return Fr((-1) ** sum(b) * math.prod(_h(x + y) for x, y in zip(a, b)))


PINS3 = ((0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1))   # contact jet [LP] (3.4)
JETS3 = {'f_xxy1': (2, 1, 0), 'f_xxy2': (2, 0, 1), 'O11': (1, 2, 0), 'O12': (1, 1, 1), 'O22': (1, 0, 2),
         'T111': (0, 3, 0), 'T112': (0, 2, 1), 'T122': (0, 1, 2), 'T222': (0, 0, 3),
         'A11': (0, 2, 0), 'A12': (0, 1, 1), 'A22': (0, 0, 2)}
CLAIM3 = {'f_xxy1': 2, 'f_xxy2': 2, 'O11': 2, 'O12': 1, 'O22': 2, 'T111': 6, 'T112': 2, 'T122': 2, 'T222': 6,
          'A11': 2, 'A12': 1, 'A22': 2}


def _solve(M, rhs):
    n = len(M)
    A = [row[:] + [x] for row, x in zip(M, rhs)]
    for c in range(n):
        p = next(i for i in range(c, n) if A[i][c] != 0)
        A[c], A[p] = A[p], A[c]
        for i in range(n):
            if i != c and A[i][c] != 0:
                f = A[i][c] / A[c][c]
                A[i] = [x - f * y for x, y in zip(A[i], A[c])]
    return [A[i][n] / A[i][i] for i in range(n)]


def contact_law_3():
    """Conditional law of the jets at 0 given the contact pins (f, f_x, f_xx, f_xxx, f_y1, f_xy1, f_y2, f_xy2) = (b, 0, 0,
    12k, 0, 0, 0, 0): returns (means as linear forms in (b, k), conditional covariance), exact rationals."""
    S = [[_cov(p, q) for q in PINS3] for p in PINS3]
    names = list(JETS3)
    W = {n: _solve(S, [_cov(JETS3[n], q) for q in PINS3]) for n in names}
    mean = {n: (W[n][0], 12 * W[n][3]) for n in names}          # coefficient of b and of k
    C = {(m, n): _cov(JETS3[m], JETS3[n]) - sum(W[m][i] * _cov(JETS3[n], PINS3[i]) for i in range(len(PINS3)))
         for m in names for n in names}
    return mean, C


def check_contact_law():
    mean, C = contact_law_3()
    for n in JETS3:
        want_b = Fr(-1) if n in ('A11', 'A22') else Fr(0)
        require(mean[n] == (want_b, Fr(0)), 'contact mean of ' + n)
        require(C[(n, n)] == CLAIM3[n], 'contact variance of ' + n)
    for m in JETS3:
        for n in JETS3:
            require(m == n or C[(m, n)] == 0, 'contact covariance %s, %s' % (m, n))
    return True


# ----------------------------------------------------------------------------- Gaussian and chi-square moments (intervals)
SQRTPI = isqrt_iv(PI)


def dfact(n):
    return math.prod(range(n, 0, -2)) if n > 0 else 1


def e_abs_std(n):
    """E|Z|^n, Z ~ N(0,1)."""
    if n % 2 == 0:
        return IV(dfact(n - 1))
    return IV(dfact(n - 1)) * SQRT2 / SQRTPI


def e_v1(j):
    """E[v1^(2j)], v1 ~ N(0, 1/2)."""
    return Fr(dfact(2 * j - 1), 2 ** j)


def e_om(n):
    """E|Omega11|^n, Omega11 ~ N(0, 2)."""
    s = IV(2 ** (n // 2)) * (SQRT2 if n % 2 else IV(1))
    return s * e_abs_std(n)


def e_Q(j, k):
    """E[Q^j], Q = 2 v1^2 + 6k |Omega11| (independent factors)."""
    tot = IV(0)
    for l in range(j + 1):
        tot = tot + IV(math.comb(j, l) * 2 ** l * e_v1(l) * (6 * k) ** (j - l)) * e_om(j - l)
    return tot


def gamma_half(two_s):
    """Gamma(s) for s = two_s / 2 > 0 an integer or half-integer, as (rational, uses_sqrt_pi)."""
    if two_s % 2 == 0:
        return Fr(math.factorial(two_s // 2 - 1)), False
    n = two_s // 2                                         # s = n + 1/2
    return Fr(math.factorial(2 * n), 4 ** n * math.factorial(n)), True


def chi_tail(nu, sigma2, p, t):
    """Upper bound (interval) of E[Z^p 1{Z > t}] for Z^2 = sigma2 * chi2_nu, t > 0 rational, p >= 0 integer."""
    t = Fr(t)
    y0 = t * t / sigma2                                   # Y = chi2_nu > y0
    x = y0 / 2
    two_s = nu + p                                         # s = nu/2 + p/2
    g_s, pi_s = gamma_half(two_s)
    g_n, pi_n = gamma_half(nu)
    ratio = IV(g_s / g_n)
    if pi_s and not pi_n:
        ratio = ratio * SQRTPI
    elif pi_n and not pi_s:
        ratio = ratio / SQRTPI
    # regularized upper incomplete gamma Q(s, x)
    if two_s % 2 == 0:
        n = two_s // 2
        Q = exp_neg(IV(x)) * IV(sum(x ** j / math.factorial(j) for j in range(n)))
    else:
        n = two_s // 2                                     # s = n + 1/2
        sx = isqrt_iv(IV(x)) if x > 0 else IV(0)
        erfc = 2 * (1 - Phi(IV(t) / isqrt_iv(IV(sigma2))))   # erfc(sqrt x) = 2 Phibar(t / sigma)
        acc = IV(0)
        for j in range(n):
            g, _ = gamma_half(2 * j + 3)                   # Gamma(j + 3/2) = g sqrt(pi)
            acc = acc + IV(x ** j) * sx / (IV(g) * SQRTPI)
        Q = erfc + exp_neg(IV(x)) * acc
    sig_p = IV(sigma2 ** (p // 2)) * (isqrt_iv(IV(sigma2)) if p % 2 else IV(1))
    two_q = IV(2 ** (p // 2)) * (SQRT2 if p % 2 else IV(1))     # 2^(p/2)
    val = sig_p * two_q * ratio * Q
    return IV(0, val.hi)


# ----------------------------------------------------------------------------- the bracket of J^(d)(k)
LAWS = {2: {'v': (1, 2), 'om': (1, 2), 'Y': (1, 6)},      # (nu, sigma2): Z_v = 2|v1|, Z_Om = |Omega11|, Z_Y = |T111|
        3: {'v': (2, 2), 'om': (3, 2), 'Y': (4, 6)}}      # Z_v = 2||v||, Z_Om = ||Omega||_F, Z_Y = weighted Frobenius


def tail_hi(d, k):
    """Upper bound of T^(d)(k) = E[(P(8M^2) - P(1152 k^2)) 1{X > 12k}] (NOTE section 3), via
    E[(512/3 Zbar^6 + 64 Q Zbar^4 + 8 Q^2 Zbar^2) 1{Zbar > t}] <= sum over the three blocks."""
    t = 12 * k
    law = LAWS[d]
    tot = IV(0)
    coef = {0: Fr(512, 3), 1: Fr(64), 2: Fr(8)}           # coefficient of Q^j Z^(6 - 2j)
    for j in (0, 1, 2):
        p = 6 - 2 * j
        # block Y: independent of Q
        if MUT != 'no-y-tail':
            nu, s2 = law['Y']
            tot = tot + IV(coef[j]) * e_Q(j, k) * chi_tail(nu, s2, p, t)
        # block Omega: Q <= 2 v1^2 + 6k Z_Om
        nu, s2 = law['om']
        if MUT == 'frob-omega-off' and d == 3:
            nu = 1                                         # wrong: |Omega11| does not dominate ||Omega||_op
        for l in range(j + 1):
            tot = tot + IV(coef[j] * math.comb(j, l) * 2 ** l * e_v1(l) * (6 * k) ** (j - l)) * chi_tail(nu, s2, p + j - l, t)
        # block v: Q <= Z_v^2 / 2 + 6k |Omega11|
        nu, s2 = law['v']
        for l in range(j + 1):
            tot = tot + IV(coef[j] * math.comb(j, l) * Fr(1, 2 ** l) * (6 * k) ** (j - l)) * e_om(j - l) * chi_tail(nu, s2, p + 2 * l, t)
    return tot.hi


def J_bracket(d, k):
    """Self-contained bracket of J^(d)(k):  216 k^3 J = 512 t^6/3 - 6 t^2 - E[P(m0)] + T,  t = 12k,
    |E[P(m0)]| <= B_P = 2 E[Q^3],  -2 E[Q^6] / (8 t^2)^3 <= T <= tail_hi."""
    k = Fr(k)
    t = 12 * k
    base = Fr(512) * t ** 6 / 3 - 6 * t ** 2
    bp = (2 * e_Q(3, k)).hi
    if MUT == 'no-bp':
        bp = Fr(0)
    tlo = (2 * e_Q(6, k)).hi / (8 * t * t) ** 3
    thi = tail_hi(d, k)
    den = 216 * k ** 3
    return IV((base - bp - tlo) / den, (base + bp + thi) / den), {'B_P': bp, 'T_lo': tlo, 'T_hi': thi}


def J3_from_G(k):
    """Monotone coupling: J^(2)(k) <= J^(3)(k) <= J^(2)(k) + tail_hi(3, k) / (216 k^3), with J^(2) from [G]."""
    lo, hi = (Fr(x) for x in JG[k])
    return IV(lo, hi + tail_hi(3, k) / (216 * k ** 3))


def R3_0():
    """R_3(0) = N_3(0) m_(2,0) / m_(3,0) = 2 sqrt2 / ((7 - 4 sqrt2)/2) = (32 + 28 sqrt2)/17 (NOTE section 2)."""
    if MUT == 'r3-one':
        return IV(1)
    return (32 + 28 * SQRT2) / 17


R_0 = 1 / (2 * SQRTPI)                                     # R(0) = phi_2(0) / m_(2,0) = (1/(2 sqrt pi)) / 1


def m3_0_polar():
    """m_(3,0) = (1/(4 sqrt(2 pi))) * int rho^6 e^(-rho^2/4) drho * int_{pi/4}^{pi/2} c^2 s^2 (s - c) dtheta
    = (30/sqrt2) * (7/(30 sqrt2) - 2/15) = (7 - 4 sqrt2)/2 (NOTE section 2): returns both sides."""
    lhs = IV(30) / SQRT2 * (IV(7) / (30 * SQRT2) - Fr(2, 15))
    rhs = (7 - 4 * SQRT2) / 2
    return lhs, rhs
