

# ----------------------------------------------------------------------------- closed forms for the contact transverse Hessian
# B = m I + GOE_2 (diagonal N(0, 2), off-diagonal N(0, 1)): its eigenvalues are s -+ rho with s = m + (G11 + G22)/2 ~ N(m, 1)
# and rho = sqrt(((G11 - G22)/2)^2 + G12^2) Rayleigh, independent; det B = s^2 - rho^2, tr B = 2 s, B > 0 iff s > rho.
def upper_moments_iv(x, J):
    """[M_0(x), ..., M_J(x)] over an interval x whose interior does not contain 0 (each M_j is monotone there)."""
    if x.lo == x.hi:
        return upper_moments(x.lo, J)
    require(not (x.lo < 0 < x.hi), 'incomplete moment over an interval containing 0')
    A, B = upper_moments(x.lo, J), upper_moments(x.hi, J)
    return [IV(min(a.lo, b.lo), max(a.hi, b.hi)) for a, b in zip(A, B)]


def es_pos(m, n):
    """[E[s^j 1{s > 0}] for j = 0..n], s ~ N(m, 1), rational m (intervals)."""
    M = upper_moments(-Fr(m), n)
    return [sum((M[i] * (math.comb(j, i) * Fr(m) ** (j - i)) for i in range(j + 1)), IV(0)) for j in range(n + 1)]


def isqrt2_pow(i):
    """2^(-i/2) (interval)."""
    return IV(Fr(1, 2 ** (i // 2))) if i % 2 == 0 else IV(Fr(1, 2 ** (i // 2))) / SQRT2


def es_pos_exp(m, n):
    """[E[s^j exp(-s^2/2) 1{s > 0}] for j = 0..n], s ~ N(m, 1), rational m:
    = exp(-m^2/4)/sqrt2 . E[(m/2 + U/sqrt2)^j 1{U > -m/sqrt2}], U ~ N(0, 1)."""
    m = Fr(m)
    M = upper_moments_iv(-(IV(m) / SQRT2), n)
    pre = exp_neg_point(m * m / 4) / SQRT2
    out = []
    for j in range(n + 1):
        tot = IV(0)
        for i in range(j + 1):
            tot = tot + M[i] * (math.comb(j, i) * (m / 2) ** (j - i)) * isqrt2_pow(i)
        out.append(pre * tot)
    return out


def inner_rho_polys(a):
    """integral_0^s (s^2 - rho^2)^a rho exp(-rho^2/2) d rho = P_a(s) - exp(-s^2/2) Q_a(s) (exact polynomial coefficient
    dicts {power: Fraction}); with u = rho^2/2 it is integral_0^(s^2/2) (s^2 - 2u)^a e^(-u) du."""
    P, Q = {}, {}
    for i in range(a + 1):
        c = Fr(math.comb(a, i) * (-2) ** i * math.factorial(i))
        P[2 * (a - i)] = P.get(2 * (a - i), Fr(0)) + c
        for l in range(i + 1):
            pw = 2 * (a - i) + 2 * l
            Q[pw] = Q.get(pw, Fr(0)) + c / (2 ** l * math.factorial(l))
    return P, Q


def contact_moment(a, c, m):
    """E[det(B)^a tr(B)^c 1{B > 0}] for B = m I + GOE_2, rational m (interval)."""
    P, Q = inner_rho_polys(a)
    n = 2 * a + c
    E1 = es_pos(m, n)
    E2 = es_pos_exp(m, n)
    tot = IV(0)
    for pw, cf in P.items():
        tot = tot + E1[pw + c] * (cf * 2 ** c)
    for pw, cf in Q.items():
        tot = tot - E2[pw + c] * (cf * 2 ** c)
    return tot


def m3(m):
    """m_(3,m) = E[det(B)^2 1{B > 0}] = E[(s^4 - 4 s^2 + 8) 1{s > 0}] - 4 sqrt2 exp(-m^2/4) Phi(m/sqrt2)."""
    return contact_moment(2, 0, m)


def gauss_half_moment(j, m, sig):
    """J_j(m, sig) = integral_0^inf lambda^j exp(-(lambda - m)^2/(2 sig^2)) d lambda for rational m, sig > 0 (interval):
    sig sqrt(2 pi) E[(m + sig Z)^j 1{m + sig Z > 0}]."""
    m, sig = Fr(m), Fr(sig)
    M = upper_moments(-m / sig, j)
    tot = sum((M[i] * (math.comb(j, i) * m ** (j - i) * sig ** i) for i in range(j + 1)), IV(0))
    return tot * SQRT2PI * sig
