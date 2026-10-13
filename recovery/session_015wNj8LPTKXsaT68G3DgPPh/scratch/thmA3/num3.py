def phi_pt(x):
    """phi(x) for rational x (interval)."""
    x = Fr(x)
    return exp_neg_point(x * x / 2) / SQRT2PI


def Phibar_pt(x):
    """1 - Phi(x) for rational x (interval)."""
    x = Fr(x)
    if abs(x) <= 12:
        p = phi_point(x)
        return IV(1 - p.hi, 1 - p.lo)
    if x > 0:
        return IV(0, (phi_pt(x) / x).hi)            # Mills: Phibar(x) <= phi(x)/x
    return IV(1 - (phi_pt(-x) / (-x)).hi, 1)


def dfact(n):
    """(n)!! for n >= -1."""
    return math.prod(range(n, 0, -2)) if n > 0 else 1


def gauss_moment(j):
    """E[Z^j] for Z ~ N(0, 1)."""
    return 0 if j % 2 else dfact(j - 1)


def upper_moments(x, J):
    """[M_0(x), ..., M_J(x)], M_j(x) = integral_x^inf t^j phi(t) dt, for rational x (intervals).  For x < 0,
    M_j(x) = E[Z^j] - (-1)^j M_j(-x)."""
    x = Fr(x)
    if x < 0:
        M = upper_moments(-x, J)
        return [IV(gauss_moment(j)) - M[j] * (-1) ** j for j in range(J + 1)]
    ph = phi_pt(x)
    M = [Phibar_pt(x), ph]
    for j in range(2, J + 1):
        M.append(ph * x ** (j - 1) + M[j - 2] * (j - 1))
    return M[:J + 1]


def rup_rel(x, bits=200):
    """Round a rational x >= 0 up to a dyadic with `bits` significant bits (upper bounds stay upper bounds)."""
    x = Fr(x)
    if x <= 0:
        return x
    e = x.numerator.bit_length() - x.denominator.bit_length()
    sh = bits - e
    if sh >= 0:
        return Fr(-((-x.numerator << sh) // x.denominator), 1 << sh)
    return Fr(-((-x.numerator) // (x.denominator << -sh)) << -sh)


def rdown_rel(x, bits=200):
    """Round a rational x >= 0 down to a dyadic with `bits` significant bits."""
    x = Fr(x)
    if x <= 0:
        return x
    e = x.numerator.bit_length() - x.denominator.bit_length()
    sh = bits - e
    if sh >= 0:
        return Fr((x.numerator << sh) // x.denominator, 1 << sh)
    return Fr((x.numerator // (x.denominator << -sh)) << -sh)


def sqrt_up(x):
    """Rational upper bound of sqrt(x), x >= 0 rational, relative error below 2^-200 (no absolute rounding floor)."""
    x = Fr(x)
    if x == 0:
        return Fr(0)
    n, d = x.numerator, x.denominator
    nd = n * d
    m = max(0, (420 - nd.bit_length()) // 2 + 1)
    return rup_rel(Fr(math.isqrt(nd << (2 * m)) + 1, d << m))


def sqrt_down(x):
    """Rational lower bound of sqrt(x), x >= 0 rational."""
    x = Fr(x)
    if x <= 0:
        return Fr(0)
    n, d = x.numerator, x.denominator
    nd = n * d
    m = max(0, (420 - nd.bit_length()) // 2 + 1)
    return rdown_rel(Fr(math.isqrt(nd << (2 * m)), d << m))


def iroot(x, p):
    """Upper bound (rational) of x^(1/p) for rational x >= 0 and p a power of two."""
    v = Fr(x)
    while p > 1:
        v = sqrt_up(v)
        p //= 2
    return v


def pow2(p):
    q = 1
    while q < p:
        q *= 2
    return q


ZN = {}


def znorm(p):
    """Upper bound of ||N(0,1)||_p for p a power of two (p = 1 uses sqrt(2/pi) < 4/5)."""
    if p == 1:
        return Fr(4, 5)
    if p not in ZN:
        ZN[p] = iroot(Fr(dfact(p - 1)), p)
    return ZN[p]


def znorm1(p):
    """max(1, ||N(0,1)||_p): the factor in ||(|mu| + ||Sigma^(1/2) z||)||_p <= |mu| + sqrt(tr Sigma) znorm1(p)."""
    return Fr(1) if p <= 2 else znorm(p)


def frob_norm(mu_abs, tr_hi, p):
    """Upper bound of || ||X|| ||_p for a Gaussian vector X with ||E X|| <= mu_abs and tr Cov X <= tr_hi (Minkowski in
    L^(p/2) on the eigen-decomposition for p >= 2; Lyapunov for p <= 2)."""
    return rup_rel(mu_abs + sqrt_up(tr_hi) * znorm1(pow2(p)))


def gamma_half(n2):
    """Gamma(n2 / 2) for an integer n2 >= 1, as an interval (sqrt(pi) for odd n2)."""
    if n2 % 2 == 0:
        return IV(math.factorial(n2 // 2 - 1))
    n = (n2 - 1) // 2                                 # Gamma(n + 1/2) = (2n - 1)!! sqrt(pi) / 2^n
    return isqrt_iv(PI) * Fr(dfact(2 * n - 1), 2 ** n)


def chi_moment(nu, j):
    """E[chi_nu^j] = 2^(j/2) Gamma((nu + j)/2) / Gamma(nu/2) (interval), integer j >= 0."""
    if j % 2 == 0:
        return IV(math.prod(nu + 2 * i for i in range(j // 2)))
    out = gamma_half(nu + j) / gamma_half(nu)
    for _ in range(j):
        out = out * SQRT2
    return out


CN = {}


def chi_norm(nu, p):
    """Upper bound of ||chi_nu||_p for p a power of two."""
    if p == 1:
        return sqrt_up(Fr(nu))
    key = (nu, p)
    if key not in CN:
        CN[key] = iroot(Fr(math.prod(nu + 2 * i for i in range(p // 2))), p)
    return CN[key]


CC = {}


def chi_const(nu):
    """C_nu sqrt(2 pi) with C_nu = 1 / (2^(nu/2 - 1) Gamma(nu/2)): E[chi_nu^j 1{chi > x}] = C_nu sqrt(2 pi) M_(j+nu-1)(x)."""
    if nu not in CC:
        p2 = IV(Fr(2) ** ((nu - 2) // 2)) if nu % 2 == 0 else IV(Fr(2) ** ((nu - 3) // 2)) * SQRT2
        CC[nu] = SQRT2PI / (p2 * gamma_half(nu))
    return CC[nu]


def chi_upper_moments(nu, x, J):
    """[E[chi_nu^j 1{chi_nu > x}] for j = 0..J] (intervals), rational x >= 0."""
    M = upper_moments(x, J + nu - 1)
    c = chi_const(nu)
    return [M[j + nu - 1] * c for j in range(J + 1)]


def chi_tail(nu, x):
    """Rational upper bound of P(chi_nu > x) for rational x >= 0 (Mills-type bound beyond the series range)."""
    x = Fr(x)
    if x <= 0:
        return Fr(1)
    if x <= 12:
        return min(Fr(1), chi_upper_moments(nu, x, 0)[0].hi)
    # x > 12: P(chi_nu > x) = c_nu sqrt(2pi) M_(nu-1)(x), M_j(x) <= phi(x) x^(j-1) (1 + 2 (j-1)/x^2) for x^2 >= 2 j, and
    # phi(x) <= m!/(x^2/2)^m / sqrt(2 pi) (m = 40)
    y = x * x / 2
    ph = Fr(math.factorial(40)) / y ** 40 * Fr(2, 5)          # 1/sqrt(2 pi) < 2/5
    j = nu - 1
    Mj = ph * (x ** (j - 1) if j >= 1 else 1 / x) * (1 + Fr(2 * max(j - 1, 1)) / (x * x))
    return min(Fr(1), Mj * Fr(51, 20))                          # c_nu sqrt(2 pi) <= sqrt(2 pi) < 51/20


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


def gauss_norm(mu_abs, sd, p):
    """Upper bound of ||X||_p for X Gaussian with |mean| <= mu_abs and sd <= sd."""
    return rup_rel(mu_abs + sd * znorm(pow2(p)))


def absup(x):
    return max(abs(x.lo), abs(x.hi))


def pow_tail(x, q):
    """Upper bound of x^(1 - 1/q) for 0 <= x <= 1 (q a power of two)."""
    if x <= 0:
        return Fr(0)
    return iroot(rup_rel(x, 64) ** (q - 1), q)
