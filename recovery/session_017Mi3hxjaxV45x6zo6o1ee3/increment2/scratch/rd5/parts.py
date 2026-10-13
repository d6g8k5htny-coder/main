### CORE
PREC = 60
CF = Context(prec=PREC, rounding=ROUND_FLOOR, Emin=-9999999, Emax=9999999)
CC = Context(prec=PREC, rounding=ROUND_CEILING, Emin=-9999999, Emax=9999999)
CE = Context(prec=PREC, rounding=ROUND_HALF_EVEN, Emin=-9999999, Emax=9999999)
# Every arithmetic step names one of the three contexts; operator syntax on Decimals would round to the current context
# (default 28 digits), so it is widened as a safety net and rule LIBRARY_EXACT checks exp, sqrt and the negation.
setcontext(Context(prec=2 * PREC + 20, rounding=ROUND_HALF_EVEN, Emin=-9999999, Emax=9999999))
ZERO, ONE, HALF = Decimal(0), Decimal(1), Decimal("0.5")
ROOT_DIGITS = 70
K_ORDER, RHO, CELL, T_CUT = 40, Fr(6), Fr(1, 2), Fr(60)
CELLS_M4 = [(Fr(i, 2), Fr(i + 1, 2)) for i in range(60)] + [(Fr(30 + i), Fr(31 + i)) for i in range(30)]
SIDE24 = {2: ("0.07340691930603427103", "0.07340691930603427104"), 3: ("0.04177593184059834334", "0.04177593184059834335")}
D3_PIN = ("5.32318026888989689492319873968726063578438852940845576405060", "5.32318026888989689492319873968726063578438852947338433618854")   # Math-#199 v1.1
PINNED = {"D_4": "14.", "c_5_ref": "0.0", "pi": "3.14159265358979323846264338327950288419716939937510"}


def ulp(x):
    if x == 0:
        return Decimal(1).scaleb(-PREC - 30)
    return Decimal(1).scaleb(x.adjusted() - PREC + 1)


class Iv:
    """Closed interval [lo, hi] of Decimals; every operation encloses the exact result."""
    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi=None):
        if hi is None:
            hi = lo
        if lo > hi:
            raise ValueError("empty interval")
        self.lo, self.hi = lo, hi

    @staticmethod
    def frac(q):
        q = Fr(q)
        n, d = Decimal(q.numerator), Decimal(q.denominator)
        return Iv(CF.divide(n, d), CC.divide(n, d))

    @staticmethod
    def of(x):
        return x if isinstance(x, Iv) else Iv.frac(x)

    def __add__(a, b):
        b = Iv.of(b)
        return Iv(CF.add(a.lo, b.lo), CC.add(a.hi, b.hi))
    __radd__ = __add__

    def __neg__(a):
        return Iv(a.hi.copy_negate(), a.lo.copy_negate())

    def __sub__(a, b):
        b = Iv.of(b)
        return Iv(CF.subtract(a.lo, b.hi), CC.subtract(a.hi, b.lo))

    def __rsub__(a, b):
        return Iv.of(b) - a

    def __mul__(a, b):
        b = Iv.of(b)
        c = [(x, y) for x in (a.lo, a.hi) for y in (b.lo, b.hi)]
        return Iv(min(CF.multiply(x, y) for x, y in c), max(CC.multiply(x, y) for x, y in c))
    __rmul__ = __mul__

    def __truediv__(a, b):
        b = Iv.of(b)
        if b.lo <= 0 <= b.hi:
            raise ZeroDivisionError("interval division by an interval containing 0")
        c = [(x, y) for x in (a.lo, a.hi) for y in (b.lo, b.hi)]
        return Iv(min(CF.divide(x, y) for x, y in c), max(CC.divide(x, y) for x, y in c))

    def __rtruediv__(a, b):
        return Iv.of(b) / a

    def sq(a):
        if a.lo <= 0 <= a.hi:
            return Iv(ZERO, max(CC.multiply(a.lo, a.lo), CC.multiply(a.hi, a.hi)))
        return a * a

    def exp(a):
        lo, hi = CE.exp(a.lo), CE.exp(a.hi)
        return Iv(max(ZERO, CF.subtract(lo, CC.multiply(Decimal(2), ulp(lo)))), CC.add(hi, CC.multiply(Decimal(2), ulp(hi))))

    def sqrt(a):
        if a.lo < 0:
            raise ValueError("sqrt of an interval with negative part")
        lo, hi = CE.sqrt(a.lo), CE.sqrt(a.hi)
        return Iv(max(ZERO, CF.subtract(lo, CC.multiply(Decimal(2), ulp(lo)))), CC.add(hi, CC.multiply(Decimal(2), ulp(hi))))

    def width(a):
        return CC.subtract(a.hi, a.lo)

    def mag(a):
        return max(abs(a.lo), abs(a.hi))

    def contains(a, x):
        x = Iv.of(x)
        return a.lo <= x.lo and x.hi <= a.hi

    def intersects(a, b):
        return a.lo <= b.hi and b.lo <= a.hi

    def pair(a):
        return [str(a.lo), str(a.hi)]


def hull(a, b):
    return Iv(min(a.lo, b.lo), max(a.hi, b.hi))


def ipow(x, n):
    r = Iv(ONE)
    for _ in range(n):
        r = r * x
    return r


# ---------------- constants and special functions
def inroot(N, n):
    if N < 2:
        return N
    x = 1 << ((N.bit_length() + n - 1) // n)
    while True:
        y = ((n - 1) * x + N // pow(x, n - 1)) // n
        if y >= x:
            break
        x = y
    if not (pow(x, n) <= N < pow(x + 1, n)):
        raise ValueError("integer root failed")
    return x


def root_frac(q, n):
    q = Fr(q)
    if q <= 0:
        raise ValueError("root of a nonpositive number")
    p, d = q.numerator, q.denominator
    scale = 10 ** ROOT_DIGITS
    r = inroot(p * d ** (n - 1) * scale ** n, n)
    den = d * scale
    return Iv(Iv.frac(Fr(r, den)).lo, Iv.frac(Fr(r + 1, den)).hi)


def pi_interval():
    def atan_inv(n, terms):
        x = Fr(1, n)
        partial, sums = Iv(ZERO), []
        for m in range(terms + 2):
            t = Iv.frac(x ** (2 * m + 1) / (2 * m + 1))
            partial = partial + t if m % 2 == 0 else partial - t
            sums.append(partial)
        return hull(sums[-1], sums[-2])
    return atan_inv(5, 44) * 16 - atan_inv(239, 22) * 4


PI = pi_interval()
SQRTPI = PI.sqrt()
SQRT2 = Iv.frac(2).sqrt()
SQRT3 = Iv.frac(3).sqrt()
SQRT6 = Iv.frac(6).sqrt()
SQRT2PI = (2 * PI).sqrt()
TINY = Decimal(10) ** (-PREC - 8)


def erf_point(x):
    """erf(x) at a Decimal point by the positive series (2/sqrt pi) e^{-x^2} sum 2^n x^{2n+1}/(2n+1)!!; once the term ratio
    2x^2/(2n+3) is at most 1/2 the omitted tail is at most twice the first omitted term."""
    if x < 0:
        return -erf_point(-x)
    if x == 0:
        return Iv(ZERO)
    X = Iv(x)
    x2 = X * X
    t, S, n = X, Iv(ZERO), 0
    while True:
        S = S + t
        ratio = x2 * 2 / (2 * n + 3)
        t = t * ratio
        n += 1
        if ratio.hi <= HALF and t.hi < S.lo * TINY:
            break
        if n > 200000:
            raise RuntimeError("erf series did not converge")
    return (S + Iv(ZERO, CC.multiply(Decimal(2), t.hi))) * (-x2).exp() * 2 / SQRTPI


def erf_iv(x):
    return Iv(erf_point(x.lo).lo, erf_point(x.hi).hi)


def gamma_lower_point(x, a):
    """gamma(a, x) by the positive series x^a e^-x sum_n x^n/(a (a+1) ... (a+n)); a = 7/6, x a Decimal > 0."""
    X = Iv(x)
    t, S, n = Iv.frac(1 / a), Iv(ZERO), 0
    while True:
        S = S + t
        n += 1
        ratio = X / Iv.frac(a + n)
        t = t * ratio
        if ratio.hi <= HALF and t.hi < S.lo * TINY:
            break
    xa = X * root_frac(Fr(x), 6)
    return xa * (-X).exp() * (S + Iv(ZERO, CC.multiply(Decimal(2), t.hi)))


def gamma_complete(a, X=Decimal(120)):
    """Gamma(a) = gamma(a, X) + Gamma(a, X), 1 < a < 2, X^(a-1) e^-X <= Gamma(a, X) <= X^(a-1) e^-X / (1 - (a-1)/X)."""
    low = gamma_lower_point(X, a)
    up = root_frac(Fr(X), 6) * (-Iv(X)).exp()
    return low + Iv(up.lo, (up / (1 - Iv.frac(a - 1) / Iv(X))).hi)


GAMMA76 = gamma_complete(Fr(7, 6))
CBRT32 = root_frac(Fr(12), 3) / 2                       # (3/2)^(1/3)


def gamma_half_integer(twice):
    """Gamma(twice/2) for a positive integer `twice`, exactly in terms of sqrt pi."""
    if twice % 2 == 0:
        return Iv.frac(math.factorial(twice // 2 - 1))
    k = (twice - 1) // 2                                   # Gamma(k + 1/2) = (2k)! sqrt pi / (4^k k!)
    return Iv.frac(Fr(math.factorial(2 * k), 4 ** k * math.factorial(k))) * SQRTPI


def mehta_Z(m):
    """Z_m = int_{R^m} prod_{i<j} |l_i - l_j| e^{-|l|^2/4} dl (unordered); Mehta's integral at beta = 1 after l = sqrt2 x."""
    if MUT == "mehta":
        m_eff = m + 1 if m == 4 else m
    else:
        m_eff = m
    prod = Iv(ONE)
    for j in range(1, m_eff + 1):
        prod = prod * gamma_half_integer(j + 2)             # Gamma(1 + j/2)
    pw = Fr(m_eff, 2) + Fr(m_eff * (m_eff - 1), 4)         # exponent of 2
    two_pow = Iv.frac(2 ** int(pw)) * (SQRT2 if pw.denominator == 2 else Iv(ONE))
    twopi_pow = ipow(2 * PI, m_eff // 2) * (SQRT2PI if m_eff % 2 else Iv(ONE))
    return two_pow * twopi_pow * prod / ipow(gamma_half_integer(3), m_eff)


def sphere_area(d):
    """|S^{d-1}| = 2 pi^{d/2} / Gamma(d/2)."""
    if MUT == "sphere-area" and d == 5:
        return 4 * PI * PI
    pi_pow = ipow(PI, d // 2) * (SQRTPI if d % 2 else Iv(ONE))
    return 2 * pi_pow / gamma_half_integer(d)


# ---------------- exact polynomial algebra (dict of exponent tuples -> Fraction)
def padd(u, v):
    w = dict(u)
    for k, c in v.items():
        w[k] = w.get(k, Fr(0)) + c
    return {k: c for k, c in w.items() if c != 0}


def pmul(u, v):
    w = {}
    for k1, c1 in u.items():
        for k2, c2 in v.items():
            k = tuple(a + b for a, b in zip(k1, k2))
            w[k] = w.get(k, Fr(0)) + c1 * c2
    return {k: c for k, c in w.items() if c != 0}


def pscale(u, c):
    return {k: v * c for k, v in u.items() if v * c != 0}


def pconst(c, nvars):
    return {(0,) * nvars: Fr(c)} if c != 0 else {}


def pvar(i, nvars):
    e = [0] * nvars
    e[i] = 1
    return {tuple(e): Fr(1)}


def ppow(u, n, nvars):
    r = pconst(1, nvars)
    for _ in range(n):
        r = pmul(r, u)
    return r


def psubst(u, images, nvars_out):
    """Substitute variable i -> images[i] (polynomials in nvars_out variables)."""
    out = {}
    for k, c in u.items():
        term = pconst(c, nvars_out)
        for i, e in enumerate(k):
            if e:
                term = pmul(term, ppow(images[i], e, nvars_out))
        out = padd(out, term)
    return out


def univariate(u):
    """dict {(n,): c} -> list of Fractions by power."""
    if not u:
        return [Fr(0)]
    deg = max(k[0] for k in u)
    return [u.get((n,), Fr(0)) for n in range(deg + 1)]


def reduce_sector(m):
    """The m-dimensional sector integrand in (a, p_1, ..., p_{m-1}) with mu_m = -a, mu_{m-1} = -a - p_1, ...:
    returns (polynomial, quadratic form) with polynomial = prod mu_i^2 prod_{i<j} (mu_j - mu_i) and
    quadratic form = |mu|^2/4 - S^2/(4(m+3)) as dicts in m variables (a first)."""
    nv = m
    a = pvar(0, nv)
    mus = []                                               # mu_m, mu_{m-1}, ..., mu_1 (each = -(a + p_1 + ...))
    acc = a
    for i in range(m):
        if i > 0:
            acc = padd(acc, pvar(i, nv))
        mus.append(pscale(acc, Fr(-1)))
    poly = pconst(1, nv)
    for mu in mus:
        poly = pmul(poly, pmul(mu, mu))
    if MUT != "vandermonde":
        for i in range(m):
            for j in range(i + 1, m):
                poly = pmul(poly, padd(mus[i], pscale(mus[j], Fr(-1))))      # mu_(later index) is larger: mus[i] - mus[j] > 0
    S = {}
    norm2 = {}
    for mu in mus:
        S = padd(S, mu)
        norm2 = padd(norm2, pmul(mu, mu))
    shift = Fr(1, 4 * (m + 3)) if MUT != "shift-variance" or m != 4 else Fr(1, 24)
    quad = padd(pscale(norm2, Fr(1, 4)), pscale(pmul(S, S), -shift))
    return poly, quad


# ---------------- Taylor-model quadrature with Cauchy remainders
class Factor:
    """An entire factor g(s): point Taylor coefficients at a rational centre s0 (list of K+1 intervals) and a bound M on
    |g(z)| for |z - s0| <= rho."""

    def coefficients(self, s0, K):
        raise NotImplementedError

    def disc_bound(self, s0, rho):
        raise NotImplementedError


class Gauss(Factor):
    """exp(-gamma s^2)."""

    def __init__(self, gamma):
        self.gamma = Fr(gamma)

    def coefficients(self, s0, K):
        g = Iv.frac(self.gamma)
        S0 = Iv.frac(s0)
        y = [(-(g * S0.sq())).exp()]
        prev = Iv(ZERO)
        for n in range(K):
            nxt = -(2 * g) * (S0 * y[n] + prev) / (n + 1)   # (n+1) y_{n+1} = -2 gamma (s0 y_n + y_{n-1})
            prev = y[n]
            y.append(nxt)
        return y

    def disc_bound(self, s0, rho):
        g = Iv.frac(self.gamma)
        inner = max(Fr(0), abs(Fr(s0)) - Fr(rho))
        return (g * Iv.frac(rho * rho) - g * Iv.frac(inner * inner)).exp()   # |e^{-g z^2}| = e^{-g(x^2 - y^2)}


class ErfLin(Factor):
    """erf(kappa s) (sign +1) or erfc(kappa s) (sign -1); kappa an interval."""

    def __init__(self, kappa, sign):
        self.kappa, self.sign = kappa, sign

    def coefficients(self, s0, K):
        k = self.kappa
        S0 = Iv.frac(s0)
        e = erf_iv(k * S0)
        c0 = e if self.sign > 0 else 1 - e
        # derivative of erf(kappa s) is (2 kappa/sqrt pi) exp(-kappa^2 s^2): coefficients of the Gaussian in s
        k2 = k.sq()
        y = [(-(k2 * S0.sq())).exp()]
        prev = Iv(ZERO)
        for n in range(K):
            nxt = -(2 * k2) * (S0 * y[n] + prev) / (n + 1)
            prev = y[n]
            y.append(nxt)
        pref = 2 * k / SQRTPI * self.sign
        return [c0] + [pref * y[n] / (n + 1) for n in range(K)]

    def disc_bound(self, s0, rho):
        k = self.kappa
        seg = 2 * k / SQRTPI * Iv.frac(abs(Fr(s0)) + Fr(rho)) * (k.sq() * Iv.frac(Fr(rho) ** 2)).exp()
        return seg if self.sign > 0 else 1 + seg


class Poly(Factor):
    """Polynomial with rational coefficients (list by power)."""

    def __init__(self, coeffs):
        self.c = list(coeffs)

    def coefficients(self, s0, K):
        # Taylor coefficients at s0: sum_j c_j C(j, n) s0^(j-n)
        out = []
        for n in range(K + 1):
            v = Fr(0)
            for j, cj in enumerate(self.c):
                if j >= n:
                    v += cj * math.comb(j, n) * Fr(s0) ** (j - n)
            out.append(Iv.frac(v))
        return out

    def disc_bound(self, s0, rho):
        r = abs(Fr(s0)) + Fr(rho)
        return Iv.frac(sum(abs(cj) * r ** j for j, cj in enumerate(self.c)))


class Jk(Factor):
    """J_k(s) = int_0^s q^k e^{-q^2/8} dq: J_0 = sqrt(2 pi) erf(s/(2 sqrt2)), J_1 = 4 (1 - e^{-s^2/8}),
    J_{k+2}(s) = -4 s^{k+1} e^{-s^2/8} + 4 (k+1) J_k(s)."""

    def __init__(self, k):
        self.k = k

    _cache = {}

    def coefficients(self, s0, K):
        key = (s0, K)
        chain = Jk._cache.get(key)
        if chain is None or len(chain) <= self.k:
            E3 = Gauss(Fr(1, 8)).coefficients(s0, K)
            J0 = [c * SQRT2PI for c in ErfLin(1 / (2 * SQRT2), +1).coefficients(s0, K)]
            J1 = [Iv.frac(4) - 4 * E3[0]] + [-4 * c for c in E3[1:]]
            chain = [J0, J1]
            for k in range(max(self.k, 9) - 1):
                sp = Poly([Fr(0)] * (k + 1) + [Fr(1)]).coefficients(s0, K)     # s^{k+1}
                chain.append(t_add(t_scale(t_mul(sp, E3), Fr(-4)), t_scale(chain[k], Fr(4 * (k + 1)))))
            Jk._cache = {key: chain}
        return chain[self.k]

    def infinity(self):
        """J_k(inf) = (1/2) 8^{(k+1)/2} Gamma((k+1)/2), as an interval."""
        k = self.k
        eight = Iv.frac(8 ** ((k + 1) // 2)) * (SQRT2 * 2 if (k + 1) % 2 else Iv(ONE))
        return eight * gamma_half_integer(k + 1) / 2

    def disc_bound(self, s0, rho):
        r = Iv.frac(abs(Fr(s0)) + Fr(rho))
        return ipow(r, self.k + 1) * (Iv.frac(Fr(rho) ** 2 / 8)).exp() / (self.k + 1)


def t_mul(u, v):
    K = len(u) - 1
    out = []
    for n in range(K + 1):
        acc = Iv(ZERO)
        for i in range(n + 1):
            acc = acc + u[i] * v[n - i]
        out.append(acc)
    return out


def t_add(u, v):
    return [x + y for x, y in zip(u, v)]


def t_scale(u, c):
    return [x * c for x in u]


def enclose_integral(branches, K=None, cell=None, T=None, rho=None, remainder=True):
    """int_0^T of sum over branches of const * prod(prefactors) * sum_pairs poly * J, plus the tail bound.  A branch is
    (const Iv, [prefactor factors], [(Poly, Jk or None)], tail_poly, gamma): for s >= T the branch is bounded in absolute
    value by |const| tail_poly(s) e^{-gamma s^2} (tail_poly a rational polynomial by power).  On each cell the point Taylor
    coefficients of the product are integrated exactly; the tail sum_{n > K} c_n x^n of the (everywhere convergent) Taylor
    series at the centre is bounded through Cauchy's estimate at the centre, |c_n| <= M_F(rho)/rho^n, M_F the product/sum of
    the factors' disc bounds, and the geometric series in |x|/rho <= h/(2 rho) (v1.1: the v1.0 text applied the estimate
    off-centre without shrinking the radius; Codex 4148001628)."""
    K = K_ORDER if K is None else K
    cell = CELL if cell is None else cell
    T = T_CUT if T is None else T
    rho = RHO if rho is None else rho
    ncell = int(T / cell)
    if ncell * cell != T:
        raise ValueError("cell does not divide T")
    half = cell / 2
    total = Iv(ZERO)
    even_moments = [Iv.frac(2 * half ** (n + 1) / (n + 1)) if n % 2 == 0 else None for n in range(K + 1)]
    # int_{-h/2}^{h/2} sum_{n>K} |c_n| |x|^n dx <= M_F sum_{n>K} rho^-n 2 (h/2)^(n+1)/(n+1)
    #                                          <= M_F rho^-(K+1) 2 (h/2)^(K+2)/(K+2) . 1/(1 - h/(2 rho))
    if Fr(half) >= Fr(rho):
        raise ValueError("cell half-width must be below the Cauchy radius")
    rem_moment = Iv.frac(2 * half ** (K + 2) / (K + 2) / (1 - Fr(half) / Fr(rho)))
    rho_pow = Iv.frac(Fr(rho) ** (K + 1))
    for c in range(ncell):
        s0 = (2 * c + 1) * half
        cell_val = Iv(ZERO)
        M_F = Iv(ZERO)
        for const, prefactors, pairs, _, _ in branches:
            acc, M_acc = None, Iv(ZERO)
            for poly, jk in pairs:
                pc = poly.coefficients(s0, K)
                M_pair = poly.disc_bound(s0, rho)
                if jk is not None:
                    pc = t_mul(pc, jk.coefficients(s0, K))
                    M_pair = M_pair * jk.disc_bound(s0, rho)
                acc = pc if acc is None else t_add(acc, pc)
                M_acc = M_acc + M_pair
            M = M_acc
            for f in prefactors:
                acc = t_mul(acc, f.coefficients(s0, K))
                M = M * f.disc_bound(s0, rho)
            for n in range(0, K + 1, 2):
                cell_val = cell_val + const * acc[n] * even_moments[n]
            M_F = M_F + Iv(const.mag()) * M
        if remainder:
            R = (M_F / rho_pow * rem_moment).hi
            cell_val = cell_val + Iv(-R, R)
        total = total + cell_val
    # tail: int_T^inf s^j e^{-gamma s^2} ds = (1/2) gamma^{-(j+1)/2} Gamma((j+1)/2, gamma T^2),
    # Gamma(a, x) <= x^(a-1) e^-x / (1 - (a-1)/x) for a > 1, <= x^(a-1) e^-x for a <= 1
    tail = Iv(ZERO)
    for const, _, _, tail_poly, gamma in branches:
        g = Iv.of(gamma)
        x = g * Iv.frac(T * T)
        for j, cj in enumerate(tail_poly):
            if cj == 0:
                continue
            a = Fr(j + 1, 2)
            if a.denominator == 2:
                xa1 = ipow(x, int(a - Fr(1, 2))) * x.sqrt() / x
                gpow = ipow(g, int(a)) * g.sqrt()
            else:
                xa1 = ipow(x, int(a) - 1)
                gpow = ipow(g, int(a))
            up = xa1 * (-x).exp()
            if a > 1:
                up = up / (1 - Iv.frac(a - 1) / x)
            tail = tail + Iv(const.mag()) * Iv.frac(abs(cj)) * up / gpow / 2
    return total + Iv(-tail.hi, tail.hi)



### REDUCTIONS
def a_integrated_terms(m):
    """For m = 2 and m = 3: integrate the sector polynomial times e^{-Q} over a in (0, inf) exactly; returns the erfc-branch
    and polynomial-branch integrands as polynomials in the remaining variables with their (decoupled) exponents."""
    poly, quad = reduce_sector(m)
    nv = m
    # quad = alpha a^2 + a * beta(p) + rest(p)
    alpha = quad.get((2,) + (0,) * (nv - 1), Fr(0))
    beta = {k[1:]: c for k, c in quad.items() if k[0] == 1}
    rest = {k[1:]: c for k, c in quad.items() if k[0] == 0}
    # G_n = P_n(beta) G_0 + R_n(beta), G_0 = (1/2) sqrt(pi/alpha) e^{beta^2/(4 alpha)} erfc(beta/(2 sqrt alpha))
    nvr = nv - 1
    P = [pconst(1, nvr), pscale(beta, Fr(-1, 2) / alpha)]
    R = [{}, pconst(Fr(1, 2) / alpha, nvr)]
    deg_a = max(k[0] for k in poly)
    for n in range(1, deg_a):
        P.append(pscale(padd(pscale(P[n - 1], Fr(n)), pscale(pmul(beta, P[n]), Fr(-1))), Fr(1, 2) / alpha))
        R.append(pscale(padd(pscale(R[n - 1], Fr(n)), pscale(pmul(beta, R[n]), Fr(-1))), Fr(1, 2) / alpha))
    if MUT == "recurrence":
        P[3] = pscale(P[3], Fr(101, 100))
    PE, PR = {}, {}
    for k, c in poly.items():
        n = k[0]
        w = {k[1:]: c}
        PE = padd(PE, pmul(w, P[n]))
        PR = padd(PR, pmul(w, R[n]))
    # exponents: erfc branch  -(rest - beta^2/(4 alpha)),  polynomial branch  -rest
    expE = padd(rest, pscale(pmul(beta, beta), Fr(-1, 4) / alpha))
    return alpha, beta, PE, PR, expE, rest


def d2_terms():
    """m = 2: one remaining variable p; erfc branch e^{-p^2/8} erfc(p sqrt30/20), polynomial branch e^{-p^2/5}."""
    alpha, beta, PE, PR, expE, rest = a_integrated_terms(2)
    if set(expE) - {(2,)} or set(rest) - {(2,)} or set(beta) - {(1,)}:
        raise ValueError("m = 2 structure")
    gE, gR, b1 = expE[(2,)], rest[(2,)], beta[(1,)]
    kappa = Iv.frac(b1) / (2 * Iv.frac(alpha).sqrt())                            # erfc argument slope
    constE = Iv.frac(Fr(1, 2)) * (PI / Iv.frac(alpha)).sqrt()
    pref = (Iv.frac(Fr(3, 5))).sqrt() / mehta_Z(2) * 2                           # sqrt(3/(m+3)) . (m! sectors) / Z_m
    pe, pr = univariate(PE), univariate(PR)
    tE = (pref * constE, [Gauss(gE), ErfLin(kappa, -1)], [(Poly(pe), None)], [2 * abs(c) for c in pe], gE)
    tR = (pref, [Gauss(gR)], [(Poly(pr), None)], [abs(c) for c in pr], gR)
    return [tE, tR]


def d3_terms():
    """m = 3: remaining (p, q); substitute p = (s - q)/2 (Jacobian 1/2), integrate q over (0, s) exactly through J_k."""
    alpha, beta, PE, PR, expE, rest = a_integrated_terms(3)
    s, q = pvar(0, 2), pvar(1, 2)
    p_img = pscale(padd(s, pscale(q, Fr(-1))), Fr(1, 2))
    images = [p_img, q]
    PEs, PRs = psubst(PE, images, 2), psubst(PR, images, 2)
    eE, eR, bs = psubst(expE, images, 2), psubst(rest, images, 2), psubst(beta, images, 2)
    # the decoupling: exponents must be diagonal in (s, q) with q^2 coefficient 1/8, and beta must be a multiple of s
    if set(eE) != {(2, 0), (0, 2)} or set(eR) != {(2, 0), (0, 2)} or eE[(0, 2)] != Fr(1, 8) or eR[(0, 2)] != Fr(1, 8) or set(bs) != {(1, 0)}:
        raise ValueError("decoupling failed: " + repr((eE, eR, bs)))
    gE, gR = eE[(2, 0)], eR[(2, 0)]
    kappa = Iv.frac(bs[(1, 0)]) / (2 * Iv.frac(alpha).sqrt())
    constE = Iv.frac(Fr(1, 2)) * (PI / Iv.frac(alpha)).sqrt()
    pref = (Iv.frac(Fr(1, 2))).sqrt() / mehta_Z(3) * 6 / 2                     # sqrt(3/6) . 3! / Z_3 . Jacobian 1/2
    branches = []
    def by_q_power(Pxy):
        out = {}
        for (i, j), c in Pxy.items():
            out.setdefault(j, {})[(i,)] = c
        return out
    for branch, Pxy, g, const, extra in (("E", PEs, gE, pref * constE, [ErfLin(kappa, -1)]), ("R", PRs, gR, pref, [])):
        pairs, tail_poly = [], {}
        for k, coef_poly in sorted(by_q_power(Pxy).items()):
            ek = univariate(coef_poly)
            jk = Jk(k)
            pairs.append((Poly(ek), jk))
            # tail bound for s >= T: |J_k(s)| <= J_k(inf), erfc <= 2; rational upper bound of J_k(inf) from its interval
            jk_bound = Fr(jk.infinity().hi) * (2 if branch == "E" else 1)
            for j, c in enumerate(ek):
                tail_poly[j] = tail_poly.get(j, Fr(0)) + abs(c) * jk_bound
        tp = [tail_poly.get(j, Fr(0)) for j in range(max(tail_poly) + 1)]
        branches.append((const, [Gauss(g)] + extra, pairs, tp, g))
    return branches



### LIBRARY
def library_exact():
    ok = True
    for q in (Fr(3), Fr(120), Fr(1, 2), Fr(288), Fr(5, 7), Fr(48) - Fr(51035039649098175696, 10 ** 29)):
        g = -q / 64
        parts, term, ssum = [], Fr(1), Fr(0)
        for k in range(0, 60):
            ssum += term
            parts.append(ssum)
            term = term * g / (k + 1)
        lo, hi = min(parts[-1], parts[-2]), max(parts[-1], parts[-2])
        exact = Iv(Iv.frac(lo ** 64).lo, Iv.frac(hi ** 64).hi)
        X = Iv.frac(q)
        got = (-X).exp()
        ok &= got.intersects(exact) and CC.divide(got.width(), got.lo) < Decimal(10) ** (-PREC + 3)
        r = root_frac(q, 2)
        ok &= X.sqrt().intersects(r) and CC.divide(X.sqrt().width(), r.lo) < Decimal(10) ** (-PREC + 3)
        ok &= (-X).lo == X.hi.copy_negate() and (-X).hi == X.lo.copy_negate()
    return ok


