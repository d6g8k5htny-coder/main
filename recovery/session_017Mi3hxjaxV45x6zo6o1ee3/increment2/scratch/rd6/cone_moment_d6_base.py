"""HEADER_PLACEHOLDER"""
import argparse
import itertools
import json
import math
import sys
import time
from decimal import Decimal, Context, ROUND_FLOOR, ROUND_CEILING, ROUND_HALF_EVEN, setcontext
from fractions import Fraction as Fr
from functools import lru_cache

MUTANTS = ("shift-variance", "vandermonde", "erfc-branch", "jacobian", "mehta", "recurrence", "layer-init", "two-erf",
           "cumulative-half", "remainder-dropped", "erf-inside", "tail-dropped", "image-shells", "sandwich-exponent")
MUT = None

PREC = 60
CF = Context(prec=PREC, rounding=ROUND_FLOOR, Emin=-9999999, Emax=9999999)
CC = Context(prec=PREC, rounding=ROUND_CEILING, Emin=-9999999, Emax=9999999)
CE = Context(prec=PREC, rounding=ROUND_HALF_EVEN, Emin=-9999999, Emax=9999999)
setcontext(Context(prec=2 * PREC + 20, rounding=ROUND_HALF_EVEN, Emin=-9999999, Emax=9999999))
ZERO, ONE, HALF = Decimal(0), Decimal(1), Decimal("0.5")
ROOT_DIGITS = 70
K_ORDER, RHO = 40, Fr(6)
CELLS = [(Fr(i, 2), Fr(i + 1, 2)) for i in range(0, 60)] + [(Fr(i), Fr(i + 1)) for i in range(30, 90)]
# constants and pins are defined in the packet section below

# ---------------- interval arithmetic over decimal (the Math-#199/#200/#201/#202 toolkit, unchanged)
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


def atan_iv(a, N=90):
    """arctan(a) for an interval 0 <= a <= 1: halve the argument, then the alternating series bracketed by consecutive
    partial sums (for each fixed argument the terms decrease, so the value lies between them)."""
    if a.lo < 0 or a.hi > 1:
        raise ValueError("atan_iv needs 0 <= a <= 1")
    a1 = a / (1 + (1 + a.sq()).sqrt())
    a1sq, p, S, prev = a1.sq(), a1, Iv(ZERO), None
    for n in range(N + 1):
        t = p / (2 * n + 1)
        prev = S
        S = S + t if n % 2 == 0 else S - t
        p = p * a1sq
    return 2 * hull(prev, S)


def gamma_half_integer(twice):
    """Gamma(twice/2) for a positive integer `twice`, exactly in terms of sqrt pi."""
    if twice % 2 == 0:
        return Iv.frac(math.factorial(twice // 2 - 1))
    k = (twice - 1) // 2                                   # Gamma(k + 1/2) = (2k)! sqrt pi / (4^k k!)
    return Iv.frac(Fr(math.factorial(2 * k), 4 ** k * math.factorial(k))) * SQRTPI



# ---------------- exact arithmetic: surds, and the Q-span of sqrt(r) pi^(k/2) ATOM (Math-#201/#202)
def sqf(n):
    """n a positive integer -> (k, r) with sqrt(n) = k sqrt(r), r squarefree."""
    k, r, p, m = 1, 1, 2, n
    while p * p <= m:
        while m % (p * p) == 0:
            m //= p * p
            k *= p
        if m % p == 0:
            m //= p
            r *= p
        p += 1
    return k, r * m


def surd_mul(u, v):
    """(c1, r1) (c2, r2) with (c, r) = c sqrt r."""
    k, r = sqf(u[1] * v[1])
    return (u[0] * v[0] * k, r)


def surd_sqrt(q):
    """sqrt of a positive rational as a surd."""
    q = Fr(q)
    if q <= 0:
        raise ValueError("sqrt of a nonpositive rational")
    k, r = sqf(q.numerator * q.denominator)
    return (Fr(k, q.denominator), r)


def surd_inv(u):
    return (1 / (u[0] * u[1]), u[1])


def surd_sq(u):
    return u[0] * u[0] * u[1]


ATOMS = [None]                      # index 0 is the number 1; index i > 0 is arctan(c sqrt r) for ATOMS[i] = (c, r), c sqrt r <= 1


def E(c=1, r=1, twok=0, atom=0):
    """The element c sqrt(r) pi^(twok/2) ATOMS[atom]."""
    return {(r, twok, atom): Fr(c)} if c != 0 else {}


def add(u, v):
    w = dict(u)
    for k, c in v.items():
        w[k] = w.get(k, Fr(0)) + c
    return {k: c for k, c in w.items() if c != 0}


def scale(u, c):
    c = Fr(c)
    return {k: v * c for k, v in u.items() if v * c != 0}


def mul_surd(u, s):
    out = {}
    for (r, twok, atom), c in u.items():
        k, r2 = sqf(r * s[1])
        key = (r2, twok, atom)
        out[key] = out.get(key, Fr(0)) + c * s[0] * k
    return {k: c for k, c in out.items() if c != 0}


def mul_pi_half(u, j):
    """multiply by pi^(j/2)."""
    return {(r, twok + j, atom): c for (r, twok, atom), c in u.items()}


def mul(u, v):
    out = {}
    for (r1, k1, a1), c1 in u.items():
        for (r2, k2, a2), c2 in v.items():
            if a1 and a2:
                raise ValueError("product of two arctan atoms is outside the span")
            k, r = sqf(r1 * r2)
            key = (r, k1 + k2, a1 or a2)
            out[key] = out.get(key, Fr(0)) + c1 * c2 * k
    return {k: c for k, c in out.items() if c != 0}


def inv_monomial(u):
    (r, twok, atom), c = next(iter(u.items()))
    if len(u) != 1 or atom:
        raise ValueError("only monomials without atoms are inverted")
    return E(1 / (c * r), r, -twok)


def atan_elem(t):
    """arctan(c sqrt r) as an element: rational multiples of pi are recognised (arguments 1, sqrt3, 1/sqrt3), an argument
    above 1 is reflected (arctan t = pi/2 - arctan(1/t)), anything else becomes an atom with argument in (0, 1]."""
    c, r = t
    if c == 0:
        return {}
    if c < 0:
        return scale(atan_elem((-c, r)), -1)
    if surd_sq(t) > 1:
        return add(E(Fr(1, 2), 1, 2), scale(atan_elem(surd_inv(t)), -1))
    if (c, r) == (1, 1):
        return E(Fr(1, 4), 1, 2)
    if r == 3 and c == Fr(1, 3):
        return E(Fr(1, 6), 1, 2)
    for i, a in enumerate(ATOMS):
        if a == (c, r):
            return E(1, 1, 0, i)
    ATOMS.append((c, r))
    return E(1, 1, 0, len(ATOMS) - 1)


def iv_eval(u):
    """Interval evaluation of an element (60 digits)."""
    total = Iv(ZERO)
    for (r, twok, atom), c in u.items():
        term = Iv.frac(c)
        if r != 1:
            term = term * root_frac(Fr(r), 2)
        if twok:
            pp = ipow(PI, abs(twok) // 2) * (SQRTPI if twok % 2 else Iv(ONE))
            term = term * pp if twok > 0 else term / pp
        if atom:
            ca, ra = ATOMS[atom]
            arg = Iv.frac(ca) * (root_frac(Fr(ra), 2) if ra != 1 else Iv(ONE))
            term = term * atan_iv(arg)
        total = total + term
    return total


def fl_eval(u):
    """Float evaluation of an element (controls only)."""
    v = 0.0
    for (r, twok, atom), c in u.items():
        t = float(c) * math.sqrt(r) * math.pi ** (twok / 2)
        if atom:
            ca, ra = ATOMS[atom]
            t *= math.atan(float(ca) * math.sqrt(ra))
        v += t
    return v


def show(u):
    parts = []
    for (r, twok, atom) in sorted(u, key=lambda k: (k[2], -k[1], k[0])):
        c = u[(r, twok, atom)]
        s = str(c)
        if r != 1:
            s += " sqrt(%d)" % r
        if twok:
            s += " pi^(%s)" % Fr(twok, 2)
        if atom:
            ca, ra = ATOMS[atom]
            s += " arctan(%s%s)" % (ca, "" if ra == 1 else " sqrt(%d)" % ra)
        parts.append(s)
    return "  +  ".join(parts) if parts else "0"


@lru_cache(maxsize=None)
def M(n, lam, slopes):
    lam = Fr(lam)
    if n == 0:
        inv_sqrt_lam = surd_sqrt(1 / lam)
        if not slopes:                                                             # (1/2) sqrt(pi/lam)
            return mul_pi_half(mul_surd(E(Fr(1, 2)), inv_sqrt_lam), 1)
        if len(slopes) == 1:                                                       # arctan(u/sqrt lam)/sqrt(pi lam)
            u = slopes[0]
            t = surd_mul(u, inv_sqrt_lam) 
            return mul_pi_half(mul_surd(atan_elem(t), inv_sqrt_lam), -1)
        if len(slopes) == 2:                                                       # arctan(uv/sqrt(lam(lam+u^2+v^2)))/sqrt(pi lam)
            u, v = slopes
            den = surd_sqrt(lam * (lam + surd_sq(u) + surd_sq(v))) if MUT != "two-erf" else surd_sqrt(lam * (lam + surd_sq(u)))
            t = surd_mul(surd_mul(u, v), surd_inv(den))
            return mul_pi_half(mul_surd(atan_elem(t), inv_sqrt_lam), -1)
        raise ValueError("at most two erf factors")
    if n == 1 and not slopes:
        return E(1 / (2 * lam))
    # by parts: s^n e^{-lam s^2} f = s^{n-1} f . s e^{-lam s^2};  the boundary term vanishes (n >= 2, or f(0) = 0)
    out = {}
    if n >= 2:
        out = add(out, scale(M(n - 2, lam, slopes), n - 1))
    for i, u in enumerate(slopes):
        rest = slopes[:i] + slopes[i + 1:]
        term = M(n - 1, lam + surd_sq(u), rest)                                   # f' contributes (2u/sqrt pi) e^{-u^2 s^2}
        out = add(out, mul_pi_half(mul_surd(term, (2 * u[0], u[1])), -1))
    return scale(out, 1 / (2 * lam))


def gauss_moment_closed(n, lam):
    """(1/2) Gamma((n+1)/2) lam^{-(n+1)/2} as an element (the direct closed form, used only by rule GAUSS_MOMENTS)."""
    lam = Fr(lam)
    g = gamma_half(n + 1)
    if (n + 1) % 2 == 0:
        return scale(mul(g, E(Fr(1) / lam ** ((n + 1) // 2))), Fr(1, 2))
    return scale(mul_surd(mul(g, E(Fr(1) / lam ** (n // 2))), surd_sqrt(1 / lam)), Fr(1, 2))



# ---------------- exact polynomial algebra and the sector reduction (Math-#199/#200)
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
    out = {}
    for k, c in u.items():
        term = pconst(c, nvars_out)
        for i, e in enumerate(k):
            if e:
                term = pmul(term, ppow(images[i], e, nvars_out))
        out = padd(out, term)
    return out


def univariate(u):
    if not u:
        return [Fr(0)]
    deg = max(k[0] for k in u)
    return [u.get((n,), Fr(0)) for n in range(deg + 1)]


def reduce_sector(m):
    """Ordered sector mu_m = -a, mu_{m-1} = -a - p_1, ...: (prod mu_i^2 prod_{i<j} (mu_j - mu_i), |mu|^2/4 - S^2/(4(m+3)))."""
    nv = m
    a = pvar(0, nv)
    mus, acc = [], a
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
                poly = pmul(poly, padd(mus[i], pscale(mus[j], Fr(-1))))
    S, norm2 = {}, {}
    for mu in mus:
        S = padd(S, mu)
        norm2 = padd(norm2, pmul(mu, mu))
    shift = Fr(1, 4 * (m + 3)) if MUT != "shift-variance" or m != 5 else Fr(1, 28)
    quad = padd(pscale(norm2, Fr(1, 4)), pscale(pmul(S, S), -shift))
    return poly, quad


def a_integrated_terms(m):
    """Exact a-integral: erfc branch PE (times G_0) and polynomial branch PR, with exponents expE and rest."""
    poly, quad = reduce_sector(m)
    nv = m
    alpha = quad.get((2,) + (0,) * (nv - 1), Fr(0))
    beta = {k[1:]: c for k, c in quad.items() if k[0] == 1}
    rest = {k[1:]: c for k, c in quad.items() if k[0] == 0}
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
    expE = padd(rest, pscale(pmul(beta, beta), Fr(-1, 4) / alpha))
    return alpha, beta, PE, PR, expE, rest


def gamma_half(twice):
    """Gamma(twice/2) as an element."""
    if twice % 2 == 0:
        return E(math.factorial(twice // 2 - 1))
    k = (twice - 1) // 2
    return E(Fr(math.factorial(2 * k), 4 ** k * math.factorial(k)), 1, 1)


def mehta_Z(m):
    """Z_m = 2^(m/2 + m(m-1)/4) (2 pi)^(m/2) prod_{j<=m} Gamma(1 + j/2) / Gamma(3/2)^m as an element."""
    m_eff = m + 1 if (MUT == "mehta" and m == 5) else m                          # Z_6 for Z_5
    prod = E(1)
    for j in range(1, m_eff + 1):
        prod = mul(prod, gamma_half(j + 2))
    pw = Fr(m_eff, 2) + Fr(m_eff * (m_eff - 1), 4)
    two_pow = E(2 ** int(pw), 2 if pw.denominator == 2 else 1)
    twopi_pow = mul_pi_half(E(2 ** (m_eff // 2), 2 if m_eff % 2 else 1), m_eff)
    g = E(1)
    for _ in range(m_eff):
        g = mul(g, gamma_half(3))
    return mul(mul(mul(two_pow, twopi_pow), prod), inv_monomial(g))


def prefactor(m, jac):
    """sqrt(3/(m+3)) m! / Z_m times the Jacobian of the final substitution."""
    return mul_surd(scale(inv_monomial(mehta_Z(m)), math.factorial(m) * jac), surd_sqrt(Fr(3, m + 3)))


def jk_exact(kmax):
    """J_k(s) = A_k sqrt(2 pi) erf(s/(2 sqrt2)) + B_k + C_k(s) e^{-s^2/8}: (A_k, B_k, C_k by power) from J_0, J_1 and
    J_{k+2} = -4 s^{k+1} e^{-s^2/8} + 4(k+1) J_k."""
    J = [(Fr(1), Fr(0), []), (Fr(0), Fr(4), [Fr(-4)])]
    for k in range(kmax - 1):
        A, B, C = J[k]
        Cn = [4 * (k + 1) * c for c in C] + [Fr(0)] * max(0, k + 2 - len(C))
        Cn[k + 1] += Fr(-4)
        J.append((4 * (k + 1) * A, 4 * (k + 1) * B, Cn))
    return J


def jk_identities(JK):
    """J_k(0) = 0,  J_k'(s) = s^k e^{-s^2/8}  (A_k + C_k' - (s/4) C_k = s^k),  J_k(inf) = (1/2) 8^{(k+1)/2} Gamma((k+1)/2)."""
    ok = True
    for k, (A, B, C) in enumerate(JK):
        ok &= B + (C[0] if C else 0) == 0
        deriv = {}
        for i, c in enumerate(C):
            if i:
                deriv[i - 1] = deriv.get(i - 1, Fr(0)) + i * c
            deriv[i + 1] = deriv.get(i + 1, Fr(0)) - c / 4
        deriv[0] = deriv.get(0, Fr(0)) + A
        ok &= {i: c for i, c in deriv.items() if c} == {k: Fr(1)}
        at_inf = add(mul_pi_half(E(A, 2), 1), E(B))
        eight = E(8 ** ((k + 1) // 2), 2 if (k + 1) % 2 else 1)
        if (k + 1) % 2:
            eight = scale(eight, 2)                                    # 8^(1/2) = 2 sqrt2
        ok &= at_inf == scale(mul(eight, gamma_half(k + 1)), Fr(1, 2))
    return ok



# ---------------- exact layers: T-functions and the incomplete Gaussian moments (Math-#202)
def tf_norm(A):
    return {k: {n: c for n, c in v.items() if c} for k, v in A.items() if any(v.values())}


def tf_add(A, B):
    out = {k: dict(v) for k, v in A.items()}
    for key, poly in B.items():
        d = out.setdefault(key, {})
        for n, c in poly.items():
            d[n] = add(d.get(n, {}), c)
    return tf_norm(out)


def tf_scale(A, c):
    return tf_norm({k: {n: scale(e, c) for n, e in v.items()} for k, v in A.items()})


_G_CACHE = {}


def G(n, mu):
    """int_0^T v^n e^{-mu v^2} dv:  G_0 = (1/2) sqrt(pi/mu) erf(sqrt(mu) T),  G_1 = (1 - e^{-mu T^2})/(2 mu),
    G_n = [(n-1) G_{n-2} - T^{n-1} e^{-mu T^2}]/(2 mu)  (by parts)."""
    mu = Fr(mu)
    key = (n, mu)
    if key in _G_CACHE:
        return _G_CACHE[key]
    if n == 0:
        val = {("f", mu): {0: mul_pi_half(mul_surd(E(Fr(1, 2)), surd_sqrt(1 / mu)), 1)}}
    elif n == 1:
        val = {("c", Fr(0)): {0: E(1 / (2 * mu))}, ("e", mu): {0: E(-1 / (2 * mu) if MUT != "layer-init" else 1 / (2 * mu))}}
    else:
        val = tf_add(tf_scale(G(n - 2, mu), Fr(n - 1)), {("e", mu): {n - 1: E(-1)}})
        val = tf_scale(val, 1 / (2 * mu))
    _G_CACHE[key] = val
    return val


def tf_derivative(A):
    """d/dT of a T-function: (p e^{-mu T^2})' = (p' - 2 mu T p) e^{-mu T^2};  (p erf(sqrt mu T))' = p' erf + p (2 sqrt(mu)/sqrt pi) e^{-mu T^2}."""
    out = {}
    for (kind, mu), poly in A.items():
        dp = {}
        for n, c in poly.items():
            if n:
                dp[n - 1] = add(dp.get(n - 1, {}), scale(c, n))
        if kind == "c":
            out = tf_add(out, {("c", Fr(0)): dp})
        elif kind == "e":
            shifted = {n + 1: scale(c, -2 * mu) for n, c in poly.items()}
            out = tf_add(out, {("e", mu): dp})
            out = tf_add(out, {("e", mu): shifted})
        else:
            pref = mul_pi_half(mul_surd(E(2), surd_sqrt(mu)), -1)                   # 2 sqrt(mu)/sqrt(pi)
            out = tf_add(out, {("f", mu): dp})
            out = tf_add(out, {("e", mu): {n: mul(c, pref) for n, c in poly.items()}})
    return tf_norm(out)


def tf_at_zero(A):
    """Value at T = 0 (erf(0) = 0)."""
    val = {}
    for (kind, mu), poly in A.items():
        if kind != "f" and 0 in poly:
            val = add(val, poly[0])
    return val


def g_identities(nmax, mus):
    """G_n(mu, 0) = 0 and G_n' = T^n e^{-mu T^2}, exactly, for n <= nmax."""
    ok = True
    for mu in mus:
        for n in range(nmax + 1):
            ok &= tf_at_zero(G(n, mu)) == {}
            ok &= tf_derivative(G(n, mu)) == tf_norm({("e", Fr(mu)): {n: E(1)}})
    return ok


def outer_moment(n, g, kind, mu, erfc, kappa):
    """int_0^inf T^n e^{-g T^2} [erfc(kappa T)] {1, e^{-mu T^2}, erf(sqrt(mu) T)} dT as an element."""
    if kind == "c":
        lam, slopes = g, ()
    elif kind == "e":
        lam, slopes = g + mu, ()
    else:
        lam, slopes = g, (surd_sqrt(mu),)
    val = M(n, lam, slopes)
    if erfc:
        val = add(val, scale(M(n, lam, tuple(sorted(slopes + (kappa,)))), -1))
    return val


def D1_exact():
    """D_1 = (sqrt3/2) (1/(2 sqrt pi)) int_0^inf mu^2 e^{-3 mu^2/16} dmu."""
    return mul_pi_half(mul_surd(scale(M(2, Fr(3, 16), ()), Fr(1, 4)), (Fr(1), 3)), -1)


def erfc_moment(n, lam, kappa):
    """int_0^inf s^n e^{-lam s^2} erfc(kappa s) ds = M_n(lam, 1) - M_n(lam, erf(kappa .))."""
    sign = -1
    return add(M(n, lam, ()), scale(M(n, lam, (kappa,)), sign))


def D2_exact():
    """m = 2: one remaining variable p; erfc branch e^{-p^2/8} erfc(p sqrt30/20), polynomial branch e^{-p^2/5}."""
    alpha, beta, PE, PR, expE, rest = a_integrated_terms(2)
    if set(expE) - {(2,)} or set(rest) - {(2,)} or set(beta) - {(1,)}:
        raise ValueError("m = 2 structure")
    gE, gR, b1 = expE[(2,)], rest[(2,)], beta[(1,)]
    kappa = surd_mul((b1 / 2, 1), surd_sqrt(1 / alpha))
    constE = mul_pi_half(mul_surd(E(Fr(1, 2)), surd_sqrt(1 / alpha)), 1)          # (1/2) sqrt(pi/alpha)
    eb, rb = {}, {}
    for n, c in enumerate(univariate(PE)):
        eb = add(eb, scale(erfc_moment(n, gE, kappa), c))
    for n, c in enumerate(univariate(PR)):
        rb = add(rb, scale(M(n, gR, ()), c))
    return mul(prefactor(2, 1), add(mul(constE, eb), rb)), {"kappa": kappa, "lambda_erfc": gE, "lambda_poly": gR}


def D3_exact():
    """m = 3: p = (s - q)/2 (Jacobian 1/2), q over (0, s) through J_k, then the s-moments exactly."""
    alpha, beta, PE, PR, expE, rest = a_integrated_terms(3)
    s, q = pvar(0, 2), pvar(1, 2)
    images = [pscale(padd(s, pscale(q, Fr(-1))), Fr(1, 2)), q]
    PEs, PRs = psubst(PE, images, 2), psubst(PR, images, 2)
    eE, eR, bs = psubst(expE, images, 2), psubst(rest, images, 2), psubst(beta, images, 2)
    if set(eE) != {(2, 0), (0, 2)} or set(eR) != {(2, 0), (0, 2)} or eE[(0, 2)] != Fr(1, 8) or eR[(0, 2)] != Fr(1, 8) or set(bs) != {(1, 0)}:
        raise ValueError("decoupling failed: " + repr((eE, eR, bs)))
    gE, gR = eE[(2, 0)], eR[(2, 0)]
    kappa = surd_mul((bs[(1, 0)] / 2, 1), surd_sqrt(1 / alpha))
    b = (Fr(1, 4), 2)                                                              # 1/(2 sqrt2), the erf slope inside J_even
    constE = mul_pi_half(mul_surd(E(Fr(1, 2)), surd_sqrt(1 / alpha)), 1)

    def by_q_power(Pxy):
        out = {}
        for (i, j), c in Pxy.items():
            out.setdefault(j, {})[(i,)] = c
        return out
    qE, qR = by_q_power(PEs), by_q_power(PRs)
    kmax = max(max(qE), max(qR))
    JK = jk_exact(kmax)
    sqrt2pi = mul_pi_half(E(1, 2), 1)
    polys = {"e": {k: univariate(v) for k, v in sorted(qE.items())}, "r": {k: univariate(v) for k, v in sorted(qR.items())}}

    def branch(Pq, lam, erfc):
        tot = {}
        for k, coef_poly in sorted(Pq.items()):
            A, B, C = JK[k]
            for j, c in enumerate(univariate(coef_poly)):
                if c == 0:
                    continue
                term = {}
                if A:                                                             # A_k sqrt(2 pi) erf(s/(2 sqrt2))
                    x = M(j, lam, (b,))
                    if erfc:
                        x = add(x, scale(M(j, lam, (kappa, b)), -1))
                    term = add(term, scale(mul(sqrt2pi, x), A))
                if B:                                                             # the constant B_k
                    term = add(term, scale(erfc_moment(j, lam, kappa) if erfc else M(j, lam, ()), B))
                for i, ci in enumerate(C):                                        # C_k(s) e^{-s^2/8}
                    if ci:
                        term = add(term, scale(erfc_moment(j + i, lam + Fr(1, 8), kappa) if erfc else M(j + i, lam + Fr(1, 8), ()), ci))
                tot = add(tot, scale(term, c))
        return tot
    jac = Fr(1, 2)
    pref = prefactor(3, jac)
    eb = mul(mul(pref, constE), branch(qE, gE, True))
    rb = mul(pref, branch(qR, gR, False))
    info = {"alpha": alpha, "kappa": kappa, "lambda_erfc": gE, "lambda_poly": gR, "kmax": kmax,
            "q_powers_erfc": sorted(qE), "q_powers_poly": sorted(qR), "polys": polys, "JK": JK,
            "erfc_branch": eb, "poly_branch": rb, "prefactor": pref, "constE": constE}
    return add(eb, rb), info


def organize_tvp(P):
    """Polynomial dict in (T, v, p3) -> {a: {k: [coefficients in v by power]}} (a = T power, k = p3 power)."""
    out = {}
    for (i, j, k), c in P.items():
        out.setdefault(i, {}).setdefault(k, {})[j] = c
    return {a: {k: [d.get(j, Fr(0)) for j in range(max(d) + 1)] for k, d in ks.items()} for a, ks in out.items()}


def integrate_branch(inner, g, erfc, kappa, JK):
    """sum_a int_0^inf T^a e^{-g T^2} [erfc(kappa T)] L_a(T) dT with L_a(T) = int_0^T e^{-v^2/24} sum_k e_{a,k}(v) J_k(v) dv."""
    total = {}
    for a, ks in inner.items():
        La = {}
        for k, ek in ks.items():
            A, B, C = JK[k]
            if A:
                raise ValueError("an even p3-power occurred; J_k would carry an error function")
            for j, c in enumerate(ek):
                if c == 0:
                    continue
                part = tf_scale(G(j, Fr(1, 24)), B) if B else {}
                for i, ci in enumerate(C):
                    if ci:
                        part = tf_add(part, tf_scale(G(j + i, Fr(1, 6)), ci))
                La = tf_add(La, tf_scale(part, c))
        for (kind, mu), poly in La.items():
            for n, coef in poly.items():
                total = add(total, mul(coef, outer_moment(n + a, g, kind, mu, erfc, kappa)))
    return total


def m4_reduction():
    """The m = 4 reduction of Math-#200: p_1 = (T - 2p_2 - p_3)/3, then p_2 = (v - p_3)/2 (Jacobian 1/6); returns the two
    branches organised as {a: {k: e_{a,k}(v)}} with their constants."""
    m = 4
    alpha, beta, PE, PR, expE, rest = a_integrated_terms(m)
    T, p2, p3 = pvar(0, 3), pvar(1, 3), pvar(2, 3)
    p1 = pscale(padd(T, padd(pscale(p2, Fr(-2)), pscale(p3, Fr(-1)))), Fr(1, 3))
    img1 = [p1, p2, p3]
    PE1, PR1 = psubst(PE, img1, 3), psubst(PR, img1, 3)
    eE1, eR1, b1 = psubst(expE, img1, 3), psubst(rest, img1, 3), psubst(beta, img1, 3)
    T2, v, q = pvar(0, 3), pvar(1, 3), pvar(2, 3)
    img2 = [T2, pscale(padd(v, pscale(q, Fr(-1))), Fr(1, 2)), q]
    PE2, PR2 = psubst(PE1, img2, 3), psubst(PR1, img2, 3)
    eE2, eR2, b2 = psubst(eE1, img2, 3), psubst(eR1, img2, 3), psubst(b1, img2, 3)
    diag = {(2, 0, 0), (0, 2, 0), (0, 0, 2)}
    for e in (eE2, eR2):
        if set(e) != diag or e[(0, 2, 0)] != Fr(1, 24) or e[(0, 0, 2)] != Fr(1, 8):
            raise ValueError("m = 4 decoupling failed: " + repr(e))
    if set(b2) != {(1, 0, 0)}:
        raise ValueError("beta is not a multiple of T: " + repr(b2))
    gE, gR = eE2[(2, 0, 0)], eR2[(2, 0, 0)]
    kappa = surd_mul((b2[(1, 0, 0)] / 2, 1), surd_sqrt(1 / alpha))
    constE = mul_pi_half(mul_surd(E(Fr(1, 2)), surd_sqrt(1 / alpha)), 1)
    jac = Fr(1, 6)
    pref = prefactor(4, jac)
    innerE, innerR = organize_tvp(PE2), organize_tvp(PR2)
    kmax = max(max(max(ks) for ks in innerE.values()), max(max(ks) for ks in innerR.values()))
    return {"alpha": alpha, "kappa": kappa, "gE": gE, "gR": gR, "constE": constE, "pref": pref, "innerE": innerE, "innerR": innerR,
            "kmax": kmax, "JK": jk_exact(kmax), "beta_T": b2[(1, 0, 0)]}


def D4_exact():
    r = m4_reduction()
    eb = mul(mul(r["pref"], r["constE"]), integrate_branch(r["innerE"], r["gE"], True, r["kappa"], r["JK"]))
    rb = mul(r["pref"], integrate_branch(r["innerR"], r["gR"], False, r["kappa"], r["JK"]))
    r["erfc_branch"], r["poly_branch"] = eb, rb
    return add(eb, rb), r


def Z4_exact():
    """Mehta's Z_4 through the same machinery: Z_4 = 4! sqrt(pi) (1/6) int dT e^{-T^2/48} int_0^T dv e^{-v^2/24} int_0^v dp_3
    V e^{-p_3^2/8}, V the Vandermonde p_1 p_2 p_3 (p_1+p_2)(p_2+p_3)(p_1+p_2+p_3), p_1 = (T - v)/3, p_2 = (v - p_3)/2."""
    T, v, q = pvar(0, 3), pvar(1, 3), pvar(2, 3)
    p1 = pscale(padd(T, pscale(v, Fr(-1))), Fr(1, 3))
    p2 = pscale(padd(v, pscale(q, Fr(-1))), Fr(1, 2))
    V = pconst(1, 3)
    for f in (p1, p2, q, padd(p1, p2), padd(p2, q), padd(padd(p1, p2), q)):
        V = pmul(V, f)
    inner = organize_tvp(V)
    kmax = max(max(ks) for ks in inner.values())
    jac = Fr(1, 6)
    const = mul_pi_half(E(24 * jac), 1)
    return mul(const, integrate_branch(inner, Fr(1, 48), False, None, jk_exact(kmax)))



# ---------------- Taylor factors with disc bounds (Math-#199/#200)
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


def t_mul_poly(pc, tm):
    """Product of a short coefficient list pc (a polynomial's Taylor coefficients) with a full Taylor list tm."""
    K = len(tm) - 1
    out = []
    for n in range(K + 1):
        acc = Iv(ZERO)
        for i in range(min(n, len(pc) - 1) + 1):
            acc = acc + pc[i] * tm[n - i]
        out.append(acc)
    return out


def poly_taylor_short(coeffs, s0):
    """Taylor coefficients at s0 of sum_j coeffs[j] s^j, all degrees (exact rationals as intervals)."""
    deg = len(coeffs) - 1
    out = []
    for n in range(deg + 1):
        v = Fr(0)
        for j in range(n, deg + 1):
            v += coeffs[j] * math.comb(j, n) * Fr(s0) ** (j - n)
        out.append(Iv.frac(v))
    return out


def gauss_moment(j, gamma):
    """int_0^inf v^j e^{-gamma v^2} dv = (1/2) gamma^{-(j+1)/2} Gamma((j+1)/2), as an interval."""
    g = Iv.frac(gamma)
    a2 = j + 1
    gpow = ipow(g, a2 // 2) * (g.sqrt() if a2 % 2 else Iv(ONE))
    return gamma_half_integer(a2) / gpow / 2


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


def simpson(f, a, b, n=4000):
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n):
        s += (4 if i % 2 else 2) * f(a + i * h)
    return s * h / 3


def float_moment(n, lam, slopes):
    lam = float(lam)
    us = [float(c) * math.sqrt(r) for c, r in slopes]
    T = math.sqrt(140 / lam)
    def f(s):
        v = s ** n * math.exp(-lam * s * s)
        for u in us:
            v *= math.erf(u * s)
        return v
    return simpson(f, 0.0, T, 20000)


