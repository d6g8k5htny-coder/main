#!/usr/bin/env python3
"""Explicit cap-route constants for Theorem A of [LP] in dimension d = 3, for the torus kernel K_L with L >= 10 in every
frame: for 0 < r <= r_*, b in [0, 1], k in [1/2, 2],

    1 - p_r(b, k, R)  <=  Q^W(G_r^c)  <=  C3(r_*) r^3,        Z_r / r^2 >= z3(r_*),

with G_r the good event of [LP] section 7 / [CAP] (1) (partial-block operator norms on the cylinder
D = [-2r, 2r] x ball(0, 2r)) and C3(r_*), z3(r_*) certified here (NOTE.md Theorem E3).  The proof bounds
Q^W(G_r^c) = E_Q[W_r 1{G_r^c}] / Z_r from above by an exact Gaussian regression near r = 0:

  * covariances of the eight preconditioned pins and the 31 targets are exact power series in r, obtained from the Taylor
    coefficients of the kernel at 0 (every negative power cancels identically, checked); the torus remainder is bounded
    uniformly in r and in the frame by Cauchy estimates on a polydisc in C^3;
  * the failure event is reduced to {lambda_1 <= (4/(3k)) r M_3^2} u {r M_4 > 3k/10}; M_3 is bounded blockwise through exact
    averaging identities of the pinned jets and M_4 through the Taylor series at the midpoint (Frobenius norms);
  * the main term is integrated over the transverse Hessian B = -D_y^2 f(M) in eigenvalue coordinates after an exact
    regression on its three entries, against an angle-free Gaussian majorant of its density; the remaining expectations are
    closed-form incomplete Gaussian and chi moments plus Hoelder-controlled perturbations;
  * the normalizer Z_r is bounded below through a Loewner-order coupling with the contact law, whose moments
    E[det(B)^a tr(B)^c 1{B > 0}] are in closed form; the rare branches are Gaussian and chi tails.
All arithmetic is exact rational interval arithmetic (outward rounding to 2^-160); standard library only.
"""
import argparse
import json
import math
import multiprocessing
import os
import sys
from fractions import Fraction as Fr

HERE = os.path.dirname(os.path.abspath(__file__))
MUTANTS = ('no-delta', 'cap-half', 'no-zfloor-correction', 'no-om-term', 'no-lambda2')
MUT = None
PBITS = 160
L_MIN = 10
NX = 60                    # Taylor order in x of the covariance series (exact); the rest is a bounded tail
NB = 24                    # truncation order of the Taylor tails of the 4-jets at the midpoint (the rest is bounded)
X0 = Fr(1, 64)             # default threshold for the quadratic near-branch analysis (records use COMBOS)
THETA = Fr(1, 32)          # default splitting parameter in the max-moment lemma (records use COMBOS)
THETA_E = Fr(1, 16)        # splitting parameter for the ||r nu - v||^2 factor
QS = (2, 4, 8, 16)         # Hoelder exponents of the rare branches: E[W 1_E] <= ||W||_q P(E)^(1 - 1/q), smallest bound used
R0_COUPLING = 12           # radius of the Loewner coupling of the transverse Hessian with the contact law (floor, section 6)
R_STARS = (Fr(1, 4096), Fr(1, 2048), Fr(1, 1024), Fr(1, 512), Fr(1, 256), Fr(1, 128), Fr(1, 64))
BAND_B = (Fr(0), Fr(1))
BAND_K = (Fr(1, 2), Fr(2))


def require(cond, msg):
    if not cond:
        print(json.dumps({'check': 'FAIL', 'reason': msg, 'mutant': MUT}))
        sys.exit(1)


# ----------------------------------------------------------------------------- rational intervals (from Math-#195 / #203 / #206)
def rdown(x):
    return Fr(x.numerator * (1 << PBITS) // x.denominator, 1 << PBITS)


def rup(x):
    return Fr(-((-x.numerator) * (1 << PBITS) // x.denominator), 1 << PBITS)


class IV:
    __slots__ = ('lo', 'hi')

    def __init__(self, lo, hi=None):
        lo = Fr(lo)
        hi = lo if hi is None else Fr(hi)
        if hi < lo:
            raise ValueError('empty interval')
        self.lo, self.hi = rdown(lo), rup(hi)

    def __repr__(self):
        return 'IV(%s, %s)' % (float(self.lo), float(self.hi))

    def __add__(self, o):
        o = o if isinstance(o, IV) else IV(o)
        return IV(self.lo + o.lo, self.hi + o.hi)
    __radd__ = __add__

    def __neg__(self):
        return IV(-self.hi, -self.lo)

    def __sub__(self, o):
        return self + (-(o if isinstance(o, IV) else IV(o)))

    def __rsub__(self, o):
        return IV(o) - self

    def __mul__(self, o):
        o = o if isinstance(o, IV) else IV(o)
        c = (self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi)
        return IV(min(c), max(c))
    __rmul__ = __mul__

    def inv(self):
        if self.lo <= 0 <= self.hi:
            raise ZeroDivisionError('interval contains zero')
        return IV(1 / self.hi, 1 / self.lo)

    def __truediv__(self, o):
        o = o if isinstance(o, IV) else IV(o)
        return self * o.inv()

    def __rtruediv__(self, o):
        return IV(o) * self.inv()

    def sq(self):
        if self.lo >= 0:
            return IV(self.lo * self.lo, self.hi * self.hi)
        if self.hi <= 0:
            return IV(self.hi * self.hi, self.lo * self.lo)
        return IV(0, max(self.lo * self.lo, self.hi * self.hi))

    def meet(self, o):
        lo, hi = max(self.lo, o.lo), min(self.hi, o.hi)
        if hi < lo:
            raise ValueError('inconsistent enclosures')
        return IV(lo, hi)

    def width(self):
        return self.hi - self.lo

    def mid(self):
        return (self.lo + self.hi) / 2

    def contains(self, x):
        return self.lo <= x <= self.hi


def isqrt_iv(x):
    """Enclosure of sqrt(x) for an interval x with x.lo > 0."""
    if x.lo <= 0:
        raise ValueError('sqrt of interval reaching zero: %r' % x)

    def lo_root(v):
        n = v.numerator * (1 << (2 * PBITS)) // v.denominator
        s = math.isqrt(n)
        return Fr(s, 1 << PBITS)

    def hi_root(v):
        n = -((-v.numerator * (1 << (2 * PBITS))) // v.denominator)
        s = math.isqrt(n) + 1
        return Fr(s, 1 << PBITS)
    return IV(lo_root(x.lo), hi_root(x.hi))


def exp_neg_point(t):
    """Enclosure of exp(-t) for a rational t >= 0: halve t to t <= 1/2, evaluate the alternating series in fixed point
    (ulp = 2^-(PBITS+40)) with an integer bound on the accumulated rounding error, then square back."""
    t = Fr(t)
    m = 0
    while t > Fr(1, 2):
        t /= 2
        m += 1
    P2 = PBITS + 40
    ONE = 1 << P2
    X = t.numerator * ONE // t.denominator            # X / ONE <= t < (X + 1) / ONE
    # series at x_lo = X / ONE: exp(-x_lo) = sum (-1)^n x_lo^n / n!  (alternating, terms decreasing for x_lo <= 1/2)
    S = ONE
    T = ONE
    E = 0                                             # error bound of T in ulps
    Etot = 0
    n = 0
    while T > 0:
        n += 1
        T = (T * X) // (ONE * n)
        E = (E * X) // (ONE * n) + 2
        S += (-1) ** n * T
        Etot += E
    Etot += E + 1                                     # first omitted exact term is below (T_n + E) <= E + 1 ulps
    # exp(-t) in [exp(-(X + 1)/ONE), exp(-X/ONE)] and exp(-(X + 1)/ONE) >= exp(-X/ONE) - 1/ONE (Lipschitz 1)
    v = IV(Fr(S - Etot - 1, ONE), Fr(S + Etot, ONE))
    for _ in range(m):
        v = v.sq()
    return v


def exp_neg(t):
    """exp(-t) on an interval t with t.lo >= 0 (monotone decreasing)."""
    return IV(exp_neg_point(t.hi).lo, exp_neg_point(t.lo).hi)


def _atan_inv(n):
    """Enclosure of arctan(1/n), n >= 2, alternating series."""
    x = Fr(1, n)
    s = Fr(0)
    term = x
    k = 0
    while True:
        s += term / (2 * k + 1)
        k += 1
        term = term * (-x * x)
        if abs(term) / (2 * k + 1) < Fr(1, 1 << (PBITS + 8)):
            break
    return IV(s - abs(term), s + abs(term))


PI = 4 * (4 * _atan_inv(5) - _atan_inv(239))
SQRT2 = isqrt_iv(IV(2))
SQRT2PI = isqrt_iv(2 * PI)


def phi_point(x):
    """Enclosure of Phi(x) = P(N(0,1) <= x) for a rational x with |x| <= 12, from the series
    Phi(x) = 1/2 + (1/sqrt(2 pi)) sum_n (-1)^n x^(2n+1) / (2^n n! (2n+1)), evaluated in fixed point (ulp = 2^-(PBITS+40))
    with an explicit integer bound E on the accumulated rounding error (in ulps) and the alternating-tail bound."""
    x = Fr(x)
    if x < 0:
        p = phi_point(-x)
        return IV(1 - p.hi, 1 - p.lo)
    if x > 12:
        raise ValueError('phi_point range')
    P2 = PBITS + 40
    ONE = 1 << P2
    X = x.numerator * ONE // x.denominator          # x_lo = X / ONE <= x < (X + 1) / ONE ; |Phi(x) - Phi(x_lo)| <= 1 ulp
    if X == 0:                                       # 0 <= x < 1 ulp: Phi(x) in [1/2, 1/2 + ulp]
        return IV(Fr(1, 2), Fr(1, 2) + Fr(1, ONE))
    X2u = (X + 1) * (X + 1)                          # upper bound of x^2 ONE^2
    S = 0
    T = X                                            # T_n ~ x^(2n+1) / (2^n n!) in ulps
    E = 1                                            # error bound of T_n in ulps
    Etot = 1                                         # error of S in ulps (starts with the x rounding)
    n = 0
    while T > 0:
        S += (-1) ** n * (T // (2 * n + 1))
        Etot += E + 1
        n += 1
        T = (T * X * X) // (ONE * ONE * 2 * n)
        E = (E * X2u) // (ONE * ONE * 2 * n) + 2
    # once T == 0 in fixed point the remaining terms are each below 1 ulp; if x^2 >= 2 n they are still growing in
    # exact arithmetic, so continue counting until they decrease: bound the tail by the first omitted exact term
    # <= (T_n + E) ulps <= (E + 1) ulps when T_n == 0 ... but growth may resume; require the decreasing regime.
    if x * x > 2 * n:
        raise ValueError('series terminated before the decreasing regime')
    Etot += E + 1
    core = IV(Fr(S - Etot, ONE), Fr(S + Etot, ONE)) / SQRT2PI
    return IV(Fr(1, 2) + core.lo, Fr(1, 2) + core.hi)


def Phi(x):
    """Phi on an interval (monotone); arguments clipped to [-12, 12] (clipping only widens)."""
    lo = min(max(x.lo, Fr(-12)), Fr(12))
    hi = max(min(x.hi, Fr(12)), Fr(-12))
    if x.lo < -12:
        plo = Fr(0)
    else:
        plo = phi_point(lo).lo
    if x.hi > 12:
        phi_ = Fr(1)
    else:
        phi_ = phi_point(hi).hi
    return IV(plo, phi_)


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


# ----------------------------------------------------------------------------- functionals and exact covariance series (d = 3)
HALF = Fr(1, 2)
XM, XS, X0PT = -HALF, HALF, Fr(0)
JET4_IDX = {'X%d%d%d' % (a, b1, 4 - a - b1): (a, b1, 4 - a - b1) for a in range(5) for b1 in range(5 - a)}
JET4 = tuple(sorted(JET4_IDX, key=lambda n: (-JET4_IDX[n][0], -JET4_IDX[n][1])))
BLOCKS = ((4, 0), (3, 1), (2, 2), (1, 3), (0, 4))           # partial blocks (a, c) of the fourth derivatives, a + c = 4


def block_members(a, c):
    """[(name, Frobenius multiplicity c!/beta!)] of the 4-jet block d_x^a D_y^c f(0)."""
    out = []
    for b1 in range(c, -1, -1):
        b2 = c - b1
        out.append(('X%d%d%d' % (a, b1, b2), math.factorial(c) // (math.factorial(b1) * math.factorial(b2))))
    return out


def term(c, p, a, b1, b2, x0):
    """c r^p d_x^a d_y1^b1 d_y2^b2 f(x0 r, 0, 0)."""
    return (Fr(c), p, a, b1, b2, Fr(x0))


def functionals():
    pins = {
        'p1': [term(HALF, 0, 0, 0, 0, XM), term(HALF, 0, 0, 0, 0, XS)],
        'p2': [term(-1, -1, 0, 0, 0, XM), term(1, -1, 0, 0, 0, XS)],
        'p3p': [term(6, -2, 1, 0, 0, XM), term(6, -2, 1, 0, 0, XS), term(12, -3, 0, 0, 0, XM), term(-12, -3, 0, 0, 0, XS)],
        'p4': [term(-1, -1, 1, 0, 0, XM), term(1, -1, 1, 0, 0, XS)],
    }
    for j, (e1, e2) in ((1, (1, 0)), (2, (0, 1))):
        pins['q1%d' % j] = [term(HALF, 0, 0, e1, e2, XM), term(HALF, 0, 0, e1, e2, XS)]
        pins['q2%d' % j] = [term(-1, -1, 0, e1, e2, XM), term(1, -1, 0, e1, e2, XS)]
    targets = {
        'a11': [term(1, 0, 0, 2, 0, XM)], 'a22': [term(1, 0, 0, 0, 2, XM)], 'a12': [term(1, 0, 0, 1, 1, XM)],
        'o11': [term(1, -1, 0, 2, 0, XS), term(-1, -1, 0, 2, 0, XM)],
        'o22': [term(1, -1, 0, 0, 2, XS), term(-1, -1, 0, 0, 2, XM)],
        'o12': [term(1, -1, 0, 1, 1, XS), term(-1, -1, 0, 1, 1, XM)],
        'T': [term(1, -2, 2, 0, 0, XM), term(4, -3, 1, 0, 0, XM), term(2, -3, 1, 0, 0, XS), term(6, -4, 0, 0, 0, XM),
              term(-6, -4, 0, 0, 0, XS)],
        'tau': [term(1, -3, 2, 0, 0, XS), term(-1, -3, 2, 0, 0, XM), term(-6, -4, 1, 0, 0, XM), term(-6, -4, 1, 0, 0, XS),
                term(-12, -5, 0, 0, 0, XM), term(12, -5, 0, 0, 0, XS)],
    }
    for j, (e1, e2) in ((1, (1, 0)), (2, (0, 1))):
        targets['v%d' % j] = [term(1, -1, 1, e1, e2, XM), term(1, -2, 0, e1, e2, XM), term(-1, -2, 0, e1, e2, XS)]
        targets['nu%d' % j] = [term(1, -2, 1, e1, e2, XM), term(1, -2, 1, e1, e2, XS), term(2, -3, 0, e1, e2, XM),
                               term(-2, -3, 0, e1, e2, XS)]
    for b1 in range(3, -1, -1):
        targets['y%s' % ('1' * b1 + '2' * (3 - b1))] = [term(1, 0, 0, b1, 3 - b1, X0PT)]
    for n, (a, b1, b2) in JET4_IDX.items():
        targets[n] = [term(1, 0, a, b1, b2, X0PT)]
    return pins, targets


PIN_NAMES = ('p1', 'p2', 'p3p', 'p4', 'q11', 'q21', 'q12', 'q22')
BNAMES = ('a11', 'a22', 'a12')                              # the transverse Hessian A = D_y^2 f(M) = -B
VNAMES = ('v1', 'v2')
ONAMES = ('o11', 'o22', 'o12')
YNAMES = ('y111', 'y112', 'y122', 'y222')
FREE = VNAMES + ONAMES + YNAMES                              # the free third-derivative blocks, in Cholesky order
FREE_W = (1, 1, 1, 1, 2, 1, 3, 3, 1)                         # Frobenius multiplicities (Euclidean norms in these weights)
TARGET_NAMES = BNAMES + FREE + ('T', 'tau', 'nu1', 'nu2') + JET4
NP = len(PIN_NAMES)


def weight(F):
    """Scaling weight a - (power of r), common to all terms of a preconditioned functional."""
    ws = {t[2] - t[1] for t in F}
    require(len(ws) == 1, 'functional not homogeneous')
    return ws.pop()


def hcoef(n):
    """d^n/dx^n exp(-x^2/2) at 0."""
    if n % 2:
        return 0
    return (-1) ** (n // 2) * dfact(n - 1)


PAIR_CACHE = {}


def pair_series(F, G, key):
    """Exact data of Cov_K(F, G) = sum_gamma kappa_gamma pi_gamma r^(gx - wF - wG), gamma = (gx, gy1, gy2):
    returns (reference series {power: rational}, weight w, list of (|c c'|, a+a', gy1, gy2) for the remainder bounds)."""
    if key in PAIR_CACHE:
        return PAIR_CACHE[key]
    wF, wG = weight(F), weight(G)
    w = wF + wG
    pi = {}
    absterms = []
    for (cf, pf, af, bf1, bf2, xf) in F:
        for (cg, pg, ag, bg1, bg2, xg) in G:
            g1, g2 = bf1 + bg1, bf2 + bg2
            sgn = (-1) ** (ag + bg1 + bg2)
            d = xf - xg
            absterms.append((abs(cf * cg), af + ag, g1, g2))
            for gx in range(af + ag, NX + 1):
                m = gx - af - ag
                val = cf * cg * sgn * d ** m / math.factorial(m)
                if val != 0:
                    pi[(gx, g1, g2)] = pi.get((gx, g1, g2), Fr(0)) + val
    for (gx, g1, g2), v in pi.items():
        require(v == 0 or gx - w >= 0, 'negative power of r in a covariance (regularity fails)')
    ser = {}
    for (gx, g1, g2), v in pi.items():
        kap = hcoef(gx) * hcoef(g1) * hcoef(g2)
        if kap and v:
            ser[gx - w] = ser.get(gx - w, Fr(0)) + kap * v
    out = ({p: c for p, c in ser.items() if c != 0}, w, absterms)
    PAIR_CACHE[key] = out
    return out


E_UP = Fr(27183, 10000)                                      # > e
SQRT3_UP = Fr(7, 4)                                          # > sqrt(3)


def torus_M1(L):
    """Bound of |sum_{n != 0} G(z + p_n)| on the polydisc |z_1|, |z_2|, |z_3| <= 1 (every frame, |p_n| = |n| L):
    exp(3/2) sum_{j >= 1} (24 j^2 + 2) exp(-(j L - sqrt 3)^2 / 2); 24 j^2 + 2 lattice points have sup-norm j."""
    L = Fr(L)
    e32 = exp_neg_point(Fr(3, 2)).inv().hi
    tot = Fr(0)
    for j in range(1, 6):
        tot += (24 * j * j + 2) * exp_neg_point((j * L - SQRT3_UP) ** 2 / 2).hi
    # j >= 6: (24 j^2 + 2) exp(-(jL - 7/4)^2/2) decreases by more than 1/2 per step for L >= 10
    tail = 2 * (24 * 36 + 2) * exp_neg_point((6 * L - SQRT3_UP) ** 2 / 2).hi
    return e32 * (tot + tail)


def theta_minus_one(L):
    L = Fr(L)
    tot = Fr(0)
    for j in range(1, 6):
        tot += (24 * j * j + 2) * exp_neg_point((j * L) ** 2 / 2).hi
    return tot + 2 * (24 * 36 + 2) * exp_neg_point((6 * L) ** 2 / 2).hi


def cov_band(F, G, key, r0, r1, M1):
    """Enclosure of the covariance (unnormalized torus kernel sum_n G(z + p_n)) of F and G, for r in [r0, r1]."""
    ser, w, absterms = pair_series(F, G, key)
    val = IV(0)
    for p, c in ser.items():
        val = val + IV(r0 ** p, r1 ** p) * c
    A = max((t[1] for t in absterms), default=0)
    # Taylor truncation tail of the reference kernel (orders gx > NX)
    tail = Fr(0)
    for (acc, aa, g1, g2) in absterms:
        tail += acc * abs(hcoef(g1) * hcoef(g2))
    tail *= 2 * Fr(NX + 1) ** A * r1 ** (NX + 1 - w)
    # torus remainder: Cauchy estimates |rho_gamma| <= gx! gy1! gy2! M1 on the unit polydisc
    tor = Fr(0)
    for (acc, aa, g1, g2) in absterms:
        tor += acc * math.factorial(g1) * math.factorial(g2) * Fr(max(w, 1)) ** aa
    tor = tor * M1 / (1 - E_UP * r1)
    err = tail + tor
    return val + IV(-err, err)


# ----------------------------------------------------------------------------- interval linear algebra
def solve_iv(A, B):
    """Interval Gaussian elimination with partial pivoting on midpoints: X[c] solves A X[c] = B[c]."""
    n = len(A)
    M = [[A[i][j] for j in range(n)] + [B[c][i] for c in range(len(B))] for i in range(n)]
    for col in range(n):
        p = max(range(col, n), key=lambda i: abs(M[i][col].mid()))
        M[col], M[p] = M[p], M[col]
        piv = M[col][col]
        for i in range(col + 1, n):
            f = M[i][col] / piv
            for j in range(col, len(M[i])):
                M[i][j] = M[i][j] - f * M[col][j]
    X = []
    for c in range(len(B)):
        x = [None] * n
        for i in reversed(range(n)):
            s = M[i][n + c]
            for j in range(i + 1, n):
                s = s - M[i][j] * x[j]
            x[i] = s / M[i][i]
        X.append(x)
    return X


def cholesky(C):
    """Interval Cholesky of an n x n SPD interval matrix (pivots must be positive)."""
    n = len(C)
    L = [[IV(0)] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = C[i][j] - sum((L[i][l] * L[j][l] for l in range(j)), IV(0))
            if i == j:
                require(s.lo > 0, 'Cholesky pivot not positive')
                L[i][j] = isqrt_iv(s)
            else:
                L[i][j] = s / L[j][j]
    return L


def opnorm_lower_tri(L, rows, cols):
    """Upper bound of the operator norm of the block L[rows][cols] of an interval lower-triangular matrix:
    max |diagonal| + Frobenius norm of the rest (square diagonal blocks), Frobenius norm otherwise."""
    if rows == cols:
        dg = max(absup(L[i][i]) for i in rows)
        off = sum(absup(L[i][j]) ** 2 for i in rows for j in cols if j < i)
        return rup_rel(dg + sqrt_up(off))
    return sqrt_up(sum(absup(L[i][j]) ** 2 for i in rows for j in cols))


def sym3_eig_bounds(C):
    """(lower bound of lambda_min, upper bound of lambda_max) of a symmetric 3 x 3 interval matrix (Gershgorin)."""
    lo = min(C[i][i].lo - sum(absup(C[i][j]) for j in range(3) if j != i) for i in range(3))
    hi = max(C[i][i].hi + sum(absup(C[i][j]) for j in range(3) if j != i) for i in range(3))
    return lo, hi


def det3(C):
    a, b, c = C[0]
    d, e, f = C[1]
    g, h, i = C[2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


# ----------------------------------------------------------------------------- the conditional law on an r-band
class Law:
    pass


def band_law(r0, r1, L=L_MIN):
    """Interval enclosures, uniform over r in [r0, r1], every frame and every torus side >= L, of the Q-law of the
    targets: mean = b alpha + k beta, covariance C (normalized kernel), and the pin energy coefficients; then the
    regression on the transverse Hessian (three entries) and the g-law of the remaining targets."""
    pins, targets = functionals()
    P = [pins[n] for n in PIN_NAMES]
    T = [targets[n] for n in TARGET_NAMES]
    M1 = torus_M1(L)
    th1 = theta_minus_one(L)
    nt = len(T)
    Spp = [[None] * NP for _ in range(NP)]
    for i in range(NP):
        for j in range(i, NP):
            Spp[i][j] = Spp[j][i] = cov_band(P[i], P[j], ('p', i, j), r0, r1, M1)
    Stp = [[cov_band(T[i], P[j], ('tp', i, j), r0, r1, M1) for j in range(NP)] for i in range(nt)]
    Stt = [[None] * nt for _ in range(nt)]
    for i in range(nt):
        for j in range(i, nt):
            Stt[i][j] = Stt[j][i] = cov_band(T[i], T[j], ('tt', i, j), r0, r1, M1)
    r = IV(r0, r1)
    kvec = [-(r * r * r) * HALF, -(r * r), IV(12)] + [IV(0)] * (NP - 3)
    e1 = [IV(1)] + [IV(0)] * (NP - 1)
    W = solve_iv(Spp, Stp)
    xs = solve_iv(Spp, [e1, kvec])
    law = Law()
    law.r0, law.r1, law.L, law.M1, law.th1 = r0, r1, L, M1, th1
    law.alpha = {n: W[i][0] for i, n in enumerate(TARGET_NAMES)}
    law.beta = {n: sum((W[i][j] * kvec[j] for j in range(NP)), IV(0)) for i, n in enumerate(TARGET_NAMES)}
    scale = IV(1 / (1 + th1), 1)          # K_L = (unnormalized) / Theta, Theta in [1, 1 + th1]
    C = {}
    for i, a in enumerate(TARGET_NAMES):
        for j in range(i, nt):
            c = TARGET_NAMES[j]
            v = Stt[i][j] - sum((W[i][l] * Stp[j][l] for l in range(NP)), IV(0))
            C[(a, c)] = C[(c, a)] = v * scale
    law.C = C
    escale = IV(1, 1 + th1)
    law.e_bb = xs[0][0] * escale
    law.e_bk = sum((kvec[j] * xs[0][j] for j in range(NP)), IV(0)) * escale
    law.e_kk = sum((kvec[j] * xs[1][j] for j in range(NP)), IV(0)) * escale
    regress_on_B(law)
    law.S = {blk: taylor_tail_coefficient(blk, r1, M1) for blk in BLOCKS}
    law.net_var = {blk: net_max_variance(law.C, blk) for blk in BLOCKS if blk[1] >= 2}
    return law


NET_M = 128                                # rational unit vectors ((1 - t^2), 2 t)/(1 + t^2), t = j/NET_M, and their rotations


def circle_net():
    """Rational unit vectors u_j covering angles [0, pi) with consecutive angular gaps <= 2/NET_M (theta = 2 arctan t has
    d theta/dt <= 2), together with the rotated copies (-u2, u1)."""
    pts = []
    for j in range(NET_M + 1):
        t = Fr(j, NET_M)
        pts.append(((1 - t * t) / (1 + t * t), 2 * t / (1 + t * t)))
    return pts + [(-u2, u1) for (u1, u2) in pts]


NET = circle_net()


def block_form_weights(blk, u):
    """w_beta = (c!/beta!) u1^beta1 u2^beta2: d_x^a d_u^c f(0) = sum_beta w_beta X_(a,beta)."""
    a, c = blk
    return [(n, m * u[0] ** int(n[2]) * u[1] ** int(n[3])) for n, m in block_members(a, c)]


def net_max_variance(C, blk):
    """Upper bound of max_j Var_Q(d_x^a d_(u_j)^c f(0)) over the net."""
    best = Fr(0)
    for u in NET:
        w = block_form_weights(blk, u)
        v = sum(abs(wi) * abs(wj) * absup(C[(ni, nj)]) if ni != nj else wi * wi * C[(ni, ni)].hi
                for ni, wi in w for nj, wj in w)
        best = max(best, v)
    return best


FROB_B = (IV(1), IV(1), SQRT2)               # (B11, B22, B12) -> Frobenius-isometric coordinates (B11, B22, sqrt2 B12)


def regress_on_B(law):
    """Regression of every target on the three entries of the transverse Hessian, in the Frobenius-isometric coordinates
    b~ = (B11, B22, sqrt2 B12) of B = -A: F = F_g + beta~_F . (b~ - mu~), with F_g independent of B."""
    C = law.C
    # covariance of b~ (B = -A: signs cancel in the covariance)
    Cb = [[C[(BNAMES[i], BNAMES[j])] * FROB_B[i] * FROB_B[j] for j in range(3)] for i in range(3)]
    law.Cb = Cb
    law.detCb = det3(Cb)
    require(law.detCb.lo > 0, 'transverse Hessian covariance not positive')
    lo, hi = sym3_eig_bounds(Cb)
    require(lo > 0, 'transverse Hessian covariance eigenvalue bound')
    law.sB_lo2, law.sB_hi2 = lo, hi
    law.sB_lo = sqrt_down(lo)
    # regression coefficients of F on b~: beta~_F = Cov(F, b~) Cb^-1, Cov(F, b~_i) = -Cov(F, A_i) FROB_B[i]
    others = [n for n in TARGET_NAMES if n not in BNAMES]
    rhs = [[-C[(n, BNAMES[i])] * FROB_B[i] for i in range(3)] for n in others]
    sol = solve_iv(Cb, rhs)
    law.bt = {n: sol[i] for i, n in enumerate(others)}
    law.bnorm = {n: sqrt_up(sum(absup(x) ** 2 for x in law.bt[n])) for n in others}
    Cg = {}
    for a in others:
        for c in others:
            Cg[(a, c)] = C[(a, c)] - sum((law.bt[a][i] * (-C[(c, BNAMES[i])] * FROB_B[i]) for i in range(3)), IV(0))
    law.Cg = Cg
    # the free blocks in Frobenius coordinates: v, omega = (o11, o22, sqrt2 o12), Y~ = (y111, sqrt3 y112, sqrt3 y122, y222)
    fw = [isqrt_iv(IV(w)) if w != 1 else IV(1) for w in FREE_W]
    law.Lfree = cholesky([[Cg[(FREE[i], FREE[j])] * fw[i] * fw[j] for j in range(9)] for i in range(9)])
    law.fw = fw
    law.sdg = {n: sqrt_up(max(Cg[(n, n)].hi, Fr(0))) for n in others}
    law.sdQ = {n: sqrt_up(C[(n, n)].hi) for n in TARGET_NAMES}


def sd0_sq(g, M1):
    """Upper bound of Var d^g f(0) = |d^(2g) K_L(0)| (normalized kernel, every frame, L >= L_MIN), g = (gx, gy1, gy2):
    the reference value prod (2 g_i - 1)!! plus the Cauchy bound (2g)! M1 of the torus remainder."""
    return math.prod(dfact(2 * x - 1) for x in g) + math.prod(math.factorial(2 * x) for x in g) * M1


def taylor_tail_coefficient(blk, r1, M1):
    """S_(a,c)(r1) = sum_{gamma != 0} sigma_(a,c),gamma (2 r1)^|gamma| / gamma!, sigma^2 = sum_beta (c!/beta!) s^2_(a+gx, beta+gy):
    the Frobenius norm of the Taylor tail of the block d_x^a D_y^c f at the midpoint over D."""
    a, c = blk
    tot = Fr(0)
    for n in range(1, NB + 1):
        for gx in range(n + 1):
            for g1 in range(n - gx + 1):
                g2 = n - gx - g1
                s2 = Fr(0)
                for b1 in range(c + 1):
                    b2 = c - b1
                    mult = math.factorial(c) // (math.factorial(b1) * math.factorial(b2))
                    s2 += mult * sd0_sq((a + gx, b1 + g1, b2 + g2), M1)
                tot += sqrt_up(s2) * (2 * r1) ** n / (math.factorial(gx) * math.factorial(g1) * math.factorial(g2))
    # tail |gamma| = n > NB: s_delta <= 2^|delta| delta! sqrt(1 + M1) and (alpha + gamma)!/gamma! <= (n + 4)^4, so
    # sigma (2 r1)^n / gamma! <= 2^(c/2) 2^(4 + n) (n + 4)^4 (2 r1)^n sqrt(1 + M1); (n+1)(n+2)/2 multi-indices of order n;
    # the ratio of consecutive bounds is below 2 q for these n (q = 4 r1 <= 1/16)
    q = 4 * r1
    require(q <= Fr(1, 16), 'Taylor tail ratio too large')
    n = NB + 1
    base = 2 * 2 ** 4 * sqrt_up(Fr(2) ** c * (1 + M1))
    tail = base * Fr((n + 1) * (n + 2), 2) * Fr(n + 4) ** 4 * q ** n / (1 - 2 * q)
    return rup_rel(tot + tail)


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
# ----------------------------------------------------------------------------- the (k, b) boxes of the sweep
F_ = Fr
GRID = (
    ((F_(1, 2), F_(3, 4)), (F_(0), F_(1, 2), F_(1))),
    ((F_(3, 4), F_(1)), (F_(0), F_(1, 2), F_(1))),
    ((F_(1), F_(5, 4)), (F_(0), F_(1, 2), F_(1))),
    ((F_(5, 4), F_(3, 2)), (F_(0), F_(1, 2), F_(1))),
    ((F_(3, 2), F_(13, 8)), (F_(0), F_(243, 512), F_(1))),
    ((F_(13, 8), F_(7, 4)), (F_(0), F_(309, 1024), F_(899, 1024), F_(1))),
    ((F_(7, 4), F_(15, 8)), (F_(0), F_(37, 256), F_(105, 256), F_(933, 1024), F_(1))),
    ((F_(15, 8), F_(31, 16)), (F_(0), F_(37, 512), F_(207, 1024), F_(225, 512), F_(113, 128), F_(1))),
    ((F_(31, 16), F_(63, 32)), (F_(0), F_(37, 1024), F_(103, 1024), F_(111, 512), F_(439, 1024), F_(421, 512), F_(1))),
    ((F_(63, 32), F_(2)), (F_(0), F_(1, 512), F_(5, 1024), F_(5, 512), F_(19, 1024), F_(35, 1024), F_(1, 16), F_(29, 256),
                           F_(209, 1024), F_(377, 1024), F_(685, 1024), F_(1))),
)


# ----------------------------------------------------------------------------- bands, boxes, runs
TOP_OCTAVES = 7            # octaves [2^-(j+1), 2^-j], j = 6..12 (r up to 1/64), split into 8 sub-bands
LOW_OCTAVES = 16           # octaves j = 13..28, split into 4 sub-bands
R_LAST = Fr(1, 2 ** 29)    # last band [0, 2^-29]
COMBO = (Fr(1, 64), Fr(1, 32))     # (X_0, theta) of NOTE sections 4-5


def bands():
    """Sub-bands [r0, r1] covering (0, 1/64]; each octave split into equal sub-bands; the last band is [0, R_LAST]."""
    out = []
    for j in range(6, 6 + TOP_OCTAVES + LOW_OCTAVES):
        hi = Fr(1, 2 ** j)
        lo = hi / 2
        m = 8 if j < 6 + TOP_OCTAVES else 4
        for i in range(m):
            out.append((lo + (hi - lo) * i / m, lo + (hi - lo) * (i + 1) / m))
    out.append((Fr(0), R_LAST))
    require(min(b[0] for b in out) == 0 and max(b[1] for b in out) == Fr(1, 64), 'band cover')
    out = sorted(out)
    require(all(out[i][1] == out[i + 1][0] for i in range(len(out) - 1)), 'bands contiguous')
    return out


def dec_down(x, places=4):
    x = Fr(x)
    n = (x.numerator * 10 ** places) // x.denominator
    s = str(abs(n)).rjust(places + 1, '0')
    return ('-' if n < 0 else '') + s[:-places] + '.' + s[-places:]


def dec_up(x, places=4):
    x = Fr(x)
    n = -((-x.numerator * 10 ** places) // x.denominator)
    s = str(abs(n)).rjust(places + 1, '0')
    return ('-' if n < 0 else '') + s[:-places] + '.' + s[-places:]


def grid_boxes():
    out = []
    for kb, bg in GRID:
        require(bg[0] == 0 and bg[-1] == 1 and all(bg[i] < bg[i + 1] for i in range(len(bg) - 1)), 'b grid')
        for i in range(len(bg) - 1):
            out.append(((Fr(bg[i]), Fr(bg[i + 1])), (Fr(kb[0]), Fr(kb[1]))))
    ks = [kb for kb, _ in GRID]
    require(ks[0][0] == BAND_K[0] and ks[-1][1] == BAND_K[1] and all(ks[i][1] == ks[i + 1][0] for i in range(len(ks) - 1)),
            'k boxes cover K')
    return tuple(out)


CG3_02_LO = Fr('22424320.655069')     # certified lower end of c_G^(3)(0, 2) (Math-#208)


def band_record(args):
    """Per band: the certified bound of sup Q^W(G_r^c)/r^3 (rounded up, 4 places) and the floor of Z_r/r^2 (rounded down,
    6 places) for each box in BOXES order, and the largest tail part (rounded up, 12 places)."""
    r0, r1 = args
    law = band_law(r0, r1)
    last = (r0 == 0)
    bound, zfloor, tails = [], [], []
    for bb, kb in BOXES:
        res = box_bound(law, bb, kb, r0, r1, last_band=last, x0=COMBO[0], theta=COMBO[1])
        bound.append(dec_up(res['total']))
        zfloor.append(dec_down(36 * kb[0] * kb[0] * res['zt'], 6))
        tails.append(res['tail'])
    return {'r': [str(r0), str(r1)], 'bound': bound, 'zfloor': zfloor, 'tail_max': dec_up(max(tails), 12)}


def full_run(procs):
    bl = bands()
    with multiprocessing.Pool(procs) as pool:
        recs = pool.map(band_record, bl)
    return assemble(recs)


def assemble(recs):
    table = {}
    for rs in R_STARS:
        mx, arg, zmin = Fr(0), None, None
        for rec in recs:
            if Fr(rec['r'][1]) <= rs:
                for (bb, kb), bs, zs in zip(BOXES, rec['bound'], rec['zfloor']):
                    v = Fr(bs)
                    if v > mx:
                        mx, arg = v, {'r': rec['r'], 'b': [str(bb[0]), str(bb[1])], 'k': [str(kb[0]), str(kb[1])]}
                    z = Fr(zs)
                    zmin = z if zmin is None or z < zmin else zmin
        table[str(rs)] = {'C': str(mx), 'argmax': arg, 'C_over_cG3_02': dec_up(mx / CG3_02_LO, 6),
                          'C_rstar3': dec_up(mx * rs ** 3, 10), 'z_star': str(zmin)}
    return {'schema': 1, 'object': 'CL-C8-THEOREM-A-CONSTANTS-D3-20261001-v1', 'scientific_effect': 'NONE',
            'certified': True, 'mutant': MUT, 'L_min': L_MIN, 'dimension': 3,
            'band': {'b': [str(BAND_B[0]), str(BAND_B[1])], 'k': [str(BAND_K[0]), str(BAND_K[1])]},
            'parameters': params(), 'C_table': table, 'bands': recs}


def params():
    return {'PBITS': PBITS, 'NX': NX, 'NB': NB, 'COMBO_X0_THETA': [str(COMBO[0]), str(COMBO[1])], 'THETA_E': str(THETA_E),
            'QS': list(QS), 'R0_COUPLING': R0_COUPLING, 'NET_M': NET_M, 'R_STARS': [str(x) for x in R_STARS],
            'TOP_OCTAVES': TOP_OCTAVES, 'LOW_OCTAVES': LOW_OCTAVES, 'R_LAST': str(R_LAST),
            'GRID': [[[str(kb[0]), str(kb[1])], [str(x) for x in bg]] for kb, bg in GRID]}


def results_text(res):
    """RESULTS.json layout: the summary indented, then one line per band record."""
    head = json.dumps({key: val for key, val in res.items() if key != 'bands'}, indent=1, sort_keys=True)
    lines = ',\n'.join('  ' + json.dumps(rec, sort_keys=True) for rec in res['bands'])
    return head[:-2] + ',\n "bands": [\n' + lines + '\n ]\n}\n'


def write_results(res):
    with open(os.path.join(HERE, 'RESULTS.json'), 'w') as fh:
        fh.write(results_text(res))


def load_results():
    with open(os.path.join(HERE, 'RESULTS.json')) as fh:
        return json.load(fh)


def replay_sample():
    """Band indices replayed by --check: the band ending at each r_*, the bands ending at 2^-14, 2^-20, 2^-26, the last band."""
    bl = bands()
    idx = []
    for rs in R_STARS:
        j = max(i for i, b in enumerate(bl) if b[1] == rs)
        idx.append(j)
    idx += [i for i, b in enumerate(bl) if b[1] in (Fr(1, 2 ** 14), Fr(1, 2 ** 20), Fr(1, 2 ** 26))]
    idx.append(0)
    return sorted(set(idx))


def check_run(procs, full=False):
    ref = load_results()
    with open(os.path.join(HERE, 'RESULTS.json')) as fh:
        require(fh.read() == results_text(ref), 'RESULTS.json is not in the canonical layout')
    require(ref['parameters'] == params(), 'parameters differ from RESULTS.json')
    require(ref['mutant'] is None, 'RESULTS.json produced under a mutant')
    require(abs(PI.mid() - Fr('3.14159265358979323846')) < Fr(1, 10 ** 19) and PI.width() < Fr(1, 10 ** 30), 'pi enclosure')
    require(abs(Phi(IV(1)).mid() - Fr('0.841344746068542948585')) < Fr(1, 10 ** 19), 'Phi(1) enclosure')
    require(abs(m3(0).mid() - (Fr(7, 2) - 2 * SQRT2.mid())) < Fr(1, 10 ** 40), 'm_(3,0) closed form')
    bl = bands()
    require([rec['r'] for rec in ref['bands']] == [[str(a), str(b)] for a, b in bl], 'band list differs')
    idx = list(range(len(bl))) if full else replay_sample()
    with multiprocessing.Pool(procs) as pool:          # in order; the first differing band exits (and stops the pool)
        for i, rec in zip(idx, pool.imap(band_record, [bl[i] for i in idx])):
            require(rec == ref['bands'][i], 'band %d (%s) differs from RESULTS.json' % (i, rec['r']))
    # the table is the maximum over the stored records
    asm = assemble(ref['bands'])
    require(asm['C_table'] == ref['C_table'], 'C table is not the maximum of the stored bounds')
    # sanity: every C exceeds the sharp reference-kernel cap coefficient c_G^(3)(0, 2) of Math-#208 (a lower bound for every
    # d = 3 cap-route constant valid at (0, 2) for the reference kernel, which this certificate covers)
    for rs, row in ref['C_table'].items():
        require(Fr(row['C']) > CG3_02_LO, 'C below the sharp cap coefficient')
    print(json.dumps({'check': 'ok', 'bands_replayed': len(idx), 'C_table': {rs: row['C'] for rs, row in ref['C_table'].items()},
                      'z_star': {rs: row['z_star'] for rs, row in ref['C_table'].items()}, 'mutant': MUT}))


def main():
    global MUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--check-full', action='store_true')
    ap.add_argument('--mutant', choices=MUTANTS)
    ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--mc', nargs=5, metavar=('R', 'B', 'K', 'N', 'SEED'), help='floating end-to-end control')
    args = ap.parse_args()
    MUT = args.mutant
    if args.mc:
        rr, bb, kk, nn, sd = args.mc
        print(json.dumps(mc_event(float(Fr(rr)), float(Fr(bb)), float(Fr(kk)), int(nn), int(sd))))
        return
    if args.check or args.check_full:
        check_run(args.procs, full=args.check_full)
        return
    res = full_run(args.procs)
    write_results(res)
    print(json.dumps({'C': {rs: row['C'] for rs, row in res['C_table'].items()},
                      'z': {rs: row['z_star'] for rs, row in res['C_table'].items()}}))
BOXES = grid_boxes()
