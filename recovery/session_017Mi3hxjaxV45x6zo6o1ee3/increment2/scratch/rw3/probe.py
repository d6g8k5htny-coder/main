"""probe"""
import argparse
import json
import math
import sys
import time
from decimal import Decimal, Context, ROUND_FLOOR, ROUND_CEILING, ROUND_HALF_EVEN, setcontext
from fractions import Fraction as Fr

MUTANTS = ("image-shells", "lambda-min", "sandwich-power", "window-scale", "d3-cross", "schur-sign", "sphere", "tau-cross",
           "tail-dropped", "no-owen", "gamma-shape")
MUT = None

PREC = 60
CF = Context(prec=PREC, rounding=ROUND_FLOOR, Emin=-9999999, Emax=9999999)
CC = Context(prec=PREC, rounding=ROUND_CEILING, Emin=-9999999, Emax=9999999)
CE = Context(prec=PREC, rounding=ROUND_HALF_EVEN, Emin=-9999999, Emax=9999999)
# Every arithmetic step below names one of the three contexts; operator syntax on Decimals would round to the thread's
# current context, which is widened as a safety net (rule LIBRARY_EXACT checks exp, sqrt and the negation).
setcontext(Context(prec=2 * PREC + 20, rounding=ROUND_HALF_EVEN, Emin=-9999999, Emax=9999999))
ZERO, ONE, HALF = Decimal(0), Decimal(1), Decimal("0.5")
INF = math.inf
ROOT_DIGITS = 70
SIDE24_D3 = ("0.04177593184059834334", "0.04177593184059834335")   # coefficients/side24_v1/ENCLOSURE.json, dimensions.3



# ---------------- interval arithmetic and special functions (Math-#197 toolkit, unchanged)
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
        return Iv(a.hi.copy_negate(), a.lo.copy_negate())          # exact: unary minus would round to the default context

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

    def contains(a, x):
        x = Iv.of(x)
        return a.lo <= x.lo and x.hi <= a.hi

    def intersects(a, b):
        return a.lo <= b.hi and b.lo <= a.hi

    def mag(a):
        return max(abs(a.lo), abs(a.hi))

    def mid_float(a):
        return float((a.lo + a.hi) / 2)

    def pair(a):
        return [str(a.lo), str(a.hi)]


def hull(a, b):
    return Iv(min(a.lo, b.lo), max(a.hi, b.hi))


def inroot(N, n):
    """floor(N ** (1/n)) for integers N >= 0, n >= 1: integer Newton iteration started above the root."""
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
    """Rigorous bracket of q ** (1/n) for a positive rational q, width 10^-ROOT_DIGITS."""
    q = Fr(q)
    if q <= 0:
        raise ValueError("root of a nonpositive number")
    p, d = q.numerator, q.denominator
    scale = 10 ** ROOT_DIGITS
    r = inroot(p * d ** (n - 1) * scale ** n, n)
    den = d * scale
    return Iv(Iv.frac(Fr(r, den)).lo, Iv.frac(Fr(r + 1, den)).hi)


def root_iv(x, n):
    """x ** (1/n) for an interval x > 0 (monotone: endpoint brackets)."""
    return Iv(root_frac(Fr(x.lo), n).lo, root_frac(Fr(x.hi), n).hi)


def pi_interval():
    def atan_inv(n, terms):
        x = Fr(1, n)
        partial, sums = Iv(ZERO), []
        for m in range(terms + 2):
            t = Iv.frac(x ** (2 * m + 1) / (2 * m + 1))
            partial = partial + t if m % 2 == 0 else partial - t
            sums.append(partial)
        return hull(sums[-1], sums[-2])              # alternating series with decreasing terms
    return atan_inv(5, 44) * 16 - atan_inv(239, 22) * 4


PI = pi_interval()
SQRT2 = Iv.frac(2).sqrt()
SQRT3 = Iv.frac(3).sqrt()
SQRT2PI = (2 * PI).sqrt()
SQRTPI = PI.sqrt()
TINY = Decimal(10) ** (-PREC - 8)


def erf_point(x, coarse=False):
    """erf(x) at a Decimal point: (2/sqrt pi) exp(-x^2) sum_n 2^n x^(2n+1)/(2n+1)!!, positive terms; once the term ratio
    2x^2/(2n+3) is at most 1/2 the omitted tail is at most twice the first omitted term.  coarse=True stops at that point
    (a valid, wider enclosure used by TRUNCATION_NESTING); the production run continues until the tail is negligible."""
    tail = MUT != "tail-dropped"
    coarse = coarse or not tail
    if x < 0:
        return -erf_point(-x, coarse)
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
        if ratio.hi <= HALF and (coarse or t.hi < S.lo * TINY):
            break
        if n > 200000:
            raise RuntimeError("erf series did not converge")
    rest = Iv(ZERO, CC.multiply(Decimal(2), t.hi)) if tail else Iv(ZERO)
    return (S + rest) * (-x2).exp() * 2 / SQRTPI


def erf_iv(x, coarse=False):
    return Iv(erf_point(x.lo, coarse).lo, erf_point(x.hi, coarse).hi)


def Phi(x):
    return (1 + erf_iv(x / SQRT2)) / 2


def phi(x):
    return (-(x.sq()) / 2).exp() / SQRT2PI


def atan_iv(a, N=90):
    """arctan(a) for an interval 0 <= a <= 1: halve the argument, then the alternating series bracketed by consecutive
    partial sums (for each fixed argument the terms decrease, so the value lies between them)."""
    a1 = a / (1 + (1 + a.sq()).sqrt())
    a1sq, p, S, prev = a1.sq(), a1, Iv(ZERO), None
    for n in range(N + 1):
        t = p / (2 * n + 1)
        prev = S
        S = S + t if n % 2 == 0 else S - t
        p = p * a1sq
    return 2 * hull(prev, S)


def owen_T(h, a, N=160, bracket=True):
    """Owen's T(h, a) = (1/2 pi) int_0^a exp(-h^2 (1+x^2)/2)/(1+x^2) dx for 0 <= a <= 1, by Owen's series
    T = (1/2 pi)[arctan a - sum_j c_j a^(2j+1)],  c_j = (-1)^j/(2j+1) [1 - exp(-h^2/2) sum_{i<=j} (h^2/2)^i/i!].
    The bracket is P(Poisson(h^2/2) > j), decreasing in j, so for fixed (h, a) the terms alternate with decreasing size and
    the sum lies between consecutive partial sums; interval evaluation of both encloses it for every point of the intervals."""
    if MUT == "no-owen":
        return Iv(ZERO)
    h2 = h.sq() / 2
    e = (-h2).exp()
    pois, term, apow, a2 = Iv(ONE), Iv(ONE), a, a.sq()
    S, prev = Iv(ZERO), None
    for j in range(N + 1):
        if j > 0:
            term = term * h2 / j
            pois = pois + term
        cj = apow / (2 * j + 1) * (1 - e * pois)
        prev = S
        S = S + cj if j % 2 == 0 else S - cj
        apow = apow * a2
    val = hull(prev, S) if bracket else S
    return (atan_iv(a) - val) / (2 * PI)


def owen_T_eval(h, a):
    if MUT == "tail-dropped":
        return owen_T(h, a, N=6, bracket=False)
    return owen_T(h, a)


def SHAPE():
    """Shape parameter of the gap-window incomplete gamma factor (mutant gamma-shape perturbs it)."""
    return Fr(4, 3) if MUT == "gamma-shape" else Fr(7, 6)


def gamma_lower_point(x, a, coarse=False):
    """gamma(a, x) = int_0^x s^(a-1) e^-s ds = x^a e^-x sum_n x^n/(a (a+1) ... (a+n)), positive terms; once the term ratio
    x/(a+n) is at most 1/2 the omitted tail is at most twice the first omitted term. x a Decimal point >= 0, a = 7/6 or 4/3."""
    tail = MUT != "tail-dropped"
    coarse = coarse or not tail
    if x == 0:
        return Iv(ZERO)
    X = Iv(x)
    t, S, n = Iv.frac(1 / a), Iv(ZERO), 0
    while True:
        S = S + t
        n += 1
        ratio = X / Iv.frac(a + n)
        t = t * ratio
        if ratio.hi <= HALF and (coarse or t.hi < S.lo * TINY):
            break
        if n > 200000:
            raise RuntimeError("incomplete gamma series did not converge")
    rest = Iv(ZERO, CC.multiply(Decimal(2), t.hi)) if tail else Iv(ZERO)
    xa = X * root_frac(Fr(x), 6) if a == Fr(7, 6) else X * root_frac(Fr(x), 3)      # x^a
    return xa * (-X).exp() * (S + rest)


def gamma_lower(x, a, coarse=False):
    """gamma(a, x) for an interval x >= 0 or x = INF (monotone in x)."""
    if isinstance(x, float):
        return GAMMA[a]
    return Iv(gamma_lower_point(x.lo, a, coarse).lo, gamma_lower_point(x.hi, a, coarse).hi)


def gamma_complete(a, X=Decimal(120)):
    """Gamma(a) = gamma(a, X) + Gamma(a, X), 1 < a < 2, with X^(a-1) e^-X <= Gamma(a, X) <= X^(a-1) e^-X / (1 - (a-1)/X)."""
    low = gamma_lower_point(X, a)
    xa1 = root_frac(Fr(X), 6) if a == Fr(7, 6) else root_frac(Fr(X), 3)
    up = xa1 * (-Iv(X)).exp()
    return low + Iv(up.lo, (up / (1 - Iv.frac(a - 1) / Iv(X))).hi)


GAMMA = {Fr(7, 6): gamma_complete(Fr(7, 6)), Fr(4, 3): gamma_complete(Fr(4, 3))}
GAMMA76 = GAMMA[Fr(7, 6)]
CBRT32 = root_frac(Fr(12), 3) / 2                       # (3/2)^(1/3) = 12^(1/3)/2
C72 = 1 / (72 * root_frac(Fr(72), 6))                   # 72^(-7/6)
SIGMA_REF = Iv.frac(Fr(2, 3)).sqrt()                    # sd of f | V=0 for the reference kernel
D2_EXACT = Iv.frac(Fr(29, 6)) - Iv.frac(6).sqrt()        # SIDE24 cone moment m = 2
C3_REF = GAMMA76 * CBRT32 * D2_EXACT / (2 * SQRT3 * PI * PI * SQRTPI)             # SIDE24 (1), d = 3


# ---------------- the Gaussian integrals int_{-inf}^x t^n phi(t) Phi(a t) dt, n <= 4 (Math-#197 section 2, unchanged)
def G_finite(x, a):
    rho2 = 1 + a.sq()
    rho = rho2.sqrt()
    Phx, phx, Phax = Phi(x), phi(x), Phi(a * x)
    T = owen_T_eval(x, a)
    Erho = (-(rho2 * x.sq()) / 2).exp()
    Phrx = Phi(rho * x)
    x2 = x.sq()
    G0 = Phx / 2 - T
    G1 = -phx * Phax + a / (rho * SQRT2PI) * Phrx
    G2 = G0 - x * phx * Phax - a / (2 * PI * rho2) * Erho
    G3 = -x2 * phx * Phax + a / (2 * PI) * (-(x / rho2) * Erho + SQRT2PI / (rho2 * rho) * Phrx) + 2 * G1
    G4 = -x * x2 * phx * Phax - a / (2 * PI) * (x2 / rho2 + 2 / rho2.sq()) * Erho + 3 * G2
    return [G0, G1, G2, G3, G4]


def G_limit(a, sign):
    if sign < 0:
        return [Iv(ZERO)] * 5
    rho2 = 1 + a.sq()
    rho = rho2.sqrt()
    g1 = a / (rho * SQRT2PI)
    return [Iv.frac(Fr(1, 2)), g1, Iv.frac(Fr(1, 2)), a / (rho2 * rho * SQRT2PI) + 2 * g1, Iv.frac(Fr(3, 2))]


def G_at(x, a):
    """x an interval or +-INF."""
    if isinstance(x, float):
        return G_limit(a, 1 if x > 0 else -1)
    return G_finite(x, a)


def P_reg(s):
    """Regularized lower incomplete gamma P(7/6, s) (s an interval >= 0 or INF)."""
    a = SHAPE()
    return gamma_lower(s, a) / GAMMA76


def G_ref(K):
    lo, hi = K
    def s(k):
        return k if isinstance(k, float) else 12 * Iv.frac(k * k)
    return P_reg(s(hi)) - P_reg(s(lo))


def library_exact():
    """exp and sqrt of the decimal module, and the interval negation, against exact rational brackets: e^-q by the
    rational series of e^(-q/64) raised to the 64th power (alternating tail below the last term), sqrt by an integer root."""
    ok = True
    args = [Fr(48) - Fr(51035039649098175696, 10 ** 29), Fr(3), Fr(120), Fr(1, 2), Fr(288), Fr(5, 7)]
    for q in args:
        g = -q / 64
        parts = []
        term, ssum = Fr(1), Fr(0)
        for k in range(0, 60):
            ssum += term
            parts.append(ssum)
            term = term * g / (k + 1)
        lo, hi = min(parts[-1], parts[-2]), max(parts[-1], parts[-2])           # alternating series with decreasing terms
        exact = Iv(Iv.frac(lo ** 64).lo, Iv.frac(hi ** 64).hi)                  # e^-q = (e^(-q/64))^64, both endpoints positive
        X = Iv.frac(q)
        got = (-X).exp()
        ok &= got.intersects(exact) and CC.divide(got.width(), got.lo) < Decimal(10) ** (-PREC + 3)
        r = root_frac(q, 2)
        ok &= X.sqrt().intersects(r) and CC.divide(X.sqrt().width(), r.lo) < Decimal(10) ** (-PREC + 3)
        ok &= (-X).lo == X.hi.copy_negate() and (-X).hi == X.lo.copy_negate()
    return ok


def simpson(f, lo, hi, n=4000):
    h = (hi - lo) / n
    s = f(lo) + f(hi)
    for i in range(1, n):
        s += (4 if i % 2 else 2) * f(lo + i * h)
    return s * h / 3


def fPhi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def fphi(x):
    return math.exp(-x * x / 2) / math.sqrt(2 * math.pi)


def m3_float(b):                                  # direct double integral: int_0^inf g(v^2) phi(v - b) dv, g from the Rayleigh integral
    def g(w):
        return w * w - 4 * w + 8 - 8 * math.exp(-w / 2)
    return simpson(lambda v: g(v * v) * fphi(v - b), 0.0, 14.0 + abs(b), 1200)



# ---------------- the d = 3 jet of the reference kernel exp(-|z|^2/2), frame (u, w1, w2)
# Cov(d^a f, d^b f) = (-1)^|a| d^(a+b) K(0), and d^al exp(-|z|^2/2) at 0 = prod_i (-1)^(al_i/2) (al_i - 1)!! for even al, else 0.
EVEN = [(0, 0, 0), (2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)]   # f, f_uu, f_uw1, f_uw2, A11, A12, A22
ODD = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (3, 0, 0)]                                    # f_u, f_w1, f_w2, t = f_uuu
IDX_X, IDX_V = [0, 4, 5, 6], [1, 2, 3]                                                 # (f, A11, A12, A22) and V = (f_uu, f_uw1, f_uw2)


def dfact2(n):
    r = 1
    while n > 1:
        r *= n
        n -= 2
    return r


def ref_cov(a, b):
    al = [x + y for x, y in zip(a, b)]
    if any(x % 2 for x in al):
        return Fr(0)
    v = Fr((-1) ** sum(a))
    for x in al:
        v *= (-1) ** (x // 2) * dfact2(x - 1)
    if MUT == "tau-cross" and sorted([a, b]) == sorted([(3, 0, 0), (1, 0, 0)]):
        return Fr(0)
    return v


def ref_matrix(idx):
    return [[ref_cov(a, b) for b in idx] for a in idx]


def det3(M):
    return (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
            + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))


def inv3(M):
    """Inverse by the adjugate; works for Fractions and for intervals."""
    d = det3(M)
    cof = [[None] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            r = [x for x in range(3) if x != i]
            c = [y for y in range(3) if y != j]
            minor = M[r[0]][c[0]] * M[r[1]][c[1]] - M[r[0]][c[1]] * M[r[1]][c[0]]
            cof[i][j] = minor if (i + j) % 2 == 0 else -minor
    return [[cof[j][i] / d for j in range(3)] for i in range(3)], d


def schur(C, X, V):
    """Cov(X | V) = C_XX - C_XV C_VV^-1 C_VX for index lists into C (Fractions or intervals)."""
    CVV = [[C[i][j] for j in V] for i in V]
    Vinv, detV = inv3(CVV)
    out = []
    for i in X:
        row = []
        for j in X:
            s = None
            for p in range(3):
                for q in range(3):
                    t = C[i][V[p]] * Vinv[p][q] * C[V[q]][j]
                    s = t if s is None else s + t
            row.append(C[i][j] + s if MUT == "schur-sign" else C[i][j] - s)
        out.append(row)
    return out, detV


def sylvester_pd(M):
    """Exact positive definiteness of a symmetric rational matrix by its leading principal minors (fraction-free elimination)."""
    n = len(M)
    A = [list(map(Fr, r)) for r in M]
    for k in range(n):
        if A[k][k] <= 0:
            return False
        for i in range(k + 1, n):
            f = A[i][k] / A[k][k]
            for j in range(k, n):
                A[i][j] -= f * A[k][j]
    return True


# exact reference: C'_ref = Cov((f, A11, A12, A22) | V = 0) and its smallest-eigenvalue floor
C_EVEN_REF = ref_matrix(EVEN)
C_ODD_REF = ref_matrix(ODD)
CPRIME_REF, DETV_REF = schur(C_EVEN_REF, IDX_X, IDX_V)
LAMBDA0 = Fr(1) if MUT == "lambda-min" else Fr(1, 3)


def lambda_floor_ok():
    """C'_ref - LAMBDA0 I is positive definite (exact), so C'_ref >= LAMBDA0 I; the smallest eigenvalue is 2 - sqrt(8/3)."""
    M = [[CPRIME_REF[i][j] - (LAMBDA0 if i == j else 0) for j in range(4)] for i in range(4)]
    return sylvester_pd(M)


# ---------------- image bound for the torus kernel in d = 3
def image_bound3(L):
    """E^(3)_L = 1404 (76 L^6 + 15) exp(-L^2/2): the shell |n|_inf = j carries (2j+1)^3 - (2j-1)^3 = 24 j^2 + 2 <= 26 j^2 points
    with |n|^2 <= 3 j^2, so sum_{n != 0} |n|^6 exp(-L^2 |n|^2/2) <= 702 sum_j j^8 exp(-L^2 j^2/2) <= 1404 exp(-L^2/2) (successive
    terms ratio at most 2^8 exp(-3L^2/2) < 1/2 for L >= 10); the contraction bound 76|x|^6 phi(x) and the normalization S >= 1
    are those of Math-#197 section 3."""
    shells = 13 if MUT == "image-shells" else 26
    L = Fr(L)
    return Iv.frac(2 * shells * 27 * (76 * L ** 6 + 15)) * (-Iv.frac(L * L / 2)).exp()


# ---------------- the d = 3 pipeline over an entry box
class Space3:
    """Every covariance entry of the 3-jet (other than Var f = 1, exact by the normalization) in [ref - E, ref + E].
    Odd block (G, t) independent of the even block (f, V, A) for any even kernel ([LP] section 15)."""

    def __init__(self, E):
        E = Iv.of(E)
        def box(M):
            n = len(M)
            B = [[None] * n for _ in range(n)]
            for i in range(n):
                for j in range(i, n):
                    v = Iv.frac(M[i][j]) + Iv(-E.hi, E.hi)
                    B[i][j] = B[j][i] = v
            return B
        CE_ = box(C_EVEN_REF)
        CE_[0][0] = Iv(ONE)
        CO = box(C_ODD_REF)
        G = [[CO[i][j] for j in range(3)] for i in range(3)]
        Ginv, detG = inv3(G)
        c = [CO[3][j] for j in range(3)]
        self.tau2 = CO[3][3] - sum((c[i] * Ginv[i][j] * c[j] for i in range(3) for j in range(3)), Iv(ZERO))
        self.pG = 1 / (2 * PI * SQRT2PI * detG.sqrt())                        # (2 pi)^(-3/2) det(Cov G)^(-1/2)
        self.Cp, detV = schur(CE_, IDX_X, IDX_V)
        self.pV = 1 / (2 * PI * SQRT2PI * detV.sqrt())
        self.tau43 = root_iv(self.tau2.sq(), 3)
        # epsilon: ||C_ref'^(-1/2) (C' - C_ref') C_ref'^(-1/2)|| <= ||C' - C_ref'||_F / LAMBDA0, entrywise sup of |C'_ij - ref|
        dev2 = Iv(ZERO)
        for i in range(4):
            for j in range(4):
                r = Iv.frac(CPRIME_REF[i][j])
                d = max(abs(CC.subtract(self.Cp[i][j].hi, r.lo)), abs(CF.subtract(self.Cp[i][j].lo, r.hi)))
                dev2 = dev2 + Iv(d).sq()
        self.eps = (dev2.sqrt() / Iv.frac(LAMBDA0)).hi

    def k_integral(self, K):
        lo, hi = K
        def s(k):
            return k if isinstance(k, float) else 72 * Iv.frac(k * k) / self.tau2
        a = SHAPE()
        return self.tau43 * C72 * (gamma_lower(s(hi), a) - gamma_lower(s(lo), a)) / (2 * SQRT2PI)

    def b_integral(self, B):
        return sandwich(B, self.eps)

    def coefficient(self, B, K):
        sphere = 2 * PI if MUT == "sphere" else 4 * PI
        return 144 * sphere * self.pG * self.pV * self.b_integral(B) * self.k_integral(K)


# ---------------- the d = 3 reference birth integral with interval endpoints (Math-#197 (4.1) and its three primitives)
def endpoint(b):
    return b if isinstance(b, (float, Iv)) else Iv.frac(b)


def I3_ref(lo, hi):
    """int_lo^hi phi_{2/3}(b) m_{3,b} db, m_{3,b} = (b^3 + b) phi(b) + (b^4 + 2 b^2 + 7) Phi(b) - 4 sqrt2 exp(-b^2/4) Phi(b/sqrt2);
    lo, hi are rationals, intervals or +-INF."""
    sig = SIGMA_REF
    def term_i(b):
        if isinstance(b, float):
            return Iv(ZERO)
        return -(b.sq() * 2 / 5 + Iv.frac(Fr(18, 25))) * (-(b.sq()) * 5 / 4).exp() / (2 * PI * sig)
    def term_ii(b):
        G = G_at(b if isinstance(b, float) else b / sig, sig)
        return sig.sq().sq() * G[4] + 2 * sig.sq() * G[2] + 7 * G[0]
    def term_iii(b):
        G = G_at(b if isinstance(b, float) else b * SQRT2, Iv.frac(Fr(1, 2)))
        return 4 / sig * G[0]
    lo, hi = endpoint(lo), endpoint(hi)
    cross = Iv.frac(Fr(1, 2)) if MUT == "d3-cross" else Iv(ONE)
    return (term_i(hi) - term_i(lo)) + (term_ii(hi) - term_ii(lo)) - cross * (term_iii(hi) - term_iii(lo))


def scaled(b, s):
    if isinstance(b, float):
        return b
    b = endpoint(b)
    return b * s if MUT == "window-scale" else b / s


def sandwich(B, eps):
    """I'_B for every covariance C' of (f, A) | V=0 with (1-eps) C_ref' <= C' <= (1+eps) C_ref':
    (1-eps)^4/(1+eps)^2 I^ref_{B/sqrt(1-eps)} <= I'_B <= (1+eps)^4/(1-eps)^2 I^ref_{B/sqrt(1+eps)}  (NOTE section 2)."""
    e = Iv(eps) if isinstance(eps, Decimal) else Iv.frac(eps)
    if e.hi >= 1:
        raise ValueError("eps must be below 1")
    up, dn = 1 + e, 1 - e
    s_up, s_dn = up.sqrt(), dn.sqrt()
    lo, hi = B
    upper = up.sq().sq() / dn.sq() * I3_ref(scaled(lo, s_up), scaled(hi, s_up))
    lower = dn.sq().sq() / up.sq() * I3_ref(scaled(lo, s_dn), scaled(hi, s_dn))
    if lower.lo > upper.hi:
        raise ValueError("sandwich bounds out of order")
    return Iv(lower.lo, upper.hi)


def scaled_exact(B, eta):
    """Exact I'_B for C' = (1 + eta) C_ref': (1 + eta)^2 I^ref_{B/sqrt(1+eta)} (degree-4 homogeneity of det(A)^2)."""
    s = (1 + Iv.frac(eta)).sqrt()
    lo, hi = B
    return (1 + Iv.frac(eta)).sq() * I3_ref(lo if isinstance(lo, float) else endpoint(lo) / s, hi if isinstance(hi, float) else endpoint(hi) / s)


# ---------------- float control for a non-scalar perturbation (not part of the certificate)
def gauss_legendre(n):
    xs, ws = [], []
    for i in range(1, n + 1):
        x = math.cos(math.pi * (i - 0.25) / (n + 0.5))
        for _ in range(100):
            p0, p1 = 1.0, x
            for k in range(2, n + 1):
                p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
            dp = n * (x * p1 - p0) / (x * x - 1)
            dx = p1 / dp
            x -= dx
            if abs(dx) < 1e-15:
                break
        xs.append(x)
        ws.append(2 / ((1 - x * x) * dp * dp))
    return xs, ws


def cone_moment_float(mu, Sig, b, nodes):
    """E[det(A)^2 1{A < 0}] for A = [[a, c], [c, d]], (a, c, d) ~ N(b mu, Sig): the c-integral over |c| < sqrt(ad) in closed form
    (truncated normal moments of orders 0, 2, 4), then Gauss-Legendre over a, d in (-R, 0)."""
    xs, ws = nodes
    ma, mc, md = (b * m for m in mu)
    saa, sac, sad = Sig[0]
    scc, scd, sdd = Sig[1][1], Sig[1][2], Sig[2][2]
    det2 = saa * sdd - sad * sad
    ia, iad, idd = sdd / det2, -sad / det2, saa / det2
    ka, kd = (sac * sdd - scd * sad) / det2, (scd * saa - sac * sad) / det2           # regression of c on (a, d)
    vc = scc - (ka * sac + kd * scd)
    sv = math.sqrt(vc)
    R = 12 * math.sqrt(max(saa, sdd)) + abs(ma) + abs(md)
    tot = 0.0
    for xa, wa in zip(xs, ws):
        a = -R * (xa + 1) / 2
        for xd, wd in zip(xs, ws):
            d = -R * (xd + 1) / 2
            ya, yd = a - ma, d - md
            q = ia * ya * ya + 2 * iad * ya * yd + idd * yd * yd
            dens = math.exp(-q / 2) / (2 * math.pi * math.sqrt(det2))
            RR = a * d
            r = math.sqrt(RR)
            m = mc + ka * ya + kd * yd
            z1, z2 = (-r - m) / sv, (r - m) / sv
            p1, p2 = fphi(z1), fphi(z2)
            mom = [fPhi(z2) - fPhi(z1), p1 - p2]
            for j in range(2, 5):
                mom.append((j - 1) * mom[j - 2] + z1 ** (j - 1) * p1 - z2 ** (j - 1) * p2)
            def M(k):
                return sum(math.comb(k, j) * m ** (k - j) * sv ** j * mom[j] for j in range(k + 1))
            tot += wa * wd * dens * (RR * RR * M(0) - 2 * RR * M(2) + M(4))
    return tot * (R / 2) ** 2


def I_float(Cp, B, nb=40, na=48):
    """int_B p_f(b) E[det(A)^2 1{A<0} | f = b] db for the covariance Cp of (f, A11, A12, A22) (floats)."""
    s2 = Cp[0][0]
    mu = [Cp[i][0] / s2 for i in (1, 2, 3)]
    Sig = [[Cp[i][j] - Cp[i][0] * Cp[0][j] / s2 for j in (1, 2, 3)] for i in (1, 2, 3)]
    nodes = gauss_legendre(na)
    xs, ws = gauss_legendre(nb)
    lo, hi = float(B[0]), float(B[1])
    tot = 0.0
    for x, w in zip(xs, ws):
        b = lo + (hi - lo) * (x + 1) / 2
        tot += w * math.exp(-b * b / (2 * s2)) / math.sqrt(2 * math.pi * s2) * cone_moment_float(mu, Sig, b, nodes)
    return tot * (hi - lo) / 2


CONTROL_DELTA = {(0, 5): Fr(1, 1000), (4, 4): Fr(2, 1000), (4, 6): Fr(1, 1000), (1, 5): Fr(-1, 1000), (2, 4): Fr(1, 1000)}


def control_matrices():
    C = [list(r) for r in C_EVEN_REF]
    for (i, j), v in CONTROL_DELTA.items():
        C[i][j] += v
        if i != j:
            C[j][i] += v
    Cp, _ = schur(C, IDX_X, IDX_V)
    dev2 = sum((Cp[i][j] - CPRIME_REF[i][j]) ** 2 for i in range(4) for j in range(4))
    eps = (Iv.frac(dev2).sqrt() / Iv.frac(LAMBDA0)).hi
    return Cp, eps
