"""Rigorous interval arithmetic on IEEE-754 binary64 (round-to-nearest +, -, *, /, sqrt), widened outward by one ulp
after every operation; no libm transcendental is trusted (exp and Phi are computed by series with explicit remainders).
An interval is a pair (lo, hi) of floats with lo <= hi."""
import math
from fractions import Fraction as Fr

INF = math.inf
_na = math.nextafter


def dn(x):
    return _na(x, -INF)


def up(x):
    return _na(x, INF)


def iv(x):
    """Exact float point as an interval."""
    return (x, x)


def from_frac(q):
    """Tightest float interval containing the rational q."""
    f = float(q)
    fq = Fr(f)
    if fq == q:
        return (f, f)
    return (f, up(f)) if fq < q else (dn(f), f)


def add(a, b):
    return (dn(a[0] + b[0]), up(a[1] + b[1]))


def sub(a, b):
    return (dn(a[0] - b[1]), up(a[1] - b[0]))


def neg(a):
    return (-a[1], -a[0])


def mul(a, b):
    p = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    return (dn(min(p)), up(max(p)))


def div(a, b):
    if b[0] <= 0.0 <= b[1]:
        raise ZeroDivisionError('interval division by an interval containing 0')
    p = (a[0] / b[0], a[0] / b[1], a[1] / b[0], a[1] / b[1])
    return (dn(min(p)), up(max(p)))


def scal(c, a):
    """c * a for an exact float c."""
    if c >= 0:
        return (dn(c * a[0]), up(c * a[1]))
    return (dn(c * a[1]), up(c * a[0]))


def divc(a, c):
    """a / c for an exact float c > 0."""
    return (dn(a[0] / c), up(a[1] / c))


def sqr(a):
    if a[0] >= 0:
        return (max(0.0, dn(a[0] * a[0])), up(a[1] * a[1]))
    if a[1] <= 0:
        return (max(0.0, dn(a[1] * a[1])), up(a[0] * a[0]))
    m = max(-a[0], a[1])
    return (0.0, up(m * m))


def pw(a, n):
    """a ** n for an integer n >= 0 (monotone pieces)."""
    if n == 0:
        return (1.0, 1.0)
    if n == 1:
        return a
    if n % 2 == 1:
        lo, hi = a[0], a[1]
        r_lo, r_hi = lo, hi
        # odd power is monotone: bound by repeated outward multiplication on each endpoint
        return (_pw_pt(lo, n, False), _pw_pt(hi, n, True))
    r = pw(sqr(a), n // 2)
    return (max(0.0, r[0]), r[1])


def _pw_pt(x, n, upper):
    """Outward bound of x ** n at a point (odd n): upper -> rounded up, else rounded down."""
    r = (x, x)
    for _ in range(n - 1):
        r = mul(r, (x, x))
    return r[1] if upper else r[0]


def hull(*ivs):
    return (min(x[0] for x in ivs), max(x[1] for x in ivs))


def absmax(a):
    return max(-a[0], a[1])


def isqrt(a):
    if a[0] < 0:
        raise ValueError('sqrt of a negative interval')
    lo = math.sqrt(a[0])
    hi = math.sqrt(a[1])
    return (max(0.0, dn(lo)), up(hi))


def contains(a, x):
    return a[0] <= x <= a[1]


# ---------------------------------------------------------------- constants (exact rationals, then outward floats)
def _pi_frac(digits=40):
    """pi by Machin's formula with an explicit rational enclosure."""
    def arctan_inv(n, terms):
        s = Fr(0)
        for k in range(terms):
            s += Fr((-1) ** k, (2 * k + 1) * n ** (2 * k + 1))
        return s
    T = 40
    lo = 16 * arctan_inv(5, T) - 4 * arctan_inv(239, T)
    err = Fr(16, (2 * T + 1) * 5 ** (2 * T + 1)) + Fr(4, (2 * T + 1) * 239 ** (2 * T + 1))
    return lo - err, lo + err


_PI_LO, _PI_HI = _pi_frac()
PI = (from_frac(_PI_LO)[0], from_frac(_PI_HI)[1])
SQRT2PI = isqrt(scal(2.0, PI))
SQRT2 = isqrt((2.0, 2.0))
INV_SQRT2PI = div((1.0, 1.0), SQRT2PI)


# ---------------------------------------------------------------- exp(-y) for y >= 0
_INVFACT = [1.0]
for _i in range(1, 19):
    _INVFACT.append(_INVFACT[-1] / _i)
_U = 2.0 ** -53


def _ln2_frac():
    """ln 2 = sum_{j>=1} 1/(j 2^j), with the tail after J terms below 2/(J 2^J)."""
    J = 80
    s = Fr(0)
    for j in range(1, J + 1):
        s += Fr(1, j * 2 ** j)
    return s, s + Fr(2, J * 2 ** J)


_L2LO, _L2HI = _ln2_frac()
LN2 = (from_frac(_L2LO)[0], from_frac(_L2HI)[1])


def _poly_exp_neg(z):
    """Binary64 Horner value p of the degree-18 Taylor polynomial of exp(-z) at a float z with |z| <= 0.75, and a bound
    of |p - exp(-z)|: Horner and coefficient rounding (coefficients 1/i! carry relative error <= 18u) give at most
    (2*18 + 18 + 2) u e^|z| (1 + 1e-12), truncation at most |z|^19/19! e^|z| < 1e-19 e^|z|; exp(-z) >= e^-|z| >= 0.47."""
    p = _INVFACT[18]
    for i in range(17, -1, -1):
        p = _INVFACT[i] - z * p
    ez = 2.12                                   # >= e^0.75
    err = (56.0 * _U * (1.0 + 1e-12) + 1e-19) * ez
    return p, err


def _exp_neg_point(y):
    """Enclosure of exp(-y) for a float y >= 0: y = n ln2 + r with n = floor(y / ln2) and r in an interval inside
    [-1e-9, 0.75] (from the rational enclosure of ln 2), exp(-y) = 2^-n exp(-r) (exact power-of-two scaling)."""
    if y == 0.0:
        return (1.0, 1.0)
    if y > 700.0:
        return (0.0, math.ldexp(1.0, -1000))     # exp(-700) < 2^-1000
    n = int(y / 0.6931471805599453)
    nL = mul((float(n), float(n)), LN2)
    r = sub((y, y), nL)
    if not (-1e-9 <= r[0] <= r[1] <= 0.75):
        raise RuntimeError('exp range reduction failed at y=%r' % y)
    plo, elo = _poly_exp_neg(r[1])               # exp(-r) is decreasing: lower end at r[1]
    phi_, ehi = _poly_exp_neg(r[0])
    lo = dn(plo - elo)
    hi = up(phi_ + ehi)
    return (max(0.0, math.ldexp(lo, -n)), math.ldexp(hi, -n))


def exp_neg(a):
    """Enclosure of exp(-y) for y in the interval a, a[0] >= 0 (decreasing)."""
    if a[0] < 0:
        # a lower bound that is negative only by underflow-level rounding (y >= -1e-300): exp(-y) <= exp(1e-300) < up(1)
        if a[0] < -1e-300:
            raise ValueError('exp_neg needs y >= 0')
        return (_exp_neg_point(max(0.0, a[1]))[0], up(1.0))
    return (_exp_neg_point(a[1])[0], _exp_neg_point(a[0])[1])


def phi_std(t):
    """Standard normal density at the interval t."""
    return mul(exp_neg(divc(sqr(t), 2.0)), INV_SQRT2PI)


def _Phi_pos_point(t):
    """Enclosure of Phi(t) for a float t >= 0: Phi(t) = 1/2 + phi(t) S(t), S(t) = sum t^(2n+1)/(2n+1)!! (positive terms),
    summed in binary64 with a priori relative error <= (3N + 3) u <= 1e-13 (N <= 300), plus the geometric tail bound once
    the term ratio t^2/(2n+3) is below 1/2."""
    if t == 0.0:
        return (0.5, 0.5)
    if t < 1e-100:
        return (0.5, up(0.5 + t))
    if t > 9.0:
        ph = phi_std((t, t))
        return (dn(1.0 - up(ph[1] / t)), 1.0)
    t2 = t * t
    term = t
    S = t
    n = 0
    while True:
        n += 1
        if n > 300:
            raise RuntimeError('Phi series did not terminate at t=%r' % t)
        term = term * t2 / (2 * n + 1)
        S += term
        ratio = t2 * (1.0 + 4 * _U) / (2 * n + 3)
        if ratio < 0.5 and term < 1e-17 * S:
            tail = term * ratio / (1.0 - ratio) * (1.0 + 1e-12)
            break
    Slo, Shi = dn(S * (1.0 - 1e-13)), up((S + tail) * (1.0 + 1e-13))
    r = add((0.5, 0.5), mul(phi_std((t, t)), (Slo, Shi)))
    return (r[0], min(r[1], 1.0))


def _mills(t):
    """Enclosure of the Mills ratio R(t) = Q(t)/phi(t), t >= 2 float: Laplace's continued fraction
    R(t) = 1/(t + 1/(t + 2/(t + 3/(t + ...)))) has positive partial numerators and denominators, so consecutive
    approximants bracket R(t) (Stieltjes); the hull of the depth-n and depth-(n+1) approximants is returned once its
    relative width is below 1e-14 (each approximant evaluated in outward interval arithmetic)."""
    ti = (t, t)
    n = 16
    while True:
        res = []
        for depth in (n, n + 1):
            v = ti
            for j in range(depth, 0, -1):
                v = add(ti, div((float(j), float(j)), v))
            res.append(div((1.0, 1.0), v))
        h = (min(res[0][0], res[1][0]), max(res[0][1], res[1][1]))
        if h[1] - h[0] <= 1e-14 * h[0] or n > 4000:
            return h
        n *= 2


def Q_point(t):
    """Enclosure of the upper tail Q(t) = 1 - Phi(t) for a float t >= 0, with relative precision for large t."""
    if t >= 2.0:
        return mul(phi_std((t, t)), _mills(t))
    r = _Phi_pos_point(t)
    return (max(0.0, dn(1.0 - r[1])), up(1.0 - r[0]))


def Phi_point(t):
    if t >= 0:
        if t >= 2.0:
            q = Q_point(t)
            return (dn(1.0 - q[1]), min(1.0, up(1.0 - q[0])))
        return _Phi_pos_point(t)
    q = Q_point(-t)
    return q


def Phi(a):
    """Normal CDF over an interval argument (increasing)."""
    return (Phi_point(a[0])[0], Phi_point(a[1])[1])
