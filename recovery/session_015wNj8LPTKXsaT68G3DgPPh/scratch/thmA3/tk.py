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


