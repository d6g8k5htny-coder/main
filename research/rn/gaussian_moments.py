"""Exact correlated Gaussian determinant moments with affine mark means.

This implements an algebraic candidate, not an RN field-law or spatial
certificate. Source identification, conditioning, and uniform covariance
families are separate obligations. No claim, grade or gate changes here.
"""
from dataclasses import dataclass
from fractions import Fraction as F
from math import comb

from research.interval import Interval

ORDER = ("xx", "yy", "xy")


def rational(x):
    if type(x) not in (int, F):
        raise TypeError("exact int/Fraction required; no float or bool")
    return F(x)


@dataclass
class Cost:
    """Explicit finite cost model, not timing or general usefulness."""
    budget: int = 1_000_000
    used: int = 0

    def charge(self, n=1):
        self.used += n
        if self.used > self.budget:
            raise RuntimeError("explicit arithmetic budget exhausted")

    def add(self, a, b):
        self.charge()
        return a + b

    def mul(self, a, b):
        self.charge()
        return a * b

    def div(self, a, b):
        self.charge()
        return a / b


def trim(p):
    p = tuple(p)
    while len(p) > 1 and not p[-1]:
        p = p[:-1]
    return p


def add(a, b, cost):
    return trim(tuple(cost.add(a[i] if i < len(a) else F(0),
                               b[i] if i < len(b) else F(0))
                      for i in range(max(len(a), len(b)))))


def multiply(a, b, cost):
    out = [F(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i+j] = cost.add(out[i+j], cost.mul(x, y))
    return trim(out)


def evaluate(p, t, cost=None):
    c = cost if cost is not None else Cost()
    t = rational(t)
    value = F(0)
    for x in reversed(p):
        value = c.add(c.mul(value, t), rational(x))
    return value


def covariance_psd(covariance, cost=None):
    """Exact Schur elimination including singular PSD matrices, no tolerance."""
    c = cost if cost is not None else Cost()
    if len(covariance) != 3 or any(len(row) != 3 for row in covariance):
        raise ValueError("3 by 3 covariance required")
    s = tuple(tuple(rational(x) for x in row) for row in covariance)
    for i in range(3):
        for j in range(3):
            c.charge()
            if s[i][j] != s[j][i]:
                raise ValueError("asymmetric covariance")
    a = [list(row) for row in s]
    for k in range(3):
        c.charge()
        if a[k][k] < 0:
            raise ValueError("covariance is not positive semidefinite")
        if a[k][k] == 0:
            for j in range(k+1, 3):
                c.charge()
                if a[k][j] != 0:
                    raise ValueError("zero PSD pivot has nonzero row")
            continue
        for i in range(k+1, 3):
            for j in range(k+1, 3):
                a[i][j] = c.add(a[i][j], -c.div(c.mul(a[i][k], a[k][j]), a[k][k]))
    return s


class MomentEngine:
    """Memoized Stein recurrence, order (xx, yy, xy), total degree at most 8."""

    def __init__(self, intercept, slope, covariance, *, order=ORDER, cost=None):
        if tuple(order) != ORDER:
            raise ValueError("coordinate order must be (xx, yy, xy)")
        if len(intercept) != 3 or len(slope) != 3:
            raise ValueError("three affine coordinate means required")
        self.cost = cost if cost is not None else Cost()
        self.mean = tuple(trim((rational(a), rational(b))) for a, b in zip(intercept, slope))
        self.cov = covariance_psd(covariance, self.cost)
        self.cache = {(0, 0, 0): (F(1),)}
        self.cost.charge()  # cache insertion

    def moment(self, alpha):
        alpha = tuple(alpha)
        if len(alpha) != 3 or any(type(x) is not int or x < 0 for x in alpha) or sum(alpha) > 8:
            raise ValueError("three nonnegative exponents, total at most 8 required")
        self.cost.charge()  # cache lookup
        if alpha in self.cache:
            return self.cache[alpha]
        i = next(i for i, a in enumerate(alpha) if a)
        beta = list(alpha)
        beta[i] -= 1
        p = multiply(self.mean[i], self.moment(beta), self.cost)
        for j in range(3):
            if beta[j]:
                gamma = beta.copy()
                gamma[j] -= 1
                weight = self.cost.mul(F(beta[j]), self.cov[i][j])
                p = add(p, multiply((weight,), self.moment(gamma), self.cost), self.cost)
        self.cost.charge()  # cache insertion
        self.cache[alpha] = p
        return p

    def determinant(self, degree):
        if type(degree) is not int or degree not in (2, 4):
            raise ValueError("determinant degree must be 2 or 4")
        p = (F(0),)
        for j in range(degree+1):
            weight = F((-1)**j * comb(degree, j))
            term = self.moment((degree-j, degree-j, 2*j))
            p = add(p, multiply((weight,), term, self.cost), self.cost)
        return p


def second_moment_trace(mean, covariance):
    """Independent E[(X'QX)^2] identity; Q encodes xx*yy-xy^2."""
    m = tuple(rational(x) for x in mean)
    if len(m) != 3:
        raise ValueError("three means required")
    s = covariance_psd(covariance)
    q = ((F(0), F(1, 2), F(0)), (F(1, 2), F(0), F(0)), (F(0), F(0), F(-1)))
    qm = [sum(q[i][j]*m[j] for j in range(3)) for i in range(3)]
    qs = [[sum(q[i][k]*s[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    expectation = sum(m[i]*qm[i] + qs[i][i] for i in range(3))
    return (expectation**2 + 2*sum(qs[i][j]*qs[j][i] for i in range(3) for j in range(3))
            + 4*sum(qm[i]*s[i][j]*qm[j] for i in range(3) for j in range(3)))


def bernstein_enclosure(polynomial, domain, *, depth=0):
    """Full closed-interval enclosure, including all dyadic subintervals."""
    p = trim(tuple(rational(x) for x in polynomial))
    if not p or not isinstance(domain, Interval):
        raise ValueError("nonempty polynomial and Interval required")
    if type(depth) is not int or not 0 <= depth <= 12:
        raise ValueError("subdivision depth must be 0..12")
    if depth:
        mid = (domain.lo + domain.hi)/2
        left = bernstein_enclosure(p, Interval(domain.lo, mid), depth=depth-1)
        right = bernstein_enclosure(p, Interval(mid, domain.hi), depth=depth-1)
        return Interval(min(left.lo, right.lo), max(left.hi, right.hi))
    n = len(p)-1
    lo, width = domain.lo, domain.hi-domain.lo
    power = [sum(p[k]*comb(k, j)*lo**(k-j)*width**j for k in range(j, n+1)) for j in range(n+1)]
    b = [sum(power[j]*F(comb(k, j), comb(n, j)) for j in range(k+1)) for k in range(n+1)]
    return Interval(min(b), max(b))


def coefficient_cap(polynomial, domain):
    """RN5's centered absolute-coefficient cap (also valid on any domain)."""
    h = max(abs(domain.lo), abs(domain.hi))
    return rational(polynomial[0]) + sum(abs(rational(x))*h**j for j, x in enumerate(polynomial[1:], 1))


def moment_cap(polynomial, domain, *, depth=2):
    """Use either valid upper bound; no weakening when Bernstein is loose."""
    result = min(coefficient_cap(polynomial, domain), bernstein_enclosure(polynomial, domain, depth=depth).hi)
    if result < 0:
        raise ValueError("negative upper bound cannot describe an even moment")
    return result


def holder_pow4(caps):
    """Best of the three valid (4,4,2) assignments; no joint independence."""
    if len(caps) != 3 or any(set(c) != {2, 4} for c in caps):
        raise ValueError("three dictionaries of second/fourth moment caps required")
    for cap in caps:
        if any(rational(x) < 0 for x in cap.values()):
            raise ValueError("negative even-moment cap")
    return min(caps[i][2]**2*caps[(i+1)%3][4]*caps[(i+2)%3][4] for i in range(3))


def cauchy_schwarz_squared(second_moment, event_probability):
    """(E[|D| 1_A])^2 <= E[D^2] P(A), under one fixed common law."""
    second_moment, event_probability = rational(second_moment), rational(event_probability)
    if second_moment < 0 or not 0 <= event_probability <= 1:
        raise ValueError("nonnegative moment and probability in [0,1] required")
    return second_moment*event_probability
