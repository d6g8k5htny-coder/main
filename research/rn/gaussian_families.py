"""Exact interval families of affine Gaussian determinant moments.

Bounds quantify every fixed PSD covariance member, constant in the mark.
No actual RN field identification, spatial cover, novelty, utility, scientific
status change or organizational independence is established.
"""
from fractions import Fraction as F
from math import comb

from research.interval import Interval
from research.rn.gaussian_moments import (ORDER, Cost, MomentEngine,
    bernstein_enclosure, covariance_psd, rational)


def interval(x):
    if isinstance(x, Interval):
        return x
    return Interval.exact(rational(x))


def point(x):
    return x.lo == x.hi


def trim_intervals(p):
    p = tuple(interval(x) for x in p)
    while len(p) > 1 and p[-1].lo == p[-1].hi == 0:
        p = p[:-1]
    return p


def covariance_family(covariance):
    """Sufficient all-PSD test; otherwise quantify PSD members only.

    Symmetric pairs denote the SAME uncertain scalar, not independent draws.
    A PSD midpoint establishes nonemptiness, never full-box feasibility.
    """
    if len(covariance) != 3 or any(len(row) != 3 for row in covariance):
        raise ValueError('3 by 3 covariance family required')
    s = tuple(tuple(interval(x) for x in row) for row in covariance)
    if any(s[i][j] != s[j][i] for i in range(3) for j in range(3)):
        raise ValueError('symmetric covariance intervals required')
    if any(s[i][i].hi < 0 for i in range(3)):
        raise ValueError('negative diagonal excludes all PSD members')
    singleton = all(point(x) for row in s for x in row)
    mid = tuple(tuple((x.lo+x.hi)/2 for x in row) for row in s)
    try:
        covariance_psd(mid)
        witness = mid
    except ValueError:
        if singleton:
            raise ValueError('singleton covariance is not PSD') from None
        witness = None
    margins = tuple(s[i][i].lo-sum(max(abs(s[i][j].lo),abs(s[i][j].hi))
                                 for j in range(3) if j != i) for i in range(3))
    all_psd = singleton or all(x >= 0 for x in margins)
    return s, {'quantifier':'ALL_BOX_MEMBERS' if all_psd else 'PSD_MEMBERS_ONLY',
               'all_members_psd':all_psd,
               'criterion':'EXACT_SINGLETON_PSD' if singleton else 'SYMMETRIC_DIAGONAL_DOMINANCE' if all_psd else 'SUFFICIENT_TEST_INCONCLUSIVE',
               'diagonal_dominance_margins':margins,
               'nonempty_established':witness is not None, 'psd_witness':witness}


class FamilyMomentEngine(MomentEngine):
    """Stein recurrence with containing coefficient intervals, degree <= 8.

    Uses the existing polynomial recurrence and interval arithmetic. Repeated
    uncertain parameters may widen results; no independence is asserted.
    """
    def __init__(self, intercept, slope, covariance, *, order=ORDER, cost=None):
        if tuple(order) != ORDER:
            raise ValueError('coordinate order must be (xx, yy, xy)')
        if len(intercept) != 3 or len(slope) != 3:
            raise ValueError('three affine coordinate means required')
        self.cost = cost if cost is not None else Cost()
        self.intercept = tuple(interval(x) for x in intercept)
        self.slope = tuple(interval(x) for x in slope)
        self.mean = tuple(trim_intervals((a,b)) for a,b in zip(self.intercept,self.slope))
        self.cov, self.feasibility = covariance_family(covariance)
        self.cache = {(0,0,0):(Interval.exact(1),)}
        self.cost.charge()

    def moment(self, alpha):
        return trim_intervals(super().moment(alpha))

    def determinant(self, degree):
        return trim_intervals(super().determinant(degree))


def family_bernstein(polynomial, domain, *, depth=0):
    """Hull over all coefficient members and ALL closed mark subdivisions."""
    if not polynomial or not isinstance(domain, Interval):
        raise ValueError('nonempty polynomial and closed Interval required')
    if type(depth) is not int or not 0 <= depth <= 12:
        raise ValueError('subdivision depth must be 0..12')
    p = trim_intervals(polynomial)
    if len(p) > 9:
        raise ValueError('degree at most eight required')
    if all(point(x) for x in p):
        return bernstein_enclosure(tuple(x.lo for x in p), domain, depth=depth)
    if depth:
        mid = (domain.lo+domain.hi)/2
        left = family_bernstein(p, Interval(domain.lo,mid), depth=depth-1)
        right = family_bernstein(p, Interval(mid,domain.hi), depth=depth-1)
        return Interval(min(left.lo,right.lo),max(left.hi,right.hi))
    n = len(p)-1; lo = domain.lo; width = domain.hi-domain.lo
    power = [sum((p[k]*comb(k,j)*lo**(k-j)*width**j for k in range(j,n+1)), Interval.exact(0)) for j in range(n+1)]
    b = [sum((power[j]*F(comb(k,j),comb(n,j)) for j in range(k+1)), Interval.exact(0)) for k in range(n+1)]
    return Interval(min(x.lo for x in b),max(x.hi for x in b))


def family_moment_cap(polynomial, domain, *, depth=2):
    p = trim_intervals(polynomial)
    if not p:
        raise ValueError('nonempty polynomial required')
    h = max(abs(domain.lo),abs(domain.hi))
    coefficient = p[0].hi + sum(max(abs(x.lo),abs(x.hi))*h**j for j,x in enumerate(p[1:],1))
    cap = min(coefficient, family_bernstein(p,domain,depth=depth).hi)
    if cap < 0:
        raise ValueError('negative bound cannot enclose a nonempty even-moment family')
    return cap
