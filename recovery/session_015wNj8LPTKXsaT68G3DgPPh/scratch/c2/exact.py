"""Exact derivation of c and c2 (Math-#216 (0.1)-(0.2); #218 (0.1)) for the Gaussian kernel, d = 1, 2, 3.
Standard library only: Fractions, truncated power series in r (pseries.py), multivariate polynomials, and closed-form
numbers  sum q * pi^(h/2) * sqrt(m) * theta^e  (theta the one cone angle atom of d = 3)."""
from fractions import Fraction as Fr
import itertools
import math

import pseries as P

NR = 6                       # polynomial truncation in r (A_r needs r^4 of the integrand)
VARS = ('r', 'b', 'k', 'x1', 'x2')
NV = len(VARS)


# ------------------------------------------------------------------------------------------- multivariate polynomials
class Poly:
    __slots__ = ('t',)

    def __init__(self, t=None):
        self.t = {e: c for e, c in (t or {}).items() if c}

    @staticmethod
    def var(name, power=1):
        e = [0] * NV
        e[VARS.index(name)] = power
        return Poly({tuple(e): Fr(1)})

    @staticmethod
    def const(c):
        return Poly({(0,) * NV: Fr(c)})

    def __add__(self, o):
        if not isinstance(o, Poly):
            o = Poly.const(o)
        t = dict(self.t)
        for e, c in o.t.items():
            t[e] = t.get(e, 0) + c
        return Poly(t)
    __radd__ = __add__

    def __neg__(self):
        return Poly({e: -c for e, c in self.t.items()})

    def __sub__(self, o):
        return self + (-o if isinstance(o, Poly) else Poly.const(-o))

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, o):
        if not isinstance(o, Poly):
            o = Fr(o)
            return Poly({e: c * o for e, c in self.t.items()})
        t = {}
        for e1, c1 in self.t.items():
            for e2, c2 in o.t.items():
                if e1[0] + e2[0] > NR:
                    continue
                e = tuple(a + b for a, b in zip(e1, e2))
                t[e] = t.get(e, 0) + c1 * c2
        return Poly(t)
    __rmul__ = __mul__

    def rcoef(self, n):
        """coefficient of r^n, as a Poly in the other variables"""
        return Poly({(0,) + e[1:]: c for e, c in self.t.items() if e[0] == n})

    def degree(self, name):
        i = VARS.index(name)
        return max((e[i] for e in self.t), default=0)

    def coef_of(self, name, p):
        i = VARS.index(name)
        return Poly({e[:i] + (0,) + e[i + 1:]: c for e, c in self.t.items() if e[i] == p})

    def subs_linear(self, name, poly):
        """substitute variable `name` by a polynomial"""
        i = VARS.index(name)
        out = Poly()
        powers = [Poly.const(1)]
        for e, c in self.t.items():
            n = e[i]
            while len(powers) <= n:
                powers.append(powers[-1] * poly)
            rest = Poly({e[:i] + (0,) + e[i + 1:]: c})
            out = out + rest * powers[n]
        return out

    def is_zero(self):
        return not self.t


def from_series(s, upto=NR):
    return Poly({(i,) + (0,) * (NV - 1): c for i, c in enumerate(s.c[:upto + 1]) if c})


# ------------------------------------------------------------------------------------------- closed-form numbers
def squarefree(n):
    """n = q^2 m with m squarefree; returns (q, m)"""
    q, m, p = 1, 1, 2
    while p * p <= n:
        while n % (p * p) == 0:
            n //= p * p
            q *= p
        if n % p == 0:
            n //= p
            m *= p
        p += 1
    return q, m * n


def sqrt_frac(x):
    """sqrt of a positive rational as (rational coefficient, squarefree m)"""
    x = Fr(x)
    assert x > 0
    q1, m = squarefree(x.numerator * x.denominator)
    return Fr(q1, x.denominator), m


class Num:
    """sum of q * pi^(h/2) * sqrt(m) * theta^e; keys (h, m, e)"""
    __slots__ = ('t',)

    def __init__(self, t=None):
        self.t = {k: v for k, v in (t or {}).items() if v}

    @staticmethod
    def rat(q):
        return Num({(0, 1, 0): Fr(q)})

    @staticmethod
    def sqrt(x):
        q, m = sqrt_frac(x)
        return Num({(0, m, 0): q})

    @staticmethod
    def pi_half(h):
        return Num({(h, 1, 0): Fr(1)})

    def __add__(self, o):
        t = dict(self.t)
        for k, v in o.t.items():
            t[k] = t.get(k, 0) + v
        return Num(t)

    def __neg__(self):
        return Num({k: -v for k, v in self.t.items()})

    def __sub__(self, o):
        return self + (-o)

    def __mul__(self, o):
        if not isinstance(o, Num):
            return Num({k: v * Fr(o) for k, v in self.t.items()})
        t = {}
        for (h1, m1, e1), v1 in self.t.items():
            for (h2, m2, e2), v2 in o.t.items():
                g = math.gcd(m1, m2)
                q, m = g, (m1 // g) * (m2 // g)
                key = (h1 + h2, m, e1 + e2)
                t[key] = t.get(key, 0) + v1 * v2 * q
        return Num(t)
    __rmul__ = __mul__

    def __repr__(self):
        parts = []
        for (h, m, e), v in sorted(self.t.items()):
            s = str(v)
            if m != 1:
                s += '*sqrt(%d)' % m
            if h:
                s += '*pi^(%d/2)' % h
            if e:
                s += '*theta^%d' % e
            parts.append(s)
        return ' + '.join(parts) or '0'

    def value(self, theta=0.0):
        return math.fsum(float(v) * math.pi ** (h / 2) * math.sqrt(m) * theta ** e for (h, m, e), v in self.t.items())


# ------------------------------------------------------------------------------------------- the pinned two-point kernel
def forms_for(d):
    X = lambda be, sg: P.deriv_at(be, sg, d)
    forms = {}
    for nm, sg in (('M', -1), ('S', 1)):
        for i in range(d):
            for j in range(i, d):
                be = [0] * d
                be[i] += 1
                be[j] += 1
                forms[(nm, i, j)] = X(tuple(be), sg)
    trans = []
    for i in range(1, d):
        for j in range(i, d):
            be = [0] * d
            be[i] += 1
            be[j] += 1
            nm = ('T', i, j)
            forms[nm] = {tuple(be): P.S.const(1)}
            trans.append(nm)
    return forms, trans


def matchings(lst):
    if not lst:
        yield []
        return
    f, rest = lst[0], lst[1:]
    for mm in matchings(rest):
        yield mm
    for i, y in enumerate(rest):
        for mm in matchings(rest[:i] + rest[i + 1:]):
            yield [(f, y)] + mm


def perm_sign(p):
    return -1 if sum(1 for i in range(len(p)) for j in range(i + 1, len(p)) if p[i] > p[j]) % 2 else 1


def series_pow(xpoly, alpha, order):
    """(1 + x)^alpha for a Poly x with zero r^0 part, truncated"""
    out, term = Poly.const(1), Poly.const(1)
    coef = Fr(1)
    for n in range(1, order + 1):
        coef = coef * (Fr(alpha) - (n - 1)) / n
        term = term * xpoly
        if term.is_zero():
            break
        out = out + term * coef
    return out


def series_exp(xpoly, order):
    out, term = Poly.const(1), Poly.const(1)
    for n in range(1, order + 1):
        term = term * xpoly * Fr(1, n)
        if term.is_zero():
            break
        out = out + term
    return out


def kernel_data(d):
    """Returns (G, q0, Q0, const) such that, with w the cone weight,
       A_r(b, k) = const * (cone integral of exp(-(q0 + Q0)/2) * G(r) * w) / r^2,
    G a Poly in (r, b, k, x1, x2) truncated at r^NR."""
    forms, trans = forms_for(d)
    pin = P.Pinned(d, forms)
    p = pin.p
    r, b, k = Poly.var('r'), Poly.var('b'), Poly.var('k')
    v = [b - k * r * r * r * Fr(1, 2), -(k * r * r), Poly(), k * 12] + [Poly()] * (p - 4)
    mean = {u: sum((from_series(pin.G[u][j]) * v[j] for j in range(p)), Poly()) for u in pin.names}
    hn = [u for u in pin.names if u not in trans]
    nt = len(trans)
    if nt:
        CTT = [[pin.C[(x, y)] for y in trans] for x in trans]
        CTTinv = P.mat_inv(CTT)
        detT = P.mat_det(CTT)
        B = {u: [sum((pin.C[(u, trans[j])] * CTTinv[j][i] for j in range(nt)), P.S()) for i in range(nt)] for u in hn}
        cp = {(u, w): from_series(pin.C[(u, w)] - sum((B[u][i] * pin.C[(trans[i], w)] for i in range(nt)), P.S()))
              for u in hn for w in hn}
        # transverse values: d = 2: a = x1; d = 3: diag(x1, x2), off-diagonal 0
        tval = {('T', 1, 1): Poly.var('x1')}
        if d == 3:
            tval[('T', 1, 2)] = Poly()
            tval[('T', 2, 2)] = Poly.var('x2')
        zT = [tval[nm] - mean[nm] for nm in trans]
        aff = {u: mean[u] + sum((from_series(B[u][i]) * zT[i] for i in range(nt)), Poly()) for u in hn}
    else:
        cp = {(u, w): from_series(pin.C[(u, w)]) for u in hn for w in hn}
        aff = {u: mean[u] for u in hn}
        detT = P.S.const(1)
    # E[det H_M det H_S | T, pins] by Isserlis
    F = Poly()
    for pp in itertools.permutations(range(d)):
        for qq in itertools.permutations(range(d)):
            fac = [('M', min(i, pp[i]), max(i, pp[i])) for i in range(d)] + \
                  [('S', min(i, qq[i]), max(i, qq[i])) for i in range(d)]
            sg = perm_sign(pp) * perm_sign(qq)
            for mt in matchings(list(range(2 * d))):
                used = set(i for pr in mt for i in pr)
                c = Poly.const(sg)
                for (i, j) in mt:
                    c = c * cp[(fac[i], fac[j])]
                    if c.is_zero():
                        break
                if c.is_zero():
                    continue
                for i in range(2 * d):
                    if i not in used:
                        c = c * aff[fac[i]]
                F = F + c
    # exponents
    q = sum((v[i] * from_series(pin.Sinv[i][j]) * v[j] for i in range(p) for j in range(p)), Poly())
    Q = sum((zT[i] * from_series(CTTinv[i][j]) * zT[j] for i in range(nt) for j in range(nt)), Poly()) if nt else Poly()
    q0, Q0 = q.rcoef(0), Q.rcoef(0)
    dq = (q - q0) + (Q - Q0)
    detS0, detT0 = pin.detS.c[0], detT.c[0]
    ratio = from_series(pin.detS) * (1 / detS0) * from_series(detT) * (1 / detT0) - Poly.const(1)
    G = series_pow(ratio, Fr(-1, 2), NR) * series_exp(dq * Fr(-1, 2), NR) * F
    # constant: -12 (2 pi)^(-(p + nt)/2) (detS0 detT0)^(-1/2), times pi for the d = 3 eigenvalue Jacobian
    nn = p + nt
    const = Num.rat(-12) * Num.sqrt(Fr(1, 2 ** nn)) * Num.pi_half(-nn) * Num.sqrt(Fr(1) / (detS0 * detT0))
    if d == 3:
        const = const * Num.pi_half(2)
    return G, q0, Q0, const, pin


def gamma_shift(n_half_num, base_num, base_den):
    """Gamma(a) / Gamma(a0) for a = n_half_num/2 and a0 = base_num/base_den with a - a0 an integer: exact rational"""
    a = Fr(n_half_num, 2)
    a0 = Fr(base_num, base_den)
    n = a - a0
    assert n.denominator == 1
    n = int(n)
    out = Fr(1)
    if n >= 0:
        for i in range(n):
            out *= a0 + i
    else:
        for i in range(-n):
            out /= a0 - 1 - i
    return out


# ------------------------------------------------------------------------------------------- the integrations
SPHERE = {1: Num.rat(2), 2: Num.pi_half(2) * 2, 3: Num.pi_half(2) * 4}


def half_moment(j, c):
    """int_0^oo t^j e^(-c t^2) dt = Gamma((j+1)/2) c^(-(j+1)/2) / 2, as a Num (c rational > 0)"""
    if j % 2 == 0:       # Gamma(j/2 + 1/2) = sqrt(pi) (1/2)(3/2)...; c^(-(j+1)/2) = c^(-j/2) sqrt(1/c)
        g = Fr(1)
        for i in range(j // 2):
            g *= Fr(2 * i + 1, 2)
        return Num.pi_half(1) * Num.sqrt(Fr(1) / c) * (g * Fr(1) / c ** (j // 2) / 2)
    m = (j + 1) // 2
    return Num.rat(Fr(math.factorial(m - 1)) / c ** m / 2)


def quadrant_moments(al, be, ga, nmax):
    """M(i, j) = int int_(s,t>0) s^i t^j exp(-(al s^2 + be s t + ga t^2)) as Nums in the atom
    theta = (pi/2 - atan(be/sqrt(D)))/sqrt(D) = M(0, 0), D = 4 al ga - be^2"""
    D = 4 * al * ga - be * be
    assert D > 0 and al > 0 and ga > 0
    M = {(0, 0): Num({(0, 1, 1): Fr(1)})}
    for n in range(nmax):
        for i in range(n + 1):
            j = n - i
            R1 = (M[(i - 1, j)] * i if i else Num()) + (half_moment(j, ga) if i == 0 else Num())
            R2 = (M[(i, j - 1)] * j if j else Num()) + (half_moment(i, al) if j == 0 else Num())
            a_ = (R1 * (2 * ga) - R2 * be) * (Fr(1) / D)
            b_ = (R2 * (2 * al) - R1 * be) * (Fr(1) / D)
            for key, val in (((i + 1, j), a_), ((i, j + 1), b_)):
                if key in M:
                    diff = M[key] - val
                    assert not diff.t, ('recurrence inconsistent', key, diff)
                else:
                    M[key] = val
    return M, D


def coefficient(d, which):
    """which = 0: c (k^(-2/3), r^0 of A_r); which = 2: c2 (finite part of k^(-4/3), r^2 of A_r).
    Returns (Num coefficient, K-atom label, theta data)."""
    G, q0, Q0, const, pin = kernel_data(d)
    assert G.rcoef(0).is_zero() and G.rcoef(1).is_zero(), 'A_r must start at r^0 after division by r^2'
    assert G.rcoef(3).is_zero(), 'r^1 term of A_r'
    H = G.rcoef(which + 2)
    E = (q0 + Q0) * Fr(1, 2)
    # k: E = ak k^2 + (k-free); no k-linear or mixed terms
    ak = E.coef_of('k', 2)
    assert set(ak.t) == {(0,) * NV}, 'k^2 coefficient must be a constant'
    ak = ak.t[(0,) * NV]
    assert E.coef_of('k', 1).is_zero() and E.degree('k') == 2
    Ek = E.coef_of('k', 0)
    power = Fr(-2, 3) if which == 0 else Fr(-4, 3)
    Hk = Poly()
    for j in range(H.degree('k') + 1):
        cj = H.coef_of('k', j)
        if cj.is_zero():
            continue
        assert j % 2 == 0, 'odd power of k'
        # Gamma((j + power + 1)/2) ak^(-(j + power + 1)/2) / 2 = [Gamma(a0) ak^(-a0)] * shift * ak^(-j/2) / 2
        if which == 0:     # a = j/2 + 1/6, a0 = 1/6
            sh = gamma_shift(j + 0, 0, 1) if False else None
            a = Fr(j, 2) + Fr(1, 6)
            a0 = Fr(1, 6)
        else:              # a = j/2 - 1/6, a0 = 5/6 (j = 0: Gamma(-1/6) = -6 Gamma(5/6))
            a = Fr(j, 2) - Fr(1, 6)
            a0 = Fr(5, 6)
        ratio = Fr(1)
        x = a0
        if a >= a0:
            while x < a:
                ratio *= x
                x += 1
        else:
            while x > a:
                x -= 1
                ratio /= x
        assert x == a
        # ak^(-a) = ak^(-a0) * ak^(a0 - a); a0 - a = (a0 - a) integer shift... a0 - a = -(j/2) + (a0 - a00)
        shift = a0 - a
        assert shift.denominator == 1
        Hk = Hk + cj * (ratio * Fr(ak) ** int(shift) / 2)
    katom = ('Gamma(1/6) * 12^(-1/6)' if which == 0 else 'Gamma(5/6) * 12^(-5/6)')
    assert ak == 12
    # b: Ek = ab b^2 + b L(x) + R(x)
    ab = Ek.coef_of('b', 2)
    assert set(ab.t) <= {(0,) * NV}
    ab = ab.t[(0,) * NV]
    Lx = Ek.coef_of('b', 1)
    Rx = Ek.coef_of('b', 0)
    assert Ek.degree('b') == 2
    shifted = Hk.subs_linear('b', Poly.var('b') - Lx * (Fr(1) / (2 * ab)))
    # int y^n e^(-ab y^2) dy over R = 2 half_moment(n, ab) for even n, 0 for odd n
    Hb = {}   # polynomial in x with Num coefficients, as dict exponent -> Num
    for e, c in shifted.t.items():
        n = e[1]
        if n % 2:
            continue
        key = (0, 0, 0) + e[3:]
        Hb[key] = Hb.get(key, Num()) + half_moment(n, ab) * (2 * c)
    Ex = Rx - Lx * Lx * (Fr(1) / (4 * ab))
    total = Num()
    if d == 1:
        assert Ex.is_zero()
        for key, val in Hb.items():
            total = total + val
    elif d == 2:
        gx = Ex.coef_of('x1', 2)
        assert set(gx.t) <= {(0,) * NV} and Ex.degree('x1') == 2 and Ex.coef_of('x1', 1).is_zero()
        gx = gx.t[(0,) * NV]
        for key, val in Hb.items():
            n = key[3]
            total = total + val * half_moment(n, gx) * (-1) ** n
    else:
        # x1 = -s, x2 = -s - t, weight t (the pi is in const); exponent al s^2 + be s t + ga t^2
        s_, t_ = Poly.var('x1'), Poly.var('x2')       # reuse slots: x1 -> s, x2 -> t
        sub = Ex.subs_linear('x1', -s_)
        # careful: substitute x2 first using a temporary, since x1 -> -s must not affect x2 -> -s - t
        sub = Ex.subs_linear('x2', -s_ - t_).subs_linear('x1', -s_) if False else None
        X1, X2 = -s_, -s_ - t_
        def subst(poly):
            out = Poly()
            for e, c in poly.t.items():
                term = Poly({e[:3] + (0, 0): c})
                for _ in range(e[3]):
                    term = term * X1
                for _ in range(e[4]):
                    term = term * X2
                out = out + term
            return out
        Est = subst(Ex)
        al = Est.coef_of('x1', 2).t.get((0,) * NV, Fr(0))
        ga = Est.coef_of('x2', 2).t.get((0,) * NV, Fr(0))
        be = Est.coef_of('x1', 1).coef_of('x2', 1).t.get((0,) * NV, Fr(0))
        chk = Poly.var('x1', 2) * al + Poly.var('x1') * Poly.var('x2') * be + Poly.var('x2', 2) * ga
        assert not (Est - chk).t, 'cone exponent not a quadratic form'
        nmax = max(key[3] + key[4] for key in Hb) + 4
        M, D = quadrant_moments(al, be, ga, nmax)
        for key, val in Hb.items():
            poly = subst(Poly({(0, 0, 0) + key[3:]: Fr(1)})) * Poly.var('x2')     # times the weight t
            for e, c in poly.t.items():
                total = total + val * M[(e[3], e[4])] * c
        return SPHERE[d] * Fr(1, 3) * const * total, katom, (al, be, ga, D)
    return SPHERE[d] * Fr(1, 3) * const * total, katom, None
