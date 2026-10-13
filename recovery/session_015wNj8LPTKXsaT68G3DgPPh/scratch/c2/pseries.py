"""Exact truncated power series in r over Q, and the pinned two-point structure of [R] for the Gaussian kernel.

Rows/forms are dicts jet -> Series.  Conditioning is done in Q[[r]] (the r^0 pin covariance is invertible)."""
from fractions import Fraction as Fr
import itertools

R = 10  # truncation order (results valid to r^(R-3))


class S:
    __slots__ = ('c',)

    def __init__(self, c=None):
        c = list(c or [])
        self.c = (c + [Fr(0)] * (R + 1))[:R + 1]

    @staticmethod
    def const(x):
        return S([Fr(x)])

    def __add__(self, o):
        if not isinstance(o, S):
            o = S.const(o)
        return S([a + b for a, b in zip(self.c, o.c)])
    __radd__ = __add__

    def __neg__(self):
        return S([-a for a in self.c])

    def __sub__(self, o):
        return self + (-o if isinstance(o, S) else S.const(-o))

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, o):
        if not isinstance(o, S):
            return S([a * o for a in self.c])
        out = [Fr(0)] * (R + 1)
        for i, a in enumerate(self.c):
            if a:
                for j in range(R + 1 - i):
                    if o.c[j]:
                        out[i + j] += a * o.c[j]
        return S(out)
    __rmul__ = __mul__

    def inv(self):
        a0 = self.c[0]
        assert a0 != 0
        out = [Fr(0)] * (R + 1)
        out[0] = 1 / a0
        for n in range(1, R + 1):
            out[n] = -sum(self.c[k] * out[n - k] for k in range(1, n + 1)) / a0
        return S(out)

    def __truediv__(self, o):
        if not isinstance(o, S):
            return S([a / o for a in self.c])
        return self * o.inv()

    def is_zero(self):
        return not any(self.c)

    def __repr__(self):
        return 'S(%s)' % [str(x) for x in self.c]


def dfact(n):
    out = 1
    while n > 1:
        out *= n
        n -= 2
    return out


def rho_der(g):
    if any(x % 2 for x in g):
        return 0
    out = 1
    for x in g:
        out *= (-1) ** (x // 2) * dfact(x - 1)
    return out


def cov(u, w):
    """Cov of two linear forms (dict jet -> S) for the Gaussian kernel: Cov(d^a f, d^b f) = (-1)^|b| d^(a+b) rho(0)."""
    tot = S()
    for a, sa in u.items():
        for b, sb in w.items():
            v = rho_der(tuple(x + y for x, y in zip(a, b)))
            if v:
                tot = tot + sa * sb * ((-1) ** sum(b) * v)
    return tot


def deriv_at(beta, sign, d, J=R + 3):
    """d^beta f at x = sign * r/2 along e_1, as jets: sum_j (sign/2)^j r^j / j! d^(beta + j e1) f(0)."""
    out = {}
    fact = 1
    for j in range(J + 1):
        if j:
            fact *= j
        c = [Fr(0)] * (R + 1)
        if j <= R:
            c[j] = Fr(sign, 2) ** j / fact
            g = (beta[0] + j,) + tuple(beta[1:])
            out[g] = S(c)
    return out


def lin(*terms):
    """sum of coef * form, coef a Fraction or S; forms dict jet -> S"""
    out = {}
    for coef, form in terms:
        for g, s in form.items():
            out[g] = out.get(g, S()) + s * coef
    return {g: s for g, s in out.items() if not s.is_zero()}


def shift(form, p):
    """divide by r^p (exact when the low coefficients vanish)"""
    out = {}
    for g, s in form.items():
        assert all(x == 0 for x in s.c[:p]), (g, s)
        out[g] = S(s.c[p:])
    return out


def pin_rows(d):
    e0 = (0,) * d
    ex = (1,) + (0,) * (d - 1)
    fa, fc = deriv_at(e0, -1, d, R + 4), deriv_at(e0, 1, d, R + 4)
    fxa, fxc = deriv_at(ex, -1, d, R + 4), deriv_at(ex, 1, d, R + 4)
    rows = [lin((Fr(1, 2), fa), (Fr(1, 2), fc)),
            shift(lin((1, fc), (-1, fa)), 1),
            shift(lin((1, fxc), (-1, fxa)), 1)]
    num = lin((1, fxa), (1, fxc), (-2, shift(lin((1, fc), (-1, fa)), 1)))
    rows.append({g: s * 6 for g, s in shift(num, 2).items()})
    for t in range(1, d):
        ey = tuple(1 if i == t else 0 for i in range(d))
        ya, yc = deriv_at(ey, -1, d), deriv_at(ey, 1, d)
        rows += [lin((Fr(1, 2), ya), (Fr(1, 2), yc)), shift(lin((1, yc), (-1, ya)), 1)]
    return rows


def mat_inv(M):
    """inverse of a square matrix of series (r^0 part invertible), Gauss-Jordan over Q[[r]]"""
    n = len(M)
    A = [row[:] + [S.const(int(i == j)) for j in range(n)] for i, row in enumerate(M)]
    for c in range(n):
        p = next(i for i in range(c, n) if A[i][c].c[0] != 0)
        A[c], A[p] = A[p], A[c]
        iv = A[c][c].inv()
        A[c] = [x * iv for x in A[c]]
        for i in range(n):
            if i != c and not A[i][c].is_zero():
                f = A[i][c]
                A[i] = [x - f * y for x, y in zip(A[i], A[c])]
    return [row[n:] for row in A]


def mat_det(M):
    n = len(M)
    A = [row[:] for row in M]
    det = S.const(1)
    for c in range(n):
        p = next(i for i in range(c, n) if A[i][c].c[0] != 0)
        if p != c:
            A[c], A[p] = A[p], A[c]
            det = -det
        det = det * A[c][c]
        iv = A[c][c].inv()
        for i in range(c + 1, n):
            if not A[i][c].is_zero():
                f = A[i][c] * iv
                A[i] = [x - f * y for x, y in zip(A[i], A[c])]
    return det


class Pinned:
    """conditional law of linear forms given the pins: mean = G v, covariance C (series)"""

    def __init__(self, d, forms):
        self.d = d
        self.rows = pin_rows(d)
        p = len(self.rows)
        self.p = p
        Sm = [[cov(a, b) for b in self.rows] for a in self.rows]
        self.Spin = Sm
        self.Sinv = mat_inv(Sm)
        self.detS = mat_det(Sm)
        names = list(forms)
        self.names = names
        CF = {u: [cov(forms[u], row) for row in self.rows] for u in names}
        self.G = {u: [sum((CF[u][q] * self.Sinv[q][j] for q in range(p)), S()) for j in range(p)] for u in names}
        self.C = {}
        for i, u in enumerate(names):
            for w in names[i:]:
                val = cov(forms[u], forms[w]) - sum((self.G[u][q] * CF[w][q] for q in range(p)), S())
                self.C[(u, w)] = self.C[(w, u)] = val
