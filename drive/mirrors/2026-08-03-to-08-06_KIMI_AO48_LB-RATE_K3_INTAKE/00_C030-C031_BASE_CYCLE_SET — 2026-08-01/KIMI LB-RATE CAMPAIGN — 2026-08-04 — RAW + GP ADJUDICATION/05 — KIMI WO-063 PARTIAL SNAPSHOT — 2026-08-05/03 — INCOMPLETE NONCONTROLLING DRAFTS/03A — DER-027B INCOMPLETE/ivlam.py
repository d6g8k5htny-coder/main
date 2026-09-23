"""Interval Laurent-series engine + interval limit-intensity evaluator (prototype).
Mirrors c026_series.py / c027 station.py semantics with mpmath.iv coefficients.
Structural orders are handled by ABSOLUTE-ORDER extraction coef(K), so interval
cancellation widths never corrupt the r-order bookkeeping.
"""
from mpmath import iv, mp
from fractions import Fraction as Fr

iv.dps = 70
NM = 17          # global series order cap (mirrors cs.NM; station uses 17 then 12)
INF = iv.mpf('inf')

class Split(Exception):
    """Box must be subdivided (fatal interval straddle)."""
    def __init__(self, why): self.why = why

IVT = type(iv.mpf(0))

def V(x):
    """exact iv from int/Fraction/str"""
    if isinstance(x, IVT):
        return x
    if isinstance(x, Fr):
        return iv.mpf(x.numerator) / iv.mpf(x.denominator)
    return iv.mpf(x)

def is_exact_zero(t):
    return t.a == 0 and t.b == 0

class ILS:
    """Laurent series in r with interval coefficients; off = bookkeeping offset
    (<= true order). Arrays hold coefficients for orders off..off+len(c)-1,
    capped at absolute order NM (set globally before frame builds)."""
    __slots__ = ('off', 'c')
    def __init__(self, off=0, c=None):
        self.off = off
        self.c = list(c) if c else []
    def copy(self):
        return ILS(self.off, self.c[:])
    def coef(self, K):
        """coefficient of r^K (absolute order); exact zero if beyond array"""
        i = K - self.off
        if i < 0 or i >= len(self.c):
            return iv.mpf(0)
        return self.c[i]
    def __add__(a, b):
        off = min(a.off, b.off)
        end = max(a.off + len(a.c), b.off + len(b.c))
        end = min(end, NM + 1)
        c = [iv.mpf(0)] * (end - off)
        for i, x in enumerate(a.c):
            if a.off + i <= NM: c[a.off - off + i] = c[a.off - off + i] + x
        for i, x in enumerate(b.c):
            if b.off + i <= NM: c[b.off - off + i] = c[b.off - off + i] + x
        return ILS(off, c)
    def __neg__(a):
        return ILS(a.off, [-x for x in a.c])
    def __sub__(a, b):
        return a + (-b)
    def __mul__(a, b):
        if isinstance(b, (int, Fr)):
            b = V(b)
        if isinstance(b, IVT):
            return ILS(a.off, [x * b for x in a.c])
        if not a.c or not b.c:
            return ILS(a.off + b.off, [])
        off = a.off + b.off
        n = NM - off + 1
        if n <= 0:
            return ILS(off, [])
        c = [iv.mpf(0)] * n
        for i, x in enumerate(a.c):
            if i >= n: break
            for j, y in enumerate(b.c):
                if i + j >= n: break
                c[i + j] = c[i + j] + x * y
        return ILS(off, c)
    __rmul__ = __mul__
    def shift(self, k):
        return ILS(self.off + k, self.c[:])

def ISc(x):
    return ILS(0, [V(x)])

HE = [[1], [0, 1], [-1, 0, 1], [0, -3, 0, 1], [3, 0, -6, 0, 1]]

def cser(a, b, du):
    """Cov(d^a f(p), d^b f(p')) as series in r, du = p - p' (scaled), interval du.
    Gaussian (Bargmann-Fock) kernel: sum_m (-cc)^m/m! r^{2m} * Hermite products."""
    n1 = a[0] + b[0]; n2 = a[1] + b[1]
    s = (-1) ** ((b[0] + b[1]) + (n1 + n2))
    u1, u2 = V(du[0]), V(du[1])
    h1 = [V(HE[n1][k]) * u1 ** k for k in range(n1 + 1)]
    h2 = [V(HE[n2][k]) * u2 ** k for k in range(n2 + 1)]
    cc = (u1 * u1 + u2 * u2) / 2
    # h1*h2 polynomial
    hp = [iv.mpf(0)] * (n1 + n2 + 1)
    for i, x in enumerate(h1):
        for j, y in enumerate(h2):
            hp[i + j] = hp[i + j] + x * y
    # multiply by exp series ex[2m] = (-cc)^m/m!
    out = [iv.mpf(0)] * (NM + 1)
    m = 0; term = iv.mpf(1)
    while 2 * m <= NM:
        for i, x in enumerate(hp):
            if 2 * m + i <= NM:
                out[2 * m + i] = out[2 * m + i] + x * term
        m += 1
        term = term * (-cc) / m
    if s < 0:
        out = [-x for x in out]
    return ILS(0, out)

def mat(n, m=None):
    m = m or n
    return [[ILS(0, []) for _ in range(m)] for _ in range(n)]

def mmul(A, B):
    n = len(A); k = len(B); m = len(B[0])
    C = mat(n, m)
    for i in range(n):
        for j in range(m):
            s = ILS(0, [])
            for t in range(k):
                s = s + A[i][t] * B[t][j]
            C[i][j] = s
    return C

def mT(A):
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

def _pivot_ok(t):
    return not (t.a <= 0 <= t.b)

def frmat_inv(M):
    """interval Gauss-Jordan inverse; raises Split on pivot straddle"""
    n = len(M)
    A = [[M[i][j] for j in range(n)] + [iv.mpf(1 if i == j else 0) for j in range(n)]
         for i in range(n)]
    for col in range(n):
        p = None
        best = None
        for r in range(col, n):
            t = A[r][col]
            if _pivot_ok(t):
                mag = max(abs(t.a), abs(t.b))
                if best is None or mag > best:
                    best = mag; p = r
        if p is None:
            raise Split('frmat_inv pivot straddle')
        A[col], A[p] = A[p], A[col]
        pv = A[col][col]
        A[col] = [x / pv for x in A[col]]
        for r in range(n):
            if r != col and not is_exact_zero(A[r][col]):
                f = A[r][col]
                A[r] = [A[r][j] - f * A[col][j] for j in range(2 * n)]
    return [[A[i][n + j] for j in range(n)] for i in range(n)]

def const_congr(W, A):
    n = len(A)
    B = mat(n)
    for i in range(n):
        for j in range(n):
            s = ILS(0, [])
            for p in range(n):
                if is_exact_zero(W[i][p]): continue
                for q in range(n):
                    if is_exact_zero(W[j][q]): continue
                    s = s + A[p][q] * (W[i][p] * W[j][q])
            B[i][j] = s
    return B

def build_frame(FUN, MON, gexp, cexp):
    n = len(FUN)
    Spp = mat(n)
    for i, (ai, pi) in enumerate(FUN):
        for j, (aj, pj) in enumerate(FUN):
            Spp[i][j] = cser(ai, aj, (V(pi[0]) - V(pj[0]), V(pi[1]) - V(pj[1])))
    Vh = [[None] * n for _ in range(n)]
    for i, (ai, pi) in enumerate(FUN):
        for a_idx, (px, py) in enumerate(MON):
            x, y = pi
            if ai == (0, 0): v = x ** px * y ** py
            elif ai == (1, 0): v = px * x ** (px - 1) * y ** py if px > 0 else 0
            elif ai == (0, 1): v = py * x ** px * y ** (py - 1) if py > 0 else 0
            Vh[i][a_idx] = V(v)
    W = frmat_inv(Vh)
    A = [[Spp[i][j].shift(gexp[i] + gexp[j]) for j in range(n)] for i in range(n)]
    B = const_congr(W, A)
    G = [[B[i][j].shift(-(cexp[i] + cexp[j])) for j in range(n)] for i in range(n)]
    G0 = [[G[i][j].coef(0) for j in range(n)] for i in range(n)]
    return Spp, Vh, W, G, G0

def neumann_inv(G, G0, K=11):
    n = len(G)
    G0i = frmat_inv(G0)
    def gk(k):
        return [[G[i][j].coef(k) for j in range(n)] for i in range(n)]
    def fm(Aa, Bb):
        return [[sum(Aa[i][t] * Bb[t][j] for t in range(n)) for j in range(n)] for i in range(n)]
    zero = [[iv.mpf(0)] * n for _ in range(n)]
    X = [G0i]
    for k in range(1, K + 1):
        S = [row[:] for row in zero]
        for j in range(1, k + 1):
            P = fm(gk(j), X[k - j])
            S = [[S[i][t] + P[i][t] for t in range(n)] for i in range(n)]
        X.append([[-sum(G0i[i][t] * S[t][j] for t in range(n)) for j in range(n)] for i in range(n)])
    Xs = mat(n)
    for i in range(n):
        for j in range(n):
            Xs[i][j] = ILS(0, [X[k][i][j] for k in range(K + 1)])
    return Xs
