"""Generic Laurent-series engine + station DAG, parameterized by scalar class.
Scalar classes supported:
  mpf  (plain mpmath mpf: exact-point value evaluation)
  J2   (value+gradient+Hessian jet, exact-point)
  JS   (global sup-bound arithmetic over a seeded domain)
The DAG is an exact transcription of c027 station.py (C027 limit evaluator);
series coefficients carry the scalar class; r-order bookkeeping uses
absolute-order extraction coef(K) (no trimming).
"""
from mpmath import mp, mpf
from fractions import Fraction as Fr
from jets import J2, JS, _j2, _js, js_exp_neg_half, js_sqrt, js_Phi_diff, PHI0

NM = 17

def zero_of(x):
    return x * 0

def _ez(x):
    """exact-zero test (no threshold): skipping exact zeros is sound."""
    if isinstance(x, J2):
        return (x.v == 0 and x.gx == 0 and x.gy == 0
                and x.hxx == 0 and x.hxy == 0 and x.hyy == 0)
    if isinstance(x, JS):
        return x.m0 == 0
    return x == 0

class GLS:
    __slots__ = ('off', 'c')
    def __init__(self, off=0, c=None):
        self.off = off
        self.c = list(c) if c else []
    def coef(self, K):
        i = K - self.off
        if i < 0 or i >= len(self.c):
            return self.c[0] * 0 if self.c else None
        return self.c[i]
    def __add__(a, b):
        off = min(a.off, b.off)
        end = min(max(a.off + len(a.c), b.off + len(b.c)), NM + 1)
        if not a.c and not b.c: return GLS(off, [])
        if not a.c: return b._slice(off, end)
        if not b.c: return a._slice(off, end)
        z = a.c[0] * 0
        c = [z] * (end - off)
        for i, x in enumerate(a.c):
            if a.off + i <= NM: c[a.off - off + i] = c[a.off - off + i] + x
        for i, x in enumerate(b.c):
            if b.off + i <= NM: c[b.off - off + i] = c[b.off - off + i] + x
        return GLS(off, c)
    def _slice(self, off, end):
        z = self.c[0] * 0 if self.c else None
        c = [self.c[i - self.off] if self.off <= i < self.off + len(self.c) else z
             for i in range(off, end)]
        return GLS(off, c)
    def __neg__(a):
        return GLS(a.off, [-x for x in a.c])
    def __sub__(a, b):
        return a + (-b)
    def __mul__(a, b):
        if isinstance(b, GLS):
            if not a.c or not b.c:
                off = a.off + b.off
                return GLS(off, [])
            off = a.off + b.off
            n = NM - off + 1
            if n <= 0: return GLS(off, [])
            z = a.c[0] * 0
            c = [z] * n
            for i, x in enumerate(a.c):
                if i >= n: break
                if _ez(x): continue
                for j, y in enumerate(b.c):
                    if i + j >= n: break
                    if _ez(y): continue
                    c[i + j] = c[i + j] + x * y
            return GLS(off, c)
        else:
            return GLS(a.off, [x * b for x in a.c])
    __rmul__ = __mul__
    def shift(self, k):
        return GLS(self.off + k, self.c[:])

# transcendental dispatch per scalar class
def s_exp(x):   # exp(x) for scalar x (assembly only)
    return mp.exp(x)

HE = [[1], [0, 1], [-1, 0, 1], [0, -3, 0, 1], [3, 0, -6, 0, 1]]

def cser(a, b, du, one):
    n1 = a[0] + b[0]; n2 = a[1] + b[1]
    s = (-1) ** ((b[0] + b[1]) + (n1 + n2))
    u1, u2 = du
    h1 = [one * HE[n1][k] * u1 ** k for k in range(n1 + 1)]
    h2 = [one * HE[n2][k] * u2 ** k for k in range(n2 + 1)]
    cc = (u1 * u1 + u2 * u2) / 2
    hp = [one * 0] * (n1 + n2 + 1)
    for i, x in enumerate(h1):
        for j, y in enumerate(h2):
            hp[i + j] = hp[i + j] + x * y
    out = [one * 0] * (NM + 1)
    m = 0; term = one * 1
    while 2 * m <= NM:
        for i, x in enumerate(hp):
            if 2 * m + i <= NM:
                out[2 * m + i] = out[2 * m + i] + x * term
        m += 1
        term = term * (-cc) / m
    if s < 0:
        out = [-x for x in out]
    return GLS(0, out)

def mat(n, m=None):
    m = m or n
    return [[GLS(0, []) for _ in range(m)] for _ in range(n)]

def mmul(A, B):
    n = len(A); k = len(B); m = len(B[0])
    C = mat(n, m)
    for i in range(n):
        for j in range(m):
            s = GLS(0, [])
            for t in range(k):
                s = s + A[i][t] * B[t][j]
            C[i][j] = s
    return C

def mT(A):
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

# ---- matrix inverse per scalar class ----
def mat_inv(M, one, minkey=None, minima=None):
    if isinstance(one, JS):
        return _mat_inv_JS(M, minkey, minima)
    if isinstance(one, J2):
        return _mat_inv_J2(M)
    return _mat_inv_num(M)

def _mat_inv_num(M):
    n = len(M)
    A = [[M[i][j] for j in range(n)] + [mpf(1 if i == j else 0) for j in range(n)] for i in range(n)]
    for col in range(n):
        p = max(range(col, n), key=lambda r: abs(A[r][col]))
        if A[p][col] == 0:
            raise ZeroDivisionError('singular matrix (numeric inverse)')
        A[col], A[p] = A[p], A[col]
        pv = A[col][col]
        A[col] = [x / pv for x in A[col]]
        for r in range(n):
            if r != col and A[r][col] != 0:
                f = A[r][col]
                A[r] = [A[r][j] - f * A[col][j] for j in range(2 * n)]
    return [[A[i][n + j] for j in range(n)] for i in range(n)]

def _mat_inv_J2(M):
    n = len(M)
    Mv = [[M[i][j].v for j in range(n)] for i in range(n)]
    Mi = _mat_inv_num(Mv)
    def wrap(x):
        return J2(x)
    out = [[wrap(Mi[i][j]) for j in range(n)] for i in range(n)]
    for dname in ('gx', 'gy'):
        D = [[getattr(M[i][j], dname) for j in range(n)] for i in range(n)]
        E = [[-sum(Mi[i][t] * D[t][u] * Mi[u][j] for t in range(n) for u in range(n))
              for j in range(n)] for i in range(n)]
        for i in range(n):
            for j in range(n):
                setattr(out[i][j], dname, E[i][j])
    for d1, d2, hn in (('gx', 'gx', 'hxx'), ('gx', 'gy', 'hxy'), ('gy', 'gy', 'hyy')):
        D1 = [[getattr(M[i][j], d1) for j in range(n)] for i in range(n)]
        D2 = [[getattr(M[i][j], d2) for j in range(n)] for i in range(n)]
        H = [[getattr(M[i][j], hn) for j in range(n)] for i in range(n)]
        # d2(Mi) = Mi*(D1*Mi*D2 + D2*Mi*D1 - H)*Mi
        def mm(A, B):
            return [[sum(A[i][t] * B[t][j] for t in range(n)) for j in range(n)] for i in range(n)]
        P1 = mm(D1, mm(Mi, D2)); P2 = mm(D2, mm(Mi, D1))
        E = mm(Mi, [[P1[i][j] + P2[i][j] - H[i][j] for j in range(n)] for i in range(n)])
        E = mm(E, Mi)
        for i in range(n):
            for j in range(n):
                setattr(out[i][j], hn, E[i][j])
    return out

def _fro(M, k):
    return mp.sqrt(sum(getattr(M[i][j], 'm%d' % k) ** 2 for i in range(len(M)) for j in range(len(M))))

def _mat_inv_JS(M, minkey, minima):
    """JS inverse via certified smallest-singular-value lower bound
    minima[minkey] = smin <= sigma_min(M) on the domain (scalar, tight).
    Returns uniform-entry JS bounds (sound: entry <= Frobenius)."""
    n = len(M)
    smin = minima[minkey]
    if smin <= 0:
        raise DagError('minima[%s] nonpositive: basis not certified on domain' % minkey)
    q = 1 / smin
    dM1 = _fro(M, 1); dM2 = _fro(M, 2); dM3 = _fro(M, 3)
    rn = mp.sqrt(n)
    MiJS = JS(rn * q,
              rn * q ** 2 * dM1,
              rn * (2 * q ** 3 * dM1 ** 2 + q ** 2 * dM2),
              rn * (6 * q ** 4 * dM1 ** 3 + 6 * q ** 3 * dM1 * dM2 + q ** 2 * dM3))
    return [[MiJS for _ in range(n)] for _ in range(n)]

B = Fr(6, 5)
Mt = (0, 0); St = (1, 0)
FUN6 = [((0, 0), Mt), ((1, 0), Mt), ((0, 1), Mt), ((0, 0), St), ((1, 0), St), ((0, 1), St)]
MON6 = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (3, 0)]
G6E = [0, 1, 1, 0, 1, 1]; C6E = [0, 1, 1, 2, 2, 3]
MON9A = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2)]
MON9B = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (0, 3)]
G9E = [0, 1, 1, 0, 1, 1, 0, 1, 1]; C9E = [0, 1, 1, 2, 2, 2, 3, 3, 3]

class DagError(Exception):
    def __init__(self, why): self.why = why

def _Trows(W, ge, ce, vecs):
    n = len(W); m = len(vecs)
    rows = []
    for j in range(n):
        s = GLS(0, [])
        for p in range(m):
            if not _iszero(W[j][p]):
                s = s + vecs[p].shift(ge[p]) * W[j][p]
        rows.append(s.shift(-ce[j]))
    return rows

def _iszero(x):
    if isinstance(x, J2):
        return x.v == 0 and x.gx == 0 and x.gy == 0 and x.hxx == 0 and x.hxy == 0 and x.hyy == 0
    if isinstance(x, JS):
        return x.m0 == 0
    return x == 0

def _Krows(OUT, FUN, W, ge, ce, Yt, one):
    K = []
    for (ao, po) in OUT:
        py = Yt if po is None else (one * po[0], one * po[1])
        Sop = []
        for p in range(len(FUN)):
            p2 = Yt if FUN[p][1] is None else (one * FUN[p][1][0], one * FUN[p][1][1])
            Sop.append(cser(ao, FUN[p][0], (py[0] - p2[0], py[1] - p2[1]), one))
        K.append(_Trows(W, ge, ce, Sop))
    return K

def build_frame(FUN, MON, gexp, cexp, Yt, one, minima, basistag):
    n = len(FUN)
    Spp = mat(n)
    for i, (ai, pi) in enumerate(FUN):
        for j, (aj, pj) in enumerate(FUN):
            ui = Yt if pi is None else (one * pi[0], one * pi[1])
            uj = Yt if pj is None else (one * pj[0], one * pj[1])
            Spp[i][j] = cser(ai, aj, (ui[0] - uj[0], ui[1] - uj[1]), one)
    Vh = [[None] * n for _ in range(n)]
    for i, (ai, pi) in enumerate(FUN):
        x, y = (Yt if pi is None else (one * pi[0], one * pi[1]))
        for a_idx, (px, py) in enumerate(MON):
            if ai == (0, 0): v = x ** px * y ** py
            elif ai == (1, 0): v = px * x ** (px - 1) * y ** py if px > 0 else one * 0
            elif ai == (0, 1): v = py * x ** px * y ** (py - 1) if py > 0 else one * 0
            Vh[i][a_idx] = v
    W = mat_inv(Vh, one, minkey='Vh' + basistag, minima=minima)
    A = [[Spp[i][j].shift(gexp[i] + gexp[j]) for j in range(n)] for i in range(n)]
    Bm = const_congr(W, A)
    G = [[Bm[i][j].shift(-(cexp[i] + cexp[j])) for j in range(n)] for i in range(n)]
    G0 = [[G[i][j].coef(0) for j in range(n)] for i in range(n)]
    return Spp, Vh, W, G, G0

def const_congr(W, A):
    n = len(A)
    Bm = mat(n)
    for i in range(n):
        for j in range(i, n):
            s = GLS(0, [])
            for p in range(n):
                if _iszero(W[i][p]): continue
                for q in range(n):
                    if _iszero(W[j][q]): continue
                    s = s + A[p][q] * (W[i][p] * W[j][q])
            Bm[i][j] = s
            Bm[j][i] = s
    return Bm

def neumann_inv(G, G0, one, K=11, minkey=None, minima=None):
    if isinstance(one, JS):
        return _neumann_inv_JS(G, G0, one, K, minkey, minima)
    n = len(G)
    G0i = mat_inv(G0, one, minkey=minkey, minima=minima)
    def gk(k):
        return [[G[i][j].coef(k) for j in range(n)] for i in range(n)]
    def fm(Aa, Bb):
        return [[sum(Aa[i][t] * Bb[t][j] for t in range(n)) for j in range(n)] for i in range(n)]
    X = [G0i]
    for k in range(1, K + 1):
        S = [[Gk * 0 for Gk in row] for row in X[0]]
        for j in range(1, k + 1):
            P = fm(gk(j), X[k - j])
            S = [[S[i][t] + P[i][t] for t in range(n)] for i in range(n)]
        X.append([[-sum(G0i[i][t] * S[t][j] for t in range(n)) for j in range(n)] for i in range(n)])
    Xs = mat(n)
    for i in range(n):
        for j in range(n):
            Xs[i][j] = GLS(0, [X[k][i][j] for k in range(K + 1)])
    return Xs

def _neumann_inv_JS(G, G0, one, K, minkey, minima):
    """Formal series inverse with SCALAR Frobenius-norm recurrences (tight:
    linear growth in k; no factorial/adjugate blowup). Uniform-entry JS
    bounds (sound: every entry <= Frobenius norm of its matrix)."""
    n = len(G)
    smin = minima[minkey]
    if smin <= 0:
        raise DagError('minima[%s] nonpositive' % minkey)
    rn = mp.sqrt(n)
    dM1 = _fro(G0, 1); dM2 = _fro(G0, 2); dM3 = _fro(G0, 3)
    q0 = rn / smin
    q1 = rn * dM1 / smin ** 2
    q2 = rn * (2 * dM1 ** 2 / smin ** 3 + dM2 / smin ** 2)
    q3 = rn * (6 * dM1 ** 3 / smin ** 4 + 6 * dM1 * dM2 / smin ** 3 + dM3 / smin ** 2)
    def gk(k, s):
        return mp.sqrt(sum(getattr(G[i][j].coef(k), 'm%d' % s) ** 2
                           for i in range(n) for j in range(n)))
    g0 = [mpf(0)] + [gk(k, 0) for k in range(1, K + 1)]
    g1 = [mpf(0)] + [gk(k, 1) for k in range(1, K + 1)]
    g2 = [mpf(0)] + [gk(k, 2) for k in range(1, K + 1)]
    g3 = [mpf(0)] + [gk(k, 3) for k in range(1, K + 1)]
    X = [(q0, q1, q2, q3)]
    for k in range(1, K + 1):
        t0 = sum(g0[j] * X[k - j][0] for j in range(1, k + 1))
        t1 = sum(g1[j] * X[k - j][0] + g0[j] * X[k - j][1] for j in range(1, k + 1))
        t2 = sum(g2[j] * X[k - j][0] + 2 * g1[j] * X[k - j][1] + g0[j] * X[k - j][2]
                 for j in range(1, k + 1))
        t3 = sum(g3[j] * X[k - j][0] + 3 * g2[j] * X[k - j][1] + 3 * g1[j] * X[k - j][2]
                 + g0[j] * X[k - j][3] for j in range(1, k + 1))
        X.append((q0 * t0,
                  q1 * t0 + q0 * t1,
                  q2 * t0 + 2 * q1 * t1 + q0 * t2,
                  q3 * t0 + 3 * q2 * t1 + 3 * q1 * t2 + q0 * t3))
    Xs = mat(n)
    for i in range(n):
        for j in range(n):
            Xs[i][j] = GLS(0, [JS(X[k][0], X[k][1], X[k][2], X[k][3]) for k in range(K + 1)])
    return Xs

_CACHE = {}

def _to_const_js(x):
    """freeze an mpf value into a constant JS (y-independent object)"""
    return JS(abs(x), 0, 0, 0)

def _freeze_series_js(s):
    return GLS(s.off, [_to_const_js(x) for x in s.c])

def frame6(one, minima=None):
    key = 'f6_' + type(one).__name__
    if key not in _CACHE:
        global NM
        NM = 17
        if isinstance(one, JS):
            # y-independent objects: compute once in mpf mode, freeze as exact
            # constants (junk allowance is stated in the certificate header)
            W6m, X6m, u6m = frame6(mpf(1))
            W6 = [[_to_const_js(x) for x in row] for row in W6m]
            X6 = [[_freeze_series_js(s) for s in row] for row in X6m]
            u6 = [_freeze_series_js(s) for s in u6m]
            _CACHE[key] = (W6, X6, u6)
            return _CACHE[key]
        Spp6, Vh6, W6, G6, G06 = build_frame(FUN6, MON6, G6E, C6E, None, one, minima, '6')
        X6 = neumann_inv(G6, G06, one, K=13, minkey='G06', minima=minima)
        vals6 = [GLS(0, [one * B]), GLS(0, []), GLS(0, []),
                 GLS(0, [one * B]) - GLS(3, [one * Fr(1, 6)]), GLS(0, []), GLS(0, [])]
        u6 = _Trows(W6, G6E, C6E, vals6)
        _CACHE[key] = (W6, X6, u6)
    return _CACHE[key]

def sdiv(A, Bs, border, one, minima=None):
    b0 = Bs.coef(border)
    if isinstance(one, JS):
        b0inv = _js(1).div(b0, minima['detS6'])
    else:
        b0inv = 1 / b0
        if not isinstance(one, J2) and b0 == 0:
            raise DagError('sdiv zero leading')
    off = A.off - border
    a = A.c
    c = []
    for k in range(len(a)):
        x = a[k]
        for j in range(min(k, len(c))):
            bi = k - j
            x = x - c[j] * Bs.coef(border + bi)
        c.append(x * b0inv)
    return GLS(off, c)

def Phi_J2(z):
    v = mp.ncdf(z.v)
    p = mp.exp(-z.v * z.v / 2) * PHI0
    pp = -z.v * p
    return J2(v, p * z.gx, p * z.gy,
              pp * z.gx * z.gx + p * z.hxx,
              pp * z.gx * z.gy + p * z.hxy,
              pp * z.gy * z.gy + p * z.hyy)

def exp_J2(x):
    v = mp.exp(x.v)
    return J2(v, v * x.gx, v * x.gy,
              v * (x.gx * x.gx + x.hxx),
              v * (x.gx * x.gy + x.hxy),
              v * (x.gy * x.gy + x.hyy))

def sqrt_J2(x):
    v = mp.sqrt(x.v)
    q = 1 / (2 * v)
    return J2(v, x.gx * q, x.gy * q,
              x.hxx * q - x.gx * x.gx / (4 * v ** 3),
              x.hxy * q - x.gx * x.gy / (4 * v ** 3),
              x.hyy * q - x.gy * x.gy / (4 * v ** 3))

def lam_assemble_J2(res):
    """Assemble lambda as a J2 jet from station_dag ingredients (typed branch)."""
    cM, cS, cY, trc = res['cM'], res['cS'], res['cY'], res['trc']
    if not (cM.v > 0 and cS.v < 0 and cY.v < 0 and trc.v < 0):
        z = J2(0)
        return dict(lam=z, A=J2(0), Pw=J2(0), N=J2(0), typed=False)
    A = exp_J2(res['qv'] * (-1 / 2)) / (2 * mp.pi * sqrt_J2(res['c6v']))
    m = res['m']; st = sqrt_J2(res['st2'])
    Pw = Phi_J2(-m / st) - Phi_J2((-1 - m) / st)
    N = cM * (-cS) * (-cY)
    lam = A * Pw * N / DEN_J2
    return dict(lam=lam, A=A, Pw=Pw, N=N, typed=True)

def lam_value_mpf(res):
    cM, cS, cY, trc = res['cM'], res['cS'], res['cY'], res['trc']
    if not (cM > 0 and cS < 0 and cY < 0 and trc < 0):
        return mpf(0)
    A = mp.exp(-res['qv'] / 2) / (2 * mp.pi * mp.sqrt(res['c6v']))
    st = mp.sqrt(res['st2'])
    Pw = mp.ncdf(-res['m'] / st) - mp.ncdf((-1 - res['m']) / st)
    return A * Pw * (cM * (-cS) * (-cY)) / DEN

def _den():
    b = mpf(6) / 5
    return (b * b + 2) * mp.ncdf(b / mp.sqrt(2)) + mp.sqrt(2) * b * mp.exp(-b * b / 4) * PHI0

DEN = _den()
DEN_J2 = J2(DEN)

def station_dag(Y1, Y2, one, minima=None, want=('lam',)):
    """Generic station DAG. Y1,Y2 scalars (mpf / J2 seeded / JS seeded).
    Returns dict of scalar quantities (in the scalar class)."""
    global NM
    W6, X6, u6 = frame6(one, minima)
    NM = 17
    Yt = (Y1, Y2)
    OUT = [((0, 0), None), ((1, 0), None), ((0, 1), None)]
    KYr = _Krows(OUT, FUN6, W6, G6E, C6E, Yt, one)
    KY = mmul(KYr, X6)
    mo = [sum((KY[i][j] * u6[j] for j in range(6)), GLS(0, [])) for i in range(3)]
    skey = 'Soo_' + type(one).__name__
    if skey not in _CACHE:
        Soo = mat(3)
        for i in range(3):
            for j in range(3):
                Soo[i][j] = cser(OUT[i][0], OUT[j][0], (one * 0, one * 0), one)
        _CACHE[skey] = Soo
    Soo = _CACHE[skey]
    KKt = mmul(KY, mT(KYr))
    So = [[Soo[i][j] - KKt[i][j] for j in range(3)] for i in range(3)]
    detS = So[1][1] * So[2][2] - So[1][2] * So[1][2]
    c6v = detS.coef(6)
    adj = [[So[2][2], -So[1][2]], [-So[1][2], So[1][1]]]
    mg = [mo[1], mo[2]]; Sfg = [So[0][1], So[0][2]]
    qn = (mg[0] * (adj[0][0] * mg[0] + adj[0][1] * mg[1])
          + mg[1] * (adj[1][0] * mg[0] + adj[1][1] * mg[1]))
    cross = Sfg[0] * (adj[0][0] * mg[0] + adj[0][1] * mg[1]) \
          + Sfg[1] * (adj[1][0] * mg[0] + adj[1][1] * mg[1])
    muN = mo[0] * detS - cross - GLS(0, [one * B]) * detS
    s2N = So[0][0] * detS - (Sfg[0] * (adj[0][0] * Sfg[0] + adj[0][1] * Sfg[1])
                             + Sfg[1] * (adj[1][0] * Sfg[0] + adj[1][1] * Sfg[1]))
    if isinstance(one, JS):
        qv = qn.coef(6).div(c6v, minima['detS6'])
        st2 = s2N.coef(12).div(c6v, minima['detS6']) * 36
        m = muN.coef(9).div(c6v, minima['detS6']) * 6
    else:
        qv = qn.coef(6) / c6v
        st2 = 36 * s2N.coef(12) / c6v
        m = 6 * muN.coef(9) / c6v
        if not isinstance(one, J2):
            if c6v == 0: raise DagError('c6 zero')
            if st2 <= 0: raise DagError('st2 nonpositive')
    # vser: clip decision
    if isinstance(one, JS):
        # domain-wide: use hull of interior/clip via derivative-neutral choice:
        # vser enters only through vals9 (mean shift); take interior series if
        # domain m within (-1,0) per minima, else clip constant; else hull both
        case = minima.get('clipcase', 'interior')
        if case == 'interior':
            vser = sdiv(muN, detS, 6, one, minima)
        elif case == 'lo':
            vser = GLS(3, [one * Fr(-1, 6)])
        elif case == 'hi':
            vser = GLS(3, [one * 0])
        else:
            raise DagError('clipcase must be region-certified pure (interior/lo/hi)')
    else:
        mv = m.v if isinstance(one, J2) else m
        if -1 < mv < 0:
            vser = sdiv(muN, detS, 6, one, minima)
        elif mv <= -1:
            vser = GLS(3, [one * Fr(-1, 6)])
        else:
            vser = GLS(3, [one * 0])
    NM = 12
    FUN9 = FUN6 + [((0, 0), None), ((1, 0), None), ((0, 1), None)]
    built = None
    for MB, bname in [(MON9A, '9A'), (MON9B, '9B')]:
        try:
            Spp9, Vh9, W9, G9, G09 = build_frame(FUN9, MB, G9E, C9E, Yt, one, minima, bname)
            X9 = neumann_inv(G9, G09, one, K=6, minkey='G0' + bname, minima=minima)
            built = (W9, X9, bname)
            break
        except (DagError, ZeroDivisionError):
            continue
    if built is None:
        raise DagError('9-frame degenerate both bases')
    W9, X9, bname = built
    H9r = [((2, 0), Mt), ((1, 1), Mt), ((0, 2), Mt),
           ((2, 0), St), ((1, 1), St), ((0, 2), St),
           ((2, 0), None), ((1, 1), None), ((0, 2), None)]
    K9r = mmul(_Krows(H9r, FUN9, W9, G9E, C9E, Yt, one), X9)
    vals9 = [GLS(0, [one * B]), GLS(0, []), GLS(0, []),
             GLS(0, [one * B]) - GLS(3, [one * Fr(1, 6)]), GLS(0, []), GLS(0, []),
             GLS(0, [one * B]) + vser, GLS(0, []), GLS(0, [])]
    u9 = _Trows(W9, G9E, C9E, vals9)
    mh = [sum((K9r[i][j] * u9[j] for j in range(9)), GLS(0, [])) for i in range(9)]
    dM = mh[0] * mh[2] - mh[1] * mh[1]
    dS = mh[3] * mh[5] - mh[4] * mh[4]
    dY = mh[6] * mh[8] - mh[7] * mh[7]
    tr = mh[0] + mh[2]
    cM = dM.coef(2); cS = dS.coef(2); cY = dY.coef(2); trc = tr.coef(1)
    return dict(c6v=c6v, qv=qv, st2=st2, m=m, cM=cM, cS=cS, cY=cY, trc=trc,
                basis=bname, detS=detS, qn=qn, muN=muN, s2N=s2N,
                Vh9=Vh9, G09=G09, X9=X9)
