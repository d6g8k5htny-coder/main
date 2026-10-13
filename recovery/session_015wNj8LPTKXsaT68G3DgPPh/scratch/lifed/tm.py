"""Taylor models in the radius: f(r) = sum_{j <= N} c_j r^j + theta e (|theta| <= 1) for every r in [0, R].
Coefficients are exact dyadic rationals (outward rounding moves into e)."""
import math
from fractions import Fraction as Fr

PB = 220                     # coefficient rounding: 2^-PB relative


def rnd(x):
    """Nearest dyadic with PB significant bits; returns (value, |error| bound)."""
    x = Fr(x)
    if x == 0:
        return x, Fr(0)
    e = abs(x.numerator).bit_length() - x.denominator.bit_length()
    sh = PB - e
    if sh >= 0:
        v = Fr(round(x * (1 << sh)), 1 << sh)
    else:
        v = Fr(round(x / (1 << -sh)) << -sh)
    return v, abs(x - v)


class TM:
    __slots__ = ('c', 'e', 'R', 'N')

    def __init__(self, c, e, R, N):
        c = list(c) + [Fr(0)] * (N + 1 - len(c))
        hi = c[N + 1:]
        c = c[:N + 1]
        e = Fr(e)
        Rp = Fr(1)
        out = []
        for j, x in enumerate(c):
            v, err = rnd(x)
            out.append(v)
            e += err * Rp
            Rp *= R
        Rp = R ** (N + 1)
        for x in hi:
            e += abs(x) * Rp
            Rp *= R
        self.c, self.e, self.R, self.N = out, e, R, N

    @classmethod
    def const(cls, x, R, N, e=0):
        return cls([Fr(x)], e, R, N)

    def bound(self):
        s, Rp = Fr(0), Fr(1)
        for x in self.c:
            s += abs(x) * Rp
            Rp *= self.R
        return s + self.e

    def __add__(self, o):
        if not isinstance(o, TM):
            o = TM.const(o, self.R, self.N)
        return TM([a + b for a, b in zip(self.c, o.c)], self.e + o.e, self.R, self.N)
    __radd__ = __add__

    def __neg__(self):
        return TM([-a for a in self.c], self.e, self.R, self.N)

    def __sub__(self, o):
        return self + (-(o if isinstance(o, TM) else TM.const(o, self.R, self.N)))

    def __rsub__(self, o):
        return TM.const(o, self.R, self.N) - self

    def __mul__(self, o):
        if not isinstance(o, TM):
            o = Fr(o)
            return TM([a * o for a in self.c], self.e * abs(o), self.R, self.N)
        N = self.N
        prod = [Fr(0)] * (2 * N + 1)
        for i, a in enumerate(self.c):
            if a == 0:
                continue
            for j, b in enumerate(o.c):
                if b:
                    prod[i + j] += a * b
        Ba = self.bound() - self.e
        Bb = o.bound() - o.e
        e = self.e * Bb + o.e * Ba + self.e * o.e
        return TM(prod, e, self.R, N)
    __rmul__ = __mul__

    def inv(self):
        c0 = self.c[0]
        if c0 == 0:
            raise ZeroDivisionError('TM inverse at zero constant term')
        h = (self - c0) * (1 / c0)                    # h(0) = 0 in the polynomial part
        H = h.bound()
        if H >= Fr(1, 2):
            raise ValueError('TM inverse: |h| too large')
        acc = TM.const(1, self.R, self.N)
        p = TM.const(1, self.R, self.N)
        for n in range(1, self.N + 1):
            p = p * (-h)
            acc = acc + p
        tail = H ** (self.N + 1) / (1 - H)
        return TM(acc.c, acc.e + tail, self.R, self.N) * (1 / c0)

    def __truediv__(self, o):
        if not isinstance(o, TM):
            return self * (1 / Fr(o))
        return self * o.inv()

    def enclose(self, r0, r1):
        """(lo, hi) rational bounds of f over [r0, r1] subset [0, R] (r0 >= 0)."""
        lo = hi = Fr(0)
        for j, x in enumerate(self.c):
            a, b = Fr(r0) ** j, Fr(r1) ** j
            if x >= 0:
                lo += x * a
                hi += x * b
            else:
                lo += x * b
                hi += x * a
        return lo - self.e, hi + self.e

    def scale_bound(self, j0, R1=None):
        """Bound B with |sum_{j >= j0} c_j r^j| <= B r^j0 for 0 <= r <= R1 (default R)."""
        R1 = self.R if R1 is None else Fr(R1)
        s, Rp = Fr(0), Fr(1)
        for x in self.c[j0:]:
            s += abs(x) * Rp
            Rp *= R1
        return s


def cov_tm(M, F, G, key, R, N, M1):
    """Taylor model over [0, R] of the covariance of F, G under the unnormalized torus kernel (the engine's exact series,
    its truncation tail and its torus remainder, as in the engine's cov_band)."""
    ser, w, absterms = M.pair_series(F, G, key)
    c = [Fr(0)] * (max(ser.keys(), default=0) + 1)
    for p, v in ser.items():
        c[p] += v
    A = max((t[1] for t in absterms), default=0)
    tail = Fr(0)
    for (acc, aa, gy) in absterms:
        tail += acc * abs(M.hcoef(gy))
    tail *= 2 * Fr(M.NX + 1) ** A * Fr(R) ** (M.NX + 1 - w)
    tor = Fr(0)
    for (acc, aa, gy) in absterms:
        tor += acc * math.factorial(gy) * Fr(max(w, 1)) ** aa
    eup = getattr(M, 'E_UP', Fr(27183, 10000))
    tor = tor * M1 / (1 - eup * Fr(R))
    return TM(c, tail + tor, Fr(R), N)


def solve_tm(A, B):
    """Gaussian elimination without pivoting (A(0) positive definite): X[c] solves A X[c] = B[c]; also det A."""
    n = len(A)
    Mx = [[A[i][j] for j in range(n)] + [B[c][i] for c in range(len(B))] for i in range(n)]
    det = None
    for col in range(n):
        piv = Mx[col][col]
        det = piv if det is None else det * piv
        ip = piv.inv()
        for i in range(col + 1, n):
            f = Mx[i][col] * ip
            for j in range(col, len(Mx[i])):
                Mx[i][j] = Mx[i][j] - f * Mx[col][j]
    X = []
    for c in range(len(B)):
        x = [None] * n
        for i in reversed(range(n)):
            s = Mx[i][n + c]
            for j in range(i + 1, n):
                s = s - Mx[i][j] * x[j]
            x[i] = s * Mx[i][i].inv()
        X.append(x)
    return X, det


class TMLaw:
    pass


def tm_law(M, names, R, N=8, L=10):
    """Taylor-model law over r in [0, R] (every frame, torus side >= L, and the reference kernel) of the targets `names`:
    alpha, beta (mean = b alpha + k beta), the unnormalized covariance block, the pin energies (unnormalized) and det."""
    pins, targets = M.functionals()
    P = [pins[n] for n in M.PIN_NAMES]
    idx = [M.TARGET_NAMES.index(n) for n in names]
    T = [targets[n] for n in names]
    M1 = M.torus_M1(L)
    n = len(P)
    R = Fr(R)
    Spp = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            Spp[i][j] = Spp[j][i] = cov_tm(M, P[i], P[j], ('p', i, j), R, N, M1)
    Stp = [[cov_tm(M, T[a], P[j], ('tp', idx[a], j), R, N, M1) for j in range(n)] for a in range(len(T))]
    Stt = [[None] * len(T) for _ in range(len(T))]
    for a in range(len(T)):
        for b in range(a, len(T)):
            i, j = idx[a], idx[b]
            key = ('tt', i, j) if i <= j else ('tt', j, i)
            if i <= j:
                Stt[a][b] = Stt[b][a] = cov_tm(M, T[a], T[b], ('tt', i, j), R, N, M1)
            else:
                Stt[a][b] = Stt[b][a] = cov_tm(M, T[b], T[a], ('tt', j, i), R, N, M1)
    rr = TM([Fr(0), Fr(1)], 0, R, N)
    kvec = [TM([0, 0, 0, Fr(-1, 2)], 0, R, N), TM([0, 0, Fr(-1)], 0, R, N), TM.const(12, R, N)] + [TM.const(0, R, N)] * (n - 3)
    e1 = [TM.const(1, R, N)] + [TM.const(0, R, N)] * (n - 1)
    X, det = solve_tm(Spp, [s for s in Stp] + [e1, kvec])
    W = X[:len(T)]
    xs0, xs1 = X[len(T)], X[len(T) + 1]
    law = TMLaw()
    law.M1, law.th1, law.R, law.N = M1, M.theta_minus_one(L), R, N
    law.alpha = {nm: W[a][0] for a, nm in enumerate(names)}
    law.beta = {nm: sum((W[a][j] * kvec[j] for j in range(n)), TM.const(0, R, N)) for a, nm in enumerate(names)}
    law.Cu = {}
    for a, na in enumerate(names):
        for b, nb in enumerate(names):
            law.Cu[(na, nb)] = Stt[a][b] - sum((W[a][l] * Stp[b][l] for l in range(n)), TM.const(0, R, N))
    law.e_bb = xs0[0]
    law.e_bk = sum((kvec[j] * xs0[j] for j in range(n)), TM.const(0, R, N))
    law.e_kk = sum((kvec[j] * xs1[j] for j in range(n)), TM.const(0, R, N))
    law.det = det
    return law
