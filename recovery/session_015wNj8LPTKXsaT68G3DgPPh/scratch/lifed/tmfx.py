"""Fixed-point Taylor models in the radius on [0, R], R = 2^-s:
f(r) = sum_{j <= N} (C_j / 2^P) r^j + theta (E / 2^P), |theta| <= 1, for every r in [0, R].
All arithmetic is in Python integers; every rounding is accounted for in E (rounded up)."""
import math
from fractions import Fraction as Fr

P = 288


def cdiv(a, b):
    return -((-a) // b)


class TM:
    __slots__ = ('C', 'E', 's', 'N')

    def __init__(self, C, E, s, N):
        self.C, self.E, self.s, self.N = C, E, s, N

    # -------------------------------------------------------------- constructors
    @classmethod
    def from_fracs(cls, coeffs, e, s, N):
        """coeffs: list or dict {power: Fraction}; e: Fraction remainder."""
        if isinstance(coeffs, dict):
            items = coeffs.items()
        else:
            items = enumerate(coeffs)
        C = [0] * (N + 1)
        E = Fr(e)
        for p, x in items:
            x = Fr(x)
            if x == 0:
                continue
            if p <= N:
                v = round(x * (1 << P))
                C[p] += v
                E += abs(x - Fr(v, 1 << P)) / Fr(2) ** (s * p)
            else:
                E += abs(x) / Fr(2) ** (s * p)
        return cls(C, cdiv(E.numerator << P, E.denominator), s, N)

    @classmethod
    def const(cls, x, s, N):
        return cls.from_fracs({0: Fr(x)}, 0, s, N)

    # -------------------------------------------------------------- helpers
    def polybound(self):
        """Upper bound (scaled int) of sum |C_j| R^j."""
        tot = 0
        for j, c in enumerate(self.C):
            if c:
                tot += cdiv(abs(c), 1 << (self.s * j))
        return tot

    def bound(self):
        return self.polybound() + self.E

    def frac_coeffs(self):
        return [Fr(c, 1 << P) for c in self.C]

    def frac_e(self):
        return Fr(self.E, 1 << P)

    # -------------------------------------------------------------- arithmetic
    def __add__(self, o):
        if not isinstance(o, TM):
            o = TM.const(o, self.s, self.N)
        return TM([a + b for a, b in zip(self.C, o.C)], self.E + o.E, self.s, self.N)
    __radd__ = __add__

    def __neg__(self):
        return TM([-a for a in self.C], self.E, self.s, self.N)

    def __sub__(self, o):
        if not isinstance(o, TM):
            o = TM.const(o, self.s, self.N)
        return self + (-o)

    def __rsub__(self, o):
        return TM.const(o, self.s, self.N) - self

    def scale(self, x):
        """Multiply by a rational x."""
        x = Fr(x)
        C = []
        E = Fr(self.E, 1 << P) * abs(x)
        for j, c in enumerate(self.C):
            v = Fr(c, 1 << P) * x
            w = round(v * (1 << P))
            C.append(w)
            E += abs(v - Fr(w, 1 << P)) / Fr(2) ** (self.s * j)
        return TM(C, cdiv(E.numerator << P, E.denominator), self.s, self.N)

    def __mul__(self, o):
        if not isinstance(o, TM):
            return self.scale(o)
        N, s = self.N, self.s
        S = [0] * (2 * N + 1)
        for i, a in enumerate(self.C):
            if a == 0:
                continue
            for j, b in enumerate(o.C):
                if b:
                    S[i + j] += a * b
        C = []
        E = 0
        for k in range(N + 1):
            v = S[k] >> P                                   # floor; error < 1 ulp, times R^k <= 1
            C.append(v)
            if S[k] - (v << P):
                E += 1
        for k in range(N + 1, 2 * N + 1):
            if S[k]:
                E += cdiv(abs(S[k]), 1 << (P + s * k))
        Ba, Bb = self.polybound(), o.polybound()
        E += cdiv(self.E * Bb + o.E * Ba + self.E * o.E, 1 << P)
        return TM(C, E, s, N)
    __rmul__ = __mul__

    def inv(self):
        c0 = Fr(self.C[0], 1 << P)
        if c0 == 0:
            raise ZeroDivisionError('TM inverse at zero constant term')
        h = (self - TM([self.C[0]] + [0] * self.N, 0, self.s, self.N)).scale(1 / c0)
        H = Fr(h.bound(), 1 << P)
        if H >= Fr(1, 2):
            raise ValueError('TM inverse: |h| too large')
        acc = TM.const(1, self.s, self.N)
        p = TM.const(1, self.s, self.N)
        mh = -h
        for _ in range(self.N):
            p = p * mh
            acc = acc + p
        tail = H ** (self.N + 1) / (1 - H)
        acc = TM(acc.C, acc.E + cdiv(tail.numerator << P, tail.denominator), self.s, self.N)
        return acc.scale(1 / c0)

    # -------------------------------------------------------------- evaluation
    def enclose(self, r0, r1):
        """(lo, hi) rational bounds of f over [r0, r1] subset [0, R]."""
        lo = hi = Fr(0)
        for j, c in enumerate(self.C):
            if c == 0:
                continue
            x = Fr(c, 1 << P)
            a, b = Fr(r0) ** j, Fr(r1) ** j
            if x >= 0:
                lo += x * a
                hi += x * b
            else:
                lo += x * b
                hi += x * a
        e = Fr(self.E, 1 << P)
        return lo - e, hi + e

    def low_coeffs(self, j0):
        return [Fr(c, 1 << P) for c in self.C[:j0]]

    def scale_bound(self, j0, R1):
        """B with |sum_{j >= j0} c_j r^j| <= B r^j0 for 0 <= r <= R1 <= R."""
        R1 = Fr(R1)
        tot, Rp = Fr(0), Fr(1)
        for c in self.C[j0:]:
            tot += abs(Fr(c, 1 << P)) * Rp
            Rp *= R1
        return tot


def cov_tm(M, F, G, key, s, N, M1):
    """Taylor model over [0, 2^-s] of the covariance of F, G under the unnormalized torus kernel: the engine's exact series,
    its truncation tail and its torus remainder, exactly as in the engine's cov_band (evaluated at r1 = R)."""
    R = Fr(1, 2 ** s)
    ser, w, absterms = M.pair_series(F, G, key)
    A = max((t[1] for t in absterms), default=0)
    tail = Fr(0)
    tor = Fr(0)
    for t in absterms:
        acc, aa, gys = t[0], t[1], t[2:]
        hprod = 1
        fprod = 1
        for g in gys:
            hprod *= M.hcoef(g)
            fprod *= math.factorial(g)
        tail += acc * abs(hprod)
        tor += acc * fprod * Fr(max(w, 1)) ** aa
    tail *= 2 * Fr(M.NX + 1) ** A * R ** (M.NX + 1 - w)
    eup = getattr(M, 'E_UP', Fr(27183, 10000))
    tor = tor * M1 / (1 - eup * R)
    return TM.from_fracs(ser, tail + tor, s, N)


def solve_tm(A, B):
    """Gaussian elimination without pivoting (A(0) positive definite): X[c] solves A X[c] = B[c]; also det A."""
    n = len(A)
    Mx = [[A[i][j] for j in range(n)] + [B[c][i] for c in range(len(B))] for i in range(n)]
    det = None
    invs = []
    for col in range(n):
        piv = Mx[col][col]
        det = piv if det is None else det * piv
        ip = piv.inv()
        invs.append(ip)
        for i in range(col + 1, n):
            f = Mx[i][col] * ip
            for j in range(col, len(Mx[i])):
                Mx[i][j] = Mx[i][j] - f * Mx[col][j]
    X = []
    for c in range(len(B)):
        x = [None] * n
        for i in reversed(range(n)):
            sacc = Mx[i][n + c]
            for j in range(i + 1, n):
                sacc = sacc - Mx[i][j] * x[j]
            x[i] = sacc * invs[i]
        X.append(x)
    return X, det


class TMLaw:
    pass


def tm_law(M, names, s, N=8, L=10):
    """Taylor-model law over r in [0, 2^-s] (every frame, every torus side >= L, and the reference kernel) of the targets
    `names`: alpha, beta (mean = b alpha + k beta), the unnormalized covariance block, the unnormalized pin energies and det."""
    pins, targets = M.functionals()
    P_ = [pins[n] for n in M.PIN_NAMES]
    idx = [M.TARGET_NAMES.index(n) for n in names]
    T = [targets[n] for n in names]
    M1 = M.torus_M1(L)
    n = len(P_)
    Spp = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            Spp[i][j] = Spp[j][i] = cov_tm(M, P_[i], P_[j], ('p', i, j), s, N, M1)
    Stp = [[cov_tm(M, T[a], P_[j], ('tp', idx[a], j), s, N, M1) for j in range(n)] for a in range(len(T))]
    nt = len(T)
    Stt = [[None] * nt for _ in range(nt)]
    for a in range(nt):
        for b in range(a, nt):
            i, j = idx[a], idx[b]
            if i <= j:
                Stt[a][b] = Stt[b][a] = cov_tm(M, T[a], T[b], ('tt', i, j), s, N, M1)
            else:
                Stt[a][b] = Stt[b][a] = cov_tm(M, T[b], T[a], ('tt', j, i), s, N, M1)
    zero = TM.const(0, s, N)
    kvec = [TM.from_fracs({3: Fr(-1, 2)}, 0, s, N), TM.from_fracs({2: Fr(-1)}, 0, s, N), TM.const(12, s, N)] + [zero] * (n - 3)
    e1 = [TM.const(1, s, N)] + [zero] * (n - 1)
    X, det = solve_tm(Spp, Stp + [e1, kvec])
    W = X[:nt]
    xs0, xs1 = X[nt], X[nt + 1]
    law = TMLaw()
    law.M1, law.th1, law.s, law.N = M1, M.theta_minus_one(L), s, N
    law.alpha = {nm: W[a][0] for a, nm in enumerate(names)}
    law.beta = {nm: sum((W[a][j] * kvec[j] for j in range(n)), zero) for a, nm in enumerate(names)}
    law.Cu = {}
    for a, na in enumerate(names):
        for b, nb in enumerate(names):
            law.Cu[(na, nb)] = Stt[a][b] - sum((W[a][l] * Stp[b][l] for l in range(n)), zero)
    law.e_bb = xs0[0]
    law.e_bk = sum((kvec[j] * xs0[j] for j in range(n)), zero)
    law.e_kk = sum((kvec[j] * xs1[j] for j in range(n)), zero)
    law.det = det
    return law
