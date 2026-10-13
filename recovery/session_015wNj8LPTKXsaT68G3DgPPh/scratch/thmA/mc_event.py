"""End-to-end float control: sample the 7-jet at 0 under Q (reference kernel), rebuild f near D by Taylor, evaluate
W and the event G_r^c (M3, M4 on a grid of D), estimate Q^W(G_r^c)/r^3."""
import math, random, sys
from fractions import Fraction as Fr
from engine import functionals, cov_series, PIN_NAMES, term, O_
NX = 50
DEG = 8
pins, _ = functionals()
P = [pins[n] for n in PIN_NAMES]
jets = [(i, n - i) for n in range(DEG + 1) for i in range(n, -1, -1)]  # (a, b) with a+b<=DEG
J = [[term(1, 0, a, b, O_)] for (a, b) in jets]

def matrices(r):
    ev = lambda s: sum(float(c) * r ** p for p, c in s.items())
    Spp = [[ev(cov_series(P[i], P[j], NX)) for j in range(6)] for i in range(6)]
    Sjp = [[ev(cov_series(J[i], P[j], NX)) for j in range(6)] for i in range(len(J))]
    Sjj = [[ev(cov_series(J[i], J[j], NX)) for j in range(len(J))] for i in range(len(J))]
    return Spp, Sjp, Sjj

def solve(A, b):
    n = len(A); M = [A[i][:] + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda i: abs(M[i][c])); M[c], M[p] = M[p], M[c]
        for i in range(c + 1, n):
            f = M[i][c] / M[c][c]
            for j in range(c, n + 1): M[i][j] -= f * M[c][j]
    x = [0] * n
    for i in reversed(range(n)):
        x[i] = (M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]
    return x

def cond_law(r, b, k):
    Spp, Sjp, Sjj = matrices(r)
    vr = [b - k * r ** 3 / 2, -k * r * r, 12 * k, 0, 0, 0]
    Wt = [solve(Spp, Sjp[i]) for i in range(len(J))]
    mu = [sum(Wt[i][j] * vr[j] for j in range(6)) for i in range(len(J))]
    C = [[Sjj[i][j] - sum(Wt[i][l] * Sjp[j][l] for l in range(6)) for j in range(len(J))] for i in range(len(J))]
    return mu, C

def eig_sqrt(C):
    # symmetric PSD square root by Jacobi
    n = len(C); A = [row[:] for row in C]; V = [[float(i == j) for j in range(n)] for i in range(n)]
    for sweep in range(100):
        off = sum(A[p][q] ** 2 for p in range(n) for q in range(p + 1, n))
        if off < 1e-26: break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(A[p][q]) < 1e-300: continue
                th = 0.5 * math.atan2(2 * A[p][q], A[q][q] - A[p][p]); c, s = math.cos(th), math.sin(th)
                for kk in range(n):
                    akp, akq = A[kk][p], A[kk][q]; A[kk][p], A[kk][q] = c * akp - s * akq, s * akp + c * akq
                for kk in range(n):
                    apk, aqk = A[p][kk], A[q][kk]; A[p][kk], A[q][kk] = c * apk - s * aqk, s * apk + c * aqk
                for kk in range(n):
                    vkp, vkq = V[kk][p], V[kk][q]; V[kk][p], V[kk][q] = c * vkp - s * vkq, s * vkp + c * vkq
    lam = [max(A[i][i], 0.0) for i in range(n)]
    return [[sum(V[i][kk] * math.sqrt(lam[kk]) * V[j][kk] for kk in range(n)) for j in range(n)] for i in range(n)]

def run(r, b, k, n, seed, grid=9):
    mu, C = cond_law(r, b, k)
    S = eig_sqrt(C)
    idx = {jt: i for i, jt in enumerate(jets)}
    fact = [math.factorial(i) for i in range(DEG + 2)]
    pts = [(-2 * r + 4 * r * i / (grid - 1), -2 * r + 4 * r * j / (grid - 1)) for i in range(grid) for j in range(grid)]
    def deriv(c, a0, b0, x, y):
        # d_x^a0 d_y^b0 of Taylor polynomial at (x, y)
        s = 0.0
        for (a, bb), i in idx.items():
            if a >= a0 and bb >= b0:
                s += c[i] * x ** (a - a0) * y ** (bb - b0) / (fact[a - a0] * fact[bb - b0])
        return s
    rng = random.Random(seed)
    num = den = 0.0
    a3 = 4 / (3 * k)
    for _ in range(n):
        z = [rng.gauss(0, 1) for _ in jets]
        c = [mu[i] + sum(S[i][j] * z[j] for j in range(len(jets))) for i in range(len(jets))]
        M, Sx = (-r / 2, 0.0), (r / 2, 0.0)
        hM = (deriv(c, 2, 0, *M), deriv(c, 1, 1, *M), deriv(c, 0, 2, *M))
        hS = (deriv(c, 2, 0, *Sx), deriv(c, 1, 1, *Sx), deriv(c, 0, 2, *Sx))
        dM = hM[0] * hM[2] - hM[1] ** 2; dS = hS[0] * hS[2] - hS[1] ** 2
        if not (hM[0] < 0 and dM > 0 and dS < 0):
            continue
        W = abs(dM * dS)
        den += W
        lam = -hM[2]
        M3 = max(abs(deriv(c, a, 3 - a, x, y)) for (x, y) in pts for a in range(4))
        M4 = max(abs(deriv(c, a, 4 - a, x, y)) for (x, y) in pts for a in range(5))
        if not (lam > a3 * r * M3 * M3 and r * M4 <= 3 * k / 10):
            num += W
    return num / den / r ** 3, den / n / r ** 2

if __name__ == '__main__':
    r, b, k, n = float(Fr(sys.argv[1])), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    est, Zr = run(r, b, k, n, 7)
    print('r=%g b=%g k=%g  Q^W(G^c)/r^3 = %.6g   Z/r^2 = %.5g' % (r, b, k, est, Zr))
