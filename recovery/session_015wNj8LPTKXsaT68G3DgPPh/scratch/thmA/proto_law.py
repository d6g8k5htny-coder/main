import math
from fractions import Fraction as Fr
from engine import functionals, cov_series, PIN_NAMES, TARGET_NAMES

NX = 44
pins, targets = functionals()
P = [pins[n] for n in PIN_NAMES]
T = [targets[n] for n in TARGET_NAMES]
series = {}
def ser(A, B, key):
    if key not in series:
        series[key] = cov_series(A, B, NX)
    return series[key]
def ev(s, r):
    return sum(float(c) * r ** p for p, c in s.items())

def law(r, b, k):
    Spp = [[ev(ser(P[i], P[j], ('p', i, j)), r) for j in range(6)] for i in range(6)]
    Stp = [[ev(ser(T[i], P[j], ('tp', i, j)), r) for j in range(6)] for i in range(12)]
    Stt = [[ev(ser(T[i], T[j], ('tt', i, j)), r) for j in range(12)] for i in range(12)]
    # solve Spp X = Stp^T columns
    import copy
    def solve(A, bvec):
        n = len(A); M = [A[i][:] + [bvec[i]] for i in range(n)]
        for c in range(n):
            p = max(range(c, n), key=lambda i: abs(M[i][c])); M[c], M[p] = M[p], M[c]
            for i in range(c + 1, n):
                f = M[i][c] / M[c][c]
                for j in range(c, n + 1): M[i][j] -= f * M[c][j]
        x = [0] * n
        for i in reversed(range(n)):
            x[i] = (M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]
        return x
    vr = [b - k * r ** 3 / 2, -k * r * r, 12 * k, 0, 0, 0]
    Wt = [solve(Spp, Stp[i]) for i in range(12)]
    mu = [sum(Wt[i][j] * vr[j] for j in range(6)) for i in range(12)]
    C = [[Stt[i][j] - sum(Wt[i][l] * Stp[j][l] for l in range(6)) for j in range(12)] for i in range(12)]
    xE = solve(Spp, vr)
    E = sum(vr[j] * xE[j] for j in range(6))
    return mu, C, E

if __name__ == '__main__':
    for r in [0.0, 1/512, 1/256]:
        mu, C, E = law(r, 0.5, 1.0)
        print('r=%g energy=%.4f' % (r, E))
        for i, n in enumerate(TARGET_NAMES):
            print('  %-4s mean %+.5f  sd %.5f   corr_w %+.4f' % (n, mu[i], math.sqrt(max(C[i][i], 0)), C[i][0] / math.sqrt(C[i][i] * C[0][0])))
