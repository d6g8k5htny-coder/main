"""Floating control: (A_r/A_0^ref - 1)/r^2 for the reference kernel by Monte Carlo of Z_r/r^2 (exact law at r) and the exact pin ratio."""
import math, random, sys, importlib.util, os
from fractions import Fraction as Fr
spec = importlib.util.spec_from_file_location('tb', sys.argv[1]); tb = importlib.util.module_from_spec(spec); spec.loader.exec_module(tb)

def chol(C):
    n = len(C); L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = C[i][j] - sum(L[i][l] * L[j][l] for l in range(j))
            L[i][j] = math.sqrt(max(s, 0.0)) if i == j else (s / L[j][j] if L[j][j] > 0 else 0.0)
    return L

def exact_cov(M, F, G, key, r):
    ser, w, _ = M.pair_series(F, G, key)
    return sum(c * r ** p for p, c in ser.items())

def fsolve(A, B):
    n = len(A)
    Mx = [list(A[i]) + [B[c][i] for c in range(len(B))] for i in range(n)]
    for col in range(n):
        for i in range(col + 1, n):
            f = Mx[i][col] / Mx[col][col]
            for j in range(col, len(Mx[i])):
                Mx[i][j] -= f * Mx[col][j]
    X = []
    for c in range(len(B)):
        x = [None] * n
        for i in reversed(range(n)):
            x[i] = (Mx[i][n + c] - sum(Mx[i][j] * x[j] for j in range(i + 1, n))) / Mx[i][i]
        X.append(x)
    return X

def law_at(d, r, b, k, names):
    # the exact reference-kernel law at rational r (series truncated at the engine's NX; reference kernel only)
    M = tb.ENG[d]
    r = Fr(r)
    pins, targets = M.functionals()
    P = [pins[n] for n in M.PIN_NAMES]
    idx = [M.TARGET_NAMES.index(n) for n in names]
    T = [targets[n] for n in names]
    n = len(P)
    Spp = [[exact_cov(M, P[i], P[j], ('p', min(i, j), max(i, j)), r) if i <= j else None for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(i):
            Spp[i][j] = Spp[j][i]
    Stp = [[exact_cov(M, T[a], P[j], ('tp', idx[a], j), r) for j in range(n)] for a in range(len(T))]
    def stt(a, c):
        i, j = idx[a], idx[c]
        return exact_cov(M, T[a], T[c], ('tt', i, j), r) if i <= j else exact_cov(M, T[c], T[a], ('tt', j, i), r)
    kvec = [-r ** 3 / 2, -r ** 2, Fr(12)] + [Fr(0)] * (n - 3)
    W = fsolve(Spp, Stp)
    mu = [float(Fr(b) * W[a][0] + Fr(k) * sum(W[a][j] * kvec[j] for j in range(n))) for a in range(len(T))]
    C = [[float(stt(a, c) - sum(W[a][l] * Stp[c][l] for l in range(n))) for c in range(len(T))] for a in range(len(T))]
    return mu, chol(C)

def pin_ratio(d, tml, r, b, k):
    ev = lambda t: sum(float(x) * r ** j for j, x in enumerate(t.frac_coeffs()))
    det = ev(tml.det); E = ev(tml.e_bb) * b * b + 2 * ev(tml.e_bk) * b * k + ev(tml.e_kk) * k * k
    return math.sqrt(12 / det) * math.exp(-(E - (1.5 * b * b + 24 * k * k)) / 2)

def m_ref(d, b):
    return float(tb.m_ref(d, Fr(b)).mid())

def sample_W_d2(r, k, mu, L, n, rng):
    tot = 0.0; tot2 = 0.0
    m = len(mu)
    for _ in range(n):
        z = [rng.gauss(0.0, 1.0) for _ in range(m)]
        x = [mu[i] + sum(L[i][j] * z[j] for j in range(i + 1)) for i in range(m)]
        w, v, om, T, tau, nu = x
        lam = -w
        c1 = 6 * k - r * T; c2 = 6 * k + r * T + r * r * tau; e = (r * nu - v) ** 2
        P1 = c1 * lam - r * v * v; P2 = c2 * (lam - r * om) + r * e
        W = P1 * P2 if (c1 > 0 and P1 > 0 and P2 > 0) else 0.0
        tot += W; tot2 += W * W
    mean = tot / n
    return mean, math.sqrt(max(tot2 / n - mean * mean, 0.0) / n)

def sample_W_d3(r, k, mu, L, n, rng):
    tot = 0.0; tot2 = 0.0
    m = len(mu)
    for _ in range(n):
        z = [rng.gauss(0.0, 1.0) for _ in range(m)]
        x = [mu[i] + sum(L[i][j] * z[j] for j in range(i + 1)) for i in range(m)]
        a11, a22, a12, v1, v2, o11, o22, o12, T, tau, nu1, nu2 = x
        B = (-a11, -a22, -a12)                       # B = -A
        trB = B[0] + B[1]; detB = B[0] * B[1] - B[2] * B[2]
        BS = (B[0] - r * o11, B[1] - r * o22, B[2] - r * o12)
        detBS = BS[0] * BS[1] - BS[2] * BS[2]
        c1 = 6 * k - r * T; c2 = 6 * k + r * T + r * r * tau
        e1, e2 = r * nu1 - v1, r * nu2 - v2
        vAv = B[1] * v1 * v1 - 2 * B[2] * v1 * v2 + B[0] * v2 * v2          # v' adj(B) v
        eAe = BS[1] * e1 * e1 - 2 * BS[2] * e1 * e2 + BS[0] * e2 * e2       # e' adj(B_S) e
        P1 = c1 * detB - r * vAv; P2 = c2 * detBS + r * eAe
        # exact typed event: H_M < 0 and index(H_S) = 2, via the eigenvalues of the 3 x 3 Hessians (scaled by 1/r on the x-row)
        ok = False
        if B[0] > 0 and detB > 0 and P1 > 0 and P2 > 0:
            # index(H_S) = 2 <=> det H_S > 0 and H_S not negative or positive definite; with -B_S block: use the trace test
            # H_S = [[r c2, r e'], [r e, -B_S]]: count negative eigenvalues by Sylvester on the order (x, y1, y2)
            l1 = -BS[0]; l2 = l1 * (-BS[1]) - BS[2] * BS[2]
            # signs of leading minors of [[-B_S, r e], [r e', r c2]] (reordered): -B_S11, det(-B_S), det(H_S) > 0
            neg = (1 if l1 < 0 else 0) + (1 if (l2 / l1 if l1 != 0 else 1) < 0 else 0) + (1 if P2 * r / (l2 if l2 != 0 else 1) < 0 else 0)
            ok = (neg == 2)
        W = P1 * P2 if ok else 0.0
        tot += W; tot2 += W * W
    mean = tot / n
    return mean, math.sqrt(max(tot2 / n - mean * mean, 0.0) / n)

if __name__ == '__main__':
    d = int(sys.argv[2]); n = int(sys.argv[3])
    tml = tb.tm_law(d)
    names = ('w', 'v', 'om', 'T', 'tau', 'nu') if d == 2 else ('a11', 'a22', 'a12', 'v1', 'v2', 'o11', 'o22', 'o12', 'T', 'tau', 'nu1', 'nu2')
    rng = random.Random(20261001)
    for r in (1 / 8, 1 / 16):
        for b, k in ((0, 0.5), (0, 2), (1, 0.5), (1, 2)):
            mu, L = law_at(d, r, b, k, names)
            mean, se = (sample_W_d2 if d == 2 else sample_W_d3)(r, k, mu, L, n, rng)
            zr = mean / (36 * k * k * m_ref(d, b)); zse = se / (36 * k * k * m_ref(d, b))
            pr = pin_ratio(d, tml, r, b, k)
            rho = pr * zr
            print('d=%d r=1/%d b=%g k=%g  Z-ratio %.5f +- %.5f  pin ratio %.6f  (A_r/A_0 - 1)/r^2 = %.3f +- %.3f' % (
                d, round(1 / r), b, k, zr, zse, pr, (rho - 1) / r ** 2, pr * zse / r ** 2), flush=True)
