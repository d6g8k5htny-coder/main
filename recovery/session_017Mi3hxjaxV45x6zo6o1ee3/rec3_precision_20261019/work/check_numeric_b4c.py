"""Numerical checks for METHOD V2 section 2 against the ORIGINAL, at finite L.

Claim M193-V2S2-XFAM-INPUT-017-20261019. Input only; scratch use.
Two text paths: V2 (h, alpha, M1-M4) and ORIGINAL (q_m, (1), (6), (8), (11), (12)).
A third, literal-law path sums the product lattice law directly in d dimensions.
"""
import random
import time
import mpmath as mp

mp.mp.dps = 80
T0 = time.time()
LOG = []


def out(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def trunc(alpha, extra=30):
    return int(mp.ceil(mp.sqrt((mp.mp.dps + extra) * mp.log(10) / alpha))) + 2


# ---- V2 text path (section 2): h=2pi/L, alpha=h^2/2, z, mu_2k
def moments_v2(L):
    h = 2 * mp.pi / L
    alpha = h**2 / 2
    M = trunc(alpha)
    ms = range(-M, M + 1)
    z = mp.fsum(mp.exp(-alpha * m * m) for m in ms)

    def mu(k):
        return mp.fsum((h * m)**(2 * k) * mp.exp(-alpha * m * m) for m in ms) / z
    a, b, c = mu(1), mu(2), mu(3)
    return dict(z=z, a=a, b=b, c=c, k4=b - 3 * a**2, k6=c - 15 * a * b + 30 * a**3)


# ---- ORIGINAL text path (section 2): q_m = 2 pi m/L, z_L, mu_2k
def moments_orig(L):
    def q(m):
        return 2 * mp.pi * m / L
    M = trunc(2 * mp.pi**2 / L**2)
    ms = range(-M, M + 1)
    zL = mp.fsum(mp.exp(-q(m)**2 / 2) for m in ms)

    def mu(k):
        return mp.fsum(q(m)**(2 * k) * mp.exp(-q(m)**2 / 2) for m in ms) / zL
    a, b, c = mu(1), mu(2), mu(3)
    return dict(z=zL, a=a, b=b, c=c, k4=b - 3 * a**2, k6=c - 15 * a * b + 30 * a**3)


def frame(e):
    """Orthonormal completion of unit e by Gram-Schmidt (columns U[:,j])."""
    d = len(e)
    basis = [list(e)]
    for k in range(d):
        v = [mp.mpf(int(i == k)) for i in range(d)]
        for w in basis:
            p = mp.fsum(v[i] * w[i] for i in range(d))
            v = [v[i] - p * w[i] for i in range(d)]
        n = mp.sqrt(mp.fsum(x * x for x in v))
        if n > mp.mpf('1e-8'):
            basis.append([x / n for x in v])
        if len(basis) == d:
            break
    return basis[1:]  # list of d-1 vectors


def pairs_of(n):
    return [(p, q) for p in range(n) for q in range(p, n)]


# closed forms shared by both texts (M2 = (6), M3 = (8), Gamma_e = (11))
def B_closed(mo, e):
    d = len(e)
    return mp.matrix([[mo['a']**2 * (int(i == j) + 2 * e[i] * e[j]) + mo['k4'] * int(i == j) * e[i]**2
                       for j in range(d)] for i in range(d)])


def V_closed(mo, e):
    S4 = mp.fsum(x**4 for x in e)
    S6 = mp.fsum(x**6 for x in e)
    a = mo['a']
    return 6 * a**3 + 9 * a * mo['k4'] * S4 + (mo['k6'] - mo['k4']**2 / a) * S6


def Gamma_closed(mo, e, U):
    d = len(e)
    n = d - 1
    P = pairs_of(n)
    a, k4 = mo['a'], mo['k4']

    def dl(x, y):
        return int(x == y)
    C = mp.matrix(len(P), len(P))
    R = mp.matrix(len(P), d)
    for r, (p1, p2) in enumerate(P):
        for s, (s1, s2) in enumerate(P):
            C[r, s] = a**2 * (dl(p1, p2) * dl(s1, s2) + dl(p1, s1) * dl(p2, s2) + dl(p1, s2) * dl(p2, s1)) \
                + k4 * mp.fsum(U[p1][i] * U[p2][i] * U[s1][i] * U[s2][i] for i in range(d))
        for i in range(d):
            R[r, i] = a**2 * dl(p1, p2) * e[i] + k4 * U[p1][i] * U[p2][i] * e[i]
    B = B_closed(mo, e)
    return C - R * mp.inverse(B) * R.T


# literal law: direct d-dimensional sum over the product lattice
def literal(L, e, U):
    d = len(e)
    h = 2 * mp.pi / L
    alpha = h**2 / 2
    M = trunc(alpha, extra=15)
    ms = list(range(-M, M + 1))
    w1 = {m: mp.exp(-alpha * m * m) for m in ms}
    z = mp.fsum(w1.values())
    n = d - 1
    P = pairs_of(n)
    acc_B = [[mp.mpf(0)] * d for _ in range(d)]
    acc_6 = mp.mpf(0)
    acc_v = [mp.mpf(0)] * d
    acc_C = [[mp.mpf(0)] * len(P) for _ in range(len(P))]
    acc_R = [[mp.mpf(0)] * d for _ in range(len(P))]

    def rec(idx, Q, w):
        nonlocal acc_6
        if idx == d:
            Qe = mp.fsum(Q[i] * e[i] for i in range(d))
            QU = [mp.fsum(Q[i] * U[j][i] for i in range(d)) for j in range(n)]
            for i in range(d):
                for j in range(d):
                    acc_B[i][j] += w * Q[i] * Q[j] * Qe**2
                acc_v[i] += w * Q[i] * Qe**3
            acc_6 += w * Qe**6
            for r, (p1, p2) in enumerate(P):
                x = QU[p1] * QU[p2]
                for s, (s1, s2) in enumerate(P):
                    acc_C[r][s] += w * x * QU[s1] * QU[s2]
                for i in range(d):
                    acc_R[r][i] += w * x * Q[i] * Qe
            return
        for m in ms:
            rec(idx + 1, Q + [h * m], w * w1[m] / z)
    rec(0, [], mp.mpf(1))
    B = mp.matrix(acc_B)
    a = mp.fsum(w1[m] * (h * m)**2 for m in ms) / z
    V = acc_6 - mp.fsum(x * x for x in acc_v) / a
    C = mp.matrix(acc_C)
    R = mp.matrix(acc_R)
    G = C - R * mp.inverse(B) * R.T
    return B, V, G


def maxrel(A, B):
    if isinstance(A, mp.matrix):
        return max(abs(A[i, j] - B[i, j]) / max(abs(B[i, j]), mp.mpf('1e-30'))
                   for i in range(A.rows) for j in range(A.cols))
    return abs(A - B) / abs(B)



def F_2x2w(G):
    """E[(det A)^2 1{A<0}] for s(A)=(A11,A12,A22) ~ N(0,G), in WHITENED (q,s) coordinates.

    p=(A11+A22)/2, q=(A11-A22)/2, s=A12; (q,s) = C (u,v) with C the Cholesky factor of Cov(q,s),
    (u,v) standard normal; p | (q,s) ~ N(mu, sig^2); the inner p-integral is closed form.
    The outer integral is over (u,v) in polar coordinates, so the partition is scale-free.
    """
    T = mp.matrix([[mp.mpf(1) / 2, 0, mp.mpf(1) / 2], [mp.mpf(1) / 2, 0, -mp.mpf(1) / 2], [0, 1, 0]])
    S = T * G * T.T
    Sqq = mp.matrix([[S[1, 1], S[1, 2]], [S[2, 1], S[2, 2]]])
    Spq = mp.matrix([[S[0, 1], S[0, 2]]])
    beta = Spq * mp.inverse(Sqq)
    sig2 = S[0, 0] - (beta * Spq.T)[0, 0]
    sig = mp.sqrt(sig2)
    c11 = mp.sqrt(Sqq[0, 0]); c21 = Sqq[1, 0] / c11; c22 = mp.sqrt(Sqq[1, 1] - c21 * c21)

    def inner(qv, sv):
        rho2 = qv * qv + sv * sv
        rho = mp.sqrt(rho2)
        mu = beta[0, 0] * qv + beta[0, 1] * sv
        x = (-rho - mu) / sig
        ph = mp.npdf(x); Ph = mp.ncdf(x)
        I0, I1, I2 = Ph, -ph, Ph - x * ph
        I3, I4 = -(x * x + 2) * ph, 3 * Ph - (x**3 + 3 * x) * ph
        c0 = (mu * mu - rho2)**2
        c1 = 4 * mu * sig * (mu * mu - rho2)
        c2 = sig2 * (6 * mu * mu - 2 * rho2)
        c3 = 4 * mu * sig**3
        c4 = sig2**2
        return c0 * I0 + c1 * I1 + c2 * I2 + c3 * I3 + c4 * I4

    def integrand(r, th):
        u, v = r * mp.cos(th), r * mp.sin(th)
        qv = c11 * u
        sv = c21 * u + c22 * v
        return inner(qv, sv) * mp.exp(-r * r / 2) / (2 * mp.pi) * r
    return mp.quad(integrand, [0, 1, 2, 4, 8, mp.inf], [0, mp.pi / 4, mp.pi / 2, 3 * mp.pi / 4, mp.pi,
                                                       5 * mp.pi / 4, 3 * mp.pi / 2, 7 * mp.pi / 4, 2 * mp.pi])


def F_mc(G, n, seed):
    Gm = [[float(G[i, j]) for j in range(3)] for i in range(3)]
    # Cholesky
    Lc = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(i + 1):
            s = Gm[i][j] - sum(Lc[i][k] * Lc[j][k] for k in range(j))
            Lc[i][j] = s**0.5 if i == j else s / Lc[j][j]
    r = random.Random(seed)
    acc = acc2 = 0.0
    for _ in range(n):
        g = [r.gauss(0, 1) for _ in range(3)]
        x = [sum(Lc[i][k] * g[k] for k in range(3)) for i in range(3)]
        a11, a12, a22 = x
        det = a11 * a22 - a12 * a12
        val = det * det if (a11 < 0 and det > 0) else 0.0
        acc += val
        acc2 += val * val
    m = acc / n
    se = ((acc2 / n - m * m) / n)**0.5
    return m, se




def unit(v):
    n = mp.sqrt(mp.fsum(mp.mpf(x)**2 for x in v))
    return [mp.mpf(x) / n for x in v]


out('== B4c (run 4) d=3 F(e) by (11a) with WHITENED quadrature; convergence at 20 vs 30 digits; Monte Carlo of (10)')
# L = infinity: independent 1-D-outer formula (p independent of (q,s); rho^2 ~ chi^2_2)
mp.mp.dps = 30
s2 = mp.mpf(5) / 3
def inner_inf(rho):
    sig = mp.sqrt(s2); x = -rho / sig
    ph = mp.npdf(x); Ph = mp.ncdf(x)
    I0, I2, I4 = Ph, Ph - x * ph, 3 * Ph - (x**3 + 3 * x) * ph
    return rho**4 * I0 - 2 * rho**2 * s2 * I2 + s2**2 * I4
Fs_1d = mp.quad(lambda r: r * mp.exp(-r * r / 2) * inner_inf(r), [0, 1, 2, 4, 8, mp.inf])
Ginf = mp.matrix([[mp.mpf(8) / 3, 0, mp.mpf(2) / 3], [0, 1, 0], [mp.mpf(2) / 3, 0, mp.mpf(8) / 3]])
Fs_w30 = F_2x2w(Ginf)
mp.mp.dps = 20
Fs_w20 = F_2x2w(Ginf)
out(f'L=inf F*_2: 1-D outer formula (30 digits) {mp.nstr(Fs_1d, 25)}; whitened 2-D quad 30 digits {mp.nstr(Fs_w30, 25)}; 20 digits {mp.nstr(Fs_w20, 18)}')
dirs3 = [unit([1, 1, 1]), unit(['0.3', '-0.5', '0.81'])]
for L in (1, 2, 3):
    mp.mp.dps = 80
    mo = moments_v2(L)
    for k, e in enumerate(dirs3):
        mp.mp.dps = 80
        U = frame(e)
        Gc = Gamma_closed(mo, e, U)
        mp.mp.dps = 30
        G30 = mp.matrix([[+Gc[i, j] for j in range(3)] for i in range(3)])
        F30 = F_2x2w(G30)
        mp.mp.dps = 20
        G20 = mp.matrix([[+Gc[i, j] for j in range(3)] for i in range(3)])
        F20 = F_2x2w(G20)
        m, se = F_mc(G20, 400000, 11 + 10 * L + k)
        out(f'd=3 L={L} e={[mp.nstr(x, 4) for x in e]}: F whitened 30 digits {mp.nstr(F30, 22)}; 20 digits {mp.nstr(F20, 16)}; '
            f'rel diff {mp.nstr(abs(F30 - F20) / abs(F30), 2)}; MC {m:.6g} +- {se:.2g} ({(m - float(F30)) / se:+.2f} sigma)')
out(f'elapsed {time.time() - T0:.1f}s')
