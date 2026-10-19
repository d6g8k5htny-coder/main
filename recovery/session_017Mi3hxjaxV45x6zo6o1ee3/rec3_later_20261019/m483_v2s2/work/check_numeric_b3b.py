"""Numerical checks for METHOD V2 section 2 against the ORIGINAL, at finite L.

Claim M193-V2S2-XFAM-INPUT-017-20261019. Input only; scratch use.
Two text paths: V2 (h, alpha, M1-M4) and ORIGINAL (q_m, (1), (6), (8), (11), (12)).
A third, literal-law path sums the product lattice law directly in d dimensions.
"""
import random
import time
import mpmath as mp

mp.mp.dps = 40
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



# ---- d=2 F: V2 path (Var(A_e)/2 from (11)) and ORIGINAL path ((12))
def F2_v2(mo, e):
    U = frame(e)
    return Gamma_closed(mo, e, U)[0, 0] / 2


def F2_orig(mo, e):
    v = [-e[1], e[0]]
    a, k4 = mo['a'], mo['k4']
    B = B_closed(mo, e)
    D = mp.matrix([e[i] * (a**2 + k4 * v[i]**2) for i in range(2)])
    return (3 * a**2 + k4 * (v[0]**4 + v[1]**4) - (D.T * mp.inverse(B) * D)[0, 0]) / 2



def K(mo, d):
    return mp.gamma(mp.mpf(7) / 6) / (mp.cbrt(24) * mp.sqrt(mp.pi) * (2 * mp.pi)**d * mo['a']**(mp.mpf(d) / 2))



out('== B3b quadrature convergence for C_alg(2,L), V2 path, dps 40')
for L in (1, 2, 3):
    mo = moments_v2(L)
    def f(th):
        e = [mp.cos(th), mp.sin(th)]
        return F2_v2(mo, e) * V_closed(mo, e)**(mp.mpf(2) / 3) / mp.sqrt(mp.det(B_closed(mo, e)))
    v1, e1 = mp.quad(f, [0, mp.pi / 8, mp.pi / 4], error=True)
    pts = [mp.pi / 4 * mp.mpf(k) / 64 for k in range(65)]
    v2, e2 = mp.quad(f, pts, error=True, maxdegree=10)
    v3 = mp.quad(f, [0, mp.pi / 4], method='gauss-legendre', maxdegree=10)
    k = K(mo, 2) * 8
    out(f'L={L}: 3-pt tanh-sinh {mp.nstr(k*v1, 28)} (err est {mp.nstr(k*e1, 2)}); 64-panel {mp.nstr(k*v2, 28)} (err est {mp.nstr(k*e2, 2)}); '
        f'gauss-legendre {mp.nstr(k*v3, 28)}; max rel spread {mp.nstr(max(abs(v1-v2), abs(v1-v3), abs(v2-v3))/abs(v2), 3)}')
