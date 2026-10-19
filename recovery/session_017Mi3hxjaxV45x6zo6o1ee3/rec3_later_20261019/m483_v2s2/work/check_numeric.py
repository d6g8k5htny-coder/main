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


# ---- B1: moments, two text paths
out('== B1 moments: V2 text path vs ORIGINAL text path')
MO = {}
for L in (1, 2, 3, 24):
    m2, mo = moments_v2(L), moments_orig(L)
    diff = max(abs(m2[k] - mo[k]) / max(abs(mo[k]), mp.mpf('1e-300')) for k in ('z', 'a', 'b', 'c'))
    MO[L] = m2
    out(f'L={L}: max rel diff (z,a,b,c) = {mp.nstr(diff, 3)}; a={mp.nstr(m2["a"], 12)} '
        f'k4={mp.nstr(m2["k4"], 6)} k6={mp.nstr(m2["k6"], 6)}')

# ---- B2: closed forms vs literal lattice law, d=2,3
out('== B2 closed forms (M2/(6), M3/(8), (11)) vs literal product-lattice law')
rng = random.Random(483)


def unit(v):
    n = mp.sqrt(mp.fsum(mp.mpf(x)**2 for x in v))
    return [mp.mpf(x) / n for x in v]


dirs = {2: [unit([1, 0]), unit([1, 1]), unit([mp.cos(mp.mpf('0.3')), mp.sin(mp.mpf('0.3'))]),
            unit([mp.cos(mp.pi / 5), mp.sin(mp.pi / 5)])],
        3: [unit([1, 0, 0]), unit([1, 1, 0]), unit([1, 1, 1]), unit(['0.3', '-0.5', '0.81']),
            unit([rng.gauss(0, 1) for _ in range(3)])]}
worst = mp.mpf(0)
for d in (2, 3):
    for L in (1, 2, 3):
        mo = MO[L]
        for e in dirs[d]:
            U = frame(e)
            Bl, Vl, Gl = literal(L, e, U)
            eB = maxrel(B_closed(mo, e), Bl)
            eV = maxrel(V_closed(mo, e), Vl)
            eG = maxrel(Gamma_closed(mo, e, U), Gl)
            worst = max(worst, eB, eV, eG)
            out(f'd={d} L={L} e={[mp.nstr(x, 4) for x in e]}: relerr B {mp.nstr(eB, 2)}, V {mp.nstr(eV, 2)}, '
                f'Gamma {mp.nstr(eG, 2)}')
out(f'B2 worst relative error = {mp.nstr(worst, 3)}')


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


out('== B3 C_alg(2,L): V2 path (M1, F=Var/2 via (11)) vs ORIGINAL path ((1) with (12))')


def K(mo, d):
    return mp.gamma(mp.mpf(7) / 6) / (mp.cbrt(24) * mp.sqrt(mp.pi) * (2 * mp.pi)**d * mo['a']**(mp.mpf(d) / 2))


def C2(mo, Ffun):
    def f(th):
        e = [mp.cos(th), mp.sin(th)]
        return Ffun(mo, e) * V_closed(mo, e)**(mp.mpf(2) / 3) / mp.sqrt(mp.det(B_closed(mo, e)))
    return K(mo, 2) * 8 * mp.quad(f, [0, mp.pi / 8, mp.pi / 4])


cinf2 = 2 / (3 * mp.pi * mp.sqrt(3)) * mp.cbrt(mp.mpf(3) / 2) * mp.gamma(mp.mpf(7) / 6) / mp.sqrt(mp.pi)
for L in (1, 2, 3, 24):
    mo = MO[L]
    v2 = C2(mo, F2_v2)
    og = C2(mo, F2_orig)
    out(f'L={L}: C_alg(2,L) V2 path = {mp.nstr(v2, 30)}; ORIGINAL path rel diff = {mp.nstr(abs(v2 - og) / og, 3)}; '
        f'ratio to c_2,inf = {mp.nstr(v2 / cinf2, 15)}')
out(f'c_2,inf (ORIGINAL (32)) = {mp.nstr(cinf2, 30)}')


# ---- d=3 F(e) via (11a) for a 2x2 block: reduce to a 2-D integral
mp.mp.dps = 20


def F_2x2(G, quad_dps=20):
    """E[(det A)^2 1{A<0}] for s(A)=(A11,A12,A22) ~ N(0,G), via (p,q,s)."""
    T = mp.matrix([[mp.mpf(1) / 2, 0, mp.mpf(1) / 2],   # p=(A11+A22)/2
                   [mp.mpf(1) / 2, 0, -mp.mpf(1) / 2],  # q=(A11-A22)/2
                   [0, 1, 0]])                          # s=A12
    S = T * G * T.T
    Sqq = mp.matrix([[S[1, 1], S[1, 2]], [S[2, 1], S[2, 2]]])
    Spq = mp.matrix([[S[0, 1], S[0, 2]]])
    Sinv = mp.inverse(Sqq)
    beta = Spq * Sinv
    sig2 = S[0, 0] - (beta * Spq.T)[0, 0]
    sig = mp.sqrt(sig2)
    detq = mp.det(Sqq)

    def inner(qv, sv):
        rho2 = qv * qv + sv * sv
        rho = mp.sqrt(rho2)
        mu = beta[0, 0] * qv + beta[0, 1] * sv
        x = (-rho - mu) / sig
        ph = mp.npdf(x)
        Ph = mp.ncdf(x)
        I0, I1, I2 = Ph, -ph, Ph - x * ph
        I3, I4 = -(x * x + 2) * ph, 3 * Ph - (x**3 + 3 * x) * ph
        # ((mu + sig t)^2 - rho^2)^2 expanded in t
        c0 = (mu * mu - rho2)**2
        c1 = 4 * mu * sig * (mu * mu - rho2)
        c2 = sig2 * (6 * mu * mu - 2 * rho2)
        c3 = 4 * mu * sig**3
        c4 = sig2**2
        return c0 * I0 + c1 * I1 + c2 * I2 + c3 * I3 + c4 * I4

    def dens(qv, sv):
        y = mp.matrix([qv, sv])
        return mp.exp(-(y.T * Sinv * y)[0, 0] / 2) / (2 * mp.pi * mp.sqrt(detq))

    def integrand(r, th):
        qv, sv = r * mp.cos(th), r * mp.sin(th)
        return inner(qv, sv) * dens(qv, sv) * r
    return mp.quad(integrand, [0, 1, 3, mp.inf], [0, mp.pi / 2, mp.pi, 3 * mp.pi / 2, 2 * mp.pi])


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


out('== B4 d=3: F(e) by (11a) (2-D reduced quadrature), validated by Monte Carlo of (10)')
Ginf = mp.matrix([[mp.mpf(8) / 3, 0, mp.mpf(2) / 3], [0, 1, 0], [mp.mpf(2) / 3, 0, mp.mpf(8) / 3]])
Fs2 = F_2x2(Ginf)
m, se = F_mc(Ginf, 400000, 1)
out(f'L=inf GOE_2+sqrt(2/3)ZI: F*_2 quad = {mp.nstr(Fs2, 15)}; MC = {m:.5f} +- {se:.5f}')
cinf3 = 4 * mp.pi / ((2 * mp.pi)**3 * mp.sqrt(3)) * mp.cbrt(mp.mpf(3) / 2) * mp.gamma(mp.mpf(7) / 6) / mp.sqrt(mp.pi) * Fs2
out(f'c_3,inf by ORIGINAL (31) with this F*_2 = {mp.nstr(cinf3, 15)}')
for L in (1, 2, 3):
    mo = {k: +v for k, v in MO[L].items()}
    for e in dirs[3][2:4]:
        U = frame(e)
        Gc = Gamma_closed(mo, e, U)
        mp.mp.dps = 40
        _, _, Gl = literal(L, e, U)
        mp.mp.dps = 20
        Fc = F_2x2(Gc)
        Fl = F_2x2(Gl)
        line = (f'd=3 L={L} e={[mp.nstr(x, 4) for x in e]}: F closed-(11) = {mp.nstr(Fc, 15)}, '
                f'F literal = {mp.nstr(Fl, 15)}, rel diff {mp.nstr(abs(Fc - Fl) / Fl, 3)}')
        if L == 3:
            m, se = F_mc(Gc, 400000, 7)
            line += f'; MC {m:.5f} +- {se:.5f}'
        out(line)
out(f'elapsed {time.time() - T0:.1f}s')
