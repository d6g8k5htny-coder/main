"""Independent high-precision recomputation of the v23 single-rung H3 floor.

Written from the mathematics stated in addendum/H3_SINGLE_RUNG_PROOF.md (C319 S02), not from
any archive code. Plain mpmath at high precision: a numerical cross-check, NOT an interval
certificate.

Model (S02 sec. 1): d=2, L=24, K(x)=k(x1)k(x2), k(s)=sum_j exp(-(s+24j)^2/2)/sum_j exp(-(24j)^2/2).
Pins M=(-r/2,0), S=(r/2,0); C=(f,fx,fy at M, f,fx,fy at S) = c=(b,0,0,b-r^3/6,0,0).
Y=(fyy(M),fyy(S),fxx(M)/r,fxx(S)/r,fxy(M)/r,fxy(S)/r); D_i=alpha_i q_i - r beta_i^2; P=-D_M D_S.
Box B (S02 sec. 3) in whitened coordinates U of Y=mu+L U, L = lower Cholesky factor.
"""
import sys
sys.path.insert(0, '/tmp/claude-0/-home-user-main/ced74b96-60a0-5d59-b364-fb9a44030ace/scratchpad/pylib')
from mpmath import mp, mpf, exp, erf, sqrt, pi, matrix, cholesky, lu_solve, nstr, fabs

mp.dps = int(sys.argv[1]) if len(sys.argv) > 1 else 160
JIMG = int(sys.argv[2]) if len(sys.argv) > 2 else 3        # image range -JIMG..JIMG (0 = planar kernel)
GAP = mpf(sys.argv[3]) if len(sys.argv) > 3 else mpf(1) / 6  # k_gap (S02: 1/6; negative control: 1/12)

r = mpf(1) / 20
b = mpf(6) / 5
Lside = 24


def hermite_prob(n, u):
    """Probabilists' Hermite He_n(u): He_0=1, He_1=u, He_{n+1}=u He_n - n He_{n-1}."""
    h0, h1 = mpf(1), u
    if n == 0:
        return h0
    for k in range(1, n):
        h0, h1 = h1, u * h1 - k * h0
    return h1


NORM = sum(exp(-mpf(Lside * j) ** 2 / 2) for j in range(-JIMG, JIMG + 1))


def kder(n, s):
    """n-th derivative of the normalized periodized kernel k at s."""
    tot = mpf(0)
    for j in range(-JIMG, JIMG + 1):
        u = s + Lside * j
        tot += (-1) ** n * hermite_prob(n, u) * exp(-u * u / 2)
    return tot / NORM


def cov(p, a, q, bb):
    """Cov(D^a f(p), D^b f(q)) = (-1)^|b| D^(a+b) K(p-q), K(x)=k(x1)k(x2)."""
    sgn = (-1) ** (bb[0] + bb[1])
    return sgn * kder(a[0] + bb[0], p[0] - q[0]) * kder(a[1] + bb[1], p[1] - q[1])


M = (-r / 2, mpf(0))
S = (r / 2, mpf(0))
Cfun = [(M, (0, 0)), (M, (1, 0)), (M, (0, 1)), (S, (0, 0)), (S, (1, 0)), (S, (0, 1))]
Hfun = [(M, (2, 0)), (M, (0, 2)), (M, (1, 1)), (S, (2, 0)), (S, (0, 2)), (S, (1, 1))]
cvec = matrix([b, 0, 0, b - GAP * r ** 3, 0, 0])

G = matrix(6, 6)
X = matrix(6, 6)
T = matrix(6, 6)
for i in range(6):
    for j in range(6):
        G[i, j] = cov(Cfun[i][0], Cfun[i][1], Cfun[j][0], Cfun[j][1])
        X[i, j] = cov(Hfun[i][0], Hfun[i][1], Cfun[j][0], Cfun[j][1])
        T[i, j] = cov(Hfun[i][0], Hfun[i][1], Hfun[j][0], Hfun[j][1])

# symmetry sanity
asym = max(fabs(G[i, j] - G[j, i]) for i in range(6) for j in range(6))
asymT = max(fabs(T[i, j] - T[j, i]) for i in range(6) for j in range(6))

Ginv_c = lu_solve(G, cvec)
muH = X * Ginv_c
GinvXt = matrix(6, 6)
for j in range(6):
    col = lu_solve(G, matrix([X[j, k] for k in range(6)]))
    for i in range(6):
        GinvXt[i, j] = col[i]
SH = T - X * GinvXt

order = [1, 4, 0, 3, 2, 5]
scale = [mpf(1), mpf(1), 1 / r, 1 / r, 1 / r, 1 / r]
mu = [muH[order[i]] * scale[i] for i in range(6)]
Sig = matrix(6, 6)
for i in range(6):
    for j in range(6):
        Sig[i, j] = SH[order[i], order[j]] * scale[i] * scale[j]
Sig = (Sig + Sig.T) / 2
Lc = cholesky(Sig)
chol_res = max(fabs((Lc * Lc.T)[i, j] - Sig[i, j]) for i in range(6) for j in range(6))

lo = [mpf(-7), mpf(-8), mpf(-6), mpf(-8), mpf(-12) / 5, mpf(-8)]
hi = [mpf(18) / 25, mpf(2), mpf(6), mpf(8), mpf(12) / 5, mpf(8)]


def affine_range(i):
    a = mu[i]
    c = mu[i]
    for j in range(6):
        t1, t2 = Lc[i, j] * lo[j], Lc[i, j] * hi[j]
        a += min(t1, t2)
        c += max(t1, t2)
    return a, c


rng = [affine_range(i) for i in range(6)]
qM, qS, aM, aS, bM, bS = rng
absbM = max(fabs(bM[0]), fabs(bM[1]))
DM_lower = aM[1] * qM[1] - r * absbM ** 2     # S02 sec. 3 sign-aware bound
DS_upper = aS[0] * qS[1]

# ---- polynomial P(U) = -D_M D_S as dict {exponent tuple: coefficient}
def lin(i):
    p = {(0,) * 6: mu[i]}
    for j in range(6):
        if Lc[i, j] != 0:
            e = [0] * 6
            e[j] = 1
            p[tuple(e)] = p.get(tuple(e), 0) + Lc[i, j]
    return p


def pmul(p, q):
    out = {}
    for e1, c1 in p.items():
        for e2, c2 in q.items():
            e = tuple(x + y for x, y in zip(e1, e2))
            out[e] = out.get(e, 0) + c1 * c2
    return out


def padd(p, q, s=1):
    out = dict(p)
    for e, c in q.items():
        out[e] = out.get(e, 0) + s * c
    return out


Y = [lin(i) for i in range(6)]
DM = padd(pmul(Y[2], Y[0]), pmul(Y[4], Y[4]), -r)
DS = padd(pmul(Y[3], Y[1]), pmul(Y[5], Y[5]), -r)
P = {e: -c for e, c in pmul(DM, DS).items()}
nmono = len(P)
maxdeg = max(sum(e) for e in P)


def phi(t):
    return exp(-t * t / 2) / sqrt(2 * pi)


def In_table(a, c, nmax):
    I = [(erf(c / sqrt(2)) - erf(a / sqrt(2))) / 2, phi(a) - phi(c)]
    for n in range(2, nmax + 1):
        I.append(a ** (n - 1) * phi(a) - c ** (n - 1) * phi(c) + (n - 1) * I[n - 2])
    return I


Itab = [In_table(lo[i], hi[i], 4) for i in range(6)]
EPB = mpf(0)
for e, cf in P.items():
    term = cf
    for i in range(6):
        term *= Itab[i][e[i]]
    EPB += term
Zlo = r ** 2 * EPB

# full-space expectation via Gaussian moments (E U^n = (n-1)!! for even n) as a code consistency check
def gmom(n):
    if n % 2:
        return 0
    out = 1
    for k in range(n - 1, 0, -2):
        out *= k
    return out


EPfull = mpf(0)
for e, cf in P.items():
    term = cf
    for i in range(6):
        term *= gmom(e[i])
    EPfull += term

a2 = -kder(2, mpf(0))
a4 = kder(4, mpf(0))

S02_EPB = mpf('3.103716695013114347708303445895408329253736141612579387226512054910408')
S02_ZLO = mpf('0.007759291737532785869270758614738520823134340354031448468066280137276019')
ZLO_RN3 = mpf('0.0077592917375327855')

pr = lambda x, n=20: nstr(x, n)
print('dps', mp.dps, 'images', JIMG, 'gap', pr(GAP, 6))
print('G,T asym', pr(asym, 3), pr(asymT, 3), ' chol residual', pr(chol_res, 3))
print('1-a2', pr(1 - a2, 8), ' a4-3', pr(a4 - 3, 8))
print('mu', [pr(x, 15) for x in mu])
print('chol diag', [pr(Lc[i, i], 15) for i in range(6)])
print('box hi(qM)', pr(qM[1], 12), ' hi(qS)', pr(qS[1], 12), ' hi(aM)', pr(aM[1], 12),
      ' lo(aS)', pr(aS[0], 12), ' max|bM|', pr(absbM, 12))
print('DM lower (S02 rule)', pr(DM_lower, 16), ' DS upper (S02 rule)', pr(DS_upper, 16))
print('P monomials', nmono, 'max degree', maxdeg)
print('E[P 1_B]', pr(EPB, 75))
print('S02 value', pr(S02_EPB, 75))
print('diff vs S02', pr(EPB - S02_EPB, 5))
print('E[P] full space', pr(EPfull, 20))
print('Zlo = r^2 E[P 1_B]', pr(Zlo, 75))
print('Zlo - S02 Zlo', pr(Zlo - S02_ZLO, 5))
print('Zlo - Z_LO_RN3', pr(Zlo - ZLO_RN3, 8))
