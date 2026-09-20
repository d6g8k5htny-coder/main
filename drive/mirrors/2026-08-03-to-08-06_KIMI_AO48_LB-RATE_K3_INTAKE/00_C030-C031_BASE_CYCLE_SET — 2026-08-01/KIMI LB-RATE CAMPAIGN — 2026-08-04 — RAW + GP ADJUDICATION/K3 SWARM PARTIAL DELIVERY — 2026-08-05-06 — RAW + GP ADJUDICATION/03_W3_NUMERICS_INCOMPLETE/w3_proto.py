# W3 float prototype (NOT the certificate) - de-risking model, hot spot, anchors.
# Model: periodized Bargmann-Fock on T^2_24; at our scale (|s|<=6) K1(s)=exp(-s^2/2)
# up to wrap corrections ~3*exp(-162); prototype uses planar kernel only.
import mpmath as mp
from fractions import Fraction as Fr

mp.mp.prec = 200  # ~60 digits; Gram conditioning handled by rational preconditioner

# ---------------- kernel (planar) ----------------
def He(m, s):
    # He_m via recurrence: He_0=1, He_1=s, He_{m+1}=s He_m - m He_{m-1}
    if m == 0:
        return mp.mpf(1)
    if m == 1:
        return s
    a, b = mp.mpf(1), s
    for k in range(1, m):
        a, b = b, s * b - k * a
    return b

def K1d(m, s):
    # d^m/ds^m exp(-s^2/2) = (-1)^m He_m(s) exp(-s^2/2)
    return (-1) ** m * He(m, s) * mp.exp(-s * s / 2)

def K2deriv(a, b, s1, s2):
    # d^a/ds1^a d^b/ds2^b of K(s)=K1(s1)K1(s2)
    return K1d(a, s1) * K1d(b, s2)

# multi-indices for jet: (f, f1, f2, f11, f12, f22)
JET = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
# pins: at each of M,S,Y: (f, f1, f2)
PINFS = [(0, 0), (1, 0), (0, 1)]

def cov2(alpha, beta, x, y):
    # Cov(d^alpha f(x), d^beta f(y)) = (-1)^|beta| d^{alpha+beta} K(x-y)
    s1, s2 = x[0] - y[0], x[1] - y[1]
    return (-1) ** (beta[0] + beta[1]) * K2deriv(alpha[0] + beta[0], alpha[1] + beta[1], s1, s2)

# ---------------- geometry (exact rationals) ----------------

def mqf(q):
    # exact Fraction -> mpf
    return mp.mpf(q.numerator) / mp.mpf(q.denominator)

r_f = Fr(1, 40)
M = (Fr(-1, 80), Fr(0))
S = (Fr(1, 80), Fr(0))
Y = (Fr(-1, 80) + Fr(1, 40) * Fr(-19, 25), Fr(1, 40) * Fr(6, 25))
PINS = [M, S, Y]
b_lev = Fr(6, 5)
r_mp = mp.mpf(1) / 40
ell = r_mp ** 3 / 6
b_mp = mp.mpf(6) / 5

print("Y =", Y, "=", (mqf(Y[0]), mqf(Y[1])))
print("ell =", mp.nstr(ell, 25))

# ---------------- rational preconditioner ----------------
# target derivative functionals at 0 for the 9 pins:
MONOS = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2)]
# pin functionals in order: for P in PINS, for alpha in PINFS  -> L_i = d^alpha at P
LFUNS = [(P, a) for P in PINS for a in PINFS]

def fall(n, k):
    # falling factorial n*(n-1)*...*(n-k+1)
    out = Fr(1)
    for j in range(k):
        out *= (n - j)
    return out

def Lmono(L, mono):
    # L_i applied to monomial x^mono, exact rational
    (P, a) = L
    if a[0] > mono[0] or a[1] > mono[1]:
        return Fr(0)
    return (fall(mono[0], a[0]) * fall(mono[1], a[1])
            * P[0] ** (mono[0] - a[0]) * P[1] ** (mono[1] - a[1]))

Vmat = [[Lmono(L, m) for m in MONOS] for L in LFUNS]

def ratinv(A):
    n = len(A)
    Mx = [row[:] + [Fr(int(i == j)) for j in range(n)] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda i2: abs(Mx[i2][col]))
        ck = Mx[piv][col] != 0
        if not ck:
            raise SystemExit("singular rational matrix")
        Mx[col], Mx[piv] = Mx[piv], Mx[col]
        pv = Mx[col][col]
        Mx[col] = [v / pv for v in Mx[col]]
        for i2 in range(n):
            if i2 != col and Mx[i2][col] != 0:
                f = Mx[i2][col]
                Mx[i2] = [a - f * b2 for a, b2 in zip(Mx[i2], Mx[col])]
    return [row[n:] for row in Mx]

Crat = ratinv(Vmat)  # exact: T = Crat . L  gives d^alpha(0) exactly on polys deg<=3 in span
# verify V*C = I exactly
for i in range(9):
    for j in range(9):
        s = sum(Vmat[i][k] * Crat[k][j] for k in range(9))
        assert s == Fr(int(i == j)), (i, j, s)
print("rational preconditioner C=V^{-1} verified exactly (V*C=I)")
Cmp = [[mqf(c) for c in row] for row in Crat]

# ---------------- Gram and conditional law ----------------
def gram():
    G = mp.matrix(9, 9)
    for i, (Pi, ai) in enumerate(LFUNS):
        for j, (Pj, aj) in enumerate(LFUNS):
            G[i, j] = cov2(ai, aj, (mqf(Pi[0]), mqf(Pi[1])), (mqf(Pj[0]), mqf(Pj[1])))
    return G

G = gram()
detG = mp.det(G)
print("det G =", mp.nstr(detG, 12), " (expect ~1.3e-42)")

# Gamma = C G C^T  (well conditioned)
Gm = [[G[i, j] for j in range(9)] for i in range(9)]
Gamma = [[sum(Cmp[i][k] * Gm[k][l] * Cmp[j][l] for k in range(9) for l in range(9))
          for j in range(9)] for i in range(9)]
Gam = mp.matrix(Gamma)
print("cond(Gamma) ~", mp.nstr(mp.norm(Gam - mp.eye(9)*0, 2), 6), "; eig min/max:",
      mp.nstr(min(mp.eig(Gam)[0]), 8), mp.nstr(max(mp.eig(Gam)[0]), 8))
Gi = Gam ** -1

# pin values: v = (b,0,0, b-ell,0,0, mu_t,0,0); first mu_t from 8-pin subsystem
# 8 pins: all except f(Y) (index 6 of 9: Y is third pin, its f is index 6)
idx8 = [0, 1, 2, 3, 4, 5, 7, 8]  # drop index 6 = f(Y)
G8 = mp.matrix(8, 8)
for a, i in enumerate(idx8):
    for c, j in enumerate(idx8):
        G8[a, c] = G[i, j]
k8 = mp.matrix(8, 1)
for a, j in enumerate(idx8):
    (Pj, aj) = LFUNS[j]
    k8[a, 0] = cov2((0, 0), aj, (mqf(Y[0]), mqf(Y[1])), (mqf(Pj[0]), mqf(Pj[1])))
v8 = mp.matrix(8, 1)
v8[0, 0] = b_mp
v8[3, 0] = b_mp - ell
mu_t = (k8.T * (G8 ** -1) * v8)[0, 0]
print("mu_t = E[f(Y)|8 pins] =", mp.nstr(mu_t, 25), " b-mu_t =", mp.nstr(b_mp - mu_t, 12),
      " (b-mu_t)/ell =", mp.nstr((b_mp - mu_t) / ell, 12))

vfull = [b_mp, 0, 0, b_mp - ell, 0, 0, mu_t, 0, 0]
# transformed values tv = C v
tv = [sum(Cmp[i][k] * vfull[k] for k in range(9)) for i in range(9)]
lam = Gi * mp.matrix(tv)  # Gamma^{-1} C v

def jetlaw(y):
    # conditional law of J(y)=(f,f1,f2,f11,f12,f22) given 9 pins
    B = mp.matrix(6, 6)
    for i, ai in enumerate(JET):
        for j, aj in enumerate(JET):
            B[i, j] = cov2(ai, aj, y, y)
    A = mp.matrix(6, 9)
    for i, ai in enumerate(JET):
        for j, (Pj, aj) in enumerate(LFUNS):
            A[i, j] = cov2(ai, aj, y, (mqf(Pj[0]), mqf(Pj[1])))
    At = A * mp.matrix(Cmp).T          # A~ = A C^T (6x9)
    muJ = At * lam                      # 6x1
    SigJ = B - At * Gi * At.T
    return muJ, SigJ

# ---------------- per-point intensity ----------------
Q3 = mp.matrix([[0, 0, mp.mpf('0.5')], [0, -1, 0], [mp.mpf('0.5'), 0, 0]])

def rho_point(y, gfun):
    muJ, SigJ = jetlaw(y)
    muf = muJ[0]; mug = mp.matrix([muJ[1], muJ[2]]); muH = mp.matrix([muJ[3], muJ[4], muJ[5]])
    Sff = SigJ[0, 0]
    Sfg = mp.matrix([[SigJ[0, 1], SigJ[0, 2]]])
    Sgg = mp.matrix([[SigJ[1, 1], SigJ[1, 2]], [SigJ[2, 1], SigJ[2, 2]]])
    SfH = mp.matrix([[SigJ[0, 3], SigJ[0, 4], SigJ[0, 5]]])
    SgH = mp.matrix([[SigJ[1, 3], SigJ[1, 4], SigJ[1, 5]],
                     [SigJ[2, 3], SigJ[2, 4], SigJ[2, 5]]])
    SHH = mp.matrix([[SigJ[3, 3], SigJ[3, 4], SigJ[3, 5]],
                     [SigJ[4, 3], SigJ[4, 4], SigJ[4, 5]],
                     [SigJ[5, 3], SigJ[5, 4], SigJ[5, 5]]])
    detSg = mp.det(Sgg)
    pgrad = mp.exp(-(mug.T * (Sgg ** -1) * mug)[0, 0] / 2) / (2 * mp.pi * mp.sqrt(detSg))
    Sggi = Sgg ** -1
    # (f,H) | grad=0
    mu_f2 = muf - (Sfg * Sggi * mug)[0, 0]
    mu_H2 = muH - SgH.T * Sggi * mug
    S_ff2 = Sff - (Sfg * Sggi * Sfg.T)[0, 0]
    S_fH2 = SfH - Sfg * Sggi * SgH
    S_HH2 = SHH - SgH.T * Sggi * SgH
    # H | f=u, grad=0
    V = S_HH2 - S_fH2.T * S_fH2 / S_ff2
    beta = S_fH2.T / S_ff2  # dm/du
    # integrate u over window
    def integrand(u):
        m = mu_H2 + beta * (u - mu_f2)
        gu = gfun(m, V)
        return mp.exp(-(u - mu_f2) ** 2 / (2 * S_ff2)) / mp.sqrt(2 * mp.pi * S_ff2) * gu
    # quadrature over tiny window: midpoint + refinement
    a, bb = b_mp - ell, b_mp
    Iu = mp.quad(integrand, [a, bb])
    return pgrad * Iu, dict(pgrad=pgrad, muf2=mu_f2, Sff2=S_ff2, muH2=mu_H2, V=V, beta=beta)

# ---------------- g(u) = E[|detH| 1{det<0}], X=(a,c,d)~N(m,V), W=X'QX ----------------
def g_quad(m, V):
    # Gil-Pelaez: E|W| = (2/pi) int_0^inf (1-Re phi(t))/t^2 dt ; EW = m'Qm + tr(QV)
    VQ = V * Q3
    # D(t) = det(I - 2 i t VQ) = 1 + c1 t + c2 t^2 + c3 t^3 with
    # c1 = -2i tr(VQ), c2 = -4 e2(VQ), c3 = 8i det(VQ)
    trA = VQ[0, 0] + VQ[1, 1] + VQ[2, 2]
    e2A = (VQ[0, 0] * VQ[1, 1] - VQ[0, 1] * VQ[1, 0]
           + VQ[0, 0] * VQ[2, 2] - VQ[0, 2] * VQ[2, 0]
           + VQ[1, 1] * VQ[2, 2] - VQ[1, 2] * VQ[2, 1])
    detA = mp.det(VQ)
    Dcoef = [1, -2j * trA, -4 * e2A, 8j * detA]
    # phi(t) = D(t)^(-1/2) exp(i t N(t)/D(t)), N(t)/D(t) = m'Q(I-2itVQ)^{-1}m
    Qm = Q3 * m
    # adj(I+sB) = I + s(trB I - B) + s^2 adj(B), B = VQ, s = -2it
    trB = trA
    adjB = VQ*VQ - trB*VQ + e2A*mp.eye(3)
    E1 = trB*mp.eye(3) - VQ
    n0 = (m.T * Q3 * m)[0, 0]
    n1 = -2j * (m.T * Q3 * E1 * m)[0, 0]
    n2 = -4 * (m.T * Q3 * adjB * m)[0, 0]
    def Npoly(t):
        return n0 + n1 * t + n2 * t * t
    def Dpoly(t):
        return 1 + Dcoef[1] * t + Dcoef[2] * t * t + Dcoef[3] * t ** 3
    def phi(t):
        D = Dpoly(t)
        return D ** (-mp.mpf('0.5')) * mp.exp(1j * t * Npoly(t) / D)
    EW = (m.T * Q3 * m)[0, 0] + mp.matrix([VQ[i, i] for i in range(3)]).T * mp.ones(3, 1)
    EW = EW[0, 0] if hasattr(EW, '__getitem__') else EW
    EW = (m.T * Q3 * m)[0, 0] + sum(VQ[i, i] for i in range(3))
    def h(t):
        return (1 - mp.re(phi(t))) / t ** 2
    t0 = mp.mpf('1e-3')
    # small-t: series via moments is overkill for prototype; use quad from t0
    I1 = mp.quad(h, [0, mp.mpf('0.5'), 5, 50, mp.inf])
    Eabs = 2 / mp.pi * I1
    return (Eabs - EW) / 2

if __name__ == "__main__":
    # ANCHOR 1: far-field BF: H|f=u has mean -u I, Var(a)=Var(d)=2, Var(c)=1 independent
    m = mp.matrix([mp.mpf('-1.2'), 0, mp.mpf('-1.2')])
    V = mp.matrix([[2, 0, 0], [0, 1, 0], [0, 0, 2]])
    gfar = g_quad(m, V)
    print("ANCHOR far-field E[|detH|1{det<0}|f=1.2] =", mp.nstr(gfar, 15),
          " (C030 KR-MB: 1.41350 +/- 0.0017)")
