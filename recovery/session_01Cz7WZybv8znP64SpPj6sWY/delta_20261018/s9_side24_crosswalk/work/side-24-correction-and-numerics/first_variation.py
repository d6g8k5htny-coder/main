"""Independent first-variation check of the d=3 periodization correction, using ONLY the
parent P eq. (15.1)-(15.2) definition of the coefficient (not LS-DER-050/055 rotational
projection, not LS-DER-060 cone formula).

c = Gamma(7/6)/(24^(1/3) sqrt(pi)) * int_{S^2} p_G(0) p_Vu(0) tau_u^(4/3) D_u dsigma(u)

Covariance K(x)=k(x1)k(x2)k(x3), k normalized 1-D periodized Gaussian with even jets
m2=k''(0), m4=k''''(0), m6=k^(6)(0) (m0=1).  First nearest-image change per unit
q=exp(-L^2/2): dm_n = 2[He_n(L)-He_n(0)].

Directional first variation of log F(u) computed by central differences in the jets;
first variation of D_u by exact Gaussian linear response with closed-form reference
moments E0[h s^2], E0[h x^2] (h = det(A)^2 1{A<0}).  Then averaged over S^2 with
product Gauss-Legendre x trapezoid quadrature.
"""
import itertools
import mpmath as mp
import sympy as sp

mp.mp.dps = 60

# ---------- 1. closed-form reference cone moments (A = [[s+x,y],[y,s-x]]) ----------
w, z = sp.symbols('w z', positive=True)
g0 = sp.integrate((w - z) ** 2 * sp.exp(-z / 2) / 2, (z, 0, w))
g1 = sp.integrate((w - z) ** 2 * z * sp.exp(-z / 2) / 2, (z, 0, w))
s = sp.symbols('s', real=True)
vs = sp.Rational(5, 3)
phi_s = sp.exp(-s ** 2 / (2 * vs)) / sp.sqrt(2 * sp.pi * vs)
D0 = sp.simplify(sp.integrate(g0.subs(w, s ** 2) * phi_s, (s, -sp.oo, 0)))
Ess = sp.simplify(sp.integrate(s ** 2 * g0.subs(w, s ** 2) * phi_s, (s, -sp.oo, 0)))
ER2 = sp.simplify(sp.integrate(g1.subs(w, s ** 2) * phi_s, (s, -sp.oo, 0)))
Exx = sp.simplify(ER2 / 2)
print('g0(w) =', sp.simplify(g0))
print('g1(w) =', sp.simplify(g1))
print('D0  =', sp.nsimplify(D0), ' equals 29/6-sqrt6:', sp.simplify(D0 - (sp.Rational(29, 6) - sp.sqrt(6))) == 0)
print('E0[h s^2] =', sp.radsimp(Ess))
print('E0[h x^2] =', sp.radsimp(Exx))
print('homogeneity check  Ess/(5/3)+2Exx-7D0 =', sp.simplify(Ess / vs + 2 * Exx - 7 * D0))

D0n = mp.mpf(sp.N(D0, 70)); Essn = mp.mpf(sp.N(Ess, 70)); Exxn = mp.mpf(sp.N(Exx, 70))
G0 = [mp.mpf(5) / 3, mp.mpf(1), mp.mpf(1)]
Wd = [Essn / G0[0] ** 2 - D0n / G0[0], Exxn - D0n, Exxn - D0n]  # dD = (1/2) sum_i W_i dGamma_ii (+0 offdiag)


def dD_linear(dG):
    """first variation of D at Gamma0=diag(5/3,1,1) in (s,x,y) coords; off-diagonal moments vanish by x->-x, y->-y."""
    return sum(Wd[i] * dG[i][i] for i in range(3)) / 2


# ---------- 1b. direct quadrature check of dD_linear for a generic (non-diagonal) perturbation ----------
def D_direct(G):
    """E[(s^2-R^2)^2 1{s<-R}] for (s,x,y)~N(0,G), by inner closed-form s-integral and 2-D (R,theta) quadrature."""
    G = mp.matrix(G)
    Gxy = G[1:3, 1:3]
    Gxy_inv = Gxy ** -1
    gsx = mp.matrix([[G[0, 1], G[0, 2]]])
    beta = gsx * Gxy_inv
    v = G[0, 0] - (beta * gsx.T)[0, 0]
    detxy = mp.det(Gxy)

    def inner(R, th):
        x, y = R * mp.cos(th), R * mp.sin(th)
        mu = beta[0, 0] * x + beta[0, 1] * y
        sd = mp.sqrt(v)
        # integral_{-inf}^{-R} (s^2-R^2)^2 phi(s;mu,v) ds; substitute s = mu + sd*t, t < tb
        tb = (-R - mu) / sd
        # E[(s^2-R^2)^2 ; t<tb]: expand polynomial in t and use truncated normal moments
        Phi = mp.ncdf(tb); ph = mp.npdf(tb)
        M = [Phi]
        M.append(-ph)
        for n in range(2, 5):
            M.append((n - 1) * M[n - 2] - tb ** (n - 1) * ph)
        # (s^2-R^2) = mu^2-R^2 + 2 mu sd t + sd^2 t^2 = a0 + a1 t + a2 t^2
        a0, a1, a2 = mu ** 2 - R ** 2, 2 * mu * sd, sd ** 2
        c = [a0 * a0, 2 * a0 * a1, a1 * a1 + 2 * a0 * a2, 2 * a1 * a2, a2 * a2]
        val = sum(c[n] * M[n] for n in range(5))
        q = mp.matrix([[x], [y]])
        dens = mp.exp(-(q.T * Gxy_inv * q)[0, 0] / 2) / (2 * mp.pi * mp.sqrt(detxy))
        return val * dens * R

    return mp.quad(inner, [0, 2, 5, 12], [0, mp.pi / 2, mp.pi, 3 * mp.pi / 2, 2 * mp.pi])


mp.mp.dps = 25
B = [[mp.mpf('0.3'), mp.mpf('-0.2'), mp.mpf('0.15')],
     [mp.mpf('-0.2'), mp.mpf('0.5'), mp.mpf('0.1')],
     [mp.mpf('0.15'), mp.mpf('0.1'), mp.mpf('-0.4')]]
eps = mp.mpf('1e-4')
Gp = [[G0[i] * (i == j) + eps * B[i][j] for j in range(3)] for i in range(3)]
Gm = [[G0[i] * (i == j) - eps * B[i][j] for j in range(3)] for i in range(3)]
Dref = D_direct([[G0[i] * (i == j) for j in range(3)] for i in range(3)])
fd = (D_direct(Gp) - D_direct(Gm)) / (2 * eps)
print('D_direct(Gamma0) - (29/6-sqrt6) =', mp.nstr(Dref - D0n, 5))
print('dD finite-diff (direct quadrature) =', mp.nstr(fd, 15), '  linear-response formula =', mp.nstr(dD_linear(B), 15))
mp.mp.dps = 60

# ---------- 2. directional log-integrand as function of jets ----------
def tensor_entry(idx, m):
    cnt = [0, 0, 0]
    for i in idx:
        cnt[i] += 1
    out = mp.mpf(1)
    for cn in cnt:
        if cn % 2:
            return mp.mpf(0)
        out *= m[cn]
    return out


def contract(vecs, m):
    tot = mp.mpf(0)
    for idx in itertools.product(range(3), repeat=len(vecs)):
        e = tensor_entry(idx, m)
        if e == 0:
            continue
        p = e
        for v, i in zip(vecs, idx):
            p *= v[i]
        tot += p
    return tot


def frame(u):
    a = [mp.mpf(1), 0, 0] if abs(u[0]) < mp.mpf('0.6') else [0, mp.mpf(1), 0]
    dot = sum(a[i] * u[i] for i in range(3))
    e2 = [a[i] - dot * u[i] for i in range(3)]
    n2 = mp.sqrt(sum(x * x for x in e2)); e2 = [x / n2 for x in e2]
    e3 = [u[1] * e2[2] - u[2] * e2[1], u[2] * e2[0] - u[0] * e2[2], u[0] * e2[1] - u[1] * e2[0]]
    return e2, e3


def blocks(u, m):
    e2, e3 = frame(u)
    E = [[mp.mpf(1), 0, 0], [0, mp.mpf(1), 0], [0, 0, mp.mpf(1)]]
    SG = mp.matrix(3, 3)
    for i in range(3):
        for j in range(3):
            SG[i, j] = -contract([E[i], E[j]], m)
    vt = -contract([u] * 6, m)
    ct = mp.matrix(3, 1)
    for i in range(3):
        ct[i] = -contract([u, u, u, E[i]], m)
    tau2 = vt - (ct.T * SG ** -1 * ct)[0, 0]
    Vp = [(u, u), (u, e2), (u, e3)]
    Ap = [(e2, e2), (e2, e3), (e3, e3)]
    def cov(P, Q):
        Mx = mp.matrix(len(P), len(Q))
        for i, (a, b) in enumerate(P):
            for j, (c, d) in enumerate(Q):
                Mx[i, j] = contract([a, b, c, d], m)
        return Mx
    SVV = cov(Vp, Vp); SAV = cov(Ap, Vp); SAA = cov(Ap, Ap)
    GA = SAA - SAV * SVV ** -1 * SAV.T
    T = mp.matrix([[mp.mpf(1) / 2, 0, mp.mpf(1) / 2], [mp.mpf(1) / 2, 0, -mp.mpf(1) / 2], [0, 1, 0]])
    Gsxy = T * GA * T.T
    logpG = -mp.mpf(3) / 2 * mp.log(2 * mp.pi) - mp.log(mp.det(SG)) / 2
    logpV = -mp.mpf(3) / 2 * mp.log(2 * mp.pi) - mp.log(mp.det(SVV)) / 2
    return logpG, logpV, tau2, Gsxy


m0 = {0: mp.mpf(1), 2: mp.mpf(-1), 4: mp.mpf(3), 6: mp.mpf(-15)}
# reference sanity at an arbitrary direction
u_t = [mp.mpf('0.36'), mp.mpf('0.48'), mp.mpf('0.8')]
lg, lv, t2, Gs = blocks(u_t, m0)
print('reference tau^2 =', mp.nstr(t2, 20), ' Gamma_sxy diag =', [mp.nstr(Gs[i, i], 12) for i in range(3)],
      ' offdiag max =', mp.nstr(max(abs(Gs[i, j]) for i in range(3) for j in range(3) if i != j), 3),
      ' detSVV =', mp.nstr(mp.exp(-2 * (lv + mp.mpf(1.5) * mp.log(2 * mp.pi))), 15))

h = mp.mpf('1e-20')


def dlogF(u, j):
    mp_ = dict(m0); mm = dict(m0)
    mp_[j] += h; mm[j] -= h
    a = blocks(u, mp_); b = blocks(u, mm)
    dlg = (a[0] - b[0]) / (2 * h)
    dlv = (a[1] - b[1]) / (2 * h)
    dlt = (mp.log(a[2]) - mp.log(b[2])) / (2 * h)
    dG = (a[3] - b[3]) / (2 * h)
    dGl = [[dG[i, k] for k in range(3)] for i in range(3)]
    dlD = dD_linear(dGl) / D0n
    return dlg + dlv + mp.mpf(2) / 3 * dlt + dlD, (dlg, dlv, dlt, dlD)


def sphere_avg(j, nz, nth):
    xs, ws = mp.gauss_legendre(nz) if hasattr(mp, 'gauss_legendre') else (None, None)
    return xs


def gl_nodes(n):
    # Gauss-Legendre nodes on [-1,1] via mpmath polyroots of Legendre P_n
    xs = []
    for k in range(1, n + 1):
        x = mp.cos(mp.pi * (k - mp.mpf(1) / 4) / (n + mp.mpf(1) / 2))
        for _ in range(100):
            p0, p1 = mp.mpf(1), x
            for l in range(2, n + 1):
                p0, p1 = p1, ((2 * l - 1) * x * p1 - (l - 1) * p0) / l
            dp = n * (x * p1 - p0) / (x * x - 1)
            dx = p1 / dp
            x -= dx
            if abs(dx) < mp.mpf(10) ** (-mp.mp.dps + 5):
                break
        p0, p1 = mp.mpf(1), x
        for l in range(2, n + 1):
            p0, p1 = p1, ((2 * l - 1) * x * p1 - (l - 1) * p0) / l
        dp = n * (x * p1 - p0) / (x * x - 1)
        xs.append((x, 2 / ((1 - x * x) * dp * dp)))
    return xs


def response(j, nz, nth):
    tot = mp.mpf(0)
    for (zz, wz) in gl_nodes(nz):
        rr = mp.sqrt(1 - zz * zz)
        for k in range(nth):
            th = 2 * mp.pi * (k + mp.mpf(1) / 3) / nth
            u = [rr * mp.cos(th), rr * mp.sin(th), zz]
            val, _ = dlogF(u, j)
            tot += wz * (2 * mp.pi / nth) * val
    return tot / (4 * mp.pi)


import sys
nz, nth = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (8, 16)
r = {}
for j in (2, 4, 6):
    r[j] = response(j, nz, nth)
    print(f'direct S^2-averaged d log c / d m{j}  (nz={nz},nth={nth}) =', mp.nstr(r[j], 30), ' identify:', mp.identify(r[j]))

# ---------- 3. chain prediction (LS-DER-050 isotropic law + LS-DER-055 radialization) ----------
m2, m4, m6 = sp.symbols('m2 m4 m6')
M2 = -3 * m2; M4 = 3 * m4 + 6 * m2 ** 2; M6 = -(3 * m6 + 18 * m4 * m2 + 6 * m2 ** 3)
a_ = M2 / 3; m4d = 3 * M4 / 15; chi = 15 * M6 / 105
Om = a_ * chi - m4d ** 2
logc = sp.Rational(2, 3) * sp.log(Om) + sp.Rational(1, 2) * sp.log(m4d) - sp.Rational(13, 6) * sp.log(a_)
ref = {m2: -1, m4: 3, m6: -15}
pred = {2: sp.diff(logc, m2).subs(ref), 4: sp.diff(logc, m4).subs(ref), 6: sp.diff(logc, m6).subs(ref)}
print('chain-law predicted d log c/d m_j:', {k: sp.nsimplify(v) for k, v in pred.items()})
for j in (2, 4, 6):
    print(f'  m{j}: direct - predicted =', mp.nstr(r[j] - mp.mpf(sp.N(pred[j], 70)), 5))

Ls = sp.symbols('L')
He = {2: Ls ** 2 - 1, 4: Ls ** 4 - 6 * Ls ** 2 + 3, 6: Ls ** 6 - 15 * Ls ** 4 + 45 * Ls ** 2 - 15}
He0 = {2: -1, 4: 3, 6: -15}
dm = {j: sp.expand(2 * (He[j] - He0[j])) for j in (2, 4, 6)}
print('dm per unit q:', dm)
P_pred = sp.expand(sum(pred[j] * dm[j] for j in (2, 4, 6)))
P_chain = sp.expand(-Ls ** 2 * (10 * Ls ** 4 - 147 * Ls ** 2 + 315) / 105)
print('chain-law polynomial =', P_pred, ' equals LS-DER-055 (27):', sp.simplify(P_pred - P_chain) == 0)
# direct polynomial (rationalized coefficients)
rr_ = {j: sp.nsimplify(mp.nstr(r[j], 25), rational=True, tolerance=1e-20) for j in (2, 4, 6)}
P_dir = sp.expand(sum(rr_[j] * dm[j] for j in (2, 4, 6)))
print('direct polynomial (rationalized) =', P_dir, ' equals P_3(L):', sp.simplify(P_dir - P_chain) == 0)
P24 = sum(r[j] * mp.mpf(sp.N(dm[j].subs(Ls, 24), 70)) for j in (2, 4, 6))
print('direct first variation at L=24 per unit q =', mp.nstr(P24, 25), ' vs -620813376/35 =', mp.nstr(mp.mpf(-620813376) / 35, 25))
print('direct first-order relative correction at L=24 =', mp.nstr(P24 * mp.exp(-288), 20))
# term split at one direction for transparency
val, parts = dlogF(u_t, 6)
print('sample direction parts (m6): dlogpG,dlogpV,dlogtau2,dlogD =', [mp.nstr(p, 10) for p in parts])
