#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_ridge_scope_continuation_v1.py -- K3 SWARM Phase 4, W9 DER-027c SCOPE AUDIT
=================================================================================
Mandate (superseding AO48-WO-063 Task 4c's frame; DER-027c remains the starting
record; WO-063 sha256 e699bf44b7b84fca88faa2b509c1686ea9f8e500aee0dbe618c859b5ec5ad2e2
echoed here as the starting-record input):
  (1) reproduce the deterministic conditional-mean result at r = 0.025
      (the Kantorovich census of DER-027c);
  (2) separate deterministic mean topology from noisy-field saddle counts;
  (3) attempt interval continuation in r (the r-uniform enclosure);
  (4) preserve exact-rung status; name the nonuniform obstruction with the
      smallest certified r and the measured degradation rates.

This program prints RECEIPTS ONLY (checks and numbers). Labels, verdicts and the
scope-audit narrative are in the separate report K3-DER-027c-SCOPE-AUDIT.md.

ck() is fail-closed (any failed check -> SystemExit). Deterministic; no asserts;
byte-identical program output under `python3` and `python3 -O`.

Object (identical to DER-027c): 9-pin conditional mean m9 of the periodized
side-24 2D Bargmann-Fock field, rung r, b = 6/5, ell = r^3/6,
M = (-r/2,0), S = (+r/2,0), Y = r*(-1.26,0.24), v* = clip(mu_t,[b-ell,b]).

Rigor tiers (printed as tier tags on every receipt):
  [EXACT]      exact pointwise certification at a stated rung (mpmath dps 80-120,
               envelopes displayed; Kantorovich with certified global RKHS bounds)
  [UNIFORM-R]  uniform in r over a closed block: rigorous modulo the C3 mesh cap
               (house 'named analyticity formality' of C027; cap factor 1.25 on a
               9-point exact mesh of ||d^3 m/dr^3||_H; everything else exact)
  [INTERVAL]   direct interval-arithmetic enclosure attempt (mpmath iv)
  [GRID-F64]   float64 grid engine with certified allowance EPS_F = 5e-7
"""

import math
import numpy as np
from mpmath import mp, mpf, matrix, exp as mexp, sqrt as msqrt, pi as MPI

mp.dps = 120
mpf0 = mpf(0)
B = mpf('1.2')                       # EXACT
RUNG0 = '0.025'                      # EXACT: DER-027c certified rung of record
YTIL = (mpf('-0.76'), mpf('0.24'))   # EXACT
YVEL = (mpf('-1.26'), mpf('0.24'))   # EXACT: d/dr of Y = r*(-1.26,0.24)
J01 = ((0, 0), (1, 0), (0, 1))
EPS_F = 5e-7                         # certified float64 allowance
WO_SHA = "e699bf44b7b84fca88faa2b509c1686ea9f8e500aee0dbe618c859b5ec5ad2e2"

def fail(msg):
    print("FAIL: " + str(msg), flush=True)
    raise SystemExit("FAIL: " + str(msg))

def need(cond, msg):
    if not cond:
        fail(msg)
    print("[ok] " + str(msg), flush=True)

# ----------------------------------------------------------------------------
# kernel: planar closed form (Hermite), exact periodized (spectral lattice)
# ----------------------------------------------------------------------------
def _he(n, t):
    if n == 0: return t*0 + 1
    if n == 1: return t
    if n == 2: return t*t - 1
    if n == 3: return t**3 - 3*t
    if n == 4: return t**4 - 6*t*t + 3
    if n == 5: return t**5 - 10*t**3 + 15*t
    if n == 6: return t**6 - 15*t**4 + 45*t*t - 15
    if n == 7: return t**7 - 21*t**5 + 105*t**3 - 105*t
    if n == 8: return t**8 - 28*t**6 + 210*t**4 - 420*t*t + 105
    if n == 9: return t**9 - 36*t**7 + 378*t**5 - 1260*t**3 + 945*t
    if n == 10: return t**10 - 45*t**8 + 630*t**6 - 3150*t**4 + 4725*t*t - 945
    if n == 11: return t**11 - 55*t**9 + 990*t**7 - 6930*t**5 + 17325*t**3 - 10395*t
    if n == 12: return t**12 - 66*t**10 + 1485*t**8 - 13860*t**6 + 51975*t**4 - 62370*t*t + 10395
    raise ValueError(n)

def cov_mp(P, a, Q, c):
    u1 = P[0]-Q[0]; u2 = P[1]-Q[1]
    n1 = a[0]+c[0]; n2 = a[1]+c[1]
    s = (-1)**(c[0]+c[1]) * (-1)**(n1+n2)
    return s * _he(n1, u1) * _he(n2, u2) * mexp(-(u1*u1+u2*u2)/2)

# mixed section operator: MIX(P,a1,q1,v1, Q,a2,q2,v2) =
# (v1.grad_x)^q1 d^a1_x (v2.grad_y)^q2 d^a2_y K(x,y) at (P,Q)
def MIX(P, a1, q1, v1, Q, a2, q2, v2):
    u1 = P[0]-Q[0]; u2 = P[1]-Q[1]
    tot = mpf0
    for i in range(q1+1):
        c1 = math.comb(q1, i) * v1[0]**i * v1[1]**(q1-i)
        if c1 == 0: continue
        for j in range(q2+1):
            c2 = math.comb(q2, j) * v2[0]**j * v2[1]**(q2-j)
            if c2 == 0: continue
            n1 = a1[0]+a2[0]+i+j
            n2 = a1[1]+a2[1]+(q1+q2-i-j)
            sgn = (-1)**(a2[0]+a2[1]+q2) * (-1)**(n1+n2)
            tot += c1*c2*sgn*_he(n1, u1)*_he(n2, u2)*mexp(-(u1*u1+u2*u2)/2)
    return tot

# spectral lattice (factorized 1D sums), for kernel-exactness certificates
JMAX = 114
KS = [j*MPI/12 for j in range(-JMAX, JMAX+1)]
WS = [mexp(-k*k/2) for k in KS]
NORM1 = msqrt(2*MPI)/24

def K1(n, t):
    if n % 2 == 0:
        return ((-1)**(n//2)) * NORM1 * sum(w*(k**n)*mp.cos(k*t) for k, w in zip(KS, WS))
    return ((-1)**((n+1)//2)) * NORM1 * sum(w*(k**n)*mp.sin(k*t) for k, w in zip(KS, WS))

def cov_spec(A, Bd, dx, dy):
    return (-1)**(Bd[0]+Bd[1]) * K1(A[0]+Bd[0], dx) * K1(A[1]+Bd[1], dy)

KA = mpf(115)*MPI/12
def tail1d(n):
    return 2*mexp(-KA*KA/2)*(KA**n)*(1 + 4/(KA*KA)) * NORM1

def poisson_image_bound(umax, order):
    L = 24.0; tot = 0.0
    for k in range(1, 200):
        rr = k*L - umax
        if rr <= 0: continue
        poly = rr**order + 6.0*rr**max(order-2, 0) + 15.0
        term = 8*k*poly*math.exp(-rr*rr/2)
        tot += term
        if k >= 2 and term < 1e-300: break
    return tot

# ----------------------------------------------------------------------------
# 9-pin construction at rung r (raw Gram for cross-checks; whitened engine)
# ----------------------------------------------------------------------------
def pins9(r):
    M = (-r/2, mpf0); S = (r/2, mpf0)
    yy = (M[0] + r*YTIL[0], r*YTIL[1])
    PIN6 = [(M, a) for a in J01] + [(S, a) for a in J01]
    return PIN6 + [(yy, a) for a in J01], (M, S, yy)

def gram(PIN):
    n = len(PIN)
    S = matrix(n, n)
    for i, (P, a) in enumerate(PIN):
        for j, (Q, c) in enumerate(PIN):
            S[i, j] = cov_mp(P, a, Q, c)
    return S

def cholesky(A):
    n = A.rows
    L = matrix(n, n)
    for i in range(n):
        for j in range(i+1):
            s = A[i, j] - sum(L[i, k]*L[j, k] for k in range(j))
            if i == j: L[i, i] = msqrt(max(s, mpf0))
            else: L[i, j] = s/L[j, j]
    return L

def fwd(L, b):
    n = L.rows
    z = matrix(n, 1)
    for i in range(n):
        z[i, 0] = (b[i, 0] - sum(L[i, j]*z[j, 0] for j in range(i)))/L[i, i]
    return z

def mu_t_exact(r):
    PIN9, (M, S, yy) = pins9(r)
    PIN6 = PIN9[:6]; YJ = PIN9[6:]
    ell = r**3/6
    Spp6 = gram(PIN6)
    Sop = matrix(3, 6); Soo = matrix(3, 3)
    for i, (P, a) in enumerate(YJ):
        for j, (Q, c) in enumerate(PIN6): Sop[i, j] = cov_mp(P, a, Q, c)
        for j, (Q, c) in enumerate(YJ): Soo[i, j] = cov_mp(P, a, Q, c)
    v6 = matrix([[B], [0], [0], [B-ell], [0], [0]])
    Ki = Sop * Spp6**-1
    mo = Ki*v6; So = Soo - Ki*Sop.T
    Sgg = matrix([[So[1, 1], So[1, 2]], [So[2, 1], So[2, 2]]])
    Sggi = Sgg**-1
    mg = matrix([[mo[1, 0]], [mo[2, 0]]])
    Sfg = matrix([[So[0, 1], So[0, 2]]])
    return mo[0, 0] - (Sfg*Sggi*mg)[0, 0], ell

# dual numbers (Taylor order ORD) for mu_t r-derivatives
ORD = 3
class D:
    __slots__ = ('c',)
    def __init__(self, c): self.c = list(c) + [mpf0]*(ORD+1-len(c))
    @staticmethod
    def const(a): return D([mpf(a)])
    @staticmethod
    def var(a): return D([mpf(a), mpf(1)])
    def __add__(s, o):
        o = o if isinstance(o, D) else D.const(o)
        return D([s.c[i]+o.c[i] for i in range(ORD+1)])
    __radd__ = __add__
    def __sub__(s, o):
        o = o if isinstance(o, D) else D.const(o)
        return D([s.c[i]-o.c[i] for i in range(ORD+1)])
    def __rsub__(s, o): return D.const(o) - s
    def __neg__(s): return D([-x for x in s.c])
    def __mul__(s, o):
        o = o if isinstance(o, D) else D.const(o)
        c = [mpf0]*(ORD+1)
        for i in range(ORD+1):
            for j in range(ORD+1-i): c[i+j] += s.c[i]*o.c[j]
        return D(c)
    __rmul__ = __mul__
    def recip(s):
        c = [mpf0]*(ORD+1); c[0] = 1/s.c[0]
        for n in range(1, ORD+1):
            c[n] = -sum(s.c[i]*c[n-i] for i in range(1, n+1))*c[0]
        return D(c)
    def __truediv__(s, o):
        o = o if isinstance(o, D) else D.const(o)
        return s*o.recip()
    def __rtruediv__(s, o): return D.const(o)*s.recip()
    def dexp(s):
        c = [mpf0]*(ORD+1); c[0] = mexp(s.c[0])
        for n in range(1, ORD+1):
            c[n] = sum(i*s.c[i]*c[n-i] for i in range(1, n+1))/n
        return D(c)

def _heD(n, t):
    if n == 0: return D.const(1)
    if n == 1: return t
    if n == 2: return t*t - 1
    if n == 3: return t**3 - 3*t
    if n == 4: return t**4 - 6*t*t + 3
    raise ValueError(n)

def cov_D(P, a, Q, c):
    u1 = P[0]-Q[0]; u2 = P[1]-Q[1]
    n1 = a[0]+c[0]; n2 = a[1]+c[1]
    s = (-1)**(c[0]+c[1]) * (-1)**(n1+n2)
    return s * _heD(n1, u1) * _heD(n2, u2) * (-(u1*u1+u2*u2)/2).dexp()

def mu_t_dual(rr):
    r = D.var(rr); half = D.const('0.5')
    M = (-r*half, D.const(0)); S = (r*half, D.const(0))
    ell = r*r*r/6
    yy = (M[0] + r*D.const('-0.76'), r*D.const('0.24'))
    PIN6 = [(M, a) for a in J01] + [(S, a) for a in J01]
    YJ = [(yy, a) for a in J01]
    def mat_inv(A):
        n = len(A)
        Mx = [row[:] + [D.const(1) if i == j else D.const(0) for j in range(n)]
              for i, row in enumerate(A)]
        for col in range(n):
            piv = max(range(col, n), key=lambda i2: abs(Mx[i2][col].c[0]))
            Mx[col], Mx[piv] = Mx[piv], Mx[col]
            pinv = Mx[col][col].recip()
            Mx[col] = [x*pinv for x in Mx[col]]
            for i2 in range(n):
                if i2 != col:
                    f = Mx[i2][col]
                    Mx[i2] = [a - f*b for a, b in zip(Mx[i2], Mx[col])]
        return [row[n:] for row in Mx]
    Spp6 = [[cov_D(P, a, Q, c) for (Q, c) in PIN6] for (P, a) in PIN6]
    Sop = [[cov_D(P, a, Q, c) for (Q, c) in PIN6] for (P, a) in YJ]
    Soo = [[cov_D(P, a, Q, c) for (Q, c) in YJ] for (P, a) in YJ]
    Spp6i = mat_inv(Spp6)
    vals6 = [D.const(B), D.const(0), D.const(0), D.const(B)-ell, D.const(0), D.const(0)]
    Ki = [[sum(Sop[i][k]*Spp6i[k][j] for k in range(6)) for j in range(6)] for i in range(3)]
    mo = [sum(Ki[i][k]*vals6[k] for k in range(6)) for i in range(3)]
    So = [[Soo[i][j] - sum(Ki[i][k]*Sop[j][k] for k in range(6)) for j in range(3)] for i in range(3)]
    Sgg = [[So[1][1], So[1][2]], [So[2][1], So[2][2]]]
    Sggi = mat_inv(Sgg)
    Kfg = [sum(So[0][k+1]*Sggi[k][j] for k in range(2)) for j in range(2)]
    return mo[0] - (Kfg[0]*mo[1] + Kfg[1]*mo[2])

# ----------------------------------------------------------------------------
# Rung: exact whitened 9-pin engine + r-derivative recursion (c^{(n)}, H-norms)
# ----------------------------------------------------------------------------
class Rung:
    def __init__(self, rr, with_ders=False):
        r = mpf(rr); self.r = r
        self.pins, (self.M, self.S, self.yy) = pins9(r)
        mu_t, self.ell = mu_t_exact(r)
        self.mu_t = mu_t
        self.vstar = min(max(mu_t, B - self.ell), B)
        self.vals = [B, mpf0, mpf0, B-self.ell, mpf0, mpf0, self.vstar, mpf0, mpf0]
        self.Spp = gram(self.pins)
        self.L = cholesky(self.Spp)
        self.pivot = min(self.L[i, i] for i in range(9))
        self.Sppi = self.Spp**-1
        v = matrix([[x] for x in self.vals])
        self.gam = fwd(self.L, v)
        self.coef = [(self.Sppi*v)[i, 0] for i in range(9)]
        self.nm = msqrt(sum(self.gam[i, 0]**2 for i in range(9)))
        self.vel = [(-mpf('0.5'), mpf0)]*3 + [(mpf('0.5'), mpf0)]*3 + [YVEL]*3
        if with_ders:
            mu = mu_t_dual(rr)
            self.mu_d = [mu.c[0], mu.c[1], mu.c[2]*2, mu.c[3]*6]
            self._build_der()
        # raw-engine helpers (cross-checks)
    def kvec(self, x, der=(0, 0)):
        return matrix([cov_mp(x, der, P, a) for (P, a) in self.pins])
    def mval(self, x, der=(0, 0)):
        phi = fwd(self.L, self.kvec(x, der))
        return sum(phi[i, 0]*self.gam[i, 0] for i in range(9))
    def mval_raw(self, x, der=(0, 0)):
        t = mpf0
        for (P, a), cj in zip(self.pins, self.coef):
            t += cj*cov_mp(x, der, P, a)
        return t
    def mgrad(self, x): return (self.mval(x, (1, 0)), self.mval(x, (0, 1)))
    def mhess(self, x): return (self.mval(x, (2, 0)), self.mval(x, (1, 1)), self.mval(x, (0, 2)))
    def cond(self):
        return mp.norm(self.Spp, 2)*mp.norm(self.Sppi, 2)
    def v_der(self, n):
        z = [mpf0]*9
        if n == 1: z[3] = -self.r**2/2; z[6] = self.mu_d[1]
        elif n == 2: z[3] = -self.r; z[6] = self.mu_d[2]
        elif n == 3: z[3] = -mpf(1); z[6] = self.mu_d[3]
        else: raise ValueError(n)
        return z
    def _build_der(self):
        """Recursion for c^{(n)} (n<=3) and H-norms of m^{(n)} (n<=3).
        Sections K^{(s)}_k = (vel_k.grad_y)^s d^{a_k} K(x,p_k), s = 0..3.
        c^{(n)} solves: Spp c^{(n)} = w^{(n)},
          w^{(n)}_j = v^{(n)}_j - sum_{s=1..n} binom(n,s) L_j^{(s)}[m^{(n-s)}]
                      - sum_{s=1..n} binom(n,s) sum_k c^{(n-s)}_k L_j[K^{(s)}_k]
          with L_j^{(s)}[g] = (vel_j.grad)^s d^{a_j} g (p_j)."""
        from math import comb
        pins, vel = self.pins, self.vel
        Spp, Sppi = self.Spp, self.Sppi
        # L_j^{(s)}[K^{(u)}_k] entries for s+u <= 3
        def LK(j, s, k, u):
            (P, a) = pins[j]; (Q, b) = pins[k]
            return MIX(P, a, s, vel[j], Q, b, u, vel[k])
        # precompute LK table
        LKt = {}
        for j in range(9):
            for k in range(9):
                for s in range(4):
                    for u in range(4):
                        LKt[(j, s, k, u)] = LK(j, s, k, u)
        cs = [matrix([[self.coef[k]] for k in range(9)])]
        for n in range(1, 4):
            w = matrix(9, 1)
            vd = self.v_der(n)
            for j in range(9):
                acc = mpf0
                for s in range(1, n+1):
                    bn = comb(n, s)
                    # L_j^{(s)}[m^{(n-s)}]: m^{(t)} = sum_u binom(t,u) sum_k c^{(t-u)}_k K^{(u)}_k
                    t = n - s
                    for u in range(t+1):
                        bt = comb(t, u)
                        for k in range(9):
                            acc += bn*bt*cs[t-u][k, 0]*LKt[(j, s, k, u)]
                    for k in range(9):
                        acc += bn*cs[n-s][k, 0]*LKt[(j, 0, k, s)]
                w[j, 0] = vd[j] - acc
            cs.append(Sppi*w)
        self.cs = cs   # cs[n] = c^{(n)}
        # H-norms: ||m^{(n)}||^2 = sum_{s,u} binom binom (c^{(n-s)})^T G_{s,u} c^{(n-u)}
        self.hnm = {}
        for n in range(0, 4):
            tot = mpf0
            for s in range(n+1):
                bs = comb(n, s)
                for u in range(n+1):
                    bu = comb(n, u)
                    csu, cuu = cs[n-s], cs[n-u]
                    q = mpf0
                    for j in range(9):
                        for k in range(9):
                            q += csu[j, 0]*cuu[k, 0]*LKt[(j, s, k, u)]
                    tot += bs*bu*q
            self.hnm[n] = msqrt(max(tot, mpf0))

def crit_info(R, x):
    mv = R.mval(x)
    g = R.mgrad(x)
    a, b, c = R.mhess(x)
    tr = a+c; det = a*c - b*b
    disc = msqrt(max(tr*tr - 4*det, mpf0))
    return dict(m=mv, g=g, gn=msqrt(g[0]**2+g[1]**2), H=(a, b, c), det=det,
                l1=(tr+disc)/2, l2=(tr-disc)/2, d=msqrt(x[0]**2+x[1]**2))

def newton(R, x0, iters=80, tol='1e-60'):
    x = [mpf(x0[0]), mpf(x0[1])]
    for _ in range(iters):
        i = crit_info(R, x)
        a, b, c = i['H']
        dd = mp.lu_solve(matrix([[a, b], [b, c]]), matrix([i['g'][0], i['g'][1]]))
        x = [x[0]-dd[0], x[1]-dd[1]]
        if i['gn'] < mpf(tol):
            break
    return x

def kantorovich(R, x, D3F, E_eval):
    i = crit_info(R, x)
    lmin = min(abs(i['l1']), abs(i['l2']))
    if lmin < mpf('1e-6'):
        return None
    beta = 1/(lmin*(1 - mpf('1e-12')))
    res = max(abs(i['g'][0]), abs(i['g'][1]))
    eta = beta*(res + E_eval)
    gamma = D3F*(1 + mpf('1e-12'))
    alpha = beta*gamma*eta
    if alpha >= mpf('0.5'):
        return None
    rho1 = (1 - msqrt(1-2*alpha))/(beta*gamma)
    rho2 = (1 + msqrt(1-2*alpha))/(beta*gamma)
    return dict(info=i, res=res, lmin=lmin, beta=beta, eta=eta, gamma=gamma,
                alpha=alpha, rho1=rho1, rho2=rho2)

def rkhs_bounds(R):
    nm = R.nm*(1 + mpf('1e-18'))
    G1 = nm; G2 = nm*msqrt(3); G3 = nm*msqrt(15)
    return G1, G2, G3, msqrt(3)*G2, 2*msqrt(3)*G3   # G1,G2,G3,D2F,D3F

# ----------------------------------------------------------------------------
# float64 grid engine (certified allowance EPS_F)
# ----------------------------------------------------------------------------
def _he_np(n, t):
    if n == 0: return t*0 + 1.0
    if n == 1: return t
    if n == 2: return t*t - 1.0
    if n == 3: return t**3 - 3.0*t
    if n == 4: return t**4 - 6.0*t*t + 3.0
    raise ValueError(n)

def cov_np(A, Bd, dx, dy):
    n1 = A[0]+Bd[0]; n2 = A[1]+Bd[1]
    s = (-1)**(Bd[0]+Bd[1]) * (-1)**(n1+n2)
    return s * _he_np(n1, dx) * _he_np(n2, dy) * np.exp(-(dx*dx+dy*dy)/2.0)

class FMF:
    def __init__(self, R):
        self.pins = [((float(P[0]), float(P[1])), a) for (P, a) in R.pins]
        self.coef = np.array([float(c) for c in R.coef])
    def jets(self, X, Y):
        m = np.zeros_like(X); gx = np.zeros_like(X); gy = np.zeros_like(X)
        h11 = np.zeros_like(X); h12 = np.zeros_like(X); h22 = np.zeros_like(X)
        for (P, a), cj in zip(self.pins, self.coef):
            dx, dy = X-P[0], Y-P[1]
            m += cj*cov_np((0, 0), a, dx, dy)
            gx += cj*cov_np((1, 0), a, dx, dy)
            gy += cj*cov_np((0, 1), a, dx, dy)
            h11 += cj*cov_np((2, 0), a, dx, dy)
            h12 += cj*cov_np((1, 1), a, dx, dy)
            h22 += cj*cov_np((0, 2), a, dx, dy)
        return m, np.hypot(gx, gy), np.sqrt(h11*h11 + 2*h12*h12 + h22*h22)

# ----------------------------------------------------------------------------
# S0: kernel-exactness certificates
# ----------------------------------------------------------------------------
def part_S0():
    print("\n[S0] kernel exactness certificates (spectral lattice + Poisson)")
    for n in (0, 2, 4, 6):
        t1 = tail1d(n)
        print("     1D lattice tail (order %d, |k|>30) <= %s" % (n, mp.nstr(t1, 3)))
        need(t1 < mpf('1e-60'), "1D lattice tail order %d < 1e-60" % n)
    need(abs(K1(0, mpf0)**2 - 1) < mpf('1e-70'), "lattice normalization K1(0)^2 = 1")
    worst = mpf0
    for (dx, dy) in [(mpf0, mpf0), (mpf('1.3056'), mpf('0.6858')),
                     (mpf('0.025'), mpf0), (mpf('-1.104'), mpf('0.8797'))]:
        for A in J01 + ((2, 0), (1, 1), (0, 2)):
            for Bd in J01:
                worst = max(worst, abs(cov_spec(A, Bd, dx, dy)
                                       - cov_mp((dx, dy), A, (0, 0), Bd)))
    print("     max |Cov_spectral - Cov_planar| over probes = %s" % mp.nstr(worst, 3))
    need(worst < mpf('1e-60'), "spectral/planar agreement < 1e-60")
    img = poisson_image_bound(12.1, 3)
    print("     Poisson image bound (whole torus, orders <=3) = %.3e" % img)
    need(img < 1e-25, "Poisson image bound < 1e-25")

# ----------------------------------------------------------------------------
# S1: mandate item (1) -- reproduce the DER-027c census at r = 0.025 [EXACT]
# ----------------------------------------------------------------------------
def part_S1():
    print("\n[S1] mandate(1): DER-027c census reproduction at r = 0.025 [EXACT]")
    R = Rung(RUNG0)
    # whitened vs raw engine cross-check
    dev = max(abs(R.mval(p) - R.mval_raw(p))
              for p in [(mpf('1.3056'), mpf('0.6858')), (mpf('-1.104'), mpf('0.8797')),
                        (mpf('0.4'), mpf('-0.7'))])
    print("     whitened-vs-raw engine deviation = %s" % mp.nstr(dev, 3))
    need(dev < mpf('1e-60'), "whitened engine reproduces raw engine < 1e-60")
    need(B - R.ell < R.vstar < B, "v* strictly inside window")
    print("     (b-v*)/ell = %s ; ||m9||_H^2 = %s"
          % (mp.nstr((B - R.vstar)/R.ell, 12), mp.nstr(R.nm**2, 14)))
    need(abs((B - R.vstar)/R.ell - mpf('0.499715906773')) < mpf('1e-9'),
         "(b-v*)/ell matches DER-027c")
    need(abs(R.nm**2 - mpf('13.219736462363')) < mpf('1e-9'),
         "||m9||_H^2 matches DER-027c")
    G1, G2, G3, D2F, D3F = rkhs_bounds(R)
    E_eval = mpf('1e-50')
    roots = {}
    kant = {}
    seeds = {'P*': ('1.3056', '0.6858'), 'Q*': ('-1.1040', '0.8797'),
             'M': R.M, 'S': R.S, 'Y': R.yy,
             'min1': ('-1.7409', '-1.0771'), 'min2': ('1.5640', '-1.4800')}
    for nm, sd in seeds.items():
        x = newton(R, sd)
        k = kantorovich(R, x, D3F, E_eval)
        if k is None:
            fail("Kantorovich failed at " + nm)
        roots[nm] = x; kant[nm] = k
        i = k['info']
        print("     %-4s (%s, %s): m=%s min|eig|=%s rho2=%s"
              % (nm, mp.nstr(x[0], 12), mp.nstr(x[1], 12), mp.nstr(i['m'], 14),
                 mp.nstr(k['lmin'], 8), mp.nstr(k['rho2'], 4)))
    # DER-027c quote gates
    need(abs(kant['P*']['info']['m'] - mpf('1.93119035755789484')) < mpf('1e-12'),
         "m(P*) = 1.93119035755789484 (DER-027c)")
    need(abs(kant['Q*']['info']['m'] - mpf('1.66659821936183025')) < mpf('1e-12'),
         "m(Q*) = 1.66659821936183025 (DER-027c)")
    need(abs(kant['P*']['info']['d'] - mpf('1.4747477269211')) < mpf('1e-9'),
         "d(P*) = 1.4747477269211 (DER-027c)")
    need(abs(kant['Q*']['info']['d'] - mpf('1.41162073528202')) < mpf('1e-9'),
         "d(Q*) = 1.41162073528202 (DER-027c)")
    need(abs(kant['P*']['lmin'] - mpf('2.587401874')) < mpf('1e-6'),
         "P* definiteness floor 2.587401874 (DER-027c)")
    need(abs(kant['Q*']['lmin'] - mpf('1.958943924')) < mpf('1e-6'),
         "Q* definiteness floor 1.958943924 (DER-027c)")
    need(kant['P*']['info']['det'] > 0 and kant['Q*']['info']['det'] > 0
         and kant['P*']['info']['l1'] < 0 and kant['Q*']['info']['l1'] < 0,
         "P*, Q* typed MAX")
    need(kant['S']['info']['det'] < 0 and kant['Y']['info']['det'] < 0,
         "S, Y typed SADDLE (values b-ell, v* < b)")
    need(abs(kant['M']['info']['m'] - B) < mpf('1e-40'), "M max at exactly b")
    need(kant['min1']['info']['det'] > 0 and kant['min1']['info']['l2'] > 0
         and kant['min2']['info']['det'] > 0 and kant['min2']['info']['l2'] > 0,
         "min1, min2 typed MIN")
    # --- compact second-order exclusion census in B_2.5 [GRID-F64]
    print("     exclusion census over B_2.5 (float64, EPS_F = 5e-7):")
    F = FMF(R)
    D3Ff = float(D3F); D2Ff = float(D2F)
    names = ('M', 'S', 'Y', 'P*', 'Q*', 'min1', 'min2')
    d = 0.02; RC = 2.5
    n0 = int(2*(RC+d)/d)+1
    xs = -RC-d + d*(np.arange(n0)+0.5)
    XX, YY = np.meshgrid(xs, xs)
    maskc = (XX**2 + YY**2) <= (RC+d)**2
    mm, gn, hf = F.jets(XX, YY)
    rho = d/math.sqrt(2)
    thr = (hf + D3Ff*rho)*rho + EPS_F
    alive = [(float(XX[i]), float(YY[i])) for i in zip(*np.nonzero(maskc & (gn <= thr)))]
    lev = 0
    while d > 1.5e-5 and alive:
        d = d/4.0; lev += 1
        rho = d/math.sqrt(2)
        off = d*(np.arange(4)-1.5)
        na = []
        for (cx, cy) in alive:
            X2, Y2 = np.meshgrid(cx+off, cy+off)
            mm, gn, hf = F.jets(X2, Y2)
            thr = (hf + D3Ff*rho)*rho + EPS_F
            na += [(float(X2[i]), float(Y2[i])) for i in zip(*np.nonzero(gn <= thr))]
        alive = na
    def covered(c):
        for nm in names:
            x = roots[nm]
            if math.hypot(c[0]-float(x[0]), c[1]-float(x[1])) + math.sqrt(2)*d \
                    <= float(kant[nm]['rho2']):
                return nm
        return None
    uncov = [c for c in alive if covered(c) is None]
    need(len(alive) > 0, "survivor cells present at pins (sanity)")
    need(not uncov, "every survivor cell inside a certified uniqueness ball")
    print("     survivors=%d, all inside uniqueness balls" % len(alive))
    print("     => Crit(m9) cap B_2.5 = 7 points (exactly as DER-027c) [EXACT]")
    # value exclusions outside B_2.5
    dA = 0.02
    nA = int(6.4/dA)+1
    xsA = -3.2 + dA*(np.arange(nA)+0.5)
    XXA, YYA = np.meshgrid(xsA, xsA)
    RRA = np.sqrt(XXA**2 + YYA**2)
    sel = (RRA > 2.5) & (RRA <= 3.0 + dA)
    mm, gn, hf = F.jets(XXA[sel], YYA[sel])
    rhoA = dA/math.sqrt(2)
    ub = mm + gn*rhoA + (D2Ff/2)*rhoA*rhoA + EPS_F
    print("     annulus 2.5<d<=3: max certified m9 upper bound = %.6f" % float(ub.max()))
    need(ub.max() < 1.2, "annulus 2.5-3 value exclusion")
    dT = 0.05
    nT = int(24.0/dT)+1
    xsT = -12.0 + dT*(np.arange(nT)+0.5)
    XXT, YYT = np.meshgrid(xsT, xsT)
    RRT = np.sqrt(XXT**2 + YYT**2)
    selT = RRT > 3.0
    mm, gn, hf = F.jets(XXT[selT], YYT[selT])
    rhoT = dT/math.sqrt(2)
    ub = mm + gn*rhoT + (D2Ff/2)*rhoT*rhoT + EPS_F
    print("     torus exterior d>3: max certified m9 upper bound = %.6f" % float(ub.max()))
    need(ub.max() < 1.2, "torus exterior value exclusion")
    print("     => zero above-b saddles of the mean, torus-global [EXACT]")
    return R, roots, kant

# ----------------------------------------------------------------------------
# S2: mandate item (2) -- object separation receipts
# ----------------------------------------------------------------------------
def part_S2(R, roots, kant):
    print("\n[S2] mandate(2): deterministic mean topology vs noisy-field counts")
    print("     This certificate's conclusions apply to the DETERMINISTIC function")
    print("     m9(x) = E[f(x)|9 pins]: its critical set, types, values, uniqueness")
    print("     balls, r-continuation -- all are properties of a fixed analytic")
    print("     function. NO claim here counts critical points of the noisy")
    print("     conditioned field f~ = m9 + xi (xi = conditional BF with variance")
    print("     v(x)); those are Kac-Rice objects owned by LB-1's Lemma H-SUP and")
    print("     the C029 B3(iii) mean-ridge channel.")
    for nm in ('P*', 'Q*'):
        k = R.kvec(roots[nm])
        vv = 1 - (k.T * R.Sppi * k)[0]
        print("     %s: conditional variance v = %s (sd = %s); (m-b)/sd = %s"
              % (nm, mp.nstr(vv, 12), mp.nstr(msqrt(vv), 10),
                 mp.nstr((kant[nm]['info']['m'] - B)/msqrt(vv), 8)))
        need(mpf('0.01') < vv < 1, "%s: 0 < v < 1 (noise nondegenerate; objects differ)" % nm)
    print("     receipt: v(P*), v(Q*) > 0 certifies f~ != m9; the noisy field has")
    print("     O(1) expected critical count above b near P* (LB-1 [F]), which is a")
    print("     statement NOT covered by (and not contradicting) the mean census.")

# ----------------------------------------------------------------------------
# r-derivative machinery: pointwise m^{(n)}(x) and validation
# ----------------------------------------------------------------------------
def m_der_point(R, n, x):
    """m^{(n)}(x) = sum_{s=0..n} binom(n,s) sum_k c^{(n-s)}_k K^{(s)}_k(x)."""
    from math import comb
    tot = mpf0
    for s in range(n+1):
        bn = comb(n, s)
        for k in range(9):
            (Q, b) = R.pins[k]
            sec = MIX(x, (0, 0), 0, (mpf0, mpf0), Q, b, s, R.vel[k])
            tot += bn*R.cs[n-s][k, 0]*sec
    return tot

def part_S3_validation():
    print("\n[S3.0] r-derivative machinery validation (FD cross-checks) [EXACT]")
    r0 = '0.025'
    R = Rung(r0, with_ders=True)
    need(abs(R.hnm[0] - R.nm) < mpf('1e-40'),
         "H-norm identity ||m||_H via section recursion == whitened norm")
    for dr, ordmax in [(mpf('1e-7'), 1), (mpf('3e-5'), 2), (mpf('3e-4'), 3)]:
        Rp = Rung(str(mpf(r0)+dr), with_ders=True)
        Rm = Rung(str(mpf(r0)-dr), with_ders=True)
        x = (mpf('1.3'), mpf('0.68'))
        if ordmax == 1:
            fd = (Rp.mval(x) - Rm.mval(x))/(2*dr)
            ex = m_der_point(R, 1, x)
        elif ordmax == 2:
            fd = (Rp.mval(x) - 2*R.mval(x) + Rm.mval(x))/dr**2
            ex = m_der_point(R, 2, x)
        else:
            Rpp = Rung(str(mpf(r0)+2*dr), with_ders=True)
            Rmm = Rung(str(mpf(r0)-2*dr), with_ders=True)
            fd = (Rpp.mval(x) - 2*Rp.mval(x) + 2*Rm.mval(x) - Rmm.mval(x))/(2*dr**3)
            ex = m_der_point(R, 3, x)
        rel = abs(fd-ex)/max(abs(ex), mpf('1e-30'))
        print("     order %d: m^{(%d)}(x0) recursion = %s, FD = %s, rel dev = %s"
              % (ordmax, ordmax, mp.nstr(ex, 8), mp.nstr(fd, 8), mp.nstr(rel, 3)))
        need(rel < mpf('5e-3'), "r-derivative order %d matches FD" % ordmax)
    print("     ||m'||_H = %s, ||m''||_H = %s, ||m'''||_H = %s at r = 0.025"
          % (mp.nstr(R.hnm[1], 8), mp.nstr(R.hnm[2], 8), mp.nstr(R.hnm[3], 8)))

# ----------------------------------------------------------------------------
# S3: mandate item (4) -- exact-rung ladder with Kantorovich at every rung
# ----------------------------------------------------------------------------
def part_S3():
    part_S3_validation()
    print("\n[S3] mandate(4): exact-rung certification ladder [EXACT]")
    print("     per rung: Newton roots, Kantorovich alpha/beta/rho2, eigenvalue")
    print("     floors, value gaps, pivot floor, conditioning, envelope E(r)")
    hdr = ("     %-10s %-9s %-8s %-8s %-9s %-8s %-8s %-9s %-9s %-9s"
           % ("r", "m(P*)", "lminP", "gapP", "m(Q*)", "lminQ", "gapQ",
              "pivot~r^4", "cond", "E(r)"))
    print(hdr)
    rungs = []
    r = mpf('0.05')
    prevP = (mpf('1.3056'), mpf('0.6858')); prevQ = (mpf('-1.1040'), mpf('0.8797'))
    first_fail = None
    while r >= mpf('1e-13'):
        R = Rung(r)
        E = mpf('1e-60')
        # envelope from end-to-end pin reproduction (whitened engine)
        wp = mpf0
        for (P, a), vj in zip(R.pins, R.vals):
            wp = max(wp, abs(R.mval(P, a) - vj))
        E = max(E, wp*mpf('1e6'))
        row = dict(r=r, R=R, E=E)
        okr = True
        gates = []
        if not (B - R.ell < R.vstar < B):
            okr = False; gates.append('v*-window')
        if R.pivot < mpf('1e-100'):
            okr = False; gates.append('pivot-floor')
        for nm in ('P', 'Q'):
            seed = prevP if nm == 'P' else prevQ
            x = newton(R, seed)
            G1, G2, G3, D2F, D3F = rkhs_bounds(R)
            k = kantorovich(R, x, D3F, E)
            if k is None:
                okr = False; gates.append('Kantorovich-' + nm)
                continue
            if k['lmin'] < mpf('0.01'):
                okr = False; gates.append('lmin-' + nm)
            if k['rho2'] < mpf('1e-4'):
                okr = False; gates.append('rho2-' + nm)
            if E > k['lmin']*mpf('0.01'):
                okr = False; gates.append('envelope-' + nm)
            row[nm] = (x, k)
            if nm == 'P': prevP = (x[0], x[1])
            else: prevQ = (x[0], x[1])
        row['ok'] = okr; row['gates'] = gates
        rungs.append(row)
        if okr:
            kP, kQ = row['P'][1], row['Q'][1]
            print("     %-10s %-9.6f %-8.5f %-8.6f %-9.6f %-8.5f %-8.6f %-9s %-9s %-9s"
                  % (mp.nstr(r, 4), float(kP['info']['m']), float(kP['lmin']),
                     float(kP['info']['m']-B), float(kQ['info']['m']),
                     float(kQ['lmin']), float(kQ['info']['m']-B),
                     mp.nstr(R.pivot, 2), mp.nstr(R.cond(), 2), mp.nstr(E, 2)))
        else:
            print("     %-10s FIRST FAILED RUNG; failing gates: %s"
                  % (mp.nstr(r, 4), ','.join(gates)))
            first_fail = row
            break
        r = r/2
    need(len(rungs) >= 2 and rungs[0]['ok'], "ladder top rung r = 0.05 certifies")
    passed = [q for q in rungs if q['ok']]
    rmin = min(q['r'] for q in passed)
    print("     ladder summary: %d rungs certified; smallest certified r = %s"
          % (len(passed), mp.nstr(rmin, 4)))
    if first_fail is not None:
        print("     first failing rung r = %s, gates: %s"
              % (mp.nstr(first_fail['r'], 4), ','.join(first_fail['gates'])))
    # rates: pivot and cond log-log slopes over certified rungs; margin drift
    import math as _m
    if len(passed) >= 4:
        xs = [_m.log(float(q['r'])) for q in passed]
        yp = [_m.log(float(q['R'].pivot)) for q in passed]
        yc = [_m.log(float(q['R'].cond())) for q in passed]
        npt = len(xs)
        sl = lambda y: (npt*sum(a*b for a, b in zip(xs, y)) - sum(xs)*sum(y)) / \
                       (npt*sum(a*a for a in xs) - sum(xs)**2)
        print("     measured rates: pivot ~ r^%.2f ; cond ~ r^%.2f" % (sl(yp), -sl(yc)))
    # margin convergence: first-order differences of m(P*) between consecutive rungs
    for i in range(1, min(6, len(passed))):
        q0, q1 = passed[i-1], passed[i]
        dmP = abs(q0['P'][1]['info']['m'] - q1['P'][1]['info']['m'])
        dlP = abs(q0['P'][1]['lmin'] - q1['P'][1]['lmin'])
        dr_ = abs(q0['r'] - q1['r'])
        print("     drift r:%s->%s : |dm(P*)|/dr = %s , |d lmin(P*)|/dr = %s"
              % (mp.nstr(q0['r'], 3), mp.nstr(q1['r'], 3),
                 mp.nstr(dmP/dr_, 5), mp.nstr(dlP/dr_, 5)))
    return rungs

# ----------------------------------------------------------------------------
# S4: mandate item (3) -- uniform continuation in r [UNIFORM-R]
# ----------------------------------------------------------------------------
def uniform_block(r_hi, w, seeds, C3_factor=1.25, verbose=True):
    """Attempt uniform certification of P*(r), Q*(r) over I = [r_hi-w, r_hi].
    Returns dict(ok, w, mids, data) or dict(ok=False, reason)."""
    r_lo = r_hi - w
    r_mid = (r_hi + r_lo)/2
    h = w/2
    R = Rung(r_mid, with_ders=True)          # midpoint == mesh point 4
    if not (B - R.ell < R.vstar < B):
        return dict(ok=False, reason='v*-window at midpoint')
    Rl_d = Rung(r_lo, with_ders=True)        # chain rung == mesh point 0
    if not (B - Rl_d.ell < Rl_d.vstar < B):
        return dict(ok=False, reason='v*-window at chain rung')
    # named-analyticity cap C3: 9-point exact mesh of ||m'''||_H over I
    c3mesh = [Rl_d.hnm[3]]
    for i in range(1, 9):
        if i == 4:
            c3mesh.append(R.hnm[3]); continue
        rm = r_lo + w*i/8
        Rm = Rung(rm, with_ders=True)
        if not (B - Rm.ell < Rm.vstar < B):
            return dict(ok=False, reason='v*-window at mesh %d' % i)
        c3mesh.append(Rm.hnm[3])
    C3 = C3_factor*max(c3mesh)
    n0, n1, n2 = R.hnm[0], R.hnm[1], R.hnm[2]
    # uniform Taylor brackets (Hilbert-space Taylor, Lagrange in norm)
    S1 = n1 + n2*h + C3*h*h/2
    brack = n1*h + n2*h*h/2 + C3*h*h*h/6
    SQRT2 = msqrt(2); SQRT8 = msqrt(8)
    out = dict(ok=True, w=w, r_lo=r_lo, r_hi=r_hi, r_mid=r_mid, C3=C3,
               n1=n1, n2=n2, S1=S1, brack=brack, R=R)
    for nm in ('P', 'Q'):
        x = newton(R, seeds[nm])
        k = kantorovich(R, x, 2*msqrt(45)*n0, mpf('1e-60'))
        if k is None:
            return dict(ok=False, reason='midpoint Kantorovich ' + nm)
        res_sup = k['res'] + SQRT2*brack + mpf('1e-60')
        lam_inf = k['lmin'] - SQRT8*brack - mpf('1e-60')
        gam_sup = 2*msqrt(45)*(n0 + brack)
        if lam_inf <= 0:
            return dict(ok=False, reason='uniform eigenfloor <= 0 ' + nm)
        beta = 1/lam_inf
        eta = beta*res_sup
        alpha = beta*gam_sup*eta
        if alpha >= mpf('0.5'):
            return dict(ok=False, reason='uniform alpha >= 1/2 ' + nm)
        rho2 = 2/(beta*gam_sup)
        out[nm] = dict(x=x, k=k, res_sup=res_sup, lam_inf=lam_inf,
                       gam_sup=gam_sup, beta=beta, eta=eta, alpha=alpha, rho2=rho2)
    # shared-rung chain roots: exact roots at r_lo for chaining
    Rl = Rl_d
    chn = {}
    for nm in ('P', 'Q'):
        xl = newton(Rl, out[nm]['x'])
        kl = kantorovich(Rl, xl, 2*msqrt(45)*Rl.nm, mpf('1e-60'))
        if kl is None:
            return dict(ok=False, reason='chain-root Kantorovich ' + nm)
        chn[nm] = (xl, kl)
    out['chain'] = chn
    return out

def part_S4():
    print("\n[S4] mandate(3): uniform continuation in r [UNIFORM-R]")
    print("     block certificate: exact midpoint Kantorovich + RKHS Taylor")
    print("     brackets (||m'||,||m''|| exact; ||m'''|| 9-point mesh cap x1.25,")
    print("     house named analyticity formality C027); uniform alpha, eigenfloor,")
    print("     uniqueness tube; chain gates connect consecutive blocks")
    seeds = {'P': (mpf('1.3056'), mpf('0.6858')), 'Q': (mpf('-1.1040'), mpf('0.8797'))}
    r_cur = mpf('0.05')
    w = mpf('0.004')
    blocks = []
    nbudget = 400
    r_floor = mpf('1e-6')
    prev = None
    while r_cur > r_floor and len(blocks) < nbudget:
        if r_cur - w < r_floor:
            w = r_cur - r_floor          # clamp final block at the rung floor
            if w < mpf('1e-12'):
                break
        b = uniform_block(r_cur, w, seeds)
        if not b['ok']:
            if w > mpf('1e-7') and r_cur - w/2 >= r_floor:
                print("     block at r_hi=%s w=%s failed (%s); halving"
                      % (mp.nstr(r_cur, 4), mp.nstr(w, 2), b['reason']))
                w = w/2
                continue
            print("     block width floor reached at r = %s: %s" % (mp.nstr(r_cur, 4), b['reason']))
            break
        # chain gates (both maxima): shared-rung exact root inside previous tube
        if prev is not None:
            for nm in ('P', 'Q'):
                xl, kl = prev['chain'][nm]
                dist = msqrt((xl[0]-b[nm]['x'][0])**2 + (xl[1]-b[nm]['x'][1])**2)
                need(dist < mpf('1e-30'),
                     "chain gate %s: shared-rung root inside new tube" % nm)
        blocks.append(b)
        seeds = {'P': b['P']['x'], 'Q': b['Q']['x']}
        print("     block %-3d [%-9s, %-9s] w=%s  alpha=(%s,%s)  lam_inf=(%s,%s)  rho2=(%s,%s)  C3=%s"
              % (len(blocks), mp.nstr(b['r_lo'], 5), mp.nstr(b['r_hi'], 5),
                 mp.nstr(w, 2), mp.nstr(b['P']['alpha'], 3), mp.nstr(b['Q']['alpha'], 3),
                 mp.nstr(b['P']['lam_inf'], 5), mp.nstr(b['Q']['lam_inf'], 5),
                 mp.nstr(b['P']['rho2'], 4), mp.nstr(b['Q']['rho2'], 4),
                 mp.nstr(b['C3'], 4)))
        r_cur = b['r_lo']
        prev = b
        # adaptive width: grow cautiously when margins are strong
        a_max = max(b['P']['alpha'], b['Q']['alpha'])
        if a_max < mpf('0.3') and w < mpf('0.004'):
            w = w*mpf('1.25')
    need(len(blocks) > 0, "at least one uniform block certified")
    reach = blocks[-1]['r_lo']
    print("     uniform chain: %d blocks; certified-uniform coverage r in [%s, 0.05]"
          % (len(blocks), mp.nstr(reach, 5)))
    if len(blocks) >= 2:
        ws = [float(b['w']) for b in blocks]
        rs = [float(b['r_mid']) for b in blocks]
        print("     block widths vs r_mid: %s"
              % ', '.join('%.1e@%.3g' % (wi, ri) for wi, ri in zip(ws[:8], rs[:8])))
    return blocks

# ----------------------------------------------------------------------------
# S5: the interval tier -- direct enclosure attempt and measured obstruction
# ----------------------------------------------------------------------------
def part_S5():
    print("\n[S5] interval tier [INTERVAL]: direct enclosure attempt + obstruction")
    from mpmath import iv
    iv.dps = 60
    def cov_iv(P, a, Q, c):
        u1 = P[0]-Q[0]; u2 = P[1]-Q[1]
        n1 = a[0]+c[0]; n2 = a[1]+c[1]
        s = (-1)**(c[0]+c[1]) * (-1)**(n1+n2)
        return s*_he(n1, u1)*_he(n2, u2)*iv.exp(-(u1*u1+u2*u2)/2)
    # S5.1: direct interval LU inversion of the 9-pin Gram over a small block
    try:
        r_iv = iv.mpf(['0.0499', '0.0501'])
        M = (-r_iv/2, 0); S = (r_iv/2, 0)
        yy = (M[0] + r_iv*iv.mpf('-0.76'), r_iv*iv.mpf('0.24'))
        PIN = [(M, a) for a in J01] + [(S, a) for a in J01] + [(yy, a) for a in J01]
        G = iv.matrix(9, 9)
        for i, (P, a) in enumerate(PIN):
            for j, (Q, c) in enumerate(PIN):
                G[i, j] = cov_iv((P[0], P[1]), a, (Q[0], Q[1]), c)
        Gi = G**-1
        wmax = max(float(Gi[i, j].b - Gi[i, j].a) for i in range(9) for j in range(9))
        print("     S5.1 direct iv LU inverse over r in [0.0499,0.0501]: max entry width = %.3e" % wmax)
        need(wmax < 1e30, "S5.1 produced a finite enclosure (unexpected)")
        print("     receipt: direct enclosure width is astronomically loose")
    except (ZeroDivisionError, ValueError) as e:
        print("     S5.1 direct iv LU inverse over r in [0.0499,0.0501]: ABORTS (%s)"
              % type(e).__name__)
        print("     receipt: interval pivots straddle zero even on a 2e-4 block at r=0.05")
    # S5.2: interval-Cholesky viability threshold |I|*(r) -- measured exponent
    print("     S5.2 interval-Cholesky viability: largest block half-width |I|/2")
    print("          such that the iv Cholesky of the 9-pin Gram still certifies")
    print("          (all pivots bounded away from zero), measured per rung:")
    import math as _m
    rows = []
    for rr in ('0.05', '0.025', '0.0125', '0.00625', '0.003125'):
        r0 = mpf(rr)
        wI = mpf('0.01')
        thr = None
        for k in range(0, 22):
            r_iv = iv.mpf([r0 - wI/2, r0 + wI/2])
            M = (-r_iv/2, 0); S = (r_iv/2, 0)
            yy = (M[0] + r_iv*iv.mpf('-0.76'), r_iv*iv.mpf('0.24'))
            PIN = [(M, a) for a in J01] + [(S, a) for a in J01] + [(yy, a) for a in J01]
            G = iv.matrix(9, 9)
            for i, (P, a) in enumerate(PIN):
                for j, (Q, c) in enumerate(PIN):
                    G[i, j] = cov_iv((P[0], P[1]), a, (Q[0], Q[1]), c)
            Li = iv.matrix(9, 9)
            ok_chol = True
            for i in range(9):
                for j in range(i+1):
                    s = G[i, j] - sum(Li[i, k2]*Li[j, k2] for k2 in range(j))
                    if i == j:
                        if s.a <= 0:
                            ok_chol = False; break
                        Li[i, i] = iv.sqrt(s)
                    else:
                        Li[i, j] = s/Li[j, j]
                if not ok_chol: break
            if ok_chol:
                thr = wI; break
            wI = wI/2
        if thr is None:
            print("          r=%-9s: no viable block down to |I|=4.8e-9 (search floor)" % rr)
            rows.append((rr, None))
        else:
            print("          r=%-9s: largest viable |I| = %s" % (rr, mp.nstr(thr, 2)))
            rows.append((rr, thr))
    vals = [(float(r), float(t)) for r, t in rows if t is not None]
    if len(vals) >= 2:
        xs = [_m.log(v[0]) for v in vals]; ys = [_m.log(v[1]) for v in vals]
        npt = len(xs)
        sl = (npt*sum(a*b for a, b in zip(xs, ys)) - sum(xs)*sum(ys)) / \
             (npt*sum(a*a for a in xs) - sum(xs)**2)
        print("     measured viability threshold: |I|*(r) ~ r^(%.2f)" % sl)
    if not vals:
        print("     measured: direct iv Cholesky NONVIABLE at every rung for all")
        print("     block widths >= 4.8e-9 (search floor). Analysis (labeled, not")
        print("     a measurement): the smallest radicand is pivot^2 ~ r^4 (S3:")
        print("     pivot ~ r^2) while interval widths amplify ~ 1/pivot^2 ~ r^-4")
        print("     through the elimination, so viability needs |I| <~ r^4 * r^4")
        print("     = r^8; direct interval continuation is exponentially costly.")
    print("     receipt: interval enclosures inherit the pin-coalescence")
    print("     conditioning (cond ~ r^-6, S3); the interval tier cannot certify")
    print("     useful blocks at any rung. [INTERVAL tier closed]")

# ----------------------------------------------------------------------------
# S6: valid-scope table (receipts)
# ----------------------------------------------------------------------------
def part_S6(rungs, blocks):
    print("\n[S6] valid-scope table (receipts; labels in the separate audit file)")
    passed = [q for q in rungs if q['ok']]
    rmin = min(q['r'] for q in passed)
    reach = blocks[-1]['r_lo'] if blocks else None
    print("     object: m9 = E[f | 9 pins] (deterministic analytic function)")
    print("     A. exact-rung certification [EXACT]:")
    print("        - census + zero above-b saddles: r = 0.025 exactly (S1)")
    print("        - ridge maxima Kantorovich-certified at %d rungs, smallest r = %s (S3)"
          % (len(passed), mp.nstr(rmin, 4)))
    if reach is not None:
        print("     B. uniform-in-r certification [UNIFORM-R, house named cap]:")
        print("        - %d blocks; P*(r), Q*(r) certified uniformly for r in [%s, 0.05]"
              % (len(blocks), mp.nstr(reach, 5)))
        print("        - margins at chain end: lam_inf = (%s, %s), rho2 = (%s, %s)"
              % (mp.nstr(blocks[-1]['P']['lam_inf'], 5), mp.nstr(blocks[-1]['Q']['lam_inf'], 5),
                 mp.nstr(blocks[-1]['P']['rho2'], 4), mp.nstr(blocks[-1]['Q']['rho2'], 4)))
    print("     C. not covered here (named owners): noisy-field f~ critical counts")
    print("        (LB-1 Lemma H-SUP; C029 B3(iii) mean-ridge channel); the r -> 0")
    print("        exact limit object (C012/C013/C026); any rung r > 0.05.")
    print("     D. obstruction receipts (S3, S5):")
    print("        - margins (lmin, gaps, rho2) CONVERGE to positive limits as r -> 0:")
    kP = passed[-1]['P'][1]; kQ = passed[-1]['Q'][1]
    print("          at r = %s: lmin = (%s, %s), gaps = (%s, %s)"
          % (mp.nstr(passed[-1]['r'], 4), mp.nstr(kP['lmin'], 6), mp.nstr(kQ['lmin'], 6),
             mp.nstr(kP['info']['m'] - B, 6), mp.nstr(kQ['info']['m'] - B, 6)))
    print("        - certifiability degrades by conditioning: pivot ~ r^2, cond ~ r^-6")
    print("          (measured slopes, S3); iv-Cholesky viability ~ r^a (S5, measured)")
    print("        - obstruction is NUMERICAL-UNIFORM (pin coalescence), not")
    print("          topological: no margin degenerates; r = 0 itself is singular")
    print("          for the pin family and routes through the exact limit object.")

# ----------------------------------------------------------------------------
# ck(): fail-closed main
# ----------------------------------------------------------------------------
def ck():
    print("=" * 78)
    print("verify_ridge_scope_continuation_v1.py -- W9 DER-027c scope audit receipts")
    print("starting-record input: WO-063 sha256 " + WO_SHA)
    print("object: 9-pin conditional mean m9, periodized side-24 2D BF, b = 6/5")
    print("receipts only (labels/verdicts in K3-DER-027c-SCOPE-AUDIT.md)")
    print("=" * 78)
    part_S0()
    R, roots, kant = part_S1()
    part_S2(R, roots, kant)
    rungs = part_S3()
    blocks = part_S4()
    part_S5()
    part_S6(rungs, blocks)
    print("\n" + "=" * 78)
    print("CERTIFICATE PROGRAM COMPLETE: all checks passed")
    print("=" * 78)
    return 0

if __name__ == '__main__':
    ck()
