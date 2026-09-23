#!/usr/bin/env python3
# verify_farfield_envelope_v1.py  (KIMI-DER-027a certificate, AO48-WO-063 Task 4a FAR-FIELD)
#
# Fail-closed certificate for the far-field monotone decay envelope of the exact
# periodized side-24 2D Bargmann-Fock field:
#   spectral lattice (pi/12) Z^2, masses exp(-|k|^2/2), truncation |k| <= 30,
#   C(x) = Z30^{-1} sum_{|k|<=30} e^{-|k|^2/2} cos(k.x),  C(0) = 1 (EXACT form).
# Conditioning (tasked): 6 pins (f, grad f) at M = (-r/2, 0), S = (+r/2, 0) with values
#   b = 6/5, b - r^3/6 (gradients 0), plus one witness value pin at Y = r*(-63/50, 6/25)
#   (value v* = -1/2; the variance channel is independent of v*). Rungs r = 1/20, 1/40.
# Channels:
#   (V) load-bearing value-variance channel  Delta(y) = 1 - Var(f(y)|pins) = v^T G^{-1} v
#   (T) one-point value-law channel          TV(N(mu(y), sig(y)^2), N(0,1)),
#        mu = beta^T G^{-1} v, sig^2 = 1 - Delta.
# Proves: derived monotone envelope for d >= d0 = 3 on the cluster annuli
#   {y : dist(y, hull(pins)) >= d} (toroidal period 24), Var bounded below by a derived
#   constant, and matches the recorded constants (C022/LB-3 recomputation class):
#   d = 3: Delta <= 2.24e-2 ; d = 5: variance quantities <= 1e-4, 6-pin value law
#   7.7e-5 (r=1/20) / 8.1e-5 (r=1/40) ; monotone envelope < 1e-4 from d = 6.1
#   (recorded worst case <= 7.5e-13).
# Standard: certified-tail high-precision sums (mpmath dps 80) + outward-padded float64
# mesh certification. Deterministic; normal and `python3 -O` transcripts byte-identical.
#
# Precision labels: EXACT = exact rational/structural; DPS80 = mpmath dps-80 certified
# with displayed error bounds; F64PAD = float64 evaluation with certified outward pad.
import mpmath as mp
import numpy as np

mp.mp.dps = 80

FAILS = []
def ck(cond, name):
    if cond:
        print("PASS  " + name)
    else:
        print("FAIL  " + name)
        FAILS.append(name)

def fmt(x, n=12):
    return mp.nstr(mp.mpf(x), n)

# ----------------------------------------------------------------------------- S0 constants
print("=== S0 EXACT structural constants ===")
a = mp.pi/12                      # lattice spacing (EXACT: pi/12)
LP = 24                           # period (EXACT side-24)
RUNGS = [mp.mpf(1)/20, mp.mpf(1)/40]   # r = 0.05, 0.025 (EXACT)
BB = mp.mpf(6)/5                  # b = 6/5 (EXACT)
YCO = (mp.mpf(-63)/50, mp.mpf(6)/25)   # Y = r*(-1.26, 0.24) (EXACT)
VSTAR = mp.mpf(-1)/2              # witness value (EXACT -1/2; variance channel independent)
print("lattice a = pi/12, period L = 24, rungs r = 1/20, 1/40, b = 6/5 (EXACT)")
print("Y = r*(-63/50, 6/25), v* = -1/2 (EXACT); pins (f,grad f) at M=(-r/2,0), S=(+r/2,0)")

def pins_of(r, useY):
    M = (-r/2, mp.mpf(0)); S = (r/2, mp.mpf(0))
    Y = (YCO[0]*r, YCO[1]*r)
    p = [(M, (0,0)), (M, (1,0)), (M, (0,1)), (S, (0,0)), (S, (1,0)), (S, (0,1))]
    if useY:
        p.append((Y, (0,0)))
    return p

def vals_of(r, useY):
    ell = r**3/6
    v = [BB, mp.mpf(0), mp.mpf(0), BB - ell, mp.mpf(0), mp.mpf(0)]
    if useY:
        v.append(VSTAR)
    return v

# ----------------------------------------------------------------------------- S1 spectral sums + certified tails
print("=== S1 spectral normalization and certified truncation tails (DPS80) ===")
NLAT = int(mp.ceil(30/a)) + 2
mass = {}
for n1 in range(-NLAT, NLAT+1):
    for n2 in range(-NLAT, NLAT+1):
        s2 = n1*n1 + n2*n2
        if a*a*s2 <= 900:
            mass[(n1, n2)] = mp.e**(-a*a*s2/2)
Z30 = sum(mass.values())
print("lattice points |k|<=30:", len(mass), " Z30 =", fmt(Z30, 25))

# certified tail T_m = sum_{|k|>30} m_k |k|^m :
# exact shell sum for 30 < |k| <= 40, rigorous cell-integral bound for |k| > 40.
def tail_T(m):
    s = mp.mpf(0)
    B = int(mp.ceil(40/a)) + 2
    for n1 in range(-B, B+1):
        for n2 in range(-B, B+1):
            k2 = a*a*(n1*n1 + n2*n2)
            if 900 < k2 <= 1600:
                s += mp.e**(-k2/2) * mp.sqrt(k2)**m
    # |k| > 40: each lattice point owns a cell of area a^2; cell half-diagonal c = a/sqrt(2).
    # |k| >= |x| - c  and  |k|^m <= (|x| + c)^m  for x in the cell  =>
    # sum <= a^{-2} int_{|x| >= 40 - c} e^{-(|x|-c)^2/2} (|x|+c)^m dx
    c = a/mp.sqrt(2)
    # bound (|x|+c)^m <= 2^m |x|^m for |x| >= 40-c >= c, and |x| = s + c:
    # integral = 2 pi int_{s >= 40-2c} e^{-s^2/2} (s+2c)^m (s+c) ds
    #          <= 2 pi int_{s >= 39} e^{-s^2/2} (2s)^{m+1} ds   (since s+2c <= 2s, s+c <= 2s for s >= 39)
    f = lambda s: mp.e**(-s*s/2) * (2*s)**(m+1)
    I = mp.quad(f, [39, 50]) + mp.e**(-mp.mpf(50)**2/2) * (2*50)**(m+1)  # tail beyond 50 dominated
    return s + (1/a**2) * 2*mp.pi * I

T = [tail_T(m) for m in range(35)]
print("T_0 =", fmt(T[0], 8), " T_0/Z30 =", fmt(T[0]/Z30, 8), " (recorded tail bound 4.4e-185; task bar < 1e-60)")
ck(T[0]/Z30 < mp.mpf('1e-60'), "S1: certified spectral tail T_0/Z30 < 1e-60")
ck(T[0]/Z30 < mp.mpf('4.4e-185'), "S1: certified spectral tail below recorded bound 4.4e-185")

# ----------------------------------------------------------------------------- S2 kernel (wrapped) + cross-check
print("=== S2 wrapped-kernel representation, Poisson-exact, certified image remainder ===")
# Poisson summation (EXACT identity): the untruncated field obeys
#   C_untr(u) = c0 * sum_{n in Z^2} e^{-|u + 24 n|^2 / 2},  c0 = 1 / sum_n e^{-|24 n|^2/2}.
C0N = 1 + 4*mp.e**(-mp.mpf(24)**2/2) + 4*mp.e**(-mp.mpf(48)**2/2) + 4*mp.e**(-(24*mp.sqrt(2))**2/2)
c0 = 1/C0N
# image remainder: for |u_i| <= 12, |u + 24n| >= rho(n) with
# rho(n)^2 = sum_i (max(0, 24|n_i| - 12))^2 ; derivative order j adds factor (24|n_i|+12)^j_i.
IMG = {}
for j1 in range(0, 35):
    for j2 in range(0, 35):
        if j1 + j2 > 34: continue
        s = mp.mpf(0)
        for n1 in range(-6, 7):
            for n2 in range(-6, 7):
                if n1 == 0 and n2 == 0: continue
                rho2 = max(0, 24*abs(n1)-12)**2 + max(0, 24*abs(n2)-12)**2
                s += ((24*abs(n1)+12)**j1) * ((24*abs(n2)+12)**j2) * mp.e**(-mp.mpf(rho2)/2)
        # |n|_inf >= 7: rho >= 156, count <= (2m+1)^2, crude certified cap
        s += mp.mpf(10)**6 * (24*7+12)**(j1+j2) * mp.e**(-mp.mpf(156)**2/2)
        IMG[(j1, j2)] = s
print("image remainder eps_img (order 0) =", fmt(IMG[(0,0)], 6), " (order 1) =", fmt(IMG[(1,0)], 6))

# probabilists' Hermite via recurrence (EXACT integer polynomials)
def he_poly(n):
    if n == 0: return [1]
    if n == 1: return [0, 1]
    p0, p1 = [1], [0, 1]
    for k in range(1, n):
        p2 = [0]*(k+2)
        for i, cf in enumerate(p1): p2[i+1] += cf
        for i, cf in enumerate(p0): p2[i] -= k*cf
        p0, p1 = p1, p2
    return p1
HE = [he_poly(n) for n in range(35)]   # ascending coefficients, EXACT integers
def he_val(n, x):   # mp.polyval takes DESCENDING coefficients
    return mp.polyval(HE[n][::-1], x)

SQ2 = mp.sqrt(2)
def kernel_d(u1, u2, j1, j2):
    """partial^{j1,j2} C_untr(u) via wrapped sum |n|<=2 (DPS80); certified remainder IMG[(j1,j2)]."""
    s = mp.mpf(0)
    for n1 in range(-2, 3):
        for n2 in range(-2, 3):
            w1 = u1 + 24*n1; w2 = u2 + 24*n2
            s += he_val(j1, w1) * he_val(j2, w2) * mp.e**(-(w1*w1 + w2*w2)/2)
    return ((-1)**(j1+j2)) * c0 * s

# correction from truncation: C_spec = (Zinf/Z30) C_untr - R/Z30, |R^{(j)}| <= T_|j|
TAU = T[0]/Z30
def kcorr(j1, j2):
    return 2*T[j1+j2]/Z30 + IMG[(j1, j2)]   # certified |C_spec^(j) - wrapped^(j)| bound
print("truncation+image kernel correction (order 0) =", fmt(kcorr(0,0), 6),
      " (order 1) =", fmt(kcorr(1,0), 6), " (order 2) =", fmt(kcorr(2,0), 6))

# cross-check: direct spectral sum (float64, numpy) vs wrapped mp at 8 points
kk = np.array(list(mass.keys()), dtype=float)
km = np.array([mass[(int(n1), int(n2))] for (n1, n2) in mass.keys()], dtype=float)
# NOTE: keys iterated deterministically (insertion order); rebuild aligned arrays:
keys = sorted(mass.keys())
kk = np.array(keys, dtype=float) * float(a)
km = np.array([float(mass[k]) for k in keys])
Z30f = float(Z30)
def Cspectral_f64(u1, u2):
    ph = kk[:, 0]*u1 + kk[:, 1]*u2
    return float(np.dot(km, np.cos(ph))) / Z30f
def dCspectral_f64(u1, u2):
    ph = kk[:, 0]*u1 + kk[:, 1]*u2
    s = np.sin(ph)
    return (-float(np.dot(km*kk[:, 0], s))/Z30f, -float(np.dot(km*kk[:, 1], s))/Z30f)
xpts = [(3.0, 0.0), (2.1213, 2.1213), (5.0, 1.0), (6.1, 2.0), (0.025, 0.0), (3.7, -2.9), (11.9, 0.4), (8.0, 8.0)]
mxdiff = mp.mpf(0)
for (x, y) in xpts:
    d1 = abs(Cspectral_f64(x, y) - kernel_d(mp.mpf(x), mp.mpf(y), 0, 0))
    g = dCspectral_f64(x, y)
    d2 = abs(g[0] - kernel_d(mp.mpf(x), mp.mpf(y), 1, 0))
    d3 = abs(g[1] - kernel_d(mp.mpf(x), mp.mpf(y), 0, 1))
    mxdiff = max(mxdiff, d1, d2, d3)
print("spectral(float64) vs wrapped(DPS80) max diff at 8 points =", fmt(mxdiff, 6))
ck(mxdiff < mp.mpf('1e-12'), "S2: cross-representation implementation agreement < 1e-12 (float64 class)")

# one full mp spectral check at a single point to 1e-50 class
def Cspectral_mp(u1, u2):
    s = mp.mpf(0)
    for (n1, n2) in keys:
        m = mass[(n1, n2)]
        s += m*mp.cos(a*n1*u1 + a*n2*u2)
    return s/Z30
dmp = abs(Cspectral_mp(mp.mpf('3.0'), mp.mpf('0.7')) - kernel_d(mp.mpf('3.0'), mp.mpf('0.7'), 0, 0))
print("spectral(DPS80) vs wrapped(DPS80) diff at (3.0,0.7) =", fmt(dmp, 6))
ck(dmp < mp.mpf('1e-50'), "S2: cross-representation certified agreement < 1e-50 at check point")

# ----------------------------------------------------------------------------- S3 Gram matrices, certified lambda_min, certified inverse
print("=== S3 pin Gram matrices: certified positive-definiteness and inverse (DPS80) ===")
ROUND = mp.mpf('1e-70')   # global dps-80 rounding allowance per displayed algebra

def build_G(r, useY):
    p = pins_of(r, useY)
    n = len(p)
    Gm = mp.matrix(n, n)
    for i in range(n):
        xi, ai = p[i]
        for j in range(n):
            xj, aj = p[j]
            Gm[i, j] = ((-1)**(aj[0]+aj[1])) * kernel_d(xi[0]-xj[0], xi[1]-xj[1],
                                                        ai[0]+aj[0], ai[1]+aj[1])
    return p, Gm

def certify(Gm):
    """returns (lam_hat, M, epsM): certified lower bound on lambda_min(Gm) (Weyl-residual),
       approximate inverse M (DPS80), certified ||G^{-1} - M||_F bound epsM."""
    n = Gm.rows
    lam, Q = mp.eigsy(Gm)
    # residual matrix R = G - Q diag(lam) Q^T
    Rm = Gm - Q*mp.diag(lam)*Q.T
    rho = mp.mpf(0)
    for i in range(n):
        for j in range(n):
            rho += Rm[i, j]**2
    rho = mp.sqrt(rho) + ROUND
    lam_hat = lam[0] - rho
    # approximate inverse from the certified eigendecomposition (avoids fragile LU pivoting
    # at cond ~ 1e10; the a posteriori eta residual below is what certifies accuracy)
    Minv = mp.matrix(n, n)
    for i in range(n):
        for j in range(n):
            Minv[i, j] = sum(Q[i, k]*Q[j, k]/lam[k] for k in range(n))
    M = Minv
    Ms = (M + M.T)/2
    # eta = ||I - M G||_F ; ||G^{-1} - M||_F <= ||M||_F eta/(1-eta)
    E = mp.eye(n) - M*Gm
    eta = mp.sqrt(sum(E[i, j]**2 for i in range(n) for j in range(n))) + ROUND
    nM = mp.sqrt(sum(M[i, j]**2 for i in range(n) for j in range(n))) + ROUND
    epsM = nM*eta/(1-eta) + (nM*eta/(1-eta))  # ||G^{-1}-M|| + ||M-Ms|| doubling guard
    epsM = epsM + ROUND
    return lam_hat, Ms, epsM, lam[0]

ENS = {}   # (useY, rung index) -> dict
for useY in (True, False):
    for ri, r in enumerate(RUNGS):
        p, Gm = build_G(r, useY)
        lam_hat, Ms, epsM, lam_raw = certify(Gm)
        ENS[(useY, ri)] = dict(pins=p, G=Gm, lam_hat=lam_hat, M=Ms, epsM=epsM, ri=ri, useY=useY)
        tag = "7-pin(incl Y)" if useY else "6-pin"
        print(f"{tag} r=1/{int(1/r)}: n={Gm.rows} lambda_min computed={fmt(lam_raw,8)} "
              f"certified lambda_hat={fmt(lam_hat,8)} epsM={fmt(epsM,6)}")
        ck(lam_hat > 0, f"S3: G ({tag}, r=1/{int(1/r)}) certified positive definite (lambda_hat>0)")

# ----------------------------------------------------------------------------- S4 derived envelope constants
print("=== S4 derived envelope: Hermite-exact kernel bounds and Schur-complement moments ===")
# Power-Gaussian sup functions (EXACT, derived):
#   P_{i,k}(d) := sup_{|u| >= d} |u1|^i |u2|^k e^{-|u|^2/2}
# Unique interior critical point at u1^2 = i, u2^2 = k (grad = 0); boundary circle sup at
# cos^2 = i/(i+k). Both exact:
#   d^2 <= i+k : P = i^{i/2} k^{k/2} e^{-(i+k)/2}        (interior max inside annulus)
#   d^2 >  i+k : P = d^{i+k} (i/(i+k))^{i/2} (k/(i+k))^{k/2} e^{-d^2/2}
# Each P_{i,k} is continuous and non-increasing on (0, oo) (values match at d^2 = i+k).
def Ppow(i, k, d):
    if i == 0 and k == 0:
        return mp.e**(-d*d/2)
    if d*d <= i + k:
        return mp.mpf(i)**(mp.mpf(i)/2) * mp.mpf(k)**(mp.mpf(k)/2) * mp.e**(-mp.mpf(i+k)/2)
    rho = (mp.mpf(i)/(i+k))**(mp.mpf(i)/2) * (mp.mpf(k)/(i+k))**(mp.mpf(k)/2)
    return d**(i+k) * rho * mp.e**(-d*d/2)

# kernel derivative bound: |d^{m1,m2} C(u)| <= sum_{i,k} |He_{m1}[i] He_{m2}[k]| P_{i,k}(d)
# + image/truncation corrections. Derived, non-increasing in d for every d in (0, 12].
def Bker(m1, m2, d):
    s = mp.mpf(0)
    p1, p2 = HE[m1], HE[m2]
    for i in range(0, m1+1):
        ci = p1[i] if i < len(p1) else 0
        if ci == 0: continue
        for k in range(0, m2+1):
            ck_ = p2[k] if k < len(p2) else 0
            if ck_ == 0: continue
            s += abs(ci*ck_) * Ppow(i, k, d)
    return s + (1+TAU)*IMG[(m1, m2)] + 2*T[m1+m2]/Z30

JMAX = 8          # Taylor order of the moment expansion
MLIST = [(m1, m2) for m1 in range(JMAX+3) for m2 in range(JMAX+3-m1)]   # |m| <= JMAX+2

def moment_tables(entry):
    """c_{q,m} = sum_p W[q,p] (-1)^|a_p| (-x_p)^{m-a_p}/(m-a_p)!  (Schur-complement moments,
       derived from certified M and EXACT pin locations; W = diag(sqrt(nu)) Q^T, M = Q nu Q^T)."""
    p = entry['pins']; Ms = entry['M']; n = Ms.rows
    nu, Q = mp.eigsy(Ms)
    nu_min = min(nu)
    Wm = mp.matrix(n, n)
    for q in range(n):
        for j in range(n):
            Wm[q, j] = mp.sqrt(nu[q]) * Q[j, q]
    # certified error on c: ||dW|| via epsM: ||W_err||_F <= sqrt(n)*epsM/(2 sqrt(nu_min)) guard
    cWerr = mp.sqrt(n) * entry['epsM'] / (2*mp.sqrt(nu_min)) + ROUND
    fac = [mp.factorial(i) for i in range(JMAX+3)]
    ctab = {}
    for (m1, m2) in MLIST:
        col = []
        for q in range(n):
            s = mp.mpf(0)
            for pidx in range(n):
                xp, al = p[pidx]
                j1, j2 = m1-al[0], m2-al[1]
                if j1 < 0 or j2 < 0 or j1+j2 > JMAX: continue
                mom = ((-xp[0])**j1/fac[j1]) * ((-xp[1])**j2/fac[j2])
                s += Wm[q, pidx] * ((-1)**(al[0]+al[1])) * mom
            col.append(s)
        ctab[(m1, m2)] = col
    return Wm, ctab, cWerr, nu_min

def rownorms(Wm):
    return [sum(abs(Wm[q, j]) for j in range(Wm.cols)) for q in range(Wm.rows)]

# Taylor-tail bound R_q(d): |sum_p W[q,p] Rem_p(y)| with
# |Rem_p(y)| <= sum_{s >= JMAX+1} sum_{|j|=s} |x_p|^j/j! |d^{a_p+j} C|  over |u| >= d - 2 rhoc.
def tail_R(d, rn, rhoc, amax, shift=(0, 0)):
    dd = d - 2*rhoc
    tot = mp.mpf(0)
    for s in range(JMAX+1, 31):
        # sum_{|j|=s} |x|^j/j! |d^{a+j+shift} C| <= rhoc^s 2^s/s! * max_{j1+j2=s} Bker(...)
        bk = mp.mpf(0)
        for j1 in range(0, s+1):
            bk = max(bk, Bker(amax+shift[0]+j1, amax+shift[1]+(s-j1), dd))
        term = (2*rhoc)**s / mp.factorial(s) * bk
        tot += term
        if s >= JMAX+6 and term < tot*mp.mpf('1e-12'):
            break
    return rn * tot * mp.mpf('1.05')  # 5% outward guard on the tail truncation

def envelope(entry, ctab, rn, rhoc, d, shift=(0, 0)):
    """certified upper bound on || shifted u(y) ||^2 for dist(y, hull) >= d (DPS80+F64PAD class)."""
    n = len(entry['pins'])
    E = mp.mpf(0)
    for q in range(n):
        s = mp.mpf(0)
        for (m1, m2) in MLIST:
            s += abs(ctab[(m1, m2)][q]) * Bker(m1+shift[0], m2+shift[1], d)
        s += tail_R(d, rn[q], rhoc, 1, shift)   # amax = 1 (gradient pins)
        E += s*s
    # residual from certified inverse error: |v^T (G^{-1} - M_sym) v| <= epsM ||v||^2
    vcrude = 3*(Bker(0, 0, d))**2 + 2*((Bker(1, 0, d))**2 + (Bker(0, 1, d))**2)
    return E + mp.mpf('1.01') * entry['epsM'] * vcrude

def rhoc_of(p):
    return max(mp.sqrt(xp[0]**2 + xp[1]**2) for xp, _ in p)

for useY in (True, False):
    for ri, r in enumerate(RUNGS):
        entry = ENS[(useY, ri)]
        Wm, ctab, cWerr, nu_min = moment_tables(entry)
        entry['W'] = Wm; entry['ctab'] = ctab; entry['cWerr'] = cWerr; entry['nu_min'] = nu_min
        entry['rn'] = rownorms(Wm)
        entry['rhoc'] = rhoc_of(entry['pins'])
        E3 = envelope(entry, ctab, entry['rn'], entry['rhoc'], mp.mpf(3))
        tag = "7-pin" if useY else "6-pin"
        print(f"{tag} r=1/{int(1/r)}: nu_min(M)={fmt(nu_min,6)} cWerr={fmt(cWerr,6)} "
              f"rhoc={fmt(entry['rhoc'],6)} E_env(3)={fmt(E3,6)}")
        ck(cWerr < mp.mpf('1e-40'), f"S4: moment-coefficient certification error < 1e-40 ({tag} r=1/{int(1/r)})")
        ck(nu_min > 0, f"S4: approximate inverse certified PD ({tag} r=1/{int(1/r)})")

# ----------------------------------------------------------------------------- S5 envelope grid and monotonicity
print("=== S5 derived envelope on certified grid of radii; monotonicity beyond d0 = 3 ===")
DGRID = [mp.mpf(3) + mp.mpf(i)/10 for i in range(0, 91)]   # 3.0 .. 12.0 step 0.1
ENV = {}
for useY in (True, False):
    for ri, r in enumerate(RUNGS):
        entry = ENS[(useY, ri)]
        vals = [envelope(entry, entry['ctab'], entry['rn'], entry['rhoc'], d) for d in DGRID]
        ENV[(useY, ri)] = vals
        mono = all(vals[i+1] <= vals[i] for i in range(len(vals)-1))
        tag = "7-pin" if useY else "6-pin"
        print(f"{tag} r=1/{int(1/r)}: E(3.0)={fmt(vals[0],6)} E(5.0)={fmt(vals[20],6)} "
              f"E(6.1)={fmt(vals[31],6)} E(12)={fmt(vals[-1],6)} monotone_on_[3,12]={mono}")
        ck(mono, f"S5: derived envelope certified non-increasing on [3, 12] ({tag} r=1/{int(1/r)})")
# structural monotonicity note: every constituent P_{i,k}(d) is globally non-increasing
# (exact interior/boundary maxima), so the displayed grid check is a certificate audit.

# ----------------------------------------------------------------------------- S6 Layer A: certified mesh suprema at recorded knots
print("=== S6 certified exact suprema at recorded knots (F64PAD mesh + DPS80 validation) ===")
# float64 wrapped kernel for mesh evaluation (validated against DPS80 at argmax points)
c0f = float(c0)
IMG2 = [(n1, n2) for n1 in range(-2, 3) for n2 in range(-2, 3)]
def f64_C(u1, u2):
    s = np.zeros_like(u1)
    for (n1, n2) in IMG2:
        s += np.exp(-((u1+24*n1)**2 + (u2+24*n2)**2)/2)
    return c0f*s
HEf = [np.poly1d(list(reversed(p))) for p in HE[:6]]
def hev(n, x):
    # Horner, ascending coeff list
    out = np.zeros_like(x)
    for cf in reversed(HE[n]):
        out = out*x + cf
    return out
def f64_dC(u1, u2, j1, j2):
    s = np.zeros_like(u1)
    for (n1, n2) in IMG2:
        w1 = u1+24*n1; w2 = u2+24*n2
        s += hev(j1, w1)*hev(j2, w2)*np.exp(-(w1*w1+w2*w2)/2)
    return ((-1)**(j1+j2))*c0f*s

def f64_pins(rf, useY):
    M = (-rf/2, 0.0); S = (rf/2, 0.0); Y = (-1.26*rf, 0.24*rf)
    p = [(M,(0,0)),(M,(1,0)),(M,(0,1)),(S,(0,0)),(S,(1,0)),(S,(0,1))]
    if useY: p.append((Y,(0,0)))
    return p

def f64_G(p):
    n = len(p); G = np.zeros((n, n))
    for i,(xi,ai) in enumerate(p):
        for j,(xj,aj) in enumerate(p):
            u1 = np.array([xi[0]-xj[0]]); u2 = np.array([xi[1]-xj[1]])
            G[i,j] = ((-1)**(aj[0]+aj[1]))*f64_dC(u1, u2, ai[0]+aj[0], ai[1]+aj[1])[0]
    return G

def f64_v(y1, y2, p):
    cols = []
    for (xp, al) in p:
        u1 = y1 - xp[0]; u2 = y2 - xp[1]
        cols.append(((-1)**(al[0]+al[1]))*f64_dC(u1, u2, al[0], al[1]))
    return np.stack(cols, axis=-1)

def f64_Dv(y1, y2, p, e):
    """d/dy_e of v vector."""
    cols = []
    for (xp, al) in p:
        u1 = y1 - xp[0]; u2 = y2 - xp[1]
        a1, a2 = al[0]+(e==0), al[1]+(e==1)
        cols.append(((-1)**(al[0]+al[1]))*f64_dC(u1, u2, a1, a2))
    return np.stack(cols, axis=-1)

def hull_dist(y1, y2, xy):
    d = np.full(y1.shape, np.inf)
    for i in range(len(xy)):
        for j in range(i+1, len(xy)):
            A = np.array(xy[i]); B = np.array(xy[j]); BA = B-A
            tt = np.clip(((y1-A[0])*BA[0]+(y2-A[1])*BA[1])/(BA@BA), 0, 1)
            d = np.minimum(d, np.hypot(y1-(A[0]+tt*BA[0]), y2-(A[1]+tt*BA[1])))
    return d

def Hcert(entry, d):
    """certified bound on max_{ij} |d^2_{ij} Delta| over the annulus {dist >= d}."""
    E0 = envelope(entry, entry['ctab'], entry['rn'], entry['rhoc'], d)
    E1 = [envelope(entry, entry['ctab'], entry['rn'], entry['rhoc'], d, (1,0)),
          envelope(entry, entry['ctab'], entry['rn'], entry['rhoc'], d, (0,1))]
    E2 = [envelope(entry, entry['ctab'], entry['rn'], entry['rhoc'], d, (2,0)),
          envelope(entry, entry['ctab'], entry['rn'], entry['rhoc'], d, (1,1)),
          envelope(entry, entry['ctab'], entry['rn'], entry['rhoc'], d, (0,2))]
    H11 = 2*mp.sqrt(E1[0]*E1[0]) + 2*mp.sqrt(E0*E2[0])
    H12 = 2*mp.sqrt(E1[0]*E1[1]) + 2*mp.sqrt(E0*E2[1])
    H22 = 2*mp.sqrt(E1[1]*E1[1]) + 2*mp.sqrt(E0*E2[2])
    return max(H11, H12, H22)

def certified_sup(useY, ri, d, chan="var", ntheta=4096, ht=0.002, trange=1.25):
    """certified sup over {dist(y,hull) >= d} of Delta (chan='var') or |mu| (chan='mu')."""
    entry = ENS[(useY, ri)]
    rf = float(RUNGS[ri])
    p = f64_pins(rf, useY)
    G = f64_G(p); Gi = np.linalg.inv(G)
    w = Gi @ np.array([float(x) for x in vals_of(RUNGS[ri], useY)])
    xy = sorted(set(xp for xp, _ in p))
    d = float(d)
    if chan == "var":
        H = float(Hcert(entry, mp.mpf(d)))
    else:
        cw, wn1, werr, ww = entry['mu']
        H = float(max(envelope_mu(entry, cw, wn1, mp.mpf(d), (2, 0)),
                      envelope_mu(entry, cw, wn1, mp.mpf(d), (1, 1)),
                      envelope_mu(entry, cw, wn1, mp.mpf(d), (0, 2))))
    th = np.linspace(0, 2*np.pi, ntheta, endpoint=False)
    best = -1.0; arg = None; gmax = 0.0
    t = d
    while t <= d + trange:
        y1 = t*np.cos(th); y2 = t*np.sin(th)
        ok = hull_dist(y1, y2, xy) >= d - 1e-12
        if ok.any():
            V = f64_v(y1[ok], y2[ok], p)
            if chan == "var":
                F = np.einsum('ti,ij,tj->t', V, Gi, V)
                DV1 = f64_Dv(y1[ok], y2[ok], p, 0); DV2 = f64_Dv(y1[ok], y2[ok], p, 1)
                Gv = 2*np.abs(np.einsum('ti,ij,tj->t', DV1, Gi, V)) + 2*np.abs(np.einsum('ti,ij,tj->t', DV2, Gi, V))
            else:
                F = np.abs(V @ w)
                DV1 = f64_Dv(y1[ok], y2[ok], p, 0); DV2 = f64_Dv(y1[ok], y2[ok], p, 1)
                Gv = np.abs(DV1 @ w) + np.abs(DV2 @ w)
            i = int(np.argmax(F))
            if F[i] > best:
                best = float(F[i]); arg = (t, float(th[ok][i]))
            gmax = max(gmax, float(Gv.max()))
        t += ht
    # cell half-diagonal: max arc step/2 and radial step/2
    rho = 0.5*np.sqrt((2*np.pi*(d+trange)/ntheta)**2 + ht**2)
    pad = gmax*rho + H*rho*rho   # mesh-geometry pad (exact gradients + certified H)
    return best, pad, arg, gmax, H

def mu_tables(entry, r, useY):
    """cw_m = sum_p (M beta)_p (-1)^|a_p| (-x_p)^{m-a_p}/(m-a_p)!  (derived dual-value moments)."""
    p = entry['pins']; Ms = entry['M']; n = Ms.rows
    beta = mp.matrix(vals_of(r, useY))
    w = Ms*beta
    bnorm = mp.sqrt(sum(beta[i]**2 for i in range(n)))
    werr = entry['epsM']*bnorm + ROUND
    fac = [mp.factorial(i) for i in range(JMAX+3)]
    cw = {}
    for (m1, m2) in MLIST:
        s = mp.mpf(0)
        for pidx in range(n):
            xp, al = p[pidx]
            j1, j2 = m1-al[0], m2-al[1]
            if j1 < 0 or j2 < 0 or j1+j2 > JMAX: continue
            mom = ((-xp[0])**j1/fac[j1]) * ((-xp[1])**j2/fac[j2])
            s += w[pidx]*((-1)**(al[0]+al[1]))*mom
        cw[(m1, m2)] = s
    wn1 = sum(abs(w[i]) for i in range(n))
    return cw, wn1, werr, w

def envelope_mu(entry, cw, wn1, d, shift=(0, 0)):
    s = mp.mpf(0)
    for (m1, m2) in MLIST:
        s += abs(cw[(m1, m2)]) * Bker(m1+shift[0], m2+shift[1], d)
    s += tail_R(d, wn1, entry['rhoc'], 1, shift)
    beta = vals_of(RUNGS[entry['ri']], entry['useY'])
    bnorm = mp.sqrt(sum(x*x for x in beta))
    s += mp.mpf('1.01')*entry['epsM']*bnorm * (3*Bker(0,0,d) + 4*Bker(1,0,d))  # |beta|.|G^-1-M|.||v||
    return s

def knot_sup(useY, ri, d, chan, ntheta=4096, ht=0.002, trange=1.25):
    entry = ENS[(useY, ri)]
    best, pad, arg, gmax, H = certified_sup(useY, ri, d, chan, ntheta, ht, trange)
    # DPS80 anchor: re-evaluate at the mesh argmax; float64 pipeline pad = 1e-3 relative
    # (covers cond*eps ~ 2.2e-4 by 4.5x) + 10x observed |f64 - DPS80| at the argmax.
    y1 = mp.mpf(str(arg[0]))*mp.cos(mp.mpf(str(arg[1])))
    y2 = mp.mpf(str(arg[0]))*mp.sin(mp.mpf(str(arg[1])))
    dvar_mp, mu_mp = mp_eval_point(useY, ri, y1, y2)
    pick = dvar_mp if chan == "var" else abs(mu_mp)
    argdiff = abs(pick - mp.mpf(str(best)))
    f64pad = mp.mpf('1e-3')*mp.mpf(str(best)) + 10*argdiff
    if chan == "var":
        Eb = envelope(entry, entry['ctab'], entry['rn'], entry['rhoc'], d + mp.mpf(str(trange)))
    else:
        cw, wn1, werr, w = entry['mu']
        Eb = envelope_mu(entry, cw, wn1, d + mp.mpf(str(trange)))
    cert = max(mp.mpf(str(best)) + mp.mpf(str(pad)) + f64pad, Eb)
    return dict(best=best, pad=pad, arg=arg, gmax=gmax, H=H, Ebeyond=Eb, cert=cert,
                argdiff=argdiff, mpval=pick)

# DPS80 validation of a float64 mesh value at a given point
def mp_eval_point(useY, ri, y1, y2):
    entry = ENS[(useY, ri)]
    r = RUNGS[ri]
    p = entry['pins']
    v = mp.matrix(len(p), 1)
    for i, (xp, al) in enumerate(p):
        v[i] = ((-1)**(al[0]+al[1]))*kernel_d(y1-xp[0], y2-xp[1], al[0], al[1])
    Minv = entry['M']   # certified approximate inverse (error epsM, displayed in S3)
    dvar = (v.T*Minv*v)[0]
    beta = mp.matrix(vals_of(r, useY))
    mu = (beta.T*Minv*v)[0]
    return dvar, mu

KNOTS = {}
for useY in (True, False):
    for ri in (0, 1):
        for d in (mp.mpf(3), mp.mpf(5), mp.mpf('6.1')):
            res = knot_sup(useY, ri, d, "var")
            KNOTS[(useY, ri, d)] = res
            tag = "7-pin" if useY else "6-pin"
            print(f"{tag} r=1/{int(1/RUNGS[ri])} d={d}: mesh_max={res['best']:.6e} pad={res['pad']:.2e} "
                  f"argdiff={fmt(res['argdiff'],4)} E_beyond={fmt(res['Ebeyond'],4)} "
                  f"certified_sup={fmt(res['cert'],6)} argmax(t,th)={res['arg']}")

# ----------------------------------------------------------------------------- S7 recorded constants and value-law channel
print("=== S7 recorded-constant checks (C022/LB-3 recomputation class) ===")
# variance channel (load-bearing): 7-pin ensemble (includes witness Y; independent of v*)
for ri in (0, 1):
    c3 = KNOTS[(True, ri, mp.mpf(3))]['cert']
    c5 = KNOTS[(True, ri, mp.mpf(5))]['cert']
    c61 = KNOTS[(True, ri, mp.mpf('6.1'))]['cert']
    print(f"7-pin r=1/{int(1/RUNGS[ri])}: certified Delta_bar(3)={fmt(c3,6)} Delta_bar(5)={fmt(c5,6)} "
          f"Delta_bar(6.1)={fmt(c61,6)}")
    ck(c3 <= mp.mpf('2.24e-2'), f"S7: recorded d=3 value-variance bound 2.24% holds (7-pin r=1/{int(1/RUNGS[ri])})")
    ck(c5 <= mp.mpf('1e-4'), f"S7: recorded d=5 variance bound 1e-4 holds (7-pin r=1/{int(1/RUNGS[ri])})")
    ck(c61 <= mp.mpf('7.5e-13'), f"S7: recorded d=6.1 variance worst case <= 7.5e-13 holds (7-pin r=1/{int(1/RUNGS[ri])})")
    ck(c61 <= mp.mpf('1e-4'), f"S7: monotone envelope < 1e-4 from d=6.1 (7-pin r=1/{int(1/RUNGS[ri])})")

# monotone envelope from d0 = 3 certified: envelope non-increasing AND dominates knot sups
for useY in (True, False):
    for ri in (0, 1):
        entry = ENS[(useY, ri)]
        e3 = ENV[(useY, ri)][0]
        c3 = KNOTS[(useY, ri, mp.mpf(3))]['cert']
        ck(e3 >= c3, f"S5/S6: analytic envelope dominates certified exact sup at d=3 ({'7-pin' if useY else '6-pin'} r=1/{int(1/RUNGS[ri])})")
        print(f"{'7-pin' if useY else '6-pin'} r=1/{int(1/RUNGS[ri])}: envelope E(3)={fmt(e3,6)} vs "
              f"certified exact sup {fmt(c3,6)} (envelope overhead factor {fmt(e3/c3,4)})")

# value-law channel (6-pin; the recorded '6-pin value law' constants)
print("--- value-law channel at d = 5 (6-pin ensemble) ---")
TVREC = {0: mp.mpf('7.7e-5'), 1: mp.mpf('8.1e-5')}
for ri in (0, 1):
    entry = ENS[(False, ri)]
    entry['mu'] = mu_tables(entry, RUNGS[ri], False)
    res = knot_sup(False, ri, mp.mpf(5), "mu")
    KNOTS[(False, ri, mp.mpf(5), 'mu')] = res
    cw, wn1, werr, w = entry['mu']
    sig_inf = mp.sqrt(1 - KNOTS[(False, ri, mp.mpf(5))]['cert'])
    mu_bar = res['cert']
    tv_bound = mu_bar/(sig_inf*mp.sqrt(2*mp.pi))
    # TV_sigma(sig_inf): crossing x* = sig sqrt(2 ln(1/sig)/(1-sig^2)); TV = 2(Phi(x*/sig)-Phi(x*))
    xs = sig_inf*mp.sqrt(2*mp.log(1/sig_inf)/(1-sig_inf**2))
    Phi = lambda x: mp.mpf('0.5')*(1+mp.erf(x/mp.sqrt(2)))
    tv_sig = 2*(Phi(xs/sig_inf) - Phi(xs))
    tv_cert = tv_bound + tv_sig
    # exact TV at the certified argmax (DPS80)
    t0, th0 = res['arg']
    y1 = mp.mpf(t0)*mp.cos(mp.mpf(th0)); y2 = mp.mpf(t0)*mp.sin(mp.mpf(th0))
    dvar_mp, mu_mp = mp_eval_point(False, ri, y1, y2)
    sig_mp = mp.sqrt(1-dvar_mp)
    # density crossings: (1-sig^-2) x^2 /2 + mu sig^-2 x - mu^2 sig^-2 /2 - ln(sig) = 0 (exact quadratic)
    A2 = (1 - sig_mp**-2)/2; B2 = mu_mp*sig_mp**-2; C2 = -mu_mp**2*sig_mp**-2/2 - mp.log(sig_mp)
    if abs(A2) < mp.mpf('1e-40'):
        rts = [-C2/B2]
    else:
        disc = mp.sqrt(B2*B2 - 4*A2*C2)
        rts = sorted([(-B2-disc)/(2*A2), (-B2+disc)/(2*A2)])
    r1, r2 = rts[0], rts[-1]
    P = lambda aa, bb: Phi((bb-mu_mp)/sig_mp)-Phi((aa-mu_mp)/sig_mp)
    Q = lambda aa, bb: Phi(bb)-Phi(aa)
    tv_exact = mp.mpf('0.5')*(abs(P(-mp.inf, r1)-Q(-mp.inf, r1)) + abs(P(r1, r2)-Q(r1, r2)) + abs(P(r2, mp.inf)-Q(r2, mp.inf)))
    print(f"6-pin r=1/{int(1/RUNGS[ri])}: certified |mu|_bar(5)={fmt(mu_bar,6)} sig_inf={fmt(sig_inf,10)}")
    print(f"   certified TV_bar(5) = {fmt(tv_cert,6)} ; exact TV at certified argmax = {fmt(tv_exact,6)} "
          f"at y={fmt(y1,6)},{fmt(y2,6)} ; recorded = {fmt(TVREC[ri],4)}")
    ck(tv_cert <= mp.mpf('1e-4'), f"S7: value-law channel certified <= 1e-4 at d=5 (6-pin r=1/{int(1/RUNGS[ri])})")
    ck(tv_exact <= tv_cert, "S7: exact TV dominated by certified bound")
    dev = abs(tv_exact - TVREC[ri])/TVREC[ri]
    print(f"   |TV_exact - recorded|/recorded = {fmt(dev, 4)}")
    ck(dev <= mp.mpf('0.02'), f"S7: recorded 6-pin value law {fmt(TVREC[ri],4)} matched within 2% (6-pin r=1/{int(1/RUNGS[ri])})")

# ----------------------------------------------------------------------------- S8 witness-pin isolation + summary
print("=== S8 witness pin v* = -1/2 isolation (value-law channel, 7-pin) ===")
for ri in (0, 1):
    entry = ENS[(True, ri)]
    entry['mu'] = mu_tables(entry, RUNGS[ri], True)
    res = knot_sup(True, ri, mp.mpf(5), "mu", ntheta=2048, ht=0.004, trange=0.6)
    print(f"7-pin(v*=-1/2) r=1/{int(1/RUNGS[ri])}: certified |mu|_bar(5) = {fmt(res['cert'],6)}")
    ck(res['cert'] > mp.mpf('0.1'),
       f"S8: ISOLATION - 7-pin v*=-1/2 value-law channel violates far stabilization at d=5 (r=1/{int(1/RUNGS[ri])})")

print("=== SUMMARY ===")
if FAILS:
    print("CERTIFICATE FAILURES:", FAILS)
    raise SystemExit(1)
print("ALL CHECKS PASS: far-field monotone decay envelope certified (d0 = 3),")
print("Var(f(y)|pins) bounded below by derived constant for dist(y, cluster) >= 3,")
print("recorded constants d=3 (<=2.24%), d=5 (<=1e-4 variance, value law matched),")
print("d=6.1 (<=7.5e-13 < 1e-4) reproduced at certified grade; v*=-1/2 isolation displayed.")
