# LPW_CONSTANT certifier — converts the endorsed LPW-CAND-20260912 existential
# constant expression into a CERTIFIED EXPLICIT lower bound, per the acceptance
# contract 03_CONSTANT_CERTIFICATE_CONTRACT.md (LPW_Review_Reconciliation_2026-09-12).
#
# Exact object: normalized periodized side-24 2D Bargmann-Fock field.
#   Spectral form: frequencies k in (pi/12) Z^2, weight w(k) = exp(-|k|^2/2),
#   Z-normalized; Cov(f(z1),f(z2)) = (1/Z) sum_k w(k) cos(k.(z1-z2)).
#   This equals the candidate's real-space K_24 by Poisson summation (checked).
# Conditioning: six-pin continuous Gaussian regression version, Hermite-normalized
#   U_r -> U_0 = (f, f_x, f_y, f_xx/2, f_xy, f_xxx/6),
#   J = (f_yy, f_xxy/2, f_xyy/2, f_yyy/6).  b = 6/5.
#
# Method (contract): exact finite-torus moments a2,a4,a6,... (never planar
# rationals); endpoint eigenfloor of record 31/250 (02_ENDPOINT_SUPPLEMENT.md,
# author-side PROVED, lead-verified) used at r=0; explicit radius R certified by
# an INTERVAL modulus ||Gamma_r - Gamma_0||_F <= L1 * R from analytic profile
# derivative bounds (no sampled rungs); density floor by the contract's boxed
# formula on an explicitly tightened compact set; B3/B4 by contract section 4.
#
# Fail-closed: any violated check prints FAIL and exits nonzero (no bare asserts).
# Deterministic: no randomness, no wall clock, no environment-dependent output.
# Mutation hooks (LPW_MUTATE env var) exist ONLY so falsify.py can demonstrate
# that specific defects are rejected with nonzero exit.

import os
import sys
from mpmath import mp, mpf, pi, exp, sqrt, cos, sin, matrix, fabs, nstr, eigsy

MUTATE = os.environ.get("LPW_MUTATE", "")

mp.dps = 60

NFAIL = [0]
def ck(cond, msg):
    if not cond:
        NFAIL[0] += 1
        print("FAIL:", msg)
        print("CERTIFICATE REJECTED")
        raise SystemExit(1)

def note(msg):
    print(msg)

# ---------------------------------------------------------------- parameters
h  = pi/12                      # spectral lattice spacing (exact object)
a  = h*h/2                      # weight exp(-a |n|^2) per coordinate
NB = 70                         # 1D spectral truncation |n| <= NB; tails from NB+1
bb = mpf(6)/5                   # b = 6/5
delta = mpf(1)/1024             # contract delta
R  = mpf(1)/100000              # chosen certified radius R = 1e-5 (see report)
LAM_END = mpf(31)/250           # endpoint eigenfloor of record (supplement sec.3)
ROUND = mpf('1e-45')            # blanket per-quantity rounding allowance at 60 dps
                                # (all quantities <= ~1e6, <= ~1e5 elementary ops;
                                #  mpf eps = 2^-199 ~ 1.2e-60; 1e-45 is generous)

note("LPW_CONSTANT certifier (contract 03, reconciliation 2026-09-12)")
note("candidate pinned: 02_LOCAL_PATH_LOWER_BOUND_CANDIDATE.md full-file sha256 "
     "cf58f72eb0399c626160c143b50f674ff3bd5c449225211edd6fd378a374e1a5")
note("addenda pinned: 02_ENDPOINT_SUPPLEMENT.md, 03_CONSTANT_CERTIFICATE_CONTRACT.md "
     "(hashes in RECEIPT.json)")
note(f"parameters: NB={NB}, dps={mp.dps}, R={nstr(R,4)}, delta=1/1024, b=6/5, "
     f"lambda_endpoint=31/250, ROUND={nstr(ROUND,3)}")

# -------------------------------------------------- 1D lattice engine + tails
ns  = list(range(-NB, NB+1))
w1  = [exp(-a*n*n) for n in ns]
k1s = [h*n for n in ns]
Z1  = sum(w1)
Z   = Z1*Z1

# 1D moment tail: sum_{|n| >= NB+1} e^{-a n^2} (h|n|)^p
# terms decrease for n >= NB+1 when 2 a (NB+1) > p/(NB+1)*... checked below;
# ratio_{n} <= e^{-a(2n+1)} (1+1/n)^p, bounded by rho = e^{-a(2N0+1)} (1+1/N0)^p.
def tail1d(p):
    N0 = NB+1
    ck(2*a*N0*N0 > p, f"tail decrease condition fails for p={p}")
    rho = exp(-a*(2*N0+1)) * (1 + mpf(1)/N0)**p
    ck(rho < mpf('0.5'), f"tail ratio not < 1/2 for p={p}: {nstr(rho,4)}")
    first = 2 * exp(-a*N0*N0) * (h*N0)**p
    return first/(1-rho)

TMOM = {p: tail1d(p) for p in range(0, 13)}
note("certified 1D spectral tails sum_{|n|>=%d} e^{-a n^2}(h|n|)^p:" % (NB+1))
note("  " + "  ".join(f"p={p}: <= {nstr(TMOM[p],3)}" for p in (0,2,4,6,8,10,12)))

# exact 1D even moments ME[2m] = (1/Z1) sum w1 (h n)^{2m}, with certified error
ME, ME_ERR = [], []
for m in range(7):
    p = 2*m
    val = sum(w*(k**p) for w, k in zip(w1, k1s))/Z1
    err = (TMOM[p] + TMOM[0]*val)/Z1 + ROUND     # numerator tail + Z1 tail + rounding
    ME.append(val); ME_ERR.append(err)
note("exact finite-torus even moments a_{2m} = ME[2m] (with certified +-err):")
note("  " + "  ".join(f"a{2*m}={nstr(ME[m],12)}(+/-{nstr(ME_ERR[m],2)})"
                      for m in range(1, 5)))

# Poisson / normalization cross-checks (exact object identities)
errZ = fabs(Z - 288/pi)
ck(errZ < 2*TMOM[0] + ROUND, f"Poisson identity fails: |Z - 288/pi| = {nstr(errZ,3)}")
note(f"Poisson check: |Z - 288/pi| = {nstr(errZ,3)} <= {nstr(2*TMOM[0]+ROUND,3)} OK")

# a2 deviation from planar 1 (exact finite-torus correction), at 150 dps
with mp.workdps(150):
    hh = pi/12; aa = hh*hh/2
    NB2 = 200
    Z1b = sum(exp(-aa*n*n) for n in range(-NB2, NB2+1))
    a2b = sum(exp(-aa*n*n)*(hh*n)**2 for n in range(-NB2, NB2+1))/Z1b
    dev_spec = 1 - a2b
    # supplement formula: 1 - a2 = 576 sum n^2 e^{-288 n^2} / sum e^{-288 n^2}
    S0 = sum(exp(-288*mpf(n)*n) for n in range(-50, 51))
    dev_real = 576*sum(n*n*exp(-288*mpf(n)*n) for n in range(-50, 51))/S0
    dd = fabs(dev_spec - dev_real)
    note(f"a2 deviation (150 dps): 1-a2 = {nstr(dev_spec,12)} ; "
         f"576-sum form = {nstr(dev_real,12)} ; |diff| = {nstr(dd,3)}")
    if MUTATE == "planar_moments":
        dev_spec = mpf(0)   # planar substitution: deviation forced to zero
    ck(dev_spec > 0, "planar-moment substitution detected: 1-a2 is not positive "
                     "(exact finite-torus moments required)")
    ck(dev_spec < mpf('1e-100'), f"a2 deviation larger than expected: {nstr(dev_spec,4)}")
    ck(dd < mpf('1e-130'), f"a2 spectral/real-space identity mismatch: {nstr(dd,3)}")
    note("a2 = 1 - (positive correction ~1e-122) < 1 certified; "
         "spectral/real-space forms agree")

a2, a4, a6, a8 = ME[1], ME[2], ME[3], ME[4]

# ------------------------------------------------ Hermite profiles (exact r-dependence)
# The Hermite combinations U_r act on each spectral mode e^{ik.z} by
#   i^{e_i} k1^{A_i} k2^{B_i} g_i(theta),  theta = k1 r/2,
# so every covariance entry of Gamma_r = Cov((U_r, J)) is the exact 1D sum
#   Cov_ij(r) = [e_i==e_j] ME[B_i+B_j] (1/Z1) sum_n w1 k1^{A_i+A_j} g_i g_j,
# smooth down to r = 0 (no 1/r singularities; cross-validated against the
# displayed T_r matrix route to <= 6e-51 at four rungs, see falsify.py).
APOW = [(0,0),(1,0),(0,1),(2,0),(1,1),(3,0),(0,2),(2,1),(1,2),(0,3)]
EPAR = [0,1,1,0,0,1,0,1,1,1]

def gfun(i, th):
    if i == 0: return cos(th) + (th/2)*sin(th)
    if i == 1:
        if th == 0: return mpf(1)
        return 3*sin(th)/(2*th) - cos(th)/2
    if i == 2: return cos(th)
    if i == 3:
        if th == 0: return mpf(-1)/2
        return -sin(th)/(2*th)
    if i == 4:
        if th == 0: return mpf(-1)
        return -sin(th)/th
    if i == 5:
        if th == 0: return mpf(-1)/6
        return cos(th)/(2*th*th) - sin(th)/(2*th*th*th)
    if i == 6: return mpf(-1)
    if i == 7: return mpf(-1)/2
    if i == 8: return mpf(-1)/2
    if i == 9: return mpf(-1)/6
    raise SystemExit("bad profile index")

def cov_ij(i, j, r):
    if EPAR[i] != EPAR[j]: return mpf(0)
    A = APOW[i][0] + APOW[j][0]
    B = APOW[i][1] + APOW[j][1]
    if B % 2 == 1: return mpf(0)
    hr2 = h*r/2
    s = mpf(0)
    for n, w, k in zip(ns, w1, k1s):
        th = hr2*n
        s += w*(k**A)*gfun(i, th)*gfun(j, th)
    return ME[B//2]*s/Z1

def build_Gamma(r):
    G = matrix(10, 10)
    for i in range(10):
        for j in range(i, 10):
            G[i, j] = cov_ij(i, j, r)
            G[j, i] = G[i, j]
    return G

# entry error for Cov_ij(r): numerator tail bound
#   |(1/Z1) sum_{|n|>NB} w1 k1^A g_i g_j| with |g_i| <= pb_i(|k1|),
# plus Z1 tail and rounding.  pb bounds are below (theta <= |k1| R / 2).
# For r=0 endpoint values use the r<=1 profile bounds (theta <= |k1|/2).
def pb(i, x, thcap):   # |g_i(theta)| <= pb_i(|k1|=x) for |theta| <= x*thcap
    if i == 0: return 1 + x*thcap/2
    if i == 1: return mpf(2)
    if i == 2: return mpf(1)
    if i == 3: return mpf(1)/2
    if i == 4: return mpf(1)
    if i == 5: return mpf(1)/4
    if i == 6: return mpf(1)
    if i == 7: return mpf(1)/2
    if i == 8: return mpf(1)/2
    if i == 9: return mpf(1)/6
    raise SystemExit("bad profile index")

def pb1(i, x, thcap):  # |g_i'(theta)| <= pb1
    if i == 0: return mpf(1)/2 + x*thcap/2
    if i == 1: return mpf(5)/4
    if i == 2: return mpf(1)
    if i == 3: return mpf(1)/4
    if i == 4: return mpf(1)/2
    if i == 5: return mpf(5)/4
    return mpf(0)

def tailabs(q):
    # sum_{|n|>NB} w1 |k1|^q, upward (CS for odd q)
    if q % 2 == 0:
        return TMOM[q]
    return sqrt(TMOM[q-1]*TMOM[q+1])

def pb_full_poly(i, j, thcap):
    # polynomial in x = |k1| bounding |g_i(theta) g_j(theta)|, |theta| <= x thcap
    p0 = {0: pb(i, 0, thcap)} if i != 0 else {0: mpf(1), 1: thcap/2}
    p1 = {0: pb(j, 0, thcap)} if j != 0 else {0: mpf(1), 1: thcap/2}
    return polymul(p0, p1)

def polymul(p, q):
    out = {}
    for a_, ca in p.items():
        for b_, cb in q.items():
            out[a_+b_] = out.get(a_+b_, mpf(0)) + ca*cb
    return out

def polyadd(p, q):
    out = dict(p)
    for b_, cb in q.items():
        out[b_] = out.get(b_, mpf(0)) + cb
    return out

def absmom(q):
    # (1/Z1) sum w1 |k1|^q, upward bound with certified error
    if q % 2 == 0:
        v, e = ME[q//2], ME_ERR[q//2]
    else:
        v = sqrt(ME[(q-1)//2]*ME[(q+1)//2])       # Cauchy-Schwarz
        e = v*(ME_ERR[(q-1)//2]/ME[(q-1)//2] + ME_ERR[(q+1)//2]/ME[(q+1)//2]) + ROUND
    return v + e   # returned already rounded UPWARD

def entry_tail(i, j, thcap):
    # certified tail+round error for the Cov_ij lattice sum:
    # (ME[B]/Z1) * sum_{|n|>NB} w1 |k1|^A |g_i g_j|  +  Z1-tail share + rounding
    A = APOW[i][0] + APOW[j][0]
    B = APOW[i][1] + APOW[j][1]
    if EPAR[i] != EPAR[j] or B % 2 == 1: return mpf(0)
    poly = pb_full_poly(i, j, thcap)
    tv = mpf(0)
    full = mpf(0)
    for pw, cf in poly.items():
        tv += cf*tailabs(A+pw)
        full += cf*absmom(A+pw)
    return (ME[B//2]+ME_ERR[B//2])*(tv + full*(TMOM[0]/Z1))/Z1 + ROUND

# ------------------------------------------------ endpoint r = 0 (exact)
G0 = build_Gamma(mpf(0))
E0, Q0 = eigsy(G0)
Lam0 = matrix(10, 10)
for i in range(10): Lam0[i, i] = E0[i]
Res0 = G0 - Q0*Lam0*Q0.T
fro0 = sqrt(sum(Res0[i, j]**2 for i in range(10) for j in range(10)))
e_tail0 = sqrt(mpf(100)*sum(entry_tail(i, j, mpf('0.5'))**2
                             for i in range(10) for j in range(10)))
lam0_num = min(E0)
lam0_lo = lam0_num - fro0 - e_tail0 - ROUND
note(f"endpoint: lambda_min(Gamma_0) = {nstr(lam0_num,15)}")
note(f"  eig Frobenius residual = {nstr(fro0,3)}, spectral+entry tail = {nstr(e_tail0,3)}")
note(f"  independently certified endpoint floor: {nstr(lam0_lo,15)}")

if MUTATE == "eigenfloor_up":
    lam_used = lam0_num + mpf('1.5e-10')   # upward-rounded floor (defect)
else:
    lam_used = LAM_END
ck(lam_used <= lam0_lo, f"endpoint eigenfloor of record {nstr(lam_used,15)} is NOT "
   f"certified (independent floor {nstr(lam0_lo,15)}); upward rounding rejected")
ck(LAM_END <= lam0_lo, "supplement eigenfloor 31/250 inconsistent with this engine")
note(f"  endpoint floor USED in chain: {nstr(lam_used,15)} "
     f"(record 31/250 = {nstr(LAM_END,6)}; mutation hook: {MUTATE or 'off'})")

# endpoint Schur/mean identities in the exact moments (supplement sec.1)
GU0 = G0[:6, :6]; GX0 = G0[:6, 6:]; CJ0 = G0[6:, 6:]
S0 = CJ0 - GX0.T*(GU0**-1)*GX0
u0 = matrix([bb, 0, 0, 0, 0, mpf(1)/3])
mu0 = GX0.T*(GU0**-1)*u0
S0_formula = [a4 - a2*a2, a2*(a4-a2*a2)/4, a2*(a4-a2*a2)/4,
              (a6 - a4*a4/a2)/36]
dS = max(fabs(S0[i, i] - S0_formula[i]) for i in range(4))
dSo = max(fabs(S0[i, j]) for i in range(4) for j in range(4) if i != j)
dm = max(fabs(mu0[0] + a2*bb), fabs(mu0[1]), fabs(mu0[2]), fabs(mu0[3]))
ck(dS < mpf('1e-50'), f"endpoint Schur diagonal identity fails: {nstr(dS,3)}")
ck(dSo < mpf('1e-50'), f"endpoint Schur off-diagonal vanishing fails: {nstr(dSo,3)}")
ck(dm < mpf('1e-50'), f"endpoint conditional-mean identity fails: {nstr(dm,3)}")
note(f"endpoint identities (exact moments): S0 vs diag(a4-a2^2, a2(a4-a2^2)/4,"
     f" ..., (a6-a4^2/a2)/36): max dev {nstr(dS,3)}; off-diag {nstr(dSo,3)};")
note(f"  E[J|U_0=u_0] vs (-a2 b,0,0,0): max dev {nstr(dm,3)}  "
     f"(mu0[0]={nstr(mu0[0],12)})")

# -------------------------------- interval modulus: ||Gamma_r - Gamma_0||_F <= L1 R
# Certified uniform bounds on |d/dr Cov_ij(r)| for r in [0,R] via analytic
# profile derivative bounds (theta-cap |theta| <= |k1| R/2).  These are global
# interval/modulus certificates, NOT sampled rungs.
def pbpoly(i, thcap):   # polynomial bound for |g_i| in x = |k1|
    if i == 0: return {0: mpf(1), 1: thcap/2}
    return {0: pb(i, 0, thcap)}

def pb1poly(i, thcap):  # polynomial bound for |g_i'|
    if i == 0: return {0: mpf(1)/2, 1: thcap/2}
    return {0: pb1(i, 0, thcap)}

THC = R/2
C1 = {}
for i in range(10):
    for j in range(10):
        if EPAR[i] != EPAR[j]:
            C1[(i, j)] = mpf(0); continue
        A = APOW[i][0] + APOW[j][0]
        B = APOW[i][1] + APOW[j][1]
        if B % 2 == 1:
            C1[(i, j)] = mpf(0); continue
        poly = polyadd(polymul(pb1poly(i, THC), pbpoly(j, THC)),
                       polymul(pbpoly(i, THC), pb1poly(j, THC)))
        s = mpf(0); sd = mpf(0)
        for pw, cf in poly.items():
            s += cf*absmom(A+1+pw)
            sd += cf*tailabs(A+1+pw)
        C1[(i, j)] = (ME[B//2]+ME_ERR[B//2])*(s + (sd + s*(TMOM[0]/Z1))/Z1)/2 + ROUND

L1  = sqrt(sum(C1[(i, j)]**2 for i in range(10) for j in range(10))) + ROUND
LGU = sqrt(sum(C1[(i, j)]**2 for i in range(6) for j in range(6))) + ROUND
LGX = sqrt(sum(C1[(i, j)]**2 for i in range(6) for j in range(6, 10))) + ROUND
note(f"interval modulus (r in [0,R]): ||dGamma/dr||_F <= L1 = {nstr(L1,8)}")
note(f"  block moduli: L_GU = {nstr(LGU,8)}, L_GX = {nstr(LGX,8)}")
note(f"  => ||Gamma_r - Gamma_0||_F <= L1*R = {nstr(L1*R,6)} for r in [0,{nstr(R,4)}]")

lam = lam_used - L1*R - ROUND
ck(lam > 0, "certified covariance floor lam = lam_endpoint - L1*R is not positive; "
            "shrink R")
note(f"CERTIFIED: lambda_min(Gamma_r) >= lam = {nstr(lam,12)} for all r in [0,R]")
note(f"  (endpoint {nstr(lam_used,12)} minus modulus {nstr(L1*R,4)}; "
     f"Schur/6-pin blocks inherit the same floor by S^-1 = (Gamma^-1)_JJ "
     f"and Cauchy interlacing)")

# diagonal / trace / norm sups over [0,R] (interval bounds, no sampling)
diag_sup = [fabs(G0[j, j]) + C1[(j, j)]*R + ROUND for j in range(10)]
tr10_sup = sum(diag_sup) + ROUND
tr6_sup  = sum(diag_sup[:6]) + ROUND
GUfro2_sup = sum((fabs(G0[i, j]) + C1[(i, j)]*R + ROUND)**2
                 for i in range(6) for j in range(6))
GX0F = sqrt(sum(G0[i, j]**2 for i in range(6) for j in range(6, 10))) + ROUND
note(f"sup diag Gamma_jj: max = {nstr(max(diag_sup),8)}; "
     f"tr6_sup = {nstr(tr6_sup,8)}, tr10_sup = {nstr(tr10_sup,8)}")
note(f"||GX_0||_F = {nstr(GX0F,8)}; sup ||GU_r||_F^2 = {nstr(GUfro2_sup,8)}")

# sup over [0,R] of |u_r| (exact): |u_r|^2 <= b^2 + 1/16 + 1/9 = 5809/3600
u_sup = sqrt(mpf(5809)/3600)
ck(fabs(u_sup*u_sup - (bb*bb + mpf(1)/16 + mpf(1)/9)) < mpf('1e-40'),
   "u_sup rational identity fails")
du_sup = R*R*sqrt(mpf(1)/16 + R*R/144)          # |u_r - u_0| <= du_sup
# sup over [0,R] of |mu_r|, mu_r = E[J | U_r = u_r] (conditional mean):
M_mean = (a2+ME_ERR[1])*bb \
    + (LGX*R/lam + GX0F*LGU*R/(lam_used*lam))*u_sup \
    + (GX0F/lam_used)*du_sup + 10*ROUND
note(f"sup |mu_r| over [0,R]: M = {nstr(M_mean,10)} "
     f"(|mu_0| = a2*b = {nstr((a2+ME_ERR[1])*bb,8)}, interval correction displayed)")

# -------------------------------- tightened compact set (contract item 1)
# E_r: |q/(2r)+5|<=delta, |A-2|<=delta, |B|<=delta, |D3|<=delta.
# For 0 < r <= R the union of the thin boxes is contained in
#   JBOX(R) = [-(10+2 delta) R, 0] x [2-delta, 2+delta] x [-delta, delta]^2.
# This set is stated explicitly; ALL dependent bounds (density floor, B4,
# conditional means) are redone on it.  Containment check (exact rationals):
from fractions import Fraction as Fr
dF = Fr(1, 1024); RF = Fr(1, 100000)
q_lo_needed = -(10+2*dF)*RF          # box lower endpoint
q_lo_actual = -10*RF - 2*RF*dF       # worst thin-box endpoint at r = R
ck(q_lo_actual >= q_lo_needed, "thin-box containment fails at r=R (lower)")
ck(-10*RF + 2*RF*dF < 0, "thin-box containment fails (sign of q)")
note("compact set (CHANGED from candidate's K_J, stated explicitly):")
note(f"  JBOX(R) = [-(10+2d)R, 0] x [2-d, 2+d] x [-d, d]^2, d=1/1024, R=1e-5;")
note(f"  contains every thin box E_r for 0<r<=R (exact rational containment OK)")
RJ2 = ((10+2*delta)*R)**2 + (2+delta)**2 + 2*delta*delta
RJ = sqrt(RJ2) + ROUND
note(f"  R_J = sup_(j in JBOX) |j| = {nstr(RJ,10)} (R_J^2 = {nstr(RJ2,10)})")

# -------------------------------- density floor (contract boxed formula)
# m = (2pi)^-2 Lambda^-2 exp(-(R_J+M)^2/(2 lambda)),
#   lambda I <= Sigma_r <= Lambda I, |mu_r| <= M, |j| <= R_J.
# Lambda: Gershgorin upper bound for lambda_max(Cov(J)) >= lambda_max(S_r)
# (Schur complement is Loewner-below the J covariance, which is r-independent).
grow = [a4+ME_ERR[2],
        (a2+ME_ERR[1])*(a4+ME_ERR[2])/4 + (a2+ME_ERR[1])*(a4+ME_ERR[2])/12,
        (a2+ME_ERR[1])*(a4+ME_ERR[2])/4,
        (a6+ME_ERR[3])/36 + (a2+ME_ERR[1])*(a4+ME_ERR[2])/12]
LAMBDA = max(grow) + ROUND
ck(grow[0] >= max(grow[1:]), "Gershgorin row 0 is not maximal; Lambda logic fails")
note(f"Lambda (Gershgorin upper, lambda_max(Cov J)) = {nstr(LAMBDA,10)}")

QEXP = (RJ + M_mean)**2/(2*lam) + ROUND     # upper enclosure of exponent
m_est = (2*pi)**-2 * LAMBDA**-2 * exp(-QEXP)
# outward enclosures: (2pi)^-2 > 1/40 exactly (since (2pi)^2 < 40)
ck((2*pi)**2 < 40, "(2 pi)^2 < 40 fails; rational enclosure invalid")
# power-of-ten certified floor:  log10(m) >= -(L10), L10 upward-enclosed
from mpmath import log10 as mlog10
L10 = 2*mlog10(2*pi) + 2*mlog10(LAMBDA) + QEXP*mlog10(exp(1)) + mpf('1e-30')
import math as _math
Nm = int(_math.floor(L10)) + 1
ck(mpf(Nm) > L10, "power-of-ten floor for m misaligned")
m_floor = mpf(10)**(-Nm)
ck(m_floor < m_est, "10^-N floor exceeds the estimate; enclosure error")
note(f"density exponent (R_J+M)^2/(2 lam) <= {nstr(QEXP,10)}")
note(f"CERTIFIED density floor on JBOX(R), all r in [0,R]:")
note(f"  m ~= {nstr(m_est,6)}  with certified rational floor  m >= 10^(-{Nm})")

# cross-check against review evidence (RB thin-box center diagnostic)
RB_DIAG = mpf('1.2832e-3')
ck(m_est < RB_DIAG, "m must not exceed the review's center diagnostic 1.2832e-3")
note(f"  cross-check: m <= 1.2832e-3 (review center diagnostic) OK; "
     f"m > 0 certified")

# thin-box volume ledger (exact rationals) and mutation hook
if MUTATE == "box_width":
    volF = (2*dF)**4                 # DEFECT: omits the shrinking width 4 d r
else:
    volF = (4*dF*RF)*(2*dF)**3
ck(volF == 32*dF**4*RF, "thin-box volume is not 32 delta^4 r (rare-box width error)")
note("thin-box volume ledger: vol(E_r) = (4 d r)(2 d)^3 = 32 d^4 r  "
     "(coefficient 32 verified exactly)")

# -------------------------------- global field C^k moments (full Fourier tail)
# ||f||_{C^p} <= Y_p = sum_k sqrt(w_k/Z) (1+|k|)^p zeta_k,
#   zeta_k = |xi_k| + |eta_k|, E zeta = 2 sqrt(2/pi), E zeta^4 = 12 + 16/pi.
def tail2d(p):
    N0 = NB+1
    # sum over shells max|n_i| = s >= N0: 8s e^{-a s^2/2} (1 + h sqrt2 s)^p
    ck(a*N0*N0 > p, f"2D tail decrease condition fails p={p}")
    def term(s):
        return 8*s*exp(-a*s*s/2)*(1 + h*sqrt(2)*s)**p
    t0v = term(N0)
    # ratio of consecutive shell terms
    rho = term(N0+1)/t0v
    ck(rho < mpf('0.9'), f"2D tail ratio not < 0.9 for p={p}: {nstr(rho,4)}")
    return t0v/(1-rho)

S = {}
for p in (3, 4):
    acc = mpf(0)
    for n1 in ns:
        w1n = w1[n1+NB]; k1 = k1s[n1+NB]
        for n2 in ns:
            w = w1n*w1[n2+NB]
            rk = sqrt(k1*k1 + k1s[n2+NB]**2)
            acc += sqrt(w)*(1+rk)**p
    S[p] = acc/sqrt(Z)
    S_err = tail2d(p)/sqrt(Z) + S[p]*(TMOM[0]/Z1) + ROUND
    note(f"S_{p} = sum sqrt(w/Z) (1+|k|)^{p} = {nstr(S[p],10)} "
         f"(+/- tail {nstr(S_err,3)})")
    S[p] += S_err   # round UPWARD (used in upper bounds)

mu1 = 2*sqrt(2/pi) + ROUND
mu4 = 12 + 16/pi + ROUND
EfC3_4 = mu4*S[3]**4          # E||f||_{C^3}^4 <= mu4 S_3^4
EfC4   = mu1*S[4]             # E||f||_{C^4}   <= mu1 S_4
note(f"E||f||_C3^4 <= mu4 S_3^4 = {nstr(EfC3_4,8)}  (mu4 = 12+16/pi = {nstr(mu4,8)})")
note(f"E||f||_C4   <= mu1 S_4   = {nstr(EfC4,8)}  (mu1 = 2 sqrt(2/pi) = {nstr(mu1,8)})")

# -------------------------------- B3 and B4 (contract section 4)
# A_k = sup_{r,x,|alpha|<=k} ||d^alpha C_r(x)||_2 * sup_r ||Gamma_r^{-1}||_2
#      <= sqrt(M_{2k} * tr_sup) / lam,
# M_{2k} = max_{|alpha|<=k} E|d^alpha f|^2 = max ME[a1] ME[a2] (exact moments).
def Mmax(k):
    best = mpf(0)
    for a1 in range(k+1):
        for a2_ in range(k+1-a1):
            v = (ME[a1]+ME_ERR[a1])*(ME[a2_]+ME_ERR[a2_])
            if v > best: best = v
    return best + ROUND
M3B, M4B = Mmax(3), Mmax(4)
note(f"M_6 = max_(|a|<=3) E|d^a f|^2 = {nstr(M3B,8)}; "
     f"M_8 = max_(|a|<=4) = {nstr(M4B,8)} (exact finite-torus moments)")

A3 = sqrt(M3B*tr6_sup)/lam + ROUND
A4 = sqrt(M4B*tr10_sup)/lam + ROUND
note(f"A_3 = {nstr(A3,8)} (6-pin),  A_4 = {nstr(A4,8)} (10-jet)")

EU4 = tr6_sup**2 + 2*GUfro2_sup            # E||U_r||^4 = (tr GU)^2 + 2||GU||_F^2
EU4q = EU4**mpf('0.25')
EV10 = sqrt(tr10_sup)                       # E||V10_r|| <= sqrt(tr Gamma)
v_sup = sqrt(u_sup**2 + RJ2) + ROUND        # sup ||(u_r, j)|| over [0,R] x JBOX
note(f"(E||U||^4)^(1/4) <= {nstr(EU4q,8)};  E||V10|| <= {nstr(EV10,8)}; "
     f"sup||(u_r,j)|| <= {nstr(v_sup,8)}")

B3 = (1 + EfC3_4**mpf('0.25') + A3*(EU4q + u_sup))**4
B4 = EfC4 + A4*(EV10 + v_sup)
note(f"B_3 = [1 + (E||f||_C3^4)^(1/4) + A_3((E||U||^4)^(1/4)+sup|u_r|)]^4")
note(f"    <= {nstr(B3,10)}")
note(f"B_4 = E||f||_C4 + A_4(E||V10|| + sup||(u_r,j)||) <= {nstr(B4,10)}")
ck(B3 > 0 and B3 < mpf('1e300'), "B3 not finite positive")
ck(B4 > 0 and B4 < mpf('1e300'), "B4 not finite positive")

# integer outward enclosures (exact rational arithmetic from here on)
B3_rat = int(_math.ceil(B3))
K = max(mpf(1), 2*B4)
ck(2*B4 > 1, "K = max(1, 2 B4) branch check")
K_rat = int(_math.ceil(2*B4))
CZ_rat = 4*B3_rat
r0 = Fr(1, 256*K_rat)                       # rounded DOWNWARD (K_rat >= K)
ck(Fr(1, 256*K_rat) <= 1/Fr(256)/Fr(int(_math.floor(K))),
   "r0 downward rounding invalid")
ck(r0 <= RF and r0 <= 1, "r0 must be <= min(R,1); raise R or check K")
note(f"K = max(1, 2 B_4) <= K_rat = {K_rat}")
note(f"C_Z = 4 B_3 <= {CZ_rat}")
note(f"CERTIFIED radius: r_0 = 1/(256 K_rat) = 1/{256*K_rat} "
     f"~= {nstr(mpf(r0.numerator)/r0.denominator,6)}  (> 0, exact rational)")

# -------------------------------- power/weight ledger (exact rationals)
if MUTATE == "hessian_power":
    HESS_POW = 3                            # DEFECT: wrong Hessian scaling power
else:
    HESS_POW = 4
ck(HESS_POW == 4, "Hessian pair-weight scaling is r^4 (r^2 per determinant)")
ck(Fr(81, 16)*Fr(1673, 128) >= 65, "event weight base 65 not verified")
ck(65*16 == 1040 and 1040 == 4*260, "weight/Markov ledger 65*16=1040=4*260 fails")
ck(1 + HESS_POW - 2 == 3, "final exponent ledger 1+4-2=3 fails")
ck(4 + 4 + 2 + 6 == 16, "jet-box basis-norm sum 16 fails")
ck(Fr(2089, 384) < 8, "Taylor lift total 2089/384 < 8 fails")
ck(Fr(1, 6) - Fr(99, 1280) - Fr(1, 16) == Fr(103, 3840) > 0,
   "path clearance 103/3840 > 0 fails")
hard = 16*dF + 8*K_rat*r0
ck(hard <= Fr(3, 64) and Fr(3, 64) < Fr(1, 16),
   f"hard geometric check 16 d + 8 K r0 <= 3/64 < 1/16 fails: {hard}")
note("ledger (exact rationals): W >= 65 r^4 on event; vol 32 d^4 r; "
     "Z_r <= C_Z r^2; exponent 1+4-2 = 3; 16d + 8 K r0 = "
     f"{hard} <= 3/64 < 1/16; clearance 103/3840 > 0")

# -------------------------------- final constant c
# c = 1040 m delta^4 / C_Z = 260 m delta^4 / B_3, delta^4 = 2^-40 exactly.
ck(dF**4 == Fr(1, 2**40), "delta^4 = 2^-40 fails")
c_est = 260*m_est*delta**4/B3
c_lo = Fr(260, 1) / (Fr(B3_rat, 1) * (2**40) * (10**Nm))
# certified power-of-ten floor for c:
c_est_lo = 260*m_floor*delta**4/B3_rat
log_c = mlog10(c_est_lo)
Nc = int(_math.floor(log_c))
ck(mpf(Nc) < log_c, "power-of-ten floor for c misaligned")
ck(Fr(10)**Nc < c_lo, "certified c floor exceeds c_lo")
note(f"CERTIFIED CONSTANT: c = 260 m delta^4 / B_3")
note(f"  estimate c ~= {nstr(c_est,6)}")
note(f"  certified: c >= 260/(B3_rat 2^40) * 10^(-{Nm}) >= 10^({Nc})  with")
note(f"  B3_rat = {B3_rat};  in particular c > 0.")
ck(c_lo > 0, "c is not positive")

note("")
note("=================== CERTIFIED SUMMARY ===================")
note(f"1 - q(r, 6/5) >= c r^3  for all 0 < r <= r_0, with")
note(f"  c  >= 10^({Nc})   (estimate {nstr(c_est,4)}; rational form above)")
note(f"  r_0 = 1/{256*K_rat} ~= {nstr(mpf(r0.numerator)/r0.denominator,6)}")
note(f"  m  >= 10^(-{Nm})  (density floor on JBOX(R), estimate {nstr(m_est,4)})")
note(f"  lam = {nstr(lam,10)} (uniform covariance floor on [0,R])")
note(f"  B_3 <= {B3_rat},  B_4 <= {nstr(B4,6)} (K_rat = {K_rat}),  R = 1e-5")
note("ALL_CHECKS_PASSED")
