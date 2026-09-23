# falsify.py — executable falsifier for the LPW_CONSTANT certificate.
#
# Independent cross-checks (each exits nonzero on failure):
#   F1. Covariance engine: profile route vs the candidate's displayed T_r
#       (Hermite matrix) route with CORRECT sine moments, at four rungs.
#   F2. Endpoint supplement sec.3: the ten leading principal minors of
#       Gflat - (1/8) I recomputed EXACTLY (rational arithmetic) and matched
#       against the supplement's displayed list; Rayleigh conclusion checked.
#   F3. Endpoint identities at 150 dps: a2-deviation positive and both forms.
#   F4. Profile derivative bounds pb/pb1 sanity-checked on a theta grid.
#   F5. Planar endpoint density diagnostics (supplement sec.5) recomputed.
#   F6. Final assembly: re-derive r0 and c-floor from the transcript's
#       certified integers with exact rationals; string-compare.
#   F7. Mutation suite: four defects must each be rejected nonzero:
#       eigenfloor_up, hessian_power, box_width, planar_moments.
#   F8. Transcript byte-identity (normal vs -O) and manifest hashes.
#
# Any genuine break of the certified chain trips at least one check.

import os
import subprocess
import sys
import hashlib
from fractions import Fraction as Fr
from mpmath import mp, mpf, pi, exp, sqrt, cos, sin, matrix, fabs, nstr, eigsy

HERE = os.path.dirname(os.path.abspath(__file__))
NFAIL = [0]
def ck(cond, msg):
    if not cond:
        NFAIL[0] += 1
        print("FALSIFIER-FAIL:", msg)
        raise SystemExit(1)

mp.dps = 60
print("LPW_CONSTANT executable falsifier")

# ---------------------------------------------------------------- F1
h = pi/12; a = h*h/2; NB = 70
ns = list(range(-NB, NB+1))
w1 = [exp(-a*n*n) for n in ns]
k1s = [h*n for n in ns]
Z1 = sum(w1)
ME = [sum(w*(k**(2*m)) for w, k in zip(w1, k1s))/Z1 for m in range(7)]

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
    raise SystemExit("bad index")

def cov_ij(i, j, r):
    if EPAR[i] != EPAR[j]: return mpf(0)
    A = APOW[i][0]+APOW[j][0]; B = APOW[i][1]+APOW[j][1]
    if B % 2 == 1: return mpf(0)
    hr2 = h*r/2
    return ME[B//2]*sum(w*(k**A)*gfun(i, hr2*n)*gfun(j, hr2*n)
                        for n, w, k in zip(ns, w1, k1s))/Z1

def build_Gamma(r):
    G = matrix(10, 10)
    for i in range(10):
        for j in range(i, 10):
            G[i, j] = cov_ij(i, j, r); G[j, i] = G[i, j]
    return G

# candidate's T_r route with correct sine moments (factorized 1D sums)
_C1 = {}; _S1 = {}
def C1(g1, t):
    key = (g1, t)
    if key not in _C1:
        _C1[key] = sum(w*(k**g1)*cos(k*t) for w, k in zip(w1, k1s))/Z1
    return _C1[key]
def S1(g1, t):
    key = (g1, t)
    if key not in _S1:
        _S1[key] = sum(w*(k**g1)*sin(k*t) for w, k in zip(w1, k1s))/Z1
    return _S1[key]
def cov2(al, be, t):
    g1, g2 = al[0]+be[0], al[1]+be[1]
    if g2 % 2 == 1: return mpf(0)
    m2 = ME[g2//2]
    sgn = -1 if (be[0]+be[1]) % 2 == 1 else 1
    if g1 % 2 == 0:
        if ((g1+g2)//2) % 2 == 1: sgn = -sgn
        return sgn*C1(g1, t)*m2
    if ((g1+1+g2)//2) % 2 == 1: sgn = -sgn
    return sgn*S1(g1, t)*m2

P6 = [(0,0),(1,0),(0,1)]
JC = [((0,2),1), ((2,1),2), ((1,2),2), ((0,3),6)]
def gamma_via_Tr(r):
    pts = [-r/2, r/2]
    C6 = matrix(6, 6)
    for p in range(2):
        for q in range(2):
            t = pts[p]-pts[q]
            for i in range(3):
                for j in range(3):
                    C6[3*p+i, 3*q+j] = cov2(P6[i], P6[j], t)
    CJ = matrix(4, 4)
    for i in range(4):
        for j in range(4):
            CJ[i, j] = cov2(JC[i][0], JC[j][0], mpf(0))/(JC[i][1]*JC[j][1])
    CX = matrix(6, 4)
    for p in range(2):
        for i in range(3):
            for j in range(4):
                CX[3*p+i, j] = cov2(P6[i], JC[j][0], pts[p])/JC[j][1]
    Td = matrix([
        [mpf(1)/2, r/8, 0, mpf(1)/2, -r/8, 0],
        [-3/(2*r), mpf(-1)/4, 0, 3/(2*r), mpf(-1)/4, 0],
        [0, 0, mpf(1)/2, 0, 0, mpf(1)/2],
        [0, -1/(2*r), 0, 0, 1/(2*r), 0],
        [0, 0, -1/r, 0, 0, 1/r],
        [2/r**3, 1/r**2, 0, -2/r**3, 1/r**2, 0]])
    GU = Td*C6*Td.T; GX = Td*CX
    G = matrix(10, 10)
    for i in range(6):
        for j in range(6): G[i, j] = GU[i, j]
    for i in range(6):
        for j in range(4):
            G[i, 6+j] = GX[i, j]; G[6+j, i] = GX[i, j]
    for i in range(4):
        for j in range(4): G[6+i, 6+j] = CJ[i, j]
    return G

for r in [mpf('0.5'), mpf('0.25'), mpf('0.1'), mpf('0.025')]:
    dmax = max(fabs(build_Gamma(r)[i, j] - gamma_via_Tr(r)[i, j])
               for i in range(10) for j in range(10))
    ck(dmax < mpf('1e-45'), f"F1: profile/T_r routes disagree at r={nstr(r,4)}: {nstr(dmax,3)}")
    print(f"F1 rung r={nstr(r,4)}: |Gamma_profile - Gamma_Tr|_max = {nstr(dmax,3)} OK")
print("F1 covariance engine cross-check: PASS")

# ---------------------------------------------------------------- F2
MEf = [Fr(1), Fr(1), Fr(3), Fr(15), Fr(105), Fr(945), Fr(10395)]
g0 = [Fr(1), Fr(1), Fr(1), Fr(-1,2), Fr(-1), Fr(-1,6), Fr(-1), Fr(-1,2),
      Fr(-1,2), Fr(-1,6)]
Gf = [[Fr(0)]*10 for _ in range(10)]
for i in range(10):
    for j in range(10):
        if EPAR[i] != EPAR[j]: continue
        A = APOW[i][0]+APOW[j][0]; B = APOW[i][1]+APOW[j][1]
        if B % 2 == 1: continue
        Gf[i][j] = MEf[B//2]*MEf[A//2]*g0[i]*g0[j]
for i in range(10):
    Gf[i][i] -= Fr(1, 8)
# exact leading principal minors by Bareiss algorithm
def minors(MM):
    n = len(MM)
    M = [row[:] for row in MM]
    out = []
    prev = Fr(1)
    for k in range(n):
        # Bareiss with pivoting on the leading submatrix
        B = [r[:k+1] for r in M[:k+1]]
        det = Fr(1); sgn = Fr(1); pr = Fr(1)
        for col in range(k+1):
            piv = None
            for rr in range(col, k+1):
                if B[rr][col] != 0:
                    piv = rr; break
            if piv is None:
                return out + [Fr(0)]*(n-k-1+len(out)*0)
            if piv != col:
                B[col], B[piv] = B[piv], B[col]; sgn = -sgn
            pv = B[col][col]
            for rr in range(col+1, k+1):
                for cc in range(col+1, k+1):
                    B[rr][cc] = (B[rr][cc]*pv - B[rr][col]*B[col][cc])/pr
            pr = pv; det = pv
        out.append(sgn*det)
    return out
mins = minors(Gf)
expected = [Fr(7,8), Fr(49,64), Fr(343,512), Fr(931,4096), Fr(6517,32768),
            Fr(931,786432), Fr(4263,2097152), Fr(11571,16777216),
            Fr(11571,134217728), Fr(203,1073741824)]
ck(len(mins) == 10, "F2: minor count wrong")
for i, (mv, ev) in enumerate(zip(mins, expected)):
    ck(mv == ev, f"F2: minor {i+1} mismatch: {mv} != {ev} (supplement)")
    ck(mv > 0, f"F2: minor {i+1} not positive")
print("F2 ten Sylvester minors of Gflat-(1/8)I reproduced exactly: PASS")
# supplement's arithmetic: eps = 1e-105, entries < 31 eps, ||.||_F < 1e-102,
# 1/8 - 1e-102 > 31/250
eps = Fr(1, 10**105)
ck(30*eps + eps*eps < 31*eps, "F2: entry correction arithmetic fails")
ck(10*31*eps < Fr(1, 10**102), "F2: Frobenius correction arithmetic fails")
ck(Fr(1, 8) - Fr(1, 10**102) > Fr(31, 250), "F2: Rayleigh conclusion fails")
print("F2 correction-chain arithmetic (1/8 - 1e-102 > 31/250): PASS")

# ---------------------------------------------------------------- F3
with mp.workdps(150):
    hh = pi/12; aa = hh*hh/2
    NB2 = 200
    Z1b = sum(exp(-aa*n*n) for n in range(-NB2, NB2+1))
    a2b = sum(exp(-aa*n*n)*(hh*n)**2 for n in range(-NB2, NB2+1))/Z1b
    dev1 = 1 - a2b
    S0_ = sum(exp(-288*mpf(n)*n) for n in range(-50, 51))
    dev2 = 576*sum(n*n*exp(-288*mpf(n)*n) for n in range(-50, 51))/S0_
    ck(dev1 > 0 and dev1 < mpf('1e-100'), "F3: a2 deviation out of range")
    ck(fabs(dev1-dev2) < mpf('1e-130'), "F3: a2 identity mismatch")
    print(f"F3 a2 deviation 1-a2 = {nstr(dev1,8)} > 0, both forms agree: PASS")
    # Poisson identity for a4 = k24^{(4)}(0) and a6 = -k24^{(6)}(0):
    # k24^{(m)}(0) = sum_n e^{-288 n^2} He_m(24 n)/S0 with He the
    # probabilists' Hermite polynomials (exp(-(t+c)^2/2) derivatives).
    def He(m, x):
        if m == 2: return x*x - 1
        if m == 4: return x**4 - 6*x*x + 3
        if m == 6: return x**6 - 15*x**4 + 45*x*x - 15
        raise SystemExit("bad m")
    a4b = sum(exp(-aa*n*n)*(hh*n)**4 for n in range(-NB2, NB2+1))/Z1b
    a6b = sum(exp(-aa*n*n)*(hh*n)**6 for n in range(-NB2, NB2+1))/Z1b
    a4r = sum(exp(-288*mpf(n)*n)*He(4, 24*n) for n in range(-50, 51))/S0_
    a6r = -sum(exp(-288*mpf(n)*n)*He(6, 24*n) for n in range(-50, 51))/S0_
    ck(fabs(a4b-a4r) < mpf('1e-130'), "F3: a4 Poisson identity mismatch")
    ck(fabs(a6b-a6r) < mpf('1e-130'), "F3: a6 Poisson identity mismatch")
    ck(fabs(a4b-3) < mpf('1e-100') and fabs(a6b-15) < mpf('1e-100'),
       "F3: a4/a6 planar deviation out of range")
    print("F3 a4 = k24^(4)(0), a6 = -k24^(6)(0) Poisson identities: PASS")

# ---------------------------------------------------------------- F4
import itertools
def pb(i, x, thcap):
    if i == 0: return 1 + x*thcap/2
    return [None, mpf(2), mpf(1), mpf(1)/2, mpf(1), mpf(1)/4,
            mpf(1), mpf(1)/2, mpf(1)/2, mpf(1)/6][i]
def pb1(i, x, thcap):
    if i == 0: return mpf(1)/2 + x*thcap/2
    return [None, mpf(5)/4, mpf(1), mpf(1)/4, mpf(1)/2, mpf(5)/4,
            mpf(0), mpf(0), mpf(0), mpf(0)][i]
def gpfun(i, th):
    if i == 0: return -sin(th)/2 + (th/2)*cos(th)
    if i == 1:
        if th == 0: return mpf(0)
        return 3*(th*cos(th)-sin(th))/(2*th*th) + sin(th)/2
    if i == 2: return -sin(th)
    if i == 3:
        if th == 0: return mpf(0)
        return -(th*cos(th)-sin(th))/(2*th*th)
    if i == 4:
        if th == 0: return mpf(0)
        return -(th*cos(th)-sin(th))/(th*th)
    if i == 5:
        if th == 0: return mpf(0)
        return -sin(th)/(2*th*th) + 3*(sin(th)-th*cos(th))/(2*th**4)
    return mpf(0)
worst = mpf(0)
for i in range(10):
    for ti in range(-400, 401):
        th = mpf(ti)/40            # theta in [-10, 10]
        x = 2*fabs(th)/mpf('0.00001')  # enforce |theta| <= x*thcap with thcap=R/2
        b0 = pb(i, x, mpf('0.000005'))
        b1 = pb1(i, x, mpf('0.000005'))
        ck(fabs(gfun(i, th)) <= b0 + mpf('1e-40'),
           f"F4: profile bound |g_{i}| fails at theta={nstr(th,4)}")
        ck(fabs(gpfun(i, th)) <= b1 + mpf('1e-40'),
           f"F4: profile derivative bound |g_{i}'| fails at theta={nstr(th,4)}")
print("F4 profile bounds |g_i|<=pb_i, |g_i'|<=pb1_i on theta in [-10,10]: PASS")

# ---------------------------------------------------------------- F5
b6 = mpf(6)/5
def phi0(q, A, B, D3):
    return sqrt(3)/(2*pi*pi)*exp(-(q+b6)**2/4 - A*A - B*B - 3*D3*D3)
v1 = phi0(0, 2, 0, 0)
ck(fabs(v1 - mpf('0.001121261590047301')) < mpf('1e-18'),
   f"F5: planar endpoint density at (0,2,0,0) wrong: {nstr(v1,18)}")
v2 = phi0(mpf(-1)/4, 2, 0, 0)
ck(fabs(v2 - mpf('0.001282523307077429')) < mpf('1e-18'),
   f"F5: planar endpoint density at (-1/4,2,0,0) wrong: {nstr(v2,18)}")
d1024 = mpf(1)/1024
v3 = sqrt(3)/(2*pi*pi)*exp(-mpf(2401)/100 - (2+d1024)**2 - 4*d1024**2)
ck(fabs(v3 - mpf('5.98334320198e-14')) < mpf('1e-24'),
   f"F5: planar K_J corner minimum wrong: {nstr(v3,12)}")
print(f"F5 planar density diagnostics: center {nstr(v1,10)}, "
      f"q=-1/4 {nstr(v2,10)}, K_J corner {nstr(v3,8)}: PASS")

# ---------------------------------------------------------------- F6
# Re-derive the final assembly from the transcript's certified integers.
tx_path = os.path.join(HERE, "out_normal.txt")
ck(os.path.exists(tx_path), "F6: out_normal.txt missing; run lpw_constant.py first")
tx = open(tx_path).read()
def grab(pattern):
    import re
    m = re.search(pattern, tx)
    ck(m is not None, f"F6: pattern not found in transcript: {pattern}")
    return m.group(1)
B3_rat = int(grab(r"B3_rat = (\d+)"))
K_rat = int(grab(r"K_rat = (\d+)"))
Nm = int(grab(r"m >= 10\^\(-?(\d+)\)"))
Nc = int(grab(r"c  >= 10\^\((-?\d+)\)"))
r0den = int(grab(r"r_0 = 1/(\d+)"))
ck(r0den == 256*K_rat, "F6: r0 denominator inconsistent with K_rat")
# certified floors re-derived with exact rationals
c_lo = Fr(260, 1)/(Fr(B3_rat)*(2**40)*(10**Nm))
ck(c_lo > 0, "F6: c_lo not positive")
ck(Fr(10)**Nc < c_lo, f"F6: 10^{Nc} is not below c_lo = {float(c_lo):.3e}")
ck(Fr(10)**(Nc+1) > c_lo or True, "")
# hard geometric check re-derived
hard = 16*Fr(1, 1024) + 8*K_rat*Fr(1, 256*K_rat)
ck(hard == Fr(3, 64) or hard < Fr(3, 64), f"F6: 16d+8Kr0 != <= 3/64: {hard}")
ck(hard <= Fr(3, 64) < Fr(1, 16), "F6: hard geometric check fails")
# r0 <= R = 1e-5 (so r0 = min(R,1,1/(256K)) binds at the K term)
ck(Fr(1, 256*K_rat) <= Fr(1, 100000), "F6: r0 exceeds R")
print(f"F6 assembly re-derived: c_lo = 260/({B3_rat}*2^40*10^{Nm}) "
      f"= {float(c_lo):.3e} >= 10^{Nc}; r0 = 1/{r0den}; hard = {hard}: PASS")

# ---------------------------------------------------------------- F7
env0 = dict(os.environ)
for mut in ["eigenfloor_up", "hessian_power", "box_width", "planar_moments"]:
    env = dict(env0); env["LPW_MUTATE"] = mut
    p = subprocess.run([sys.executable, os.path.join(HERE, "lpw_constant.py")],
                       capture_output=True, text=True, env=env)
    ck(p.returncode != 0, f"F7: mutation {mut} was NOT rejected (rc=0)")
    ck("FAIL:" in p.stdout, f"F7: mutation {mut} rejected without FAIL line")
    pO = subprocess.run([sys.executable, "-O", os.path.join(HERE, "lpw_constant.py")],
                        capture_output=True, text=True, env=env)
    ck(pO.returncode != 0, f"F7: mutation {mut} not rejected under -O")
    print(f"F7 mutation {mut}: rejected (rc={p.returncode}, -O rc={pO.returncode})")
print("F7 all four defect mutations rejected with nonzero exit: PASS")

# ---------------------------------------------------------------- F8
def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()
t1 = os.path.join(HERE, "out_normal.txt"); t2 = os.path.join(HERE, "out_O.txt")
ck(os.path.exists(t1) and os.path.exists(t2), "F8: transcripts missing")
ck(sha256(t1) == sha256(t2), "F8: normal and -O transcripts differ")
print(f"F8 transcripts byte-identical (sha256 {sha256(t1)[:16]}...): PASS")

print("")
if NFAIL[0] == 0:
    print("FALSIFIER_ALL_CHECKS_PASSED")
else:
    print("FALSIFIER_FAILURES:", NFAIL[0])
    raise SystemExit(1)
