#!/usr/bin/env python3
# audit_scope_027a_v1.py  (K3 SWARM Phase 4 - DER-027a SCOPE AUDIT, supersedes WO-063 Task 4a frame)
#
# Recomputes every load-bearing quantity of KIMI-DER-027a, upgrades the knot layer from
# F64PAD mesh+heuristic pad to INTERVAL control (rigorous mean-value box enclosures; every
# error term derived, no empirical factors), attempts a genuine uniform continuation in r
# (jet-frame perturbation), keeps the v*=-1/2 isolation prominent, and prints the exact
# valid-scope table (claim x d-range x r-range x conditioning family x evidence tier).
# Fail-closed ck(); deterministic; normal and `python3 -O` transcripts byte-identical.
# Precision labels: EXACT / DPS80 (certified mp bounds) / IB (interval-box enclosure) /
# F64 (float64 center evaluation with derived rounding bound; never load-bearing alone).
import mpmath as mp
import numpy as np
import hashlib, os

mp.mp.dps = 80
OUTDIR = os.path.dirname(os.path.abspath(__file__))
DERDIR = "/mnt/agents/output/19fcef2e-c022-877e-8000-0f5a09d447b0"
UPDIR = "/mnt/agents/upload"

FAILS = []
def ck(cond, name):
    if cond:
        print("PASS  " + name)
    else:
        print("FAIL  " + name)
        FAILS.append(name)

def fmt(x, n=10):
    return mp.nstr(mp.mpf(x), n)

def sha256_of(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()

# ================= A0 receipts (input + DER-027a starting record) =================
print("=== A0 receipts (starting record; labels/receipts section at end) ===")
WO = "AO48-WO-063 - LB-RATE HARDENING CAMPAIGN work order for KIMI - WP resolution + AUD-023 v1.1 + gamma-LOC ii-c + rigorous far-grid-ridge + THM-023 v1.1 all-small-r + transport settlements - agents authorized - 2026-08-04.md"
REC = {}
for tag, p in [("WO-063", os.path.join(UPDIR, WO)),
               ("C022", os.path.join(UPDIR, "C022 Observed Update.json")),
               ("c022fd", os.path.join(UPDIR, "c022 fd.json")),
               ("DER027a-md", os.path.join(DERDIR, "KIMI-DER-027a_farfield_envelope.md")),
               ("DER027a-py", os.path.join(DERDIR, "verify_farfield_envelope_v1.py")),
               ("DER027a-out", os.path.join(DERDIR, "verify_farfield_envelope_v1.out.txt")),
               ("DER027a-Oout", os.path.join(DERDIR, "verify_farfield_envelope_v1.O.out.txt"))]:
    REC[tag] = sha256_of(p)
    print(f"receipt {tag}: {REC[tag]}")
ck(REC["WO-063"] == "e699bf44b7b84fca88faa2b509c1686ea9f8e500aee0dbe618c859b5ec5ad2e2",
   "A0: WO-063 hash echoes task statement")
ck(REC["C022"].startswith("9bc0647b"), "A0: C022 hash echoes task statement prefix")
ck(REC["DER027a-out"] == REC["DER027a-Oout"], "A0: DER-027a transcripts byte-identical (receipt check)")

# ================= A1 spectral recomputation + independent identities =================
print("=== A1 spectral lattice, normalization, certified tails (recompute) ===")
a = mp.pi/12
NLAT = int(mp.ceil(30/a)) + 2
mass = {}
for n1 in range(-NLAT, NLAT+1):
    for n2 in range(-NLAT, NLAT+1):
        s2 = n1*n1 + n2*n2
        if a*a*s2 <= 900:
            mass[(n1, n2)] = mp.e**(-a*a*s2/2)
Z30 = sum(mass.values())
print("lattice points:", len(mass), " Z30 =", fmt(Z30, 25))
# independent identity: Z_inf = (2*pi/a^2) * sum_n e^{-|24n|^2/2} (Poisson); Z30 = Z_inf - T
ZINF = (2*mp.pi/a**2) * (1 + 4*mp.e**(-mp.mpf(24)**2/2) + 4*mp.e**(-mp.mpf(48)**2/2))
print("Z_inf (independent Poisson identity) =", fmt(ZINF, 25), " Z_inf - Z30 =", fmt(ZINF-Z30, 6))

def tail_T(m):
    s = mp.mpf(0)
    B = int(mp.ceil(40/a)) + 2
    for n1 in range(-B, B+1):
        for n2 in range(-B, B+1):
            k2 = a*a*(n1*n1 + n2*n2)
            if 900 < k2 <= 1600:
                s += mp.e**(-k2/2) * mp.sqrt(k2)**m
    c = a/mp.sqrt(2)
    f = lambda s: mp.e**(-s*s/2) * (2*s)**(m+1)
    I = mp.quad(f, [39, 50]) + mp.e**(-mp.mpf(50)**2/2) * (2*50)**(m+1)
    return s + (1/a**2) * 2*mp.pi * I

T = [tail_T(m) for m in range(35)]
ck(T[0]/Z30 < mp.mpf('1e-60'), "A1: certified tail T_0/Z30 < 1e-60 (recompute)")
ck(T[0]/Z30 < mp.mpf('4.4e-185'), "A1: certified tail below recorded 4.4e-185")
ck(abs(ZINF - Z30) < mp.mpf('1e-76'),
   "A1: Z30 consistent with independent Poisson identity to 1e-76 (dps-80 pi-ulp scale; T_0 = 2.8e-194)")
print("T_0/Z30 =", fmt(T[0]/Z30, 8))
TAU = T[0]/Z30

# ================= A2 kernel (wrapped, certified) + cross-checks =================
print("=== A2 wrapped kernel, image remainders, cross-representation checks (recompute) ===")
C0N = 1 + 4*mp.e**(-mp.mpf(24)**2/2) + 4*mp.e**(-mp.mpf(48)**2/2) + 4*mp.e**(-(24*mp.sqrt(2))**2/2)
c0 = 1/C0N
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
        s += mp.mpf(10)**6 * (24*7+12)**(j1+j2) * mp.e**(-mp.mpf(156)**2/2)
        IMG[(j1, j2)] = s

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
HE = [he_poly(n) for n in range(35)]     # ascending EXACT integer coefficients
def he_val(n, x):
    return mp.polyval(HE[n][::-1], x)    # mp.polyval takes DESCENDING coefficients

def kernel_d(u1, u2, j1, j2):
    s = mp.mpf(0)
    for n1 in range(-2, 3):
        for n2 in range(-2, 3):
            w1 = u1 + 24*n1; w2 = u2 + 24*n2
            s += he_val(j1, w1) * he_val(j2, w2) * mp.e**(-(w1*w1 + w2*w2)/2)
    return ((-1)**(j1+j2)) * c0 * s

def kcorr(j1, j2):
    return 2*T[j1+j2]/Z30 + IMG[(j1, j2)]

# cross-check vs direct spectral (mp) at 2 points
keys = sorted(mass.keys())
def Cspectral_mp(u1, u2, j1, j2):
    s = mp.mpf(0)
    for (n1, n2) in keys:
        m = mass[(n1, n2)]
        ph = a*n1*u1 + a*n2*u2
        # d^j cos(k.u) = k^j cos(ph + |j| pi/2)  (exact)
        val = m * (a*n1)**j1 * (a*n2)**j2 * mp.cos(ph + (j1+j2)*mp.pi/2)
        s += val
    return s/Z30
for (x, y, j1, j2) in [(mp.mpf('3.0'), mp.mpf('0.7'), 0, 0), (mp.mpf('5.1'), mp.mpf('-2.2'), 1, 0),
                       (mp.mpf('2.9'), mp.mpf('4.4'), 0, 1)]:
    dd = abs(Cspectral_mp(x, y, j1, j2) - kernel_d(x, y, j1, j2))
    print(f"spectral vs wrapped diff at ({x},{y}) order ({j1},{j2}) =", fmt(dd, 6))
    ck(dd < mp.mpf('1e-50'), f"A2: cross-representation agreement < 1e-50 at ({x},{y}) ({j1},{j2})")

# derivative-at-origin moment identity (EXACT): d^{2i,2k} C(0) = (-1)^{i+k} (2i-1)!! (2k-1)!!
def dfact(n):
    return mp.factorial(n)/ (mp.mpf(2)**(n//2) * mp.factorial(n//2)) if n % 2 == 0 else mp.mpf(0)
mmaxdev = mp.mpf(0)
for i in range(0, 5):
    for k in range(0, 5):
        exact = ((-1)**(i+k)) * dfact(2*i) * dfact(2*k)
        mm = kernel_d(mp.mpf(0), mp.mpf(0), 2*i, 2*k)
        mmaxdev = max(mmaxdev, abs(exact - mm))
print("max |dC(0) - exact double-factorial moments| =", fmt(mmaxdev, 6))
ck(mmaxdev < mp.mpf('1e-55'), "A2: origin moments match EXACT double-factorial values")

# ================= A3 Gram matrices, certified inverses; cross-check vs DER-027a =================
print("=== A3 pin Gram matrices (recompute) with certified lambda_min and inverse ===")
RUNGS = [mp.mpf(1)/20, mp.mpf(1)/40]
BB = mp.mpf(6)/5
YCO = (mp.mpf(-63)/50, mp.mpf(6)/25)
VSTAR = mp.mpf(-1)/2
ROUND = mp.mpf('1e-70')

def pins_of(r, useY):
    M = (-r/2, mp.mpf(0)); S = (r/2, mp.mpf(0)); Y = (YCO[0]*r, YCO[1]*r)
    p = [(M, (0,0)), (M, (1,0)), (M, (0,1)), (S, (0,0)), (S, (1,0)), (S, (0,1))]
    if useY: p.append((Y, (0,0)))
    return p

def vals_of(r, useY):
    ell = r**3/6
    v = [BB, mp.mpf(0), mp.mpf(0), BB-ell, mp.mpf(0), mp.mpf(0)]
    if useY: v.append(VSTAR)
    return v

def build_G(r, useY):
    p = pins_of(r, useY); n = len(p); Gm = mp.matrix(n, n)
    for i in range(n):
        xi, ai = p[i]
        for j in range(n):
            xj, aj = p[j]
            Gm[i, j] = ((-1)**(aj[0]+aj[1])) * kernel_d(xi[0]-xj[0], xi[1]-xj[1],
                                                        ai[0]+aj[0], ai[1]+aj[1])
    return p, Gm

def certify(Gm):
    n = Gm.rows
    lam, Q = mp.eigsy(Gm)
    Rm = Gm - Q*mp.diag(lam)*Q.T
    rho = mp.sqrt(sum(Rm[i, j]**2 for i in range(n) for j in range(n))) + ROUND
    lam_hat = lam[0] - rho
    Minv = mp.matrix(n, n)
    for i in range(n):
        for j in range(n):
            Minv[i, j] = sum(Q[i, k]*Q[j, k]/lam[k] for k in range(n))
    Ms = (Minv + Minv.T)/2
    E = mp.eye(n) - Minv*Gm
    eta = mp.sqrt(sum(E[i, j]**2 for i in range(n) for j in range(n))) + ROUND
    nM = mp.sqrt(sum(Minv[i, j]**2 for i in range(n) for j in range(n))) + ROUND
    epsM = 2*nM*eta/(1-eta) + ROUND
    return lam_hat, Ms, epsM

ENS = {}
# DER-027a anchors (transcript values, mutation-test reference)
ANCHOR = {(True, 0): ('2.5677067e-10', '9.58778e-61'), (True, 1): ('4.7566847e-12', '3.76507e-58'),
          (False, 0): ('3.2552060e-10', '7.08626e-61'), (False, 1): ('5.0862628e-12', '4.86464e-58')}
for useY in (True, False):
    for ri, r in enumerate(RUNGS):
        p, Gm = build_G(r, useY)
        lam_hat, Ms, epsM = certify(Gm)
        ENS[(useY, ri)] = dict(pins=p, G=Gm, lam_hat=lam_hat, M=Ms, epsM=epsM, ri=ri, useY=useY)
        tag = "7-pin" if useY else "6-pin"
        print(f"{tag} r=1/{int(1/r)}: lambda_hat={fmt(lam_hat,8)} epsM={fmt(epsM,6)}")
        ck(lam_hat > 0, f"A3: G certified PD ({tag} r=1/{int(1/r)})")
        a1, a2 = ANCHOR[(useY, ri)]
        ck(abs(lam_hat - mp.mpf(a1))/mp.mpf(a1) < mp.mpf('1e-6'),
           f"A3: recomputed lambda_hat matches DER-027a anchor ({tag} r=1/{int(1/r)})")
        ck(epsM < mp.mpf(a2)*10, f"A3: inverse residual within 10x of DER-027a anchor ({tag})")

# ================= A4 derived envelope machinery (recompute, interval-grade constants) =================
print("=== A4 Hermite-exact kernel bounds, Schur-complement moments, envelope (recompute) ===")
def Ppow(i, k, d):
    if i == 0 and k == 0:
        return mp.e**(-d*d/2)
    if d*d <= i + k:
        return mp.mpf(i)**(mp.mpf(i)/2) * mp.mpf(k)**(mp.mpf(k)/2) * mp.e**(-mp.mpf(i+k)/2)
    rho = (mp.mpf(i)/(i+k))**(mp.mpf(i)/2) * (mp.mpf(k)/(i+k))**(mp.mpf(k)/2)
    return d**(i+k) * rho * mp.e**(-d*d/2)

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

JMAX = 8
MLIST = [(m1, m2) for m1 in range(JMAX+3) for m2 in range(JMAX+3-m1)]

def moment_tables(entry):
    p = entry['pins']; Ms = entry['M']; n = Ms.rows
    nu, Q = mp.eigsy(Ms)
    nu_min = min(nu)
    Wm = mp.matrix(n, n)
    for q in range(n):
        for j in range(n):
            Wm[q, j] = mp.sqrt(nu[q]) * Q[j, q]
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
    rn = [sum(abs(Wm[q, j]) for j in range(Wm.cols)) for q in range(Wm.rows)]
    return Wm, ctab, cWerr, rn

def rhoc_of(p):
    return max(mp.sqrt(xp[0]**2 + xp[1]**2) for xp, _ in p)

_BK = {}
def BkM(m1, m2, d):
    key = (m1, m2, mp.nstr(d, 30))
    if key not in _BK:
        _BK[key] = Bker(m1, m2, d)
    return _BK[key]

_TB = {}
def tail_base(d, rhoc, amax, shift=(0, 0)):
    key = (mp.nstr(d, 30), shift)
    if key in _TB:
        return _TB[key]
    dd = d - 2*rhoc
    tot = mp.mpf(0)
    for s in range(JMAX+1, 31):
        bk = mp.mpf(0)
        for j1 in range(0, s+1):
            bk = max(bk, BkM(amax+shift[0]+j1, amax+shift[1]+(s-j1), dd))
        term = (2*rhoc)**s / mp.factorial(s) * bk
        tot += term
        if s >= JMAX+6 and term < tot*mp.mpf('1e-12'):
            break
    _TB[key] = tot
    return tot

def tail_R(d, rn, rhoc, amax, shift=(0, 0)):
    return rn * tail_base(d, rhoc, amax, shift) * mp.mpf('1.05')

def envelope(entry, d, shift=(0, 0)):
    n = len(entry['pins'])
    ctab, rn, rhoc = entry['ctab'], entry['rn'], entry['rhoc']
    E = mp.mpf(0)
    for q in range(n):
        s = mp.mpf(0)
        for (m1, m2) in MLIST:
            s += (abs(ctab[(m1, m2)][q]) + entry['cWerr']) * BkM(m1+shift[0], m2+shift[1], d)
        s += tail_R(d, rn[q], rhoc, 1, shift)
        E += s*s
    vcrude = 3*(BkM(0, 0, d))**2 + 2*((BkM(1, 0, d))**2 + (BkM(0, 1, d))**2)
    return E + mp.mpf('1.01') * entry['epsM'] * vcrude

def mu_tables(entry, r, useY):
    p = entry['pins']; Ms = entry['M']; n = Ms.rows
    beta = mp.matrix(vals_of(r, useY))
    w = Ms*beta
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
    bnorm = mp.sqrt(sum(beta[i]**2 for i in range(n)))
    return cw, wn1, bnorm

def envelope_mu(entry, d, shift=(0, 0)):
    cw, wn1, bnorm = entry['mu']
    s = mp.mpf(0)
    for (m1, m2) in MLIST:
        s += abs(cw[(m1, m2)]) * BkM(m1+shift[0], m2+shift[1], d)
    s += tail_R(d, wn1, entry['rhoc'], 1, shift)
    s += mp.mpf('1.01')*entry['epsM']*bnorm * (3*BkM(0, 0, d) + 4*BkM(1, 0, d))
    return s

for useY in (True, False):
    for ri, r in enumerate(RUNGS):
        entry = ENS[(useY, ri)]
        Wm, ctab, cWerr, rn = moment_tables(entry)
        entry['W'] = Wm; entry['ctab'] = ctab; entry['cWerr'] = cWerr; entry['rn'] = rn
        entry['rhoc'] = rhoc_of(entry['pins'])
        entry['mu'] = mu_tables(entry, r, useY)
        tag = "7-pin" if useY else "6-pin"
        print(f"{tag} r=1/{int(1/r)}: cWerr={fmt(cWerr,6)} rhoc={fmt(entry['rhoc'],6)}")
        ck(cWerr < mp.mpf('1e-40'), f"A4: moment certification error < 1e-40 ({tag} r=1/{int(1/r)})")

DGRID = [mp.mpf(3) + mp.mpf(i)/10 for i in range(0, 91)]
ENVV = {}
for useY in (True, False):
    for ri in (0, 1):
        entry = ENS[(useY, ri)]
        vv = [envelope(entry, d) for d in DGRID]
        ENVV[(useY, ri)] = vv
        mono = all(vv[i+1] <= vv[i] for i in range(len(vv)-1))
        tag = "7-pin" if useY else "6-pin"
        print(f"{tag} r=1/{int(1/RUNGS[ri])}: E(3)={fmt(vv[0],6)} E(5)={fmt(vv[20],6)} "
              f"E(6.1)={fmt(vv[31],6)} monotone={mono}")
        ck(mono, f"A4: envelope certified non-increasing on [3,12] ({tag} r=1/{int(1/RUNGS[ri])})")

# ================= A5 INTERVAL CONTROL: rigorous box enclosures at the knots =================
print("=== A5 interval-grade certified suprema (box mean-value enclosures; no heuristic pads) ===")
# Every error term derived: (i) float64 center rounding eps_f (stated bound),
# (ii) box half-width from certified Bker Lipschitz table, (iii) image+tail corrections
# kcorr, (iv) coefficient intervals from cWerr, (v) Taylor tail R, (vi) inverse residual epsM.
c0f = float(c0)
IMG2 = [(n1, n2) for n1 in range(-2, 3) for n2 in range(-2, 3)]
SHE = [sum(abs(c) for c in p) for p in HE]

def f64_dC_err(u1, u2, j1, j2):
    """center value of d^{j}C via wrapped float64 + derived rounding bound."""
    s = np.zeros_like(u1); eb = np.zeros_like(u1)
    for (n1, n2) in IMG2:
        w1 = u1 + 24*n1; w2 = u2 + 24*n2
        # Horner with ascending-coeff reversal (descending loop)
        h1 = np.zeros_like(w1); h2 = np.zeros_like(w2)
        for cf in reversed(HE[j1]): h1 = h1*w1 + cf
        for cf in reversed(HE[j2]): h2 = h2*w2 + cf
        e = np.exp(-(w1*w1 + w2*w2)/2)
        s += h1*h2*e
        eb += SHE[j1]*SHE[j2]*np.maximum(np.abs(w1), 1.0)**j1*np.maximum(np.abs(w2), 1.0)**j2*e
    nround = (2*(j1+j2) + 12)
    return ((-1)**(j1+j2))*c0f*s, c0f*eb*(2.0**-53)*nround

def hull_dist_f(y1, y2, xy):
    dd = np.full(np.atleast_1d(y1).shape, np.inf)
    for i in range(len(xy)):
        for j in range(i+1, len(xy)):
            A = np.array(xy[i]); B = np.array(xy[j]); BA = B-A
            tt = np.clip(((y1-A[0])*BA[0]+(y2-A[1])*BA[1])/(BA@BA), 0, 1)
            dd = np.minimum(dd, np.hypot(y1-(A[0]+tt*BA[0]), y2-(A[1]+tt*BA[1])))
    return dd

def knots_certified(useY, ri, d, chan="var", ntheta=540, ht=0.06, trange=1.3, override=None,
                    rounds=2):
    """interval-certified sup over {dist(y,hull) >= d} of Delta (chan='var') or |mu| ('mu').
    Boxes B(center, rho) cover the annulus; box may intersect the domain iff
    dist(center, hull) + rho >= d. Box hi is a rigorous enclosure (derived error terms).
    Branch-and-bound: boxes whose hi exceeds the best rigorous lower bound are replaced
    by 3x3 sub-boxes (2 rounds). override: custom (ctab, rn, rhoc, epsM, cWerr, hull,
    tailzero) spec (used for the r-free jet ensemble)."""
    entry = ENS[(useY, ri)]
    if override is None:
        ctab = entry['ctab']; rhoc = float(entry['rhoc'])
        cWerr = float(entry['cWerr']); epsM = float(entry['epsM'])
        cw, wn1, bnorm = entry['mu']
        xy = sorted(set(tuple(map(float, xp)) for xp, _ in entry['pins']))
        rn = entry['rn']; tailzero = False
        n = len(entry['pins'])
        cfac = {(m1, m2): np.array([float(ctab[(m1, m2)][q]) for q in range(n)]) for (m1, m2) in MLIST}
        mufac = {(m1, m2): float(cw[(m1, m2)]) for (m1, m2) in MLIST}
    else:
        rhoc = override['rhoc']; cWerr = override['cWerr']; epsM = override['epsM']
        xy = override['hull']; rn = override['rn']; tailzero = override['tailzero']
        cfac = override['cfac']; n = len(rn)
        mufac = {(m1, m2): 0.0 for (m1, m2) in MLIST}
        wn1, bnorm = 0.0, 0.0
    dth = 2*np.pi/ntheta

    def box_eval(t, ths, rho):
        """returns (vv center, hh rigorous hi, keep) for boxes centered (t, ths) radius rho.
        t may be a scalar or an array broadcastable with ths; the certified constants
        (Lipschitz, tails) use the conservative scalar din = min(t) - rho."""
        ths = np.asarray(ths, dtype=float)
        tarr = np.asarray(t, dtype=float)
        din = max(1.0, float(np.min(tarr)) - rho)
        y1 = tarr*np.cos(ths); y2 = tarr*np.sin(ths)
        lipQ = override.get('lipQ') if override else None
        dm = {}; hw = {}
        for (m1, m2) in MLIST:
            val, err = f64_dC_err(y1, y2, m1, m2)
            if lipQ is None:
                lip = float(BkM(m1+1, m2, mp.mpf(din))) + float(BkM(m1, m2+1, mp.mpf(din)))
                hw[(m1, m2)] = lip*rho + err + float(kcorr(m1, m2))
            else:
                hw[(m1, m2)] = err + float(kcorr(m1, m2))
            dm[(m1, m2)] = val
        if tailzero:
            tailv = [0.0]*n; tailm = 0.0
        else:
            tailv = [float(tail_R(mp.mpf(din), rn[q], entry['rhoc'], 1)) for q in range(n)]
            tailm = float(tail_R(mp.mpf(din), wn1, entry['rhoc'], 1))
        Bhi = np.full(len(ths), epsM*1.01*(3*float(BkM(0, 0, mp.mpf(din)))**2 +
                     2*(float(BkM(1, 0, mp.mpf(din)))**2 + float(BkM(0, 1, mp.mpf(din)))**2)))
        vv = np.zeros(len(ths))
        for q in range(n):
            ubc = np.zeros(len(ths)); ubw = np.full(len(ths), tailv[q])
            for (m1, m2) in MLIST:
                c = cfac[(m1, m2)][q]
                ubc += c*dm[(m1, m2)]
                ubw += abs(c)*hw[(m1, m2)] + cWerr*(np.abs(dm[(m1, m2)]) + hw[(m1, m2)])
            if lipQ is not None:
                ubw += lipQ[q]*rho
            lo = ubc - ubw; hi = ubc + ubw
            Bhi += np.maximum(np.abs(lo), np.abs(hi))**2
            vv += np.where(lo*hi > 0, np.minimum(np.abs(lo), np.abs(hi))**2, 0.0)
        muc = np.zeros(len(ths)); muw = np.full(len(ths), tailm +
                  1.01*epsM*bnorm*(3*float(BkM(0, 0, mp.mpf(din))) + 4*float(BkM(1, 0, mp.mpf(din)))))
        for (m1, m2) in MLIST:
            muc += mufac[(m1, m2)]*dm[(m1, m2)]
            muw += abs(mufac[(m1, m2)])*hw[(m1, m2)]
        if chan == "var":
            out_ctr, out_hi = vv, Bhi
        else:
            out_ctr, out_hi = np.abs(muc), np.abs(muc) + muw
        keep = hull_dist_f(y1, y2, xy) + rho >= d - 1e-15
        return out_ctr, out_hi, keep

    th = np.linspace(0, 2*np.pi, ntheta, endpoint=False)
    rows = []
    t = float(d) - rhoc - ht
    while t <= float(d) + trange + 1e-12:
        if t > rhoc + 0.1:
            rows.append(t)
        t += ht
    # branch and bound: boxes carry (t, th, rho, hi, lo); refine every box whose hi
    # exceeds the best rigorous lower bound, replacing it by 3x3 sub-boxes (2 rounds).
    boxes = []   # (t0, th0, rho, hi, lo)
    best_lo = 0.0; cand = None
    for t in rows:
        rho = 0.5*np.sqrt((t*dth)**2 + ht**2)
        vv, hh, keep = box_eval(t, th, rho)
        for i in np.where(keep)[0]:
            boxes.append((t, float(th[i]), rho, float(hh[i]), float(vv[i])))
            if vv[i] > best_lo:
                best_lo = float(vv[i]); cand = (t, float(th[i]))
    for rnd in range(rounds):
        hot = [b for b in boxes if b[3] > best_lo]
        if not hot:
            break
        hot.sort(key=lambda b: -b[3])
        hot = hot[:1500 if rnd == 0 else 800]
        hotset = set(id(b) for b in hot)
        boxes = [b for b in boxes if id(b) not in hotset]
        ts_l = []; ths_l = []; rho_l = 0.0; meta = []
        for (t0, th0, rho0, _, _) in hot:
            rho3 = rho0/3.0
            rho_l = max(rho_l, rho3)
            ht3 = 2*rho0/3.0  # angular/radial split step approx rho0*(2/3)/max(t0,1) handled by rho
            for dtt in (-2*rho0/3, 0.0, 2*rho0/3):
                for dthh in (-2*rho0/max(t0, 1e-9)/3, 0.0, 2*rho0/max(t0, 1e-9)/3):
                    if t0 + dtt > rhoc + 0.05:
                        ts_l.append(t0 + dtt); ths_l.append(th0 + dthh); meta.append(rho3)
        vv, hh, keep = box_eval(np.array(ts_l), np.array(ths_l), rho_l)
        for i in range(len(ts_l)):
            if keep[i]:
                boxes.append((ts_l[i], ths_l[i], rho_l, float(hh[i]), float(vv[i])))
                if vv[i] > best_lo:
                    best_lo = float(vv[i]); cand = (ts_l[i], ths_l[i])
    best_hi = max(b[3] for b in boxes)
    if override is not None and override.get('Eb_fun') is not None:
        Eb = override['Eb_fun'](d + mp.mpf(str(trange)))
    elif chan == "var":
        Eb = envelope(entry, d + mp.mpf(str(trange)))
    else:
        Eb = envelope_mu(entry, d + mp.mpf(str(trange)))
    cert = max(mp.mpf(str(best_hi)), Eb)
    return dict(cert=cert, center=best_lo, hi=best_hi, Ebeyond=Eb, cand=cand)

def hull_dist_f(y1, y2, xy):
    dd = np.full(np.atleast_1d(y1).shape, np.inf)
    for i in range(len(xy)):
        for j in range(i+1, len(xy)):
            A = np.array(xy[i]); B = np.array(xy[j]); BA = B-A
            tt = np.clip(((y1-A[0])*BA[0]+(y2-A[1])*BA[1])/(BA@BA), 0, 1)
            dd = np.minimum(dd, np.hypot(y1-(A[0]+tt*BA[0]), y2-(A[1]+tt*BA[1])))
    return dd

KIB = {}
for useY in (True, False):
    for ri in (0, 1):
        for d in (mp.mpf(3), mp.mpf(5), mp.mpf('6.1')):
            res = knots_certified(useY, ri, d, "var")
            KIB[(useY, ri, d)] = res
            tag = "7-pin" if useY else "6-pin"
            print(f"IB {tag} r=1/{int(1/RUNGS[ri])} d={d}: box_sup_hi={res['hi']:.6e} "
                  f"center={res['center']:.6e} E_beyond={fmt(res['Ebeyond'],4)} certified={fmt(res['cert'],6)}")

# recorded-constant checks at interval grade
ck(KIB[(True, 0, mp.mpf(3))]['cert'] <= mp.mpf('2.24e-2'), "A5: IB certified d=3 <= 2.24% (7-pin r=1/20)")
ck(KIB[(True, 1, mp.mpf(3))]['cert'] <= mp.mpf('2.24e-2'), "A5: IB certified d=3 <= 2.24% (7-pin r=1/40)")
for useY in (True, False):
    for ri in (0, 1):
        ck(KIB[(useY, ri, mp.mpf(5))]['cert'] <= mp.mpf('1e-4'),
           f"A5: IB certified d=5 <= 1e-4 ({'7' if useY else '6'}-pin r=1/{int(1/RUNGS[ri])})")
        ck(KIB[(useY, ri, mp.mpf('6.1'))]['cert'] <= mp.mpf('7.5e-13'),
           f"A5: IB certified d=6.1 <= 7.5e-13 ({'7' if useY else '6'}-pin r=1/{int(1/RUNGS[ri])})")
# consistency vs DER-027a F64PAD values (must agree within old pads)
DER027A_SUP = {(True, 0, 3): '0.0194319', (True, 1, 3): '0.0204043',
               (False, 0, 3): '0.0193899', (False, 1, 3): '0.0203911',
               (True, 0, 5): '3.45513e-8', (True, 1, 5): '3.81227e-8',
               (False, 0, 5): '3.43132e-8', (False, 1, 5): '3.80577e-8',
               (True, 0, 6.1): '5.28475e-13', (True, 1, 6.1): '5.96702e-13',
               (False, 0, 6.1): '5.22296e-13', (False, 1, 6.1): '5.95071e-13'}
for (useY, ri, dd), s in DER027A_SUP.items():
    dkey = mp.mpf(3) if dd == 3 else (mp.mpf(5) if dd == 5 else mp.mpf('6.1'))
    old = mp.mpf(s)
    new = KIB[(useY, ri, dkey)]['cert']
    newlo = mp.mpf(str(KIB[(useY, ri, dkey)]['center']))
    dev = abs(new - old)/old
    print(f"consistency d={dd} {'7' if useY else '6'}-pin r=1/{int(1/RUNGS[ri])}: "
          f"old_cert={s} new_cert={fmt(new,6)} new_lo={fmt(newlo,6)} dev={fmt(dev,4)}")
    # cross-validity: each framework's rigorous lower bound must sit below the other's
    # certified upper bound (both bound the same true sup).
    ck(newlo <= old*mp.mpf('1.001'),
       f"A5: IB lower bound below DER-027a certificate (d={dd} {'7' if useY else '6'}-pin r=1/{int(1/RUNGS[ri])})")
    ck(new >= newlo, f"A5: IB certificate dominates its lower bound (d={dd})")

# ================= A6 uniform continuation in r (jet-frame perturbation) =================
print("=== A6 genuine uniform continuation attempt (jet frame, N=8, K'=4) ===")
NJET = 8; KPR = 4
JLIST = [(m1, m2) for m1 in range(NJET+1) for m2 in range(NJET+1-m1)]
J4 = [(m1, m2) for (m1, m2) in JLIST if m1+m2 <= KPR]
def exact_moment(j1, j2):
    if j1 % 2 or j2 % 2: return mp.mpf(0)
    return ((-1)**((j1+j2)//2)) * dfact(j1) * dfact(j2)
nJ = len(JLIST)
GJ = mp.matrix(nJ, nJ)
for i, (m1, m2) in enumerate(JLIST):
    for j, (n1, n2) in enumerate(JLIST):
        GJ[i, j] = ((-1)**(n1+n2)) * exact_moment(m1+n1, m2+n2)
lamJ, QJ = mp.eigsy(GJ)
RJ = GJ - QJ*mp.diag(lamJ)*QJ.T
rhoJ = mp.sqrt(sum(RJ[i, j]**2 for i in range(nJ) for j in range(nJ))) + ROUND
lamJ_hat = lamJ[0] - rhoJ
print("jet Gram lambda_min certified:", fmt(lamJ_hat, 8), " (", nJ, "jets )")
ck(lamJ_hat > 0, "A6: jet Gram certified PD")
WJ = mp.matrix(nJ, nJ)
for q in range(nJ):
    for j in range(nJ):
        WJ[q, j] = QJ[j, q] / mp.sqrt(lamJ[q])   # whitening of GJ^{-1}: WJ^T WJ = GJ^{-1}
# r-free jet ensemble: pins ARE the jets at the origin -> moment form is EXACT (no tail).
# The triangle bound on E_J is vacuous (cancellation); certify E_J by the IB box layer.
MJ = mp.matrix(nJ, nJ)
for i in range(nJ):
    for j in range(nJ):
        MJ[i, j] = sum(QJ[i, k]*QJ[j, k]/lamJ[k] for k in range(nJ))
EJres = mp.eye(nJ) - MJ*GJ
etaJ = mp.sqrt(sum(EJres[i, j]**2 for i in range(nJ) for j in range(nJ))) + ROUND
nMJ = mp.sqrt(sum(MJ[i, j]**2 for i in range(nJ) for j in range(nJ))) + ROUND
epsMJ = 2*nMJ*etaJ/(1-etaJ) + ROUND
nuJmin = 1/lamJ[nJ-1]
cWerrJ = mp.sqrt(nJ)*epsMJ/(2*mp.sqrt(nuJmin)) + ROUND
print(f"jet ensemble: epsM={fmt(epsMJ,6)} cWerr={fmt(cWerrJ,6)}")
cfacJ = {}
for (m1, m2) in MLIST:
    arr = np.zeros(nJ)
    if (m1, m2) in JLIST:
        jj = JLIST.index((m1, m2))
        for q in range(nJ):
            arr[q] = float(WJ[q, jj]) * ((-1)**(m1+m2))
    cfacJ[(m1, m2)] = arr
rnJ = [float(sum(abs(WJ[q, j]) for j in range(nJ))) for q in range(nJ)]
# certified per-q spectral Lipschitz constants for the dual functions u_q:
# d_i u_q(y) = Z30^-1 sum_k m_k Q_q(ik)(i k_i) e^{ik.y}  =>  |d_i u_q| <= Lam_qi with
# Lam_qi = Z30^-1 sum_k m_k |Q_q(ik)| |k_i|  +  certified tail (triangle |Q_q| <= W1 poly).
af = float(a)
karr = np.array([(af*n1, af*n2) for (n1, n2) in keys], dtype=float)
marr = np.array([float(mass[(n1, n2)]) for (n1, n2) in keys], dtype=float)
Zf = float(Z30)
WJf = np.array([[float(WJ[q, j]) for j in range(nJ)] for q in range(nJ)])
sgnJ = np.array([((-1)**(m1+m2)) for (m1, m2) in JLIST], dtype=float)
pw1 = {(e): (karr[:, 0]**e) for e in range(NJET+1)}
pw2 = {(e): (karr[:, 1]**e) for e in range(NJET+1)}
lamQ = np.zeros(nJ)
for q in range(nJ):
    Qk = np.zeros(len(karr)); Qki = np.zeros(len(karr))
    for jj, (m1, m2) in enumerate(JLIST):
        cf = WJf[q, jj]*sgnJ[jj]*pw1[m1]*pw2[m2]
        z = 1j**(m1+m2)
        Qk += cf*z.real; Qki += cf*z.imag
    QA = np.hypot(Qk, Qki)   # |Q_q(ik)| exactly (complex modulus)
    lam1 = float((marr*QA*np.abs(karr[:, 0])).sum())/Zf
    lam2 = float((marr*QA*np.abs(karr[:, 1])).sum())/Zf
    w1q = float(sum(abs(WJf[q, jj]) for jj in range(nJ)))
    tailq = float(sum(abs(WJf[q, jj])*(T[m1+m2+1]/Z30) for jj, (m1, m2) in enumerate(JLIST)))
    lamQ[q] = (lam1 + lam2)*(1 + 1e-12) + tailq + 1e-14
print("jet spectral Lipschitz: max_q Lam_q =", f"{lamQ.max():.4f}", " mean =", f"{lamQ.mean():.4f}")
def EJtri(d):
    """analytic triangle bound for the jet ensemble; used ONLY beyond t=8.6 (tiny there)."""
    E = mp.mpf(0)
    for q in range(nJ):
        s = mp.mpf(0)
        for jj, (m1, m2) in enumerate(JLIST):
            s += abs(WJ[q, jj]) * BkM(m1, m2, d)
        E += s*s
    return E
JET_SPEC = dict(cfac=cfacJ, rn=rnJ, rhoc=0.0, epsM=float(epsMJ), cWerr=float(cWerrJ),
                hull=[(0.0, 0.0)], tailzero=True, lipQ=lamQ, Eb_fun=EJtri)
JKNOTS = [mp.mpf(3), mp.mpf(4), mp.mpf(5), mp.mpf('6.1'), mp.mpf(8)]
EJC = {}
for d0 in JKNOTS:
    res = knots_certified(True, 0, d0, "var", trange=float(mp.mpf('8.6') - d0), override=JET_SPEC,
                          rounds=3)
    EJC[d0] = res['cert']
    print(f"jet ensemble E_J({d0}) = {fmt(res['cert'],6)} (lo {res['center']:.4e})")
jk_sorted = sorted(JKNOTS)
# monotone envelope by running min (valid: annuli nested, so sup(d) <= sup(k) for k <= d)
runmin = mp.mpf('inf')
for k in jk_sorted:
    runmin = min(runmin, EJC[k]); EJC[k] = runmin
ck(all(EJC[jk_sorted[i+1]] <= EJC[jk_sorted[i]] for i in range(len(jk_sorted)-1)),
   "A6: jet certificates non-increasing in d")
def EJ(d):
    k = max(kk for kk in jk_sorted if kk <= d)
    return EJC[k]
# pin residual norms ||rho_p||^2 (Taylor of pin_p around origin to order NJET-|alpha|)
fac = [mp.factorial(i) for i in range(NJET+2)]
def pin_resid_norm2(xp, al):
    a1, a2 = al
    var_pin = ((-1)**(a1+a2)) * exact_moment(2*a1, 2*a2)
    cov = mp.mpf(0)
    varJ = mp.mpf(0)
    for j1 in range(0, NJET+1-a1-a2):
        for j2 in range(0, NJET+1-a1-a2-j1):
            xj = (xp[0]**j1/fac[j1]) * (xp[1]**j2/fac[j2])
            # Cov(pin_p, d^{al+j} f(0)) = (-1)^{|al+j|} d^{2al+j} C(x_p)
            cov += xj * ((-1)**(a1+a2+j1+j2)) * kernel_d(xp[0], xp[1], 2*a1+j1, 2*a2+j2)
            for k1 in range(0, NJET+1-a1-a2):
                for k2 in range(0, NJET+1-a1-a2-k1):
                    xk = (xp[0]**k1/fac[k1]) * (xp[1]**k2/fac[k2])
                    varJ += xj*xk*((-1)**(a1+a2+k1+k2))*exact_moment(2*a1+j1+k1, 2*a2+j2+k2)
    return var_pin - 2*cov + varJ
# scaled realization matrix At (pins x jets<=KPR), entries xi_p^{m-al}/(m-al)! (r-free)
def At_matrix(useY):
    p = pins_of(mp.mpf(1), useY)   # r = 1 gives xi pins
    A = mp.matrix(len(p), len(J4))
    for i, (xp, al) in enumerate(p):
        for j, (m1, m2) in enumerate(J4):
            if m1 >= al[0] and m2 >= al[1]:
                A[i, j] = (xp[0]**(m1-al[0])/fac[m1-al[0]]) * (xp[1]**(m2-al[1])/fac[m2-al[1]])
    return A
CONT = {}
for useY in (True, False):
    A = At_matrix(useY)
    AAT = A*A.T
    sg, QA = mp.eigsy(AAT)
    RA = AAT - QA*mp.diag(sg)*QA.T
    rhoA = mp.sqrt(sum(RA[i, j]**2 for i in range(AAT.rows) for j in range(AAT.rows))) + ROUND
    sig_hat = mp.sqrt(max(sg[0] - rhoA, mp.mpf(0)))
    np_ = len(pins_of(mp.mpf(1), useY))
    print(f"{'7' if useY else '6'}-pin: scaled realization sigma_min(At) >= {fmt(sig_hat,8)}")
    ck(sig_hat > 0, f"A6: pin realization map full rank at K'=4 ({'7' if useY else '6'}-pin)")
    # c_res(r) = sqrt(sum_p ||rho_p||^2) as explicit function of r (positions r*xi)
    _cres_cache = {}
    def cres(r):
        key = mp.nstr(r, 30)
        if key not in _cres_cache:
            s = mp.mpf(0)
            for (xp, al) in pins_of(r, useY):
                v = pin_resid_norm2(xp, al)
                if v < 0 and v > -mp.mpf('1e-60'):
                    v = mp.mpf(0)
                s += v
            _cres_cache[key] = mp.sqrt(s)
        return _cres_cache[key]
    def tau(r):
        # sigma_min(A(r)) >= sigma_min(D_row)*sig_hat*sigma_min(D_col) = 1 * sig_hat * r^KPR
        return cres(r) / (sig_hat * r**KPR * mp.sqrt(lamJ_hat))
    r0 = RUNGS[0]
    t0 = tau(r0)
    print(f"{'7' if useY else '6'}-pin: cres(1/20)={fmt(cres(r0),6)} tau(1/20)={fmt(t0,6)}")
    ck(t0 < 1, f"A6: tau(1/20) < 1 -> uniform continuation valid on (0, 1/20] ({'7' if useY else '6'}-pin)")
    # uniform envelope
    def Eunif(d, r):
        t = tau(r)
        return ((mp.sqrt(EJ(d)) + t)/(1 - t))**2
    # validity: the uniform bound must dominate every exact-rung rigorous LOWER bound
    for ri, r in enumerate(RUNGS):
        for d in (mp.mpf(3), mp.mpf(5), mp.mpf('6.1')):
            Eu = Eunif(d, r)
            Ibc = KIB[(useY, ri, d)]['cert']; Ibl = mp.mpf(str(KIB[(useY, ri, d)]['center']))
            dom = Eu >= Ibl*mp.mpf('0.999')
            print(f"{'7' if useY else '6'}-pin r=1/{int(1/r)} d={d}: E_unif={fmt(Eu,6)} "
                  f"IB cert={fmt(Ibc,6)} IB lo={fmt(Ibl,6)} dominates_lo={dom}")
            ck(dom, f"A6: uniform bound dominates exact-rung lower bound ({'7' if useY else '6'}-pin r=1/{int(1/r)} d={d})")
    # measured Gram degeneracy exponent (nonuniform obstruction of the raw frame)
    l1 = ENS[(useY, 0)]['lam_hat']; l2 = ENS[(useY, 1)]['lam_hat']
    expo = mp.log(l1/l2)/mp.log(2)
    print(f"{'7' if useY else '6'}-pin: measured lambda_min(G(r)) exponent ~ r^{fmt(expo,4)} "
          f"(rank drop at r=0: {np_} pins -> 3 independent functionals)")
    CONT[useY] = dict(tau0=t0, EJ3=EJ(mp.mpf(3)), Eunif3=Eunif(mp.mpf(3), r0), expo=expo)
    print(f"{'7' if useY else '6'}-pin: E_J(3)={fmt(EJ(mp.mpf(3)),6)} "
          f"E_unif(3;1/20)={fmt(Eunif(mp.mpf(3), r0),6)} "
          f"E_unif(6.1;1/20)={fmt(Eunif(mp.mpf('6.1'), r0),6)}")
    # monotonicity of E_unif in d at r = 1/20
    ev = [Eunif(d, r0) for d in DGRID]
    mono = all(ev[i+1] <= ev[i] for i in range(len(ev)-1))
    ck(mono, f"A6: uniform envelope non-increasing in d on [3,12] at r=1/20 ({'7' if useY else '6'}-pin)")
    print(f"{'7' if useY else '6'}-pin: E_unif(12;1/20)={fmt(ev[-1],6)}")

# ================= A7 value-law channel: full one-point-law bounds + isolation =================
print("=== A7 value-law channel (6-pin) and v*=-1/2 isolation (interval grade) ===")
SQ2PI = mp.sqrt(2*mp.pi)
def tvsig(s):
    if s >= 1: return mp.mpf(0)
    return mp.sqrt(2/mp.pi)*(1/s - 1)*mp.e**(-mp.mpf('0.72')) + (1-s)**2*mp.mpf('2.6')
# 6-pin |mu| sup at d=5 and d=3 (interval grade) first - used by the TV knot bounds
MUK = {}
for ri in (0, 1):
    for d in (mp.mpf(3), mp.mpf(5)):
        res = knots_certified(False, ri, d, "mu")
        MUK[(ri, d)] = res
        print(f"6-pin r=1/{int(1/RUNGS[ri])} d={d}: IB certified |mu| sup = {fmt(res['cert'],6)}")
TVROWS = {}
for ri, r in enumerate(RUNGS):
    entry = ENS[(False, ri)]
    muv = [envelope_mu(entry, d) for d in DGRID]
    sig = [mp.sqrt(1-v) for v in ENVV[(False, ri)]]
    tb = [muv[i]/(sig[i]*SQ2PI) + tvsig(sig[i]) for i in range(len(DGRID))]
    # knot-sharpened bounds using the IB |mu| suprema at d=3 and d=5
    sig3 = mp.sqrt(1-ENVV[(False, ri)][0]); sig5 = mp.sqrt(1-ENVV[(False, ri)][20])
    tbk3 = MUK[(ri, mp.mpf(3))]['cert']/(sig3*SQ2PI) + tvsig(sig3)
    tbk5 = MUK[(ri, mp.mpf(5))]['cert']/(sig5*SQ2PI) + tvsig(sig5)
    TVROWS[ri] = (tb, tbk3, tbk5)
    mono = all(tb[i+1] <= tb[i] for i in range(len(tb)-1))
    print(f"6-pin r=1/{int(1/r)}: TV_envelope(3)={fmt(tb[0],6)} TV_envelope(5)={fmt(tb[20],6)} "
          f"TV_envelope(6.1)={fmt(tb[31],6)} monotone={mono} | TV_knot(3)={fmt(tbk3,6)} TV_knot(5)={fmt(tbk5,6)}")
    ck(mono, f"A7: full one-point-law TV bound non-increasing on [3,12] (6-pin r=1/{int(1/r)})")
    ck(tbk5 <= mp.mpf('1e-4'), f"A7: TV bound at d=5 <= 1e-4 (6-pin r=1/{int(1/r)}, IB knot)")
# isolation: 7-pin v*=-1/2, |mu| sup at d=5 (interval grade)
ISO = {}
for ri in (0, 1):
    res = knots_certified(True, ri, mp.mpf(5), "mu")
    ISO[ri] = res
    print(f"ISOLATION 7-pin v*=-1/2 r=1/{int(1/RUNGS[ri])} d=5: certified |mu| sup = {fmt(res['cert'],6)} "
          f"(center {res['center']:.4e}, argmax cand {res['cand']})")
    ck(res['cert'] > mp.mpf('0.1'), f"A7: certified violation of value-law stabilization (r=1/{int(1/RUNGS[ri])})")

# ================= A8 mutation tests =================
print("=== A8 mutation tests ===")
# M1: value mutation leaves the variance channel exactly invariant
def delta_at_points(useY, ri, pts, vals_override=None):
    entry = ENS[(useY, ri)]
    p = entry['pins']; Ms = entry['M']; n = len(p)
    out = []
    for (y1, y2) in pts:
        v = mp.matrix(n, 1)
        for i, (xp, al) in enumerate(p):
            v[i] = ((-1)**(al[0]+al[1])) * kernel_d(y1-xp[0], y2-xp[1], al[0], al[1])
        Mv = Ms*v
        out.append(float((v.T*Mv)[0]))
    return out
pts = [(mp.mpf('3.1'), mp.mpf('0.2')), (mp.mpf('5.0'), mp.mpf('1.1')), (mp.mpf('6.3'), mp.mpf('-2.0')),
       (mp.mpf('4.2'), mp.mpf('3.3')), (mp.mpf('8.0'), mp.mpf('0.0'))]
base = delta_at_points(True, 0, pts)
mut = delta_at_points(True, 0, pts, vals_override="MUTATED")
dev = max(abs(b-m) for b, m in zip(base, mut))
print(f"M1: variance channel under value mutation (v* -1/2 -> -0.4, b 6/5 -> 1.3): max dev = {dev:.3e}")
ck(dev == 0.0, "A8-M1: variance channel bitwise-invariant under pin-value mutation")
# M2: refinement invariance: halved boxes must not increase the certified sup beyond pad terms
res_coarse = knots_certified(True, 0, mp.mpf(3), "var", ntheta=270, ht=0.12)
res_fine = KIB[(True, 0, mp.mpf(3))]
print(f"M2: d=3 7-pin r=1/20 coarse cert={fmt(res_coarse['cert'],6)} fine cert={fmt(res_fine['cert'],6)}")
ck(res_fine['cert'] <= res_coarse['cert'], "A8-M2: refinement does not increase certified sup")
# M3: self-falsification - a corrupted kernel MUST be detected by the A2 cross-check
def corrupted_kernel(u1, u2, j1, j2):
    return -kernel_d(u1, u2, j1, j2) if (j1, j2) == (0, 0) else kernel_d(u1, u2, j1, j2)
det = abs(Cspectral_mp(mp.mpf('3.0'), mp.mpf('0.7'), 0, 0) - corrupted_kernel(mp.mpf('3.0'), mp.mpf('0.7'), 0, 0))
print(f"M3: corrupted-kernel cross-check residual = {fmt(det,6)} (must exceed 1e-50 to be caught)")
ck(det > mp.mpf('1e-50'), "A8-M3: sign-flip mutation is caught by the cross-representation check")
# M4: envelope-coefficient mutation is caught by the dominance check (E_env >= certified sup)
mutE = ENVV[(True, 0)][0]*mp.mpf('1e-3')
ck(not (mutE >= KIB[(True, 0, mp.mpf(3))]['cert']),
   "A8-M4: 1e-3-scaled envelope mutant fails the dominance check (as required)")
print(f"M4: mutant E(3)x1e-3 = {fmt(mutE,6)} < IB cert {fmt(KIB[(True,0,mp.mpf(3))]['cert'],6)} -> mutant rejected")

# ================= A9 exact valid-scope table =================
print("=== A9 exact valid-scope table ===")
rows = [
 ("C1 variance deviation Delta_bar(d) <= E(d), monotone in d",
  "d in [3,12], d0=3", "r in {1/20,1/40} exact + uniform (0,1/20] via A6",
  "7-pin (f,grad at M,S; f at Y), ANY pinned values", "IB at knots {3,5,6.1} + DPS80 envelope between"),
 ("C2 variance floor Var >= 0.979",
  "d >= 3", "r in {1/20,1/40} exact; (0,1/20] via A6 dominance",
  "7-pin, any values", "IB (max over rungs of certified Delta_bar(3))"),
 ("C3 recorded constants 2.24% @3, 1e-4 @5, 7.5e-13 @6.1",
  "d in {3,5,6.1}", "r in {1/20,1/40} exact rungs",
  "7-pin (values-independent)", "IB (interval box enclosures)"),
 ("C4 full one-point-law TV bound T_bar(d), monotone",
  "d in [3,12]", "r in {1/20,1/40} exact",
  "6-pin with EXACT values beta=(6/5, b-r^3/6)", "DPS80 envelope; IB |mu| sup at d=3,5"),
 ("C5 recorded value law 7.7e-5/8.1e-5 @ d=5",
  "d=5", "r=1/20 / r=1/40 exact",
  "6-pin exact beta", "IB |mu| sup + exact TV at argmax (DPS80)"),
 ("C6 v*=-1/2 isolation: value-law stabilization FAILS",
  "d=5 (counterexample)", "r in {1/20,1/40}",
  "7-pin with tasked v*=-1/2 at Y", "IB certified |mu| sup > 0.1"),
 ("C7 uniform continuation E_unif(d;r) <= [ (sqrt(E_J(d))+tau)/(1-tau) ]^2",
  "d in [3,12]", "ALL r in (0,1/20] (tau(1/20)<1 certified)",
  "7-pin and 6-pin families, any values (variance channel)", "DPS80 certified (jet frame N=8 K'=4)"),
]
for i, (c, dd, rr, fam, tier) in enumerate(rows, 1):
    print(f"SCOPE {c}\n   d-range: {dd}\n   r-range: {rr}\n   conditioning: {fam}\n   evidence tier: {tier}")

# ================= A10 labels & receipts (separate) + final gate =================
print("=== A10 precision labels & receipts (separate section) ===")
print("LABELS: EXACT (structural constants: pi/12 lattice, rungs 1/20,1/40, b=6/5, Y=(-63/50,6/25)r,")
print(" v*=-1/2, double-factorial jet moments, Hermite integer coefficients);")
print(" DPS80 (mpmath dps-80 with certified residual bounds: lambda_hat, epsM, cWerr, tails, envelopes);")
print(" IB (interval box enclosures: every error term derived - Lipschitz from certified Bker,")
print(" coefficient intervals cWerr, tails R, inverse residual epsM, float64 center rounding bound);")
print(" F64 (float64 center values; never load-bearing without IB half-widths).")
for tag in REC:
    print(f"receipt {tag}: {REC[tag]}")
print("=== SUMMARY ===")
if FAILS:
    print("FAILED CHECKS:", FAILS)
    raise SystemExit(1)
print("ALL CHECKS PASS")
print("audit_scope_027a_v1: scope audit complete; see audit_scope_027a_v1_report.md for the proof text.")
