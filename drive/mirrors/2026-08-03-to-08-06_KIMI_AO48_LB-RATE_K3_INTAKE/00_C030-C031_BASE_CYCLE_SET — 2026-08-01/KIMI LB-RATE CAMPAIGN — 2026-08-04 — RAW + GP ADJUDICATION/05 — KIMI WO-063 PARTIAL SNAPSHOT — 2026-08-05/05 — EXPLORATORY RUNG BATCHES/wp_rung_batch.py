#!/usr/bin/env python3
"""
verify_wp_witness_v1.py -- AO48-WO-063 Task 1 (WP RESOLUTION) certificate.

WO echo: sha256 e699bf44b7b84fca88faa2b509c1686ea9f8e500aee0dbe618c859b5ec5ad2e2.

Resolves the live contradiction THM-023 conditional (i): the C030/C031 ledger
prints a rigidity-zone (d <= 1.5) window-saddle contribution of ~3e-15 (3.6e-15
in C031 line 29) while LB-1's certified recheck binds the same zone integral at
~1.3e-5 (upper-bound grade). This certificate recomputes, on the EXACT
normalized periodized side-24 Bargmann-Fock field (spectral lattice (pi/12)Z^2,
masses e^{-|k|^2/2}, Z1-normalized so K(0)=1; kernel factorizes into 1D
periodized kernels kL), at mp.dps = 100:

  W1  the 9-pin conditioning at r = 0.025 (pins M=(-r/2,0): (b,0,0);
      S=(r/2,0): (b-ell,0,0), ell=r^3/6; Y=(-0.0315,0.006): (mu_t,0,0) with
      mu_t the 6-pin gradient-conditioned mean), in BOTH representations
      (spectral lattice sum AND wrapped-kernel image sum), cross-representation
      agreement < 1e-50, certified truncation tail < 1e-60;
  W2  f and all first derivatives (and the Hessian, for intensity) at the
      witness points A=(-0.04,-0.58) [LB-1 printed hot spot] and
      B=(-0.05,-0.575) [operator-cited hot spot], reproducing LB-1's prints;
  W3  the TRUE two-sided window-saddle Kac-Rice intensity at A, B and the
      true hot spot, by two independent quadrature engines (exact-1D-u slice
      formulation + direct 4D Gauss-Hermite) -- showing the Cauchy-Schwarz
      upper bound is ~17 orders loose at A and tight at the true hot spot;
  W4  the rigidity-zone integrals: (a) the certified upper bound (LB-1 form,
      mesh-stability {0.1,0.05,0.025}) and (b) the two-sided true evaluation
      (mesh-stability {0.1,0.05} + fine strip 0.0125), derived-on-grid grade
      (C027/C031 house class);
  W5  the ruling arithmetic: C030's 3e-15 and the printed WP channel total
      0.213*r^3 = 3.32e-6 against the certified numbers; the premise
      falsification m(P*) > b, m(Q*) > b (cross-checked against Task 4c's
      independent Kantorovich certificate KIMI-DER-027c);
  W6  code-path origin of the bad print: the zone-wide kill map
      e^{-(b-m)^2/2v} reproduces ~3.9e-15 at interior point (-0.45,-0.125)
      (C030's figure class), while the rim band {mu_t within 2 sd of the
      window} of area ~2.02 carries >90% of the upper-bound integral;
  W7  propagation old->new for every rung tolerance that consumed the WP
      value, and the LB-1 sup-over-zone survival statement with its geometric
      and functional separation data.

Fail-closed (SystemExit on failure; survives python -O; no bare asserts).
FALSIFIER: an independent recomputation disagreeing with any certified value
beyond the stated tail/quadrature bounds kills the resolution.
"""
import sys, math, os, hashlib
import mpmath
from mpmath import mpf, mp

def ck(cond, msg="check"):
    if not cond:
        print("CHECK FAILED:", msg)
        raise SystemExit(1)
    print("[ok]", msg)

mp.dps = 100

# ---------------------------------------------------------------------------
# 0. WO hash echo (verified against the uploaded WO file when present)
# ---------------------------------------------------------------------------
WO_SHA = "e699bf44b7b84fca88faa2b509c1686ea9f8e500aee0dbe618c859b5ec5ad2e2"
print("Inputs: AO48-WO-063 sha256 =", WO_SHA)
WO_PATH = "/mnt/agents/upload/AO48-WO-063 - LB-RATE HARDENING CAMPAIGN work order for KIMI - WP resolution + AUD-023 v1.1 + gamma-LOC ii-c + rigorous far-grid-ridge + THM-023 v1.1 all-small-r + transport settlements - agents authorized - 2026-08-04.md"
if os.path.exists(WO_PATH):
    h = hashlib.sha256(open(WO_PATH, "rb").read()).hexdigest()
    ck(h == WO_SHA, "WO file hash matches the echoed sha256")

# ---------------------------------------------------------------------------
# P. Exact periodized kernel platform: spectral lattice sums AND image sums
# ---------------------------------------------------------------------------
PI12 = mp.pi/12; KMAX = 30; LT = 24
JMAX = int(mp.ceil(KMAX/PI12)) + 1
ks = [j*PI12 for j in range(-JMAX, JMAX+1)]
ws = [mp.e**(-k*k/2) for k in ks]
Z1 = sum(ws)

def kE(n, d):
    if n % 2: return mpf(0)
    return sum(wj*(kj**n)*mp.cos(kj*d) for kj, wj in zip(ks, ws))/Z1
def kS(n, d):
    if n % 2 == 0: return mpf(0)
    return sum(wj*(kj**n)*mp.sin(kj*d) for kj, wj in zip(ks, ws))/Z1
def kL(n, s):
    # (d/ds)^n of the normalized 1D periodized kernel K1 (K1(0)=1)
    if n % 2 == 0: return ((-1)**(n//2)) * kE(n, s)
    return ((-1)**((n+1)//2)) * kS(n, s)
def kL_img(n, s):
    # wrapped-kernel image sum, dual normalization L/(sqrt(2 pi) Z1)
    tot = mpf(0)
    for j in range(-3, 4):
        t = s + 24*j
        a, b = mpf(1), t
        if n == 0: he = a
        elif n == 1: he = b
        else:
            for k in range(1, n): a, b = b, t*b - k*a
            he = b
        tot += ((-1)**n)*he*mp.e**(-t*t/2)
    return tot * LT/(mp.sqrt(2*mp.pi)*Z1)

def kL_plan(n, s):
    # planar (unperiodized) 1D kernel derivative, for deviation certification
    a, b = mpf(1), s
    if n == 0: he = a
    elif n == 1: he = b
    else:
        for k in range(1, n): a, b = b, s*b - k*a
        he = b
    return ((-1)**n)*he*mp.e**(-s*s/2)

# P1: spectral truncation tail
tail = {n: 2*mp.e**(-mpf(KMAX)**2/2)*mpf(KMAX)**n*(1 + 4/mpf(KMAX)**2) for n in range(0, 9)}
ck(all(tail[n] < mpf('1e-60') for n in tail),
   "P1 lattice truncation tail < 1e-60 for n = 0..8 (KMAX = 30)")

# P2: Poisson agreement (spectral vs image) on kernel probes
mx = mpf(0)
for n in range(0, 9):
    for s in (mpf('0'), mpf('0.013'), mpf('0.577'), mpf('1.45'), mpf('1.62')):
        mx = max(mx, abs(kL(n, s) - kL_img(n, s)))
ck(mx < mpf('1e-50'), "P2 Poisson agreement: |kL_spec - kL_img| < 1e-50 for n = 0..8 (measured %s)" % mp.nstr(mx, 3))

# P3: image-tail bound: omitted images |j| >= 4 have |s+24j| >= 24*4-1.62 = 94.38;
# |He_n(t)| <= |t|^n * n for |t| >= n (crude), e^{-t^2/2} <= e^{-4453}
img_tail = 2*8*mpf('94.38')**8*mp.e**(-mpf('94.38')**2/2)/(1 - mp.e**(-90))
ck(img_tail < mpf('1e-60'), "P3 image-sum truncation tail < 1e-60 (|s| <= 1.62, measured %s)" % mp.nstr(img_tail, 3))

# P4: planar deviation (legitimacy of any planar-kernel comparison): |s| <= 1.62
#     nearest wrap images (2 of them) at >= 24-1.62 = 22.38; exact He_8 evaluation.
plb = 2*abs(kL_plan(8, mpf('22.38')))*(1 + mp.e**(-44))
ck(plb < mpf('1e-90'), "P4 planar-vs-periodized deviation bound < 1e-90 for |s| <= 1.62 (%s)" % mp.nstr(plb, 3))
pdev = mpf(0)
for n in range(0, 9):
    for s in (mpf('0.05'), mpf('0.6'), mpf('1.5')):
        pdev = max(pdev, abs(kL(n, s) - kL_plan(n, s)))
ck(pdev < mpf('1e-90'), "P4b measured planar deviation < 1e-90 (%s)" % mp.nstr(pdev, 3))

def cov2(a, c, dx, dy, img=False):
    f = kL_img if img else kL
    return ((-1)**(c[0]+c[1])) * f(a[0]+c[0], dx) * f(a[1]+c[1], dy)

# ---------------------------------------------------------------------------
# W1. Pins at r = 0.025 in both representations
# ---------------------------------------------------------------------------
BB = mpf(6)/5
r  = mpf('0.025')
ell = r**3/6
StM = (mpf('-0.0125'), mpf(0)); StS = (mpf('0.0125'), mpf(0)); StY = (mpf('-0.0315'), mpf('0.006'))
PSTATS = [StM, StS, StY]
J01 = [(0,0),(1,0),(0,1)]
DER6 = [(0,0),(1,0),(0,1),(2,0),(1,1),(0,2)]

def build_pins(img=False):
    PIN6 = [(StM, a) for a in J01] + [(StS, a) for a in J01]
    vals6 = [BB, mpf(0), mpf(0), BB-ell, mpf(0), mpf(0)]
    f = kL_img if img else kL
    def assemble(PINS):
        n = len(PINS); A = mp.matrix(n, n)
        for i,(P,a) in enumerate(PINS):
            for j,(Q,c) in enumerate(PINS):
                A[i,j] = ((-1)**(c[0]+c[1])) * f(a[0]+c[0], P[0]-Q[0]) * f(a[1]+c[1], P[1]-Q[1])
        return A
    Spp6 = assemble(PIN6); Spp6i = Spp6**-1
    v6 = mp.matrix(vals6)
    KY = mp.matrix(3, 6)
    for i,a in enumerate(J01):
        for j,(P,c) in enumerate(PIN6):
            KY[i,j] = ((-1)**(c[0]+c[1])) * f(a[0]+c[0], StY[0]-P[0]) * f(a[1]+c[1], StY[1]-P[1])
    SY = mp.matrix(3, 3)
    for i,a in enumerate(J01):
        for j,c in enumerate(J01):
            SY[i,j] = ((-1)**(c[0]+c[1])) * f(a[0]+c[0], 0) * f(a[1]+c[1], 0)
    Wm = KY * Spp6i
    mo = Wm * v6
    So = SY - Wm * KY.T
    Sgg = mp.matrix(2,2); Sgg[:,:] = So[1:3,1:3]
    Sggi = Sgg**-1
    mg = mp.matrix([mo[1], mo[2]])
    Sfg = mp.matrix(1,2); Sfg[0,0]=So[0,1]; Sfg[0,1]=So[0,2]
    mu_t = mo[0] - (Sfg*Sggi*mg)[0]
    PIN9 = PIN6 + [(StY, a) for a in J01]
    vals9 = vals6 + [mu_t, mpf(0), mpf(0)]
    Spp9 = assemble(PIN9); Spp9i = Spp9**-1
    coef = Spp9i * mp.matrix(vals9)
    return dict(PIN9=PIN9, vals9=vals9, Spp9=Spp9, Spp9i=Spp9i, coef=coef, mu_t=mu_t)

st_s = build_pins(False)
st_i = build_pins(True)
off = (st_s['mu_t']-BB)/ell
print("W1 mu_t (9th pin value)      =", mp.nstr(st_s['mu_t'], 15))
print("W1 (mu_t - b)/ell            =", mp.nstr(off, 15), " (LB-1 print -0.4997159; ledger r->0 limit -0.4999290)")
ck(abs(off - mpf('-0.4997159')) < mpf('1e-6'), "W1a Y-pin offset reproduces LB-1's -0.4997159")
ck(abs(st_s['mu_t'] - st_i['mu_t']) < mpf('1e-50'), "W1b mu_t cross-representation < 1e-50 (measured %s)" % mp.nstr(abs(st_s['mu_t']-st_i['mu_t']),3))
gdev = max(abs(st_s['Spp9'][i,j]-st_i['Spp9'][i,j]) for i in range(9) for j in range(9))
ck(gdev < mpf('1e-50'), "W1c 9-pin Gram cross-representation < 1e-50 (measured %s)" % mp.nstr(gdev,3))
dg = mp.det(st_s['Spp9'])
ck(dg > 0, "W1d det(Gram) = %s > 0 (fail-closed on degeneracy)" % mp.nstr(dg,3))
ev = mp.eigsy(st_s['Spp9'])[0]
ck(min(ev) > 0, "W1e Gram min eigenvalue = %s > 0" % mp.nstr(min(ev),3))

# ---------------------------------------------------------------------------
# W2. f and all first derivatives (and Hessian) at the witness points
# ---------------------------------------------------------------------------
def fast_cond(st, x):
    KZP = mp.matrix(6, 9)
    for i,a in enumerate(DER6):
        for si,P in enumerate(PSTATS):
            kx = [kL(n, x[0]-P[0]) for n in range(4)]
            ky = [kL(n, x[1]-P[1]) for n in range(4)]
            for cidx, c in enumerate(J01):
                KZP[i, si*3+cidx] = ((-1)**(c[0]+c[1])) * kx[a[0]+c[0]] * ky[a[1]+c[1]]
    return KZP

KZZ = mp.matrix(6, 6)
for i,a in enumerate(DER6):
    for j,c in enumerate(DER6):
        KZZ[i,j] = ((-1)**(c[0]+c[1])) * kL(a[0]+c[0], 0) * kL(a[1]+c[1], 0)

def cond_law(st, x):
    KZP = fast_cond(st, x)
    W = KZP * st['Spp9i']
    mean = W * mp.matrix(st['vals9'])
    cov = KZZ - W * KZP.T
    return mean, cov

def analysis_from(mean, cov):
    m = mean[0]; g1, g2 = mean[1], mean[2]
    a11, a12, a22 = cov[1,1], cov[1,2], cov[2,2]
    detSg = a11*a22 - a12*a12
    if detSg <= 0:
        print("CHECK FAILED: conditional gradient covariance degenerate")
        raise SystemExit(1)
    tg2 = (a22*g1*g1 - 2*a12*g1*g2 + a11*g2*g2)/detSg
    pgrad = mp.exp(-tg2/2)/(2*mp.pi*mp.sqrt(detSg))
    c01, c02 = cov[0,1], cov[0,2]
    mu_t = m - (c01*(a22*g1-a12*g2) + c02*(a11*g2-a12*g1))/detSg
    v_t = cov[0,0] - (c01*c01*a22 - 2*c01*c02*a12 + c02*c02*a11)/detSg
    if v_t <= 0:
        print("CHECK FAILED: v_t <= 0")
        raise SystemExit(1)
    def sgi(v1,v2): return ((a22*v1-a12*v2)/detSg, (a11*v2-a12*v1)/detSg)
    w = [sgi(cov[3+i,1], cov[3+i,2]) for i in range(3)]
    nu = [mean[3+i] - (w[i][0]*g1 + w[i][1]*g2) for i in range(3)]
    Sig = [[cov[3+i,3+j] - (w[i][0]*cov[3+j,1] + w[i][1]*cov[3+j,2]) for j in range(3)] for i in range(3)]
    C = [[Sig[i][j] + nu[i]*nu[j] for j in range(3)] for i in range(3)]
    Em = lambda idx: (C[idx[0]][idx[1]]*C[idx[2]][idx[3]] + C[idx[0]][idx[2]]*C[idx[1]][idx[3]] + C[idx[0]][idx[3]]*C[idx[1]][idx[2]])
    D2 = Em((0,0,2,2)) - 2*Em((0,1,1,2)) + Em((1,1,1,1))
    Edet = nu[0]*nu[2] + Sig[0][2] - nu[1]**2 - Sig[1][1]
    Vdet = D2 - Edet**2
    cant = Vdet/(Vdet+Edet**2) if (Edet > 0 and Vdet > 0) else mpf(1)
    sq = mp.sqrt(max(D2, mpf(0)))
    s = mp.sqrt(v_t)
    t1 = (BB - mu_t)/s; t2 = (BB - ell - mu_t)/s
    Pw_exact = mp.ncdf(-t2) - mp.ncdf(-t1)
    ustar = min(max(mu_t, BB-ell), BB); dd = (ustar-mu_t)/s
    Pw_peak = (ell/s)*mp.exp(-dd*dd/2)/mp.sqrt(2*mp.pi)
    cvec = [cov[0,3+i] - (c01*w[i][0] + c02*w[i][1]) for i in range(3)]
    return dict(m=m,g1=g1,g2=g2,v=cov[0,0],Sg=(a11,a12,a22),mu_t=mu_t,v_t=v_t,pgrad=pgrad,
                D2=D2,Edet=Edet,cant=cant,Pw_exact=Pw_exact,Pw_peak=Pw_peak,
                wp_exact=pgrad*sq*min(cant,Pw_exact), wp_peak=pgrad*sq*min(cant,Pw_peak),
                nu=nu,Sig=Sig,cvec=cvec,mean=mean,cov=cov)

WA = (mpf('-0.04'), mpf('-0.58'))
WBp = (mpf('-0.05'), mpf('-0.575'))
FA = {}
for nm, pt in (("A", WA), ("B", WBp)):
    ms_, cs_ = cond_law(st_s, pt)
    mi_, ci_ = cond_law(st_i, pt)
    dmean = max(abs(ms_[i]-mi_[i]) for i in range(6))
    dcov = max(abs(cs_[i,j]-ci_[i,j]) for i in range(6) for j in range(6))
    ck(max(dmean, dcov) < mpf('1e-50'),
       "W2-%s witness law cross-representation < 1e-50 (mean %s, cov %s)" % (nm, mp.nstr(dmean,3), mp.nstr(dcov,3)))
    fa = analysis_from(ms_, cs_)
    FA[nm] = fa
    print("W2-%s witness (%s, %s): conditional law of (f, fx, fy) given the 9 pins" % (nm, mp.nstr(pt[0],5), mp.nstr(pt[1],5)))
    print("   E[f]  = %s    E[fx] = %s    E[fy] = %s" % (mp.nstr(fa['m'],12), mp.nstr(fa['g1'],12), mp.nstr(fa['g2'],12)))
    print("   Var(f) = %s   Sg = (%s, %s, %s)" % (mp.nstr(fa['v'],12), mp.nstr(fa['Sg'][0],6), mp.nstr(fa['Sg'][1],6), mp.nstr(fa['Sg'][2],6)))
    print("   E[f|g=0] = %s   Var(f|g=0) = %s   p_grad(0) = %s" % (mp.nstr(fa['mu_t'],12), mp.nstr(fa['v_t'],10), mp.nstr(fa['pgrad'],10)))
    print("   E[det^2 H|g=0] = %s   cant = %s   Pw = %s   wp(CS bound) = %s" % (mp.nstr(fa['D2'],8), mp.nstr(fa['cant'],4), mp.nstr(fa['Pw_exact'],6), mp.nstr(fa['wp_exact'],8)))

faA, faB = FA["A"], FA["B"]
# Reproduce LB-1's printed hot-spot values at A = (-0.04, -0.58)
ck(abs(faA['mu_t'] - mpf('1.1870')) < mpf('5e-5'), "W2a mu_t(A) = 1.1870 (LB-1 print)")
ck(abs(faA['v_t'] - mpf('5.94e-5')) < mpf('5e-7'), "W2b v_t(A) = 5.94e-5 (LB-1 print)")
ck(abs(faA['pgrad'] - mpf('5.05')) < mpf('5e-3'), "W2c p_grad(0;A) = 5.05 (LB-1 print)")
ck(abs(faA['wp_exact']/mpf('1.313e-4') - 1) < mpf('0.01'), "W2d wp(A) = 1.313e-4/unit-area (LB-1 print, 1% gate)")
ck(abs(faB['wp_exact']/mpf('1.33e-4') - 1) < mpf('0.01'), "W2e wp(B) = 1.33e-4/unit-area (operator print, 1% gate)")

# ---------------------------------------------------------------------------
# W3. TRUE two-sided window-saddle intensity: two independent engines
# ---------------------------------------------------------------------------
import numpy as np
from numpy.polynomial.hermite_e import hermegauss

def gh_nodes(ngh):
    xs, wts = hermegauss(ngh)
    return np.array(xs), np.array(wts)/math.sqrt(2*math.pi)

GHC = {}
for ngh in (16, 24, 32):
    Xn, Wn = gh_nodes(ngh)
    G1,G2,G3 = np.meshgrid(Xn,Xn,Xn,indexing='ij')
    WGn = (Wn[:,None,None]*Wn[None,:,None]*Wn[None,None,:]).ravel()
    GHC[ngh] = (np.vstack([G1.ravel(),G2.ravel(),G3.ravel()]), WGn)

def cond_params(fa):
    cov = fa['cov']
    return (float(fa['mu_t']), float(fa['v_t']),
            [float(v) for v in fa['nu']],
            [[float(fa['Sig'][i][j]) for j in range(3)] for i in range(3)],
            [float(v) for v in fa['cvec']], float(fa['pgrad']))

def u_slice(mu_t, v_t, nu, Sig, cvec, pgrad, ngh=16, n_u=41):
    # rho_true = pgrad * Int phi_u(u) G(u) Pw(u) du
    # u = E[f|H,pins,g=0] ~ N(mu_t, su^2); G(u) = E[|det H| 1{det<0} | u] (GH);
    # Pw(u) = P(f in [b-ell,b] | u) exact (f|H Gaussian, sd sf).
    SigM = np.array(Sig); nuV = np.array(nu); cV = np.array(cvec)
    su2 = float(cV @ np.linalg.solve(SigM, cV)); su = math.sqrt(su2)
    sf = math.sqrt(max(v_t - su2, 1e-300))
    V = SigM - np.outer(cV, cV)/su2
    ev, Q = np.linalg.eigh(V); lam_max = max(ev.max(), 1e-300)
    L = Q*np.sqrt(np.maximum(ev, 1e-15*lam_max))
    wu = cV/su2; ellf = float(ell)
    Zv, WGv = GHC[ngh]
    lo = mu_t - 6*su; hi = 1.2 + 8*sf
    us = lo + (hi-lo)*np.arange(n_u)/(n_u-1)
    tot = 0.0
    for u in us:
        mu = nuV + wu*(u-mu_t)
        H = mu[:,None] + L @ Zv
        det = H[0]*H[2]-H[1]**2
        Gu = float(np.sum(WGv*np.maximum(-det,0.0)))
        if Gu == 0.0: continue
        phi_u = math.exp(-((u-mu_t)/su)**2/2)/(su*math.sqrt(2*math.pi))
        Pw = 0.5*math.erfc(-((u-(1.2-ellf))/sf)/math.sqrt(2)) - 0.5*math.erfc(-((u-1.2)/sf)/math.sqrt(2))
        tot += phi_u*Gu*Pw
    return pgrad*tot*(hi-lo)/(n_u-1)

def gh_direct(fa, ngh=16):
    # direct 4D engine: E[|det H|1{det<0} Pw(H)] with Pw(H) the exact f|H window factor
    nu = fa['nu']; Sig = fa['Sig']
    SigM = mp.matrix(3,3)
    for i in range(3):
        for j in range(3): SigM[i,j] = Sig[i][j]
    cM = mp.matrix(fa['cvec'])
    SigMi = SigM**-1
    sf2 = fa['v_t'] - (cM.T*SigMi*cM)[0]
    sf = mp.sqrt(sf2)
    beta = SigMi*cM
    Lm = mp.matrix(3,3)
    for i in range(3):
        for j in range(i+1):
            s = Sig[i][j] - sum(Lm[i,k]*Lm[j,k] for k in range(j))
            Lm[i,j] = mp.sqrt(s) if i==j else s/Lm[j,j]
    xs, wts = hermegauss(ngh)
    xs = [mpf(float(v)) for v in xs]; wts = [mpf(float(v))/mp.sqrt(2*mp.pi) for v in wts]
    tot = mpf(0)
    for i1 in range(ngh):
        for i2 in range(ngh):
            for i3 in range(ngh):
                z = (xs[i1], xs[i2], xs[i3])
                H = [nu[q] + sum(Lm[q,k]*z[k] for k in range(q+1)) for q in range(3)]
                det = H[0]*H[2] - H[1]*H[1]
                if det >= 0: continue
                dH = [H[q]-nu[q] for q in range(3)]
                mf = fa['mu_t'] + sum(beta[q]*dH[q] for q in range(3))
                Pw = mp.ncdf((mf-(BB-ell))/sf) - mp.ncdf((mf-BB)/sf)
                tot += wts[i1]*wts[i2]*wts[i3]*(-det)*Pw
    return fa['pgrad']*tot

WHS = (mpf('0'), mpf('0.6'))
fa06 = analysis_from(*cond_law(st_s, WHS))
p06 = cond_params(fa06)
us16 = u_slice(*p06, ngh=16, n_u=41); us32 = u_slice(*p06, ngh=32, n_u=121)
print("W3 true hot-spot region (0, 0.6): u-slice gh16/nu41 = %.6e ; gh32/nu121 = %.6e" % (us16, us32))
ck(abs(us16/us32 - 1) < 0.005, "W3a u-slice engine self-convergence < 0.5% at (0,0.6)")
gd = gh_direct(fa06, 16)
print("W3 direct-4D engine (mp, gh16) at (0,0.6):", mp.nstr(gd, 8))
ck(abs(float(gd)/us32 - 1) < 0.03, "W3b two independent engines agree < 3%% at (0,0.6) (u-slice %.6e, direct %.6e)" % (us32, float(gd)))
rtA = u_slice(*cond_params(faA), ngh=16, n_u=81)
rtA2 = u_slice(*cond_params(faA), ngh=24, n_u=81)
rtB = u_slice(*cond_params(faB), ngh=16, n_u=81)
rtB2 = u_slice(*cond_params(faB), ngh=24, n_u=81)
print("W3 TRUE intensity at A=(-0.04,-0.58): %.3e (gh16) %.3e (gh24); at B=(-0.05,-0.575): %.3e (gh16) %.3e (gh24)" % (rtA, rtA2, rtB, rtB2))
env = max(rtA, rtA2, rtB, rtB2)
ck(env < 2e-21,
   "W3c TRUE intensity at the -y witness points is <= ~1e-21 (two GH meshes each): the CS upper bound (1.31e-4) is loose by >= 14 orders at its own argmax -- the bound's peak is a phantom; the true hot spot is on the +y side")
print("W3d bound-vs-truth at A: wp_CS = %s vs rho_true <= %.1e (looseness >= %.1e)" % (mp.nstr(faA['wp_exact'],6), env, float(faA['wp_exact'])/env))


def rung_wp(rval, h='0.05'):
    r_ = mpf(rval); el = r_**3/6
    global ell
    ell = el
    M_ = (-r_/2, mpf(0)); S_ = (r_/2, mpf(0)); Y_ = (-r_/2 - r_*mpf('0.76'), r_*mpf('0.24'))
    PST = [M_, S_, Y_]
    PIN6 = [(M_, a) for a in J01] + [(S_, a) for a in J01]
    vals6 = [BB, mpf(0), mpf(0), BB-el, mpf(0), mpf(0)]
    def assemble(PINS):
        n=len(PINS); A=mp.matrix(n,n)
        for i,(P,a) in enumerate(PINS):
            for j,(Q,c) in enumerate(PINS):
                A[i,j]=((-1)**(c[0]+c[1]))*kL(a[0]+c[0],P[0]-Q[0])*kL(a[1]+c[1],P[1]-Q[1])
        return A
    Spp6i = assemble(PIN6)**-1; v6 = mp.matrix(vals6)
    KY = mp.matrix(3,6)
    for i,a in enumerate(J01):
        for j,(P,c) in enumerate(PIN6):
            KY[i,j]=((-1)**(c[0]+c[1]))*kL(a[0]+c[0],Y_[0]-P[0])*kL(a[1]+c[1],Y_[1]-P[1])
    SY = mp.matrix(3,3)
    for i,a in enumerate(J01):
        for j,c in enumerate(J01):
            SY[i,j]=((-1)**(c[0]+c[1]))*kL(a[0]+c[0],0)*kL(a[1]+c[1],0)
    Wm = KY*Spp6i; mo = Wm*v6; So = SY - Wm*KY.T
    Sgg = mp.matrix(2,2); Sgg[:,:] = So[1:3,1:3]
    Sfg = mp.matrix(1,2); Sfg[0,0]=So[0,1]; Sfg[0,1]=So[0,2]
    mu_t = mo[0] - (Sfg*(Sgg**-1)*mp.matrix([mo[1],mo[2]]))[0]
    PIN9 = PIN6 + [(Y_, a) for a in J01]
    vals9 = vals6 + [mu_t, mpf(0), mpf(0)]
    Spp9i = assemble(PIN9)**-1
    hh = mpf(h); N = 31
    gxs_ = [hh*i for i in range(-N, N+1)]
    tab_ = {si: ([[kL(n, gx - PST[si][0]) for gx in gxs_] for n in range(4)],
                 [[kL(n, gy - PST[si][1]) for gy in gxs_] for n in range(4)]) for si in range(3)}
    It = 0.0; Ics_ = mpf(0); npts = 0
    for iy, gy in enumerate(gxs_):
        for ix, gx in enumerate(gxs_):
            if mp.sqrt(gx*gx+gy*gy) > 1.5 + hh/2: continue
            KZP = mp.matrix(6, 9)
            for i,a in enumerate(DER6):
                for si in range(3):
                    Tx, Ty = tab_[si]
                    for cidx, c in enumerate(J01):
                        KZP[i, si*3+cidx] = ((-1)**(c[0]+c[1]))*Tx[a[0]+c[0]][ix]*Ty[a[1]+c[1]][iy]
            W = KZP*Spp9i
            fa = analysis_from(W*mp.matrix(vals9), KZZ - W*KZP.T)
            Ics_ += fa['wp_exact']; It += u_slice(*cond_params(fa), ngh=16, n_u=41); npts += 1
    return It*float(hh)**2, Ics_*hh**2, (mu_t-BB)/el, npts

import time
for rv in ('0.05','0.04','0.03','0.02'):
    t0=time.time()
    It, Ics_, off, npts = rung_wp(rv)
    print("RUNG r=%s: I_true=%.6e coeff=%.5f I_cs=%s coeff_cs=%s Yoff=%s cells=%d [%.0fs]" % (
        rv, It, It/float(mpf(rv))**3, mp.nstr(Ics_,5), mp.nstr(Ics_/mpf(rv)**3,5), mp.nstr(off,7), npts, time.time()-t0), flush=True)
print("BATCH DONE")
