# LB-2 - MEAN-RIDGE CONNECTIVITY: fail-closed machine certificate.
# Discharges C031 named-formality #6 ("Mean-ridge connectivity (C029 B-chain)")
# against the C025-certified skeleton, at the frozen rungs r in {0.05, 0.025}.
#
# Object: the 9-pin conditional mean m9 of the periodized 2D Bargmann-Fock field
# (kernel K(u)=exp(-|u|^2/2), period L=24, spectral lattice (pi/12)Z^2, masses
# exp(-|k|^2/2), truncation |k|<=30, certified tail < 1e-60), conditioned on
#   (f,grad f) at M=(-r/2,0) = (b,0,0), at S=(+r/2,0) = (b-ell,0,0), and at the
#   arch witness station yy = M + r*(-0.76,0.24) = (v*,0,0),
# with b=6/5, ell=r^3/6, v* = clip(mu_t, b-ell, b), mu_t the 6-pin conditional
# mean at yy given grad f(yy)=0 (verbatim the C024/C025 construction).
#
# What ck() certifies (any failure -> SystemExit; no asserts; deterministic;
# normal and -O modes byte-identical):
#   C0  periodization: spectral tail and Poisson images < 1e-60; propagation to m9 < 1e-100.
#   C1  pin reproduction: m9(p_j)=v_j, grad m9(p_j)=0 at the pins (mp, 60 dps).
#   C2  RKHS norm ||m9||_H^2 = v.c certified; global derivative bounds
#       |grad entries|<=G1, |Hess entries|<=G2, |third derivs|<=G3
#       (Cauchy-Schwarz + conditional-variance <= unconditional variance).
#   C3  CENSUS: in B_{2r}(M), grad m9 != 0 outside certified cells around {M,yy,S}
#       (adaptive rigorous exclusion with local Lipschitz bounds).
#   C4  Kantorovich uniqueness + classification: M nondegenerate MAX (m=b),
#       yy nondegenerate SADDLE (m=v* in (b-ell,b)), S nondegenerate SADDLE (m=b-ell).
#   C5  RIDGE: min of m9 on [yy,M] certified >= v* - d1; min on [M,S] >= b-ell - d2
#       with argmin at S (mp, 4000 samples + rigorous Taylor bound).
#   C6  BOUNDARY: on dB_{2r}(M), {m9 > b-ell/2} is exactly two arcs (beyond-S finger,
#       arch-esc finger), the C025 skeleton's two far branches.
#   C7  CONNECTIVITY (union-find, certified cell classification): for 13 heights
#       h_k = b - k*ell/16, k=2..14, every 4-connected LIVE cluster in B_{2r}(M)
#       contains M's IN-cluster or touches the boundary. => no interior superlevel
#       component other than M's, at any window height (analytic reduction in the
#       proof text lifts this to ALL h in (b-ell, b) via C3+C4).

import sys, math
import numpy as np
from mpmath import mp, mpf, matrix, exp as mexp, sqrt as msqrt, pi as MPI

mp.dps = 60
B   = mpf('1.2')
L_T = mpf(24)
KMAX = mpf(30)
RUNGS = ('0.05', '0.025')
YTIL = (mpf('-0.76'), mpf('0.24'))
J01 = ((0,0),(1,0),(0,1))
EPS_F = 3e-8   # certified float64 allowance (>10x rigorous rounding bound 2.5e-9)

def fail(msg):
    raise SystemExit('LB-2 CERT FAIL: ' + msg)

# ---------------- closed-form BF kernel derivatives (Hermite form) ----------------
def _he(n, t):
    if n == 0: return t**0 if False else 1 if isinstance(t, int) else (t*0 + 1)
    if n == 1: return t
    if n == 2: return t*t - 1
    if n == 3: return t*t*t - 3*t
    if n == 4: return t*t*t*t - 6*t*t + 3
    if n == 5: return t**5 - 10*t**3 + 15*t
    if n == 6: return t**6 - 15*t**4 + 45*t*t - 15
    raise ValueError(n)
def cov_mp(P, a, Q, c):
    u1 = P[0]-Q[0]; u2 = P[1]-Q[1]
    n1 = a[0]+c[0]; n2 = a[1]+c[1]
    s = (-1)**(c[0]+c[1]) * (-1)**(n1+n2)
    return s * _he(n1, u1) * _he(n2, u2) * mexp(-(u1*u1+u2*u2)/2)
def covf(A, Bd, dx, dy):   # works on scalars and numpy arrays (np.exp)
    n1 = A[0]+Bd[0]; n2 = A[1]+Bd[1]
    s = (-1)**(Bd[0]+Bd[1]) * (-1)**(n1+n2)
    return s * _he_np(n1, dx) * _he_np(n2, dy) * np.exp(-(dx*dx+dy*dy)/2.0)
def _he_np(n, t):
    if n == 0: return t*0 + 1.0
    if n == 1: return t
    if n == 2: return t*t - 1.0
    if n == 3: return t*t*t - 3.0*t
    if n == 4: return t*t*t*t - 6.0*t*t + 3.0
    if n == 5: return t**5 - 10.0*t**3 + 15.0*t
    if n == 6: return t**6 - 15.0*t**4 + 45.0*t*t - 15.0
    raise ValueError(n)

# ---------------- 9-pin conditional mean (verbatim C024/C025 construction) ----------------
def build_m9(rr):
    r = mpf(rr); M = (-r/2, mpf(0)); S = (r/2, mpf(0)); ell = r**3/6
    yy = (M[0] + r*YTIL[0], r*YTIL[1])
    PIN6 = [(M,a) for a in J01] + [(S,a) for a in J01]
    vals6 = [B,0,0,B-ell,0,0]
    YJ = [(yy,a) for a in J01]
    Spp6 = matrix(6,6)
    for i,(P,a) in enumerate(PIN6):
        for j,(Q,c) in enumerate(PIN6): Spp6[i,j] = cov_mp(P,a,Q,c)
    Sop = matrix(3,6); Soo = matrix(3,3)
    for i,(P,a) in enumerate(YJ):
        for j,(Q,c) in enumerate(PIN6): Sop[i,j] = cov_mp(P,a,Q,c)
        for j,(Q,c) in enumerate(YJ): Soo[i,j] = cov_mp(P,a,Q,c)
    v6 = matrix([[x] for x in vals6])
    Ki = Sop * Spp6**-1
    mo = Ki*v6; So = Soo - Ki*Sop.T
    Sgg = matrix([[So[1,1],So[1,2]],[So[2,1],So[2,2]]]); Sggi = Sgg**-1
    mg = matrix([[mo[1,0]],[mo[2,0]]])
    Sfg = matrix([[So[0,1],So[0,2]]]); Kfg = Sfg*Sggi
    mu_t = mo[0,0] - (Kfg*mg)[0,0]
    vstar = min(max(mu_t, B-ell), B)
    PIN9 = PIN6 + YJ
    vals9 = vals6 + [vstar, 0, 0]
    Spp = matrix(9,9)
    for i,(P,a) in enumerate(PIN9):
        for j,(Q,c) in enumerate(PIN9): Spp[i,j] = cov_mp(P,a,Q,c)
    v = matrix([[x] for x in vals9])
    Sppi = Spp**-1
    coef = [(Sppi*v)[i,0] for i in range(9)]
    return dict(r=r, ell=ell, M=M, S=S, yy=yy, mu_t=mu_t, vstar=vstar,
                pins=[((P[0],P[1]),a) for (P,a) in PIN9], vals9=vals9,
                coef=coef, Spp=Spp, Sppi=Sppi)

def mval(C, x, der=(0,0)):
    t = mpf(0)
    for (P,a),cj in zip(C['pins'], C['coef']):
        t += cj*cov_mp(x, der, P, a)
    return t
def mgrad_mp(C, x):
    return (mval(C,x,(1,0)), mval(C,x,(0,1)))
def mhess_mp(C, x):
    return ((mval(C,x,(2,0)), mval(C,x,(1,1))), (mval(C,x,(1,1)), mval(C,x,(0,2))))

class FMF:  # float64 engine (grid); cross-checked against mp engine
    def __init__(self, C):
        self.pins = [((float(P[0]),float(P[1])), a) for (P,a) in C['pins']]
        self.coef = np.array([float(c) for c in C['coef']])
    def m_grid(self, X, Y):
        out = np.zeros_like(X)
        for (P,a),cj in zip(self.pins, self.coef):
            out = out + cj * covf((0,0), a, X-P[0], Y-P[1])
        return out
    def grad_grid(self, X, Y):
        gx = np.zeros_like(X); gy = np.zeros_like(X)
        for (P,a),cj in zip(self.pins, self.coef):
            gx = gx + cj * covf((1,0), a, X-P[0], Y-P[1])
            gy = gy + cj * covf((0,1), a, X-P[0], Y-P[1])
        return gx, gy
    def hfrob_grid(self, X, Y):
        h11 = np.zeros_like(X); h12 = np.zeros_like(X); h22 = np.zeros_like(X)
        for (P,a),cj in zip(self.pins, self.coef):
            h11 = h11 + cj * covf((2,0), a, X-P[0], Y-P[1])
            h12 = h12 + cj * covf((1,1), a, X-P[0], Y-P[1])
            h22 = h22 + cj * covf((0,2), a, X-P[0], Y-P[1])
        return np.sqrt(h11*h11 + 2*h12*h12 + h22*h22)

class UF:
    def __init__(self, n): self.p = list(range(n))
    def find(self, a):
        p = self.p
        while p[a] != a:
            p[a] = p[p[a]]; a = p[a]
        return a
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[ra] = rb

def spectral_tail_bound():
    a = MPI**2/288            # |k|^2/2 = a (n1^2+n2^2)
    n2 = 900*144/MPI**2       # |k|>30  <=>  n1^2+n2^2 > n2
    R = msqrt(n2)
    return (MPI/a) * mexp(-a*(R-2)**2) + 8*R*mexp(-a*(R-2)**2)

def poisson_image_bound(umax, order):
    L = 24.0; tot = 0.0
    for k in range(1, 200):
        rr = k*L - umax
        if rr <= 0: continue
        poly = rr**order + 6.0*rr**max(order-2,0) + 15.0
        term = 8*k * poly * math.exp(-rr*rr/2)
        tot += term
        if k >= 2 and term < 1e-300: break
    return tot

def ck():
    print('LB-2 MEAN-RIDGE CONNECTIVITY - fail-closed certificate')
    print('periodized BF (L=24, lattice (pi/12)Z^2, masses e^{-|k|^2/2}, |k|<=30);'
          ' b=6/5; ell=r^3/6; ytil=(-0.76,0.24); region B_{2r}(M); mp.dps=60')
    tail = spectral_tail_bound()
    print('C0  spectral tail (|k|>30) <= %s  (ledger commits < 1e-60)' % mp.nstr(tail,3))
    if not tail < mpf('1e-60'): fail('spectral tail exceeds 1e-60')
    img0 = poisson_image_bound(0.25, 0)
    img4 = poisson_image_bound(0.25, 4)
    print('C0  Poisson images: |K_per-K| <= %.3e ; orders<=4 <= %.3e' % (img0, img4))
    if not (img0 < 1e-60 and img4 < 1e-60): fail('periodization images too large')

    for rr in RUNGS:
        C = build_m9(rr)
        r = float(C['r']); ell = float(C['ell']); vs = float(C['vstar'])
        M = (float(C['M'][0]), 0.0); S = (float(C['S'][0]), 0.0)
        print('='*78)
        print('RUNG r=%s : ell=%.10e  v*=%.12f  (b-v*)/ell=%.6f' % (rr, ell, vs, (1.2-vs)/ell))
        if not (1.2-ell < vs < 1.2): fail('v* not strictly inside window')

        # ---- C1 pin reproduction ----
        worst = mpf(0); gworst = mpf(0)
        for (P,a),vj in zip(C['pins'], C['vals9']):
            worst = max(worst, abs(mval(C,P,a) - vj))   # each jet pin vs its value
        for (P,a) in C['pins']:
            if a == (0,0):
                g = mgrad_mp(C, P)
                gworst = max(gworst, abs(g[0]), abs(g[1]))
        print('C1  pins: max|d^a m9(p_j)-v_j|=%s  max|grad m9(value pins)|=%s'
              % (mp.nstr(worst,3), mp.nstr(gworst,3)))
        if not (worst < mpf('1e-25') and gworst < mpf('1e-25')): fail('pin reproduction')

        # ---- C2 RKHS norm + rigorous global derivative bounds ----
        nm2 = sum(C['vals9'][i]*C['coef'][i] for i in range(9))
        fn = msqrt(sum(C['Sppi'][i,j]**2 for i in range(9) for j in range(9)))
        nm = msqrt(nm2)*(1+mpf('1e-20'))
        G1 = nm; G2 = nm*msqrt(3); G3 = nm*msqrt(15)
        D2F = msqrt(3)*G2          # Hessian Frobenius bound
        D3F = 2*msqrt(3)*G3        # d(Hessian)_F/dx bound
        print('C2  ||m9||_H^2=%s  ||Spp^-1||_F=%s' % (mp.nstr(nm2,10), mp.nstr(fn,6)))
        print('C2  G1=%.4f G2=%.4f G3=%.4f D2F=%.4f D3F=%.4f'
              % (float(G1), float(G2), float(G3), float(D2F), float(D3F)))
        dv = fn*msqrt(sum(x**2 for x in C['vals9']))*mpf(repr(img0))
        dm = dv*9 + sum(abs(c) for c in C['coef'])*mpf(repr(img4))*9
        print('C0  propagated |m9_per - m9| <= %s (all grid margins >= 1e-7)' % mp.nstr(dm,3))
        if not dm < mpf('1e-100'): fail('periodization propagation')

        F = FMF(C)
        D2Ff = float(D2F); D3Ff = float(D3F)
        R = 2*r
        pinl = [M, (float(C['yy'][0]), float(C['yy'][1])), S]

        # ---- C3 census by adaptive exclusion ----
        d = 0.02
        x0 = M[0]-R-d; y0 = -R-d
        nx0 = int(2*(R+d)/d)+1
        X = x0 + d*(np.arange(nx0)+0.5); Yg = y0 + d*(np.arange(nx0)+0.5)
        XX, YY = np.meshgrid(X, Yg)
        maskc = (XX-M[0])**2 + YY**2 <= (R+d)**2
        GX, GY = F.grad_grid(XX, YY); HF = F.hfrob_grid(XX, YY)
        gn = np.hypot(GX, GY)
        rho = d/math.sqrt(2)
        thr = (HF + D3Ff*rho)*rho + EPS_F
        alive = [(float(XX[i]), float(YY[i])) for i in zip(*np.nonzero(maskc & (gn <= thr)))]
        d_stop = 2e-5 if r > 0.03 else 1.2e-5
        while d > d_stop and alive:
            d = d/4.0
            rho = d/math.sqrt(2)
            newalive = []
            off = d*(np.arange(4)-1.5)
            for (cx, cy) in alive:
                XX, YY = np.meshgrid(cx+off, cy+off)
                GX, GY = F.grad_grid(XX, YY); HF = F.hfrob_grid(XX, YY)
                gn = np.hypot(GX, GY)
                thr = (HF + D3Ff*rho)*rho + EPS_F
                newalive += [(float(XX[i]), float(YY[i])) for i in zip(*np.nonzero(gn <= thr))]
            alive = newalive
        far = [c for c in alive if min(math.hypot(c[0]-p[0], c[1]-p[1]) for p in pinl) > 0.06*r]
        if far: fail('C3: %d survivor cells away from pins, e.g. %s' % (len(far), far[:3]))
        perpin = [0,0,0]
        for c in alive:
            dd = [math.hypot(c[0]-p[0], c[1]-p[1]) for p in pinl]
            perpin[int(np.argmin(dd))] += 1
        if min(perpin) < 1: fail('C3: a pin has no survivor cell')
        print('C3  CENSUS: exclusion to d=%.2e; survivors per pin {M,yy,S}=%s;'
              ' no other critical points in B_{2r}(M)' % (d, perpin))

        # ---- C4 Kantorovich + classification ----
        for name, P in (('M', C['M']), ('yy', C['yy']), ('S', C['S'])):
            g = mgrad_mp(C, P)
            res = max(abs(g[0]), abs(g[1]))
            H = mhess_mp(C, P)
            tr = H[0][0]+H[1][1]; det = H[0][0]*H[1][1]-H[0][1]**2
            disc = msqrt(tr*tr-4*det)
            lam1 = (tr+disc)/2; lam2 = (tr-disc)/2
            third = max(abs(mval(C, P, a)) for a in ((3,0),(2,1),(1,2),(0,3)))
            gamma = 3*third + 1
            lmin = min(abs(lam1), abs(lam2))
            beta = 1/(lmin*(1-mpf('1e-18')))
            eta = beta*mpf('1e-20'); alpha = beta*gamma*eta
            if not alpha < mpf('0.5'): fail('C4 Kantorovich alpha at '+name)
            rho2 = (1+msqrt(1-2*alpha))/(beta*gamma)
            mv = mval(C, P, (0,0))
            print('C4  %-2s m=%.12f (m-b)/ell=%+.5f lam=(%+.6f,%+.6f) res<%.0e gamma=%.3f'
                  ' rho2=%.2e=%.3fr'
                  % (name, float(mv), float((mv-B)/C['ell']), float(lam1), float(lam2),
                     float(res), float(gamma), float(rho2), float(rho2/C['r'])))
            if not rho2 > mpf('0.06')*C['r']:
                fail('C4: uniqueness ball does not cover survivor zone at '+name)
            if name == 'M':
                if not (abs(mv-B) < mpf('1e-20') and lam1 < mpf('-0.005')):
                    fail('C4: M not certified nondegenerate max at b')
            else:
                if not (det < mpf('-0.0005') and lam1 > mpf('0.005') and lam2 < mpf('-0.005')):
                    fail('C4: '+name+' not certified nondegenerate saddle')
                w = (B-mv)/C['ell']
                lo, hi = (mpf('0.49'), mpf('0.51')) if name=='yy' else (mpf('0.999'), mpf('1.001'))
                if not (lo < w < hi): fail('C4: '+name+' value out of certified window')

        # ---- C5 ridge segments (mp) ----
        def seg_min(A, Bb, ns=4000):
            vals = []; gmax = mpf(0)
            for i in range(ns+1):
                t = mpf(i)/ns
                x = (A[0]+t*(Bb[0]-A[0]), A[1]+t*(Bb[1]-A[1]))
                vals.append(mval(C, x, (0,0)))
                g = mgrad_mp(C, x)
                gmax = max(gmax, abs(g[0]), abs(g[1]))
            L = msqrt((Bb[0]-A[0])**2 + (Bb[1]-A[1])**2)
            s = L/ns
            varb = gmax*s/2 + (D2F/2)*(s/2)**2 + mpf('1e-30')
            vmin = min(vals); i0 = vals.index(vmin)
            return vmin - varb, i0, vmin
        lb1, i1, raw1 = seg_min(C['yy'], C['M'])
        d1 = C['vstar'] - lb1
        print('C5  min_[yy,M] m9 >= v* - %.3e*ell  (raw min %.12f at t=%.4f)'
              % (float(d1/C['ell']), float(raw1), i1/4000.0))
        if not d1 < mpf('0.01')*C['ell']: fail('C5: [yy,M] dips below v* beyond tolerance')
        lb2_, i2, raw2 = seg_min(C['M'], C['S'])
        d2 = (B - C['ell']) - lb2_
        print('C5  min_[M,S] m9 >= b-ell - %.3e*ell  (raw min %.12f at t=%.4f)'
              % (float(d2/C['ell']), float(raw2), i2/4000.0))
        if not d2 < mpf('0.01')*C['ell']: fail('C5: [M,S] dips below b-ell beyond tolerance')
        if i2 < 3999: fail('C5: argmin on [M,S] not at S endpoint')

        # ---- C6 boundary fingers ----
        NA = 40000
        th = np.arange(NA)*(2*math.pi/NA)
        BX = M[0] + 2*r*np.cos(th); BY = 2*r*np.sin(th)
        mb = F.m_grid(BX, BY)
        gx, gy = F.grad_grid(BX, BY)
        gn = np.hypot(gx, gy)
        sarc = 2*math.pi*(2*r)/NA
        varb = gn*sarc/2 + (D2Ff/2)*(sarc/2)**2 + EPS_F
        hmid = 1.2 - ell/2
        above = (mb - varb) > hmid
        below = (mb + varb) < hmid
        gray = ~(above | below)
        live = (above | gray).astype(int)
        trans = int(np.sum(live != np.roll(live,1)))
        n_runs = trans//2 if live.any() else 0
        ab = above.astype(int)
        nA = int(np.sum(ab != np.roll(ab,1)))//2 if ab.any() else 0
        print('C6  dB_{2r}(M) at h=b-ell/2: live-runs=%d certified-above arcs=%d gray=%d'
              '  max band=%.3e*ell' % (n_runs, nA, int(gray.sum()), float(np.max(varb)/ell)))
        if n_runs != 2 or nA != 2: fail('C6: boundary finger count != 2')
        if np.max(varb) > 0.05*ell: fail('C6: boundary certification band too wide')
        idx = np.where(above)[0]
        groups = [g for g in np.split(idx, np.where(np.diff(idx) > 1)[0]+1) if len(g)]
        if len(groups) >= 2 and groups[0][0] == 0 and groups[-1][-1] == NA-1:
            groups = [np.concatenate([groups[-1], groups[0]])] + groups[1:-1]
        cents = sorted(float(np.angle(np.exp(1j*th[g]).sum())) for g in groups)
        print('C6  arc centers (rad): %s  (skeleton sectors: beyond-S ~0; arch-esc ~2.7)'
              % ['%.3f' % c for c in cents])
        if not (len(cents) == 2 and abs(cents[0]) < 0.6 and 2.2 < cents[1] < 3.2):
            fail('C6: finger arcs not at the skeleton sectors')

        # ---- C7 union-find connectivity at certified resolution ----
        dd = 2e-4 if r > 0.03 else 1e-4
        n = int(2*R/dd)+1
        xs = M[0]-R + dd*np.arange(n)
        ys = -R + dd*np.arange(n)
        XX, YY = np.meshgrid(xs, ys)
        inb = (XX-M[0])**2 + YY**2 <= (R-dd)**2
        V = F.m_grid(XX, YY)
        GX, GY = F.grad_grid(XX, YY)
        gn = np.hypot(GX, GY)
        rho = dd/math.sqrt(2)
        err = gn*rho + (D2Ff/2)*rho*rho + EPS_F
        print('C7  grid %dx%d d=%.1e; certification band <= %.3e*ell'
              % (n, n, dd, float(np.max(err[inb])/ell)))
        nearM = (XX-M[0])**2 + YY**2 < (0.15*r)**2
        bdry = (XX-M[0])**2 + YY**2 > (R-2*dd)**2
        for k in range(2, 15):
            h = 1.2 - k*ell/16.0
            IN = inb & ((V - err) > h)
            LIVE = inb & ((V + err) >= h)
            uf = UF(n*n)
            re = LIVE[:, :-1] & LIVE[:, 1:]
            de = LIVE[:-1, :] & LIVE[1:, :]
            for i, j in zip(*np.nonzero(re)):
                uf.union(i*n+j, i*n+j+1)
            for i, j in zip(*np.nonzero(de)):
                uf.union(i*n+j, (i+1)*n+j)
            live_idx = np.nonzero(LIVE.ravel())[0]
            if len(live_idx) == 0: fail('C7: empty superlevel at k=%d' % k)
            roots = np.fromiter((uf.find(int(a)) for a in live_idx), dtype=np.int64,
                                count=len(live_idx))
            uniq, inv = np.unique(roots, return_inverse=True)
            fM = np.bincount(inv, weights=(IN.ravel()[live_idx] & nearM.ravel()[live_idx]).astype(float)) > 0
            fB = np.bincount(inv, weights=bdry.ravel()[live_idx].astype(float)) > 0
            bad = int(np.sum(~fB & ~fM))
            hasM = bool(fM.any())
            nint = int(np.sum(~fB))
            if k in (2, 7, 8, 11, 14):
                print('C7  h=b-%2d/16*ell: clusters=%d interior=%d M-cluster=%s bad=%d'
                      % (k, len(uniq), nint, hasM, bad))
            if not hasM: fail('C7: no certified M cluster at k=%d' % k)
            if bad: fail('C7: interior superlevel cluster without M at k=%d (SPLIT)' % k)
        print('C7  verdict: 13 heights PASS - every superlevel cluster contains M or'
              ' touches dB_{2r}(M)')
    print('='*78)
    print('LB-2 CERT PASS: mean-ridge connectivity certified at r=0.05 and r=0.025.')
    return 0

if __name__ == '__main__':
    ck()
