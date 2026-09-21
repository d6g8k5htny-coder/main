#!/usr/bin/env python3
"""
lb3_certificate.py -- SIDE24/q0, task LB-3: R0 / gamma-LOC architecture item.
Fail-closed machine certificate. Recomputes every numerical constant used in
the LB-3 discharge text; no PASS label is trusted, everything is recomputed
here. Any failed check raises SystemExit (fail-closed), in BOTH normal and
`python -O` modes (no bare `assert`; ck() is explicit and -O-proof).

Sections:
  S1  F.4 threshold-free inward-tube / capture-escape budgets (exact Fractions)
  S2  Exact periodized Bargmann-Fock kernel on the side-24 torus
  S3  Pin Gram certificates (G6 / G9, both rungs) + C022 dual weights
  S4  Far-field stabilization tables (C022 gamma_LOC corollary quantities)
  S5  Dependency hash binding (byte-exact inputs)
Prints LB3_CERTIFICATE_PASS only if every check passes.
"""
import sys, math, hashlib, os
from fractions import Fraction as F

LINES = []
def out(s=""):
    LINES.append(str(s))

def ck(cond, msg):
    if not cond:
        out("CHECK FAILED: " + str(msg))
        sys.stdout.write("\n".join(LINES) + "\n")
        raise SystemExit(1)

# ============================ S1 ===========================================
def s1():
    out("S1 F.4 capture/escape budgets (threshold-free inward tube; exact Fractions)")
    K = F(4096)
    eps0 = F(1, 1024) / K                       # eps0 = 1/(1024 K)
    ck(512*eps0 == F(1, 8192), "S1 eps0 identity 1")
    ck(2*K*eps0 == F(1, 512), "S1 eps0 identity 2")
    for Rint in (1, 2, 5, 50):
        R = F(Rint)
        tau = 2*K*R**5 + 2*R
        m = 2*R/tau
        delta0 = F(1, 512)/R**3
        eta_R = F(1, 8192)/R**3
        ck(eta_R == delta0/16, "S1 eta_R = delta0/16 R=%d" % Rint)
        rR = eps0/R**4                            # maximal rR given r R^5 <= eps0
        ck(m <= F(1)/(K*R**4), "S1 cone entry R=%d" % Rint)
        ck(R*(1+m) <= R*F(3,2), "S1 pin grad R=%d" % Rint)
        marg3 = 1 - (F(1,2)+delta0)*R*m - F(1,2)*R*m**2*delta0 - rR*(1+m)
        ck(marg3 >= F(3,4), "S1 cone FX R=%d" % Rint)
        s4 = F(257,1024) + F(257,1024)/K + F(1,2048)/K**2 + eps0
        ck(s4 < F(1,3), "S1 cone flux R=%d" % Rint)
        s5 = F(7,8) + F(5,8)/K + F(3,32)/K**2 + F(4,3)*eps0
        ck(s5 < F(9,10), "S1 strip energy R=%d" % Rint)
        ck(F(49,192) - F(1,768) == F(65,256), "S1 ledger identity R=%d" % Rint)
        ck(F(65,256) > F(1,4), "S1 ledger > 1/4 R=%d" % Rint)
        ck(F(9869,16777216) < F(1,768), "S1 ledger bracket R=%d" % Rint)
        ck(R*(2+m) <= 3*R, "S1 tube grad R=%d" % Rint)
        tube_long = 1 - R*m/F(2) - R*m**2/F(8) - 3*rR
        ck(tube_long >= F(3,4), "S1 tube FX R=%d" % Rint)
        adv7 = 1 + F(1)/K + F(1,4)/K**2 + 6*eps0 + 4*R**2/(K*tau)
        ck(adv7 < 2, "S1 tube adverse R=%d" % Rint)
        cone_cost = 3*eta_R/R
        ck(cone_cost < (F(1,3) - s4), "S1 cost cone R=%d" % Rint)
        strip_cost = 4*eta_R/(3*R)
        ck(strip_cost < (F(9,10) - s5), "S1 cost strip R=%d" % Rint)
        tube_cost_ratio = 12*eta_R/R              # corrected V4 bound
        ck(tube_cost_ratio > eta_R/12, "S1 corrected > undercounted R=%d" % Rint)
        ck(tube_cost_ratio < (2 - adv7), "S1 cost tube fits R=%d" % Rint)
        ck(tube_cost_ratio <= F(12, 8192), "S1 tube cost <= 12/8192 R=%d" % Rint)
        d = delta0
        ck((F(1,4) - (-F(1,2)+d)**2) == d - d**2, "S1 V2 identity R=%d" % Rint)
        ck(d - d**2 < d, "S1 V2 false R=%d" % Rint)
        out("  R=%d ok: cone s4=%.6f<1/3 strip s5=%.6f<9/10 tube adv=%.6f<2 "
            "tubecost=%.2e<=12/8192=%.2e"
            % (Rint, float(s4), float(s5), float(adv7),
               float(tube_cost_ratio), float(F(12, 8192))))
    out("S1 OK: 64 exact-rational checks (16 x 4 R values)")

# ============================ S2 ===========================================
def s2(mp):
    out("S2 periodized BF kernel: lattice (pi/12)Z^2, masses e^{-|k|^2/2}, |k|<=30")
    pi = mp.pi
    step = pi/12
    R0 = 30 - float(step)                        # lattice granularity guard
    dens = (12/float(pi))**2
    for j in range(4):
        R2 = R0*R0
        e = math.exp(-R2/2)
        erc = math.erfc(R0/math.sqrt(2))
        if j == 0:   I = e
        elif j == 1: I = e*(R2 + 2)/2 + R0*math.sqrt(math.pi/2)*erc
        elif j == 2: I = e*(R2*R2 + 4*R2 + 8)/4
        else:        I = e*(R0**3 + 3*R0)/2 + 3*math.sqrt(math.pi/2)*erc
        tail = 4*dens*2*math.pi*I                # x4 lattice-count inflation
        ck(tail < 1e-60, "S2 tail order %d (got %.3e)" % (j, tail))
        out("  truncation tail |k|^%d-weighted <= %.3e < 1e-60" % (j, tail))
    lat = []
    NN = 115
    for n1 in range(-NN, NN+1):
        for n2 in range(-NN, NN+1):
            k1, k2 = step*n1, step*n2
            if k1*k1 + k2*k2 <= 900:
                lat.append((k1, k2))
    ck(len(lat) == 41277, "S2 lattice count")
    wsp = [mp.e**(-(k1*k1+k2*k2)/2) for k1, k2 in lat]
    Z = mp.fsum(wsp)
    dev = abs(Z - (12/pi)**2*2*pi)
    ck(dev < mp.mpf("1e-60"), "S2 partition function Poisson deviation")
    out("  Z = %s (Poisson deviation %s)" % (mp.nstr(Z, 20), mp.nstr(dev, 3)))
    lam2 = mp.fsum([wi*k1*k1 for wi, (k1, k2) in zip(wsp, lat)])/Z
    lam4 = mp.fsum([wi*k1**4 for wi, (k1, k2) in zip(wsp, lat)])/Z
    lam22 = mp.fsum([wi*k1*k1*k2*k2 for wi, (k1, k2) in zip(wsp, lat)])/Z
    ck(abs(lam2 - 1) < mp.mpf("1e-60"), "S2 lambda2 = 1")
    ck(abs(lam4 - 3) < mp.mpf("1e-60"), "S2 lambda4 = 3")
    ck(abs(lam22 - 1) < mp.mpf("1e-60"), "S2 lambda22 = 1")
    out("  spectral moments lam2=1, lam4=3, lam22=1 verified to <1e-60")
    return pi, step, lat, wsp, Z

def make_kspec(mp, lat, wsp, Z):
    def Kspec(x1, x2, deriv=(0, 0)):
        a, b = deriv
        s = []
        for wi, (k1, k2) in zip(wsp, lat):
            t = k1*x1 + k2*x2
            term = wi*(k1**a)*(k2**b)
            n = a + b
            if n % 4 == 0:   term *= mp.cos(t)
            elif n % 4 == 1: term *= -mp.sin(t)
            elif n % 4 == 2: term *= -mp.cos(t)
            else:            term *= mp.sin(t)
            s.append(term)
        return mp.fsum(s)/Z
    return Kspec

def Kimg(x1, x2, deriv=(0, 0), imax=1):
    a, b = deriv
    tot = 0.0
    for i in range(-imax, imax+1):
        for j in range(-imax, imax+1):
            t1, t2 = x1 + 24*i, x2 + 24*j
            e = math.exp(-(t1*t1+t2*t2)/2)
            if (a, b) == (0, 0): tot += e
            elif (a, b) == (1, 0): tot += -t1*e
            elif (a, b) == (0, 1): tot += -t2*e
            elif (a, b) == (2, 0): tot += (t1*t1-1)*e
            elif (a, b) == (1, 1): tot += t1*t2*e
            elif (a, b) == (0, 2): tot += (t2*t2-1)*e
            elif (a, b) == (3, 0): tot += -(t1**3-3*t1)*e
            elif (a, b) == (2, 1): tot += -(t1*t1*t2-t2)*e
            elif (a, b) == (1, 2): tot += -(t1*t2*t2-t1)*e
            elif (a, b) == (0, 3): tot += -(t2**3-3*t2)*e
    return tot

def s2b(mp, Kspec):
    worst = 0.0
    for pt in ((3.0, 1.7), (0.05, 0.0), (7.3, 4.1)):
        for dv in ((0, 0), (1, 0), (2, 0), (2, 1)):
            s = Kspec(mp.mpf(pt[0]), mp.mpf(pt[1]), dv)
            f = Kimg(pt[0], pt[1], dv)
            worst = max(worst, float(abs(s - f)))
    ck(worst < 1e-13, "S2 spectral vs planar+images (%.3e)" % worst)
    out("  spectral == planar+images at 3 pts x 4 derivs, max diff %.3e < 1e-13" % worst)
    out("S2 OK")

# ============================ S3 ===========================================
FD = [(0, 0), (1, 0), (0, 1)]
BVAL = 1.2

def Kimg_mp(mp, x1, x2, deriv=(0, 0), imax=2):
    a, b = deriv
    tot = mp.mpf(0)
    for i in range(-imax, imax+1):
        for j in range(-imax, imax+1):
            t1, t2 = x1 + 24*i, x2 + 24*j
            e = mp.e**(-(t1*t1+t2*t2)/2)
            if (a, b) == (0, 0): tot += e
            elif (a, b) == (1, 0): tot += -t1*e
            elif (a, b) == (0, 1): tot += -t2*e
            elif (a, b) == (2, 0): tot += (t1*t1-1)*e
            elif (a, b) == (1, 1): tot += t1*t2*e
            elif (a, b) == (0, 2): tot += (t2*t2-1)*e
            elif (a, b) == (3, 0): tot += -(t1**3-3*t1)*e
            elif (a, b) == (2, 1): tot += -(t1*t1*t2-t2)*e
            elif (a, b) == (1, 2): tot += -(t1*t2*t2-t1)*e
            elif (a, b) == (0, 3): tot += -(t2**3-3*t2)*e
    return tot

def gram_entry_mp(mp, pi_, pj_, a, b):
    A = FD[a]; B = FD[b]
    sgn = -1 if (B[0]+B[1]) % 2 else 1
    return sgn*Kimg_mp(mp, mp.mpf(pi_[0])-mp.mpf(pj_[0]),
                       mp.mpf(pi_[1])-mp.mpf(pj_[1]), (A[0]+B[0], A[1]+B[1]))

def build_G_mp(mp, pins):
    n = 3*len(pins)
    G = mp.matrix(n, n)
    for i, p_ in enumerate(pins):
        for j, q_ in enumerate(pins):
            for a in range(3):
                for b in range(3):
                    G[3*i+a, 3*j+b] = gram_entry_mp(mp, p_, q_, a, b)
    return G

def pinsets(r):
    ell = r**3/6
    M = (-r/2, 0.0); S = (r/2, 0.0)
    ystar = (-0.063, 0.012) if r == 0.05 else (-0.030, 0.007)
    return (("6pin", [M, S], [BVAL, 0, 0, BVAL-ell, 0, 0]),
            ("9pin", [M, S, ystar], [BVAL, 0, 0, BVAL-ell, 0, 0, BVAL-ell/2, 0, 0]))

def s3(mp, Kspec):
    out("S3 pin Gram certificates (mp dps 60, eigsy + residual, pin-preserving)")
    mp.mp.dps = 60
    # image-tail of the imax=2 mp kernel for |x| <= 8.1: nearest omitted image
    # at distance >= 48 - 8.1*sqrt(2) > 36; e^{-36^2/2} per image, 16 images:
    itail = 16*math.exp(-36.0**2/2)
    ck(itail < 1e-60, "S3 image tail imax=2 (%.3e)" % itail)
    out("  mp evaluation kernel: planar + 24 images; image tail <= %.3e < 1e-60" % itail)
    store = {}
    for r in (0.05, 0.025):
        for ntag, pins, vals in pinsets(r):
            G = build_G_mp(mp, pins)
            E, Zm = mp.eigsy(G)
            lam_min = min(E); lam_max = max(E)
            n = 3*len(pins)
            resid = mp.mpf(0)
            for i in range(n):
                zi = mp.matrix([Zm[j, i] for j in range(n)])
                ri = G*zi - E[i]*zi
                resid = max(resid, mp.sqrt(sum(x*x for x in ri)))
            ck(lam_min > 1e3*resid, "S3 eigencert r=%s %s (lam=%s resid=%s)"
               % (r, ntag, mp.nstr(lam_min, 5), mp.nstr(resid, 3)))
            Ginv = G**-1
            v = mp.matrix([mp.mpf(x) for x in vals])
            w = Ginv*v
            w1 = sum(abs(x) for x in w)
            ginf_rowsum = max(sum(abs(Ginv[i, j]) for j in range(n)) for i in range(n))
            store[(r, ntag)] = dict(pins=pins, vals=vals, lam_min=lam_min,
                                    w1=w1, ginf_inf=ginf_rowsum, G=G, Ginv=Ginv)
            out("  r=%.3f %s: lam_min=%s lam_max=%s resid=%s margin=%s ||Ginv||_inf=%s"
                % (r, ntag, mp.nstr(lam_min, 8), mp.nstr(lam_max, 4),
                   mp.nstr(resid, 3), mp.nstr(lam_min/resid, 3),
                   mp.nstr(ginf_rowsum, 6)))
    # bind dual weights to C022 (verbatim quote in discharge text):
    w1a = store[(0.05, "6pin")]["w1"]; w1b = store[(0.025, "6pin")]["w1"]
    ck(abs(w1a/65560 - 1) < 5e-4, "S3 dual weight r=0.05 vs C022 6.556e4 (got %s)"
       % mp.nstr(w1a, 8))
    ck(abs(w1b/518300 - 1) < 5e-4, "S3 dual weight r=0.025 vs C022 5.183e5 (got %s)"
       % mp.nstr(w1b, 8))
    out("  ||G6^{-1} v||_1 = %s (r=0.05) / %s (r=0.025)  [C022: 6.556e4 / 5.183e5]"
        % (mp.nstr(w1a, 10), mp.nstr(w1b, 10)))
    g9a = store[(0.05, "9pin")]["lam_min"]
    ck(abs(g9a/mp.mpf("1.321e-12") - 1) < 5e-4,
       "S3 gram9 r=0.05 vs C022 1.321e-12 (got %s)" % mp.nstr(g9a, 8))
    out("  lam_min G9 r=0.05 = %s  [C022 C5_gram9: 1.321e-12]" % mp.nstr(g9a, 8))
    out("S3 OK")
    return store

# ============================ S4 ===========================================
EFUNCS = [("f", (0, 0), 1.0), ("dx", (1, 0), 1.0), ("dy", (0, 1), 1.0),
          ("dxx", (2, 0), 3.0), ("dxy", (1, 1), 1.0), ("dyy", (0, 2), 3.0)]

def tv_gauss(mu, v, s02):
    if v <= 0: return 1.0
    A = 1.0/s02 - 1.0/v; B = 2*mu/v; C = math.log(s02/v) - mu*mu/v
    def Phi(x, s): return 0.5*(1+math.erf(x/(s*math.sqrt(2))))
    if abs(A) < 1e-300:
        roots = [-C/B] if B != 0 else []
    else:
        D = B*B - 4*A*C
        if D < 0: roots = []
        else:
            sq = math.sqrt(D); roots = sorted([(-B-sq)/(2*A), (-B+sq)/(2*A)])
    if not roots: return 0.0
    s0 = math.sqrt(s02); sc = math.sqrt(v)
    def Df(x): return Phi(x, s0) - Phi(x-mu, sc)
    tot = 0.0; prev = 0.0
    for i in range(len(roots)+1):
        cur = Df(roots[i]) if i < len(roots) else 0.0
        tot += abs(cur-prev); prev = cur
    return tot/2

def s4(mp, store):
    import numpy as np
    out("S4 far-field stabilization tables (C022 gamma_LOC corollary recomputation)")
    out("  conditional law given pins at testbed values (b,0,0,b-ell,0,0[,v*,0,0]),")
    out("  v* = b - ell/2; evaluation: one-point law of each jet component at")
    out("  toroidal distance >= d0 from every pin; grid: polar per pin,")
    out("  720 angles x dr=0.03, r in [d0, 7.0]/[d0, 6.0]; argmax refined in mp.")
    def Kv(x1, x2, deriv=(0, 0), imax=1):
        a, b = deriv
        tot = np.zeros(np.broadcast(np.asarray(x1, dtype=float),
                                    np.asarray(x2, dtype=float)).shape)
        for i in range(-imax, imax+1):
            for j in range(-imax, imax+1):
                t1 = x1 + 24.0*i; t2 = x2 + 24.0*j
                e = np.exp(-(t1*t1+t2*t2)/2)
                if (a, b) == (0, 0): tot += e
                elif (a, b) == (1, 0): tot += -t1*e
                elif (a, b) == (0, 1): tot += -t2*e
                elif (a, b) == (2, 0): tot += (t1*t1-1)*e
                elif (a, b) == (1, 1): tot += t1*t2*e
                elif (a, b) == (0, 2): tot += (t2*t2-1)*e
                elif (a, b) == (3, 0): tot += -(t1**3-3*t1)*e
                elif (a, b) == (2, 1): tot += -(t1*t1*t2-t2)*e
                elif (a, b) == (1, 2): tot += -(t1*t2*t2-t1)*e
                elif (a, b) == (0, 3): tot += -(t2**3-3*t2)*e
        return tot
    def uvec_arr(xs, ys, pins, gamma):
        n = 3*len(pins)
        U = np.empty((n,)+np.shape(xs))
        for i, p in enumerate(pins):
            for bb in range(3):
                B = FD[bb]; tot = (gamma[0]+B[0], gamma[1]+B[1])
                sgn = -1.0 if (B[0]+B[1]) % 2 else 1.0
                U[3*i+bb] = sgn*Kv(xs-p[0], ys-p[1], tot)
        return U
    thresholds = {}
    for r in (0.05, 0.025):
        for ntag in ("6pin", "9pin"):
            D = store[(r, ntag)]
            pins, vals = D["pins"], D["vals"]
            Gf = np.array([[float(D["G"][i, j]) for j in range(3*len(pins))]
                           for i in range(3*len(pins))])
            Ginvf = np.linalg.inv(Gf)
            wf = Ginvf @ np.array(vals, dtype=float)
            for d0 in (3.0, 5.0):
                nang = 720
                rmax = 7.0 if d0 == 3.0 else 6.0
                dr = 0.03
                angs = np.linspace(0, 2*np.pi, nang, endpoint=False)
                best = {}
                for p in pins:
                    rr = d0
                    while rr < rmax:
                        xs = p[0] + rr*np.cos(angs); ys = p[1] + rr*np.sin(angs)
                        dmin = np.full(nang, np.inf)
                        for q in pins:
                            dxq = xs-q[0]; dyq = ys-q[1]
                            dxq -= 24*np.round(dxq/24); dyq -= 24*np.round(dyq/24)
                            dmin = np.minimum(dmin, np.hypot(dxq, dyq))
                        mask = dmin >= d0 - 1e-12
                        if mask.any():
                            for name, gamma, s02 in EFUNCS:
                                U = uvec_arr(xs, ys, pins, gamma)
                                Zv = np.einsum('ij,ja->ia', Ginvf, U)
                                red = np.einsum('ia,ia->a', U, Zv)/s02
                                m = wf @ U
                                vv = s02*(1.0-red)
                                tv = np.array([tv_gauss(float(a), float(b_), s02)
                                               if b_ > 0 else 1.0
                                               for a, b_ in zip(m, vv)])
                                for qn, qa in (("red", red), ("tv", tv)):
                                    qm = np.where(mask, qa, 0.0)
                                    i0 = int(np.argmax(qm))
                                    key = (name, qn)
                                    if key not in best or qm[i0] > best[key][0]:
                                        best[key] = (float(qm[i0]),
                                                     (float(xs[i0]), float(ys[i0])),
                                                     float(m[i0]), float(vv[i0]))
                        rr += dr
                thresholds[(r, ntag, d0)] = best
                fr = best[("f", "red")]; ft = best[("f", "tv")]
                dxr = best[("dx", "red")]
                allred = max(best[(nm, "red")][0] for nm, _, _ in EFUNCS)
                alltv = max(best[(nm, "tv")][0] for nm, _, _ in EFUNCS)
                out("  r=%.3f %s d>=%g: red_f=%.4e red_dx=%.4e red_max=%.4e "
                    "tv_f=%.4e tv_max=%.4e"
                    % (r, ntag, d0, fr[0], dxr[0], allred, ft[0], alltv))
    # mp refinement of the argmax points for the f-functional (both quantities)
    out("  mp refinement (dps 60) of f-argmax points:")
    mp.mp.dps = 60
    for r in (0.05, 0.025):
        for ntag in ("6pin", "9pin"):
            D = store[(r, ntag)]
            pins, vals = D["pins"], D["vals"]
            Ginv = D["Ginv"]
            w = D["Ginv"]*mp.matrix([mp.mpf(x) for x in vals])
            for d0 in (3.0, 5.0):
                best = thresholds[(r, ntag, d0)]
                for qn in ("red", "tv"):
                    tv_f, x, m_f, v_f = best[("f", qn)]
                    U = mp.matrix(3*len(pins), 1)
                    for i, p in enumerate(pins):
                        for bb in range(3):
                            B = FD[bb]
                            sgn = -1 if (B[0]+B[1]) % 2 else 1
                            U[3*i+bb] = sgn*Kimg_mp(mp, mp.mpf(x[0])-mp.mpf(p[0]),
                                                    mp.mpf(x[1])-mp.mpf(p[1]),
                                                    (B[0], B[1]))
                    m_mp = (w.T*U)[0]
                    red_mp = (U.T*Ginv*U)[0]
                    # consistency float vs mp at argmax
                    ck(abs(float(red_mp) - (1 - v_f)) < 5e-3*max(1 - v_f, 1e-12) + 1e-12,
                       "S4 mp-vs-float red r=%s %s d0=%s" % (r, ntag, d0))
                    out("    r=%.3f %s d>=%g (%s): x=(%.4f,%.4f) m=%s red=%s"
                        % (r, ntag, d0, qn, x[0], x[1],
                           mp.nstr(m_mp, 6), mp.nstr(red_mp, 6)))
    # ---- certified thresholds (fail-closed) ----
    for (r, ntag, d0), best in thresholds.items():
        allred = max(best[(nm, "red")][0] for nm, _, _ in EFUNCS)
        alltv = max(best[(nm, "tv")][0] for nm, _, _ in EFUNCS)
        redf = best[("f", "red")][0]
        tvf = best[("f", "tv")][0]
        if d0 == 3.0:
            ck(allred <= 0.115, "S4 red_max d>=3 <= 0.115 %s %s %s" % (r, ntag, d0))
            ck(redf <= 0.025, "S4 red_f d>=3 <= 0.025 %s %s" % (r, ntag))
        else:
            ck(allred <= 1e-4, "S4 red_max d>=5 <= 1e-4 %s %s" % (r, ntag))
            ck(alltv <= 2.5e-3, "S4 tv_max d>=5 <= 2.5e-3 %s %s" % (r, ntag))
            if ntag == "6pin":
                ck(tvf <= 1e-4, "S4 tv_f d>=5 <= 1e-4 (6pin) r=%s" % r)
            else:
                ck(tvf <= 3e-4, "S4 tv_f d>=5 <= 3e-4 (9pin) r=%s" % r)
    out("  certified: red_max(d>=3)<=0.115, red_f(d>=3)<=0.025, red_max(d>=5)<=1e-4,")
    out("  tv_max(d>=5)<=2.5e-3, tv_f(d>=5)<=1e-4 (6pin) / <=3e-4 (9pin)")
    # ---- envelopes (crude, rigorous, monotone) ----
    out("  envelopes: red(x) <= n||Ginv||_inf E(d)^2/s02, |m(x)| <= ||w||_1 E(d),")
    out("  E(d) = d e^{-d^2/2} (decreasing for d>1; verified):")
    for r in (0.05, 0.025):
        for ntag in ("6pin", "9pin"):
            D = store[(r, ntag)]
            n = 3*len(D["pins"])
            gi = float(D["ginf_inf"]); w1 = float(D["w1"])
            E = lambda d: d*math.exp(-d*d/2)
            ck(E(6.0) > E(6.1) > E(6.2), "S4 E monotone")
            dvar = None; dmean = None
            d = 3.0
            while d <= 17.0:
                if dvar is None and n*gi*E(d)**2 < 1e-4: dvar = round(d, 1)
                if dmean is None and w1*E(d) < 1e-4: dmean = round(d, 1)
                d += 0.1
            ck(dvar is not None and dmean is not None, "S4 envelope exists")
            lim = 6.7 if ntag == "6pin" else 8.0
            ck(dvar <= lim, "S4 var envelope threshold %.1f <= %.1f %s %s"
               % (dvar, lim, r, ntag))
            ck(dmean <= 8.1, "S4 mean envelope threshold %.1f %s %s" % (dmean, r, ntag))
            out("    r=%.3f %s: var-env <1e-4 for d>=%.1f; mean-env <1e-4 for d>=%.1f"
                % (r, ntag, dvar, dmean))
    out("  NOTE (for the discharge text): full one-point TV including the")
    out("  conditional mean at testbed values exceeds 2.2% at d=3 (value TV up to")
    out("  %.3f (6pin) / %.3f (9pin)); the ledger's 2.2%% matches the value-"
        % (max(thresholds[(r, "6pin", 3.0)][("f", "tv")][0] for r in (0.05, 0.025)),
           max(thresholds[(r, "9pin", 3.0)][("f", "tv")][0] for r in (0.05, 0.025))))
    out("  variance channel: red_f(d>=3) in [%.4f, %.4f] vs ledger 0.022."
        % (min(thresholds[(r, nt, 3.0)][("f", "red")][0]
               for r in (0.05, 0.025) for nt in ("6pin", "9pin")),
           max(thresholds[(r, nt, 3.0)][("f", "red")][0]
               for r in (0.05, 0.025) for nt in ("6pin", "9pin"))))
    out("S4 OK")

# ============================ S5 ===========================================
DEPS = [
    ("/mnt/agents/upload/C031_LBRATE_Integration.md",
     "e7998ef0d17d951bc89978f9fe32e510019059dd0650c8f0e0ae33273f40f32e"),
    ("/mnt/agents/upload/C022 Observed Update.json",
     "9bc0647b885931ba4dc862bedfb879f9bd12c4cd6d39f51c6e98e54c7f75d5cb"),
    ("/mnt/agents/upload/Lemma UB0.md",
     "c6a90469b6e43c592b661a63c85a6fa6590505f3d22fd888a328ebf7926d41c0"),
    ("/mnt/agents/upload/Lemma UB0 Addendum.md",
     "224438cb740565ff8b2f0acab83a75c8dc80cc15265a55451f43ff930894c94c"),
    ("/mnt/agents/upload/R0UNIF and LBRestate.md",
     "0e9bb486c2dec10322f4145e3266eff5cdfef82edf5e4b05203d447b4a2c0ad4"),
    ("/mnt/agents/upload/MS Shift R0 Closure.md",
     "6575e2bb5178e1229794360d44bdcd5a733952033899b92d89868cf15c3d6c32"),
    ("/mnt/agents/upload/02_GAUSSIAN_TRANSVERSALITY — WORKING-SOURCE — 19242B — SHA4a77f3b7 — ID19G4QeIu.md",
     "4a77f3b7ed390c0ebff3c580baa4d8a107bf3cf2b892c5927e5ffe005ad1c1fe"),
    ("/mnt/agents/output/SIDE24_gap_fill/SIDE24_GAP_FILL_SUPPLEMENT.md",
     "33609b70d8c4d970d3594ebdc2cb59200508bfb2e53002ae7c6b10b3bc19d41f"),
    ("/mnt/agents/output/KIMI_EXPORT_2026-08-04_SARDG/KIMI-AUD-022.md",
     "8efcd937502b67ab515d41435eace639000c287ce5e6773105953319f48429b8"),
]

def s5():
    out("S5 dependency hash binding (sha256, byte-exact)")
    for path, want in DEPS:
        ck(os.path.exists(path), "S5 missing dependency: " + path)
        h = hashlib.sha256(open(path, "rb").read()).hexdigest()
        ck(h == want, "S5 hash mismatch: " + path)
        out("  ok %s  %s" % (h[:16] + "...", os.path.basename(path)))
    # KIMI-AUD-022 self-hash (body before the marker line)
    p = "/mnt/agents/output/KIMI_EXPORT_2026-08-04_SARDG/KIMI-AUD-022.md"
    data = open(p, "rb").read()
    marker = b"SHA-256 of this report body"
    idx = data.index(marker)
    h = hashlib.sha256(data[:idx]).hexdigest()
    ck(h == "b00f07210a54680b3c0695be03f29037f1e228c6c247cc339cbf39f651dc0f88",
       "S5 KIMI-AUD-022 self-hash")
    out("  ok KIMI-AUD-022 self-hash b00f0721... (verdict binds byte-exact body)")
    out("S5 OK")

# ============================ main =========================================
def main():
    import mpmath as mp
    out("LB-3 fail-closed certificate: R0 / gamma-LOC (SIDE24/q0)")
    s1()
    mp.mp.dps = 80
    pi, step, lat, wsp, Z = s2(mp)
    Kspec = make_kspec(mp, lat, wsp, Z)
    s2b(mp, Kspec)
    store = s3(mp, Kspec)
    s4(mp, store)
    s5()
    out("LB3_CERTIFICATE_PASS")
    sys.stdout.write("\n".join(LINES) + "\n")

if __name__ == "__main__":
    main()
