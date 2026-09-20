# Per-box certified lower bound driver:
#   int_{band} rho dy >= sum_boxes boxint_lo
# Usage: python3 w3_lbox.py x0 x1 y0 y1 sz gtol outlog
import math, sys, time
from mpmath import iv, mpi
import mpmath as mp
iv.prec = 350
import importlib.util
_spec = importlib.util.spec_from_file_location("w3c", "/mnt/agents/output/K3_SIDE24_LB/W3_numerics/w3_cert.py")
w3c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w3c)
_spec2 = importlib.util.spec_from_file_location("w3t", "/mnt/agents/output/K3_SIDE24_LB/W3_numerics/w3_taylor.py")
w3t = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(w3t)
_spec3 = importlib.util.spec_from_file_location("w3tm", "/mnt/agents/output/K3_SIDE24_LB/W3_numerics/w3_tm.py")
w3tm = importlib.util.module_from_spec(_spec3)
_spec3.loader.exec_module(w3tm)
_spec4 = importlib.util.spec_from_file_location("w3L", "/mnt/agents/output/K3_SIDE24_LB/W3_numerics/w3_lower.py")
w3L = importlib.util.module_from_spec(_spec4)
_spec4.loader.exec_module(w3L)
# unify module instances (isinstance identity across files)
w3c.__dict__['w3tm'] = w3tm
w3L.__dict__['w3c'] = w3c

ivm = iv.mpf
lo, hi, mid, width = w3c.lo, w3c.hi, w3c.mid, w3c.width
ck = w3c.ck

LISTMODE = (sys.argv[1] == 'LIST')
if LISTMODE:
    sz = float(sys.argv[3])
    gtol = float(sys.argv[4])
    LG = open(sys.argv[5], "w", buffering=1)
    boxes = []
    for ln in open(sys.argv[2]):
        if ln.strip():
            p = ln.split()
            boxes.append((float(p[2]), float(p[3])))
    x0 = x1 = y0 = y1 = 0.0
else:
    x0, x1, y0, y1, sz, gtol = map(float, sys.argv[1:7])
    LG = open(sys.argv[7], "w", buffering=1)
    boxes = None

t00 = time.time()
w3c.init_caches()
G = w3c.gram_interval()
C, Es = w3c.build_whitener(G)
law = w3c.build_law(G, C)
mu_t = w3c.compute_mu_t(G)
tv = w3c.matvec_iv(law['C'], w3c.pin_values(mu_t))
w3t.w3c._CACHE.update(w3c._CACHE)
LG.write("# setup %.1fs\n" % (time.time() - t00))
iv.prec = 160

b_iv = w3c.civ(w3c.b_rat)
ell_iv = w3c.civ(w3c.ell_rat)
u0w = float(lo(b_iv - ell_iv))
u1w = float(hi(b_iv))
u_c = (u0w + u1w) / 2.0

acc = ivm(0)
acc_up = ivm(0)
nb = 0
t0 = time.time()


def pder(p, which):
    co = [ivm(0)] * len(p.Gs)
    idx = {g: k for k, g in enumerate(p.Gs)}
    for k, g in enumerate(p.Gs):
        if g[which] > 0:
            gg = list(g)
            gg[which] -= 1
            co[idx[tuple(gg)]] = p.co[k] * g[which]
    return w3tm.P(co, p.Gs)


if LISTMODE:
    nx = len(boxes)
    ny = 1
else:
    nx = int(round((x1 - x0) / sz))
    ny = int(round((y1 - y0) / sz))
    ck(abs(nx * sz - (x1 - x0)) < 1e-9 and abs(ny * sz - (y1 - y0)) < 1e-9, "grid")
for i in range(nx):
    for j in range(ny):
        if LISTMODE:
            xc, yc = boxes[i]
            xa, xb = xc - sz / 2, xc + sz / 2
            ya, yb_ = yc - sz / 2, yc + sz / 2
        else:
            xa, xb = x0 + i * sz, x0 + (i + 1) * sz
            ya, yb_ = y0 + j * sz, y0 + (j + 1) * sz
        xc, yc = (xa + xb) / 2, (ya + yb_) / 2
        r1, r2 = (xb - xa) / 2, (yb_ - ya) / 2
        # ensure exact representability of box endpoints
        xa, xb, ya, yb_ = xc - r1, xc + r1, yc - r2, yc + r2
        area = ivm(repr(xb - xa)) * ivm(repr(yb_ - ya))
        ybx = (mpi(repr(xa), repr(xb)), mpi(repr(ya), repr(yb_)))
        tt0 = time.time()
        TL = w3t.TaylorLaw((ivm(repr(xc)), ivm(repr(yc))), 5, law, tv)
        tt1 = time.time()
        RT = w3tm.RhoTM(TL, ybx)
        F = RT.build()
        F['_RT'] = RT
        tt2 = time.time()
        vt, pgrad = F['vt'], F['pgrad']
        # ---- mt model ----
        mu0p = RT.mu[0]
        det2p = F['det2p']
        Sg11, Sg12, Sg22 = RT.sig[1][1], RT.sig[1][2], RT.sig[2][2]
        a1p = Sg22 * RT.mu[1] - Sg12 * RT.mu[2]
        a2p = -Sg12 * RT.mu[1] + Sg11 * RT.mu[2]
        sfp = RT.sig[0][1] * a1p + RT.sig[0][2] * a2p
        mt_num = mu0p * det2p - sfp
        det2v = F['det2']
        det2_lo = lo(det2v)
        ck(det2_lo > 0, "det2 lbox")
        # center values (constant coefficients)
        mt_c = mt_num.co[0] / det2p.co[0]
        # gradient at center
        gmt = []
        for w in (0, 1):
            dnum = pder(mt_num, w).co[0]
            dden = pder(det2p, w).co[0]
            gmt.append((dnum * det2p.co[0] - mt_num.co[0] * dden) / (det2p.co[0] ** 2))
        # Hessian sup over box -> Rm
        Rm = ivm(0)
        det2lo3 = ivm(lo(det2v)) ** 3
        for w in (0, 1):
            for v_ in (0, 1):
                Hn = pder(pder(mt_num, w), v_)
                Hd = pder(pder(det2p, w), v_)
                t1 = RT.evp(Hn) / det2v
                dnw = RT.evp(pder(mt_num, w))
                dnv = RT.evp(pder(mt_num, v_))
                ddw = RT.evp(pder(det2p, w))
                ddv = RT.evp(pder(det2p, v_))
                t2 = (dnw * ddv + dnv * ddw) / (det2v * det2v)
                t3 = RT.evp(mt_num) * RT.evp(Hd) / (det2v * det2v)
                t4 = 2 * RT.evp(mt_num) * ddw * ddv / (det2v * det2v * det2v)
                Hij = t1 - t2 - t3 + t4
                Rm += iv.fabs(Hij) * (ivm(r1) if w == 0 else ivm(r2)) * (ivm(r1) if v_ == 0 else ivm(r2)) / (2 if w == v_ else 1)
        Rm_f = float(hi(Rm)) * (1 + 1e-12)
        # ---- Hessian law center + deltas ----
        # pre-check: skip if mt is > 5 sigma from the window (contribution negligible -> 0, valid)
        mtc_f = float(mid(mt_c))
        sig_hi_f0 = float(hi(iv.sqrt(vt)))
        if abs(mtc_f - u_c) > 5 * sig_hi_f0 + 0.5 * (u1w - u0w):
            LG.write(("box (%.4f,%.4f): SKIP mt=%.4f sig=%.3g" + chr(10)) % (xc, yc, mtc_f, sig_hi_f0))
            acc += ivm(0)
            # crude rigorous upper for skipped boxes: rho <= pgrad_hi * sqrt(EW2sup) * sqrt(Pwin_hi)
            # Pwin <= ell*phi(z*)/sig_lo with z* the window-nearest standardized distance
            zstar = max(0.0, (abs(mtc_f - u_c) - 0.5 * (u1w - u0w)) / sig_hi_f0)
            pwin_hi = (u1w - u0w) * math.exp(-0.5 * zstar * zstar) / math.sqrt(2 * math.pi) / float(lo(iv.sqrt(vt)))
            # EW2 = kap2 + kap1^2 <= 6 smax^2 + (3 smax + m0^2)^2 (||Q||=1)
            smax = max(float(hi(w3tm.poly_eval_bound(F['SHH'][i][i], r1, r2))) for i in range(3))
            det3_lo_s = float(lo(F['det3']))
            m0u2 = 0.0
            for i2 in range(3):
                muH_s = float(hi(w3tm.poly_eval_bound(RT.mu[3 + i2], r1, r2)))
                mH_s = muH_s
                for k in range(3):
                    w3_s = float(hi(w3tm.poly_eval_bound(F['W3num'][i2][k], r1, r2))) / det3_lo_s
                    mu_s = float(hi(w3tm.poly_eval_bound(RT.mu[k], r1, r2)))
                    if k == 0:
                        mu_s += max(abs(u0w), abs(u1w))
                    mH_s += w3_s * mu_s
                m0u2 += mH_s * mH_s
            ew2_hi = (6.0 * smax * smax + (3.0 * smax + m0u2) ** 2) * (1 + 1e-12)
            upb = float(hi(F['pgrad'])) * math.sqrt(max(0.0, ew2_hi)) * math.sqrt(min(1.0, pwin_hi)) * float(mid(area))
            acc_up += ivm(upb)
            nb += 1
            continue
        # interval g-quad over the box (clamped hval: rigorous for all V in box)
        r_tm = w3c.rho_tm(TL, ybx, law, tv, gtol, "lbox", need_exact=True, _F=F)
        tt3 = time.time()
        LG.write(("   phases: taylor=%.1f tmbuild=%.1f rhotm=%.1f" + chr(10)) % (tt1 - tt0, tt2 - tt1, tt3 - tt2))
        ck(r_tm['mode'] == 'exact', "lbox exact")
        g_c = r_tm['g']
        pgrad = r_tm['pgrad']
        vt = r_tm['vt']
        g_lo = lo(g_c)
        g_ok = g_lo > 0
        dg = width(g_c)
        # ---- convolution ----
        A = float(hi(iv.fabs(gmt[0]))) * r1
        B = float(hi(iv.fabs(gmt[1]))) * r2
        LG.write(("   conv: A=%.3g B=%.3g Rm=%.3g sig_lo=%.3g mt_c=%.6f" + chr(10)) % (
            A, B, Rm_f, float(lo(iv.sqrt(vt))), mtc_f))
        if A < B:
            A, B = B, A
        sig_lo_f = float(lo(iv.sqrt(vt)))
        sig_hi = iv.sqrt(vt)
        cr_lo = float(lo(b_iv - ell_iv - mt_c))
        cr_hi = float(hi(b_iv - mt_c))
        ca = sorted([abs(cr_lo), abs(cr_hi)])
        nsub = 64
        Fmin = None
        for kk in range(nsub + 1):
            cc = ca[0] + (ca[1] - ca[0]) * kk / nsub
            Fv = w3L.F_conv(ivm(cc), Rm_f, ivm(sig_lo_f), A, B)
            if Fmin is None or float(lo(Fv)) < float(lo(Fmin)):
                Fmin = Fv
        F_lo = max(0.0, float(lo(Fmin))) - (0.2420 / (sig_lo_f ** 2)) * (ca[1] - ca[0]) / (2 * nsub)
        # ---- assemble (lower bound: failures contribute 0, always valid) ----
        if g_ok and F_lo > 0:
            boxint = ivm(lo(pgrad)) * ivm(g_lo) * ell_iv / sig_hi * area * ivm(F_lo)
        else:
            boxint = ivm(0)
            LG.write(("box (%.4f,%.4f): ZERO (g_ok=%s F_lo=%.3g g_c=[%.3g,%.3g] dg=%.2g)" + chr(10)) % (
                xc, yc, g_ok, float(F_lo), float(lo(g_c)), float(hi(g_c)), dg))
        acc += boxint
        upb = min(float(hi(r_tm['cheap'])), float(hi(r_tm['mid']))) * area
        acc_up += ivm(upb)
        nb += 1
        LG.write(("box %d (%.4f,%.4f): boxint_lo=%.4e acc=%.6e acc_up=%.6e g_c=[%.3e,%.3e] dg=%.2e F_lo=%.3e t=%.0f" + chr(10)) % (
            nb, xc, yc, float(lo(boxint)), float(lo(acc)), float(lo(acc_up)), float(lo(g_c)), float(hi(g_c)), dg,
            float(F_lo), time.time() - t0))
LG.write("DONE nb=%d\n" % nb)
LG.write("LOWER=%.10e\n" % float(lo(acc)))
LG.write("UPPERBAND=%.10e\n" % float(hi(acc_up)))
LG.write("TIME=%.0f\n" % (time.time() - t0))
LG.write("EXITOK\n")
