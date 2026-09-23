# Final spatial drivers for I_WP rigorous bounds. Fail-closed.
# Usage:
#   python3 w3_run.py lower x0 x1 y0 y1 sz gtol outlog   (exact, area*lo(rho))
#   python3 w3_run.py upper x0 x1 y0 y1 sz0 tolbox gtol outlog  (min(cheap,mid))
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

ivm = iv.mpf
lo, hi, mid, width = w3c.lo, w3c.hi, w3c.mid, w3c.width
ck = w3c.ck
PINS = [(-0.0125, 0.0), (0.0125, 0.0), (-0.0315, 0.006)]
SMIN = 1e-6

def pin_dist(x, y):
    return min(math.hypot(x - px, y - py) for (px, py) in PINS)

MODE = sys.argv[1]
LG = open(sys.argv[-1], "w", buffering=1)

t00 = time.time()
w3c.init_caches()
G = w3c.gram_interval()
C, Es = w3c.build_whitener(G)
law = w3c.build_law(G, C)
mu_t = w3c.compute_mu_t(G)
tv = w3c.matvec_iv(law['C'], w3c.pin_values(mu_t))
w3t.w3c._CACHE.update(w3c._CACHE)
LG.write("# setup %.1fs mode=%s args=%s\n" % (time.time() - t00, MODE, sys.argv[1:-1]))

def taylor(xc, yc, N):
    return w3t.TaylorLaw((ivm(repr(xc)), ivm(repr(yc))), N, law, tv)

acc_lo = ivm(0)
acc_hi = ivm(0)
nb = 0
skipped_pin = 0
t0 = time.time()

if MODE == 'lower':
    x0, x1, y0, y1, sz, gtol = map(float, sys.argv[2:8])
    nx = int(round((x1 - x0) / sz))
    ny = int(round((y1 - y0) / sz))
    ck(abs(nx * sz - (x1 - x0)) < 1e-12 and abs(ny * sz - (y1 - y0)) < 1e-12, "grid")
    for i in range(nx):
        for j in range(ny):
            xa, xb = x0 + i * sz, x0 + (i + 1) * sz
            ya, yb_ = y0 + j * sz, y0 + (j + 1) * sz
            xc, yc = (xa + xb) / 2, (ya + yb_) / 2
            if pin_dist(xc, yc) < 3 * SMIN:
                skipped_pin += 1
                continue
            ybx = (mpi(repr(xa), repr(xb)), mpi(repr(ya), repr(yb_)))
            N = 6 if sz > 0.03 else 5
            JS = taylor(xc, yc, N).eval_box(ybx)
            r = w3c.rho_box(ybx, law, tv, gtol, "low", need_exact=True, JS=JS)
            ck(r['mode'] == 'exact', "lower exact")
            ck(lo(r['g']) > 0, "lower g>0")
            area = ivm(repr(sz)) ** 2
            acc_lo += lo(r['rho']) * area
            acc_hi += hi(r['rho']) * area
            nb += 1
            if nb % 25 == 0:
                LG.write("t=%.0f nb=%d acc_lo=%.6e acc_hi=%.6e\n" % (
                    time.time() - t0, nb, float(lo(acc_lo)), float(hi(acc_hi))))
    LG.write("DONE nb=%d skipped_pin=%d\n" % (nb, skipped_pin))
    LG.write("LOWER=%.10e\n" % float(lo(acc_lo)))
    LG.write("UPPERBAND=%.10e\n" % float(hi(acc_hi)))
    LG.write("TIME=%.0f\n" % (time.time() - t0))

elif MODE == 'upper':
    x0, x1, y0, y1, sz0, tolbox, gtol = map(float, sys.argv[2:9])
    # adaptive: accept if min(cheap,mid)*area <= tolbox; else split to sz_min=0.005;
    # at min size accept with min(cheap,mid) regardless (recorded)
    sz_min = 0.005
    stack = []
    nx = max(1, int(math.ceil((x1 - x0) / sz0)))
    ny = max(1, int(math.ceil((y1 - y0) / sz0)))
    for i in range(nx):
        for j in range(ny):
            stack.append((x0 + (x1 - x0) * i / nx, x0 + (x1 - x0) * (i + 1) / nx,
                          y0 + (y1 - y0) * j / ny, y0 + (y1 - y0) * (j + 1) / ny))
    nforced = 0
    while stack:
        xa, xb, ya, yb_ = stack.pop()
        xc, yc = (xa + xb) / 2, (ya + yb_) / 2
        sz = max(xb - xa, yb_ - ya)
        dp = pin_dist(xc, yc)
        rad = math.hypot(xb - xc, yb_ - yc)
        if dp + rad <= SMIN:
            skipped_pin += 1
            continue
        if dp - rad < SMIN and sz > 0.25 * SMIN:
            # straddles pin chart: split until tiny, then pin-chart handles
            if sz > 2 * SMIN:
                xm, ym = (xa + xb) / 2, (ya + yb_) / 2
                stack += [(xa, xm, ya, ym), (xm, xb, ya, ym), (xa, xm, ym, yb_), (xm, xb, ym, yb_)]
                continue
        ybx = (mpi(repr(xa), repr(xb)), mpi(repr(ya), repr(yb_)))
        N = 6 if sz > 0.03 else 5
        area = ivm(repr(xb - xa)) * ivm(repr(yb_ - ya))
        try:
            JS = taylor(xc, yc, N).eval_box(ybx)
            r = w3c.rho_box(ybx, law, tv, gtol, "up", need_exact=False, JS=JS)
        except SystemExit:
            r = None
        if r is not None:
            bd = min(hi(r['cheap']), hi(r.get('mid', r['cheap'])))
            if float(bd * hi(area)) <= tolbox:
                acc_hi += bd * area
                nb += 1
                if nb % 500 == 0:
                    LG.write("t=%.0f nb=%d acc_hi=%.6e\n" % (time.time() - t0, nb, float(hi(acc_hi))))
                continue
        if sz > sz_min:
            xm, ym = (xa + xb) / 2, (ya + yb_) / 2
            stack += [(xa, xm, ya, ym), (xm, xb, ya, ym), (xa, xm, ym, yb_), (xm, xb, ym, yb_)]
        else:
            # forced leaf at min size: use exact
            r = w3c.rho_box(ybx, law, tv, gtol, "upleaf", need_exact=True, JS=JS)
            bd = hi(r['rho'])
            if r['mode'] != 'exact':
                bd = min(hi(r['cheap']), hi(r.get('mid', r['cheap'])))
            acc_hi += bd * area
            nforced += 1
            nb += 1
    LG.write("DONE nb=%d nforced=%d skipped_pin=%d\n" % (nb, nforced, skipped_pin))
    LG.write("UPPER=%.10e\n" % float(hi(acc_hi)))
    LG.write("TIME=%.0f\n" % (time.time() - t0))
LG.write("EXITOK\n")
