# Spatial driver: rigorous upper/lower bounds for I_WP over regions, using
# TaylorLaw conditional-law models + rho_box (cheap/exact). Fail-closed.
import math, sys, time, hashlib, json
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
SMIN = 1e-6          # pin disks handled by charts
N_TAYLOR = 5
N_COARSE = 6

def pin_dist(x, y):
    return min(math.hypot(x - px, y - py) for (px, py) in PINS)

class Driver:
    def __init__(self, mode, tol_box, min_sz, logpath, subregion=None, gtol=1e-7):
        # mode: 'upper' or 'lower'
        self.mode = mode
        self.tol_box = tol_box
        self.min_sz = min_sz
        self.gtol = gtol
        self.sub = subregion
        self.log = open(logpath, "w")
        w3c.init_caches()
        G = w3c.gram_interval()
        C, Es = w3c.build_whitener(G)
        self.law = w3c.build_law(G, C)
        mu_t = w3c.compute_mu_t(G)
        self.tv = w3c.matvec_iv(self.law['C'], w3c.pin_values(mu_t))
        w3t.w3c._CACHE.update(w3c._CACHE)
        self.mu_t = mu_t
        self.acc_hi = ivm(0)     # upper accumulator (sum of hi contributions)
        self.acc_lo = ivm(0)     # lower accumulator (sum of lo contributions, exact only)
        self.n_cheap = 0
        self.n_exact = 0
        self.n_boxes = 0
        self.t_start = time.time()
        self.log.write("# driver %s tol_box=%.2e min_sz=%.4f\n" % (mode, tol_box, min_sz))
        self.log.flush()

    def taylor(self, xc, yc, N):
        return w3t.TaylorLaw((ivm(repr(xc)), ivm(repr(yc))), N, self.law, self.tv)

    def box_rho(self, x0, x1, y0, y1):
        """returns (contrib_hi, contrib_lo) area-weighted intervals of rho over the box"""
        xc, yc = (x0 + x1) / 2, (y0 + y1) / 2
        sz = max(x1 - x0, y1 - y0)
        N = N_COARSE if sz > 0.03 else N_TAYLOR
        area = ivm(repr(x1 - x0)) * ivm(repr(y1 - y0))
        TL = self.taylor(xc, yc, N)
        yb = (mpi(repr(x0), repr(x1)), mpi(repr(y0), repr(y1)))
        JS = TL.eval_box(yb)
        r = w3c.rho_box(yb, self.law, self.tv, self.gtol, "box", need_exact=(self.mode == 'lower'), JS=JS)
        return r, area

    def accept_cheap(self, r, area, sz):
        return float(hi(r['cheap'] * area)) <= self.tol_box

    def process(self, x0, x1, y0, y1):
        sz = max(x1 - x0, y1 - y0)
        xc, yc = (x0 + x1) / 2, (y0 + y1) / 2
        dp = pin_dist(xc, yc)
        rad = math.hypot(x1 - xc, y1 - yc)
        # pin disk: fully inside a chart disk
        if dp + rad <= SMIN:
            return 'pin', None
        # skip pin-adjacent boxes here (handled by annulus loop) if too close
        if dp - rad < 2 * SMIN and sz > 0.25 * SMIN:
            pass
        r, area = self.box_rho(x0, x1, y0, y1)
        self.n_boxes += 1
        if self.mode == 'upper':
            if self.accept_cheap(r, area, sz):
                self.acc_hi += hi(r['cheap'] * area)
                self.n_cheap += 1
                return 'cheap', float(hi(r['cheap'] * area))
            if sz > self.min_sz:
                return 'split', None
            # leaf: use min(cheap, exact)
            ybx = (mpi(repr(x0), repr(x1)), mpi(repr(y0), repr(y1)))
            ex = r if r['mode'] == 'exact' else w3c.rho_box(
                ybx, self.law, self.tv, self.gtol, "leaf", need_exact=True,
                JS=self.taylor(xc, yc, N_TAYLOR).eval_box(ybx))
            contrib = min(float(hi(r['cheap'])), float(hi(ex['rho']))) * float(hi(area))
            self.acc_hi += ivm(repr(contrib))
            self.n_exact += 1
            return 'exact', contrib
        else:
            # lower mode: only over subregion; must be exact with lo>0
            ck(r['mode'] == 'exact', "lower needs exact")
            glo = lo(r['g'])
            ck(glo > 0, "lower: g_lo>0")
            self.acc_lo += lo(r['rho']) * area
            self.acc_hi += hi(r['rho']) * area
            self.n_exact += 1
            return 'exact', (float(lo(r['rho'] * area)), float(hi(r['rho'] * area)))

    def run(self, region, sz0):
        (x0, x1), (y0, y1) = region
        stack = [(x0, x1, y0, y1)]
        # initial grid of sz0
        init = []
        nx = max(1, int(math.ceil((x1 - x0) / sz0)))
        ny = max(1, int(math.ceil((y1 - y0) / sz0)))
        for i in range(nx):
            for j in range(ny):
                xa = x0 + (x1 - x0) * i / nx
                xb = x0 + (x1 - x0) * (i + 1) / nx
                ya = y0 + (y1 - y0) * j / ny
                yb_ = y0 + (y1 - y0) * (j + 1) / ny
                init.append((xa, xb, ya, yb_))
        stack = init
        while stack:
            (xa, xb, ya, yb_) = stack.pop()
            act, val = self.process(xa, xb, ya, yb_)
            if act == 'split':
                xc, yc = (xa + xb) / 2, (ya + yb_) / 2
                stack += [(xa, xc, ya, yc), (xc, xb, ya, yc), (xa, xc, yc, yb_), (xc, xb, yc, yb_)]
            if self.n_boxes % 200 == 0:
                self.log.write("t=%.0fs boxes=%d cheap=%d exact=%d acc_hi=%.6e acc_lo=%.6e\n" % (
                    time.time() - self.t_start, self.n_boxes, self.n_cheap, self.n_exact,
                    float(hi(self.acc_hi)), float(lo(self.acc_lo))))
                self.log.flush()
            if time.time() - self.t_start > 20000:
                self.log.write("TIMEOUT mid-run\n")
                self.log.flush()
                ck(False, "driver timeout (fail-closed)")
        self.log.write("DONE boxes=%d cheap=%d exact=%d\n" % (self.n_boxes, self.n_cheap, self.n_exact))
        self.log.write("UPPER=%.10e\n" % float(hi(self.acc_hi)))
        self.log.write("LOWER=%.10e\n" % float(lo(self.acc_lo)))
        self.log.flush()

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == 'land':
        # cheap landscape scan on a grid: prints rho cheap_hi per cell
        d = Driver('upper', 1e-30, 0.01, "/dev/null")
        x0, x1, y0, y1 = -0.4, 0.4, -0.9, 0.9
        n = int(sys.argv[2])
        for i in range(n):
            for j in range(n):
                xa = x0 + (x1 - x0) * i / n; xb = x0 + (x1 - x0) * (i + 1) / n
                ya = y0 + (y1 - y0) * j / n; yb_ = y0 + (y1 - y0) * (j + 1) / n
                xc, yc = (xa + xb) / 2, (ya + yb_) / 2
                if pin_dist(xc, yc) < 0.03:
                    continue
                try:
                    r, area = d.box_rho(xa, xb, ya, yb_)
                except SystemExit as e:
                    print("%.4f %.4f CKFAIL %s" % (xc, yc, e), flush=True)
                    continue
                ch = float(hi(r['cheap']))
                if ch > 1e-12:
                    print("%.4f %.4f cheap=%.4e z1=%.2f pgrad=%.3e" % (
                        xc, yc, ch, float(mid(r['z1'])), float(mid(r['pgrad']))), flush=True)
    elif mode == 'upper':
        d = Driver('upper', float(sys.argv[2]), float(sys.argv[3]), sys.argv[4])
        d.run(((-1.5, 1.5), (-1.5, 1.5)), float(sys.argv[5]))
    elif mode == 'lower':
        # lower over subregion given as x0 x1 y0 y1 sz
        x0, x1, y0, y1, sz = map(float, sys.argv[2:7])
        d = Driver('lower', 0.0, sz, sys.argv[7])
        d.run(((x0, x1), (y0, y1)), sz)
