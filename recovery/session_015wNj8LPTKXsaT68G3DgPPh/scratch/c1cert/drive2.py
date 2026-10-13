import sys, time, math
sys.path.insert(0, '/home/user/wt_cusp/frontiers/cusp_coefficient_certified_20261001')
sys.dont_write_bytecode = True
from fractions import Fraction as Fr
from ia import dn, up, add, sub, mul, div, scal, from_frac
import gauss as G
import cusp as CU
import cx as CX
R, C = CU.RealOps, CU.CplxOps
def safe_bound(box):
    return CX.cabs_up(CU.g2(C, box))
def panel_cert(a, b, tol, rhos=(4.0, 3.0, 2.4, 2.0, 1.7, 1.5, 1.35, 1.25, 1.18, 1.12), m=24, nmax=60):
    h = 0.5*(b - a)
    cands = []
    for rho in rhos:
        slabs = G.ellipse_slabs(a, b, rho, m)
        Ms = []
        for bf in (safe_bound, CU.g2_bound_direct):
            try:
                Ms.append(max(bf(bx) for bx in slabs))
            except (ValueError, ZeroDivisionError, OverflowError):
                pass
        if not Ms:
            continue
        M = min(Ms)
        for n in range(4, nmax + 1):
            e = G.error_bound(n, rho, M, h)
            if e <= tol:
                cands.append((n, e, rho, M)); break
    if not cands:
        raise RuntimeError('panel [%g, %g]: no admissible rule' % (a, b))
    n, e, rho, M = min(cands)
    xs, ws = G.panel(n, a, b)
    Q = (0.0, 0.0)
    for x, w in zip(xs, ws):
        Q = add(Q, mul(w, CU.g2(R, x)))
    return (dn(Q[0] - e), up(Q[1] + e)), n, rho, M, e
edges = [0.0, 0.4, 0.7, 0.9, 1.0, 1.1, 1.3, 1.6, 2.0, 2.6, 3.4, 4.5, 6.0, 8.0, 11.0, 15.0]
t0 = time.time()
tot = (0.0, 0.0)
for a, b in zip(edges, edges[1:]):
    iv, n, rho, M, e = panel_cert(a, b, 5e-16)
    tot = add(tot, iv)
    print('[%5.2f, %5.2f]  n=%2d rho=%.2f M=%.3e err=%.2e  Q=[%.17g, %.17g]' % (a, b, n, rho, M, e, iv[0], iv[1]), flush=True)
U = 15.0
tail_main = div((1.0, 1.0), (7.0 * U**7, 7.0 * U**7))
c = from_frac(Fr(36, 19))
tail_err = up(1.5 * U**-14 / 14)   # (36/19)^(5/8) < 1.5
tail = (dn(tail_main[0] - tail_err), up(tail_main[1] + tail_err))
tot = add(tot, tail)
print('int g2 in [%.17g, %.17g]  width %.2e   (%.1fs)' % (tot[0], tot[1], tot[1]-tot[0], time.time()-t0))
# float reference: int g2 = J/(4 C0) with J = 8.037781790926847623751347054690236550348
J = 8.037781790926847623751347054690236550348
C0 = (3/16)**(-5/8)
print('float reference int g2 =', J/(4*C0))
