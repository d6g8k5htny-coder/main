import mpmath as mp
from mpmath import iv, mpi
iv.prec = 350
import importlib.util, time
spec = importlib.util.spec_from_file_location("w3c", "/mnt/agents/output/K3_SIDE24_LB/W3_numerics/w3_cert.py")
w3c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w3c)
spec2 = importlib.util.spec_from_file_location("w3t", "/mnt/agents/output/K3_SIDE24_LB/W3_numerics/w3_taylor.py")
w3t = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(w3t)
w3c._DEBUG_G = True
w3c.init_caches()
G = w3c.gram_interval()
C, Es = w3c.build_whitener(G)
law = w3c.build_law(G, C)
mu_t = w3c.compute_mu_t(G)
tv = w3c.matvec_iv(law['C'], w3c.pin_values(mu_t))
w3t.w3c._CACHE.update(w3c._CACHE)
sz = 0.0025
import sys
pts = [(0.0, 0.59875), (-0.03, 0.55)] if len(sys.argv) < 3 else [(float(sys.argv[1]), float(sys.argv[2]))]
for (xc, yc) in pts:
    ybx = (mpi(repr(xc - sz/2), repr(xc + sz/2)), mpi(repr(yc - sz/2), repr(yc + sz/2)))
    t0 = time.time()
    TL = w3t.TaylorLaw((iv.mpf(repr(xc)), iv.mpf(repr(yc))), 5, law, tv)
    t1 = time.time()
    r = w3c.rho_tm(TL, ybx, law, tv, 3e-7, "one", need_exact=True)
    t2 = time.time()
    print("(%.4f,%.4f) mode=%s rho=[%.4e,%.4e] vt=[%.2e,%.2e] taylor=%.1fs exact=%.1fs" % (
        xc, yc, r['mode'], w3c.lo(r['rho']), w3c.hi(r['rho']), w3c.lo(r['vt']), w3c.hi(r['vt']),
        t1 - t0, t2 - t1), flush=True)
    if r['mode'] == 'exact':
        print("   g=[%.3e,%.3e] npan=%d pgrad=[%.3e,%.3e] mt=[%.3e,%.3e] z1=[%.2e,%.2e] Pwin=[%.2e,%.2e]" % (
            w3c.lo(r['g']), w3c.hi(r['g']), r['npan'],
            w3c.lo(r['pgrad']), w3c.hi(r['pgrad']), w3c.lo(r['mt']), w3c.hi(r['mt']),
            w3c.lo(r['z1']), w3c.hi(r['z1']), w3c.lo(r['Pwin']), w3c.hi(r['Pwin'])), flush=True)
