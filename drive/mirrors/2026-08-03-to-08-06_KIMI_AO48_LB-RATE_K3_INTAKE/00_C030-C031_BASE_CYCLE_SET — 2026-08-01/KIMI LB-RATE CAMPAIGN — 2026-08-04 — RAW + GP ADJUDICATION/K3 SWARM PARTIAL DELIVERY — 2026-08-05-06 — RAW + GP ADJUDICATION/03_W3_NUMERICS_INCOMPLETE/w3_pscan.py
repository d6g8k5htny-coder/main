import mpmath as mp
from mpmath import iv, mpi
iv.prec = 350
import importlib.util, sys
spec = importlib.util.spec_from_file_location("w3c", "/mnt/agents/output/K3_SIDE24_LB/W3_numerics/w3_cert.py")
w3c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w3c)
w3c.init_caches()
G = w3c.gram_interval()
C, Es = w3c.build_whitener(G)
law = w3c.build_law(G, C)
mu_t = w3c.compute_mu_t(G)
tv = w3c.matvec_iv(law['C'], w3c.pin_values(mu_t))
n = int(sys.argv[1])
x0, x1, y0, y1 = -0.8, 0.8, -1.2, 1.2
for i in range(n + 1):
    for j in range(n + 1):
        x = x0 + (x1 - x0) * i / n
        y = y0 + (y1 - y0) * j / n
        if min((x-px)**2 + (y-py)**2 for (px, py) in [(-0.0125,0),(0.0125,0),(-0.0315,0.006)]) < 0.0012:
            continue
        yb = (mpi(x, x), mpi(y, y))
        try:
            r = w3c.rho_box(yb, law, tv, 1e-7, "pscan", need_exact=False)
            ch = float(w3c.hi(r['cheap']))
            if ch > 1e-14:
                print("%.4f %.4f cheap=%.4e z1=%.2f pgrad=%.3e vt=%.2e" % (
                    x, y, ch, float(w3c.mid(r['z1'])), float(w3c.mid(r['pgrad'])), float(w3c.mid(r['vt']))), flush=True)
        except SystemExit as e:
            print("%.4f %.4f FAIL %s" % (x, y, e), flush=True)
