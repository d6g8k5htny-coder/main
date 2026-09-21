# Strip scan: identify boxes whose mt(y) is within nsig of the window.
# Writes a box list for w3_lbox.py. Uses cheap point evaluations.
import math, sys, time
from mpmath import iv, mpi
import mpmath as mp
iv.prec = 350
import importlib.util
spec = importlib.util.spec_from_file_location("w3c", "/mnt/agents/output/K3_SIDE24_LB/W3_numerics/w3_cert.py")
w3c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w3c)
w3c.init_caches()
G = w3c.gram_interval()
C, Es = w3c.build_whitener(G)
law = w3c.build_law(G, C)
mu_t = w3c.compute_mu_t(G)
tv = w3c.matvec_iv(law['C'], w3c.pin_values(mu_t))

x0, x1, y0, y1, sz = map(float, sys.argv[1:6])
out = open(sys.argv[6], "w")
b = float(w3c.mid(w3c.civ(w3c.b_rat)))
ell = float(w3c.mid(w3c.civ(w3c.ell_rat)))
u_c = b - ell / 2
nx = int(round((x1 - x0) / sz))
ny = int(round((y1 - y0) / sz))
nsel = 0
t0 = time.time()
for i in range(nx):
    for j in range(ny):
        xc = x0 + (i + 0.5) * sz
        yc = y0 + (j + 0.5) * sz
        if min((xc-px)**2 + (yc-py)**2 for (px, py) in [(-0.0125,0),(0.0125,0),(-0.0315,0.006)]) < 1e-8:
            continue
        yb = (mpi(xc, xc), mpi(yc, yc))
        try:
            muJ, SigJ = w3c.jetlaw_iv(yb, law, tv)
            # mt = mu_f - S_fg S_g^-1 mu_g
            Sg = [[SigJ[1][1], SigJ[1][2]], [SigJ[1][2], SigJ[2][2]]]
            Sgi, detg = w3c.inv2x2_spd(Sg, "strip")
            t = [Sgi[0][0]*muJ[1] + Sgi[0][1]*muJ[2], Sgi[1][0]*muJ[1] + Sgi[1][1]*muJ[2]]
            mt = muJ[0] - (SigJ[0][1]*t[0] + SigJ[0][2]*t[1])
            vt = SigJ[0][0] - (SigJ[0][1]*t[0] + SigJ[0][2]*t[1])
            mtf = float(w3c.mid(mt))
            vtf = float(w3c.mid(vt))
            sig = math.sqrt(max(vtf, 1e-30))
            # contributing if |mt - window center| < 5 sig + ell + sz*grad-margin
            if abs(mtf - u_c) < 3.5 * sig + ell + 0.005:
                out.write("%d %d %.6f %.6f %.6f %.3e\n" % (i, j, xc, yc, mtf, sig))
                nsel += 1
        except SystemExit:
            pass
    if (i * ny) % 200 == 0:
        print("row %d/%d sel=%d t=%.0f" % (i, nx, nsel, time.time() - t0), flush=True)
out.close()
print("SELECTED=%d of %d" % (nsel, nx * ny))
