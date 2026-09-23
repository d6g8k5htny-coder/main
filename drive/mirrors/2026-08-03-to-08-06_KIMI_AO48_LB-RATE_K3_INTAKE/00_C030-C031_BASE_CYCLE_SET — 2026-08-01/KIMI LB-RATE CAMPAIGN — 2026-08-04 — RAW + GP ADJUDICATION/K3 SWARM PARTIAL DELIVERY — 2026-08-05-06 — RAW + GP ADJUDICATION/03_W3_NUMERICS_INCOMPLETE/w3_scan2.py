# W3 scan v2: (a) zone landscape on 0.05 grid; (b) near-M radial probes, high precision.
import mpmath as mp
import importlib.util, sys

PREC = int(sys.argv[1]) if len(sys.argv) > 1 else 120
MODE = sys.argv[2] if len(sys.argv) > 2 else "landscape"
mp.mp.prec = PREC

spec = importlib.util.spec_from_file_location("w3p", "/mnt/agents/output/K3_SIDE24_LB/W3_numerics/w3_proto.py")
w3p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w3p)
spec2 = importlib.util.spec_from_file_location("w3s", "/mnt/agents/output/K3_SIDE24_LB/W3_numerics/w3_scan.py")
w3s = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(w3s)

b_mp, ell = w3p.b_mp, w3p.ell
Mpt = (mp.mpf(-1)/80, mp.mpf(0))
Spt = (mp.mpf(1)/80, mp.mpf(0))
Ypt = (w3p.mqf(w3p.Y[0]), w3p.mqf(w3p.Y[1]))

if MODE == "landscape":
    h = 0.05
    vals = {}
    for i in range(-30, 31):
        for j in range(-30, 31):
            gx, gy = i*h, j*h
            if gx*gx + gy*gy > 1.5*1.5: continue
            rv, info = w3s.rho((mp.mpf(gx), mp.mpf(gy)))
            vals[(gx, gy)] = (float(rv), info)
    top = sorted(vals.items(), key=lambda kv: -kv[1][0])[:15]
    print("TOP 15 (0.05 grid):")
    for k, v in top:
        print("   ", k, mp.nstr(v[0], 8))
    tot = sum(v[0] for v in vals.values()) * h*h
    print("crude I over |y|<=1.5 (0.05 grid, midpoint) ~", mp.nstr(tot, 8))
    # hot spot detail
    for key in [t[0] for t in top[:5]]:
        rv, info = vals[key]
        print("at", key, "rho=", mp.nstr(rv,8), {k:(round(v,8) if isinstance(v,float) else v) for k,v in info.items() if k in ('pgrad','z','g','muf2','Sff2')})
elif MODE == "pinrad":
    # radial probes around M in Ndir directions, log-spaced s
    import math
    Ndir = 12
    for idir in range(Ndir):
        th = 2*math.pi*idir/Ndir
        dx, dy = math.cos(th), math.sin(th)
        line = []
        for s in [0.03, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]:
            y = (Mpt[0] + mp.mpf(dx)*s, Mpt[1] + mp.mpf(dy)*s)
            try:
                rv, info = w3s.rho(y)
                line.append((s, float(rv), info['pgrad'], info['z'], info['g']))
            except Exception as e:
                line.append((s, None, None, None, str(e)[:40]))
        print("dir %2d th=%.3f:" % (idir, th))
        for row in line:
            if row[1] is None:
                print("    s=%.5f  FAILED %s" % (row[0], row[4]))
            else:
                print("    s=%.5f rho=%.3e pgrad=%.3e z=%+.2f g=%.3e" % row)
