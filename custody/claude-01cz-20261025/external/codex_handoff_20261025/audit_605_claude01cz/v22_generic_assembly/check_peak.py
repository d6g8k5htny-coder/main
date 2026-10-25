# Diagnostic only: max over s (fine grid near the crossover) of Shell(r,s)/r^5, to test per-shell uniformity in r.
import math, importlib.util, sys
spec = importlib.util.spec_from_file_location("cs", sys.argv[1]); cs = importlib.util.module_from_spec(spec)
src = open(sys.argv[1]).read().split("worst = 0.0")[0]; exec(src, cs.__dict__)
for r in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
    sc = r**0.75; best = (0, 0)
    for j in range(-40, 41):
        s = sc*2**(j/10)
        if s < 4*r or s > 0.1: continue
        v = r**2*s*cs.I(r, s)/r**5
        if v > best[0]: best = (v, s/sc)
    print(f"r={r:g}: max Shell/r^5 = {best[0]:.3f} at s = {best[1]:.2f} * r^(3/4)")
