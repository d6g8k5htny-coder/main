import time, sys
from fractions import Fraction as Fr
import lifecore as LC
import tmfx
import lifebound as LB
d = int(sys.argv[1])
M = LC.E2 if d == 2 else LC.E3
names = ['w', 'om'] if d == 2 else ['a11', 'a22', 'a12', 'o11', 'o22', 'o12']
t = time.time()
tml = tmfx.tm_law(M, names, 6, N=8)
print('TM law %.1fs' % (time.time() - t), flush=True)
for R in (Fr(1, 64), Fr(1, 512), Fr(1, 4096)):
    t = time.time()
    il = M.band_law(Fr(0), R)
    worst = {'ua': 0, 'ue': 0, 'da': 0, 'de': 0}
    rows = []
    for i in range(8):
        for j in range(12):
            Q = (Fr(i, 8), Fr(i + 1, 8), Fr(1, 2) + Fr(j, 8), Fr(1, 2) + Fr(j + 1, 8))
            (up, dn), info = (LB.bounds_d2 if d == 2 else LB.bounds_d3)(Q, R, tml, il)
            worst['ua'] = max(worst['ua'], up[0]); worst['ue'] = max(worst['ue'], up[1])
            worst['da'] = max(worst['da'], dn[0]); worst['de'] = max(worst['de'], dn[1])
            if (i in (0, 7)) and (j in (0, 11)):
                rows.append((Q, up, dn, info))
    print('R=1/%d  %.1fs  worst a_up=%.4f eps_up=%.3e a_dn=%.4f eps_dn=%.3e   (a R^2: %.3e)' % (
        1 / R, time.time() - t, float(worst['ua']), float(worst['ue']), float(worst['da']), float(worst['de']), float(worst['ua'] * R * R)), flush=True)
    for Q, up, dn, info in rows:
        print('   b[%s,%s] k[%s,%s] up=(%.3f, %.2e) dn=(%.3f, %.2e)' % (Q[0], Q[1], Q[2], Q[3], float(up[0]), float(up[1]), float(dn[0]), float(dn[1])),
              {k: (tuple('%.3g' % float(x) for x in v) if isinstance(v, tuple) and not isinstance(v[0], tuple) else
                   ('%.3g' % float(v) if not isinstance(v, (tuple, list)) else ([('%.3g' % float(y)) for y in v] if isinstance(v, list) else str([[('%.3g' % float(z)) for z in y] for y in v]))))
               for k, v in info.items()}, flush=True)
