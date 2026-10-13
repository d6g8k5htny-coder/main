import sys, time
sys.path.insert(0, '.')
import certified_d2 as c
from fractions import Fraction as Q
def show(x, d=22):
    return '[%s, %s] w=%.1e' % (format(float(x.lo), '.15e'), format(float(x.hi), '.15e'), float(x.hi - x.lo))
for name, L in (('24', c.I(24)), ('8', c.I(8)), ('2pi', 2*c.PI), ('4', c.I(4)), ('3', c.I(3)), ('2', c.I(2))):
    t = time.time()
    mi = c.moments_image(L)
    md = c.moments_dual(L)
    ok = all(a.lo <= b.hi and b.lo <= a.hi for a, b in zip(mi, md[:3]))
    k = c.cumulants(*mi)
    print(name, 'intersect', ok, 'k2', show(k[0]), 'k4/k2^2 %.6e' % float(k[1].lo/(k[0].lo**2)), '%.1fs' % (time.time()-t))
