"""Stability of the tiny-but-positive Cov(O) eigenvalues from p5 under doubled precision."""
import sys
sys.path.insert(0, '.')
import mpmath as mp
import p5_smallL_pairs_taylor as P  # reruns p5 at dps 80 on import; acceptable cost
for dps in (80, 160):
    mp.mp.dps = dps
    for d, L, y in [(2, mp.mpf('0.7'), [mp.mpf('0.035'), 0]), (2, mp.mpf('0.7'), [mp.mpf('0.35'), 0]),
                    (2, mp.mpf('0.7'), [mp.mpf('0.35'), mp.mpf('0.35')]), (3, mp.mpf(1), [mp.mpf('0.05'), 0, 0])]:
        pts = P.mp_lattice(L, d, dig=dps)
        print('dps=%d d=%d L=%s y=%s: min eig corr(O) = %s' % (dps, d, mp.nstr(L, 3), [mp.nstr(mp.mpf(x), 3) for x in y],
              mp.nstr(P.covO(pts, [mp.mpf(x) for x in y], d), 8)))
