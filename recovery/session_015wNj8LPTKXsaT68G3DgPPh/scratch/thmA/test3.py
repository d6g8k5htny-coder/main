import time
from fractions import Fraction as Fr
import theorem_a as ta
r1 = Fr(1, 512); r0 = r1 * Fr(1000, 1044)
t0 = time.time(); law = ta.band_law(r0, r1); print('law', round(time.time() - t0, 2))
print('mu_w k-part width', float(law.beta['w'].width()))
for kb in [(Fr(2) - Fr(1, 64), Fr(2)), (Fr(2) - Fr(1, 4096), Fr(2)), (Fr(1, 2), Fr(1)), (Fr(15, 8), Fr(31, 16))]:
    for bb in [(Fr(0), Fr(1, 64)), (Fr(1, 4), Fr(1))]:
        t0 = time.time()
        res = ta.box_bound(law, bb, kb, r0, r1)
        cg = 5324360.44 if kb[1] == 2 else None
        print('b', [float(x) for x in bb], 'k', [float(x) for x in kb], 'total %.6g' % float(res['total']), 'main %.6g' % float(res['main']),
              'tail %.3g' % float(res['tail']), 'El2 %.5f corr %.5f' % (float(res['El2']), float(res['corr'])),
              'B/A %.4f C/A %.5f' % (float(res['termB'] / res['termA']), float(res['termC'] / res['termA'])),
              ('ratio %.5f' % (float(res['total']) / cg)) if cg else '', 't %.2fs' % (time.time() - t0))
