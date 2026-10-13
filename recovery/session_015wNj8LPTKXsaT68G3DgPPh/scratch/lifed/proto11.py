import time
from fractions import Fraction as Fr
import lifecore as LC
E2, E3 = LC.E2, LC.E3
# a mid band and the last band, refined boxes near k = 1/2
for M, nm in ((E2, 'd2'), (E3, 'd3')):
    bl = M.bands()
    for (r0, r1) in (bl[0], bl[len(bl) // 2], bl[-1]):
        t = time.time(); law = M.band_law(r0, r1); tl = time.time() - t
        last = (r0 == 0)
        t = time.time()
        outs = []
        for kb in ((Fr(1, 2), Fr(17, 32)), (Fr(17, 32), Fr(9, 16)), (Fr(31, 32), Fr(1))):
            for bb in ((Fr(0), Fr(1, 8)), (Fr(7, 8), Fr(1))):
                if M is E2:
                    res = M.best_box(law, bb, kb, r0, r1, last)
                else:
                    res = M.box_bound(law, bb, kb, r0, r1, last_band=last, x0=M.COMBO[0], theta=M.COMBO[1])
                outs.append((bb, kb, float(res['total'])))
        dt = (time.time() - t) / len(outs)
        print(nm, 'band [%s,%s] law %.2fs  per box %.2fs' % (r0, r1, tl, dt), ' '.join('b%s k%s:%.4g' % (o[0][0], o[1][0], o[2]) for o in outs), flush=True)
