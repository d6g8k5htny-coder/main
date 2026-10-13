import json, sys, time
from fractions import Fraction as Fr
import refint as RI
d = int(sys.argv[1]); nh = 4096 if d == 2 else 1024
M = RI.E2 if d == 2 else RI.E3
res = json.load(open('results_planar.json' if d == 2 else 'results_d3.json'))
BOXES = M.BOXES
cells = RI.b_cells(d, nh=nh, block=16)
K = RI.K_const(M, d)
Jcache = {}
def J13(kb):
    if kb not in Jcache:
        Jcache[kb] = RI.k_integral(M, 1, 3, kb[0], kb[1])
    return Jcache[kb]
Ibox = [K * RI.b_integral(M, cells, nh, bb[0], bb[1]) * J13(kb) for (bb, kb) in BOXES]
tot = sum((x.hi for x in Ibox), Fr(0))
print('total k^-5/3 integral %.10e' % float(tot))
bands = res['bands']
for rs in M.R_STARS:
    Cb = [Fr(0)] * len(BOXES)
    for b in bands:
        r1 = Fr(b['r'][1])
        if r1 <= rs:
            for i, x in enumerate(b['bound']):
                Cb[i] = max(Cb[i], Fr(x))
    S = sum((c * I.hi for c, I in zip(Cb, Ibox)), Fr(0))
    Cmax = max(Cb)
    # contributions by k-box
    byk = {}
    for (bb, kb), c, I in zip(BOXES, Cb, Ibox):
        byk[kb] = byk.get(kb, 0) + float(c * I.hi)
    print('r_* = 1/%d: sum C_box I_box = %.6e   (uniform max-C bound %.6e)  top k-boxes: %s' % (
        1 / rs, float(S), float(Cmax * tot), ', '.join('[%s,%s]:%.3g' % (kb[0], kb[1], v / float(S)) for kb, v in sorted(byk.items(), key=lambda t: -t[1])[:3])))
