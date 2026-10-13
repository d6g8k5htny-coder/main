"""Refined cap sweep: per band, the engines' certified bound of sup Q^W(G_r^c)/r^3 on refined (b, k) boxes near k = 1/2."""
import json, sys, time, multiprocessing
from fractions import Fraction as Fr
import importlib.util, os
HERE = os.path.dirname(os.path.abspath(__file__))

def load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

B_REF = tuple((Fr(i, 4), Fr(i + 1, 4)) for i in range(4))
K_REF = tuple((Fr(1, 2) + Fr(i, 32), Fr(1, 2) + Fr(i + 1, 32)) for i in range(8)) \
    + tuple((Fr(3, 4) + Fr(i, 16), Fr(3, 4) + Fr(i + 1, 16)) for i in range(4)) + ((Fr(1), Fr(3, 2)), (Fr(3, 2), Fr(2)))
CAP_BOXES = tuple((bb, kb) for kb in K_REF for bb in B_REF)
M = None

def init(d):
    global M
    M = load('eng%d' % d, 'theorem_a.py' if d == 2 else 'theorem_a3.py')

def band(args):
    d, r0, r1 = args
    law = M.band_law(r0, r1)
    last = (r0 == 0)
    x0, th = (M.COMBOS[0] if d == 2 else M.COMBO)
    out = []
    for bb, kb in CAP_BOXES:
        res = M.box_bound(law, bb, kb, r0, r1, last_band=last, x0=x0, theta=th)
        out.append(M.dec_up(res['total']))
    return {'r': [str(r0), str(r1)], 'bound': out}

if __name__ == '__main__':
    d = int(sys.argv[1]); procs = int(sys.argv[2])
    init(d)
    bl = M.bands()
    t = time.time()
    with multiprocessing.Pool(procs, initializer=init, initargs=(d,)) as pool:
        recs = pool.map(band, [(d, r0, r1) for r0, r1 in bl], chunksize=1)
    json.dump({'d': d, 'boxes': [[str(bb[0]), str(bb[1]), str(kb[0]), str(kb[1])] for bb, kb in CAP_BOXES], 'bands': recs},
              open('cap_sweep_d%d.json' % d, 'w'), separators=(',', ':'))
    print('d=%d bands=%d boxes=%d time %.1fs' % (d, len(bl), len(CAP_BOXES), time.time() - t))
