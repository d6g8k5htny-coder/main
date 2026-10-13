"""Runtime trace: which engine/theorem_b functions actually execute on #227's certificate paths (one box per path)."""
import os, sys, time, json
from fractions import Fraction as Fr
sys.path.insert(0, os.getcwd())
import theorem_v as V
hit = {}
cur = ['?']
def prof(frame, event, arg):
    if event == 'call':
        fn = os.path.basename(frame.f_code.co_filename)
        if fn in ('engine_e2.py', 'engine_e3.py', 'theorem_b.py'):
            hit.setdefault(cur[0], set()).add('%s:%s' % (fn, frame.f_code.co_name))
for d in (2, 3):
    M = V.ENG[d]; t = time.time()
    sys.setprofile(prof)
    cur[0] = 'law'; law = V.tv_law(d, torus=True); V.tv_law(d, torus=False)
    for R in (V.R_STARS[0], V.R_STARS[-1]):
        cur[0] = 'band_law'; il = M.band_law(Fr(0), R)
        fn = V.bounds_d2_v if d == 2 else V.bounds_d3_v
        cur[0] = 'bounds_v'
        for bb, kb in V.SUBBOXES[:3] + V.SUBBOXES[-3:]:
            fn((bb[0], bb[1], kb[0], kb[1]), R, law, il)
    cur[0] = 'weights'; V.k_weights(d); V.TB.reference(d)
    cur[0] = 'eps0'
    tml = V.TB.tm_law(d); il = M.band_law(Fr(0), V.R_STARS[0])
    f2 = V.TB.bounds_d2 if d == 2 else V.TB.bounds_d3
    bb, kb = V.SUBBOXES[0]; f2((bb[0], bb[1], kb[0], kb[1]), V.R_STARS[0], tml, il)
    sys.setprofile(None)
    print('d=%d %.0fs' % (d, time.time() - t), flush=True)
json.dump({k: sorted(v) for k, v in hit.items()}, open(sys.argv[1], 'w'), indent=1)
