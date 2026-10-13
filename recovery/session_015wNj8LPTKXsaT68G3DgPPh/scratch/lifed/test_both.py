import sys, time, importlib.util
from fractions import Fraction as Fr
spec = importlib.util.spec_from_file_location('tb', 'pkt/theorem_b.py'); tb = importlib.util.module_from_spec(spec); spec.loader.exec_module(tb)
for d in (2, 3):
    tml = tb.tm_law(d)
    M = tb.ENG[d]
    fn = tb.bounds_d2 if d == 2 else tb.bounds_d3
    for R in (Fr(1, 4096), Fr(1, 512), Fr(1, 64)):
        il = M.band_law(Fr(0), R)
        wu = wd = 0
        for (bb, kb) in tb.SUBBOXES:
            up, dn = fn((bb[0], bb[1], kb[0], kb[1]), R, tml, il)
            wu = max(wu, up[0]); wd = max(wd, dn[0])
        print('d=%d R=%s a_up=%.4f a_dn=%.4f' % (d, R, float(wu), float(wd)), flush=True)
