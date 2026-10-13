import sys, time, importlib.util
from fractions import Fraction as Fr
spec = importlib.util.spec_from_file_location('tb', 'pkt/theorem_b.py'); tb = importlib.util.module_from_spec(spec); spec.loader.exec_module(tb)
t = time.time(); tml = tb.tm_law(3); print('tm %.1fs' % (time.time() - t))
for R in (Fr(1, 4096), Fr(1, 512), Fr(1, 64)):
    il = tb.E3.band_law(Fr(0), R)
    wu = wd = 0
    for (bb, kb) in tb.SUBBOXES:
        up, dn = tb.bounds_d3((bb[0], bb[1], kb[0], kb[1]), R, tml, il)
        wu = max(wu, up[0]); wd = max(wd, dn[0])
    print('R=%s a_up=%.4f a_dn=%.4f' % (R, float(wu), float(wd)), flush=True)
