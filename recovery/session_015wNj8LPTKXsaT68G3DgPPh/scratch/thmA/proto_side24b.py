import sys, time
sys.argv = ['x']
import importlib.util
spec = importlib.util.spec_from_file_location('ta', 'theorem_a.py'); ta = importlib.util.module_from_spec(spec); spec.loader.exec_module(ta)
from fractions import Fraction as Fr
pts = {'side24': ((Fr(6, 5), Fr(6, 5)), (Fr(1, 6), Fr(1, 6))), 'corner02': ((Fr(0), Fr(1, 64)), (Fr(63, 32), Fr(2)))}
bands = ((Fr(0), Fr(1, 2**29)), (Fr(31, 131072), Fr(1, 4096)), (Fr(31, 65536), Fr(1, 2048)))
laws = {b: ta.band_law(*b) for b in bands}
for X0 in (Fr(1, 64), Fr(1, 1024), Fr(1, 16384)):
    for th in (Fr(1, 32), Fr(1, 256), Fr(1, 2048)):
        ta.X0 = X0; ta.THETA = th
        row = []
        for name, (bb, kb) in pts.items():
            for b in bands:
                try:
                    res = ta.box_bound(laws[b], bb, kb, b[0], b[1], last_band=(b[0] == 0))
                    row.append('%s@%s=%.1f' % (name, b[1], float(res['total'])))
                except SystemExit:
                    row.append('%s@%s=FAIL' % (name, b[1]))
        print('X0', X0, 'theta', th, ' '.join(row), flush=True)
