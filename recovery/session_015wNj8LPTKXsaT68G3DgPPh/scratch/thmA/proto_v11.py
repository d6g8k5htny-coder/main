import sys, time
sys.argv = ['x']
import importlib.util
spec = importlib.util.spec_from_file_location('ta', 'theorem_a.py'); ta = importlib.util.module_from_spec(spec); spec.loader.exec_module(ta)
from fractions import Fraction as Fr
for b in ((Fr(0), Fr(1, 2**29)), (Fr(31, 131072), Fr(1, 4096)), (Fr(31, 65536), Fr(1, 2048)), (Fr(31, 32768), Fr(1, 1024)), (Fr(31, 16384), Fr(1, 512)), (Fr(15, 1024), Fr(1, 64))):
    t0 = time.time()
    rec = ta.band_record(b)
    old_max = max(Fr(x) for x in rec['bound'])
    print(b[1], 'band max %.4f' % float(old_max), 'side24', rec['side24'], 'zmin', min(rec['zfloor'], key=Fr), '%.1fs' % (time.time() - t0), flush=True)
