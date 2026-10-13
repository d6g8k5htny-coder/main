import sys, time, json
sys.argv = ['x']
import importlib.util
spec = importlib.util.spec_from_file_location('ta', 'theorem_a.py'); ta = importlib.util.module_from_spec(spec); spec.loader.exec_module(ta)
from fractions import Fraction as Fr
old = {tuple(r['r']): r for r in (json.loads(l) for l in open('RESULTS.json').read().split('"bands": [\n')[1].split('\n')[:-3] if l.strip().startswith('{'))} if False else None
for b in ((Fr(15, 4096), Fr(1, 256)), (Fr(5, 1024), Fr(11, 2048)), (Fr(1, 512), Fr(9, 4096))):
    t0 = time.time()
    rec = ta.band_record(b)
    mx = max(Fr(x) for x in rec['bound'])
    print(b[1], 'band max %.4f' % float(mx), 'tail_max', rec['tail_max'], 'side24', rec['side24'], '%.1fs' % (time.time() - t0), flush=True)
