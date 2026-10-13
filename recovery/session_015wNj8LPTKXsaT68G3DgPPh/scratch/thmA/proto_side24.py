import sys, time, json
sys.argv = ['x']
import importlib.util
spec = importlib.util.spec_from_file_location('ta', 'theorem_a.py'); ta = importlib.util.module_from_spec(spec); spec.loader.exec_module(ta)
from fractions import Fraction as Fr
pt_b = (Fr(6, 5), Fr(6, 5)); pt_k = (Fr(1, 6), Fr(1, 6))
for th in (Fr(1, 32), Fr(1, 256)):
    ta.THETA = th
    for (r0, r1) in ((Fr(0), Fr(1, 2**29)), (Fr(31, 131072), Fr(1, 4096)), (Fr(31, 65536), Fr(1, 2048)), (Fr(31, 32768), Fr(1, 1024))):
        t0 = time.time()
        law = ta.band_law(r0, r1)
        try:
            res = ta.box_bound(law, pt_b, pt_k, r0, r1, last_band=(r0 == 0))
            print('theta', th, 'band', r1, 'total %.1f main %.1f tail %.3g zt %.5f termA %.4g termB %.4g termC %.4g pbar %.4g' % (
                float(res['total']), float(res['main']), float(res['tail']), float(res['zt']), float(res['termA']), float(res['termB']), float(res['termC']), float(res['pbar'])),
                'qA2 %.3g qfar %.3g qx %.3g' % (float(res['qA2']), float(res['qfar']), float(res['qx'])), '%.1fs' % (time.time() - t0), flush=True)
        except SystemExit:
            print('theta', th, 'band', r1, 'require failed', flush=True)
