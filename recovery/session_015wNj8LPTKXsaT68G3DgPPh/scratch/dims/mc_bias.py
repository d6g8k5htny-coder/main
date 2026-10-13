import sys, math, random
sys.argv = ['x']
import importlib.util
spec = importlib.util.spec_from_file_location('cd', 'cap_dims.py'); cd = importlib.util.module_from_spec(spec); spec.loader.exec_module(cd)
from fractions import Fraction as Fr
import multiprocessing
def job(seed):
    return cd.mc_J(2, Fr(1, 2), 500000, seed)
if __name__ == '__main__':
    with multiprocessing.Pool(4) as p:
        rows = p.map(job, [101, 202, 303, 404])
    ms = [r['J_over_2359296k3'] for r in rows]; ses = [r['se'] / 294912 for r in rows]
    m = sum(ms) / 4; se = math.sqrt(sum(s * s for s in ses)) / 4
    print('per seed', [round(x, 6) for x in ms], 'mean %.6f +- %.6f' % (m, se), 'certified 1.0231677 -> z = %.2f' % ((m - 1.0231677) / se))
