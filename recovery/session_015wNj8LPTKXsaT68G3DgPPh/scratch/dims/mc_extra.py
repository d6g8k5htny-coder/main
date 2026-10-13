import sys, json, multiprocessing
sys.argv = ['x']
import importlib.util
spec = importlib.util.spec_from_file_location('cd', '/home/user/wt_dims/frontiers/c8_cap_failure_coefficient_dims_20261001/cap_coefficient_dims.py'); cd = importlib.util.module_from_spec(spec); spec.loader.exec_module(cd)
from fractions import Fraction as Fr
def job(seed):
    return cd.mc_J(2, Fr(1, 2), 500000, seed)
if __name__ == '__main__':
    with multiprocessing.Pool(4) as p:
        rows = p.map(job, [101, 202, 303, 404])
    path = '/home/user/wt_dims/frontiers/c8_cap_failure_coefficient_dims_20261001/RESULTS.json'
    d = json.load(open(path))
    d['controls']['mc_planar_extra'] = rows
    with open(path, 'w') as fh:
        json.dump(d, fh, indent=1, sort_keys=True); fh.write('\n')
    print([round(r['J_over_2359296k3'], 6) for r in rows])
