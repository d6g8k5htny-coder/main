"""Compute the eight pristine direct_mass rows once and pickle them for fast
runner-level demonstrations (the runner only reads status/normalized_interval/evaluations)."""
import pickle, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'root/experiments/universality/two_parameter_cusp_population'))
import population_certificate as pc
rows, t0 = {}, time.monotonic()
for j in range(1, 9):
    t = time.monotonic()
    rows[j] = pc.direct_mass(j)
    print(j, rows[j]['status'], rows[j]['evaluations'], round(time.monotonic() - t, 1), 's', flush=True)
pickle.dump(rows, (HERE / 'direct_cache.pkl').open('wb'))
print('total', round(time.monotonic() - t0, 1), 's')
