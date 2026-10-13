import sys, pickle, time
from pathlib import Path
root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root / 'experiments/universality/two_parameter_cusp_population'))
import population_certificate as pc
out = {}
for j in range(1, 9):
    t = time.time()
    out[j] = pc.direct_mass(j)
    print(j, out[j]['status'], out[j]['evaluations'], out[j]['normalized_interval'], round(time.time()-t,1), flush=True)
pickle.dump(out, open(sys.argv[2], 'wb'))
print('DONE')
