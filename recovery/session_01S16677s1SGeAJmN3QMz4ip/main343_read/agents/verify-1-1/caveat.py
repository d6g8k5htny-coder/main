# Reader's caveat: would a shared factor-2 loss be caught by the coefficient predicates?
# Use the frozen CERTIFICATE.json intersection intervals, halve them, reclassify.
import sys, json
from fractions import Fraction as Q
sys.path.insert(0, sys.argv[1])
import population_certificate as pc
cert = json.load(open(sys.argv[2]))
b = pc.coefficient_interval()
for row in cert['rows']:
    j = row['j']
    obs = tuple(Q(x) for x in row['intersection_normalized'])
    halved = (obs[0] / 2, obs[1] / 2)
    err = pc.error_interval(j)
    print(j, 'frozen:', pc.classify_coefficient(obs, b, err), ' halved(M1):', pc.classify_coefficient(halved, b, err))
# Also the overall disposition rule
from importlib import util
spec = util.spec_from_file_location('runner', sys.argv[1] + '/run_population_certificate.py')
runner = util.module_from_spec(spec); spec.loader.exec_module(runner)
st = [pc.classify_coefficient((Q(x[0])/2, Q(x[1])/2), b, pc.error_interval(r['j'])) for r in cert['rows'] for x in [r['intersection_normalized']]]
print('overall under M1 (coefficient only):', runner._overall(st))
