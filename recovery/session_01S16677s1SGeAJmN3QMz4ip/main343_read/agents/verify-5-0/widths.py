import sys
from fractions import Fraction as Q
sys.path.insert(0, sys.argv[1] + '/experiments/universality/two_parameter_cusp_population')
import population_certificate as pc
row = pc.direct_mass(1)
cdf = pc.cdf_mass(Q(1, 8))
n = row['n']
L, U = row['normalized_interval']
c0, c1 = n * cdf[0], n * cdf[1]
f = lambda x: format(float(x), '.4e')
print('tree', sys.argv[1].rsplit('/', 1)[-1], 'status', row['status'], 'gates_all', all(row['gates'].values()))
print('  normalized width         ', f(U - L))
print('  interval (per-copy) width', f(row['interval'][1] - row['interval'][0]))
print('  cdf normalized width     ', f(c1 - c0))
print('  sliver_width             ', f(row['sliver_width']))
print('  disc core / tail         ', f(row['discretization_components']['core']), f(row['discretization_components']['tail']))
print('  arithmetic_width         ', f(row['arithmetic_width']))
print('  overlap margin lower (c1 - L)', f(c1 - L), ' upper (U - c0)', f(U - c0))
print('  composition check: lower == core0+tail0', L == row['core_interval'][0] + row['tail_interval'][0],
      ' upper == core1+tail1+sliver', U == row['core_interval'][1] + row['tail_interval'][1] + row['sliver_width'])
print('  tau,n', row['tau'], n, ' failing gates', [k for k, v in row['gates'].items() if not v])
