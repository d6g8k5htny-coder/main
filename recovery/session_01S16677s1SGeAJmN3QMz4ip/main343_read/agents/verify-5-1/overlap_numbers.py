import sys, pathlib
from fractions import Fraction as Q
root = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root / 'experiments/universality/two_parameter_cusp_population'))
import population_certificate as pc
row = pc.direct_mass(1); cdf = pc.cdf_mass(Q(1, 8)); d = row['interval']; n = row['n']
print('direct(unnorm) width %.4e; normalized width %.4e; arithmetic %.4e' % (d[1]-d[0], row['normalized_width'], row['arithmetic_width']))
print('cdf inside direct:', d[0] <= cdf[0] and cdf[1] <= d[1])
print('max inward shift of direct LOWER before overlap breaks (unnorm) %.4e (normalized %.4e)' % (cdf[1]-d[0], n*(cdf[1]-d[0])))
print('max inward shift of direct UPPER before overlap breaks (unnorm) %.4e (normalized %.4e)' % (d[1]-cdf[0], n*(d[1]-cdf[0])))
print('M02 per-point perturbation ~2^-128 = %.3e; M04 sliver = %.3e; M06 lost tail margin = %.3e' % (2.0**-128, row['sliver_width'], row['discretization_components']['tail']))
