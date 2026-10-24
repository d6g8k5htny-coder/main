import sys, importlib
from fractions import Fraction as Q
popdir = sys.argv[1]
sys.path.insert(0, popdir)
import population_certificate as pc
print('module:', pc.__file__)
# (a) saturation value and the three mass gates
nc = pc.negative_controls()
print('negative_controls status:', nc['status'])
print('mass gates:', {k: nc['gates'][k] for k in ('whole_rectangle_mass','missing_sign_detected','conditioning_detected')})
print('cdf_mass(9/4) =', pc.cdf_mass(Q(9,4)), ' cdf_mass(3) =', pc.cdf_mass(Q(3)))
# (b) does any route code run at tau>=9/4?  Break every primitive cdf_mass uses and retry.
orig = {n: getattr(pc, n) for n in ('root_bracket','_scale','_add','_subtract','_multiply','_reciprocal','_sqrt','_outward','quarter_power_bounds')}
def boom(*a, **k): raise RuntimeError('route primitive called')
for n in orig: setattr(pc, n, boom)
try:
    print('cdf_mass(9/4) with ALL route primitives replaced by raise:', pc.cdf_mass(Q(9,4)))
    nc2 = pc.negative_controls()
    print('negative_controls with ALL route primitives replaced by raise:', nc2['status'], {k: nc2['gates'][k] for k in ('whole_rectangle_mass','missing_sign_detected','conditioning_detected')})
finally:
    for n, f in orig.items(): setattr(pc, n, f)
# (c) computed values the routes actually produce
m = pc.cdf_mass(Q(1,8)); print('row-1 CDF normalized 4*m(1/8) ~', float(4*m[0]), float(4*m[1]))
near = pc.cdf_mass(Q(9,4) - Q(1, 1<<40)); print('computed cdf_mass(9/4 - 2^-40) ~', float(near[0]), float(near[1]), ' (1/5 =', 0.2, ')')
# (d) coefficient predicate per row from the CDF route alone (reader caveat)
b = pc.coefficient_interval()
for j in range(1, 9):
    tau, n = pc._schedule(j)
    cm = pc.cdf_mass(tau); obs = (n*cm[0], n*cm[1])
    print(' row', j, 'CDF-normalized ~ %.5f..%.5f' % (float(obs[0]), float(obs[1])), 'classify_coefficient:', pc.classify_coefficient(obs, b, pc.error_interval(j)))
