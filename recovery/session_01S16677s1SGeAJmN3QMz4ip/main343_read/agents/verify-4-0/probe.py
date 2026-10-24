import sys
from fractions import Fraction as Q
root = sys.argv[1]; label = sys.argv[2]
sys.path.insert(0, root + '/experiments/universality/two_parameter_cusp_population')
import population_certificate as pc
print('===', label)
# (1) literal branch: cdf_mass(9/4) must not touch root_bracket or arithmetic
calls = []
orig = pc.root_bracket
def spy(tau):
    calls.append(tau); raise AssertionError('root_bracket consulted')
pc.root_bracket = spy
try:
    print('cdf_mass(9/4) =', pc.cdf_mass(Q(9, 4)), '| cdf_mass(3) =', pc.cdf_mass(Q(3)), '| root_bracket calls:', calls)
finally:
    pc.root_bracket = orig
# (2) negative controls
nc = pc.negative_controls()
print('negative_controls status:', nc['status'], '| all gates true:', all(nc['gates'].values()), '| n gates:', len(nc['gates']), '| n controls:', len(nc['controls']))
print('gates:', nc['gates'])
print('whole_rectangle_mass:', nc['whole_rectangle_mass'])
# (3) computed closed form just below saturation
lo, hi = pc.cdf_mass(Q(9, 4) - Q(1, 1 << 40))
print('cdf_mass(9/4 - 2^-40): 1/5 - lo =', float(Q(1, 5) - lo), '| hi <= 1/5:', hi <= Q(1, 5), '| hi =', float(hi))
# (4) computed closed form at tau_1 = 1/8 vs the frozen certificate's expectation
print('cdf_mass(1/8) =', [float(x) for x in pc.cdf_mass(Q(1, 8))])
# (5) patch critical_fixture so the theta=2 adversarial control reports the fixed-left 32 a^4 lifetime
orig_cf = pc.critical_fixture
def bad(a, h, sign=1):
    row = orig_cf(a, h, sign)
    if h == 2 * a:
        row['lifetime'] = 32 * a ** 4
    return row
pc.critical_fixture = bad
nc2 = pc.negative_controls()
print('with fixed-left adversarial lifetime: status =', nc2['status'], '| actual_theta2_gap =', nc2['gates'].get('actual_theta2_gap'), '| reject_fixed_left =', nc2['gates'].get('reject_fixed_left', 'ABSENT'))
pc.critical_fixture = orig_cf
