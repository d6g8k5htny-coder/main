import sys
from fractions import Fraction as Q
sys.path.insert(0, sys.argv[1])
import population_certificate as pc
label = sys.argv[2]
nc = pc.negative_controls()
print(label, 'negative_controls status =', nc['status'])
for k in ('whole_rectangle_mass', 'missing_sign_detected', 'conditioning_detected'):
    print(label, ' gate', k, '=', nc['gates'][k])
print(label, ' reported whole_rectangle_mass =', nc['whole_rectangle_mass'])
c = pc.cdf_mass(Q(1, 8))
print(label, ' cdf_mass(1/8) normalized (n=4) ~', float(4*c[0]), float(4*c[1]))
print(label, ' cdf_mass(9/4) =', pc.cdf_mass(Q(9,4)), ' cdf_mass(3) =', pc.cdf_mass(Q(3)))
if len(sys.argv) > 3 and sys.argv[3] == 'direct':
    d = pc.direct_mass(1)
    print(label, ' direct_mass(1) status', d['status'], 'normalized ~', float(d['normalized_interval'][0]), float(d['normalized_interval'][1]))
