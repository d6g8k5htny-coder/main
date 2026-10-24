import sys
from fractions import Fraction as Q
sys.path.insert(0, sys.argv[1] + '/experiments/universality/two_parameter_cusp_population')
import population_certificate as pc
print('tree', sys.argv[1].rsplit('/',1)[-1])
viol = []
for v in (Q(2), Q(3), Q(1, 3), Q(5, 7)):
    lo, hi = pc._sqrt((v, v))
    q = pc.quarter_power_bounds(v)
    c = pc.cube_root_bounds(v)
    ok_s = lo <= hi and lo*lo <= v <= hi*hi
    ok_q = q[0] <= q[1] and q[0]**4 <= v <= q[1]**4
    ok_c = c[0] <= c[1] and c[0]**3 <= v <= c[1]**3
    print(f'  v={v}: sqrt lo<=hi {lo<=hi} lo^2<=v {lo*lo<=v} v<=hi^2 {v<=hi*hi} | quarter ok {ok_q} | cube ok {ok_c}')
    if not (ok_s and ok_q and ok_c):
        viol.append(v)
print('  perfect powers: quarter(16)=', pc.quarter_power_bounds(Q(16)), ' cube(8)=', pc.cube_root_bounds(Q(8)))
print('  violations:', viol)
