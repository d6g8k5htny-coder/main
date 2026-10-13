import sys, pathlib
from fractions import Fraction as Q
from unittest import mock
root = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root / 'experiments/universality/two_parameter_cusp_population'))
import population_certificate as pc
tag = root.name
f = lambda q: f"{float(q):.6e}"
# (a) polarity of elementary enclosures
viol = []
for v in (Q(2), Q(3), Q(1, 3), Q(5, 7)):
    lo, hi = pc._sqrt((v, v))
    if not (lo < hi and lo * lo <= v <= hi * hi): viol.append(('_sqrt', str(v), 'lo^2-v=' + f(lo*lo - v)))
    lo, hi = pc.quarter_power_bounds(v)
    if not (lo < hi and lo ** 4 <= v <= hi ** 4): viol.append(('quarter', str(v), 'lo^4-v=' + f(lo**4 - v)))
    lo, hi = pc.cube_root_bounds(v)
    if not (lo < hi and lo ** 3 <= v <= hi ** 3): viol.append(('cube', str(v)))
print(tag, 'perfect powers:', pc.quarter_power_bounds(Q(16)), pc.cube_root_bounds(Q(8)), '| polarity violations:', viol or 'none')
# (b) j=1 direct row and cdf overlap
row = pc.direct_mass(1); cdf = pc.cdf_mass(Q(1, 8))
d = row['interval']; nm = row['normalized_interval']
print(tag, 'status', row['status'], 'gates_all', all(row['gates'].values()), 'failed_gates', [k for k, v in row['gates'].items() if not v])
print(tag, 'direct width', f(d[1]-d[0]), 'cdf width', f(cdf[1]-cdf[0]), 'overlap slack', f(min(d[1], cdf[1]) - max(d[0], cdf[0])))
print(tag, 'direct', f(d[0]), f(d[1]), '| cdf', f(cdf[0]), f(cdf[1]))
print(tag, 'sliver_width', f(row['sliver_width']), 'disc', {k: f(v) for k, v in row['discretization_components'].items()})
core, tail, sl = row['core_interval'], row['tail_interval'], row['sliver_width']
print(tag, 'composition replay: lo ok', nm[0] == core[0] + tail[0], '| hi == core+tail+sliver', nm[1] == core[1] + tail[1] + sl, '| hi == core+tail (no sliver)', nm[1] == core[1] + tail[1])
print(tag, 'tail_interval', f(tail[0]), f(tail[1]), 'tail hi-lo', f(tail[1]-tail[0]))
# (d) failing-gate row via coarse core panels
with mock.patch.object(pc, 'CORE_PANELS', 64):
    r64 = pc.direct_mass(1)
print(tag, 'CORE_PANELS=64 -> status', r64['status'], 'normalized_width gate', r64['gates'].get('normalized_width'), 'failed', [k for k, v in r64['gates'].items() if not v])
