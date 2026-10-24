"""Can an exact-value test detect the inward-rounding survivors? Probe root vs M02/M03a/M03b."""
import sys
from fractions import Fraction as Q
from pathlib import Path
target = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(target / 'experiments/universality/two_parameter_cusp_population'))
import population_certificate as pc
from mean_adaptive_confidence import log_bounds
bad = []
# sqrt / quarter / cube on non-perfect powers
for v in (Q(2), Q(3), Q(1, 3), Q(5, 7)):
    lo, hi = pc._sqrt((v, v))
    if not (lo < hi and lo * lo <= v <= hi * hi): bad.append(('_sqrt', v))
    lo, hi = pc.quarter_power_bounds(v)
    if not (lo < hi and lo ** 4 <= v <= hi ** 4): bad.append(('quarter', v))
    lo, hi = pc.cube_root_bounds(v)
    if not (lo < hi and lo ** 3 <= v <= hi ** 3): bad.append(('cube', v))
# void_interval against exact rational powers (1-m)^n
for m, n in ((Q(1, 5), 4), (Q(1, 3), 2), (Q(1, 7), 3), (Q(2, 3), 5), (Q(1, 2), 1), (Q(3, 11), 7), (Q(9, 10), 2), (Q(1, 1000), 65536)):
    lo, hi = pc.void_interval((m, m), n)
    if not (lo <= (1 - m) ** n <= hi): bad.append(('void', m, n))
# poisson_target: exp(-log(k)) = 1/k exactly; pass b = log_bounds(k) (an enclosure of log k)
for k in (Q(2), Q(3), Q(5), Q(10), Q(7, 3)):
    b = log_bounds(k, bits=pc.BITS)
    lo, hi = pc.poisson_target(b)
    if not (lo <= 1 / k <= hi): bad.append(('poisson', k))
print(target.name, 'violations:', bad if bad else 'none')
