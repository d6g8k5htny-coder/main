"""Check of the first-collision claims by direct real-zero counts (fine scan,
step 0.02, with critical-point refinement so close pairs are not missed).

Usage: python3 dbn/count_check.py <a> <b> <t>; stdout saved as results/dbn/count_<a>_<b>_<t>.out.
Why counts confirm "first collision": in s = -t, H solves the forward heat equation, so the
number of sign changes on a fixed window is non-increasing apart from flux through the window
edges (Sturm / zero-number property, from memory; README section 5.2). Real zeros in a fixed
window therefore cannot be created as t decreases, degenerate events included; Proposition 1
describes only the nondegenerate case. Zero 46 sits 0.5 from the z = 270 edge and moves
inward (269.513 at t = 0, 269.496 at t = -0.38), so there is no boundary flux.
"""
import sys, mpmath as mp
from hflow import HFlow
from zeros import scan
hf = HFlow(dps=80)
a, b, t = mp.mpf(sys.argv[1]), mp.mpf(sys.argv[2]), mp.mpf(sys.argv[3])
zs, cs = scan(hf, t, a, b, mp.mpf('0.02'))
print('t=%s window [%s,%s]: %d real zeros; first/last: %s %s' % (sys.argv[3], sys.argv[1], sys.argv[2], len(zs), mp.nstr(zs[0], 10), mp.nstr(zs[-1], 10)))
