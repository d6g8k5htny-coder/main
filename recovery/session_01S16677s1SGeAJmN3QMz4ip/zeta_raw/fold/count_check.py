"""Independent check of the first-collision claims by direct real-zero counts (fine scan,
step 0.02, with critical-point refinement so close pairs are not missed)."""
import sys, mpmath as mp
from hflow import HFlow
from zeros import scan
hf = HFlow(dps=80)
a, b, t = mp.mpf(sys.argv[1]), mp.mpf(sys.argv[2]), mp.mpf(sys.argv[3])
zs, cs = scan(hf, t, a, b, mp.mpf('0.02'))
print('t=%s window [%s,%s]: %d real zeros; first/last: %s %s' % (sys.argv[3], sys.argv[1], sys.argv[2], len(zs), mp.nstr(zs[0], 10), mp.nstr(zs[-1], 10)))
