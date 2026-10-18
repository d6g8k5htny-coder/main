import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import contain, ref, mpmath as mp
from fractions import Fraction as F
mp.mp.dps = 130
summ = json.load(open(sys.argv[1]))
rows = []
for s in summ:
    if s["status"] != "ENCLOSED": continue
    c = json.load(open(s["path"]))
    lo, hi = F(c["request"]["L_lo"]), F(c["request"]["L_hi"])
    for L in contain.samples(lo, hi, 24):
        d = ref.direct(L)
        for k in ("mu2", "mu4", "mu6"):
            bl, bh = map(F, c["bounds"][k])
            v = d[k]
            ml, mh = v - contain.mpf(bl), contain.mpf(bh) - v
            rows.append((min(ml, mh) / v, s["tag"], k, str(L), mp.nstr(v, 8), mp.nstr(ml, 4), mp.nstr(mh, 4)))
rows.sort(key=lambda r: r[0])
print("min relative margins (margin/value):")
for r in rows[:10]: print(mp.nstr(r[0], 4), r[1:])
print("negatives:", sum(1 for r in rows if r[0] < 0))
