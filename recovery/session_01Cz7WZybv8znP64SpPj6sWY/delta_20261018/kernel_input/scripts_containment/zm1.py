import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import contain, mpmath as mp
from fractions import Fraction as F
mp.mp.dps = 130
summ = json.load(open(sys.argv[1]))
worst = None; nf = 0; n = 0
for s in summ:
    if s["status"] != "ENCLOSED": continue
    c = json.load(open(s["path"]))
    lo, hi = F(c["request"]["L_lo"]), F(c["request"]["L_hi"])
    zl, zh = (F(x) - 1 for x in c["bounds"]["z"])
    for L in contain.samples(lo, hi, 24):
        Lm = contain.mpf(L); a = 2*mp.pi**2/Lm**2
        N = int(mp.ceil(mp.sqrt(400/a))) + 5
        zm1 = 2*mp.fsum(mp.exp(-a*m*m) for m in range(1, N+1))
        n += 1
        ml = zm1 - contain.mpf(zl); mh = contain.mpf(zh) - zm1
        if ml < 0 or mh < 0:
            nf += 1; print("FAIL", s["tag"], L, mp.nstr(zm1, 20), zl, zh)
        r = min(ml/abs(zm1), mh/abs(zm1))
        if worst is None or r < worst[0]: worst = (r, s["tag"], str(L), mp.nstr(zm1, 10), mp.nstr(ml, 5), mp.nstr(mh, 5))
print("z-1 checks", n, "failures", nf, "worst relative margin", worst)
