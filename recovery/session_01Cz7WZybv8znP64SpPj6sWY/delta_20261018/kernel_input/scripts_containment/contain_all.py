import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import contain, mpmath as mp
summ = json.load(open(sys.argv[1]))
use_dual = len(sys.argv) > 2 and sys.argv[2] == "dual"
total_fail = 0; nsamp = 0; ncert = 0
tight = []
for s in summ:
    if s["status"] != "ENCLOSED": continue
    c = json.load(open(s["path"]))
    fails, mm, ns = contain.check(c, 24, use_dual=use_dual)
    ncert += 1; nsamp += ns; total_fail += len(fails)
    for f in fails: print("FAIL", s["tag"], f)
    # record tightest relative margin vs width
    from fractions import Fraction as F
    for k,(m,L) in mm.items():
        if not k.startswith("raw"):
            lo,hi = map(F, c["bounds"][k])
            w = hi-lo
            if w > 0:
                tight.append((m/ (mp.mpf(w.numerator)/w.denominator), s["tag"], k, L, mp.nstr(m,5)))
            else:
                tight.append((mp.mpf(-1) if m < 0 else mp.mpf(0), s["tag"], k, L, "zero-width"))
print("certs", ncert, "samples", nsamp, "containment failures", total_fail)
tight.sort(key=lambda t: t[0])
for t in tight[:12]: print(" tightest margin/width", mp.nstr(t[0],5), t[1:])
