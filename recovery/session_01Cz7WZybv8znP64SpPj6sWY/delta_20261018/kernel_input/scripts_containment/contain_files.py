import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import contain, mpmath as mp
tot = 0; nf = 0
for fn in sys.argv[1:]:
    if not fn.endswith(".json") or fn.endswith("mutations.json"): continue
    c = json.load(open(fn))
    fails, mm, ns = contain.check(c, 24)
    tot += ns; nf += len(fails)
    for f in fails: print("FAIL", fn, f)
    print(os.path.basename(fn), "L=[%s,%s]" % (c["request"]["L_lo"][:20], c["request"]["L_hi"][:20]), "samples", ns, "fails", len(fails), "min mu2 margin", mp.nstr(mm["mu2"][0], 4))
print("TOTAL samples", tot, "fails", nf)
