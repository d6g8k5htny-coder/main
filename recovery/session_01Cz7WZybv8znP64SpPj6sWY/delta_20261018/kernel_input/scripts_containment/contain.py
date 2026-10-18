"""mpmath containment checker (venv python -I). argv: cert.json [nsamples]
Checks every certified bound contains the high-precision reference at endpoints and
interior sample points of [L_lo, L_hi]."""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref, mpmath as mp
from fractions import Fraction as F
mp.mp.dps = 130

def samples(lo, hi, n):
    if lo == hi:
        return [lo]
    pts = [lo, hi]
    for j in range(1, n + 1):
        pts.append(lo + (hi - lo) * F(j, n + 1))
    # a few irregular interior points
    for num, den in [(1, 997), (996, 997), (1, 3), (2, 3), (1, 7), (355, 1130)]:
        pts.append(lo + (hi - lo) * F(num, den))
    return sorted(set(pts))

def mpf(q):
    return mp.mpf(q.numerator) / q.denominator

def check(cert, n=24, use_dual=False):
    r = cert["request"]
    lo, hi = F(r["L_lo"]), F(r["L_hi"])
    b = {k: (F(v[0]), F(v[1])) for k, v in cert["bounds"].items()}
    raw = {k: (F(v[0]), F(v[1])) for k, v in cert["raw"].items()}
    fails = []
    minmargin = {}
    for L in samples(lo, hi, n):
        d = ref.direct(L)
        vals = {"z": d["z"], "mu2": d["mu2"], "mu4": d["mu4"], "mu6": d["mu6"]}
        rawv = {"0": d["S"][0], "2": d["S"][1], "4": d["S"][2], "6": d["S"][3]}
        if use_dual:
            u = ref.dual(L)
            for k in vals:
                if abs(vals[k] - u[k]) > mp.mpf(10) ** -100 * max(abs(vals[k]), mp.mpf(10) ** -200):
                    fails.append(("DUAL_MISMATCH", str(L), k))
        for k, v in list(vals.items()) + [("raw" + k2, v2) for k2, v2 in rawv.items()]:
            lo_b, hi_b = b[k] if not k.startswith("raw") else raw[k[3:]]
            ml, mh = v - mpf(lo_b), mpf(hi_b) - v
            if ml < 0 or mh < 0:
                fails.append(("NONCONTAINMENT", str(L), k, mp.nstr(v, 30), [str(lo_b), str(hi_b)], mp.nstr(ml, 5), mp.nstr(mh, 5)))
            m = min(ml, mh)
            if k not in minmargin or m < minmargin[k][0]:
                minmargin[k] = (m, str(L))
    return fails, minmargin, len(samples(lo, hi, n))

if __name__ == "__main__":
    cert = json.load(open(sys.argv[1]))
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    fails, mm, ns = check(cert, n, use_dual=True)
    print("samples", ns, "fails", len(fails))
    for f in fails: print(" ", f)
    for k, (m, L) in mm.items(): print(" minmargin", k, mp.nstr(m, 6), "at L=", L)
