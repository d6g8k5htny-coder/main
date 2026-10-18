import json, math, sys
from fractions import Fraction as F
out = sys.argv[1]
def s(q): q = F(q); return "%d/%d" % (q.numerator, q.denominator)
def mmin(Lhi):
    a = 2*math.pi**2/float(Lhi)**2
    return max(1, math.ceil(math.sqrt(3.05/a)))
reqs = []
pts = [F(3,10), F(1,3), F(2,5), F(1,2), F(3,5), F(7,10), F(4,5), F(9,10), F(1), F(11,10), F(5,4), F(3,2), F(7,4), F(2), F(5,2),
       F(3), F(355,113), F(7,2), F(4), F(5), F(6), F(7), F(8), F(10), F(12), F(15), F(20), F(25), F(30),
       F(31415926535,10**10), F(1,7), F(1,20), F(1,50), F(7,400), F(1,58), F(60), F(100), F(150), F(160), F(165)]
for L in pts:
    mlo = mmin(L)
    for M in sorted(set([mlo, min(64, max(mlo, 2*mlo)), 64])):
        for bits, pairs, piT in [(128, 24, 32), (80, 16, 20)]:
            reqs.append(["pt L=%s M=%d b=%d p=%d" % (s(L), M, bits, pairs),
                         {"L_lo": s(L), "L_hi": s(L), "cutoff": M, "bits": bits, "pi_terms": piT, "exp_pairs": pairs, "input_role": "TEST_ONLY"}])
    # coarse settings
    reqs.append(["pt-coarse L=%s" % s(L), {"L_lo": s(L), "L_hi": s(L), "cutoff": mlo, "bits": 16, "pi_terms": 1, "exp_pairs": 1, "input_role": "TEST_ONLY"}])
ivs = [(F(3,10), F(31,100)), (F(3,10), F(1,2)), (F(1), F(2)), (F(1), 1+F(1,10**12)), (F(1,2), F(3)), (F(29,10), F(31,10)), (F(5), F(6)),
       (F(10), F(21,2)), (F(29), F(30)), (F(3,10), F(30)), (F(1), F(30)), (F(15), F(30)), (F(100), F(160)), (F(7,400), F(1,20)),
       (F(2), F(2)+F(1,2**200)), (F(1,3), F(355,113))]
for lo, hi in ivs:
    mlo = mmin(hi)
    for M in sorted(set([mlo, 64])):
        for bits, pairs, piT in [(128, 24, 32), (80, 16, 20), (16, 1, 1)]:
            reqs.append(["iv [%s,%s] M=%d b=%d p=%d" % (s(lo), s(hi), M, bits, pairs),
                         {"L_lo": s(lo), "L_hi": s(hi), "cutoff": M, "bits": bits, "pi_terms": piT, "exp_pairs": pairs,
                          "width_goal": "1/1000000", "input_role": "TEST_ONLY"}])
json.dump(reqs, open(out, "w"))
print(len(reqs))
