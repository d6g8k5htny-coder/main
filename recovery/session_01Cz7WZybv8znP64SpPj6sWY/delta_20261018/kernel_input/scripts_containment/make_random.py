import json, math, random, sys
from fractions import Fraction as F
rnd = random.Random(20261018)
def s(q): return "%d/%d" % (q.numerator, q.denominator)
reqs = []
for i in range(200):
    lo = F(math.exp(rnd.uniform(math.log(0.3), math.log(30)))).limit_denominator(rnd.choice([10, 1000, 10**6, 2**40]))
    if rnd.random() < 0.5:
        hi = lo
    else:
        hi = lo * (1 + F(rnd.choice([1, 10, 1000, 10**5, 10**9]), 10**rnd.randint(1, 12)))
        hi = min(hi, F(30)).limit_denominator(10**15) if hi > lo else lo
        if hi < lo: hi = lo
    a_lo = 2*math.pi**2/float(hi)**2; a_hi = 2*math.pi**2/float(lo)**2
    mmin = max(1, math.ceil(math.sqrt(3.05/a_lo)))
    mmax = min(64, int(math.sqrt(65000/a_hi)))
    if mmax < mmin: mmax = mmin
    M = rnd.randint(mmin, mmax)
    req = {"L_lo": s(lo), "L_hi": s(hi), "cutoff": M, "bits": rnd.randint(16, 128), "pi_terms": rnd.randint(1, 32),
           "exp_pairs": rnd.randint(1, 24), "input_role": rnd.choice(["TEST_ONLY", "UNASSIGNED_PARAMETER_INPUT"])}
    if rnd.random() < 0.5: req["width_goal"] = s(F(1, 10**rnd.randint(1, 30)))
    reqs.append(["rand%03d" % i, req])
json.dump(reqs, open(sys.argv[1], "w"))
