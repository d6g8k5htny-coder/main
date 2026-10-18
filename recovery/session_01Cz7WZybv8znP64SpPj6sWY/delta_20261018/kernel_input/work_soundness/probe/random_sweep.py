import sys, json, random
from fractions import Fraction as Q
sys.path.insert(0, sys.argv[1])
import mpmath as mp
from moment_kernel import certify, KernelHold
from verify_certificate import verify
mp.mp.dps = 50
rng = random.Random(20261018)
def mpq(q): q = Q(q); return mp.mpf(q.numerator)/q.denominator
def truth(L):
    h = 2*mp.pi/mpq(L); N = int(40/h)+5
    S = [mp.mpf(1), mp.mpf(0), mp.mpf(0), mp.mpf(0)]
    for m in range(1, N+1):
        q2 = (h*m)**2; w = mp.e**(-q2/2)
        for k in range(4): S[k] += 2*q2**k*w
    return {"z": S[0], "mu2": S[1]/S[0], "mu4": S[2]/S[0], "mu6": S[3]/S[0]}
stats = {"ok": 0, "hold": {}, "reject": 0, "viol": 0}
for it in range(300):
    lo = Q(rng.randint(1, 4000), rng.randint(1, 40))
    if lo < Q(1, 5) or lo > 165: continue
    hi = lo if rng.random() < 0.4 else lo * (1 + Q(rng.randint(1, 500), 1000))
    M = rng.randint(1, 64)
    req = dict(L_lo=f"{lo.numerator}/{lo.denominator}", L_hi=f"{hi.numerator}/{hi.denominator}", cutoff=M,
               bits=rng.choice([16, 24, 40, 80, 128]), pi_terms=rng.randint(1, 32), exp_pairs=rng.randint(1, 24),
               input_role="TEST_ONLY")
    if rng.random() < 0.5: req["width_goal"] = "1/" + str(10**rng.randint(0, 20))
    try:
        c = J = json.loads(json.dumps(certify(req)))
    except KernelHold as e:
        stats["hold"][e.reason] = stats["hold"].get(e.reason, 0) + 1; continue
    try: verify(c)
    except ValueError as e:
        stats["reject"] += 1; print("REJECT", req, e); continue
    for L in {lo, hi, lo + (hi-lo)*Q(rng.randint(0, 1000), 1000), lo + (hi-lo)*Q(1, 3)}:
        tr = truth(L)
        for k, v in tr.items():
            a, b = (mpq(x) for x in c["bounds"][k])
            if not a <= v <= b:
                stats["viol"] += 1; print("VIOLATION", req, L, k, v, a, b)
    stats["ok"] += 1
print(stats)
