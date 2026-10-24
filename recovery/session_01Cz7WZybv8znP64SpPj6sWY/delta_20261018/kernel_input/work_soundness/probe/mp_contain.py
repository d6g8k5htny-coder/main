"""Independent mpmath containment check of producer certificates (forall-L claim)."""
import sys, json
from fractions import Fraction as Q
KDIR = sys.argv[1]
sys.path.insert(0, KDIR)
import mpmath as mp
from moment_kernel import certify, KernelHold
from verify_certificate import verify
mp.mp.dps = 80

def truth(L):
    L = mp.mpf(L.numerator)/L.denominator
    h = 2*mp.pi/L
    N = int(40/h) + 5
    S = [mp.mpf(0)]*4
    for m in range(-N, N+1):
        w = mp.e**(-(h*m)**2/2)
        q2 = (h*m)**2
        for k in range(4):
            S[k] += q2**k * w if (k or True) else 0
    # m=0: q2**0 = 1 -> contributes 1 to S0, 0 to others (0**k for k>0 is 0)
    return {"z": S[0], "mu2": S[1]/S[0], "mu4": S[2]/S[0], "mu6": S[3]/S[0]}

def mpq(s):
    q = Q(s); return mp.mpf(q.numerator)/q.denominator

cases = []
def add(lo, hi, **kw): cases.append((lo, hi, kw))
add("1/1","1/1", cutoff=2)
add("1/1","2/1", cutoff=2)
add("3/2","3/2", cutoff=3)
add("1/1","1/1", cutoff=1, bits=16, pi_terms=1, exp_pairs=1)
add("1/1","3/1", cutoff=3, bits=16, pi_terms=1, exp_pairs=1)
add("5/1","7/1", cutoff=4)
add("10/1","10/1", cutoff=4)
add("10/1","20/1", cutoff=8)
add("40/1","40/1", cutoff=16)
add("100/1","100/1", cutoff=40)
add("150/1","160/1", cutoff=64)
add("163/1","163/1", cutoff=64)
add("2/1","50/1", cutoff=64)
add("6/5","6/5", cutoff=64)
add("1/2","1/2", cutoff=1)
add("1/10","1/10", cutoff=1)
add("1/100","1/100", cutoff=1)
add("10/1","10/1", cutoff=4, bits=16, pi_terms=1, exp_pairs=1)
add("30/1","30/1", cutoff=12, bits=16, pi_terms=2, exp_pairs=1)
add("7/3","9/2", cutoff=2, bits=128, pi_terms=32, exp_pairs=24)
bad = 0
for lo, hi, kw in cases:
    req = dict(L_lo=lo, L_hi=hi, input_role="TEST_ONLY", **kw)
    try:
        c = certify(req)
    except KernelHold as e:
        print("HOLD", lo, hi, kw, e.reason); continue
    c = json.loads(json.dumps(c))
    try:
        verify(c); vr = "ACCEPT"
    except ValueError as e:
        vr = "REJECT " + str(e)
    Llo, Lhi = Q(lo), Q(hi)
    pts = [Llo + (Lhi-Llo)*Q(i, 16) for i in range(17)] if Llo != Lhi else [Llo]
    worst = {}
    for L in pts:
        t = truth(L)
        for k, v in t.items():
            a, b = (mpq(x) for x in c["bounds"][k])
            if not (a <= v <= b):
                bad += 1
                print("  VIOLATION", lo, hi, kw, "L=", L, k, v, a, b)
            # relative margin
            w = b - a
            worst[k] = min(worst.get(k, mp.inf), min(v-a, b-v)/w if w else mp.inf)
    widths = {k: mp.nstr(mpq(c["bounds"][k][1]) - mpq(c["bounds"][k][0]), 3) for k in c["bounds"]}
    print(vr, lo, hi, kw, c["outcome"], "widths", widths, "minmargfrac", {k: mp.nstr(v, 3) for k, v in worst.items()})
print("VIOLATIONS", bad)
