import sys, random
from fractions import Fraction as Q
sys.path.insert(0, sys.argv[1])
import mpmath as mp
from moment_kernel import Arithmetic, pi_witness, exp_negative, KernelHold
mp.mp.dps = 120
def mpq(s): q = Q(s); return mp.mpf(q.numerator)/q.denominator
bad = 0
for bits in (16, 17, 40, 80, 128):
    for n in range(1, 33):
        rec, p = pi_witness(n, Arithmetic(bits))
        if not (p[0] <= Q(int(mp.floor(mp.pi*2**400)), 2**400) and Q(int(mp.ceil(mp.pi*2**400)), 2**400) <= p[1]):
            bad += 1; print("PI VIOL", bits, n, p)
rng = random.Random(7)
xs = [Q(0), Q(1), Q(2), Q(65536), Q(1, 2**200), Q(65535, 1)] + [Q(rng.randint(0, 10**8), rng.randint(1, 10**4)) for _ in range(200)]
for x in xs:
    for bits, pairs in ((16, 1), (40, 3), (80, 16), (128, 24)):
        try:
            w = exp_negative(x, bits, pairs)
        except KernelHold as e:
            continue
        lo, hi = (mpq(v) for v in w["result"])
        tv = mp.e**(-mpq(f"{x.numerator}/{x.denominator}"))
        if not lo <= tv <= hi:
            bad += 1; print("EXP VIOL", x, bits, pairs, lo, tv, hi)
print("violations", bad)
