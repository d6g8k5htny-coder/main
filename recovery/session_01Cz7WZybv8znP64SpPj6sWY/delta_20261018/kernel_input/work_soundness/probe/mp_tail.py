"""Check each certified tail_upper against the true omitted tail at many L in the interval,
and finite enclosure against the true finite sum (both independently with mpmath)."""
import sys, json
from fractions import Fraction as Q
sys.path.insert(0, sys.argv[1])
import mpmath as mp
from moment_kernel import certify, KernelHold
from verify_certificate import verify
mp.mp.dps = 60
def mpq(s):
    q = Q(s); return mp.mpf(q.numerator)/q.denominator
def parts(L, M):
    L = mp.mpf(L.numerator)/L.denominator
    h = 2*mp.pi/L
    N = int(40/h) + M + 5
    fin = [mp.mpf(1), 0, 0, 0]; tail = [mp.mpf(0)]*4
    for m in range(1, N+1):
        w = mp.e**(-(h*m)**2/2); q2 = (h*m)**2
        for k in range(4):
            (fin if m <= M else tail)[k] += 2*q2**k*w
    return fin, tail
cases = [("164/1","164/1",64), ("163/1","164/1",64), ("41/1","41/1",16), ("40/1","41/1",16),
         ("10/1","10/1",4), ("2/1","2/1",1), ("5/2","5/2",1), ("14/5","14/5",1), ("1/1","20/1",8),
         ("3/1","3/1",1), ("2/1","3/1",1)]
for lo, hi, M in cases:
    try:
        c = certify(dict(L_lo=lo, L_hi=hi, cutoff=M, input_role="TEST_ONLY"))
    except KernelHold as e:
        print("HOLD", lo, hi, M, e.reason); continue
    verify(json.loads(json.dumps(c)))
    al = mpq(c["alpha"][0])
    print("case", lo, hi, M, "alpha_lo*M^2=", mp.nstr(al*M*M, 6))
    Llo, Lhi = Q(lo), Q(hi)
    pts = [Llo + (Lhi-Llo)*Q(i, 8) for i in range(9)] if Llo != Lhi else [Llo]
    for L in pts:
        fin, tail = parts(L, M)
        for k, key in enumerate(("0","2","4","6")):
            tu = mpq(c["tail_upper"][key]); fa, fb = (mpq(x) for x in c["finite"][key])
            ok_t = tail[k] <= tu; ok_f = fa <= fin[k] <= fb
            if not (ok_t and ok_f):
                print("  VIOLATION L=", L, key, "tail", tail[k], "<=", tu, ok_t, "fin", fa, fin[k], fb, ok_f)
        if L in (pts[0], pts[-1]):
            print("  L=", L, "tail/true ratio:", [mp.nstr(mpq(c["tail_upper"][kk])/t, 4) if t else "inf" for kk, t in zip(("0","2","4","6"), tail)])
