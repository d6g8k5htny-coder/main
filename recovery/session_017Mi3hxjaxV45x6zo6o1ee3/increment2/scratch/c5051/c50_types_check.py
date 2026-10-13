#!/usr/bin/env python3
"""Types of the extra critical points of the typed pinned cubic, by the signs of Delta and H.

P = 2k u^3 - 3k u/2 - k/2 + s Z^2/2 + (B/2) u Z^2 + (D/3) Z^3   (sheared; unimodular, so det/inertia preserved)
Extras: conic C = 6k(u^2-1/4) + (B/2) Z^2 = 0, line L = s + B u + D Z = 0, Z != 0.
Elimination (D != 0): A2 u^2 + A1 u + A0 = 0, A2 = Delta = 12kD^2 + B^3, A1 = 2B^2 s, A0 = B s^2 - 3kD^2.
det Hess at an extra = -3k(B + 4 s u).
Claims (exact Vieta identities, D != 0, Delta != 0):
  (V1) A1^2 - 4 A2 A0 = 12 k D^2 H,                H = Delta - 4 B s^2
  (V2) det1 * det2 = 9 k^2 (B^2 - 4 s^2) H / Delta
  (V3) det1 + det2 = -6 k B H / Delta
Under strict typing s < -|B|/2 (so B^2 - 4s^2 < 0):
  H < 0            -> no extra critical point (and then Delta > 0)
  H > 0, Delta > 0 -> one extra saddle and one extra local minimum
  Delta < 0        -> H > 0 and two extra saddles
"""
from fractions import Fraction as Fr
import random, math
random.seed(11)

def rq(n=40, d=9):
    return Fr(random.randint(-n, n), random.randint(1, d))

ok = True
for _ in range(2000):
    k = Fr(random.randint(1, 40), random.randint(1, 9))
    B = rq(); D = rq() or Fr(1, 3)
    s = -abs(B) / 2 - Fr(random.randint(1, 60), random.randint(1, 9))
    Delta = 12 * k * D * D + B ** 3
    if Delta == 0:
        continue
    H = Delta - 4 * B * s * s
    A2, A1, A0 = Delta, 2 * B * B * s, B * s * s - 3 * k * D * D
    S, Pr = -A1 / A2, A0 / A2                        # u1 + u2, u1 * u2
    prod = 9 * k * k * (B * B + 4 * s * B * S + 16 * s * s * Pr)   # det1*det2
    summ = -3 * k * (2 * B + 4 * s * S)                             # det1+det2
    ok &= (A1 * A1 - 4 * A2 * A0 == 12 * k * D * D * H)
    ok &= (prod == 9 * k * k * (B * B - 4 * s * s) * H / Delta)
    ok &= (summ == -6 * k * B * H / Delta)
    # sign consequences
    if Delta < 0:
        ok &= (H > 0) and (B < 0)
print("exact Vieta identities (2000 rational draws):", "pass" if ok else "FAIL")

# D = 0 branch: u = -s/B, Z^2 = -12k(u^2-1/4)/B, det = -3k(B^2-4s^2)/B ; Delta = B^3
# B = 0 branch: (u,Z) = (+-1/2, -s/D), det = -/+ 6ks ; Delta = 12kD^2 > 0, H = 12kD^2 > 0
ok2 = True
for _ in range(2000):
    k = Fr(random.randint(1, 40), 7); B = rq() or Fr(-1); s = -abs(B) / 2 - Fr(random.randint(1, 60), 9)
    u = -s / B; Z2 = -12 * k * (u * u - Fr(1, 4)) / B; det = -3 * k * (B + 4 * s * u)
    ok2 &= (det == -3 * k * (B * B - 4 * s * s) / B)
    if Z2 > 0:                       # extras exist iff B < 0 iff Delta=B^3 < 0
        ok2 &= (B < 0) and (det < 0)
    else:
        ok2 &= (B > 0)
    D = rq() or Fr(1)
    ok2 &= (-3 * k * (4 * s * Fr(1, 2)) > 0) and (-3 * k * (4 * s * Fr(-1, 2)) < 0)
print("degenerate branches D=0, B=0:", "pass" if ok2 else "FAIL")

# float sweep in the ORIGINAL parameters theta=(s,a,beta,q): classify every extra root
cnt = {}
for _ in range(200000):
    k = random.uniform(0.2, 3); a, beta, q = (random.uniform(-5, 5) for _ in range(3))
    B = beta - a * a / (12 * k); D = (q - a * beta / (4 * k) + a ** 3 / (72 * k * k)) / 2
    s = -abs(B) / 2 - random.expovariate(1.0)
    Delta = 12 * k * D * D + B ** 3; H = Delta - 4 * B * s * s
    A2, A1, A0 = Delta, 2 * B * B * s, B * s * s - 3 * k * D * D
    disc = A1 * A1 - 4 * A2 * A0
    types = []
    if disc > 0:
        for sg in (1, -1):
            u = (-A1 + sg * math.sqrt(disc)) / (2 * A2); Z = -(s + B * u) / D
            Puu, PuZ, PZZ = 12 * k * u, B * Z, s + B * u + 2 * D * Z
            det = Puu * PZZ - PuZ * PuZ
            types.append("saddle" if det < 0 else ("min" if Puu > 0 else "MAX"))
    key = ("Delta<0" if Delta < 0 else "Delta>0", "H<0" if H < 0 else "H>0", tuple(sorted(types)))
    cnt[key] = cnt.get(key, 0) + 1
for kk in sorted(cnt):
    print(" ", kk, cnt[kk])
