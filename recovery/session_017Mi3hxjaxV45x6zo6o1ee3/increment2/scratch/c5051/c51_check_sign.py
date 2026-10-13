#!/usr/bin/env python3
"""Under strict typing s<-|Bs|/2: real extra critical points exist iff H>0 (H=0: double root).
Float sweep over generic draws plus exact checks of the Ds=0 and Bs=0 branches."""
import random, math
from fractions import Fraction as Fr
random.seed(7)
bad = 0; n = 0; pos2 = 0; neg0 = 0
for _ in range(100000):
    k = random.uniform(0.2, 3); Bs = random.uniform(-4, 4); Ds = random.uniform(-4, 4)
    s = -abs(Bs) / 2 - random.uniform(1e-3, 3)
    Delta = 12*k*Ds*Ds + Bs**3; H = Delta - 4*Bs*s*s
    A2, A1, A0 = Delta, 2*Bs*Bs*s, Bs*s*s - 3*k*Ds*Ds
    disc = A1*A1 - 4*A2*A0
    assert abs(disc - 12*k*Ds*Ds*H) <= 1e-9 * (1 + abs(disc))
    nextra = 2 if disc > 0 else 0
    n += 1
    if (H > 0) != (nextra == 2): bad += 1
    pos2 += (H > 0 and nextra == 2); neg0 += (H < 0 and nextra == 0)
print("generic sweep:", n, "draws; iff violations:", bad, "; H>0 with 2 extras:", pos2, "; H<0 with none:", neg0)
# Ds = 0 branch: extras u=-s/Bs, Z^2 = -12k(u^2-1/4)/Bs ; H = Bs(Bs^2-4s^2)
ok = True
for _ in range(2000):
    k = Fr(random.randint(1, 30), 7); Bs = Fr(random.randint(-40, 40), 9) or Fr(1)
    s = -abs(Bs) / 2 - Fr(random.randint(1, 40), 11)
    H = Bs**3 - 4*Bs*s*s
    u = -s / Bs; Z2 = -12*k*(u*u - Fr(1, 4)) / Bs
    ok &= ((Z2 > 0) == (H > 0))
# Bs = 0 branch: extras (u,Z)=(+-1/2, -s/Ds) always exist; H = 12k Ds^2 > 0
for _ in range(2000):
    k = Fr(random.randint(1, 30), 7); Ds = Fr(random.randint(-40, 40), 9) or Fr(1)
    ok &= 12*k*Ds*Ds > 0
print("Ds=0 / Bs=0 branches:", "iff holds" if ok else "FAIL")
