#!/usr/bin/env python3
"""Stdlib-only reviewer checks for C51 GLOBAL.md Lemmas 1-2, 5 and PROOF.md O13/O14.

Exact Fraction arithmetic on a tiny bivariate polynomial class; float only for
the root-count sanity sweep. Identities are checked as polynomial identities
by exact evaluation at many random rational points (Schwartz-Zippel)."""
from fractions import Fraction as Fr
import random, math, itertools

random.seed(20261001)

def rq(lo=-5, hi=5, den=7):
    return Fr(random.randint(lo * den, hi * den), random.randint(1, den))

# ---- tiny polynomial in (x, z): dict {(i, j): coeff} --------------------
def padd(*ps):
    out = {}
    for p in ps:
        for m, c in p.items():
            out[m] = out.get(m, 0) + c
    return {m: c for m, c in out.items() if c != 0}

def pmul(p, q):
    out = {}
    for (i, j), c in p.items():
        for (k, l), d in q.items():
            out[(i + k, j + l)] = out.get((i + k, j + l), 0) + c * d
    return {m: c for m, c in out.items() if c != 0}

def pscale(p, a):
    return {m: c * a for m, c in p.items() if c * a != 0}

def dx(p):
    return {(i - 1, j): c * i for (i, j), c in p.items() if i > 0}

def dz(p):
    return {(i, j - 1): c * j for (i, j), c in p.items() if j > 0}

def ev(p, x, z):
    return sum(c * x ** i * z ** j for (i, j), c in p.items())

X = {(1, 0): Fr(1)}
Z = {(0, 1): Fr(1)}
ONE = {(0, 0): Fr(1)}

def P_orig(k, s, a, beta, q):
    # P = 2kX^3 - 3kX/2 - k/2 + sZ^2/2 + (a/2)(X^2-1/4)Z + (beta/2)XZ^2 + (q/6)Z^3
    X2 = pmul(X, X); Z2 = pmul(Z, Z)
    return padd(pscale(pmul(X2, X), 2 * k), pscale(X, -Fr(3) * k / 2),
                pscale(ONE, -k / 2), pscale(Z2, s / 2),
                pscale(pmul(padd(X2, pscale(ONE, -Fr(1, 4))), Z), a / 2),
                pscale(pmul(X, Z2), beta / 2), pscale(pmul(Z2, Z), q / 6))

def H3_orig(k, a, beta, q):
    X2 = pmul(X, X); Z2 = pmul(Z, Z)
    return padd(pscale(pmul(X2, X), 2 * k), pscale(pmul(X2, Z), a / 2),
                pscale(pmul(X, Z2), beta / 2), pscale(pmul(Z2, Z), q / 6))

def derived(k, s, a, beta, q):
    Bs = beta - a * a / (12 * k)
    Ds = (q - a * beta / (4 * k) + a ** 3 / (72 * k * k)) / 2
    Delta = 12 * k * Ds ** 2 + Bs ** 3
    H = 12 * k * Ds ** 2 + Bs ** 3 - 4 * Bs * s * s
    Sigma = 12 * k * Ds ** 2 + (s - Bs) ** 2 * (Bs + 2 * s)
    return Bs, Ds, Delta, H, Sigma

fails = []
def check(name, ok):
    if not ok:
        fails.append(name)

N = 300
for _ in range(N):
    k = abs(rq()) + Fr(1, 3)
    s, a, beta, q = rq(), rq(), rq(), rq()
    Bs, Ds, Delta, H, Sigma = derived(k, s, a, beta, q)
    P = P_orig(k, s, a, beta, q)
    H3 = H3_orig(k, a, beta, q)
    # (S1) shear: P(X,Z) == 2ku^3 - 3ku/2 - k/2 + sZ^2/2 + (Bs/2)uZ^2 + (Ds/3)Z^3, u = X + aZ/(12k)
    for _ in range(3):
        x, z = rq(), rq()
        u = x + a * z / (12 * k)
        sheared = 2 * k * u ** 3 - 3 * k * u / 2 - k / 2 + s * z * z / 2 + Bs / 2 * u * z * z + Ds / 3 * z ** 3
        check("S1 shear P", ev(P, x, z) == sheared)
        check("S1 shear H3", ev(H3, x, z) == 2 * k * u ** 3 + Bs / 2 * u * z * z + Ds / 3 * z ** 3)
    # H_poly numerator
    U = 12 * k * beta - a * a
    V = 72 * k * k * q - 18 * k * a * beta + a ** 3
    check("H_poly", H == (V * V + U ** 3 - 576 * k * k * U * s * s) / (1728 * k ** 3))
    # Sheared-coordinate gradient/Hessian (shear is unimodular: det and inertia preserved)
    # L = s + Bs u + Ds Z (line), C = 6k(u^2 - 1/4) + (Bs/2)Z^2 (conic)
    for _ in range(3):
        u, z = rq(), rq()
        L = s + Bs * u + Ds * z
        C = 6 * k * (u * u - Fr(1, 4)) + Bs / 2 * z * z
        Puu, PuZ, PZZ = 12 * k * u, Bs * z, s + Bs * u + 2 * Ds * z
        det = Puu * PZZ - PuZ ** 2
        # (3a) det Hess = -3k(Bs+4su) + 24k u L - 2 Bs C  (ideal membership)
        check("3a det", det == -3 * k * (Bs + 4 * s * u) + 24 * k * u * L - 2 * Bs * C)
        # (3b) H Z^2 = 3k(Bs+4su)^2 + 12k L (L - 2s - 2Bs u) + 2(Bs^2 - 4s^2) C
        check("3b HZ2", H * z * z == 3 * k * (Bs + 4 * s * u) ** 2
              + 12 * k * L * (L - 2 * s - 2 * Bs * u) + 2 * (Bs * Bs - 4 * s * s) * C)
    # det of original-coordinate Hessian equals sheared det at matching points
    x, z = rq(), rq()
    u = x + a * z / (12 * k)
    Dxx, Dxz, Dzz = ev(dx(dx(P)), x, z), ev(dz(dx(P)), x, z), ev(dz(dz(P)), x, z)
    check("det invariance", Dxx * Dzz - Dxz ** 2 == 12 * k * u * (s + Bs * u + 2 * Ds * z) - (Bs * z) ** 2)
    # pins: critical, determinants 3k(Bs-2s), 3k(Bs+2s), axial entries -6k, 6k
    for xp, dexp, axial in ((Fr(-1, 2), 3 * k * (Bs - 2 * s), -6 * k), (Fr(1, 2), 3 * k * (Bs + 2 * s), 6 * k)):
        check("pin grad", ev(dx(P), xp, 0) == 0 and ev(dz(P), xp, 0) == 0)
        dd = ev(dx(dx(P)), xp, 0) * ev(dz(dz(P)), xp, 0) - ev(dz(dx(P)), xp, 0) ** 2
        check("pin det", dd == dexp)
        check("pin axial", ev(dx(dx(P)), xp, 0) == axial)
    check("pin values", ev(P, Fr(-1, 2), 0) == 0 and ev(P, Fr(1, 2), 0) == -k)
    # elimination quadratic; discriminant identity disc = 12 k Ds^2 H
    if Ds != 0:
        A2, A1, A0 = Delta, 2 * Bs * Bs * s, Bs * s * s - 3 * k * Ds * Ds
        for _ in range(3):
            u = rq()
            z = -(s + Bs * u) / Ds
            C = 6 * k * (u * u - Fr(1, 4)) + Bs / 2 * z * z
            check("elim", 2 * Ds * Ds * C == A2 * u * u + A1 * u + A0)
        check("disc", A1 * A1 - 4 * A2 * A0 == 12 * k * Ds * Ds * H)
    # Lemma 2: homogeneous critical direction with Z != 0, Bs != 0: u = -Ds Z/Bs
    if Bs != 0:
        zz = rq() or Fr(1)
        uu = -Ds * zz / Bs
        check("L2 elim", 6 * k * uu * uu + Bs / 2 * zz * zz == zz * zz * Delta / (2 * Bs * Bs))
    # Lemma 5 / (8) for the exact pinned cubic at scale r: grad[r^3 P(x/r)] - grad H3(x)
    #   = r^2 (-3k/2, -a/8) + (0, r s z): only C r^2 + C r |x| terms (no |x|^2, |x|^3 terms).
    r = Fr(1, random.randint(2, 50))
    for _ in range(2):
        x, z = rq(), rq()
        gx = r * r * ev(dx(P), x / r, z / r) - ev(dx(H3), x, z)
        gz = r * r * ev(dz(P), x / r, z / r) - ev(dz(H3), x, z)
        check("(8) cubic x", gx == -Fr(3) * k * r * r / 2)
        check("(8) cubic z", gz == -a * r * r / 8 + r * s * z)

print("exact identity checks:", N, "parameter draws;", "FAIL " + str(sorted(set(fails))) if fails else "all pass")

# ---- float sweep: Lemma 1 conclusions on random typed generic theta ------
def roots_float(k, s, a, beta, q):
    Bs, Ds, Delta, H, _ = [float(v) for v in derived(Fr(k), Fr(s), Fr(a), Fr(beta), Fr(q))]
    out = [(-0.5, 0.0), (0.5, 0.0)]
    A2, A1, A0 = Delta, 2 * Bs * Bs * s, Bs * s * s - 3 * k * Ds * Ds
    disc = A1 * A1 - 4 * A2 * A0
    if disc >= 0 and A2 != 0:
        for sg in (1, -1):
            u = (-A1 + sg * math.sqrt(disc)) / (2 * A2)
            z = -(s + Bs * u) / Ds
            out.append((u, z))
    return out, Bs, Ds, H

stats = {"draws": 0, "extra2": 0, "extra0": 0, "Hneg_extra": 0, "maxcount_ne1": 0, "degenerate": 0, "notcrit": 0}
for _ in range(20000):
    k = random.uniform(0.2, 3)
    a, beta, q = (random.uniform(-4, 4) for _ in range(3))
    Bs = beta - a * a / (12 * k)
    s = -abs(Bs) / 2 - random.uniform(0.01, 3)          # typed: s < -|Bs|/2
    rts, Bs, Ds, H = roots_float(k, s, a, beta, q)
    stats["draws"] += 1
    nextra = len(rts) - 2
    stats["extra2" if nextra == 2 else "extra0"] += 1
    if H < 0 and nextra:
        stats["Hneg_extra"] += 1
    nmax = 0
    for (u, z) in rts:
        gu = 6 * k * (u * u - 0.25) + Bs / 2 * z * z
        gz = z * (s + Bs * u + Ds * z)
        if abs(gu) + abs(gz) > 1e-7 * (1 + abs(u) + abs(z)) ** 3:
            stats["notcrit"] += 1
        Puu, PuZ, PZZ = 12 * k * u, Bs * z, s + Bs * u + 2 * Ds * z
        det = Puu * PZZ - PuZ * PuZ
        if abs(det) < 1e-9:
            stats["degenerate"] += 1
        if det > 0 and Puu < 0:
            nmax += 1
    if nmax != 1:
        stats["maxcount_ne1"] += 1
print("float sweep:", stats)

# ---- O13 / O14 integer and pointwise inequalities ------------------------
ok = True
for n in range(0, 5001):
    pos = max(n - 1, 0)
    ok &= 0 <= pos <= n * (n >= 2)
    ok &= n - (n > 0) == pos
    ok &= 2 * (n >= 2) <= n * (n >= 2)
for _ in range(20000):
    n = random.randint(0, 12)
    phis = [random.uniform(-1, 1) for _ in range(n)]
    S = sum(phis)
    lhs = abs(S - ((S / n) if n > 0 else 0.0))
    ok &= lhs <= 1.0 * max(n - 1, 0) + 1e-12
# TV identity: TV(Law(N|N>0), delta_1) = P(N>=2|N>0) <= E[N 1{N>=2}]/(2 P(N>0))
for _ in range(2000):
    w = [random.random() for _ in range(8)]
    tot = sum(w); p = [x / tot for x in w]
    PN0 = sum(p[1:]); P2 = sum(p[2:])
    cond = [p[n] / PN0 for n in range(1, 8)]
    tv = 0.5 * (abs(cond[0] - 1) + sum(cond[1:]))
    ok &= abs(tv - P2 / PN0) < 1e-12
    ok &= P2 / PN0 <= sum(n * p[n] for n in range(2, 8)) / (2 * PN0) + 1e-12
print("O13/O14/TV checks:", "all pass" if ok else "FAIL")
