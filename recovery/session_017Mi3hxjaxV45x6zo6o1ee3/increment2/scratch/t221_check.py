# Exact checks for Math-#221 T1-T3 (stdlib only; Fractions + a crude quadrature sanity check).
from fractions import Fraction as F
import math, random

# T1: for probability laws P_E, P_D and weights a=e/n, b=D/n (a+b=1),
# C/n = a P_E + b P_D, and sup_A |C/n(A) - P_E(A)| = b sup_A |P_D(A) - P_E(A)| <= b.
random.seed(1)
worst = 0
for _ in range(2000):
    m = random.randint(2, 6)
    pe = [random.random() for _ in range(m)]; s = sum(pe); pe = [x/s for x in pe]
    pd = [random.random() for _ in range(m)]; s = sum(pd); pd = [x/s for x in pd]
    b = random.random(); a = 1 - b
    pc = [a*x + b*y for x, y in zip(pe, pd)]
    # sup over sets = sum of positive parts
    tv = sum(max(0, x - y) for x, y in zip(pc, pe))
    worst = max(worst, tv / b)
print("T1: max TV/(D/n) over random cases =", round(worst, 12), "(<= 1, equality iff singular)")
# singular case attains 1
pe = [1, 0]; pd = [0, 1]; b = F(1, 7); pc = [(1-b)*x + b*y for x, y in zip(pe, pd)]
print("T1: singular case TV =", sum(max(0, x-y) for x, y in zip(pc, pe)), "= D/n =", b)

# T2 near range: for fixed ell, rho<r<r0 <=> ell/r0^3 < k < ell/rho^3, and
# (ell^(-1/3)/3) * int_0^(ell/rho^3) k^(-2/3) dk = 1/rho exactly.
for ell, rho in [(F(1, 1000), F(1, 5)), (F(1, 64), F(1, 2)), (F(3, 10**6), F(1, 50))]:
    K = ell / rho**3
    val = (float(ell) ** (-1/3) / 3) * 3 * float(K) ** (1/3)
    print("T2 k-integral: ell=%s rho=%s -> %.15g vs 1/rho=%.15g" % (ell, rho, val, 1/float(rho)))

# T2 independent route in (r,k): count <= int_rho^r0 r dr int_0^(t/r^3) h(k) dk with sup h <= Hb
# gives <= Hb t int_rho^inf r^(-2) dr = Hb t/rho.  Check against the (ell,k) route numerically
# for h(k) = exp(-k^2) (any bounded integrable profile), t <= rho^3.
def route_rk(t, rho, r0, n=4000):
    s = 0.0
    for i in range(n):
        r = rho + (r0 - rho) * (i + 0.5) / n
        K = t / r**3
        s += r * (math.sqrt(math.pi) / 2 * math.erf(K)) * (r0 - rho) / n
    return s
def route_lk(t, rho, r0, n=4000, m=4000):
    s = 0.0
    for i in range(n):
        ell = t * (i + 0.5) / n
        lo, hi = ell / r0**3, ell / rho**3
        # substitute k = w^3 to remove the k^(-2/3) singularity: dk k^(-2/3) = 3 dw
        wl, wh = lo ** (1/3), hi ** (1/3)
        inner = sum(3 * math.exp(-((wl + (wh - wl) * (j + 0.5) / m) ** 3) ** 2) for j in range(m)) * (wh - wl) / m
        s += ell ** (-1/3) / 3 * inner * t / n
    return s
for t, rho, r0 in [(1e-4, 0.1, 0.5), (1e-6, 0.02, 0.3)]:
    a, b = route_rk(t, rho, r0), route_lk(t, rho, r0, 800, 800)
    print("T2 two routes: t=%g rho=%g: (r,k) %.8g  (ell,k) %.8g  bound t/rho %.8g" % (t, rho, a, b, t/rho))

# T2 without t <= rho^3: total actual mass <= C t^(2/3) and t/rho >= t^(2/3) iff t >= rho^3.
for t, rho in [(F(1, 10**3), F(1, 20)), (F(1, 10**6), F(1, 1000))]:
    print("t > rho^3:", t > rho**3, " t/rho >= t^(2/3):", float(t/rho) >= float(t) ** (2/3))
