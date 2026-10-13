"""Exact check of the #181 comment 5908646514 (sharp Theta(1/T) scalar rate). stdlib only."""
from fractions import Fraction as F
from math import comb, log
# h(z) = z^7 (z+1)^4 ; H0 = 6401/3960 ; a(rho,z) = (rho z+1)^4 (rho z+rho+1)^7 / [(rho+1)^5 (2 rho z+rho+1)^9]
H0 = sum(F(comb(4, j), 8 + j) for j in range(5))
print("H0 =", H0, H0 == F(6401, 3960))
I5 = sum(F(comb(5, j), 8 + j) for j in range(6))            # int_0^1 z^7 (z+1)^5 dz
print("int z^7(z+1)^5 =", I5, I5 == F(31745, 10296))
mu = I5 / H0; print("mu = E[Z+1] =", mu, mu == F(158725, 83213), float(mu))
m = F(13, 14) * mu; print("m = (13/14) mu =", m, float(m))
# d/drho log a at rho = 0: 4z + 7(z+1) - 5 - 9(2z+1) = -7(z+1)  (check as polynomial in z)
coef = {1: 4 + 7 - 18, 0: 7 - 5 - 9}
print("d_rho log a(0,z) coefficients {z:%d, 1:%d} == -7(z+1):" % (coef[1], coef[0]), coef == {1: -7, 0: -7})
# second-order Taylor check of a(rho,z) = 1 - 7(z+1) rho + O(rho^2) numerically at a few z
def a(rho, z): return (rho*z + 1)**4 * (rho*z + rho + 1)**7 / ((rho + 1)**5 * (2*rho*z + rho + 1)**9)
ok = all(abs((a(r, z) - 1 + 7*(z + 1)*r) / r**2) < 200 for z in (0.0, 0.3, 0.7, 1.0) for r in (1e-2, 1e-3, 1e-4))
print("a(rho,z) = 1 - 7(z+1) rho + O(rho^2) uniformly (bounded second-order ratio):", ok)
# E[Y] for density 13 y^12 on (0,1) is 13/14 ; first-order correction integrates to zero: E[m - Y(Z+1)] = m - (13/14) mu = 0
print("E[Y] = 13/14 and m - E[Y]E[Z+1] = 0:", F(13, 14) * mu - m == 0)
# tail: P(V>T) = (165888/D) int_{rho<1/T} rho^12 h a = (165888 H0/(13 D)) T^-13 [1 - 7 m / T + O(T^-2)]
# coefficient check: int_0^{1/T} rho^12 (1 - 7(z+1) rho) drho = T^-13/13 - 7(z+1) T^-14/14 ; times h and integrate: H0 T^-13/13 [1 - (13/14) 7 mu / T]
print("tail correction 7 m = 7 (13/14) mu:", 7 * m == F(13, 2) * mu)
# K = (7/2) E|Y(Z+1) - m| exactly: E|W - m| with W = Y(Z+1), Y ~ 13 y^12, Z ~ h/H0.
# E|W-m| = E[W] - m + 2 E[(m - W)_+] = 2 E[(m-W)_+] since E W = m.  E[(m-W)_+] = int_z h(z)/H0 int_0^{min(1, m/(z+1))} (m - y(z+1)) 13 y^12 dy dz
# inner integral with upper limit c: 13 [ m c^13/13 - (z+1) c^14/14 ] = m c^13 - (13/14)(z+1) c^14
# case c = 1 (z+1 <= m): m - (13/14)(z+1) ; case c = m/(z+1) (z+1 > m): m^14/(z+1)^13 - (13/14) m^14/(z+1)^13 = m^14/(14 (z+1)^13)
# with w = z+1 in (1,2), h = (w-1)^7 w^4 ; split at w = m (m < 2 since m = (13/14) mu and mu < 2)
assert 1 < m < 2
# part 1: int_1^m (w-1)^7 w^4 [m - (13/14) w] dw  (polynomial)  ; part 2: int_m^2 (w-1)^7 w^4 * m^14 / (14 w^13) dw = (m^14/14) int_m^2 (w-1)^7 w^-9 dw
def poly_mul(a, b):
    d = {}
    for i, c1 in a.items():
        for j, c2 in b.items(): d[i + j] = d.get(i + j, 0) + c1 * c2
    return d
wm1_7 = {0: F(1)}
for _ in range(7): wm1_7 = poly_mul(wm1_7, {1: F(1), 0: F(-1)})
w4 = {4: F(1)}
def integ_poly(p, lo, hi): return sum(c * (hi ** (k + 1) - lo ** (k + 1)) / (k + 1) for k, c in p.items())
p1 = poly_mul(poly_mul(wm1_7, w4), {0: m, 1: -F(13, 14)})
part1 = integ_poly(p1, F(1), m)
# part 2: (w-1)^7 w^-9 = sum_k c_k w^(k-9), k=0..7 ; integrate: k-9 != -1 always (k=8 not present) -> no log term
lau = {k - 9: c for k, c in wm1_7.items()}
part2 = (m ** 14 / 14) * sum(c * (F(2) ** (k + 1) - m ** (k + 1)) / (k + 1) for k, c in lau.items())
Eplus = (part1 + part2) / H0
K = F(7, 2) * 2 * Eplus
print("E[(m-W)_+] =", Eplus, " (rational, no log term because w^-1 never appears)")
print("K = (7/2) E|W - m| =", K, "≈", float(K))
# floating Monte-Carlo-free cross-check by 2D midpoint quadrature
import math
n = 2000; tot = 0.0
for i in range(n):
    y = (i + 0.5) / n
    for j in range(n):
        z = (j + 0.5) / n
        tot += abs(y * (z + 1) - float(m)) * 13 * y ** 12 * (z ** 7 * (z + 1) ** 4) / float(H0)
print("float E|W-m| ≈", tot / n / n, " exact", float(2 * Eplus))
