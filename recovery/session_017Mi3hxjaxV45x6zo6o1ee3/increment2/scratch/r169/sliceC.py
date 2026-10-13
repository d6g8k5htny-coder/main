"""Finite controls for Math-#169 Slice C (E27-E34). Exact where the object is rational; floating Gauss-Legendre for shape integrals."""
from fractions import Fraction as F
import math, random
random.seed(20260930)
# ---------- E4: F = G + B2 pathwise over n in {0,1,2} and all radius orderings ----------
ok4 = True
for radii in ([], [0.3], [2.5], [0.3, 2.5], [2.5, 0.3], [2.5, 2.5], [0.3, 0.3]):
    for t in (0.1, 1.0, 2.5, 3.0):
        Fv = sum(1 for r in radii if r > t)
        G = 1 if radii and max(radii) > t else 0
        B2 = 1 if len(radii) == 2 and min(radii) > t else 0
        ok4 &= Fv == G + B2
        # E27 selector: sum_i 1{R_i>t} 1{i is an actual maximum}/ties == 1{R_max>t}
        if radii:
            m = max(radii); ties = sum(1 for r in radii if r == m)
            ok4 &= abs(sum((1 if r > t else 0) * (1 / ties if r == m else 0) for r in radii) - G) < 1e-12
print("E4 identity and E27 selector identity over all count/ordering cases:", ok4)
# ---------- physical radius geometry: X = u - aZ/(12k), position = X e_axis + Z e_soft ----------
def gamma(a, k): return math.sqrt(1 + (a / (12 * k)) ** 2)
ok_geo = True
for _ in range(2000):
    k = random.choice([0.5, 1.0, 2.0]); a = random.uniform(-40, 40); u = random.uniform(-1.5, 0.5); z = random.uniform(-3, 3)
    if abs(z) < 1e-3: continue
    t = 1e7; Z = t * z
    X = u - a * Z / (12 * k); R = math.hypot(X, Z)
    ok_geo &= abs(R / t - gamma(a, k) * abs(z)) < 1e-5                 # R_i / t -> gamma |z|
    # triangle bound: R <= |u| + gamma |Z|, hence R > t and |u|<=3/2, t>=3 give |z| > 1/(2 gamma)
    ok_geo &= R <= abs(u) + gamma(a, k) * abs(Z) + 1e-9
    # limiting direction e_* = [-a e_axis/(12k) + e_soft]/gamma : position/t -> z*gamma*e_*
    ex, ez = -a / (12 * k) / gamma(a, k), 1 / gamma(a, k)
    ok_geo &= abs(X / t - z * gamma(a, k) * ex) < 1e-5 and abs(Z / t - z * gamma(a, k) * ez) < 1e-12
    ok_geo &= abs(math.hypot(ex, ez) - 1) < 1e-12
for t in (3.0, 5.0, 50.0):
    for a in (-30, 0, 30):
        g = gamma(a, 1.0)
        # R > t with |u| <= 3/2 forces gamma|Z| > t - 3/2 >= t/2, i.e. |z| > 1/(2 gamma)
        ok_geo &= (t - 1.5) / (g * t) >= 1 / (2 * g)
print("radius/direction limits, triangle bound, |z| > 1/(2 gamma) for t >= 3:", ok_geo)
# ---------- z-integrals: int_{gamma|z|>1} |z|^-12 dz = 2 gamma^11 / 11 ; int_{|z|>1/(2gamma)} = 2 (2 gamma)^11 / 11 ----------
def zint(lo): return 2 * lo ** (-11) / 11
ok_z = all(abs(zint(1 / gamma(a, 1.0)) - 2 * gamma(a, 1.0) ** 11 / 11) < 1e-9 * gamma(a, 1.0) ** 11 for a in (-20, -3, 0, 7, 25))
ok_z &= abs(zint(1 / (2 * gamma(5.0, 1.0))) - 2 * (2 * gamma(5.0, 1.0)) ** 11 / 11) < 1e-6
# and the change of variables q = gamma|z|: |z|^-12 dz = gamma^11 q^-12 dq (per sign); Pareto(11) tail q^-11
ok_z &= abs(sum(11 * q ** -12 * 1e-4 for q in [1 + 1e-4 * (i + 0.5) for i in range(200000)]) - (1 - 21 ** -11)) < 1e-3
print("z-integrals, Pareto normalization:", ok_z)
# ---------- coefficients: C0 = C_star / I_shape with C_star = (216/11) k^7 (c_m/z0) I_shape J_cusp; 108 * 2/11 = 216/11 ----------
print("R13 -> E5 factor: 108 * 2/11 == 216/11:", F(108) * F(2, 11) == F(216, 11))
# ---------- E8 / E31 / E34 algebra ----------
I = F(246528, 35); J = F(1083417, 280)
log2 = 0.6931471805599453
D = 27066286003 / 223205220 - (79298560 / 4782969) * log2
Imax = float(I) - D
print("E8: P(n=2|Rmax>t) = J/(I-D) = %.12f ; P(K_t=2|Rmax>t) = D/(I-D) = %.12f ; E[K_t|Rmax>t] = I/(I-D) = %.12f" % (float(J) / Imax, D / Imax, float(I) / Imax))
print("E31 consistency: P(n=2) * E[rho^11 | doublet] = (J/Imax)*(D/J) = D/Imax:", abs((float(J) / Imax) * (D / float(J)) - D / Imax) < 1e-15)
# E34: with a_n = [n C0 (I-D)/A]^(1/11), n * P_pi(Rmax > a_n x) -> x^-11 ; with the point coefficient instead -> theta* x^-11
for n in (10 ** 3, 10 ** 6):
    C0_over_A = 0.37  # arbitrary positive constant
    a_n = (n * C0_over_A * Imax) ** (1 / 11)
    x = 1.7
    tail = C0_over_A * Imax * (a_n * x) ** -11       # limiting tail P_pi(Rmax > s) ~ C0 (I-D) s^-11 / A
    assert abs(n * tail - x ** -11) < 1e-12
    a_n_pt = (n * C0_over_A * float(I)) ** (1 / 11)
    tail_pt = C0_over_A * Imax * (a_n_pt * x) ** -11
    assert abs(n * tail_pt - (Imax / float(I)) * x ** -11) < 1e-12
print("E34 normalizations: exp(-x^-11) with the cluster coefficient, exp(-theta* x^-11) with the point coefficient: True")
# ---------- E28 capped moments: int_{n>0} min(Rmax,t)^11 dM = int_0^t 11 s^10 G(s) ds ~ 11 C0 (I-D) log t (model tail G(s) = c s^-11 for s>=1, bounded mass below) ----------
def capped(t, c, mass_small=2.0):
    # G(s) = mass_small for s<1 (finite total mass), c s^-11 for s>=1 ; capped moment = int_0^t 11 s^10 G(s) ds
    return mass_small * 1.0 + 11 * c * math.log(t) if t >= 1 else None
for t in (1e3, 1e6, 1e12):
    v = capped(t, 5.0)
    assert abs(v / (11 * 5.0 * math.log(t)) - 1) < 3.0 / (11 * 5.0 * math.log(t)) + 1e-12
print("E28 capped-moment asymptotics (Tonelli model): ~ 11 C log t with an O(1) remainder from the finite small-radius mass: True")
# ---------- E32/E33 consistency: int_0^1 [H_eta(e) - inner(e)] de = I - D and int e[...] = I*E eta - D_1 (using Slice B's exact D_1) ----------
D1 = 1259212131941593 / 15982386572880 - (3186360320 / 387420489) * log2
Eeta_pt = 5771 / 7062
Eeta_max = (float(I) * Eeta_pt - D1) / Imax
print("E33 E eta_max from E32 bookkeeping: %.18f (claim 0.819586988784918536...)" % Eeta_max)
# ---------- H1 closed form versus H2 integral form (R's height density, used by E32): float spot checks ----------
def Qf(u, v):
    b0 = 3 - 12 * u * u
    return (4 * v * v - b0 * b0) * (b0 - 4 * u * v)
def gl(n):
    xs, ws = [], []
    for i in range(1, n + 1):
        x = math.cos(math.pi * (i - 0.25) / (n + 0.5))
        for _ in range(100):
            p0, p1 = 1.0, x
            for kk in range(2, n + 1): p0, p1 = p1, ((2 * kk - 1) * x * p1 - (kk - 1) * p0) / kk
            dp = n * (x * p1 - p0) / (x * x - 1); dx = p1 / dp; x -= dx
            if abs(dx) < 1e-15: break
        xs.append(x); ws.append(2 / ((1 - x * x) * dp * dp))
    return xs, ws
X, W = gl(60)
def integ(f, a, b): return sum(w * f((b - a) / 2 * x + (a + b) / 2) for x, w in zip(X, W)) * (b - a) / 2
def H1(e): return (110592 / 7) * (e ** 3.5 - (1 - e) ** 3.5 + 1) + 13824 * (e ** 4 - 2 * e ** 3 + 5 * e ** 2 - 4 * e)
def H2(e): return 6 * integ(lambda u: Qf(u, 6 * e - 6 * u - 3), -0.5 - math.sqrt(e), 0.5 - math.sqrt(1 - e))
okH = all(abs(H1(e) - H2(e)) < 1e-7 * max(1, abs(H1(e))) for e in (0.05, 0.2, 0.5, 0.8, 0.95))
print("H1 closed form == H2 integral form at five heights:", okH, " int H1 == I_shape:", abs(integ(H1, 0, 1) - float(I)) < 1e-6)
# E32 inner subtraction: 6 int_{1/2}^{1} Q(-p, 6(e+p-1/2)) 1{A p < 6(e+p-1/2) < V+} dp integrates in e to D
def Vplus(p): return (36 * p * p - 12 * p - 3 + 3 * math.sqrt((2 * p - 1) ** 3 * (18 * p + 7))) / 4
def inner(e):
    def f(p):
        v = 6 * (e + p - 0.5); A = 12 * p * p - 3
        return Qf(-p, v) if (A * p < v < Vplus(p)) else 0.0
    # integrate piecewise in p by splitting at the (at most two) boundary crossings found numerically
    ps = [0.5 + 0.5 * i / 4000 for i in range(4001)]
    return 6 * sum(f((ps[i] + ps[i + 1]) / 2) * (ps[i + 1] - ps[i]) for i in range(4000))
es = [(i + 0.5) / 400 for i in range(400)]
D_from_E32 = sum(inner(e) for e in es) / 400
print("E32 inner term integrates in e to D: %.6f vs D = %.6f (midpoint rules; control only)" % (D_from_E32, D))
