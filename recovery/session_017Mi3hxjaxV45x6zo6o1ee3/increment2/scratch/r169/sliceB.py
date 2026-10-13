"""Independent exact reconstruction of Math-#169 PROOF.md Slice B (E20-E26, E33). stdlib only."""
from fractions import Fraction as F
import math, itertools

# ---------- Laurent polynomials in one variable over Q: dict exp -> coeff ----------
class L:
    def __init__(self, d=None):
        self.d = {k: v for k, v in (d or {}).items() if v != 0}
    @staticmethod
    def c(x): return L({0: F(x)})
    @staticmethod
    def y(n=1): return L({n: F(1)})
    def __add__(s, o):
        o = o if isinstance(o, L) else L.c(o); d = dict(s.d)
        for k, v in o.d.items(): d[k] = d.get(k, 0) + v
        return L(d)
    __radd__ = __add__
    def __neg__(s): return L({k: -v for k, v in s.d.items()})
    def __sub__(s, o): return s + (-(o if isinstance(o, L) else L.c(o)))
    def __rsub__(s, o): return (-s) + o
    def __mul__(s, o):
        o = o if isinstance(o, L) else L.c(o); d = {}
        for k1, v1 in s.d.items():
            for k2, v2 in o.d.items(): d[k1 + k2] = d.get(k1 + k2, 0) + v1 * v2
        return L(d)
    __rmul__ = __mul__
    def __pow__(s, n):
        r = L.c(1)
        for _ in range(n): r = r * s
        return r
    def __eq__(s, o): return s.d == (o if isinstance(o, L) else L.c(o)).d
    def integrate_quarter_to_one(s):
        """exact integral over y in [1/4, 1]; returns (rational, coefficient of log 2)"""
        rat, lg = F(0), F(0)
        for j, cj in s.d.items():
            if j == -1: lg += 2 * cj                      # int_{1/4}^1 dy/y = log 4 = 2 log 2
            else: rat += cj * (1 - F(4) ** (-(j + 1))) / (j + 1)
        return rat, lg

# ---------- bivariate polynomials in (p, v) as dict (i,j)->coeff, for exact strip integrals ----------
def pmul(a, b):
    d = {}
    for (i1, j1), c1 in a.items():
        for (i2, j2), c2 in b.items(): d[(i1 + i2, j1 + j2)] = d.get((i1 + i2, j1 + j2), 0) + c1 * c2
    return {k: v for k, v in d.items() if v}
def padd(*ps):
    d = {}
    for p in ps:
        for k, v in p.items(): d[k] = d.get(k, 0) + v
    return {k: v for k, v in d.items() if v}
def pscale(a, c): return {k: v * c for k, v in a.items()}
def ppow(a, n):
    r = {(0, 0): F(1)}
    for _ in range(n): r = pmul(r, a)
    return r
P_ = {(1, 0): F(1)}; V_ = {(0, 1): F(1)}; ONE = {(0, 0): F(1)}
def subst_v(poly, vexpr):
    """substitute v := vexpr (a polynomial in p only, given as dict over (i,0)) into poly(p,v) -> polynomial in p"""
    out = {}
    for (i, j), c in poly.items():
        term = pmul({(i, 0): c}, ppow(vexpr, j))
        out = padd(out, term)
    return out
def int_p(poly_p, a, b):
    """exact integral over p in [a,b] of a polynomial in p alone"""
    tot = F(0)
    for (i, j), c in poly_p.items():
        assert j == 0
        tot += c * (F(b) ** (i + 1) - F(a) ** (i + 1)) / (i + 1)
    return tot
def int_v_primitive(poly):
    """primitive in v"""
    return {(i, j + 1): c / (j + 1) for (i, j), c in poly.items()}

# ---------- the shape density Q at u = -p (E11 with u=-p): b0(u) = 3-12u^2 = -(12p^2-3) = -A ----------
A_p = {(2, 0): F(12), (0, 0): F(-3)}                 # A = 12 p^2 - 3
# Q(u,v) = (4v^2 - b0^2)(b0 - 4uv); at u=-p: (4v^2 - A^2)(-A + 4pv)
Q_pv = pmul(padd(pscale(ppow(V_, 2), 4), pscale(ppow(A_p, 2), -1)), padd(pscale(A_p, -1), pscale(pmul(P_, V_), 4)))
Q_prim = int_v_primitive(Q_pv)                       # my own primitive in v (no constant)
# E20's P0
P0 = padd(pscale(pmul(P_, ppow(V_, 4)), 4), pscale(pmul(A_p, ppow(V_, 3)), F(-4, 3)),
          pscale(pmul(pmul(P_, ppow(A_p, 2)), ppow(V_, 2)), -2), pmul(ppow(A_p, 3), V_))
print("E20 primitive P0 equals d/dv-primitive of Q:", padd(P0, pscale(Q_prim, -1)) == {})

# ---------- I = int_S Q over S = {-3/2<u<1/2, |b0|/2 < v < 3-6u}, in the variable p=-u (p from -1/2 to 3/2) ----------
# u in (-1/2, 1/2) <-> p in (-1/2, 1/2): b0 = -A >= 0, lower = (3-12p^2)/2 = -A/2 ; upper = 3 - 6u = 3 + 6p
# u in (-3/2, -1/2) <-> p in (1/2, 3/2): b0 < 0, lower = A/2 ; upper = 3 + 6p
UP = {(0, 0): F(3), (1, 0): F(6)}
I_1 = int_p(padd(subst_v(Q_prim, UP), pscale(subst_v(Q_prim, pscale(A_p, F(-1, 2))), -1)), F(-1, 2), F(1, 2))
I_2 = int_p(padd(subst_v(Q_prim, UP), pscale(subst_v(Q_prim, pscale(A_p, F(1, 2))), -1)), F(1, 2), F(3, 2))
I = I_1 + I_2
print("I =", I, "== 246528/35:", I == F(246528, 35))

# ---------- J: the two outer strips (E14): p in (1/2,1): A/2 < v < A p ; p in [1,3/2): A/2 < v < 3+6p ----------
J1 = int_p(padd(subst_v(Q_prim, pmul(A_p, P_)), pscale(subst_v(Q_prim, pscale(A_p, F(1, 2))), -1)), F(1, 2), 1)
J2 = int_p(padd(subst_v(Q_prim, UP), pscale(subst_v(Q_prim, pscale(A_p, F(1, 2))), -1)), 1, F(3, 2))
print("J strips:", J1, "== 596217/1120:", J1 == F(596217, 1120), "|", J2, "== 3737451/1120:", J2 == F(3737451, 1120))
J = J1 + J2
print("J =", J, "== 1083417/280:", J == F(1083417, 280))

# ---------- the substitution (E22): q0 = sqrt((18p+7)/(2p-1)), y = (q0-3)/(q0+3) ----------
y = L.y()
p_y = (4 * y ** 2 + y + 4) * L({-1: F(1, 18)})               # (4y^2+y+4)/(18y)
A_y = 12 * p_y ** 2 - 3
two_p_minus_1 = 2 * p_y - 1
print("2p-1 == 4(1-y)^2/(9y):", two_p_minus_1 == (4 * (1 - y) ** 2) * L({-1: F(1, 9)}))
# q0 = 3(1+y)/(1-y)  (inverse of y=(q0-3)/(q0+3)); check q0^2 (2p-1) == 18p+7 as Laurent identity after clearing (1-y)^2:
# (18p+7)(1-y)^2 == 9(1+y)^2 (2p-1)
print("q0 relation:", (18 * p_y + 7) * (1 - y) ** 2 == 9 * (1 + y) ** 2 * two_p_minus_1)
# sqrt((2p-1)^3 (18p+7)) = (2p-1)^2 q0 (both factors positive for p>1/2): (2p-1)^2 q0 = 16(1-y)^4/(81y^2) * 3(1+y)/(1-y) = 16(1-y)^3(1+y)/(27 y^2)
root_y = 16 * (1 - y) ** 3 * (1 + y) * L({-2: F(1, 27)})
print("root^2 == (2p-1)^3(18p+7):", root_y ** 2 == two_p_minus_1 ** 3 * (18 * p_y + 7))
Vplus_y = (36 * p_y ** 2 - 12 * p_y - 3 + 3 * root_y) * F(1, 4)
print("V+(p(y)) == 4(1-y)^2(y+2)/(9y^2):", Vplus_y == 4 * (1 - y) ** 2 * (y + 2) * L({-2: F(1, 9)}))
print("A(y) == 8(1-y)^2(2y+1)(y+2)/(27y^2):", A_y == 8 * (1 - y) ** 2 * (2 * y + 1) * (y + 2) * L({-2: F(1, 27)}))
# derivative of p(y): p = (4/18) y + 1/18 + (4/18) y^-1 -> p' = 4/18 - (4/18) y^-2 ; -p' = (2/9)(y^-2 - 1)
minus_dp = L({-2: F(2, 9), 0: F(-2, 9)})
print("-p'(y) == (2/9)(y^-2-1): by construction; check p(1/4)=1, p(1)=1/2:",
      sum(c * F(1, 4) ** k for k, c in p_y.d.items()) == 1, sum(c for c in p_y.d.values()) == F(1, 2))
# also V+ is the upper root of H_p (E14) : H_p(v) = -2v^2 + (36p^2-12p-3) v - 72p^3 + 36p^2 + 18p - 9 ; discriminant check
Hb = {(2, 0): F(36), (1, 0): F(-12), (0, 0): F(-3)}; Hc = {(3, 0): F(-72), (2, 0): F(36), (1, 0): F(18), (0, 0): F(-9)}
disc = padd(ppow(Hb, 2), pscale(Hc, 8))   # b^2 - 4(-2)c = b^2 + 8c
disc_claim = pscale(pmul(ppow({(1, 0): F(2), (0, 0): F(-1)}, 3), {(1, 0): F(18), (0, 0): F(7)}), 9)
print("disc(H_p) == 9(2p-1)^3(18p+7):", disc == disc_claim, " (so V+ = [b + 3 sqrt((2p-1)^3(18p+7))]/4 is the upper root)")

# ---------- evaluate a (p,v)-polynomial at p=p(y), v=vexpr(y) -> Laurent in y ----------
def eval_L(poly, pL, vL):
    out = L()
    for (i, j), c in poly.items(): out = out + c * (pL ** i) * (vL ** j)
    return out
def D_j_exact(j):
    """D_j = int_{S2_inner} eta^j Q du dv with eta = 1/2 - p + v/6, inner strip 1/2<p<1, A p < v < V+(p)"""
    eta = {(0, 0): F(1, 2), (1, 0): F(-1), (0, 1): F(1, 6)}
    integrand = pmul(ppow(eta, j), Q_pv)
    prim = int_v_primitive(integrand)
    inner = eval_L(prim, p_y, Vplus_y) - eval_L(prim, p_y, A_y * p_y)
    lau = inner * minus_dp
    return lau.integrate_quarter_to_one(), lau
(D0r, D0l), lauD = D_j_exact(0)
(D1r, D1l), _ = D_j_exact(1)
(D2r, D2l), _ = D_j_exact(2)
print("D_0 = %s + (%s) log2 ; claim 27066286003/223205220 - (79298560/4782969) log2 :" % (D0r, D0l),
      D0r == F(27066286003, 223205220) and D0l == -F(79298560, 4782969))
print("D_1 claim:", D1r == F(1259212131941593, 15982386572880) and D1l == -F(3186360320, 387420489), (D1r, D1l))
print("D_2 claim:", D2r == F(72706616008642669, 1315122095139840) and D2l == -F(15686696960, 3486784401), (D2r, D2l))
print("Laurent degree range of the D_0 integrand:", min(lauD.d), max(lauD.d), "coefficient of y^-1:", lauD.d.get(-1))
# E24 primitive check against my primitive for j=1,2
def P_j_E24(j):
    out = {}
    for l in range(j + 1):
        coef = F(math.comb(j, l), 6 ** l)
        pref = pscale(ppow({(0, 0): F(1, 2), (1, 0): F(-1)}, j - l), coef)
        bracket = padd(pscale(pmul(P_, ppow(V_, l + 4)), F(16, l + 4)), pscale(pmul(A_p, ppow(V_, l + 3)), F(-4, l + 3)),
                       pscale(pmul(pmul(P_, ppow(A_p, 2)), ppow(V_, l + 2)), F(-4, l + 2)), pscale(pmul(ppow(A_p, 3), ppow(V_, l + 1)), F(1, l + 1)))
        out = padd(out, pmul(pref, bracket))
    return out
for j in (0, 1, 2):
    eta = {(0, 0): F(1, 2), (1, 0): F(-1), (0, 1): F(1, 6)}
    mine = int_v_primitive(pmul(ppow(eta, j), Q_pv))
    print("E24 P_%d equals my primitive:" % j, padd(P_j_E24(j), pscale(mine, -1)) == {})

# ---------- point moments (E33 inputs, from R H3-H4): E eta = int_S eta Q / I ----------
def point_moment(j):
    eta = {(0, 0): F(1, 2), (1, 0): F(-1), (0, 1): F(1, 6)}
    prim = int_v_primitive(pmul(ppow(eta, j), Q_pv))
    m1 = int_p(padd(subst_v(prim, UP), pscale(subst_v(prim, pscale(A_p, F(-1, 2))), -1)), F(-1, 2), F(1, 2))
    m2 = int_p(padd(subst_v(prim, UP), pscale(subst_v(prim, pscale(A_p, F(1, 2))), -1)), F(1, 2), F(3, 2))
    return (m1 + m2) / I
m1, m2 = point_moment(1), point_moment(2)
print("point E eta =", m1, "== 5771/7062:", m1 == F(5771, 7062), "| E eta^2 =", m2, "== 2440/3531:", m2 == F(2440, 3531))

# ---------- E7 exact identity ----------
# theta* = (I - D)/I = 1 - D0r/I - (D0l/I) log2
th_r, th_l = 1 - D0r / I, -D0l / I
print("E7: theta* rational part", th_r, "== 1545114756173/1572181042176:", th_r == F(1545114756173, 1572181042176),
      "| log2 coeff", th_l, "== 10841600/4605999147:", th_l == F(10841600, 4605999147))

# ---------- certified intervals: log2 = 2 sum_{j>=0} 3^-(2j+1)/(2j+1), remainder bound (E26) ----------
N = 60
lo = 2 * sum(F(1, 3 ** (2 * j + 1) * (2 * j + 1)) for j in range(N))
rem = 2 * F(1, 3 ** (2 * N + 1)) / ((2 * N + 1) * (1 - F(1, 9)))
hi = lo + rem
# independent check of the remainder bound: the exact tail is smaller than rem (compare with 40 more terms)
tail40 = 2 * sum(F(1, 3 ** (2 * j + 1) * (2 * j + 1)) for j in range(N, N + 40))
print("log2 bracket width", float(rem), "; tail(40 more terms) < rem:", tail40 < rem, "; float log2 inside:", lo < F(math.log(2)) < hi)
def affine(r, l):  # value r + l*log2 as interval
    a, b = r + l * lo, r + l * hi
    return (min(a, b), max(a, b))
def ratio(num, den):  # (r1 + l1 log2)/(r2 + l2 log2) over the log2 bracket, monotone in log2 (checked by endpoints + midpoint)
    vals = []
    for t in (lo, hi, (lo + hi) / 2):
        vals.append((num[0] + num[1] * t) / (den[0] + den[1] * t))
    return min(vals), max(vals)
def show(name, iv, lo_claim, hi_claim):
    a, b = iv
    ok = F(lo_claim) < a and b < F(hi_claim)
    print("%s in (%.21f, %.21f); claim (%s, %s) is an outward enclosure: %s" % (name, float(a), float(b), lo_claim, hi_claim, ok))
    return ok
Ir = (I, F(0)); Dv = (D0r, D0l); IminusD = (I - D0r, -D0l); Jv = (J, F(0))
ok9 = show("theta*", ratio(IminusD, Ir), "0.984415773209943375", "0.984415773209943376")
ok9 &= show("J/(I-D)", ratio(Jv, IminusD), "0.558034234065747870", "0.558034234065747871")
ok9 &= show("D/(I-D)", ratio(Dv, IminusD), "0.015830939745347846", "0.015830939745347847")
# E33
num1 = (I * F(5771, 7062) - D1r, -D1l); num2 = (I * F(2440, 3531) - D2r, -D2l)
ok33 = show("E eta_max", ratio(num1, IminusD), "0.819586988784918536", "0.819586988784918537")
ok33 &= show("E eta_max^2", ratio(num2, IminusD), "0.694438469047655310", "0.694438469047655311")
# positivity and ordering claims
Dlo, Dhi = affine(D0r, D0l)
print("D > 0:", Dlo > 0, " D approx", float((Dlo + Dhi) / 2), "; I-D-J > 0:", (I - Dhi - J) > 0, float(I - (Dlo + Dhi) / 2 - J),
      "; I-D > J:", I - Dhi > J)
print("E9 all enclosures hold:", ok9, "| E33 enclosures hold:", ok33)

# ---------- floating cross-checks (controls only): 2D quadrature of Q over S2_inner and of Q|t_c|^11 over S2_outer (E19) ----------
def Qf(p, v):
    A = 12 * p * p - 3
    return (4 * v * v - A * A) * (4 * p * v - A)
def Vplus(p): return (36 * p * p - 12 * p - 3 + 3 * math.sqrt((2 * p - 1) ** 3 * (18 * p + 7))) / 4
def gl(n):
    # Gauss-Legendre nodes/weights via Newton on Legendre polynomials
    xs, ws = [], []
    for i in range(1, n + 1):
        x = math.cos(math.pi * (i - 0.25) / (n + 0.5))
        for _ in range(100):
            p0, p1 = 1.0, x
            for k in range(2, n + 1): p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
            dp = n * (x * p1 - p0) / (x * x - 1)
            dx = p1 / dp; x -= dx
            if abs(dx) < 1e-15: break
        xs.append(x); ws.append(2 / ((1 - x * x) * dp * dp))
    return xs, ws
X, W = gl(80)
def integ(f, a, b):
    return sum(w * f((b - a) / 2 * x + (a + b) / 2) for x, w in zip(X, W)) * (b - a) / 2
def inner_v(p): return integ(lambda v: Qf(p, v), (12 * p * p - 3) * p, Vplus(p))
# substitute p = 1/2 + s^2 to soften the (2p-1)^{3/2} endpoint behaviour
D_num = integ(lambda s: inner_v(0.5 + s * s) * 2 * s, 0.0, math.sqrt(0.5))
print("float 2D quadrature of Q over S2_inner:", D_num, "vs exact D:", float((Dlo + Dhi) / 2), "rel diff %.2e" % abs(D_num / float((Dlo + Dhi) / 2) - 1))
def tc(p, v):
    b = -(12 * p * p - 3)
    u = -p
    return (4 * v * v - b * b) / (4 * v * v - 8 * b * u * v + b * b)
def outer_v(p):
    A = 12 * p * p - 3
    hi_v = A * p if p < 1 else 3 + 6 * p
    return integ(lambda v: Qf(p, v) * abs(tc(p, v)) ** 11, A / 2, hi_v)
E19_num = integ(lambda s: outer_v(0.5 + s * s) * 2 * s, 0.0, math.sqrt(0.5)) + integ(outer_v, 1.0, 1.5)
print("float int_{S2_outer} Q |t_c|^11 (E19):", E19_num, "rel diff to D %.2e" % abs(E19_num / float((Dlo + Dhi) / 2) - 1))
print("float I check:", integ(lambda p: integ(lambda v: Qf(p, v), abs(12 * p * p - 3) / 2, 3 + 6 * p), -0.5, 1.5), float(I))
