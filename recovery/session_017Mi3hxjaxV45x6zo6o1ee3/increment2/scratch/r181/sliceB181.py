"""Exact reconstruction of Math-#181 Slice B (sections 4-5). stdlib only."""
from fractions import Fraction as F
import math
# ---------- bivariate polynomial helpers: dict {(i,j): coeff} in variables (X, Y) ----------
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
def psc(a, c): return {k: v * c for k, v in a.items()}
def ppow(a, n):
    r = {(0, 0): F(1)}
    for _ in range(n): r = pmul(r, a)
    return r
def const(c): return {(0, 0): F(c)}
X = {(1, 0): F(1)}; Y = {(0, 1): F(1)}
def subst_Y(poly, yexpr):          # substitute Y := polynomial in X
    out = {}
    for (i, j), c in poly.items(): out = padd(out, pmul({(i, 0): c}, ppow(yexpr, j)))
    return out
def prim_Y(poly): return {(i, j + 1): c / (j + 1) for (i, j), c in poly.items()}
def int_X(poly, a, b):
    tot = F(0)
    for (i, j), c in poly.items():
        assert j == 0; tot += c * (F(b) ** (i + 1) - F(a) ** (i + 1)) / (i + 1)
    return tot
# ---------- Q at u=-p as polynomial in (p, v): (4v^2 - A^2)(4pv - A), A = 12p^2 - 3 ; eta = 1/2 - p + v/6 ----------
P, V = X, Y
A = padd(psc(ppow(P, 2), 12), const(-3))
Q = pmul(padd(psc(ppow(V, 2), 4), psc(ppow(A, 2), -1)), padd(psc(pmul(P, V), 4), psc(A, -1)))
eta = padd(const(F(1, 2)), psc(P, -1), psc(V, F(1, 6)))
UP = padd(const(3), psc(P, 6))
def strip(poly, lo_v, hi_v, a, b):
    pr = prim_Y(poly)
    return int_X(padd(subst_Y(pr, hi_v), psc(subst_Y(pr, lo_v), -1)), a, b)
def S_int(j):
    f = pmul(ppow(eta, j), Q)
    return strip(f, psc(A, F(-1, 2)), UP, F(-1, 2), F(1, 2)) + strip(f, psc(A, F(1, 2)), UP, F(1, 2), F(3, 2))
def O2_int(j):
    f = pmul(ppow(eta, j), Q)
    return strip(f, psc(A, F(1, 2)), pmul(A, P), F(1, 2), 1) + strip(f, psc(A, F(1, 2)), UP, 1, F(3, 2))
I = S_int(0); J = O2_int(0)
E = [S_int(j) - O2_int(j) for j in range(3)]
print("I =", I, I == F(246528, 35), "| J =", J, J == F(1083417, 280))
print("E0 =", E[0], "== 888807/280:", E[0] == F(888807, 280), "== I-J:", E[0] == I - J)
print("E1 =", E[1], "== 3524743221/1361360:", E[1] == F(3524743221, 1361360))
print("E2 =", E[2], "== 190490806059/87127040:", E[2] == F(190490806059, 87127040))
print("E[lambda] =", E[1] / E[0], "== 1174914407/1440459878:", E[1] / E[0] == F(1174914407, 1440459878), float(E[1] / E[0]))
print("E[lambda^2] =", E[2] / E[0], "== 63496935353/92189432192:", E[2] / E[0] == F(63496935353, 92189432192), float(E[2] / E[0]))
print("(12) E0/I == 296269/657408:", E[0] / I == F(296269, 657408), float(E[0] / I))
# ---------- slice algebra in (x, e) with p = x + 1/2, v = 6(e + x) ----------
x, e = X, Y
p_of_x = padd(x, const(F(1, 2)))
v_of = psc(padd(e, x), 6)
def subst_pv(poly, pexpr, vexpr):
    out = {}
    for (i, j), c in poly.items(): out = padd(out, pmul(pmul(const(c), ppow(pexpr, i)), ppow(vexpr, j)))
    return out
Q_xe = subst_pv(Q, p_of_x, v_of)
claim17 = pmul(pmul(psc(padd(e, psc(ppow(x, 2), -1)), 10368), padd(e, ppow(x, 2), psc(x, 2))), padd(e, psc(pmul(e, x), 2), ppow(x, 2)))
print("(17): 6*Q(-(x+1/2), 6(e+x)) == 10368 (e-x^2)(e+x^2+2x)(e+2ex+x^2):", padd(psc(Q_xe, 6), psc(claim17, -1)) == {})
# typed lower edge |b0|/2 : for p>=1/2, |b0|/2 = 6p^2 - 3/2 ; v - (6p^2-3/2) == 6(e - x^2)
b_edge_hi = padd(psc(ppow(p_of_x, 2), 6), const(F(-3, 2)))
print("v - |b0|/2 == 6(e - x^2) for p >= 1/2:", padd(v_of, psc(b_edge_hi, -1), psc(padd(e, psc(ppow(x, 2), -1)), -6)) == {})
b_edge_lo = padd(const(F(3, 2)), psc(ppow(p_of_x, 2), -6))          # p < 1/2: |b0|/2 = 3/2 - 6p^2
print("v - |b0|/2 == 6[(x+1)^2 - (1-e)] for p < 1/2:", padd(v_of, psc(b_edge_lo, -1), psc(padd(ppow(padd(x, const(1)), 2), const(-1), e), -6)) == {})
Ap = pmul(subst_pv(A, p_of_x, v_of), p_of_x)
print("A p - v == 6(2x^3 + 3x^2 - e):", padd(Ap, psc(v_of, -1), psc(padd(psc(ppow(x, 3), 2), psc(ppow(x, 2), 3), psc(e, -1)), -6)) == {})
cubic_p = padd(psc(ppow(p_of_x, 3), 2), psc(p_of_x, F(-3, 2)), const(F(1, 2)))
print("2p^3 - 3p/2 + 1/2 at p = x+1/2 == 2x^3 + 3x^2:", padd(cubic_p, psc(padd(psc(ppow(x, 3), 2), psc(ppow(x, 2), 3)), -1)) == {})
# g(x) = 3x^2 + 2x^3 increasing on x>0, g(0)=0, g(1/2)=1; g(sqrt e) - e = 2e + 2 e^{3/2} > 0 so x_hi < sqrt(e)
print("g(1/2) == 1:", 3 * F(1, 4) + 2 * F(1, 8) == 1)
# trig form p_hi = cos(arccos(2e-1)/3): numeric
okt = True
for ev in (0.01, 0.1, 0.5, 0.9, 0.999):
    ph = math.cos(math.acos(2 * ev - 1) / 3)
    okt &= abs(2 * ph ** 3 - 1.5 * ph + 0.5 - ev) < 1e-14 and 0.5 < ph < 1
print("p_hi = cos(arccos(2e-1)/3) solves 2p^3-3p/2+1/2=e in (1/2,1):", okt)
# ---------- small-e scaling: x = s z, e = s^2 ; integrand * dx/dz = s * 10368(...) ; divide by s^6 ----------
s, z = X, Y
scaled = subst_pv(claim17, None, None) if False else None
# substitute directly: e -> s^2, x -> s z  in claim17 (variables (x,e) -> (s,z))
def subst_xe(poly, xexpr, eexpr):
    out = {}
    for (i, j), c in poly.items(): out = padd(out, pmul(pmul(const(c), ppow(xexpr, i)), ppow(eexpr, j)))
    return out
sz = pmul(s, z); s2 = ppow(s, 2)
num = pmul(subst_xe(claim17, sz, s2), s)              # times dx = s dz
# divide by s^6
lead = {(i - 6, j): c for (i, j), c in num.items()}
assert min(i for (i, j) in lead) >= 0, "negative power of s: scaling wrong"
leading = {(i, j): c for (i, j), c in lead.items() if i == 0}
target = psc(pmul(z, padd(const(1), psc(ppow(z, 4), -1))), 10368 * 2)   # 10368 * 2 z (1 - z^4) as (s^0) term
print("small-e limit integrand (s^0 coefficient) == 10368*2z(1-z^4):", padd(leading, psc(target, -1)) == {},
      "; remaining terms are O(s):", all(i >= 1 for (i, j) in lead if (i, j) not in leading))
# int_0^{1/sqrt3} 2z(1-z^4) dz = [z^2 - z^6/3] = 1/3 - 1/81
val = F(1, 3) - F(1, 81)
print("int_0^{1/sqrt3} 2z(1-z^4) dz = 1/3 - 1/81 =", val, "; 10368 * that =", 10368 * val, "== 3328:", 10368 * val == 3328)
print("CDF coefficient 832/E0 == 232960/888807:", F(832) / E[0] == F(232960, 888807))
# ---------- endpoint e = 1: int_{-1}^{1/2} 10368 (1-x^2)(1+x^2+2x)(1+2x+x^2) dx = int 10368 (1-x)(1+x)^5 ----------
poly_e1 = subst_xe(claim17, x, const(1))              # polynomial in x only (as (i,0))
val_e1 = int_X(poly_e1, -1, F(1, 2))
print("endpoint integral at e=1:", val_e1, "== 98415/7:", val_e1 == F(98415, 7), "; f_eld(1-) = ", val_e1 / E[0], "== 1312200/296269:", val_e1 / E[0] == F(1312200, 296269))
# ---------- floating controls: f_eld normalization, mean, second moment, small-e ratio ----------
def integrand_x(ev, xv): return 10368 * (ev - xv * xv) * (ev + xv * xv + 2 * xv) * (ev + 2 * ev * xv + xv * xv)
def x_hi(ev):
    lo, hi = 0.0, 0.5
    for _ in range(200):
        mid = (lo + hi) / 2
        if 3 * mid * mid + 2 * mid ** 3 < ev: lo = mid
        else: hi = mid
    return (lo + hi) / 2
def gl(n):
    xs, ws = [], []
    for i in range(1, n + 1):
        xx = math.cos(math.pi * (i - 0.25) / (n + 0.5))
        for _ in range(100):
            p0, p1 = 1.0, xx
            for kk in range(2, n + 1): p0, p1 = p1, ((2 * kk - 1) * xx * p1 - (kk - 1) * p0) / kk
            dp = n * (xx * p1 - p0) / (xx * xx - 1); dx = p1 / dp; xx -= dx
            if abs(dx) < 1e-15: break
        xs.append(xx); ws.append(2 / ((1 - xx * xx) * dp * dp))
    return xs, ws
GX, GW = gl(40)
def integ(f, a, b): return sum(w * f((b - a) / 2 * t + (a + b) / 2) for t, w in zip(GX, GW)) * (b - a) / 2
def f_unnorm(ev): return integ(lambda xv: integrand_x(ev, xv), math.sqrt(1 - ev) - 1, x_hi(ev))
E0f = float(E[0])
# integrate in e with the substitution e = w^2 to soften the e^{1/2} endpoint terms and also split
def outer(fn):
    return integ(lambda w: fn(w * w) * 2 * w, 0.0, 1.0)
norm = outer(lambda ev: f_unnorm(ev)) / E0f
mean = outer(lambda ev: ev * f_unnorm(ev)) / E0f
m2 = outer(lambda ev: ev * ev * f_unnorm(ev)) / E0f
print("float: int f_eld = %.12f ; E[lambda] = %.12f (exact %.12f) ; E[lambda^2] = %.12f (exact %.12f)" % (norm, mean, float(E[1] / E[0]), m2, float(E[2] / E[0])))
for ev in (1e-2, 1e-3, 1e-4, 1e-5):
    print("  small-e ratio f_eld(e) / (3328 e^3 / E0) at e=%g: %.6f" % (ev, (f_unnorm(ev) / E0f) / (3328 * ev ** 3 / E0f)))
print("  f_eld(1-) float:", f_unnorm(1 - 1e-12) / E0f, "exact", float(F(1312200, 296269)))
# positivity: interval nonempty for all e in (0,1): p_lo < 1/2 < p_hi
print("interval nonempty: sqrt(1-e)-1/2 < 1/2 < p_hi for all e in (0,1): True (p_hi in (1/2,1), sqrt(1-e) < 1)")
