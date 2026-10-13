"""Math-#185 Slice A (sections 2-3) independent checks. Exact rational arithmetic where the claim is algebraic;
high-precision quadrature (Gauss-Legendre, pure Python) as a control for the asymptotic/TV claims."""
from fractions import Fraction as F
import math, random
random.seed(185)
# ---------------- bivariate polynomial helpers: dict {(i, j): coeff} meaning sum c * e^i x^j  (generic names X0, X1)
def pmul(a, b):
    out = {}
    for (i, j), c in a.items():
        for (k, l), d in b.items():
            out[(i + k, j + l)] = out.get((i + k, j + l), 0) + c * d
    return {k: v for k, v in out.items() if v != 0}
def padd(*ps):
    out = {}
    for p in ps:
        for k, v in p.items():
            out[k] = out.get(k, 0) + v
    return {k: v for k, v in out.items() if v != 0}
def psc(a, c): return {k: v * c for k, v in a.items()}
def ppow(a, n):
    r = {(0, 0): F(1)}
    for _ in range(n): r = pmul(r, a)
    return r
E, X = {(1, 0): F(1)}, {(0, 1): F(1)}           # e and x
ONE = {(0, 0): F(1)}
def peval(p, ev, xv): return sum(c * ev ** i * xv ** j for (i, j), c in p.items())
# ---------------- (1) the source density and E0
K = psc(pmul(pmul(padd(E, psc(pmul(X, X), -1)), padd(E, pmul(X, X), psc(X, 2))), padd(E, psc(pmul(E, X), 2), pmul(X, X))), 10368)
E0 = F(888807, 280)
def prim_e(p): return {(i + 1, j): c / (i + 1) for (i, j), c in p.items()}       # antiderivative in e
def int_e_between(p, lo_poly_in_x, hi):   # integral over e from lo(x) (poly in x, given as dict in X) to constant hi, result poly in x
    P = prim_e(p)
    out = {}
    for (i, j), c in P.items():
        # c * (hi^i - lo(x)^i) * x^j
        out = padd(out, {(0, j): c * F(hi) ** i}, psc(pmul(ppow(lo_poly_in_x, i), {(0, j): F(1)}), -c))
    return out
def int_x(p, a, b):   # p poly in x only (keys (0, j)); exact rational endpoints
    return sum(c * (F(b) ** (j + 1) - F(a) ** (j + 1)) / (j + 1) for (i, j), c in p.items() if i == 0)
# strips: -1<x<0, -2x-x^2<e<1 ; 0<x<1/2, 3x^2+2x^3<e<1
strip1 = int_e_between(K, padd(psc(X, -2), psc(pmul(X, X), -1)), 1)
strip2 = int_e_between(K, padd(psc(pmul(X, X), 3), psc(ppow(X, 3), 2)), 1)
mass = int_x(strip1, -1, 0) + int_x(strip2, 0, F(1, 2))
print("[1] total mass of K over the two strips == E0 = 888807/280:", mass == E0, mass)
# ---------------- Lemma 1: identity (4) exactly
p = padd(X, {(0, 0): F(1, 2)}); A = psc(pmul(X, padd(ONE, X)), 12); v = psc(padd(E, X), 6)
lhs = padd(pmul(ppow(padd(v, psc(A, -1)), 2), padd(psc(v, 2), A)), psc(ppow(padd(v, psc(pmul(A, p), -1)), 2), -12))
cub = padd(psc(ppow(X, 3), 2), psc(pmul(E, pmul(X, X)), 3), psc(pmul(E, E), -1))          # 2x^3 + 3 e x^2 - e^2
rhs = psc(pmul(padd(ONE, psc(E, -1)), cub), 432)
print("[2] identity (4): (v-A)^2(2v+A) - 12(v-Ap)^2 == 432(1-e)(2x^3+3ex^2-e^2):", lhs == rhs)
# A == 12p^2 - 3 and v < A  <=>  e < x + 2x^2
print("[2b] A == 12p^2-3:", A == padd(psc(pmul(p, p), 12), {(0, 0): F(-3)}))
# value of the cubic factor at e = x + 2x^2 equals x^2(2x-1)(x+1); derivative in e is 3x^2 - 2e
sub = {}
for (i, j), c in cub.items():
    sub = padd(sub, psc(pmul(ppow(padd(X, psc(pmul(X, X), 2)), i), {(0, j): F(1)}), c))
print("[2c] cubic factor at e = x+2x^2 == x^2(2x-1)(x+1):", sub == pmul(pmul(pmul(X, X), padd(psc(X, 2), {(0, 0): F(-1)})), padd(X, ONE)))
# ---------------- Lemma 1 against the CUB classifier as validated in the Math-#169 Slice A review (cub_n_is_2, in_S)
def b0(u): return 3 - 12 * u * u
def in_S(u, vv): return F(-3, 2) < u < F(1, 2) and abs(b0(u)) / 2 < vv < 3 - 6 * u
def cub_n_is_2(u, vv):
    b = b0(u)
    return vv < -b and 12 * (vv - b * u) ** 2 < (vv + b) ** 2 * (2 * vv - b)
def in_O2(pp, vv):                   # Math-#169 E14 outer doublets
    Ap = 12 * pp * pp - 3
    if F(1, 2) < pp < 1: return Ap / 2 < vv < Ap * pp
    if 1 <= pp < F(3, 2): return Ap / 2 < vv < 3 + 6 * pp
    return False
def rq(lo, hi, den=4001):
    d = random.randint(1, den); return lo + (hi - lo) * F(random.randint(0, d), d)
agree = in_support_E = 0; total = 0; sing_neg = 0
for _ in range(20000):
    xv = rq(F(-1, 2), F(1, 2))
    lo_e = (3 * xv * xv + 2 * xv ** 3) if xv > 0 else (-2 * xv - xv * xv)
    if lo_e >= 1: continue
    ev = rq(lo_e, F(1)) if lo_e < 1 else None
    if ev is None or ev <= lo_e or ev >= 1: continue
    total += 1
    u = -(xv + F(1, 2)); vv = 6 * (ev + xv)
    in_support_E += in_S(u, vv) and not in_O2(-u, vv)
    n2_cub = cub_n_is_2(u, vv)
    n2_lemma = peval(cub, ev, xv) > 0          # x > c(e)  <=>  2x^3 + 3 e x^2 - e^2 > 0
    agree += (n2_cub == n2_lemma)
    if xv <= 0 and n2_cub: sing_neg += 1
print("[3] random support points:", total, "| in S and not O2 (the elder region):", in_support_E, "| CUB n=2 == (x > c(e)):", agree, "| doublets with x<=0:", sing_neg)
# ---------------- (10) and (12): the two changes of variables, exact in (delta, w) and (tau, z)
D, Wv = {(1, 0): F(1)}, {(0, 1): F(1)}   # generic variable pair
def Ksub(e_expr, x_expr):
    out = {}
    for (i, j), c in K.items():
        out = padd(out, psc(pmul(ppow(e_expr, i), ppow(x_expr, j)), c))
    return out
K10 = pmul(Ksub(pmul(D, D), pmul(D, Wv)), D)              # K(delta^2, delta w) * delta   (dx = delta dw)
Phi = pmul(pmul(padd(ONE, psc(pmul(Wv, Wv), -1)), padd(psc(Wv, 2), pmul(D, padd(ONE, pmul(Wv, Wv))))), padd(ONE, pmul(Wv, Wv), psc(pmul(D, Wv), 2)))
print("[4] (10): K(e,x)dx == 10368 delta^6 Phi_delta(w) dw with Phi_delta=(1-w^2)[2w+delta(1+w^2)][1+w^2+2 delta w]:", K10 == psc(pmul(ppow(D, 6), Phi), 10368))
T, Z = D, Wv
K12 = pmul(Ksub(ppow(T, 3), pmul(pmul(T, T), Z)), pmul(T, T))     # K(tau^3, tau^2 z) * tau^2
I12 = pmul(pmul(padd(ONE, psc(pmul(T, pmul(Z, Z)), -1)), padd(psc(Z, 2), T, pmul(pmul(T, T), pmul(Z, Z)))), padd(ONE, pmul(T, pmul(Z, Z)), psc(pmul(pmul(T, T), Z), 2)))
print("[5] (12): K(e,x)dx == 10368 tau^10 (1-tau z^2)(2z+tau+tau^2 z^2)(1+tau z^2+2tau^2 z) dz:", K12 == psc(pmul(ppow(T, 10), I12), 10368))
# A0, the normalizations of (7), and C_s
A0 = F(1, 3) - F(1, 81)            # int_0^{1/sqrt3} 2w(1-w^4) dw = w^2 - w^6/3 at w^2=1/3
print("[6] A0 = 26/81:", A0 == F(26, 81), "| 10368*A0/4 == 832:", 10368 * A0 / 4 == 832, "| f_W normalizer 2/A0 == 81/13:", 2 / A0 == F(81, 13),
      "| C_s rational part 10368*(3/13)/832 == 486/169:", F(10368 * 3, 13 * 832) == F(486, 169))
# first-order coefficient of int Phi_delta (interior + moving upper boundary): c1 = (272/189 - 16/81)/sqrt3 ; kappa = 8 c1/(9 A0)
c1_r = F(272, 189) - F(16, 81)   # times 1/sqrt3
kappa_r = 8 * c1_r / (9 * A0)     # times 1/sqrt3
print("[7] predicted A_eps = 832 eps^4 [1 + kappa sqrt(eps) + O(eps)], kappa = %s/sqrt3 = %.9f" % (kappa_r, float(kappa_r) / math.sqrt(3)))
# ---------------- numerics: Gauss-Legendre
def gl(n):
    xs, ws = [], []
    for i in range(1, n + 1):
        x = math.cos(math.pi * (i - 0.25) / (n + 0.5))
        for _ in range(100):
            p0, p1 = 1.0, x
            for k in range(2, n + 1):
                p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
            dp = n * (x * p1 - p0) / (x * x - 1)
            dx = p1 / dp; x -= dx
            if abs(dx) < 1e-16: break
        xs.append(x); ws.append(2 / ((1 - x * x) * dp * dp))
    return xs, ws
GX, GW = gl(96)
def integ(f, a, b): return sum(w * f((b - a) / 2 * t + (a + b) / 2) for t, w in zip(GX, GW)) * (b - a) / 2
Kf = [(i, j, float(c)) for (i, j), c in K.items()]
def Kval(ev, xv): return sum(c * ev ** i * xv ** j for i, j, c in Kf)
def Kprim_x(ev, xv): return sum(c * ev ** i * xv ** (j + 1) / (j + 1) for i, j, c in Kf)
def root_b(ev):        # 3b^2 + 2b^3 = e
    b = math.sqrt(ev / 3)
    for _ in range(60):
        f = 3 * b * b + 2 * b ** 3 - ev; b -= f / (6 * b + 6 * b * b)
    return b
def root_c(ev):        # 2c^3 + 3 e c^2 = e^2
    c = (ev * ev / 2) ** (1 / 3)
    for _ in range(80):
        f = 2 * c ** 3 + 3 * ev * c * c - ev * ev; c -= f / (6 * c * c + 6 * ev * c)
    return c
def a_of(ev): return -ev / (1 + math.sqrt(1 - ev))
print("[8] asymptotics (5): e, b/sqrt(e)-1/sqrt3, c/e^(2/3)-2^(-1/3), a/e+1/2  (each O(sqrt e), O(e^(1/3)), O(e))")
for ev in (1e-2, 1e-4, 1e-6, 1e-8):
    print("     e=%.0e  %+.3e (/sqrt e = %+.4f)  %+.3e (/e^(1/3) = %+.4f)  %+.3e (/e = %+.4f)" % (ev, root_b(ev) / math.sqrt(ev) - 1 / math.sqrt(3), (root_b(ev) / math.sqrt(ev) - 1 / math.sqrt(3)) / math.sqrt(ev),
          root_c(ev) / ev ** (2 / 3) - 2 ** (-1 / 3), (root_c(ev) / ev ** (2 / 3) - 2 ** (-1 / 3)) / ev ** (1 / 3), a_of(ev) / ev + 0.5, (a_of(ev) / ev + 0.5) / ev))
def fe(ev): return Kprim_x(ev, root_b(ev)) - Kprim_x(ev, a_of(ev))         # unnormalized density of e
def S(ev): return Kprim_x(ev, root_c(ev)) - Kprim_x(ev, a_of(ev))          # singleton part
def A_eps(eps): return integ(lambda s: fe(eps * s * s) * 2 * eps * s, 0.0, 1.0)      # e = eps s^2
def S_eps(eps): return integ(lambda t: S(eps * t ** 3) * 3 * eps * t * t, 0.0, 1.0)  # e = eps t^3
kap = float(kappa_r) / math.sqrt(3); Cs = float(F(486, 169)) * 2 ** (-2 / 3)
print("[9] (11)/(13)/(8): eps, A/(832 eps^4)-1 over sqrt(eps) [-> kappa=%.6f], S(eps)/(10368 2^(-2/3) eps^(10/3))-1 over eps^(1/3), P(n=1|eps)/eps^(1/3) [-> C_s=%.9f], (P/eps^(1/3)-C_s)/eps^(1/3)" % (kap, Cs))
for eps in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
    A = A_eps(eps); Pn1 = S_eps(eps) / A
    print("     eps=%.0e  %.6f   %+.5f   %.9f   %+.5f" % (eps, (A / (832 * eps ** 4) - 1) / math.sqrt(eps), (S(eps) / (10368 * 2 ** (-2 / 3) * eps ** (10 / 3)) - 1) / eps ** (1 / 3), Pn1 / eps ** (1 / 3), (Pn1 / eps ** (1 / 3) - Cs) / eps ** (1 / 3)))
print("     exact-e conditional singleton probability S(e)/f(e) / [ (81/26) 2^(-2/3) e^(1/3) ]:", ["%.6f" % (S(ev) / fe(ev) / (F(81, 26) * 2 ** (-2 / 3) * ev ** (1 / 3))) for ev in (1e-3, 1e-5, 1e-7, 1e-9)])
# ---------------- (7) and (9): TV distances by quadrature with sign-change and boundary splitting
def Phi_d(d, w): return (1 - w * w) * (2 * w + d * (1 + w * w)) * (1 + w * w + 2 * d * w)
def f_W(w): return (81 / 13) * w * (1 - w ** 4) if 0 < w < 1 / math.sqrt(3) else 0.0
def tv_pair(eps, tagged):
    A = A_eps(eps)
    def inner(y):
        d = math.sqrt(eps * y); ev = eps * y
        l = -d / (1 + math.sqrt(1 - d * d)); h = root_b(ev) / d; wc = root_c(ev) / d
        def g(w): return 10368 * eps ** 4 * y ** 3 * Phi_d(d, w) / A if l < w < h else 0.0
        def diff(w):
            gg = g(w)
            if tagged and w <= wc: gg = 0.0
            return abs(gg - 4 * y ** 3 * f_W(w))
        pts = sorted(set([l, 0.0, wc, h, 1 / math.sqrt(3)]))
        # sign changes of g - F inside [0, h]
        grid = [pts[0] + (pts[-1] - pts[0]) * k / 4000 for k in range(4001)]
        def sg(w):
            gg = g(w)
            if tagged and w <= wc: gg = 0.0
            return gg - 4 * y ** 3 * f_W(w)
        vals = [sg(w) for w in grid]
        for k in range(4000):
            if vals[k] * vals[k + 1] < 0:
                lo, hi = grid[k], grid[k + 1]
                for _ in range(60):
                    mid = (lo + hi) / 2
                    if sg(lo) * sg(mid) <= 0: hi = mid
                    else: lo = mid
                pts.append((lo + hi) / 2)
        pts = sorted(set(pts))
        return sum(integ(diff, pts[k], pts[k + 1]) for k in range(len(pts) - 1))
    tot = integ(lambda s: inner(s * s) * 2 * s, 0.0, 1.0)     # y = s^2
    extra = S_eps(eps) / A if tagged else 0.0                   # the n=1 part of the tagged law has no counterpart
    return 0.5 * (tot + extra)
print("[10] (7): dTV(Law(Y,W), 4y^3 x f_W) / sqrt(eps)  and  (9): tagged TV, (TV_tag - P(n=1))/sqrt(eps), TV_tag/eps^(1/3)")
for eps in (1e-2, 1e-3, 1e-4):
    tv = tv_pair(eps, False); tvt = tv_pair(eps, True); Pn1 = S_eps(eps) / A_eps(eps)
    print("     eps=%.0e  TV=%.6e  TV/sqrt(eps)=%.5f | TV_tag=%.6e  (TV_tag-P1)/sqrt(eps)=%.5f  TV_tag/eps^(1/3)=%.5f  (C_s=%.5f)" % (eps, tv, tv / math.sqrt(eps), tvt, (tvt - Pn1) / math.sqrt(eps), tvt / eps ** (1 / 3), Cs))
