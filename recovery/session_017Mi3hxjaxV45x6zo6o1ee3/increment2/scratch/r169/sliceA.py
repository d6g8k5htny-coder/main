"""Exact controls for Math-#169 Slice A (E10-E19). stdlib only; Fractions throughout; forward-mode exact derivatives."""
from fractions import Fraction as F
import random, math
random.seed(169)
def b0(u): return 3 - 12 * u * u
def eta_of(u, v): return u + F(1, 2) + v / 6
def in_S(u, v): return F(-3, 2) < u < F(1, 2) and abs(b0(u)) / 2 < v < 3 - 6 * u
def E12(u, v):
    b = b0(u); L = 4 * v * v - 8 * b * u * v + b * b
    if L == 0: return None
    tc = (4 * v * v - b * b) / L
    uc = (2 * b * v - u * (4 * v * v + b * b)) / L
    return uc, v * tc * tc, tc
def cub_n_is_2(u, v):
    b = b0(u)                                   # CUB: n=2 iff v < -b and 12(v-bu)^2 < (v+b)^2 (2v-b)
    return v < -b and 12 * (v - b * u) ** 2 < (v + b) ** 2 * (2 * v - b)
# ---------- bivariate polynomials for symbolic identities ----------
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
Pp = {(1, 0): F(1)}; Vv = {(0, 1): F(1)}          # variables (p, v) or (p, ell)
A = padd(psc(ppow(Pp, 2), 12), const(-3))          # A = 12p^2-3
Up = padd(const(3), psc(Pp, 6))                     # U_p = 3+6p
Hp = padd(psc(ppow(Vv, 2), -2), pmul(padd(psc(ppow(Pp, 2), 36), psc(Pp, -12), const(-3)), Vv),
          psc(ppow(Pp, 3), -72), psc(ppow(Pp, 2), 36), psc(Pp, 18), const(-9))
lhs = padd(pmul(pmul(ppow(padd(Vv, psc(A, -1)), 2), padd(psc(Vv, 2), A)), const(1)), psc(ppow(padd(Vv, psc(pmul(A, Pp), -1)), 2), -12))
rhs = pmul(padd(Up, psc(Vv, -1)), Hp)
print("E14 factorization (v-A)^2(2v+A) - 12(v-Ap)^2 == (U_p - v) H_p(v):", padd(lhs, psc(rhs, -1)) == {})
def poly_p(expr_in_v):  # substitute v := polynomial in p (given as dict) into Hp
    out = {}
    for (i, j), c in Hp.items(): out = padd(out, pmul({(i, 0): c}, ppow(expr_in_v, j)))
    return out
two_p_m1 = padd(psc(Pp, 2), const(-1)); two_p_p1 = padd(psc(Pp, 2), const(1)); p_m1 = padd(Pp, const(-1))
checks = {
 "H_p(L_p) == 9(2p-1)^3(2p+1)": padd(poly_p(psc(A, F(1, 2))), psc(pmul(ppow(two_p_m1, 3), two_p_p1), -9)) == {},
 "H_p(A) == 18(p-1)(2p-1)^2(2p+1)": padd(poly_p(A), psc(pmul(pmul(p_m1, ppow(two_p_m1, 2)), two_p_p1), -18)) == {},
 "H_p(U_p) == 36(p-1)(2p+1)^2": padd(poly_p(Up), psc(pmul(p_m1, ppow(two_p_p1, 2)), -36)) == {},
 "H_p(Ap) == -9(p-1)(2p-1)^3(2p+1)^2": padd(poly_p(pmul(A, Pp)), psc(pmul(pmul(p_m1, ppow(two_p_m1, 3)), ppow(two_p_p1, 2)), 9)) == {},
 "Ap - U_p == 3(p-1)(2p+1)^2": padd(pmul(A, Pp), psc(Up, -1), psc(pmul(p_m1, ppow(two_p_p1, 2)), -3)) == {},
}
for k_, v_ in checks.items(): print("E14 sign identities:", k_, v_)
# ---------- E17 symbolic: (eta_c - eta) * (4l^2-8pl+1)^2 == 2 (l-p)(4pl-1)^3, in variables (p, l) ----------
Ll = Vv  # reuse second variable as ell
den = padd(psc(ppow(Ll, 2), 4), psc(pmul(Pp, Ll), -8), const(1))          # 4l^2-8pl+1
uc_num = padd(psc(Ll, -2), pmul(Pp, padd(psc(ppow(Ll, 2), 4), const(1))))   # u_c = uc_num/den
tc_num = padd(psc(ppow(Ll, 2), 4), const(-1))                                # t_c = tc_num/den
Ap_ = padd(psc(ppow(Pp, 2), 4), const(-1))                                   # 4p^2-1 = A/3
# eta_c - eta = (u_c + p) + (v_c - v)/6, v = l A, v_c = v t_c^2, A = 3(4p^2-1)
# multiply by den^2: (uc_num + p den) den + (l A/6)(tc_num^2 - den^2)
lhs17 = padd(pmul(padd(uc_num, pmul(Pp, den)), den), pmul(psc(pmul(Ll, Ap_), F(1, 2)), padd(ppow(tc_num, 2), psc(ppow(den, 2), -1))))
rhs17 = psc(pmul(padd(Ll, psc(Pp, -1)), ppow(padd(psc(pmul(Pp, Ll), 4), const(-1)), 3)), 2)
print("E17 symbolic identity:", padd(lhs17, psc(rhs17, -1)) == {})
# ---------- exact random-point controls ----------
def rnd_frac(lo, hi, den_max=997):
    d = random.randint(2, den_max); n = random.randint(int(lo * d) + 1, int(hi * d) - 1)
    return F(n, d)
def critical_point_residuals(k, a, s, beta, c, X, Z):
    # P(X,Z)=2kX^3-3kX/2-k/2+sZ^2/2+(a/2)(X^2-1/4)Z+(beta/2)XZ^2+(c/6)Z^3   (M2)
    dPX = 6 * k * X * X - 3 * k / 2 + a * X * Z + beta * Z * Z / 2
    dPZ = s * Z + (a / 2) * (X * X - F(1, 4)) + beta * X * Z + c * Z * Z / 2
    return dPX, dPZ
n_pts = 0; ok_inv = ok_13 = ok_line = ok_roots = ok_n2 = ok_16 = ok_17 = ok_18 = True
n_doublet = 0
# exact forward-mode derivatives: value + gradient (d/du, d/dv)
class Dual:
    def __init__(s, v, du=F(0), dv=F(0)): s.v, s.du, s.dv = F(v), F(du), F(dv)
    def __add__(s, o): o = o if isinstance(o, Dual) else Dual(o); return Dual(s.v + o.v, s.du + o.du, s.dv + o.dv)
    __radd__ = __add__
    def __sub__(s, o): o = o if isinstance(o, Dual) else Dual(o); return Dual(s.v - o.v, s.du - o.du, s.dv - o.dv)
    def __rsub__(s, o): return Dual(o) - s
    def __mul__(s, o): o = o if isinstance(o, Dual) else Dual(o); return Dual(s.v * o.v, s.du * o.v + s.v * o.du, s.dv * o.v + s.v * o.dv)
    __rmul__ = __mul__
    def __truediv__(s, o):
        o = o if isinstance(o, Dual) else Dual(o)
        return Dual(s.v / o.v, (s.du * o.v - s.v * o.du) / (o.v * o.v), (s.dv * o.v - s.v * o.dv) / (o.v * o.v))
def Qf(u, v): return (4 * v * v - b0(u) ** 2) * (b0(u) - 4 * u * v)
while n_pts < 600:
    u = rnd_frac(-1.5, 0.5); v = rnd_frac(0, 6)
    if not in_S(u, v): continue
    r = E12(u, v)
    if r is None: continue
    uc, vc, tc = r; b = b0(u)
    n_pts += 1
    # involution: T(T(u,v)) == (u,v), t_c(T) t_c == 1
    r2 = E12(uc, vc)
    ok_inv &= r2 is not None and r2[0] == u and r2[1] == v and r2[2] * tc == 1
    # E13
    ok_13 &= b0(uc) == b * tc * tc and vc - b0(uc) * uc == (v - b * u) * tc ** 3
    # line and conic at Z' = t Z
    ok_line &= b * uc + (v - b * u) * tc == v and 12 * uc * uc + b * tc * tc == 3
    # both roots are critical points of the actual cubic M2, with jets reconstructed by R4/R5 (k=1, Z=1, random a)
    k = F(1); Z = F(1); a = rnd_frac(-9, 9)
    x = k * v / Z ** 2; B = k * b / Z ** 2; D = k * (v - b * u) / Z ** 3
    s = -x; beta = a * a / (12 * k) + B; c = a ** 3 / (144 * k * k) + a * B / (4 * k) + 2 * D
    for (uu, ZZ) in ((u, Z), (uc, tc * Z)):
        X = uu - a * ZZ / (12 * k)
        ok_roots &= critical_point_residuals(k, a, s, beta, c, X, ZZ) == (0, 0)
    # n=2 (CUB criterion) <=> companion in S  <=> E14 region membership
    comp_in_S = in_S(uc, vc)
    ok_n2 &= cub_n_is_2(u, v) == comp_in_S
    p = -u
    if comp_in_S:
        n_doublet += 1
        Ap = (12 * p * p - 3) * p; Aq = 12 * p * p - 3
        Lp = Aq / 2; Upv = 3 + 6 * p
        # V_+ involves a square root; test membership via H_p sign instead: inner iff Ap < v and H_p(v) > 0 (p<1)
        Hpv = -2 * v * v + (36 * p * p - 12 * p - 3) * v - 72 * p ** 3 + 36 * p * p + 18 * p - 9
        outer_pred = (Lp < v < Ap) if p < 1 else (Lp < v < Upv)
        inner_pred = (p < 1) and (Ap < v) and (Hpv > 0)
        ok_n2 &= (outer_pred != inner_pred) and (outer_pred or inner_pred)
        # E16: t_c < 0, L < 0, |t_c| > 1 iff v > Ap ; outer <=> |t_c| < 1
        Lv = 4 * v * v - 8 * b * u * v + b * b
        ok_16 &= tc < 0 and Lv < 0 and ((abs(tc) > 1) == (v > Ap)) and (outer_pred == (abs(tc) < 1))
        # E17 at the point
        ell = v / Aq
        ok_17 &= eta_of(uc, vc) - eta_of(u, v) == 2 * (ell - p) * (4 * p * ell - 1) ** 3 / (4 * ell * ell - 8 * p * ell + 1) ** 2
        ok_17 &= (eta_of(uc, vc) < eta_of(u, v)) == outer_pred      # farther (outer) anchor is deeper
        # E18: Q(T)|det DT| == Q |t_c|^11, exact derivatives
        U = Dual(u, 1, 0); V = Dual(v, 0, 1); Bd = 3 - 12 * U * U
        Ld = 4 * V * V - 8 * Bd * U * V + Bd * Bd
        Tc = (4 * V * V - Bd * Bd) / Ld
        Uc = (2 * Bd * V - U * (4 * V * V + Bd * Bd)) / Ld
        Vc = V * Tc * Tc
        det = Uc.du * Vc.dv - Uc.dv * Vc.du
        ok_18 &= Qf(uc, vc) * abs(det) == Qf(u, v) * abs(tc) ** 11
        ok_18 &= Qf(u, v) > 0 and Qf(uc, vc) > 0
print("points in S:", n_pts, "doublets:", n_doublet)
print("E12 involution (T∘T = id, t_c t_cc = 1):", ok_inv)
print("E13 identities:", ok_13, "| line and conic at Z'=t_c Z:", ok_line)
print("both roots are critical points of the actual cubic (M2) with R4/R5 jets:", ok_roots)
print("CUB n=2 criterion <=> companion in S <=> E14 outer/inner partition (exclusive, exhaustive):", ok_n2)
print("E16: t_c<0, L<0 on doublets; |t_c|>1 iff v>Ap; outer iff |t_c|<1:", ok_16)
print("E17 pointwise + ordering (outer anchor deeper):", ok_17)
print("E18 Q(T)|det DT| == Q|t_c|^11 (exact forward-mode derivatives):", ok_18)
# ---------- CUB criterion <=> (s>B and D^2<T) in jet variables, symbolic in (u,v) with Z=1,k=1 ----------
# s=-v, B=b, D=v-bu, T=-(s-B)^2(B+2s)/12 = (v+b)^2 (2v-b)/12 ; D^2<T <=> 12(v-bu)^2 < (v+b)^2(2v-b)
okT = True
for _ in range(300):
    u = rnd_frac(-1.5, 0.5); v = rnd_frac(0, 6); b = b0(u)
    s, B, D = -v, b, v - b * u
    T = -(s - B) ** 2 * (B + 2 * s) / 12
    okT &= (T == (v + b) ** 2 * (2 * v - b) / 12) and ((s > B) == (v < -b)) and ((D * D < T) == (12 * (v - b * u) ** 2 < (v + b) ** 2 * (2 * v - b)))
print("CUB (s>B, D^2<T) in (u,v) coordinates equals the E14 criterion:", okT)
# ---------- degree drop L=0 and tie |t_c|=1 are curves: L=0 has no solutions inside S2 (checked above: L<0), tie curve v = Ap ----------
print("tie curve: |t_c| = 1 <=> v = Ap (exact algebra: |t_c| = (4v^2-A^2)/(-L), equality iff 8v^2 = 8Apv).")
# ---------- extra: det DT == -t_c^4 and Q(T) == Q |t_c|^7 (identities stated in Codex review 5363731045) ----------
random.seed(7); okd = True; cnt = 0
while cnt < 400:
    u = rnd_frac(-1.5, 0.5); v = rnd_frac(0, 6)
    if not in_S(u, v): continue
    r = E12(u, v)
    if r is None: continue
    uc, vc, tc = r; cnt += 1
    U = Dual(u, 1, 0); V = Dual(v, 0, 1); Bd = 3 - 12 * U * U
    Ld = 4 * V * V - 8 * Bd * U * V + Bd * Bd
    Tc = (4 * V * V - Bd * Bd) / Ld
    Uc = (2 * Bd * V - U * (4 * V * V + Bd * Bd)) / Ld
    Vc = V * Tc * Tc
    det = Uc.du * Vc.dv - Uc.dv * Vc.du
    okd &= det == -tc ** 4
    okd &= Qf(uc, vc) == Qf(u, v) * abs(tc) ** 7   # note: on S the sign of Q is positive only inside S; use abs when companion outside S
print("det DT == -t_c^4 and Q(T) == Q |t_c|^7 at 400 exact points of S:", okd)
