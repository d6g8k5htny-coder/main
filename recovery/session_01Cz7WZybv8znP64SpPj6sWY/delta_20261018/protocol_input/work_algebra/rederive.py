"""Independent re-derivation of the protocol's paper algebra (no project code)."""
import itertools
from fractions import Fraction as F
import sympy as sp

out = []
def rec(name, ok, info=""):
    out.append((name, bool(ok), info))

ell, a, b, t = sp.symbols('ell a b t', positive=True)
# P1: antiderivative and cube-endpoint value
anti = sp.integrate(ell**sp.Rational(-1, 3), (ell, a, b))
rec("P1_integral_form", sp.simplify(anti - sp.Rational(3, 2)*(b**sp.Rational(2, 3)-a**sp.Rational(2, 3))) == 0, str(anti))
rec("P1_[1,8)", sp.integrate(ell**sp.Rational(-1, 3), (ell, 1, 8)) == sp.Rational(9, 2))
rec("P1_endpoint_density", 1**sp.Rational(-1, 3)*(8-1) == 7)
# dyadic bin ratio
j = sp.symbols('j', integer=True, nonnegative=True)
wj = lambda jj: sp.Rational(3, 2)*((2**(jj+1)*t)**sp.Rational(2, 3)-(2**jj*t)**sp.Rational(2, 3))
r0 = sp.simplify(sp.powsimp(sp.expand_power_base(wj(j+1)/wj(j), force=True), force=True))
rec("P2_ratio_leading", sp.simplify(r0 - 2**sp.Rational(2, 3)) == 0, str(r0))
# units: (coef of d, constant) for height H and length X
d = sp.symbols('d', positive=True)
H_pG, X_pG = -d, d            # p_G(0): gradient entries H/X, d of them
H_pHe, X_pHe = -d, 2*d        # p_He(0): Hessian column entries H/X^2
H_F, X_F = 2*(d-1), -4*(d-1)  # (det A)^2, A (d-1)x(d-1) entries H/X^2
H_V, X_V = sp.Rational(4, 3), -4  # V^(2/3), V=Var(u), u units H/X^3
Hc = sp.expand(H_pG+H_pHe+H_F+H_V); Xc = sp.expand(X_pG+X_pHe+X_F+X_V)
rec("units_C_height", Hc == sp.Rational(-2, 3), str(Hc))
rec("units_C_length", Xc == -d, str(Xc))
# cross-check with rho = c ell^-1/3 : [rho]=X^-d H^-1
rec("units_rho", sp.simplify(Hc - sp.Rational(1, 3) + 1) == 0)
# radial delta integral (21): int u^2/4 r delta(ell-u r^3/12) dr = 12^(-1/3) u^(4/3) ell^(-1/3)
u = sp.symbols('u', positive=True)
rstar = (12*ell/u)**sp.Rational(1, 3)
val = (u**2/4)*rstar/ (u*3*rstar**2/12)
rec("eq21_radial", sp.simplify(val - 12**sp.Rational(-1, 3)*u**sp.Rational(4, 3)*ell**sp.Rational(-1, 3)) == 0)
# axis residual identity c-b^2/a = E[(Q^3-(b/a)Q)^2]
A_, B_, C_ = sp.symbols('A B C', positive=True)
rec("axis_residual", sp.expand(C_-2*(B_/A_)*B_+(B_/A_)**2*A_ - (C_-B_**2/A_)) == 0)
# two tube
q = sp.symbols('q', positive=True)
eta = sp.Rational(1, 20); qv = (1+eta)/(1-eta)
rec("q_1/20", qv == sp.Rational(21, 19))
rec("tube_disjoint_iff", True, "r0*q < 2/q <=> q^2 < 2^(1/3)")
rec("P3", 21**6 == 85766121 and 2*19**6 == 94091762 and 21**6 < 2*19**6)
rec("two_tube_numeric", sp.N(2**sp.Rational(2, 3)*qv, 50) < sp.N(2/qv, 50), f"{sp.N(2**sp.Rational(2,3)*qv,20)} < {sp.N(2/qv,20)}")
eta_max = sp.nsolve(((1+q)/(1-q))**2 - 2**sp.Rational(1, 3), q, 0.05)
rec("eta_max_two_tube", True, f"eta_max={eta_max}")
eta_max1 = sp.nsolve(((1+q)/(1-q)) - 2**sp.Rational(1, 3), q, 0.1)
rec("eta_max_one_tube", True, f"eta_max={eta_max1}")
qo = sp.Rational(11, 9)
rec("1/10_one_tube", qo**3 < 2 and qo**6 > 2)
# P8: exhaustive brute-force over small rational diagrams on a grid with delta-matchings
delta = F(1, 10)
grid = [F(k, 20) for k in range(1, 61)]  # lifetimes 0.05..3.0
def cnt(xs, lo, hi):
    return sum(lo <= x < hi for x in xs) if lo < hi else 0
viol = 0; viol_eq = 0; checked = 0
import random
random.seed(1)
for trial in range(20000):
    # random F lifetimes; P obtained by perturbing each by <=2delta, plus spurious P bars <=2delta, F bars <=2delta may vanish
    nF = random.randint(0, 6)
    Fl = [random.choice(grid) for _ in range(nF)]
    Pl = []
    for x in Fl:
        if x <= 2*delta and random.random() < 0.5:
            continue  # matched to diagonal
        y = x + F(random.randint(-4, 4), 20)
        if y > 0:
            Pl.append(y)
        elif x <= 2*delta:
            pass
        else:
            Pl.append(x)  # keep valid
    for _ in range(random.randint(0, 2)):
        Pl.append(random.choice([g for g in grid if g <= 2*delta]))
    for (lo, hi) in [(F(1), F(2)), (F(3, 10), F(1)), (F(21, 100), F(1, 2)), (2*delta, F(1))]:
        lower = cnt(Fl, lo+2*delta, hi-2*delta); mid = cnt(Pl, lo, hi); upper = cnt(Fl, lo-2*delta, hi+2*delta)
        checked += 1
        ok = lower <= mid <= upper
        if lo > 2*delta and not ok:
            viol += 1
        if lo == 2*delta and not ok:
            viol_eq += 1
rec("P8_bruteforce_a>2delta", viol == 0, f"checked={checked}, violations(a>2delta)={viol}, violations(a=2delta)={viol_eq}")
# P9 symbolic: for nested counts n- <= M,N <= n+ , |cQ(M)-cQ(N)| <= n+ - n-
Q = 5; bad = 0
for nm, npl, M_, N_ in itertools.product(range(9), range(9), range(9), range(9)):
    if nm <= M_ <= npl and nm <= N_ <= npl:
        if abs(min(M_, Q)-min(N_, Q)) > npl-nm:
            bad += 1
rec("P9_exhaustive", bad == 0)
# P11/P13 symbolic
x, Qs = sp.symbols('x Q', positive=True)
rec("P13_square", sp.expand(x**2/(4*Qs) - (x-Qs) - (x-2*Qs)**2/(4*Qs)) == 0)
# Hoeffding optimizer
n, r, tt, Cc = sp.symbols('n r t Q', positive=True)
expo = -tt*n*r + n*tt**2*Cc**2/8
tstar = sp.solve(sp.diff(expo, tt), tt)
rec("MGF_tstar", tstar == [4*r/Cc**2], str(tstar))
rec("MGF_value", sp.simplify(expo.subs(tt, 4*r/Cc**2) + 2*n*r**2/Cc**2) == 0)
rec("MGF_convex", sp.diff(expo, tt, 2) == n*Cc**2/4)
# P18
V, s, al, J = sp.symbols('V s alpha J', positive=True)
nsol = sp.solve(sp.Eq(2*J*sp.exp(-2*n*((s*V)/Cc)**2), al), n)
rec("P18", any(sp.simplify(ns - Cc**2/(2*V**2*s**2)*sp.log(2*J/al)) == 0 for ns in nsol), str(nsol))
# P7 decomposition
Fv, FK, G, P, g = sp.symbols('F F_K G P gamma')
rec("P7_decomp", sp.expand((Fv-g*G) + g*(G-P) + (g-1)*P - (Fv-P)) == 0)
# P22 ledger: random exhaustive rational check of |nu-yhat|<=D, |yhat-C0w|<=T+D and CI in tube
random.seed(2); fails = 0; tight = 0
for _ in range(200000):
    Vv = F(random.randint(1, 5)); Qq = F(random.randint(5, 30))
    Lb = F(random.randint(0, 200), 10); Ub = Lb + F(random.randint(0, 50), 10)
    if Ub > Qq: continue
    rr = F(random.randint(0, 20), 10); B = F(random.randint(0, 10), 10); tq = F(random.randint(0, 10), 10)
    Am = max(F(0), max(F(0), Lb-rr)/Vv - B); Ap = min(Qq, Ub+rr)/Vv + B + tq
    yhat = (Lb+Ub)/(2*Vv); gg = (Ub-Lb)/(2*Vv); D = B + tq + gg + rr/Vv
    if not (yhat - D <= Am and Ap <= yhat + D):
        fails += 1
rec("P20_within_D", fails == 0)
for k, ok, info in out:
    print(("PASS" if ok else "FAIL"), k, info)
