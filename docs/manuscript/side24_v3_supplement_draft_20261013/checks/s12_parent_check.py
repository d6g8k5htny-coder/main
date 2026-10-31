"""Exact checks of finite identities transcribed in supp_s12_parent.tex / supp_s13_cap.tex.

Finite algebra only (SymPy); not a check of any analytic estimate. Run: python3 -I s12_parent_check.py
"""
import sympy as sp

ok = []


def check(name, cond):
    ok.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name)


r, k, b, x, t, tau = sp.symbols("r k b x t tau", positive=True)

# (S12.9): |det T_r| = 12 r^{-(d+3)}; axial block on (f(M), f_x(M), f(S), f_x(S)), transverse blocks on (f_y(M), f_y(S)).
Tax = sp.Matrix([
    [sp.Rational(1, 2), 0, sp.Rational(1, 2), 0],
    [-1 / r, 0, 1 / r, 0],
    [0, -1 / r, 0, 1 / r],
    [12 / r**3, 6 / r**2, -12 / r**3, 6 / r**2],
])
Ttr = sp.Matrix([[sp.Rational(1, 2), sp.Rational(1, 2)], [-1 / r, 1 / r]])
check("axial block |det| = 12/r^4", sp.simplify(abs(Tax.det()) - 12 / r**4) == 0)
check("transverse block det = 1/r", sp.simplify(Ttr.det() - 1 / r) == 0)
for d in range(2, 7):
    m = d - 1
    check(f"|det T_r| = 12 r^-(d+3), d={d}", sp.simplify(12 / r**4 * (1 / r)**m - 12 * r**(-(d + 3))) == 0)

# (S12.10): target v_r from the pins f(M)=b, f(S)=b-kr^3, f_x=0.
pins = sp.Matrix([b, 0, b - k * r**3, 0])
check("target v_r axial part", sp.simplify(Tax * pins - sp.Matrix([b - k * r**3 / 2, -k * r**2, 0, 12 * k])) == sp.zeros(4, 1))

# Generic quintic axial profile with critical pins, used for (S13.5) Hermite identity and (S12.16)-(S12.17) averaging.
c = sp.symbols("c0:6")
g = sum(c[i] * x**i for i in range(6))
a_, c_ = -r / 2, r / 2
sol = sp.solve([sp.diff(g, x).subs(x, a_), sp.diff(g, x).subs(x, c_)], [c[1], c[2]], dict=True)[0]
gp = g.subs(sol)
lhs = gp.subs(x, c_) - gp.subs(x, a_)
rhs = -sp.Rational(1, 2) * sp.integrate((x + r / 2) * (r / 2 - x) * sp.diff(gp, x, 3), (x, a_, c_))
check("Hermite identity (S13.5) on quintic with critical pins", sp.simplify(lhs - rhs) == 0)
check("kernel integral = r^3/6", sp.simplify(sp.integrate((x + r / 2) * (r / 2 - x), (x, a_, c_)) - r**3 / 6) == 0)
# averaging identities behind (S12.17): f_xx(a) = -(1/r) int (c-x) f_xxx, f_xx(c) = (1/r) int (x-a) f_xxx
check("f_xx(a) = -(1/r) int_a^c (c-x) f_xxx dx", sp.simplify(sp.diff(gp, x, 2).subs(x, a_) + sp.integrate((c_ - x) * sp.diff(gp, x, 3), (x, a_, c_)) / r) == 0)
check("f_xx(c) = (1/r) int_a^c (x-a) f_xxx dx", sp.simplify(sp.diff(gp, x, 2).subs(x, c_) - sp.integrate((x - a_) * sp.diff(gp, x, 3), (x, a_, c_)) / r) == 0)
# constant f_xxx = 12k gives alpha_M = -6k, alpha_S = 6k exactly
check("alpha_M=-6k, alpha_S=6k for f_xxx=12k", sp.simplify(-(sp.integrate((c_ - x) * 12 * k, (x, a_, c_)) / r) / r + 6 * k) == 0
      and sp.simplify((sp.integrate((x - a_) * 12 * k, (x, a_, c_)) / r) / r - 6 * k) == 0)

# (S12.18)/(S12.21): block determinant and congruence normalization.
for m in (1, 2, 3):
    A = sp.Matrix(m, m, lambda i, j: sp.Symbol(f"A{min(i, j)}{max(i, j)}"))
    beta = sp.Matrix(m, 1, lambda i, j: sp.Symbol(f"be{i}"))
    alpha = sp.Symbol("alpha")
    H = sp.Matrix(sp.BlockMatrix([[sp.Matrix([[r * alpha]]), r * beta.T], [r * beta, A]]))
    check(f"(5.3) block identity m={m}", sp.expand(H.det() / r - (alpha * A.det() - r * (beta.T * A.adjugate() * beta)[0])) == 0)
    Dinv = sp.diag(1 / sp.sqrt(r), *([1] * m))
    K = Dinv * H * Dinv
    target = sp.Matrix(sp.BlockMatrix([[sp.Matrix([[alpha]]), sp.sqrt(r) * beta.T], [sp.sqrt(r) * beta, A]]))
    check(f"erratum congruence diag(r^-1/2,I) m={m}", sp.simplify(K - target) == sp.zeros(m + 1, m + 1)
          and sp.simplify(K.det() - H.det() / r) == 0)
    # the displayed (uncorrected) factor diag(sqrt r, I) does NOT give this matrix
    Dbad = sp.diag(sp.sqrt(r), *([1] * m))
    check(f"source factor diag(sqrt r,I) differs m={m}", sp.simplify(Dbad * H * Dbad - target) != sp.zeros(m + 1, m + 1))

# (S12.29): soft integral
Dst, Est, U = sp.symbols("D E U", positive=True)
lam = sp.symbols("lam", positive=True)
I73 = sp.integrate(lam * (lam + Est * r * U), (lam, 0, Dst * r * U**2))
check("(7.3) integral", sp.simplify(I73 - r**3 * (Dst**3 / 3 * U**6 + Est * Dst**2 / 2 * U**5)) == 0)

# (S12.31): split threshold: lam <= 1/(4Dr) and lam <= 2DrJ^2 + 2Dr lam^2  =>  lam <= 4DrJ^2
J = sp.symbols("J", positive=True)
check("(7.5) split: 2Dr lam^2 <= lam/2 when lam <= 1/(4Dr)", sp.simplify((2 * Dst * r * (1 / (4 * Dst * r))**2) - (1 / (4 * Dst * r)) / 2) == 0)

# (S12.46): b^2 <= 2(b - k r^3/2)^2 + k^2/2 for r<=1, via b^2 <= 2(b-u)^2 + 2u^2 with u = k r^3/2
u = k * r**3 / 2
check("(13.1) identity 2(b-u)^2+2u^2-b^2 = (b-2u)^2", sp.expand(2 * (b - u)**2 + 2 * u**2 - b**2 - (b - 2 * u)**2) == 0)
check("(13.1) 2u^2 = k^2 r^6/2 <= k^2/2 at r=1", sp.simplify(2 * u**2 - k**2 * r**6 / 2) == 0)

# (S12.39): r dr/dl
l = sp.symbols("l", positive=True)
rr = (l / k)**sp.Rational(1, 3)
check("(11.1) r dr/dl", sp.simplify(rr * sp.diff(rr, l) - sp.Rational(1, 3) * k**sp.Rational(-2, 3) * l**sp.Rational(-1, 3)) == 0)
check("(11.3)/(13.6) factor 12/3 = 4", sp.Rational(12, 3) == 4)
check("difference exponent l^{-1/3} r^3 k^{-2/3} = l^{2/3} k^{-5/3}", sp.simplify(l**sp.Rational(-1, 3) * (l / k) * k**sp.Rational(-2, 3) - l**sp.Rational(2, 3) * k**sp.Rational(-5, 3)) == 0)

# (S12.43)-(S12.45)
q = sp.symbols("q")
check("(12.1) int_0^t l^{-1/3} = (3/2) t^{2/3}", sp.simplify(sp.integrate(l**sp.Rational(-1, 3), (l, 0, t)) - sp.Rational(3, 2) * t**sp.Rational(2, 3)) == 0)
check("(12.1) int_0^t l^{2/3} = (3/5) t^{5/3}", sp.simplify(sp.integrate(l**sp.Rational(2, 3), (l, 0, t)) - sp.Rational(3, 5) * t**sp.Rational(5, 3)) == 0)

# (S12.56)-(S12.58): gamma factor
phi = sp.exp(-(12 * k)**2 / (2 * tau**2)) / (tau * sp.sqrt(2 * sp.pi))
val = 144 * sp.integrate(k**sp.Rational(4, 3) * phi, (k, 0, sp.oo))
check("(15.2) gamma factor", sp.simplify(val - sp.gamma(sp.Rational(7, 6)) * tau**sp.Rational(4, 3) / (24**sp.Rational(1, 3) * sp.sqrt(sp.pi))) == 0)
check("4*(6k)^2 k^{-2/3} = 144 k^{4/3}", sp.simplify(4 * (6 * k)**2 * k**sp.Rational(-2, 3) - 144 * k**sp.Rational(4, 3)) == 0)
# amplitude scaling a^{-2/3}: p_G ~ a^-d, p_V ~ a^-d, tau^{4/3} ~ a^{4/3}, D ~ a^{2m}
dd, aa = sp.symbols("d a", positive=True)
check("amplitude exponent -2/3", sp.simplify(-dd - dd + sp.Rational(4, 3) + 2 * (dd - 1) + sp.Rational(2, 3)) == 0)

# Cap constants (S13)
qq = sp.Integer(11)
omega = 2 / qq + 2 / qq**2
M3 = (qq + 5) / 8
check("omega(11) = 24/121", omega == sp.Rational(24, 121))
check("M3*omega(11) = 48/121", M3 * omega == sp.Rational(48, 121))
qs = sp.symbols("q", positive=True)
check("M3*omega = (1+6/q+5/q^2)/4", sp.simplify((qs + 5) / 8 * (2 / qs + 2 / qs**2) - (1 + 6 / qs + 5 / qs**2) / 4) == 0)
adverse = sp.Rational(48, 121) * (3 + 3 * sp.Rational(24, 121) + sp.Rational(24, 121)**2)
check("adverse = 2554128/1771561", adverse == sp.Rational(2554128, 1771561))
check("7/4 - adverse = 2184415/7086244 > 1/4", sp.Rational(7, 4) - adverse == sp.Rational(2184415, 7086244) and sp.Rational(2184415, 7086244) > sp.Rational(1, 4))
check("int_{-r/2}^{r/2}|2r-t| dt = 2r^2", sp.simplify(sp.integrate(2 * r - t, (t, -r / 2, r / 2)) - 2 * r**2) == 0)
check("|x^2-r^2/4| <= 15r^2/4 on |x|<=2r (at x=2r)", sp.simplify((4 * r**2 - r**2 / 4) - sp.Rational(15, 4) * r**2) == 0)
I13 = sp.integrate((x + r / 2) * (x - r / 2) / 8, (x, -2 * r, -r / 2))
check("(13) exterior integral = 9r^3/32", sp.simplify(I13 - sp.Rational(9, 32) * r**3) == 0)
check("(13) b - 9/32 r^3 = s - 11/96 r^3 with s=b-r^3/6", sp.simplify((b - sp.Rational(9, 32) * r**3) - (b - r**3 / 6 - sp.Rational(11, 96) * r**3)) == 0)
check("(14) 11r*(20r/11)^2 = 4400/121 r^3", sp.simplify(11 * r * (20 * r / 11)**2 - sp.Rational(4400, 121) * r**3) == 0)
check("(14) 4400/121 > 1/6", sp.Rational(4400, 121) > sp.Rational(1, 6))
check("scaling: (9/32)*6 = 27/16, (11/96)*6 = 11/16, (4400/121)*6 = 26400/121, (1/4)*6 = 3/2",
      sp.Rational(9, 32) * 6 == sp.Rational(27, 16) and sp.Rational(11, 96) * 6 == sp.Rational(11, 16)
      and sp.Rational(4400, 121) * 6 == sp.Rational(26400, 121) and sp.Rational(1, 4) * 6 == sp.Rational(3, 2))
kap = sp.symbols("kappa", positive=True)
check("scaling of (4): 8 r (M3/6k)^2 < lam/6k  <=>  lam > 4/(3k) r M3^2", sp.simplify(8 * 6 * kap / 36 / kap**2 - sp.Rational(4, 3) / kap) == 0)
check("scaling of (4): r M4/(6k) <= 1/20  <=>  r M4 <= 3k/10", sp.Rational(6, 20) == sp.Rational(3, 10))

# --- Reconciled reading (C3, C4 and the REC Section 5 added steps) ---
# C4: the cylinder [-2r,2r] x closed ball(0,2r) lies in the closed ball of radius 2*sqrt(2)*r (its farthest points),
# and 2*sqrt(2)*r < L/2  <=>  r < L/(4*sqrt(2)).
Lsym = sp.symbols("L", positive=True)
check("C4: farthest point (2r, 2r e) has norm 2*sqrt(2)*r", sp.simplify(sp.sqrt((2 * r)**2 + (2 * r)**2) - 2 * sp.sqrt(2) * r) == 0)
check("C4: 2 sqrt2 r = L/2 exactly at r = L/(4 sqrt2)", sp.simplify((2 * sp.sqrt(2) * r).subs(r, Lsym / (4 * sp.sqrt(2))) - Lsym / 2) == 0)
# REC A5: det H_M = r alpha det B' with B' = A - (r/alpha) beta beta^T; and B - (-B') = (r/(-alpha)) beta beta^T (PSD for alpha<0).
for m in (1, 2, 3):
    A = sp.Matrix(m, m, lambda i, j: sp.Symbol(f"A{min(i, j)}{max(i, j)}"))
    beta = sp.Matrix(m, 1, lambda i, j: sp.Symbol(f"be{i}"))
    alpha = sp.Symbol("alpha", nonzero=True)
    H = sp.Matrix(sp.BlockMatrix([[sp.Matrix([[r * alpha]]), r * beta.T], [r * beta, A]]))
    Bp = A - (r / alpha) * beta * beta.T
    check(f"REC A5: det H_M = r alpha det B' (m={m})", sp.simplify(H.det() - r * alpha * Bp.det()) == 0)
    check(f"REC A5: B - (-B') = (r/(-alpha)) beta beta^T (m={m})", sp.simplify((-A) - (-Bp) - (r / (-alpha)) * beta * beta.T) == sp.zeros(m, m))
    # C3 (W1): the two difference matrices have entries alpha_M+6k, alpha_S-6k, sqrt(r) beta_i, A_S - A_M
    AM = A
    AS = sp.Matrix(m, m, lambda i, j: sp.Symbol(f"S{min(i, j)}{max(i, j)}"))
    aM, aS = sp.symbols("aM aS")
    bM = sp.Matrix(m, 1, lambda i, j: sp.Symbol(f"bM{i}"))
    bS = sp.Matrix(m, 1, lambda i, j: sp.Symbol(f"bS{i}"))
    HM = sp.Matrix(sp.BlockMatrix([[sp.Matrix([[r * aM]]), r * bM.T], [r * bM, AM]]))
    HS = sp.Matrix(sp.BlockMatrix([[sp.Matrix([[r * aS]]), r * bS.T], [r * bS, AS]]))
    Dinv = sp.diag(1 / sp.sqrt(r), *([1] * m))
    diffM = Dinv * HM * Dinv - sp.diag(-6 * k, AM)
    diffS = Dinv * HS * Dinv - sp.diag(6 * k, AM)
    expM = sp.Matrix(sp.BlockMatrix([[sp.Matrix([[aM + 6 * k]]), sp.sqrt(r) * bM.T], [sp.sqrt(r) * bM, sp.zeros(m, m)]]))
    expS = sp.Matrix(sp.BlockMatrix([[sp.Matrix([[aS - 6 * k]]), sp.sqrt(r) * bS.T], [sp.sqrt(r) * bS, AS - AM]]))
    check(f"C3 (W1): difference matrices (m={m})", sp.simplify(diffM - expM) == sp.zeros(m + 1, m + 1) and sp.simplify(diffS - expS) == sp.zeros(m + 1, m + 1))

print("ALL PASS" if all(v for _, v in ok) else "SOME FAILED", f"({sum(v for _, v in ok)}/{len(ok)})")
