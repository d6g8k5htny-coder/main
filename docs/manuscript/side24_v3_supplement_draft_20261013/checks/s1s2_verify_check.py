"""Exact checks of the displays transcribed in supp_s1_setting.tex and supp_s2_contact.tex.

Run: python3 -I s1s2_verify_check.py   (prints ALL_CHECKS_PASS at the end)
Needs SymPy, mpmath and NumPy. This is the check named `chk_s1s2/verify_s1s2.py` in the
concordance (S1-S2 section); draft 10 ships it under this name, unchanged otherwise.
"""
import itertools
import sympy as sp
import mpmath as mp

ok = []


def check(name, cond):
    if not cond:
        raise AssertionError(name)
    ok.append(name)


# ---------------------------------------------------------------- S1: Poisson summation (C.1), 1-D factor
# sum_n exp(-(s+24n)^2/2) = (2pi)^{1/2}/24 * sum_m exp(-2 pi^2 m^2/24^2) exp(2 pi i m s/24); the 3-D identity is the product.
mp.mp.dps = 60
for s in [mp.mpf('0'), mp.mpf('0.37'), mp.mpf('5.1'), mp.mpf('11.9'), mp.mpf('-7.25')]:
    lhs = mp.nsum(lambda n: mp.e ** (-(s + 24 * n) ** 2 / 2), [-mp.inf, mp.inf])
    rhs = mp.sqrt(2 * mp.pi) / 24 * mp.nsum(
        lambda m: mp.e ** (-2 * mp.pi ** 2 * m ** 2 / 24 ** 2) * mp.cos(2 * mp.pi * m * s / 24), [-mp.inf, mp.inf])
    check('poisson_1d s=%s' % s, abs(lhs - rhs) < mp.mpf(10) ** -50)
# 3-D prefactor = product of three 1-D prefactors
check('poisson_prefactor_3d', sp.simplify((sp.sqrt(2 * sp.pi) / 24) ** 3 - (2 * sp.pi) ** sp.Rational(3, 2) / 24 ** 3) == 0)
# K_24(0)=1: Z_24 = sum_n exp(-288|n|^2) and |24 n|^2/2 = 288 |n|^2
n = sp.symbols('n')
check('K24_normalization', sp.expand((24 * n) ** 2 / 2 - 288 * n ** 2) == 0)

# ---------------------------------------------------------------- S1: det Cov of list (7) at the R^3 Bargmann-Fock kernel = 12
z1, z2, z3 = sp.symbols('z1 z2 z3', real=True)
Kbf = sp.exp(-(z1 ** 2 + z2 ** 2 + z3 ** 2) / 2)
Z = (z1, z2, z3)


def cov(a, b):
    # Cov(d^a f(x), d^b f(x)) = (-1)^{|b|} d^{a+b} K(0)
    expr = Kbf
    for i in range(3):
        expr = sp.diff(expr, Z[i], a[i] + b[i])
    return (-1) ** sum(b) * expr.subs({z1: 0, z2: 0, z3: 0})


# t = e1, v = e2, w = e3 (rotation invariance of the kernel)
lst = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1)]
C7 = sp.Matrix(8, 8, lambda i, j: cov(lst[i], lst[j]))
check('det_list7_BF_equals_12', sp.simplify(C7.det()) == 12)
check('list7_posdef_BF', all(ev > 0 for ev in C7.eigenvals()))
# with the factor -1/12 on d_t^3 f the corrected contact-limit covariance has determinant 12/144 = 1/12
Dsc = sp.diag(1, 1, 1, sp.Rational(-1, 12), 1, 1, 1, 1)
check('det_corrected_limit_BF', sp.simplify((Dsc * C7 * Dsc).det()) == sp.Rational(1, 12))
# remote block (8) with Hessian, at y = x + (3,0,0) (distance 3): positivity of the 18x18 covariance at the BF kernel


def cov2(a, xa, b, xb):
    expr = Kbf
    for i in range(3):
        expr = sp.diff(expr, Z[i], a[i] + b[i])
    d = [xa[i] - xb[i] for i in range(3)]
    return (-1) ** sum(b) * expr.subs({z1: d[0], z2: d[1], z3: d[2]})


rem = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)]
fam = [(a, (0, 0, 0)) for a in lst] + [(a, (3, 0, 0)) for a in rem]
C18 = sp.Matrix(18, 18, lambda i, j: cov2(fam[i][0], fam[i][1], fam[j][0], fam[j][1]))
C18n = sp.matrix2numpy(C18.evalf(30), dtype=float)
import numpy as np
check('list7_plus_remote_posdef_BF', np.linalg.eigvalsh(C18n).min() > 0)

# ---------------------------------------------------------------- S1: |det Theta| = 1 (block matrix of (x,u) -> (M,S))
for d in range(1, 6):
    I = sp.eye(d)
    Theta = sp.BlockMatrix([[I, -I / 2], [I, I / 2]]).as_explicit()
    check('det_Theta_d%d' % d, abs(Theta.det()) == 1)

# ---------------------------------------------------------------- S2: pin Jacobian |det Delta_r| = r^{-(d+3)} and target
r, b, kap = sp.symbols('r b kappa', positive=True)
for d in range(1, 6):
    # raw coordinates: f(M), f(S), then for e in frame: d_e f(M), d_e f(S); e=0 is t
    N = 2 * d + 2
    rows = []
    # V+ , V-corr
    rowVp = [sp.Rational(1, 2), sp.Rational(1, 2)] + [0] * (2 * d)
    rowVm = [-1 / r ** 3, 1 / r ** 3] + [-(r / 2) / r ** 3, -(r / 2) / r ** 3] + [0] * (2 * d - 2)
    rows += [rowVp, rowVm]
    for e in range(d):
        rp = [0, 0] + [0] * (2 * d)
        rm = [0, 0] + [0] * (2 * d)
        rp[2 + 2 * e] = sp.Rational(1, 2)
        rp[3 + 2 * e] = sp.Rational(1, 2)
        rm[2 + 2 * e] = -1 / r
        rm[3 + 2 * e] = 1 / r
        rows += [rp, rm]
    Delta = sp.Matrix(rows)
    check('det_Delta_d%d' % d, sp.simplify(Delta.det() - r ** (-(d + 3))) == 0)
    target_raw = sp.Matrix([b, b - kap * r ** 3 / 6] + [0] * (2 * d))
    y = sp.simplify(Delta * target_raw)
    check('target_d%d' % d, sp.simplify(y[0] - (b - kap * r ** 3 / 12)) == 0 and sp.simplify(y[1] + kap / 6) == 0
          and all(sp.simplify(y[i]) == 0 for i in range(2, N)))

# ---------------------------------------------------------------- S2: segment identities on a generic degree-7 polynomial
s = sp.symbols('s', real=True)
cs = sp.symbols('c0:8')
phi = sum(cs[k] * s ** k for k in range(8))
h = r / 2
Vminus = (phi.subs(s, h) - phi.subs(s, -h) - (r / 2) * (sp.diff(phi, s).subs(s, h) + sp.diff(phi, s).subs(s, -h))) / r ** 3
Vseg = -sp.integrate((r ** 2 / 4 - s ** 2) * sp.diff(phi, s, 3), (s, -h, h)) / (2 * r ** 3)
check('trapezoid_identity_Vminus', sp.simplify(Vminus - Vseg) == 0)
check('Vminus_limit', sp.simplify(sp.limit(Vminus, r, 0) + sp.diff(phi, s, 3).subs(s, 0) / 12) == 0)
check('weight_mass_r3_over_6', sp.simplify(sp.integrate(r ** 2 / 4 - s ** 2, (s, -h, h)) - r ** 3 / 6) == 0)
# Taylor identities (exact endpoint factor): for g with g(-h)=g(h)=0
g = phi  # generic; impose the two vanishing conditions by solving for c0, c1
sol = sp.solve([g.subs(s, h), g.subs(s, -h)], [cs[0], cs[1]], dict=True)[0]
g0 = sp.expand(g.subs(sol))
lhsM = sp.diff(g0, s).subs(s, -h)
rhsM = -sp.integrate((h - s) * sp.diff(g0, s, 2), (s, -h, h)) / r
lhsS = sp.diff(g0, s).subs(s, h)
rhsS = sp.integrate((s + h) * sp.diff(g0, s, 2), (s, -h, h)) / r
check('taylor_identity_M', sp.simplify(lhsM - rhsM) == 0)
check('taylor_identity_S', sp.simplify(lhsS - rhsS) == 0)
check('weight_mass_r2_over_2', sp.simplify(sp.integrate(h - s, (s, -h, h)) - r ** 2 / 2) == 0
      and sp.simplify(sp.integrate(s + h, (s, -h, h)) - r ** 2 / 2) == 0)
# alpha_M -> -(1/2) g''(0) and alpha_S -> +(1/2) g''(0) for g = d_t f (here g'' plays d_t^3 f)
alphaM = rhsM / r
alphaS = rhsS / r
check('alphaM_limit', sp.simplify(sp.limit(alphaM, r, 0) + sp.diff(g0, s, 2).subs(s, 0) / 2) == 0)
check('alphaS_limit', sp.simplify(sp.limit(alphaS, r, 0) - sp.diff(g0, s, 2).subs(s, 0) / 2) == 0)
# under the limiting pin V-corr = -kappa/6, d_t^3 f(x) = 2 kappa, so alpha_M -> -kappa, alpha_S -> kappa
check('pin_gives_2kappa', sp.solve(sp.Eq(-sp.Symbol('f3') / 12, -kap / 6), sp.Symbol('f3'))[0] == 2 * kap)

# ---------------------------------------------------------------- S2: det H = r (alpha det Q - r beta^T adj(Q) beta)
for d in range(2, 5):
    m = d - 1
    al = sp.Symbol('alpha')
    be = sp.Matrix(sp.symbols('beta0:%d' % m))
    qs = sp.symbols('q0:%d' % (m * m))
    Q = sp.Matrix(m, m, lambda i, j: qs[min(i, j) * m + max(i, j)])
    H = sp.zeros(d, d)
    H[0, 0] = r * al
    for i in range(m):
        H[0, i + 1] = r * be[i]
        H[i + 1, 0] = r * be[i]
        for j in range(m):
            H[i + 1, j + 1] = Q[i, j]
    D = al * Q.det() - r * (be.T * Q.adjugate() * be)[0, 0]
    check('detH_equals_rD_d%d' % d, sp.expand(H.det() - r * D) == 0)
    check('D_degree_d%d' % d, sp.Poly(sp.expand(D), al, *be, *qs).total_degree() == d)

# ---------------------------------------------------------------- S2: power counts, lifetime pushforward, scalars
dd = sp.Symbol('d', positive=True, integer=True)
check('factor_general_d', sp.simplify(r ** 3 / 6 * r ** (-(dd + 3)) * r ** 2 - r ** (2 - dd) / 6) == 0)
check('radial_general_d', sp.simplify(r ** 3 / 6 * r ** (-(dd + 3)) * r ** 2 * r ** (dd - 1) - r / 6) == 0)
check('powercount_d3', sp.simplify(r ** 3 / 6 * r ** -6 * r ** 2 * r ** 2 - r / 6) == 0)
pc, zc = sp.symbols('p zeta', positive=True)
J = r ** (dd - 2) * (r ** 3 / 6) * (r ** (-(dd + 3)) * pc) * (r ** 2 * zc)
check('J_exact', sp.simplify(J - pc * zc / 6) == 0)
check('radial_identity_15', sp.simplify((r ** 3 / 6) * sp.Symbol('K') * r ** (dd - 1) - r ** (dd - 2) * (r ** 3 / 6) * sp.Symbol('K') * r) == 0)
check('J_equals_Gall_over_6_d3', sp.simplify(r ** (3 - 2) * (r ** 3 / 6) * sp.Symbol('K') - r ** 4 * sp.Symbol('K') / 6) == 0)
ell = sp.Symbol('ell', positive=True)
rl = (6 * ell / kap) ** sp.Rational(1, 3)
check('ell_of_r', sp.simplify(kap * rl ** 3 / 6 - ell) == 0)
drdl = sp.diff(rl, ell)
check('dr_formula', sp.simplify(drdl - (6 / kap) ** sp.Rational(1, 3) * sp.Rational(1, 3) * ell ** sp.Rational(-2, 3)) == 0)
check('rdr_formula', sp.simplify(rl * drdl - 6 ** sp.Rational(2, 3) / 3 * kap ** sp.Rational(-2, 3) * ell ** sp.Rational(-1, 3)) == 0)
# inverse-derivative check: d ell/dr = kappa r^2/2, r dr = 2/(kappa r) d ell
check('inverse_derivative', sp.simplify(sp.diff(kap * r ** 3 / 6, r) - kap * r ** 2 / 2) == 0)
check('inverse_check_substituted', sp.simplify(2 / (kap * rl) - 6 ** sp.Rational(2, 3) / 3 * kap ** sp.Rational(-2, 3) * ell ** sp.Rational(-1, 3)) == 0)
check('scalar_6_23_over_18', sp.simplify(sp.Rational(1, 6) * 6 ** sp.Rational(2, 3) / 3 - 6 ** sp.Rational(2, 3) / 18) == 0)
check('scalar_d3_from_J', sp.simplify(6 ** sp.Rational(2, 3) / 3 * sp.Rational(1, 6) - 6 ** sp.Rational(2, 3) / 18) == 0)
# r < rho  <=> kappa > 6 ell / rho^3 at fixed ell
rho = sp.Symbol('varrho', positive=True)
check('cutoff', sp.simplify((6 * ell / kap) - rho ** 3).subs(kap, 6 * ell / rho ** 3) == 0)

# ---------------------------------------------------------------- S2: Sylvester/Schur congruence used for the type indicators
al = sp.Symbol('alpha')
be = sp.Matrix(sp.symbols('beta0:2'))
qs = sp.symbols('q0:3')
Q = sp.Matrix([[qs[0], qs[1]], [qs[1], qs[2]]])
H = sp.Matrix([[r * al, r * be[0], r * be[1]], [r * be[0], qs[0], qs[1]], [r * be[1], qs[1], qs[2]]])
Lm = sp.Matrix([[1, (r * be.T * Q.inv())[0], (r * be.T * Q.inv())[1]], [0, 1, 0], [0, 0, 1]])
schur = r * al - (r * be.T * Q.inv() * r * be)[0, 0]
check('congruence', sp.simplify(Lm * sp.diag(schur, Q) * Lm.T - H) == sp.zeros(3, 3))

print('\n'.join(ok))
print('N_CHECKS', len(ok))
print('ALL_CHECKS_PASS')
