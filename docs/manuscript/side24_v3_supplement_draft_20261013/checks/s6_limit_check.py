"""Exact checks of the displays of Section S6 (the near/far decomposition and the limit).

Run: python3 -I s6_limit_check.py   (prints ALL_CHECKS_PASS at the end)
Needs SymPy and mpmath. Draft 10 ships this script; the checks 1-9 are those of the draft-2
concordance (CONCORDANCE.md, S6 section), made into assertions; check 10 is new in draft 10
(the separation-independence of the canonical density version, after (S6.5)).
"""
import sympy as sp
import mpmath as mp

ok = []


def check(name, cond):
    if not bool(cond):
        raise AssertionError(name)
    ok.append(name)
    print('PASS', name)


r, k, l, b, s2, eps = sp.symbols('r kappa ell b sigma2 epsilon', positive=True)

# 1. lifetime pushforward (S6.3): r = (6 l / kappa)^(1/3) and r dr/dl = (6^(2/3)/3) kappa^(-2/3) l^(-1/3)
rl = (6 * l / k) ** sp.Rational(1, 3)
lhs = rl * sp.diff(rl, l)
rhs = sp.Rational(1, 3) * 6 ** sp.Rational(2, 3) * k ** sp.Rational(-2, 3) * l ** sp.Rational(-1, 3)
check('1 pushforward r dr = (6^(2/3)/3) kappa^(-2/3) ell^(-1/3) d ell', sp.simplify(lhs / rhs) == 1)

# 2. the prefactor (1/6)(6^(2/3)/3) = 6^(2/3)/18
check('2 (1/6)(6^(2/3)/3) = 6^(2/3)/18', sp.simplify(sp.Rational(1, 6) * 6 ** sp.Rational(2, 3) / 3 - 6 ** sp.Rational(2, 3) / 18) == 0)

# 3. K (r^3/6) r^2 = (r^4 K) r / 6
K = sp.Symbol('K', positive=True)
check('3 K (r^3/6) r^2 = G r/6 with G = r^4 K', sp.simplify(K * (r ** 3 / 6) * r ** 2 - (r ** 4 * K) * r / 6) == 0)

# 4. the kappa integral, (3.21) of the main text
I = sp.integrate(k ** sp.Rational(4, 3) * (2 * sp.pi * s2) ** sp.Rational(-1, 2) * sp.exp(-k ** 2 / (72 * s2)), (k, 0, sp.oo))
target = sp.gamma(sp.Rational(7, 6)) * 72 ** sp.Rational(7, 6) * s2 ** sp.Rational(2, 3) / (2 * sp.sqrt(2 * sp.pi))
check('4 kappa integral (3.21)', sp.simplify(I / target) == 1)

# 5. the constant of Corollary S6.5 equals B_3
C = sp.Rational(1, 18) * 6 ** sp.Rational(2, 3) * sp.gamma(sp.Rational(7, 6)) * 72 ** sp.Rational(7, 6) / (2 * sp.sqrt(2 * sp.pi)) * (2 * sp.pi) ** (-3)
B3 = 3 * sp.gamma(sp.Rational(7, 6)) / (2 ** sp.Rational(4, 3) * sp.pi ** sp.Rational(7, 2))
check('5 (6^(2/3)/18)(72^(7/6)/(2 sqrt(2 pi)))(2 pi)^(-3) Gamma(7/6) = B_3', sp.simplify(C / B3) == 1)

# 6. uniform injectivity of the corrected target: |y_r|^2 = b^2 - 2 eps b kappa + (eps^2 + 1/36) kappa^2, eps = r^3/12
Mq = sp.Matrix([[1, -eps], [-eps, eps ** 2 + sp.Rational(1, 36)]])
check('6a determinant of the target form is 1/36', sp.simplify(Mq.det() - sp.Rational(1, 36)) == 0)
check('6b trace of the target form is 37/36 + eps^2', sp.simplify(Mq.trace() - sp.Rational(37, 36) - eps ** 2) == 0)
lmin = (Mq.trace() - sp.sqrt(Mq.trace() ** 2 - 4 * Mq.det())) / 2
for e_val in [0, sp.Rational(1, 10), 1, 10, 144]:
    lm = sp.N(lmin.subs(eps, e_val), 30)
    bound = sp.N(sp.Rational(1, 36) / (1 + sp.Rational(1, 36) + e_val ** 2), 30)
    check('6c lambda_min >= det/trace at eps=%s' % e_val, lm >= bound)
check('6d eps = varrho^3/12 < 144 for varrho < 12', sp.Rational(12 ** 3, 12) == 144)

# 7. integrability of kappa^(-2/3)(1 + kappa^N) exp(-c kappa^2) on (0, oo), N = 6
c = sp.Symbol('c', positive=True)
J = sp.integrate(k ** sp.Rational(-2, 3) * (1 + k ** 6) * sp.exp(-c * k ** 2), (k, 0, sp.oo))
check('7 majorant integral finite: %s' % sp.simplify(J), J.is_finite is not False and not J.has(sp.oo))

# 8. corrected pin target on the pins: V+ = b - kappa r^3/12, V-corr = -kappa/6
fM = b
fS = b - k * r ** 3 / 6
check('8a V+ = b - kappa r^3/12', sp.simplify((fS + fM) / 2 - (b - k * r ** 3 / 12)) == 0)
check('8b V-corr = -kappa/6', sp.simplify((fS - fM) / r ** 3 + k / 6) == 0)

# 9. the corrected divided difference of a quartic has limit -g'''/12
t = sp.symbols('t')
a0, a1, a2, a3, a4 = sp.symbols('a0:5')
g = a0 + a1 * t + a2 * t ** 2 / 2 + a3 * t ** 3 / 6 + a4 * t ** 4 / 24
gp = sp.diff(g, t)
expr = (g.subs(t, r / 2) - g.subs(t, -r / 2) - (r / 2) * (gp.subs(t, r / 2) + gp.subs(t, -r / 2))) / r ** 3
check('9 V-corr of a quartic = -a3/12 exactly', sp.simplify(sp.expand(expr) + a3 / 12) == 0)

# 10. separation independence (draft 10, after (S6.5)): at fixed ell, kappa = 6 ell / r^3 turns
#     (6^(2/3)/18) ell^(-1/3) kappa^(-2/3) G_r d kappa, with G_r = r^4 K_r, into r^2 K_r dr.
kap = 6 * l / r ** 3
jac = sp.Abs(sp.diff(kap, r))
integrand = sp.Rational(1, 18) * 6 ** sp.Rational(2, 3) * l ** sp.Rational(-1, 3) * kap ** sp.Rational(-2, 3) * r ** 4 * jac
check('10a Jacobian identity (6^(2/3)/18) ell^(-1/3) kappa^(-2/3) r^4 |d kappa/dr| = r^2', sp.simplify(integrand - r ** 2) == 0)
# numerical test with a concrete kernel K_r(kappa) = (1 + r) exp(-kappa^2 - r^2): the near band
# kappa in (6 ell/rho'^3, 6 ell/rho^3] equals the far band rho <= r < rho' in pair coordinates.
mp.mp.dps = 40
ell0, rho, rhop = mp.mpf('0.003'), mp.mpf('0.7'), mp.mpf('1.9')
Kf = lambda rr, kk: (1 + rr) * mp.e ** (-kk ** 2 - rr ** 2)
rl_num = lambda kk: (6 * ell0 / kk) ** (mp.mpf(1) / 3)
near = mp.quad(lambda kk: mp.mpf(6) ** (mp.mpf(2) / 3) / 18 * ell0 ** (-mp.mpf(1) / 3) * kk ** (-mp.mpf(2) / 3)
               * rl_num(kk) ** 4 * Kf(rl_num(kk), kk), [6 * ell0 / rhop ** 3, 6 * ell0 / rho ** 3])
far = mp.quad(lambda rr: rr ** 2 * Kf(rr, 6 * ell0 / rr ** 3), [rho, rhop])
check('10b near band = far band numerically (rel. err %.1e)' % float(abs(near - far) / far), abs(near - far) < mp.mpf(10) ** -30 * far)

print('N_CHECKS', len(ok))
print('ALL_CHECKS_PASS')
