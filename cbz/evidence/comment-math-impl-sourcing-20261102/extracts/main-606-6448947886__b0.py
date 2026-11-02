"""Symbolic checks of identities stated in #606 PROOF.md (weighted cusp sampling), my own code."""
import sys
sys.path.insert(0, '/tmp/claude-0/-home-user-main/ced74b96-60a0-5d59-b364-fb9a44030ace/scratchpad/pylib')
import sympy as sp

a, th, t, h, u, u0, p, x, L, Th, w, eta, g0, W1, c0, v0 = sp.symbols(
    'a theta t h u u0 p x L Theta w eta g0 W1 c0 v0', positive=True)
ok = {}

gamma = a / 2 + sp.Rational(5, 4)
beta = 2 * (th + 1) / 3
ac = 3 ** sp.Rational(-1, 4)
Kc = sp.integrate(1 - sp.sqrt(3) * w ** 2, (w, -ac, ac))
ok['Kc = 4 a_c/3 = 4/3^(5/4)'] = sp.simplify(Kc - 4 * ac / 3) == 0 and sp.simplify(Kc - 4 / 3 ** sp.Rational(5, 4)) == 0
Qc = Kc ** sp.Rational(1, 3) / (2 * ac)
ok['Q_c = Kc^(1/3)/(2a_c) = (2/3)Kc^(-2/3)'] = sp.simplify(Qc - sp.Rational(2, 3) * Kc ** sp.Rational(-2, 3)) == 0

# (2)->(18): v = eps u^(3/2) eta, Jacobian u^(3/2); u^a * u^(3/2) = u^(2 gamma - 1)
ok['a+3/2 = 2gamma-1'] = sp.simplify(a + sp.Rational(3, 2) - (2 * gamma - 1)) == 0

# (4) second form: u = sqrt(t/h), u^(a-1/2)|du/dh| = (1/2) t^(gamma-1) h^(-gamma)
uu = sp.sqrt(t / h)
lhs = uu ** (a - sp.Rational(1, 2)) * sp.Abs(sp.diff(uu, h))
ok['(4) substitution'] = sp.simplify(sp.powsimp(lhs / (sp.Rational(1, 2) * t ** (gamma - 1) * h ** (-gamma)), force=True)) == 1

# (4) first form from (18): for fixed u, h = kappa(eta), ell = u^2 h -> u^(2gamma-1) * u^(-2) = u^(a-1/2)
ok['(4) first-form exponent'] = sp.simplify((2 * gamma - 1 - 2) - (a - sp.Rational(1, 2))) == 0

# (5) cumulative: integral_0^min(u0, sqrt(t/k)) u^(2gamma-1) du = min(...)^(2gamma)/(2gamma)
k = sp.symbols('k', positive=True)
ok['(5) cumulative'] = sp.simplify(sp.integrate(u ** (2 * gamma - 1), (u, 0, sp.sqrt(t / k))) - (t / k) ** gamma / (2 * gamma)) == 0

# (6): exponent after dividing by t^(beta-1): a - 1/2 - 2(beta-1) = 2(gamma-beta) - 1
ok['(6) u exponent'] = sp.simplify(a - sp.Rational(1, 2) - 2 * (beta - 1) - (2 * (gamma - beta) - 1)) == 0

# R_eps(h) ~ c * (h/Kc)^(2theta/3) * Qc h^(-1/3) = r_eps h^(beta-1), r_eps = (2/3) c Kc^(-beta)
c = sp.symbols('c', positive=True)
R_lead = c * (h / Kc) ** (2 * th / 3) * Qc * h ** sp.Rational(-1, 3)
ok['(3),(17) r_eps and power'] = sp.simplify(sp.powsimp(R_lead / (sp.Rational(2, 3) * c * Kc ** (-beta) * h ** (beta - 1)), force=True)) == 1

# (8) integrability: w kappa^(-gamma) ~ d^(theta - 3gamma/2) integrable iff theta - 3gamma/2 > -1 iff gamma < beta
ok['(8) criterion'] = sp.simplify((th - 3 * gamma / 2 + 1) - sp.Rational(3, 2) * (beta - gamma)) == 0

# Section 6/9 constants
ok['Kc/(8 a_c^3) = 1/(2 sqrt 3)'] = sp.simplify(Kc / (8 * ac ** 3) - 1 / (2 * sp.sqrt(3))) == 0
A0 = 2 * g0 * ac * Kc ** sp.Rational(-1, 3)
ok['A0^(-3) = 1/(2 sqrt3 g0^3)'] = sp.simplify(A0 ** -3 - 1 / (2 * sp.sqrt(3) * g0 ** 3)) == 0
# (36): r = g0 sqrt(u) * 2 a_c Kc^(-1/3) h^(1/3), h = t x/u^2  ->  A0 (t x)^(1/3) u^(-1/6)
r36 = g0 * sp.sqrt(u) * 2 * ac * Kc ** sp.Rational(-1, 3) * (t * x / u ** 2) ** sp.Rational(1, 3)
ok['(36) r'] = sp.simplify(sp.powsimp(r36 / (A0 * (t * x) ** sp.Rational(1, 3) * u ** sp.Rational(-1, 6)), force=True)) == 1
# k3 = ell / r^3 = t x / r^3 -> A0^(-3) sqrt(u)
ok['(36) k3'] = sp.simplify(sp.powsimp(t * x / (A0 * (t * x) ** sp.Rational(1, 3) * u ** sp.Rational(-1, 6)) ** 3 / (A0 ** -3 * sp.sqrt(u)), force=True)) == 1
# edge k(u): ell = u^2 Kc d^(3/2), r = 2 g sqrt(u) a_c sqrt(d) -> ell/r^3 = sqrt(u)/(2 sqrt3 g^3)
d, g = sp.symbols('d g', positive=True)
ok['(24) k(u)'] = sp.simplify(u ** 2 * Kc * d ** sp.Rational(3, 2) / (2 * g * sp.sqrt(u) * ac * sp.sqrt(d)) ** 3 - sp.sqrt(u) / (2 * sp.sqrt(3) * g ** 3)) == 0
# tip (31): u = sqrt(t x / kappa), r = g0 sqrt(u) D -> r/t^(1/4) = g0 x^(1/4) D/kappa^(1/4); ell/r^4 = kappa/(g0^4 D^4)
kap, D = sp.symbols('kappa D', positive=True)
ut = sp.sqrt(t * x / kap)
rt = g0 * sp.sqrt(ut) * D
ok['(31) A4'] = sp.simplify(sp.powsimp(rt / t ** sp.Rational(1, 4) - g0 * x ** sp.Rational(1, 4) * D / kap ** sp.Rational(1, 4), force=True)) == 0
ok['(31) Q4'] = sp.simplify(t * x / rt ** 4 - kap / (g0 ** 4 * D ** 4)) == 0
ok['(31) k3/t^(1/4)'] = sp.simplify(sp.powsimp((t * x / rt ** 3) / t ** sp.Rational(1, 4) - x / (g0 * x ** sp.Rational(1, 4) * D / kap ** sp.Rational(1, 4)) ** 3, force=True)) == 0
# (24) g(u): y(s) = (2/omega) asin(s/sqrt(2W1)), g0 = y'(0); y'(-sqrt(u/3)) = g0 (1 - u/(6W1))^(-1/2)
om, s = sp.symbols('omega s', positive=True)
y = 2 / om * sp.asin(s / sp.sqrt(2 * W1))
gg0 = sp.diff(y, s).subs(s, 0)
ok['(24) g(u)'] = sp.simplify(sp.diff(y, s).subs(s, -sp.sqrt(u / 3)) - gg0 * (1 - u / (6 * W1)) ** sp.Rational(-1, 2)) == 0

# Section 9: tau = t/u0^2, L = log(1/tau), u = u0 exp(-L Theta): h = t x/u^2 = x exp(-L(1-2Theta))
tau = sp.exp(-L)
hh = (tau * u0 ** 2) * x / (u0 * sp.exp(-L * Th)) ** 2
ok['(32) h'] = sp.simplify(hh - x * sp.exp(-L * (1 - 2 * Th))) == 0
ok['(32) endpoint'] = sp.simplify(sp.solve(sp.Eq(x * sp.exp(-L * (1 - 2 * Th)), sp.Rational(1, 4)), Th)[0] - (sp.Rational(1, 2) - sp.log(4 * x) / (2 * L))) == 0
# J = 1/3 - Theta/6 on Theta in (0,1/2): range (1/4,1/3), density 2*6 = 12; H = Theta/2 = 1 - 3J
J = sp.Rational(1, 3) - Th / 6
ok['J range and density'] = (J.subs(Th, 0) == sp.Rational(1, 3) and J.subs(Th, sp.Rational(1, 2)) == sp.Rational(1, 4)
                             and 2 * abs(1 / sp.diff(J, Th)) == 12)
ok['H = 1 - 3J'] = sp.simplify(Th / 2 - (1 - 3 * J)) == 0

# Section 11 recovery: a = theta = 0, w = 1, p = (4 c0 eta_c/5) u0^(5/2)
etac = 2 / (3 * sp.sqrt(3))
g11, b11 = gamma.subs(a, 0), beta.subs(th, 0)
Wsum = 2 * etac
Z = u0 ** (2 * g11) * Wsum / (2 * g11)
p11 = 4 * c0 * etac / 5 * u0 ** sp.Rational(5, 2)
C11 = sp.simplify(p11 / Z)
ok['(11) C = c0'] = sp.simplify(C11 - c0) == 0
ok['(11) W area = p/c0'] = sp.simplify(2 * sp.integrate(etac * u ** sp.Rational(3, 2), (u, 0, u0)) - p11 / c0) == 0
Rsum = 2 * sp.Rational(2, 3) * Kc ** (-b11)
D11 = C11 * Rsum * u0 ** (2 * (g11 - b11)) / (2 * (g11 - b11))
Cedge = sp.Rational(8, 7) * c0 * Kc ** sp.Rational(-2, 3) * u0 ** sp.Rational(7, 6)
ok['(11) D = TWO C_edge'] = sp.simplify(D11 - Cedge) == 0
ok['(11) edge U density exponent 7/6'] = sp.simplify(2 * (g11 - b11) - sp.Rational(7, 6)) == 0
ok['(26) TWO mass = eta_c u0^(3/2)/(5 v0) with c0 = 1/(4u0v0)'] = sp.simplify(p11.subs(c0, 1 / (4 * u0 * v0)) - etac * u0 ** sp.Rational(3, 2) / (5 * v0)) == 0

# Section 12: a = 0, w = (eta_c - eta)^theta
Ws = 2 * sp.integrate((etac - eta) ** th, (eta, 0, etac))
ok['(41) Wsum'] = sp.simplify(Ws - 2 * etac ** (th + 1) / (th + 1)) == 0
Z12 = u0 ** sp.Rational(5, 2) * Ws / sp.Rational(5, 2)
Cth = sp.simplify(p / Z12)
ok['(41) C_theta'] = sp.simplify(sp.powsimp(sp.expand_power_base(
    Cth / (5 * p * (th + 1) / (4 * u0 ** sp.Rational(5, 2) * etac ** (th + 1))), force=True), force=True)) == 1
g12 = sp.Rational(5, 4)
Dedge = Cth * 2 * sp.Rational(2, 3) * Kc ** (-beta) * u0 ** (2 * (g12 - beta)) / (2 * (g12 - beta))
ok['(12) D_theta edge'] = sp.simplify(Dedge - 8 * Cth / (7 - 8 * th) * Kc ** (-2 * (th + 1) / 3) * u0 ** ((7 - 8 * th) / 6)) == 0
ok['(12) critical theta = 7/8'] = sp.solve(sp.Eq(beta, g12), th) == [sp.Rational(7, 8)]
ok['(12) D at 7/8'] = sp.simplify((Cth * 2 * sp.Rational(2, 3) * Kc ** (-beta) / 2).subs(th, sp.Rational(7, 8)) - (2 * Cth / 3 * Kc ** sp.Rational(-5, 4)).subs(th, sp.Rational(7, 8))) == 0
rows = {0: beta.subs(th, 0), sp.Rational(1, 2): beta.subs(th, sp.Rational(1, 2)), 1: beta.subs(th, 1)}
ok['(12) table exponents'] = (rows[0] == sp.Rational(2, 3) and rows[sp.Rational(1, 2)] == 1 and rows[1] == sp.Rational(4, 3) > g12)

for name, val in ok.items():
    print('PASS' if val else 'FAIL', name)
print('all', all(ok.values()), len(ok))
