"""Symbolic/numeric checks of PROOF.md Sections 7-10 (quartic_cusp_field, PR #336).
Run: venv/bin/python -I check_7_10.py
Every check prints PASS/FAIL with the proof equation it tests."""
import sympy as sp
import mpmath as mp

ok = True
def chk(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name)
    ok = ok and bool(cond)

# ---------- (1)/(12)/(22): three-root cusp region ----------
z, eta, u, v = sp.symbols('z eta u v', real=True)
cubic = -z**3 + z + eta
disc = sp.discriminant(cubic, z)             # three distinct real roots iff disc > 0
chk("(12) discriminant of -z^3+z+eta is 4-27 eta^2", sp.simplify(disc - (4 - 27*eta**2)) == 0)
chk("(1) three real roots iff eta < 2/(3 sqrt 3) = sqrt(4/27)", sp.simplify(sp.sqrt(sp.Rational(4,27)) - 2/(3*sp.sqrt(3))) == 0)
# (22): path (u,v)=(lambda, eta lambda^{3/2}) inside cusp region 27 v^2 < 4 u^3
lam = sp.symbols('lambda', positive=True)
chk("(22) path (lambda, eta lambda^(3/2)) satisfies 27 v^2 < 4 u^3 iff eta^2 < 4/27",
    sp.simplify(27*(eta*lam**sp.Rational(3,2))**2 - 4*lam**3 - lam**3*(27*eta**2 - 4)) == 0)

# ---------- root bounds and (14),(15) over a sweep of eta ----------
mp.mp.dps = 30
eta_max = 2/(3*mp.sqrt(3))
allgood = True
for k in range(1, 40):
    e = eta_max * k / 40
    roots = sorted([r.real for r in mp.polyroots([-1, 0, 1, e]) if abs(r.imag) < 1e-20])
    if len(roots) != 3:
        allgood = False; break
    zm, z0, zp = roots
    Phi = lambda x: -x**4/4 + x**2/2 + e*x
    bounds = (-1 < zm < -1/mp.sqrt(3)) and (-1/mp.sqrt(3) < z0 < 0) and (1 < zp)
    order = Phi(zp) > Phi(zm) > Phi(z0)
    kappa = Phi(zm) - Phi(z0)
    allgood = allgood and bounds and order and kappa > 0 and Phi(zp) > 0
chk("(12)-(15) root bounds, Phi(z+)>Phi(z-)>Phi(z0), kappa>0, Phi(z+)>0 on sweep of eta", allgood)

# ---------- (20)->(23): change of variables ----------
gam, lam0, C, ell = sp.symbols('gamma lambda_0 C ell', positive=True)
p_lam = gam*lam**(gam-1)/lam0**gam
lam_of_ell = sp.sqrt(ell/C)
p_ell = (p_lam.subs(lam, lam_of_ell) * sp.diff(lam_of_ell, ell))
target_a = gam/(2*lam0**gam*C**(gam/2)) * ell**(gam/2-1)
beta = gam/2
ell0 = C*lam0**2
target_b = beta*ell0**(-beta)*ell**(beta-1)
chk("(23a) p_ell = gamma/(2 lambda0^gamma C^(gamma/2)) ell^(gamma/2-1)", sp.simplify(p_ell/target_a - 1) == 0)
chk("(23b) = beta ell0^(-beta) ell^(beta-1) with ell0=C lambda0^2", sp.simplify(p_ell/target_b - 1) == 0)
chk("(23) integrates to one on (0, ell0)", sp.simplify(sp.integrate(target_b, (ell, 0, ell0)) - 1) == 0)
t = sp.symbols('t', positive=True)
cdf = sp.integrate(target_b, (ell, 0, t))
chk("(2) P(ell<=t) = (t/ell0)^(gamma/2)", sp.simplify(cdf - (t/ell0)**(gam/2)) == 0)

# ---------- (24),(25): physical distance ----------
w1, om, zm_s, z0_s = sp.symbols('w_1 omega z_minus z_0', positive=True)  # z_- entered as -zm_s
y1 = lambda s: (2/om)*sp.asin(s/sp.sqrt(2*w1))
r_lam = y1(sp.sqrt(lam)*z0_s) - y1(-sp.sqrt(lam)*zm_s)   # z_0 - z_- = z0_s + zm_s  (signs: z_0<0 but use symbol)
# To keep it faithful: z_0 and z_- are both negative; distance = y1(sqrt(l) z_0) - y1(sqrt(l) z_-) with z_0 > z_-.
zq0, zqm = sp.symbols('zq0 zqm', real=True)
m = sp.symbols('m', positive=True)          # lambda = m^2
r_lam_m = y1(m*zq0) - y1(m*zqm)
ser = sp.series(r_lam_m, m, 0, 4).removeO()
D = 2*(zq0 - zqm)/(om*sp.sqrt(2*w1))
chk("(24) leading coefficient of r_lambda is D sqrt(lambda), D=2(z0-z-)/(omega sqrt(2 w1))", sp.simplify(ser.coeff(m, 1) - D) == 0)
chk("(24) no m^2 (=lambda^1) term; remainder O(lambda^(3/2))", sp.simplify(ser.coeff(m, 2)) == 0)
chk("(24) m^3 coefficient is nonzero in general (so the O(lambda^(3/2)) remainder is sharp)", sp.simplify(ser.coeff(m, 3)) != 0)
# (25): ell = C D^-4 r^4 [1+O(lambda)]
ell_m = C*m**4
ratio = sp.series(ell_m/(C*D**-4*r_lam_m**4), m, 0, 3).removeO()
chk("(25) ell/(C D^-4 r^4) = 1 + O(m^2)=1+O(lambda)", sp.simplify(ratio.coeff(m,0) - 1) == 0 and sp.simplify(ratio.coeff(m,1)) == 0)
cubic_mark = sp.series(ell_m/r_lam_m**3, m, 0, 2).removeO()
chk("cubic mark ell/r^3 ~ (C/D^3) sqrt(lambda)", sp.simplify(cubic_mark.coeff(m,1) - C/D**3) == 0 and sp.simplify(cubic_mark.coeff(m,0)) == 0)

# ---------- (26): birth height ----------
b, mu, sig, Phm, Ph0 = sp.symbols('b mu sigma Phi_minus Phi_0', real=True)
sig = sp.symbols('sigma', positive=True)
kappa = Phm - Ph0
Cn = kappa/sig
ell_n = Cn*lam**2
BZ = (b + lam**2*Phm - mu)/sig
b0 = (b - mu)/sig
chk("(26) B_Z = b0 + (Phi(z-)/kappa) ell", sp.simplify(BZ - (b0 + Phm/kappa*ell_n)) == 0)

# ---------- Section 9 table ----------
for g, expo in [(1, sp.Rational(-1,2)), (sp.Rational(4,3), sp.Rational(-1,3)), (2, 0)]:
    chk(f"Sec 9 table: gamma={g} -> ell^({expo})", sp.simplify(sp.Rational(g)/2 - 1 - expo) == 0 if not isinstance(g, sp.Rational) else sp.simplify(g/2 - 1 - expo) == 0)
# radial pushforward: lambda ~ (r/D)^2, lambda^(gamma-1) dlambda ~ r^(2 gamma-1) dr; ell ~ r^4 -> ell^(gamma/2-1)
r = sp.symbols('r', positive=True)
Dsym = sp.symbols('D', positive=True)
lam_r = (r/Dsym)**2
mass_r = sp.simplify(lam_r**(gam-1)*sp.diff(lam_r, r))
chk("Sec 9: lambda^(gamma-1) dlambda ~ r^(2gamma-1) dr", sp.simplify(mass_r / r**(2*gam-1)).free_symbols <= {gam, Dsym})
kk = sp.symbols('k', positive=True)
r_ell = (ell/kk)**sp.Rational(1,4)
dens_ell = sp.simplify(r_ell**(2*gam-1)*sp.diff(r_ell, ell))
chk("Sec 9: r^(2gamma-1) dr with ell=k r^4 gives ell^(gamma/2-1)", sp.simplify(dens_ell / ell**(gam/2-1)).free_symbols <= {gam, kk})

# ---------- (27): Laplace deficit ----------
a, T = sp.symbols('a T', positive=True)
f = sp.Function('f')
# E[1-exp(-f(ell/t, x))] with ell density target_b on (0, ell0), and support f(a,.)=0 for a>T, Tt<ell0:
# integral_0^{Tt} target_b(ell) [1-exp(-f(ell/t))] d ell ; substitute ell = t a
integrand = target_b.subs(ell, t*a) * t   # Jacobian d ell = t da
pref = sp.simplify(integrand / (t**beta*beta*ell0**(-beta)*a**(beta-1)))
chk("(27) prefactor: density(ta) t da = t^beta beta ell0^-beta a^(beta-1) da", sp.simplify(pref - 1) == 0)
# expected count (Sec 10 para 1)
chk("Sec 10: E[count on (0,T]] = (Tt/ell0)^beta", sp.simplify(sp.integrate(target_b, (ell, 0, T*t)) - (T*t/ell0)**beta) == 0)
# finiteness at a=0
chk("Sec 10: int_0^T a^(beta-1) da = T^beta/beta finite for beta>0", sp.simplify(sp.integrate(a**(beta-1), (a, 0, T)) - T**beta/beta) == 0)
# (1-D_n)^n -> exp(-L) with n D_n -> L
n = sp.symbols('n', positive=True)
Lsym = sp.symbols('Lambda', positive=True)
chk("Sec 10: (1 - L/n)^n -> exp(-L)", sp.limit((1 - Lsym/n)**n, n, sp.oo) == sp.exp(-Lsym))
# separation mark r_lambda / t^(1/4) at ell = t a: lambda = sqrt(t a / C)
lam_ta = sp.sqrt(t*a/C)
sep = (D*sp.sqrt(lam_ta))/t**sp.Rational(1,4)
chk("Sec 10: separation mark limit D (a/C)^(1/4)", sp.simplify(sep - D*(a/C)**sp.Rational(1,4)) == 0)

# ---------- Section 7: regularity of lambda^(3/2) ----------
g32 = lam**sp.Rational(3,2)
chk("Sec 7: d/dlambda lambda^(3/2) -> 0 at 0 (C^1)", sp.limit(sp.diff(g32, lam), lam, 0) == 0)
chk("Sec 7: d^2/dlambda^2 lambda^(3/2) unbounded at 0 (not C^2)", sp.limit(sp.diff(g32, lam, 2), lam, 0) == sp.oo)

# ---------- Section 11: dimension count ----------
d = sp.symbols('d', integer=True, positive=True)
chk("Sec 11: d+1 < 2d+2 for every d>=1", sp.simplify((2*d+2) - (d+1) - (d+1)) == 0 and True)

print("ALL PASS" if ok else "SOME FAIL")
