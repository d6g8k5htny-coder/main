"""Independent re-derivation of the C100 Collar Cubic Closure ledgers against P2.

Read-only support script. Run: venv/bin/python -I c100_check.py
"""
import sympy as sp

out = {}

# ---------------------------------------------------------------------------
# 1. Coalesced six-pin conditional law of (q,a,w,z) = (f_yy, f_xxy, f_xyy, f_yyy)
#    for planar BF covariance K(h)=exp(-|h|^2/2) (T -> infinity limit).
#    Pins in contact limit: (f, f_x, f_y, f_xx, f_xxx, f_xy) = (b,0,0,0,12k,0).
# ---------------------------------------------------------------------------
hx, hy = sp.symbols('hx hy', real=True)
K = sp.exp(-(hx**2 + hy**2) / 2)


def cov(al, be):
    # Cov(d^al f(0), d^be f(0)) = (-1)^{|be|} d^{al+be} K (0)
    n = (al[0] + be[0], al[1] + be[1])
    d = sp.diff(K, hx, n[0], hy, n[1]).subs({hx: 0, hy: 0})
    return (-1) ** (be[0] + be[1]) * d


obs = [(0, 0), (1, 0), (0, 1), (2, 0), (3, 0), (1, 1)]   # f, f_x, f_y, f_xx, f_xxx, f_xy
free = [(0, 2), (2, 1), (1, 2), (0, 3)]                  # q, a, w, z
b, k = sp.symbols('b k', real=True)
target = sp.Matrix([b, 0, 0, 0, 12 * k, 0])
S11 = sp.Matrix(len(obs), len(obs), lambda i, j: cov(obs[i], obs[j]))
S21 = sp.Matrix(len(free), len(obs), lambda i, j: cov(free[i], obs[j]))
S22 = sp.Matrix(len(free), len(free), lambda i, j: cov(free[i], free[j]))
condcov = sp.simplify(S22 - S21 * S11.inv() * S21.T)
condmean = sp.simplify(S21 * S11.inv() * target)
out['condcov_qawz'] = condcov.tolist()
out['condmean_qawz'] = list(condmean)

# ---------------------------------------------------------------------------
# 2. Scaled gradient G=(G1,G2): mean (X^2-1/4, -bY) at 6k=1; covariance; density.
# ---------------------------------------------------------------------------
X, Y = sp.symbols('X Y', real=True)
qs, as_, ws, zs = sp.symbols('q a w z', real=True)
G1 = X**2 - sp.Rational(1, 4) + as_ * X * Y + ws / 2 * Y**2
G2 = qs * Y
lin = sp.Matrix([[sp.diff(G1, v) for v in (qs, as_, ws, zs)],
                 [sp.diff(G2, v) for v in (qs, as_, ws, zs)]])
CovG = sp.simplify(lin * condcov * lin.T)
out['CovG'] = CovG.tolist()
out['detCovG'] = sp.factor(CovG.det())
m = sp.Matrix([X**2 - sp.Rational(1, 4), (-b) * Y])  # mean of G at 6k=1
expo = sp.simplify((m.T * CovG.inv() * m)[0] / 2)
out['density_exponent'] = sp.simplify(expo)
claimed_expo = (X**2 - sp.Rational(1, 4))**2 / (Y**2 * (4 * X**2 + Y**2)) + b**2 / 4
out['density_exponent_matches_C100'] = sp.simplify(expo - claimed_expo) == 0
pref = 1 / (2 * sp.pi * sp.sqrt(CovG.det()))
out['density_prefactor_matches_C100'] = sp.simplify(
    pref - 1 / (2 * sp.pi * Y**2 * sp.sqrt(4 * X**2 + Y**2))) == 0 if True else None

# ---------------------------------------------------------------------------
# 3. Cubic model P, station solve, collision factorization near M and S.
# ---------------------------------------------------------------------------
x, y = sp.symbols('x y', real=True)
a, Q, w, z = sp.symbols('a Q w z', real=True)
P = (x**3 / 3 - x / 4 - sp.Rational(1, 12)
     + a / 2 * (x**2 - sp.Rational(1, 4)) * y + Q / 2 * y**2 + w / 2 * x * y**2 + z / 6 * y**3)
out['P(-1/2,0)'] = P.subs({x: -sp.Rational(1, 2), y: 0})
out['P(1/2,0)'] = P.subs({x: sp.Rational(1, 2), y: 0})
out['Pxxx'] = sp.diff(P, x, 3)
Px, Py = sp.diff(P, x), sp.diff(P, y)
sol = sp.solve([Px.subs({x: X, y: Y}), Py.subs({x: X, y: Y})], [a, Q], dict=True)[0]
H = sp.hessian(P, (x, y))
dM = H.subs({x: -sp.Rational(1, 2), y: 0}).det()
dS = H.subs({x: sp.Rational(1, 2), y: 0}).det()
dY = H.subs({x: X, y: Y}).det()
prod = sp.together((dM * dS * dY).subs(sol))

rho, c, s = sp.symbols('rho c s', real=True)
for name, x0 in (('M', -sp.Rational(1, 2)), ('S', sp.Rational(1, 2))):
    pr = sp.factor(sp.together(prod.subs({X: x0 + rho * c, Y: rho * s})))
    num, den = sp.fraction(pr)
    # order of vanishing in rho of the numerator, and of the denominator
    vn = min(sp.Poly(num, rho).monoms(), key=lambda t: t[0])[0]
    vd = min(sp.Poly(den, rho).monoms(), key=lambda t: t[0])[0] if den.has(rho) else 0
    R = sp.simplify(pr / rho**2)
    lead = sp.factor(sp.limit(R, rho, 0))
    out[f'collision_{name}_rho_order'] = vn - vd
    out[f'collision_{name}_leading'] = lead
claimed_lead = -((-2 * c**2 + s**2 * w) * (-4 * c**3 + 3 * c * s**2 * w + s**3 * z)**2) / (4 * s**6)
out['collision_M_leading_matches_C100'] = sp.simplify(out['collision_M_leading'] - claimed_lead) == 0

# small-s behaviour of the claimed leading factor on the unit circle (c^2+s^2=1)
lead_axis = sp.limit(s**6 * claimed_lead.subs(c, sp.sqrt(1 - s**2)), s, 0)
out["claimed_leading_s6_times_limit_s0"] = lead_axis

# Forced jet a near M (station solve), to see the Gaussian exponent source
a_forced = sp.simplify(sol[a].subs({X: -sp.Rational(1, 2) + rho * c, Y: rho * s}))
out['a_forced_near_M_rho0'] = sp.simplify(sp.limit(a_forced, rho, 0))

# density exponent near M at rho->0
ex_M = sp.limit(((X**2 - sp.Rational(1, 4))**2 / (Y**2 * (4 * X**2 + Y**2))).subs(
    {X: -sp.Rational(1, 2) + rho * c, Y: rho * s}), rho, 0)
out['density_exponent_near_M_rho0'] = sp.simplify(ex_M)
out['rho2_times_density_prefactor_near_M'] = sp.simplify(sp.limit(
    rho**2 / (2 * sp.pi * Y**2 * sp.sqrt(4 * X**2 + Y**2)).subs({X: -sp.Rational(1, 2) + rho * c, Y: rho * s}),
    rho, 0))

# ---------------------------------------------------------------------------
# 4. Axis chart at X=0
# ---------------------------------------------------------------------------
wsol = sp.solve(Px.subs({x: 0, y: Y}), w)
out['X0_station_w'] = wsol
out['X0_gaussian_cost_exponent'] = sp.simplify(wsol[0]**2 / (2 * condcov[2, 2]))
out['X0_density_exponent'] = sp.simplify(((X**2 - sp.Rational(1, 4))**2 / (Y**2 * (4 * X**2 + Y**2))).subs(X, 0))
out['pair_axis_Px'] = sp.factor(Px.subs({y: 0}))

# ---------------------------------------------------------------------------
# 5. Power ledgers (exponent of r)
# ---------------------------------------------------------------------------
ledger = {
    'C100_generic': {'grad_jacobian': -3, 'three_hessians': 6, 'Z_r': -2},
    'P2_regionI_P19': {'Z_r(P14)': -2, 'grad_jacobian(P11)': -3, 'P16': 6, 'q^2/Delta^2<=1': 0},
    'P2_regionII_P20': {'Z_r': -2, 'grad_jacobian': -3, 'P17': 3, 'spatial_ratio': -2, 'chi^6 moment': -6},
}
out['ledger_totals'] = {kk: sum(v.values()) for kk, v in ledger.items()}
out['C100_count_exponent'] = out['ledger_totals']['C100_generic'] + 2  # dy = r^2 dxi

# Region II absorption: r^-10 e^{-c/(2r^2)} <= 6!(2/c)^6 r^2
rr, cc = sp.symbols('r c', positive=True)
uu = cc / (2 * rr**2)
out['regionII_absorption_identity'] = sp.simplify(rr**-10 * sp.factorial(6) / uu**6 - sp.factorial(6) * (2 / cc)**6 * rr**2) == 0

# Normalizer leading term: W ~ r^2 q^2 1{q<0} with q ~ N(-b,2): z_0 = E[q^2; q<0] > 0
qq = sp.symbols('qq', real=True)
z0 = sp.integrate(qq**2 * sp.exp(-(qq + b)**2 / 4) / sp.sqrt(4 * sp.pi), (qq, -sp.oo, 0))
out['z0(b)'] = sp.simplify(z0)
out['z0(b=6/5)'] = sp.N(z0.subs(b, sp.Rational(6, 5)), 12)

# P13 consistency: det H_M/r = -6k S0, det H_S/r = 6k S0 at 6k=1
Mm = H.subs({x: -sp.Rational(1, 2), y: 0})
out['scaledHess_M'] = Mm.tolist()

# Diagnostic: rho^2 * p_hat_0 at transverse station (c=0,s=1), rho=0.02, b=6/5
rv = sp.Rational(2, 100)
val = sp.exp(-sp.Rational(36, 100)) / (2 * sp.pi * sp.sqrt(1 + rv**2))
out['diag_rho2_phat0_transverse_b1.2'] = sp.N(val, 8)

for kk, v in out.items():
    print(f'{kk}: {v}')
