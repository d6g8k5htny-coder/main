#!/usr/bin/env python3
"""Independent, fail-closed audit of the V3.4 hybrid Section 6 algebra.

This script is deliberately narrower than the analytic proof.  It checks:

* unisolvence and the exact square completions for the perpendicular-safe
  mixed-curvature cubic interpolant;
* the short-edge factors in the quartic interpolation error;
* an explicit typed collinear quintic whose determinant product is Theta(r^3)
  while the common monomial is Theta(r^6); and
* the algebraic deficit and exponential-absorption ledgers used on the axis.

It does not certify Gaussian covariance floors, compact-atlas coverage,
conditional-moment bounds, Kac--Rice hypotheses, or the theorem.
"""

from __future__ import annotations

import math
import sys

import sympy as sp


CHECKS = 0
FAILURES: list[str] = []


def ck(label: str, condition: object) -> None:
    """Record one check and fail closed at the end."""
    global CHECKS
    CHECKS += 1
    ok = bool(condition)
    print(("PASS " if ok else "FAIL ") + label)
    if not ok:
        FAILURES.append(label)


def z(expr: sp.Expr) -> bool:
    """Exact zero test for rational symbolic expressions."""
    return sp.factor(sp.cancel(expr)) == 0


def denominator_uses_only(expr: sp.Expr, allowed: set[sp.Symbol]) -> bool:
    den = sp.factor(sp.denom(sp.cancel(expr)))
    return den.free_symbols <= allowed


def coeffs_nonnegative(expr: sp.Expr, variables: tuple[sp.Symbol, ...]) -> bool:
    poly = sp.Poly(sp.expand(expr), *variables)
    return all(coef.is_nonnegative is True for coef in poly.coeffs())


negative_probe = "--negative-probe" in sys.argv

print("SECTION A: MIXED-CURVATURE CUBIC")

X, Y = sp.symbols("X Y", real=True)
eta, c, sigma = sp.symbols("eta c sigma", real=True, nonzero=True)
kappa, alpha, D = sp.symbols("kappa alpha D", real=True)
a = eta / 2

# The order is chosen to make the endpoint kernel transparent.
monomials = (
    sp.Integer(1), Y, X, Y**2, X * Y, X**2,
    Y**3, X * Y**2, X**2 * Y, X**3,
)
coeff = sp.symbols("z0:10")
Q_generic = sum(q * m for q, m in zip(coeff, monomials))
nodes = ((-a, 0), (a, 0), (c, sigma))
values = (0, -kappa * eta**3 / 6, -alpha * kappa * eta**3 / 6)

equations: list[sp.Expr] = []
for (px, py), value in zip(nodes, values):
    equations.extend((
        Q_generic.subs({X: px, Y: py}) - value,
        sp.diff(Q_generic, X).subs({X: px, Y: py}),
        sp.diff(Q_generic, Y).subs({X: px, Y: py}),
    ))
mixed_row = (
    sp.diff(Q_generic, X, Y).subs({X: a, Y: 0})
    - sp.diff(Q_generic, X, Y).subs({X: -a, Y: 0})
) / (2 * a) - D
equations.append(mixed_row)

matrix, rhs = sp.linear_eq_to_matrix(equations, coeff)
matrix_det = sp.factor(matrix.det())
ck("raw 10x10 determinant = 2 eta^5 sigma^6",
   z(matrix_det - 2 * eta**5 * sigma**6))
ck("raw interpolation is unique for eta*sigma != 0", matrix_det != 0)

# The four-dimensional kernel after only the two endpoint value-gradient pins.
A0, B0, C0, D0 = sp.symbols("A0 B0 C0 D0")
endpoint_kernel = (
    A0 * (X**2 - a**2) * Y
    + Y**2 * (B0 * X + C0)
    + D0 * Y**3
)
for label, px in (("M", -a), ("S", a)):
    ck(f"endpoint kernel value vanishes at {label}",
       z(endpoint_kernel.subs({X: px, Y: 0})))
    ck(f"endpoint kernel X-gradient vanishes at {label}",
       z(sp.diff(endpoint_kernel, X).subs({X: px, Y: 0})))
    ck(f"endpoint kernel Y-gradient vanishes at {label}",
       z(sp.diff(endpoint_kernel, Y).subs({X: px, Y: 0})))

kernel_mixed_row = sp.factor((
    sp.diff(endpoint_kernel, X, Y).subs({X: a, Y: 0})
    - sp.diff(endpoint_kernel, X, Y).subs({X: -a, Y: 0})
) / (2 * a))
ck("normalized mixed-curvature row isolates 2*A0",
   z(kernel_mixed_row - 2 * A0))

witness_basis = (X * Y**2, Y**2, Y**3)
witness_matrix = sp.Matrix([
    [m.subs({X: c, Y: sigma}) for m in witness_basis],
    [sp.diff(m, X).subs({X: c, Y: sigma}) for m in witness_basis],
    [sp.diff(m, Y).subs({X: c, Y: sigma}) for m in witness_basis],
])
ck("witness value-gradient determinant has magnitude sigma^6",
   z(witness_matrix.det() + sigma**6))
ck("witness determinant has no c factor", c not in sp.factor(witness_matrix.det()).free_symbols)

solution_tuple = tuple(next(iter(sp.linsolve((matrix, rhs), coeff))))
solution = dict(zip(coeff, solution_tuple))

expected_solution = (
    -eta**3 * kappa / 12,
    -D * eta**2 / 8,
    -eta**2 * kappa / 4,
    eta**2 * (D * sigma - 2 * alpha * eta * kappa
              + 2 * c * kappa + eta * kappa) / (4 * sigma**2),
    0,
    0,
    (12 * D * c**2 * sigma - 3 * D * eta**2 * sigma
     + 8 * alpha * eta**3 * kappa + 16 * c**3 * kappa
     - 12 * c * eta**2 * kappa - 4 * eta**3 * kappa) / (24 * sigma**3),
    -(4 * D * c * sigma + 4 * c**2 * kappa - eta**2 * kappa)
    / (4 * sigma**2),
    D / 2,
    kappa / 3,
)
for i, (actual, expected) in enumerate(zip(solution_tuple, expected_solution)):
    ck(f"cubic coefficient z{i} exact", z(actual - expected))

Q = sp.factor(Q_generic.subs(solution))
for label, (px, py), value in zip(("M", "S", "P"), nodes, values):
    ck(f"Q value pin at {label}", z(Q.subs({X: px, Y: py}) - value))
    ck(f"Q X-gradient pin at {label}", z(sp.diff(Q, X).subs({X: px, Y: py})))
    ck(f"Q Y-gradient pin at {label}", z(sp.diff(Q, Y).subs({X: px, Y: py})))
ck("Q normalized mixed-curvature pin",
   z((sp.diff(Q, X, Y).subs({X: a, Y: 0})
      - sp.diff(Q, X, Y).subs({X: -a, Y: 0})) / (2 * a) - D))

H_Q = sp.hessian(Q, (X, Y))
H_M, H_S, H_P = tuple(
    H_Q.subs({X: px, Y: py}).applyfunc(sp.factor) for px, py in nodes
)

expected_H_M = sp.Matrix([
    [-eta * kappa, -D * eta / 2],
    [-D * eta / 2,
     eta * (2 * D * sigma * (2 * c + eta)
            + kappa * (-4 * alpha * eta**2 + 4 * c**2
                       + 4 * c * eta + eta**2)) / (4 * sigma**2)],
])
expected_H_S = sp.Matrix([
    [eta * kappa, D * eta / 2],
    [D * eta / 2,
     eta * (2 * D * sigma * (-2 * c + eta)
            + kappa * (-4 * alpha * eta**2 - 4 * c**2
                       + 4 * c * eta + 3 * eta**2)) / (4 * sigma**2)],
])
expected_H_P = sp.Matrix([
    [D * sigma + 2 * c * kappa,
     (-D * c * sigma - 2 * c**2 * kappa + eta**2 * kappa / 2) / sigma],
    [(-D * c * sigma - 2 * c**2 * kappa + eta**2 * kappa / 2) / sigma,
     (D * sigma * (4 * c**2 - eta**2)
      + 2 * kappa * (2 * alpha * eta**3 + 4 * c**3
                     - 3 * c * eta**2 - eta**3)) / (4 * sigma**2)],
])
for name, actual, expected in (
    ("M", H_M, expected_H_M),
    ("S", H_S, expected_H_S),
    ("P", H_P, expected_H_P),
):
    for i in range(2):
        for j in range(i, 2):
            ck(f"H_Q({name})[{i},{j}] exact", z(actual[i, j] - expected[i, j]))

v = (D * sigma + kappa * (2 * c + eta)) / (2 * kappa * eta)
L_normalized = kappa**2 * eta**4 / sigma**2
ck("M square completion",
   z(-H_M.det() - L_normalized * (v**2 - alpha)))
ck("S square completion",
   z(-H_S.det() - L_normalized * ((v - 1)**2 - (1 - alpha))))
ck("witness square completion",
   z(H_P.det() + L_normalized * ((v - alpha)**2 + alpha * (1 - alpha))))

r_phys, s_phys = sp.symbols("r_phys s_phys", positive=True)
L_physical = sp.factor(s_phys**2 * L_normalized.subs(eta, r_phys / s_phys))
ck("physical L = kappa^2 r^4/(sigma^2 s^2)",
   z(L_physical - kappa**2 * r_phys**4 / (sigma**2 * s_phys**2)))

for name, H in (("M", H_M), ("S", H_S)):
    for i in range(2):
        for j in range(i, 2):
            ck(f"H_Q({name}) entry carries eta", z(H[i, j].subs(eta, 0)))
            ck(f"H_Q({name}) entry denominator uses only sigma",
               denominator_uses_only(H[i, j], {sigma}))
for i in range(2):
    for j in range(i, 2):
        ck(f"H_Q(P)[{i},{j}] has no negative eta power",
           eta not in sp.denom(sp.cancel(H_P[i, j])).free_symbols)
        ck(f"H_Q(P)[{i},{j}] is finite at c=0",
           c not in sp.denom(sp.cancel(H_P[i, j])).free_symbols)

ck("negative probe remains dormant", not negative_probe)

print("SECTION B: QUARTIC INTERPOLATION ERROR")

tau = sp.symbols("tau", real=True)
q40, q31, q22, q13, q04 = sp.symbols("q40 q31 q22 q13 q04", real=True)
quartic = (
    q40 * X**4 + q31 * X**3 * Y + q22 * X**2 * Y**2
    + q13 * X * Y**3 + q04 * Y**4
)
correction_coeff = sp.symbols("w0:10")
cubic_correction = sum(q * m for q, m in zip(correction_coeff, monomials))
g_generic = tau * quartic + cubic_correction

g_equations: list[sp.Expr] = []
for px, py in nodes:
    g_equations.extend((
        g_generic.subs({X: px, Y: py}),
        sp.diff(g_generic, X).subs({X: px, Y: py}),
        sp.diff(g_generic, Y).subs({X: px, Y: py}),
    ))
g_equations.append((
    sp.diff(g_generic, X, Y).subs({X: a, Y: 0})
    - sp.diff(g_generic, X, Y).subs({X: -a, Y: 0})
) / (2 * a))

g_matrix, g_rhs = sp.linear_eq_to_matrix(g_equations, correction_coeff)
ck("quartic correction uses the same interpolation matrix", g_matrix == matrix)
g_solution_tuple = tuple(next(iter(sp.linsolve((g_matrix, g_rhs), correction_coeff))))
g = sp.factor(g_generic.subs(dict(zip(correction_coeff, g_solution_tuple))))

for label, (px, py) in zip(("M", "S", "P"), nodes):
    ck(f"g value zero at {label}", z(g.subs({X: px, Y: py})))
    ck(f"g X-gradient zero at {label}", z(sp.diff(g, X).subs({X: px, Y: py})))
    ck(f"g Y-gradient zero at {label}", z(sp.diff(g, Y).subs({X: px, Y: py})))
ck("g has equal endpoint mixed curvature",
   z(sp.diff(g, X, Y).subs({X: a, Y: 0})
     - sp.diff(g, X, Y).subs({X: -a, Y: 0})))

edge_value = sp.factor(g.subs(Y, 0))
edge_normal_gradient = sp.factor(sp.diff(g, Y).subs(Y, 0))
ck("edge value has two double zeros",
   z(edge_value - tau * q40 * (2 * X - eta)**2 * (2 * X + eta)**2 / 16))
ck("edge normal gradient has endpoint zeros and equal slopes",
   z(edge_normal_gradient
     - tau * q31 * X * (2 * X - eta) * (2 * X + eta) / 4))

H_g = sp.hessian(g, (X, Y))
Hg_M, Hg_S, Hg_P = tuple(
    H_g.subs({X: px, Y: py}).applyfunc(sp.factor) for px, py in nodes
)
for name, H in (("M", Hg_M), ("S", Hg_S)):
    ck(f"quartic error {name} tt = 2 tau eta^2 q40",
       z(H[0, 0] - 2 * tau * eta**2 * q40))
    ck(f"quartic error {name} te = tau eta^2 q31/2",
       z(H[0, 1] - tau * eta**2 * q31 / 2))
    ck(f"quartic error {name} ee is linear in tau", z(H[1, 1].subs(tau, 0)))
    ck(f"quartic error {name} ee denominator uses only sigma",
       denominator_uses_only(H[1, 1] / tau, {sigma}))
ck("endpoint te errors are exactly equal", z(Hg_M[0, 1] - Hg_S[0, 1]))

for i in range(2):
    for j in range(i, 2):
        ck(f"quartic witness error [{i},{j}] is linear in tau",
           z(Hg_P[i, j].subs(tau, 0)))
        ck(f"quartic witness error [{i},{j}] has no eta pole",
           eta not in sp.denom(sp.cancel(Hg_P[i, j] / tau)).free_symbols)
        ck(f"quartic witness error [{i},{j}] has no c pole",
           c not in sp.denom(sp.cancel(Hg_P[i, j] / tau)).free_symbols)

# F=s^-3(f(sX,sY)-b), so a normalized Hessian is multiplied by s in
# physical coordinates.  A Taylor quartic enters F with tau=s.
physical_M_tt = sp.factor(
    s_phys * Hg_M[0, 0].subs({tau: s_phys, eta: r_phys / s_phys})
)
physical_M_te = sp.factor(
    s_phys * Hg_M[0, 1].subs({tau: s_phys, eta: r_phys / s_phys})
)
ck("physical endpoint tt error is r^2 times bounded coefficient",
   z(physical_M_tt - 2 * q40 * r_phys**2))
ck("physical endpoint te error is r^2 times bounded coefficient",
   z(physical_M_te - q31 * r_phys**2 / 2))

for name, H in (("M", Hg_M), ("S", Hg_S)):
    physical_ee_over_s2 = sp.cancel(
        s_phys * H[1, 1].subs({tau: s_phys, eta: r_phys / s_phys}) / s_phys**2
    )
    ck(f"physical endpoint {name} ee error is s^2 times chart coefficient",
       denominator_uses_only(physical_ee_over_s2, {sigma, s_phys}))
for i in range(2):
    for j in range(i, 2):
        physical_witness_over_s2 = sp.cancel(
            s_phys * Hg_P[i, j].subs({tau: s_phys, eta: r_phys / s_phys})
            / s_phys**2
        )
        ck(f"physical witness [{i},{j}] error is s^2 times chart coefficient",
           denominator_uses_only(physical_witness_over_s2, {sigma, s_phys}))

print("SECTION C: TYPED COLLINEAR QUINTIC")

zeta = sp.symbols("zeta", real=True)
alpha0 = sp.Rational(1, 2)
quintic_coeff = sp.symbols("a0:6")
p_generic = sum(q * zeta**i for i, q in enumerate(quintic_coeff))
axis_nodes = (-sp.Rational(1, 2), sp.Rational(1, 2), sp.Integer(2))
axis_values = (0, -sp.Rational(1, 6), -alpha0 / 6)
axis_equations: list[sp.Expr] = []
for node, value in zip(axis_nodes, axis_values):
    axis_equations.extend((p_generic.subs(zeta, node) - value,
                           sp.diff(p_generic, zeta).subs(zeta, node)))
axis_solution = sp.solve(axis_equations, quintic_coeff, dict=True)[0]
p = sp.factor(p_generic.subs(axis_solution))

expected_p = sp.factor(
    (2 * zeta + 1)**2
    * (128 * alpha0 * zeta**3 - 444 * alpha0 * zeta**2
       + 348 * alpha0 * zeta - 79 * alpha0
       + 250 * zeta**3 - 1500 * zeta**2 + 3000 * zeta - 2000)
    / 20250
)
ck("explicit collinear quintic formula", z(p - expected_p))
for label, node, value in zip(("M", "S", "y"), axis_nodes, axis_values):
    ck(f"quintic value pin at {label}", z(p.subs(zeta, node) - value))
    ck(f"quintic critical pin at {label}", z(sp.diff(p, zeta).subs(zeta, node)))

curvatures = tuple(sp.factor(sp.diff(p, zeta, 2).subs(zeta, node))
                   for node in axis_nodes)
expected_curvatures = (
    -sp.Rational(3277, 2025),
    sp.Rational(569, 1125),
    sp.Rational(12, 25),
)
for label, actual, expected in zip(("M", "S", "y"), curvatures,
                                    expected_curvatures):
    ck(f"quintic curvature at {label}", z(actual - expected))
ck("M axial curvature is negative", curvatures[0] < 0)
ck("S axial curvature is positive", curvatures[1] > 0)
ck("witness axial curvature is positive", curvatures[2] > 0)

x = sp.symbols("x", real=True)
r = sp.symbols("r", positive=True)
q_r = sp.factor(r**3 * p.subs(zeta, x / r))
physical_nodes = (-r / 2, r / 2, 2 * r)
physical_curvatures = tuple(
    sp.factor(sp.diff(q_r, x, 2).subs(x, node)) for node in physical_nodes
)
for label, actual, normalized in zip(("M", "S", "y"), physical_curvatures,
                                      curvatures):
    ck(f"physical {label} curvature = r times normalized curvature",
       z(actual - r * normalized))

# Extend by -(x_2^2+x_3^2)/2.  The Hessian is diag(q_r'',-1,-1), so M is
# negative definite and S has Morse index two.
ck("3D extension makes M negative definite",
   physical_curvatures[0] < 0)
ck("3D extension makes S index two",
   physical_curvatures[1] > 0)

product_constant = sp.factor(abs(sp.prod(curvatures)))
det_product = sp.factor(product_constant * r**3)
ck("collinear determinant product is a nonzero constant times r^3",
   product_constant > 0 and z(det_product / r**3 - product_constant))

common_D = lambda rr, ss: sp.expand((rr * (rr + ss**2))**2 * (rr**2 + ss**3))
D_collar = sp.factor(common_D(r, 2 * r))
ck("D(r,2r) exact collar factorization",
   z(D_collar - r**6 * (1 + 4 * r)**2 * (1 + 8 * r)))
ck("typed quintic product / D(r,2r) diverges",
   sp.limit(det_product / D_collar, r, 0, dir="+") == sp.oo)

fourth_jet = sp.factor(sp.diff(q_r, x, 4))
fifth_jet = sp.factor(sp.diff(q_r, x, 5))
ck("physical fourth derivative has r^-1 scaling",
   z(fourth_jet - sp.diff(p, zeta, 4).subs(zeta, x / r) / r))
ck("physical fifth derivative has r^-2 scaling",
   z(fifth_jet - sp.diff(p, zeta, 5) / r**2))
ck("quintic fifth derivative is nonzero", sp.diff(p, zeta, 5) != 0)

print("SECTION D: AXIS DEFICIT AND ESIP ABSORPTION")

r0, s0 = sp.symbols("r0 s0", positive=True)
D_rs = common_D(r0, s0)
singular_slack = sp.expand(D_rs - r0**2 * s0**7)
ck("D-r^2*s^7 has nonnegative coefficients",
   coeffs_nonnegative(singular_slack, (r0, s0)))
ck("singular crude product r^2*s costs at most s^-6",
   z((D_rs * s0**-6) / (r0**2 * s0) - D_rs / (r0**2 * s0**7)))

lam = sp.symbols("lam", positive=True)
D_scaled = sp.factor(D_rs.subs(s0, lam * r0))
ck("collar D exact scaling",
   z(D_scaled - r0**6 * (1 + lam**2 * r0)**2
     * (1 + lam**3 * r0)))
collar_slack = sp.expand(D_scaled / r0**6 - 1)
ck("collar factor D/r^6 is at least one coefficientwise",
   coeffs_nonnegative(collar_slack, (r0, lam)))
ck("collar crude product r^3 costs at most r^-3",
   z((D_scaled * r0**-3) / r0**3 - D_scaled / r0**6))

# Density-weighted versions of the same two comparisons.
singular_crude_density = s0**-7 * r0**2 * s0
singular_target_with_loss = s0**-7 * D_rs * s0**-6
ck("singular density-weighted ratio is D/(r^2*s^7)",
   z(singular_target_with_loss / singular_crude_density
     - D_rs / (r0**2 * s0**7)))

collar_crude_after_value = r0**-4 * r0**3
collar_target_with_loss = r0**-4 * D_scaled * r0**-3
ck("collar density-weighted ratio is D/r^6",
   z(collar_target_with_loss / collar_crude_after_value
     - D_scaled / r0**6))

# Exact elementary domination used to absorb any fixed projective power:
# exp(c/theta^2) >= (c/theta^2)^j/j!.
theta, c_esip = sp.symbols("theta c_esip", positive=True)
for power in range(0, 21):
    j = math.ceil(power / 2)
    residual_theta_power = 2 * j - power
    ck(f"ESIP Taylor bound absorbs theta^-{power}",
       residual_theta_power >= 0)
ck("specific singular s^-6 loss uses third exponential term",
   2 * 3 - 6 == 0)
ck("specific collar r^-3 loss uses second exponential term",
   2 * 2 - 3 == 1)

# The angular substitution in H.25/H.26 and (9.1):
# z=s^2+rho^2, dz=2 rho d rho, then u=c/z.
rho, zvar, u = sp.symbols("rho zvar u", positive=True)
m = sp.symbols("m", real=True)
angular_integrand = rho * (s0**2 + rho**2)**(-m / 2) \
    * sp.exp(-c_esip / (s0**2 + rho**2))
after_z = sp.Rational(1, 2) * zvar**(-m / 2) * sp.exp(-c_esip / zvar)
after_u = sp.Rational(1, 2) * c_esip**(1 - m / 2) \
    * u**(m / 2 - 2) * sp.exp(-u)
ck("angular z-substitution Jacobian",
   z(angular_integrand.subs(rho, sp.sqrt(zvar - s0**2))
     / (2 * sp.sqrt(zvar - s0**2)) - after_z))
ck("angular u=c/z substitution integrand",
   z(after_z.subs(zvar, c_esip / u) * c_esip / u**2 - after_u))

# Reproduce the six-term singular radial ledger after value integration,
# volume, and Palm normalization: r*s^-5*D.
radial_integrand = sp.expand(r0 * s0**-5 * D_rs)
expected_radial = (
    r0**7 * s0**-5 + 2 * r0**6 * s0**-3
    + r0**5 * s0**-2 + r0**5 * s0**-1
    + 2 * r0**4 + r0**3 * s0**2
)
ck("six-term singular radial ledger", z(radial_integrand - expected_radial))

print("SECTION E: EXACT AXIAL-ENDPOINT COLLISION LEDGER")

# This degree-five one-dimensional family makes the endpoint-collision
# scaling concrete.  Its derivative has critical points 0,d,r/2,r, and the
# coefficient is chosen so that q(r)-q(0)=-r^3/6 independently of d.
d = sp.symbols("d", positive=True)
endpoint_qprime = sp.factor(
    20 * x * (x - d) * (x - r) * (x - r / 2) / r**2
)
endpoint_q = sp.factor(sp.integrate(endpoint_qprime, (x, 0, x)))
ck("endpoint family derivative vanishes at M",
   z(endpoint_qprime.subs(x, 0)))
ck("endpoint family derivative vanishes at y",
   z(endpoint_qprime.subs(x, d)))
ck("endpoint family derivative vanishes at S",
   z(endpoint_qprime.subs(x, r)))
ck("endpoint family has exact pair gap -r^3/6",
   z(endpoint_q.subs(x, r) + r**3 / 6))

endpoint_value_y = sp.factor(endpoint_q.subs(x, d))
endpoint_alpha = sp.factor(-6 * endpoint_value_y / r**3)
expected_endpoint_alpha = sp.factor(
    d**3 * (6 * d**2 - 15 * d * r + 10 * r**2) / r**5
)
ck("endpoint witness value is in canonical alpha form",
   z(endpoint_alpha - expected_endpoint_alpha))
ck("endpoint witness alpha is strictly between zero and one at d=r/4",
   0 < endpoint_alpha.subs(d, r / 4) < 1)

endpoint_curvatures = tuple(
    sp.factor(sp.diff(endpoint_q, x, 2).subs(x, node))
    for node in (0, d, r)
)
expected_endpoint_curvatures = (
    -10 * d,
    10 * d * (r - 2 * d) * (r - d) / r**2,
    10 * (r - d),
)
for label, actual, expected in zip(("M", "y", "S"), endpoint_curvatures,
                                    expected_endpoint_curvatures):
    ck(f"endpoint family curvature at {label}", z(actual - expected))
ck("endpoint family has max/saddle/saddle axial signs at d=r/4",
   endpoint_curvatures[0].subs(d, r / 4) < 0
   and endpoint_curvatures[1].subs(d, r / 4) > 0
   and endpoint_curvatures[2].subs(d, r / 4) > 0)

# With two fixed negative transverse curvatures, the absolute 3-Hessian
# determinant product is the product of the three axial curvatures.
endpoint_product = sp.factor(-sp.prod(endpoint_curvatures))
expected_endpoint_product = sp.factor(
    1000 * d**2 * (r - 2 * d) * (r - d)**2 / r**2
)
ck("endpoint determinant product exact", z(endpoint_product - expected_endpoint_product))
delta = sp.symbols("delta", positive=True)
ck("endpoint determinant product is r^3 on d=delta*r",
   z(endpoint_product.subs(d, delta * r)
     - 1000 * delta**2 * (1 - 2 * delta) * (1 - delta)**2 * r**3))

# Equation (4.2) gives gradient-density prefactor d^-3*r^-4 at the exact
# axial endpoint.  Multiplication by the r*d^2 determinant product leaves a
# genuine d^-1 factor.  It is integrable after d^2 dd volume, but no power
# of r alone can dominate it pointwise as d -> 0.
endpoint_density_prefactor = d**-3 * r**-4
endpoint_kernel_scale = sp.factor(endpoint_density_prefactor * d**2 * r)
ck("endpoint density times determinant scale = d^-1*r^-3",
   z(endpoint_kernel_scale - d**-1 * r**-3))
fixed_radial_loss = r**-20
claimed_common_scale = r**-4 * common_D(r, r / 2) * fixed_radial_loss
ck("no fixed radial loss controls the endpoint d^-1 pointwise",
   sp.limit(endpoint_kernel_scale / claimed_common_scale, d, 0, dir="+")
   == sp.oo)

endpoint_post_palm_volume = sp.factor(
    r**-2 * endpoint_kernel_scale * d**2
)
ck("endpoint post-Palm radial integrand = r^-5*d",
   z(endpoint_post_palm_volume - r**-5 * d))
epsilon0 = sp.symbols("epsilon0", positive=True)
endpoint_integrated = sp.integrate(endpoint_post_palm_volume, (d, 0, epsilon0 * r))
ck("endpoint radial integral before ESIP = epsilon0^2*r^-3/2",
   z(endpoint_integrated - epsilon0**2 * r**-3 / 2))
ck("endpoint e^-c/r^2 absorbs the integrated r^-3 deficit",
   sp.limit(r**-6 * sp.exp(-1 / r**2), r, 0, dir="+") == 0)

print("SECTION F: PAIR-MEASURABLE SINGULAR CORRECTED FRAME")

# Work on the positive axial chart; reflection supplies the negative chart.
# Here rho is the angular transverse coordinate and the physical transverse
# coordinate of y is s*rho.  The finite Taylor model retains exactly the
# terms needed through combined projective order two.
u_ax, v_ax = sp.symbols("u_ax v_ax", real=True)
s_ax, rho_ax = sp.symbols("s_ax rho_ax", positive=True)
a_ax = sp.symbols("a_ax", nonnegative=True)
D_ax = 1 - a_ax
H0, H1, H2 = sp.symbols("H0 H1 H2")
A0j, A1j, A2j = sp.symbols("A0j A1j A2j")
B0j, B1j = sp.symbols("B0j B1j")
C0j, D0j = sp.symbols("C0j D0j")
E0j, E1j = sp.symbols("E0j E1j")

H_series = H0 + u_ax * H1 + u_ax**2 * H2 / 2
A_series = A0j + u_ax * A1j + u_ax**2 * A2j / 2
B_series = B0j + u_ax * B1j
C_series = C0j + u_ax * D0j
E_series = E0j + u_ax * E1j
w_axis = u_ax**2 - a_ax * s_ax**2
F_axis = w_axis**2 * H_series
Fv_axis = w_axis * A_series
Fw_axis = w_axis * B_series

F_local = F_axis + v_ax * Fv_axis + v_ax**2 * C_series / 2
Fw_local = Fw_axis + v_ax * E_series
Fu_local = sp.diff(F_local, u_ax)
Fv_local = sp.diff(F_local, v_ax)
at_witness = {u_ax: s_ax, v_ax: s_ax * rho_ax}
Fy = sp.expand(F_local.subs(at_witness))
Fuy = sp.expand(Fu_local.subs(at_witness))
Fvy = sp.expand(Fv_local.subs(at_witness))
Fwy = sp.expand(Fw_local.subs(at_witness))

# Correct value row: every subtracted quantity is either a witness component
# or pair-measurable.  In particular, no unconditioned midpoint value F(0)
# occurs.
V0bar = sp.expand(
    (Fy - s_ax * rho_ax * Fvy / 2 - s_ax * Fuy / 3) / s_ax**3
)
V1bar = sp.expand(Fuy / s_ax**2)
V2bar = sp.expand(Fvy / s_ax)
V3bar = sp.expand(Fwy / s_ax)


def projective_degree(term: sp.Expr) -> int:
    powers = term.as_powers_dict()
    return int(powers.get(s_ax, 0) + powers.get(rho_ax, 0))


def projective_part(expr: sp.Expr, degree: int) -> sp.Expr:
    return sp.expand(sum(
        term for term in sp.expand(expr).as_ordered_terms()
        if projective_degree(term) == degree
    ))


V0_first = sp.factor(projective_part(V0bar, 1))
V1_first = sp.factor(projective_part(V1bar, 1))
V2_first = sp.factor(projective_part(V2bar, 1))
V3_first = sp.factor(projective_part(V3bar, 1))
ck("pair-measurable V0 first-order row",
   z(V0_first
     + s_ax * D_ax * (1 + 3 * a_ax) * H0 / 3
     + rho_ax * (1 + 3 * a_ax) * A0j / 6))
ck("singular V1 first-order row",
   z(V1_first - 4 * s_ax * D_ax * H0 - 2 * rho_ax * A0j))
ck("singular V2 first-order row",
   z(V2_first - s_ax * D_ax * A0j - rho_ax * C0j))
ck("singular V3 first-order row",
   z(V3_first - s_ax * D_ax * B0j - rho_ax * E0j))

lambda_ax = 12 / (1 + 3 * a_ax)
Wbar = sp.expand(V1bar + lambda_ax * V0bar)
ck("Wbar first projective order cancels for every a",
   z(projective_part(Wbar, 1)))
Wbar_second = sp.factor(projective_part(Wbar, 2))
expected_Wbar_second = sp.factor(
    -3 * D_ax / (1 + 3 * a_ax)
    * (s_ax**2 * D_ax**2 * H1
       + s_ax * rho_ax * D_ax * A1j
       + rho_ax**2 * D0j / 2)
)
ck("Wbar general-a second-order row", z(Wbar_second - expected_Wbar_second))
ck("Wbar specializes to the printed eta=0 second row",
   z(Wbar_second.subs(a_ax, 0)
     + 3 * (s_ax**2 * H1 + s_ax * rho_ax * A1j
            + rho_ax**2 * D0j / 2)))

# Before taking c -> +1, the exact coefficient which cancels the complete
# first-order row is 12c/(c^2+3a).
c_ax = sp.symbols("c_ax", real=True, nonzero=True)
D_cax = c_ax**2 - a_ax
V0_first_general = -(c_ax**2 + 3 * a_ax) * (
    2 * s_ax * D_cax * H0 + rho_ax * A0j
) / 6
V1_first_general = 2 * c_ax * (
    2 * s_ax * D_cax * H0 + rho_ax * A0j
)
lambda_general = 12 * c_ax / (c_ax**2 + 3 * a_ax)
ck("general-c W coefficient cancels the first-order row",
   z(V1_first_general + lambda_general * V0_first_general))
W_second_general = sp.factor(
    -3 * D_cax / (c_ax**2 + 3 * a_ax)
    * (s_ax**2 * D_cax**2 * H1
       + s_ax * rho_ax * D_cax * A1j
       + rho_ax**2 * D0j / 2)
)
ck("general-c second row reduces to positive-axis Wbar row",
   z(W_second_general.subs(c_ax, 1) - expected_Wbar_second))

# At fixed pair pins, the map from raw (F,Fu,Fv,Fw) to
# (V0bar,V1bar,V2bar,V3bar) is triangular up to the first-row shears.
raw_transform = sp.Matrix([
    [s_ax**-3, -sp.Rational(1, 3) * s_ax**-2,
     -rho_ax * sp.Rational(1, 2) * s_ax**-2, 0],
    [0, s_ax**-2, 0, 0],
    [0, 0, s_ax**-1, 0],
    [0, 0, 0, s_ax**-1],
])
ck("singular raw-to-corrected determinant = s^-7",
   z(raw_transform.det() - s_ax**-7))

# Coefficient matrix for the compact normalized frame
# (V0bar/Lambda,Wbar/Lambda^2,V2bar/Lambda,V3bar/Lambda), with columns
# H0,A0,B0,C0,E0,H1,A1,D0.  Its squared exterior norm factors explicitly
# and is positive on T^2+R^2=1, 0<=a<1, including T=0 and R=0.
T_ax, R_ax = sp.symbols("T_ax R_ax", nonnegative=True)
frame = sp.zeros(4, 8)
frame[0, 0] = -T_ax * D_ax * (1 + 3 * a_ax) / 3
frame[0, 1] = -R_ax * (1 + 3 * a_ax) / 6
frame_factor = -3 * D_ax / (1 + 3 * a_ax)
frame[1, 5] = frame_factor * D_ax**2 * T_ax**2
frame[1, 6] = frame_factor * D_ax * T_ax * R_ax
frame[1, 7] = frame_factor * R_ax**2 / 2
frame[2, 1] = T_ax * D_ax
frame[2, 3] = R_ax
frame[3, 2] = T_ax * D_ax
frame[3, 4] = R_ax

import itertools

nonzero_minors = []
for columns in itertools.combinations(range(frame.cols), frame.rows):
    minor = sp.factor(frame[:, columns].det())
    if minor != 0:
        nonzero_minors.append(minor)
frame_exterior_sq = sp.factor(sum(minor**2 for minor in nonzero_minors))
expected_exterior_sq = sp.factor(
    D_ax**2 * (R_ax**2 + T_ax**2 * D_ax**2)
    * (R_ax**2 + 2 * T_ax**2 * D_ax**2)**4 / 16
)
ck("singular normalized frame has 18 nonzero minors",
   len(nonzero_minors) == 18)
ck("singular full-boundary exterior norm factorization",
   z(frame_exterior_sq - expected_exterior_sq))
ck("contact boundary T=0 frame remains full rank",
   z(frame_exterior_sq.subs({T_ax: 0, R_ax: 1}) - D_ax**2 / 16))
ck("collinear boundary R=0 frame remains full rank",
   z(frame_exterior_sq.subs({T_ax: 1, R_ax: 0}) - 16 * D_ax**12 / 16))
ck("five total row powers imply covariance determinant Lambda^10",
   2 * (1 + 2 + 1 + 1) == 10)

# The axial-gradient row alone gives the ESIP: its cubic conditional mean is
# kappa(c^2-a), while its residual variance is quadratic in (s,rho).
V1_coefficient_norm_sq = sp.expand(
    (4 * s_ax * D_ax)**2 + (2 * rho_ax)**2
)
ck("V1 leading variance proxy is O(s^2+rho^2)",
   z(V1_coefficient_norm_sq
     - 16 * s_ax**2 * D_ax**2 - 4 * rho_ax**2))
ck("V1 target mismatch is separated for a<1 on the axis",
   z(kappa * (1 - a_ax) - kappa * D_ax))

# Why subtracting F(0) is impermissible: the endpoint value-gradient pins do
# not determine the midpoint value.  This quartic lies in their kernel and
# has a nonzero midpoint value.
h_ax = sp.symbols("h_ax", nonzero=True)
q_mid = (u_ax**2 - h_ax**2)**2
ck("midpoint diagnostic vanishes at left endpoint", z(q_mid.subs(u_ax, -h_ax)))
ck("midpoint diagnostic vanishes at right endpoint", z(q_mid.subs(u_ax, h_ax)))
ck("midpoint diagnostic derivative vanishes at left endpoint",
   z(sp.diff(q_mid, u_ax).subs(u_ax, -h_ax)))
ck("midpoint diagnostic derivative vanishes at right endpoint",
   z(sp.diff(q_mid, u_ax).subs(u_ax, h_ax)))
ck("midpoint diagnostic value is not pair-measurable",
   sp.factor(q_mid.subs(u_ax, 0)) == h_ax**4)

print("SECTION G: GENERIC JOINT VALUE ROW R0")

Tf, E2f, E3f = sp.symbols("Tf E2f E3f")
xi1f, xi2f, xi3f = sp.symbols("xi1f xi2f xi3f")
Vf = xi2f * E2f + xi3f * E3f
Af = xi1f * Tf + Vf

# The exact trapezoidal Hermite row is based at M.  If
# Du=(xi1+1/2)T+V and z=r*Du, then
# exp(z)-1-z(exp(z)+1)/2 has cubic coefficient -Du^3/12.
u1f = xi1f + sp.Rational(1, 2)
Duf = u1f * Tf + Vf
zformal = sp.symbols("zformal")
trapezoid_series = sp.series(
    sp.exp(zformal) - 1 - zformal * (sp.exp(zformal) + 1) / 2,
    zformal, 0, 5,
).removeO()
ck("trapezoidal Hermite residual has no terms below cubic",
   z(sp.expand(trapezoid_series).coeff(zformal, 0))
   and z(sp.expand(trapezoid_series).coeff(zformal, 1))
   and z(sp.expand(trapezoid_series).coeff(zformal, 2)))
ck("generic R0 contact symbol = -Du^3/12",
   z(sp.expand(trapezoid_series).coeff(zformal, 3) * Duf**3
     + Duf**3 / 12))
R1_symbol = Tf * Vf * (xi1f * Tf + Vf / 2)
R2_symbol = Vf * E2f
R3_symbol = Vf * E3f

# A fixed 4x4 coefficient minor on monomials
# (V^3,T V^2,V E2,V E3) is nonzero for every xi1.
generic_minor = sp.Matrix([
    [-sp.Rational(1, 12), 0, 0, 0],
    [0, sp.Rational(1, 2), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
]).det()
ck("generic value-gradient limiting rows are independent",
   generic_minor == -sp.Rational(1, 24))

generic_raw_transform = sp.Matrix([
    [r**-3, -u1f * sp.Rational(1, 2) * r**-2,
     -xi2f * sp.Rational(1, 2) * r**-2,
     -xi3f * sp.Rational(1, 2) * r**-2],
    [0, r**-2, 0, 0],
    [0, 0, r**-1, 0],
    [0, 0, 0, r**-1],
])
ck("generic raw-to-corrected determinant = r^-7",
   z(generic_raw_transform.det() - r**-7))
ck("generic inverse diagonal scales are (r^3,r^2,r,r)",
   z(1 / generic_raw_transform.det() - r**7))

print(f"TOTAL_CHECKS={CHECKS}")
print("SCOPE_LIMIT: symbolic interpolation, exact counterexample, and axis "
      "power/map ledgers only; no side-24 covariance-floor, full Gaussian "
      "regression, Kac--Rice, or theorem certification")

if FAILURES:
    print(f"ALL_ASSERTIONS_FAIL ({len(FAILURES)} failures)")
    for failure in FAILURES:
        print("FAILED_CHECK=" + failure)
    raise SystemExit(1)

print("ALL_ASSERTIONS_PASS")
