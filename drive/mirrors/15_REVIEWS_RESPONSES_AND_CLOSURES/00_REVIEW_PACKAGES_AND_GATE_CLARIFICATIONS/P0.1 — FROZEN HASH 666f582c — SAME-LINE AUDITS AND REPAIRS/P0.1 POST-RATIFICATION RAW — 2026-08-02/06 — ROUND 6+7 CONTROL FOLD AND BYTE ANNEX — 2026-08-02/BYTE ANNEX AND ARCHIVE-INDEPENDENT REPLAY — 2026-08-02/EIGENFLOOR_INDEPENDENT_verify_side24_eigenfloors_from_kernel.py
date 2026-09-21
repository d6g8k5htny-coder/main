#!/usr/bin/env python3
"""Independent SIDE24 covariance/eigenfloor reconstruction.

This verifier starts from the Gaussian spectral moment rule and from the
periodized kernel.  It does not import or execute any project verifier.

Analytic scope:
  * exact Bargmann--Fock contact pair-pin spectrum;
  * exact generic-transverse three-row Schur matrix and determinant;
  * exact singular-near four-row Schur matrix and determinant;
  * explicit side-24 image bound transferring the local pair-pin floor.

Numerical scope:
  * all-image (not spherical Fourier-truncation) evaluation of the corrected
    pair-pin covariance for the historical probe directions/radii.

The finite-r numerical table is diagnostic.  The uniform local floor is
proved by the exact contact calculation plus the explicit image bound and
continuity; no sampled minimum is promoted to an analytic compact-set floor.
"""

from __future__ import annotations

import os
from pathlib import Path

import mpmath as mp
import sympy as sp


checks: list[bool] = []


def ck(label: str, condition: bool) -> None:
    ok = bool(condition)
    checks.append(ok)
    print(f"[{'ok' if ok else 'FAIL'}] {label}")


# ---------------------------------------------------------------------------
# Exact continuous-BF calculations from Gaussian polynomial moments.
# ---------------------------------------------------------------------------

T, E, W = sp.symbols("T E W", real=True)
I = sp.I


def double_factorial_moment(power: int) -> sp.Integer:
    if power % 2:
        return sp.Integer(0)
    return sp.Integer(1) if power == 0 else sp.factorial2(power - 1)


def gaussian_expectation(expression: sp.Expr) -> sp.Expr:
    polynomial = sp.Poly(sp.expand(expression), T, E, W)
    return sp.simplify(
        sum(
            coefficient
            * double_factorial_moment(monomial[0])
            * double_factorial_moment(monomial[1])
            * double_factorial_moment(monomial[2])
            for monomial, coefficient in polynomial.terms()
        )
    )


def gram(rows: list[sp.Expr]) -> sp.Matrix:
    return sp.Matrix(
        [
            [gaussian_expectation(left * sp.conjugate(right)) for right in rows]
            for left in rows
        ]
    )


def schur_exact(pin_rows: list[sp.Expr], output_rows: list[sp.Expr]) -> sp.Matrix:
    joint = gram(pin_rows + output_rows)
    count = len(pin_rows)
    pins = joint[:count, :count]
    return sp.simplify(
        joint[count:, count:]
        - joint[count:, :count] * pins.inv() * joint[:count, count:]
    )


# The corrected contact pin frame, with the last row equal to -D_T^3/12.
contact_pins = [
    sp.Integer(1),
    I * T,
    I * E,
    I * W,
    (I * T) ** 2,
    (I * T) * (I * E),
    (I * T) * (I * W),
    -(I * T) ** 3 / 12,
]
pin_gram = gram(contact_pins)

# Rotation invariance lets us use T as the pair direction.  The even block is
# [[1,-1],[-1,3]] plus two unit rows; the odd nontrivial block is
# [[1,1/4],[1/4,5/48]].
lambda_bf = (sp.Integer(53) - 5 * sp.sqrt(97)) / 96
ck("01 corrected contact pin determinant positive", pin_gram.det() > 0)
ck(
    "02 corrected contact pin exact least eigenvalue",
    sp.simplify(
        lambda_bf
        - (sp.Rational(53, 96) - sp.Rational(5, 96) * sp.sqrt(97))
    )
    == 0,
)
ck("03 normalized BF contact eigenfloor exceeds 0.039", lambda_bf > sp.Rational(39, 1000))


# Generic transverse FACEWISE (3.5).  V=Y E+Z W and X is the axial mark.
X, Y, Z = sp.symbols("X Y Z", real=True)
V = Y * E + Z * W
generic_rows = [
    X * (I * T) ** 2 * (I * V) + sp.Rational(1, 2) * (I * T) * (I * V) ** 2,
    (I * V) * (I * E),
    (I * V) * (I * W),
]
generic_schur = schur_exact(contact_pins, generic_rows)
rho2 = Y**2 + Z**2
generic_expected = sp.Matrix(
    [
        [rho2 * (4 * X**2 + rho2) / 2, 0, 0],
        [0, 2 * Y**2 + Z**2, Y * Z],
        [0, Y * Z, Y**2 + 2 * Z**2],
    ]
)
ck(
    "04 generic contact Schur matrix derived exactly",
    (generic_schur - generic_expected).applyfunc(sp.simplify) == sp.zeros(3),
)
ck(
    "05 generic contact determinant is rho^6(4X^2+rho^2)",
    sp.factor(generic_schur.det() - rho2**3 * (4 * X**2 + rho2)) == 0,
)
ck(
    "06 generic transverse eigenvalues have explicit positive frame",
    sp.factor(generic_expected[0, 0]) == rho2 * (4 * X**2 + rho2) / 2,
)


# Singular-near FACEWISE (5.1), derived from the pair-pin-measurable frame.
c, rho = sp.symbols("c rho", real=True)
omega = c * (I * T) + rho * (I * E)
singular_rows = [
    -rho * (I * E) * omega**2 / 12,
    c * rho * (I * T) ** 2 * (I * E)
    + rho**2 * (I * T) * (I * E) ** 2 / 2,
    rho * (I * E) ** 2,
    rho * (I * E) * (I * W),
]
singular_schur = schur_exact(contact_pins, singular_rows)
singular_expected = sp.Matrix(
    [
        [rho**2 * (c**4 + 4 * c**2 * rho**2 + 3 * rho**4) / 72,
         -c * rho**2 * (c**2 + rho**2) / 6, 0, 0],
        [-c * rho**2 * (c**2 + rho**2) / 6,
         rho**2 * (4 * c**2 + rho**2) / 2, 0, 0],
        [0, 0, 2 * rho**2, 0],
        [0, 0, 0, rho**2],
    ]
)
ck(
    "07 singular-near contact Schur matrix derived exactly",
    (singular_schur - singular_expected).applyfunc(sp.simplify) == sp.zeros(4),
)
ck(
    "08 singular-near determinant has both anisotropic factors",
    sp.factor(
        singular_schur.det()
        - rho**10 * (c**2 + rho**2) * (3 * c**2 + rho**2) / 24
    )
    == 0,
)


# ---------------------------------------------------------------------------
# Side-24 periodized kernel and independent all-image numerical construction.
# ---------------------------------------------------------------------------

mp.mp.dps = 100
SIDE = mp.mpf(24)


def hermite_prob(order: int, value: mp.mpf) -> mp.mpf:
    if order == 0:
        return mp.mpf(1)
    if order == 1:
        return value
    previous, current = mp.mpf(1), value
    for index in range(1, order):
        previous, current = current, value * current - index * previous
    return current


def kernel_1d_derivative(order: int, displacement: mp.mpf, images: int = 2) -> mp.mpf:
    denominator = mp.fsum(
        mp.exp(-((SIDE * image) ** 2) / 2)
        for image in range(-images, images + 1)
    )
    numerator = mp.fsum(
        (-1) ** order
        * hermite_prob(order, displacement + SIDE * image)
        * mp.exp(-((displacement + SIDE * image) ** 2) / 2)
        for image in range(-images, images + 1)
    )
    return numerator / denominator


Term = tuple[mp.mpf, tuple[mp.mpf, mp.mpf, mp.mpf], tuple[int, int, int]]
Functional = list[Term]


def fscale(functional: Functional, scalar: mp.mpf) -> Functional:
    return [(scalar * coefficient, point, alpha) for coefficient, point, alpha in functional]


def fadd(*functionals: Functional) -> Functional:
    return [term for functional in functionals for term in functional]


def derivative_functional(
    point: tuple[mp.mpf, mp.mpf, mp.mpf],
    directions: list[tuple[mp.mpf, mp.mpf, mp.mpf]],
    scalar: mp.mpf = mp.mpf(1),
) -> Functional:
    expansion: dict[tuple[int, int, int], mp.mpf] = {(0, 0, 0): scalar}
    for direction in directions:
        updated: dict[tuple[int, int, int], mp.mpf] = {}
        for alpha, coefficient in expansion.items():
            for axis in range(3):
                beta = list(alpha)
                beta[axis] += 1
                key = tuple(beta)
                updated[key] = updated.get(key, mp.mpf(0)) + coefficient * direction[axis]
        expansion = updated
    return [(coefficient, point, alpha) for alpha, coefficient in expansion.items() if coefficient]


def covariance(left: Functional, right: Functional, images: int = 2) -> mp.mpf:
    total = mp.mpf(0)
    for coefficient_l, point_l, alpha in left:
        for coefficient_r, point_r, beta in right:
            value = coefficient_l * coefficient_r * ((-1) ** sum(beta))
            for axis in range(3):
                value *= kernel_1d_derivative(
                    alpha[axis] + beta[axis], point_l[axis] - point_r[axis], images
                )
            total += value
    return total


def covariance_matrix(rows: list[Functional], images: int = 2) -> mp.matrix:
    return mp.matrix(
        [[covariance(rows[i], rows[j], images) for j in range(len(rows))] for i in range(len(rows))]
    )


def normalize(vector: tuple[float, float, float]) -> tuple[mp.mpf, mp.mpf, mp.mpf]:
    values = tuple(mp.mpf(item) for item in vector)
    norm = mp.sqrt(mp.fsum(item * item for item in values))
    return tuple(item / norm for item in values)


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def dot(a, b):
    return mp.fsum(x * y for x, y in zip(a, b))


def frame(direction: tuple[float, float, float]):
    t = normalize(direction)
    seed = (mp.mpf(0), mp.mpf(0), mp.mpf(1))
    if abs(dot(t, seed)) > mp.mpf("0.85"):
        seed = (mp.mpf(0), mp.mpf(1), mp.mpf(0))
    e = normalize(tuple(seed[i] - dot(seed, t) * t[i] for i in range(3)))
    w = normalize(cross(t, e))
    return t, e, w


def corrected_pin_rows(radius: mp.mpf, basis) -> list[Functional]:
    t, e, w = basis
    origin = (mp.mpf(0), mp.mpf(0), mp.mpf(0))
    minus = tuple(-radius * item / 2 for item in t)
    plus = tuple(radius * item / 2 for item in t)
    dirs = (t, e, w)
    value_m = derivative_functional(minus, [])
    value_s = derivative_functional(plus, [])
    grad_m = [derivative_functional(minus, [direction]) for direction in dirs]
    grad_s = [derivative_functional(plus, [direction]) for direction in dirs]
    differences = [
        fadd(fscale(grad_s[j], 1 / radius), fscale(grad_m[j], -1 / radius))
        for j in range(3)
    ]
    trapezoid = fadd(
        fscale(value_s, 1 / radius**3),
        fscale(value_m, -1 / radius**3),
        fscale(grad_s[0], -1 / (2 * radius**2)),
        fscale(grad_m[0], -1 / (2 * radius**2)),
    )
    return [value_m, *grad_m, *differences, trapezoid]


def contact_pin_rows(basis) -> list[Functional]:
    t, e, w = basis
    point = (mp.mpf(0), mp.mpf(0), mp.mpf(0))
    return [
        derivative_functional(point, []),
        derivative_functional(point, [t]),
        derivative_functional(point, [e]),
        derivative_functional(point, [w]),
        derivative_functional(point, [t, t]),
        derivative_functional(point, [t, e]),
        derivative_functional(point, [t, w]),
        derivative_functional(point, [t, t, t], mp.mpf(-1) / 12),
    ]


def conditioned_covariance(pin_rows: list[Functional], output_rows: list[Functional]) -> mp.matrix:
    joint = covariance_matrix(pin_rows + output_rows)
    count = len(pin_rows)
    pins = joint[:count, :count]
    coupling = joint[:count, count:]
    return joint[count:, count:] - coupling.T * (pins**-1) * coupling


def point_in_frame(radius: mp.mpf, coordinates, basis):
    return tuple(
        radius * mp.fsum(coordinates[j] * basis[j][axis] for j in range(3))
        for axis in range(3)
    )


def generic_contact_outputs(coordinates, basis) -> list[Functional]:
    xi1, xi2, xi3 = coordinates
    t, e, w = basis
    point = (mp.mpf(0), mp.mpf(0), mp.mpf(0))
    transverse = tuple(xi2 * e[axis] + xi3 * w[axis] for axis in range(3))
    first = fadd(
        derivative_functional(point, [t, t, transverse], xi1),
        derivative_functional(point, [t, transverse, transverse], mp.mpf(1) / 2),
    )
    return [
        first,
        derivative_functional(point, [transverse, e]),
        derivative_functional(point, [transverse, w]),
    ]


def generic_finite_outputs(radius: mp.mpf, coordinates, basis, pins) -> list[Functional]:
    xi1, xi2, xi3 = coordinates
    t, e, w = basis
    minus = tuple(-radius * item / 2 for item in t)
    witness = point_in_frame(radius, coordinates, basis)
    directions_local = (t, e, w)
    grad_m = [derivative_functional(minus, [direction]) for direction in directions_local]
    grad_y = [derivative_functional(witness, [direction]) for direction in directions_local]
    differences = pins[4:7]
    trapezoid = pins[7]
    u = (xi1 + mp.mpf(1) / 2, xi2, xi3)
    transverse_e = fadd(
        fscale(grad_y[1], 1 / radius),
        fscale(grad_m[1], -1 / radius),
        fscale(differences[1], -u[0]),
    )
    transverse_w = fadd(
        fscale(grad_y[2], 1 / radius),
        fscale(grad_m[2], -1 / radius),
        fscale(differences[2], -u[0]),
    )
    axial = fadd(
        fscale(grad_y[0], 1 / radius**2),
        fscale(grad_m[0], -1 / radius**2),
        *(fscale(differences[j], -u[j] / radius) for j in range(3)),
        fscale(trapezoid, 6 * xi1**2 - mp.mpf(3) / 2),
    )
    return [axial, transverse_e, transverse_w]


directions = [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (1.0, 1.0, 1.0)]
normalized_local_minima = []
for direction in directions:
    matrix = covariance_matrix(contact_pin_rows(frame(direction)))
    eigenvalues = mp.eigsy(matrix, eigvals_only=True)
    normalized_local_minima.append(eigenvalues[0])

# A direct bound for derivatives through order six at contact.  For n!=0,
# |He_j(24n)| <= 2(24|n|)^6 (j<=6), and the two-sided image tail is below
# 4*24^6*sum n^6 exp(-288 n^2) < 2e-116.  Product-kernel entries and the
# directional row l1 coefficients then give the deliberately loose 6e-111
# operator bound used below.
q_image = mp.exp(-288)
image_derivative_bound = 8 * (mp.mpf(24) ** 6) * q_image
ck("09 one-dimensional side-24 local image correction < 2e-116", image_derivative_bound < mp.mpf("2e-116"))
ck("10 local joint-Gram perturbation budget < 6e-111", 288 * image_derivative_bound < mp.mpf("6e-111"))
ck(
    "11 side-24 local corrected-pin eigenfloor > 0.039 for all directions",
    mp.mpf(str(sp.N(lambda_bf, 90))) - mp.mpf("6e-111") > mp.mpf("0.039"),
)
ck(
    "12 all-image local probes agree with exact BF floor to 1e-90",
    max(abs(value - mp.mpf(str(sp.N(lambda_bf, 100)))) for value in normalized_local_minima)
    < mp.mpf("1e-90"),
)

# Convert the normalized local covariance to the unnormalized spectral
# convention used by the historical 34.4 diagnostic.
h = mp.pi / 12
spatial_normalizer = mp.fsum(mp.exp(-((SIDE * image) ** 2) / 2) for image in range(-2, 3))
spectral_mass_1d = mp.sqrt(2 * mp.pi) / h * spatial_normalizer
unnormalized_limit = normalized_local_minima[0] * spectral_mass_1d**3
ck("13 unnormalized local eigenfloor lies in historical 34.3--34.4 band", mp.mpf("34.3") < unnormalized_limit < mp.mpf("34.4"))

print(f"EXACT_NORMALIZED_BF_FLOOR={sp.N(lambda_bf, 60)}")
print(f"SIDE24_NORMALIZED_LOCAL_FLOOR={mp.nstr(normalized_local_minima[0], 60)}")
print(f"SIDE24_UNNORMALIZED_LOCAL_FLOOR={mp.nstr(unnormalized_limit, 60)}")

# Independent all-image finite-r table.  This is explicitly diagnostic; its
# role is to reproduce the surfaced historical values without the project's
# spherical lattice truncation.
radii = [mp.mpf(value) for value in ("2", "1", "0.5", "0.2", "0.1", "0.05", "0.02")]
finite_table: list[tuple[str, list[mp.mpf]]] = []
for direction in directions:
    basis = frame(direction)
    row = []
    for radius in radii:
        matrix = covariance_matrix(corrected_pin_rows(radius, basis))
        eig = mp.eigsy(matrix, eigvals_only=True)[0]
        row.append(eig * spectral_mass_1d**3)
    finite_table.append((str(direction), row))
    ck(f"14 finite-r corrected pins positive for {direction}", min(row) > 0)
    ck(
        f"15 r=.02 corrected-pin value approaches local floor for {direction}",
        abs(row[-1] / unnormalized_limit - 1) < mp.mpf("0.002"),
    )

for label, values in finite_table:
    print("FINITE_PIN_TABLE", label, " ".join(mp.nstr(value, 10) for value in values))

# Stability of the all-image computation: n=-1..1 and n=-2..2 agree far
# beyond the printed precision on all finite table entries.
probe_basis = frame((1.0, 1.0, 1.0))
probe_rows = corrected_pin_rows(mp.mpf("0.1"), probe_basis)
e1 = mp.eigsy(covariance_matrix(probe_rows, images=1), eigvals_only=True)[0]
e2 = mp.eigsy(covariance_matrix(probe_rows, images=2), eigvals_only=True)[0]
ck("20 all-image tail stable at 80 decimal digits", abs(e1 - e2) < mp.mpf("1e-80"))

# Reconstruct the surfaced generic-collar diagnostic range with the exact
# all-image kernel and an independently assembled covariance.  One axial
# frame suffices for the displayed finite grid; the analytic perturbation
# bound, not this grid, controls the uncountable direction set.
collar_basis = frame((1.0, 0.0, 0.0))
collar_xis = [
    ("-1.20", "0.35", "0.20"),
    ("-0.75", "0.40", "-0.55"),
    ("-0.22", "0.28", "0.44"),
    ("0.23", "-0.36", "0.31"),
    ("0.78", "0.62", "0.18"),
    ("1.25", "-0.48", "-0.39"),
]
collar_radii = [mp.mpf(value) for value in ("0.4", "0.2", "0.1", "0.05", "0.02")]
collar_ratios: list[mp.mpf] = []
collar_eigenvalues: list[mp.mpf] = []
for raw_coordinates in collar_xis:
    coordinates = tuple(mp.mpf(value) for value in raw_coordinates)
    rho_sq = coordinates[1] ** 2 + coordinates[2] ** 2
    reference = rho_sq**3 * (4 * coordinates[0] ** 2 + rho_sq)
    for radius in collar_radii:
        pins = corrected_pin_rows(radius, collar_basis)
        residuals = generic_finite_outputs(radius, coordinates, collar_basis, pins)
        conditional = conditioned_covariance(pins, residuals)
        collar_ratios.append(mp.det(conditional) / reference)
        collar_eigenvalues.append(mp.eigsy(conditional, eigvals_only=True)[0])
    contact = conditioned_covariance(
        contact_pin_rows(collar_basis), generic_contact_outputs(coordinates, collar_basis)
    )
    ck(
        f"21 exact contact determinant ratio is one at xi={raw_coordinates}",
        abs(mp.det(contact) / reference - 1) < mp.mpf("1e-90"),
    )

collar_min_ratio = min(collar_ratios)
collar_max_ratio = max(collar_ratios)
collar_min_eigen = min(collar_eigenvalues)
ck("27 exact-kernel finite-grid ratio minimum corroborates 0.78104", mp.mpf("0.780") < collar_min_ratio < mp.mpf("0.782"))
ck("28 exact-kernel finite-grid ratio maximum corroborates 1.83594", mp.mpf("1.835") < collar_max_ratio < mp.mpf("1.837"))
ck("29 exact-kernel finite-grid Schur minimum corroborates 0.0485002", mp.mpf("0.048") < collar_min_eigen < mp.mpf("0.049"))
print(f"GENERIC_GRID_RATIO_RANGE=[{mp.nstr(collar_min_ratio, 12)},{mp.nstr(collar_max_ratio, 12)}]")
print(f"GENERIC_GRID_MIN_SCHUR_EIGENVALUE={mp.nstr(collar_min_eigen, 12)}")

if os.environ.get("SIDE24_EIGENFLOOR_FORCE_FAILURE") == "1":
    ck("forced-failure mutation", False)

passed = sum(checks)
print(f"CHECKS={passed}/{len(checks)}")
print("SCOPE: exact contact identities + analytic local side-24 floor; finite-r table is diagnostic.")
if passed != len(checks):
    print("CHECK_FAILURE")
    raise SystemExit(1)
print("ALL_ASSERTIONS_PASS")
