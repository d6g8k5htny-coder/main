#!/usr/bin/env python3
"""Fail-closed replay of the reconstructed side-24 G.9.2 transfer.

Dependency-free: Python 3.11+ standard library only.  The all-image bound is
exact rational arithmetic.  Decimal calculations are an independent
180-digit regression and do not carry the uniform theorem by sampling.
"""

from __future__ import annotations

import os
from decimal import Decimal as D
from decimal import getcontext
from fractions import Fraction as F
from math import factorial


checks: list[bool] = []


def ck(label: str, condition: bool) -> None:
    ok = bool(condition)
    checks.append(ok)
    print(f"[{'ok' if ok else 'FAIL'}] {label}")


def poly_add(a: list[int], b: list[int]) -> list[int]:
    n = max(len(a), len(b))
    return [
        (a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0)
        for i in range(n)
    ]


def hermite_polynomials(max_order: int) -> list[list[int]]:
    polys = [[1], [0, 1]]
    for n in range(1, max_order):
        xpn = [0] + polys[n]
        previous = [-n * value for value in polys[n - 1]]
        polys.append(poly_add(xpn, previous))
    return polys[: max_order + 1]


def poly_value(poly: list[int], x: F) -> F:
    value = F(0)
    for coefficient in reversed(poly):
        value = value * x + coefficient
    return value


def poly_derivative(poly: list[int]) -> list[int]:
    return [index * poly[index] for index in range(1, len(poly))]


HERMITE = hermite_polynomials(6)
ck("01 Hermite coefficient sums 0..6", [sum(map(abs, p)) for p in HERMITE] == [1, 1, 2, 4, 10, 26, 76])
ck(
    "02 Hermite derivative recurrence",
    all(poly_derivative(HERMITE[n]) == [n * c for c in HERMITE[n - 1]] for n in range(1, 7)),
)


R_FAR = F(47, 2)
far_hermite_bound = all(
    sum(F(abs(coefficient)) * R_FAR**power for power, coefficient in enumerate(poly))
    <= 2 * R_FAR**6
    for poly in HERMITE
)
ck("03 |He_m(x)| <= 2|x|^6 for m<=6 and |x|>=23.5", far_hermite_bound)

# Exact sign/monotonicity facts on 0 <= x <= 1/2 imply |He_m(x)| <= 15.
half = F(1, 2)
inner_bound_facts = (
    poly_value(HERMITE[6], F(0)) == -15
    and poly_value(HERMITE[6], half) < 0
    and poly_value(HERMITE[5], half) < 15
    and F(3) - 6 * half**2 > 0       # He_4 >= 3-6x^2 > 0
    and F(15) - 10 * half**2 > 0    # He_5/x >= 15-10x^2 > 0
    and half**2 - 3 < 0             # He_3/x < 0
    and half**2 - 1 < 0             # He_2 < 0
)
ck("04 planar derivatives through order 6 bounded by 15 on |t|<=1/2", inner_bound_facts)


# Peano audit at d=1.  Homogeneity gives the same order and constant for all d.
# Terms are (coefficient, point, derivative order).
PEANO = {
    "L0": [(F(1, 2), -half, 0), (F(1, 2), half, 0)],
    "L1": [(F(-1), -half, 0), (F(1), half, 0)],
    "L2": [(F(-1), -half, 1), (F(1), half, 1)],
    "L3": [(F(12), -half, 0), (F(6), -half, 1), (F(-12), half, 0), (F(6), half, 1)],
    "Ty0": [(F(1, 2), -half, 0), (F(1, 2), half, 0)],
    "Ty1": [(F(-1), -half, 0), (F(1), half, 0)],
    "Tz0": [(F(1, 2), -half, 0), (F(1, 2), half, 0)],
    "Tz1": [(F(-1), -half, 0), (F(1), half, 0)],
}


def monomial_action(terms: list[tuple[F, F, int]], degree: int) -> F:
    total = F(0)
    for coefficient, point, derivative_order in terms:
        if degree >= derivative_order:
            total += (
                coefficient
                * F(factorial(degree), factorial(degree - derivative_order))
                * point ** (degree - derivative_order)
            )
    return total


def peano_order_constant(terms: list[tuple[F, F, int]]) -> tuple[int, F]:
    order = next(degree for degree in range(10) if monomial_action(terms, degree))
    constant = sum(
        (
            abs(coefficient)
            * abs(point) ** (order - derivative_order)
            / factorial(order - derivative_order)
        )
        for coefficient, point, derivative_order in terms
        if order >= derivative_order
    )
    return order, constant


peano_values = {name: peano_order_constant(terms) for name, terms in PEANO.items()}
ck(
    "05 stable-pin Peano orders",
    [peano_values[name][0] for name in PEANO] == [0, 1, 2, 3, 0, 1, 0, 1],
)
ck(
    "06 stable-pin Peano constants",
    [peano_values[name][1] for name in PEANO] == [F(1), F(1), F(1), F(2), F(1), F(1), F(1), F(1)],
)
ck("07 derivative ceilings x<=6, transverse<=4", max(v[0] for v in peano_values.values()) * 2 == 6)
ck("08 Peano/output coefficient-product ceiling", max(v[1] for v in peano_values.values()) ** 2 == 4)


# Exact exponential gate.  e > sum_{j=0}^6 1/j! = 1957/720.  Raising the
# target inequality to the eighth power removes the fractional exponent 1/8.
e_lower = sum((F(1, factorial(j)) for j in range(7)), F(0))
ck("09 exact lower bound e > 1957/720", e_lower == F(1957, 720))
ck(
    "10 exp(-2209/8) < 5/(4*10^120) by integer powers",
    e_lower**2209 > F(4 * 10**120, 5) ** 8,
)

A = F(5, 4 * 10**120)
geometric_tail_gate = F(64) * A**2 < F(1, 10**200)
ck("11 n>=2 image tail is geometrically negligible", geometric_tail_gate)
ck(
    "12 sum n^6 exp(-2209 n^2/8) < 2A",
    A + (F(64) * A**2) ** 2 / (1 - F(64) * A**2) < 2 * A,
)

IMAGE_1D = 8 * 25**6 * A
NORMALIZED_1D = 16 * IMAGE_1D
DERIVATIVE_3D = 721 * NORMALIZED_1D
JOINT_ENTRY = 4 * DERIVATIVE_3D
GAMMA_BOUND = 8400 * JOINT_ENTRY
DET_BOUND = 403200 * JOINT_ENTRY

ck("13 one-dimensional all-image derivative bound < 1e-110", IMAGE_1D < F(1, 10**110))
ck("14 normalized one-dimensional derivative bound < 1e-109", NORMALIZED_1D < F(1, 10**109))
ck("15 stable joint covariance entry bound < 2e-106", JOINT_ENTRY < F(2, 10**106))


# Exact alternating-series floor for the planar stable pin Gram.
def odd_coefficient(n: int) -> int:
    return 2**n - 2 * n


def h33_coefficient(n: int) -> int:
    return 4 * n * n - 10 * n + 4


odd_decrease = all(
    odd_coefficient(n + 1) <= (n + 1) * odd_coefficient(n)
    for n in range(3, 200)
)
h33_decrease = all(
    h33_coefficient(n + 1) <= (n + 1) * h33_coefficient(n)
    for n in range(3, 200)
)
even_decrease = all(
    (2 * (n + 1) + 2 ** (n + 1)) <= (n + 1) * (2 * n + 2**n)
    for n in range(1, 200)
)
ck("16 odd-block alternating coefficients decrease", odd_decrease)
ck("17 L3-variance alternating coefficients decrease", h33_decrease)
ck("18 even-block alternating coefficients decrease", even_decrease)
ck("19 even-block determinant floor 7/4", F(2) - 2 * F(1, 8) >= F(7, 4))
ck("20 odd-block determinant floor 21/4", F(6) - 6 * F(1, 8) >= F(21, 4))
PIN_FLOOR = F(21, 64)
ck("21 stable planar pin eigenvalue floor 21/64", F(21, 4) / 16 == PIN_FLOOR)
ck("22 periodized pin inverse Neumann gate", F(64, 21) * 8 * JOINT_ENTRY < F(1, 100))
ck("23 conditional covariance transfer bound < 1e-102", GAMMA_BOUND < F(1, 10**102))
ck("24 determinant transfer bound < 1e-100", DET_BOUND < F(1, 10**100))


def sci_fraction(value: F) -> str:
    getcontext().prec = 50
    decimal_value = D(value.numerator) / D(value.denominator)
    return f"{decimal_value:.12E}"


print(f"ALL_IMAGE_1D_BOUND={sci_fraction(IMAGE_1D)}")
print(f"STABLE_JOINT_ENTRY_BOUND={sci_fraction(JOINT_ENTRY)}")
print(f"CONDITIONAL_GAMMA_BOUND={sci_fraction(GAMMA_BOUND)}")
print(f"DETERMINANT_TRANSFER_BOUND={sci_fraction(DET_BOUND)}")


# -------------------------------------------------------------------------
# Independent 180-digit direct regression
# -------------------------------------------------------------------------
getcontext().prec = 180
SIDE = D(24)


def he_decimal(order: int, x: D) -> D:
    if order == 0:
        return D(1)
    if order == 1:
        return x
    previous, current = D(1), x
    for n in range(1, order):
        previous, current = current, x * current - D(n) * previous
    return current


def kernel_derivative(order: int, t: D, periodized: bool) -> D:
    sign = D(-1) if order % 2 else D(1)

    def raw(x: D) -> D:
        return sign * he_decimal(order, x) * (-(x * x) / 2).exp()

    if not periodized:
        return raw(t)
    normalizer = sum(((-((SIDE * n) ** 2) / 2).exp() for n in range(-2, 3)), D(0))
    return sum((raw(t + SIDE * n) for n in range(-2, 3)), D(0)) / normalizer


Term = tuple[D, tuple[D, D, D], tuple[int, int, int]]


def term_covariance(left: Term, right: Term, periodized: bool) -> D:
    coefficient_l, point_l, alpha = left
    coefficient_r, point_r, beta = right
    displacement = [point_l[i] - point_r[i] for i in range(3)]
    value = coefficient_l * coefficient_r * (D(-1) if sum(beta) % 2 else D(1))
    for i in range(3):
        value *= kernel_derivative(alpha[i] + beta[i], displacement[i], periodized)
    return value


def linear_covariance(left: list[Term], right: list[Term], periodized: bool) -> D:
    return sum((term_covariance(a, b, periodized) for a in left for b in right), D(0))


def stable_pin_rows(distance: D) -> tuple[tuple[D, D, D], list[list[Term]]]:
    h = distance / 2
    zero = D(0)
    m = (-h, zero, zero)
    s = (h, zero, zero)

    def one(coefficient: D, point: tuple[D, D, D], alpha: tuple[int, int, int]) -> list[Term]:
        return [(coefficient, point, alpha)]

    l0 = one(D(".5"), m, (0, 0, 0)) + one(D(".5"), s, (0, 0, 0))
    l1 = one(-1 / distance, m, (0, 0, 0)) + one(1 / distance, s, (0, 0, 0))
    l2 = one(-1 / distance, m, (1, 0, 0)) + one(1 / distance, s, (1, 0, 0))
    l3 = (
        one(12 / distance**3, m, (0, 0, 0))
        + one(-12 / distance**3, s, (0, 0, 0))
        + one(6 / distance**2, m, (1, 0, 0))
        + one(6 / distance**2, s, (1, 0, 0))
    )
    ty0 = one(D(".5"), m, (0, 1, 0)) + one(D(".5"), s, (0, 1, 0))
    ty1 = one(-1 / distance, m, (0, 1, 0)) + one(1 / distance, s, (0, 1, 0))
    tz0 = one(D(".5"), m, (0, 0, 1)) + one(D(".5"), s, (0, 0, 1))
    tz1 = one(-1 / distance, m, (0, 0, 1)) + one(1 / distance, s, (0, 0, 1))
    return m, [l0, l1, l2, l3, ty0, ty1, tz0, tz1]


def solve(matrix: list[list[D]], rhs: list[list[D]]) -> list[list[D]]:
    n = len(matrix)
    columns = len(rhs[0])
    augmented = [matrix[i][:] + rhs[i][:] for i in range(n)]
    for column in range(n):
        pivot_row = max(range(column, n), key=lambda row: abs(augmented[row][column]))
        if augmented[pivot_row][column] == 0:
            raise ArithmeticError("singular pin matrix")
        augmented[column], augmented[pivot_row] = augmented[pivot_row], augmented[column]
        pivot = augmented[column][column]
        for j in range(column, n + columns):
            augmented[column][j] /= pivot
        for row in range(n):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                for j in range(column, n + columns):
                    augmented[row][j] -= factor * augmented[column][j]
    return [row[n:] for row in augmented]


def conditioned_gamma(distance: D, vector: tuple[D, D, D], periodized: bool) -> list[list[D]]:
    point, pins = stable_pin_rows(distance)
    outputs: list[list[Term]] = []
    for i in range(3):
        terms: list[Term] = []
        for j in range(3):
            alpha = [0, 0, 0]
            alpha[i] += 1
            alpha[j] += 1
            terms.append((vector[j], point, tuple(alpha)))
        outputs.append(terms)
    p = [[linear_covariance(pins[i], pins[j], periodized) for j in range(8)] for i in range(8)]
    c = [[linear_covariance(pins[i], outputs[j], periodized) for j in range(3)] for i in range(8)]
    s = [[linear_covariance(outputs[i], outputs[j], periodized) for j in range(3)] for i in range(3)]
    regression = solve(p, c)
    return [
        [
            s[i][j] - sum((c[k][i] * regression[k][j] for k in range(8)), D(0))
            for j in range(3)
        ]
        for i in range(3)
    ]


def determinant3(matrix: list[list[D]]) -> D:
    return (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


def corrected_planar_formula(distance: D, vector: tuple[D, D, D]) -> D:
    q = (-(distance * distance) / 2).exp()
    g = (distance**2 * q**2 + q**2 - 1) / (q**2 - 1)
    v = (
        -distance**6 * q**2
        + distance**4 * q**4
        + distance**4 * q**2
        + 4 * distance**2 * q**4
        - 4 * distance**2 * q**2
        + 2 * q**4
        - 4 * q**2
        + 2
    ) / ((-distance**2 * q + q**2 - 1) * (distance**2 * q + q**2 - 1))
    u = vector[0] ** 2
    p = vector[1] ** 2 + vector[2] ** 2
    return (p + g * u) * (2 * g * p**2 + 2 * v * u * p + g * v * u**2)


root3 = D(3).sqrt()
root98 = D(".98").sqrt()
directions = {
    "axis": (D(1), D(0), D(0)),
    "transverse": (D(0), D(1), D(0)),
    "diagonal": (D(1) / root3, D(1) / root3, D(1) / root3),
    "skew": (D(".9") / root98, D(".1") / root98, D(".4") / root98),
}
probes = [
    ("P1", D(".5"), "axis"),
    ("P2", D(".5"), "transverse"),
    ("P3", D(".5"), "diagonal"),
    ("P4", D(".25"), "skew"),
    ("P5", D(".1"), "axis"),
    ("P6", D(".01"), "axis"),
    ("P7", D(".01"), "diagonal"),
    ("P8", D(".001"), "axis"),
]

numeric_gamma_bound = D(GAMMA_BOUND.numerator) / D(GAMMA_BOUND.denominator)
numeric_det_bound = D(DET_BOUND.numerator) / D(DET_BOUND.denominator)
max_entry_error = D(0)
max_det_error = D(0)
for label, distance, direction_name in probes:
    vector = directions[direction_name]
    planar = conditioned_gamma(distance, vector, False)
    periodized = conditioned_gamma(distance, vector, True)
    planar_det = determinant3(planar)
    periodized_det = determinant3(periodized)
    formula_det = corrected_planar_formula(distance, vector)
    entry_error = max(
        abs(periodized[i][j] - planar[i][j]) for i in range(3) for j in range(3)
    )
    det_error = abs(periodized_det - planar_det)
    max_entry_error = max(max_entry_error, entry_error)
    max_det_error = max(max_det_error, det_error)
    scale = max(D(1), abs(planar_det), abs(formula_det))
    ck(f"{label} planar Schur determinant = corrected G.9.1", abs(planar_det - formula_det) < D("1e-145") * scale)
    ck(f"{label} periodized covariance inside analytic bound", entry_error < numeric_gamma_bound)
    ck(f"{label} periodized determinant inside analytic bound", det_error < numeric_det_bound)
    ck(f"{label} conditional determinants positive", planar_det > 0 and periodized_det > 0)
    print(
        f"PROBE {label} d={distance} u={direction_name} "
        f"entry_error={entry_error:.18E} det_error={det_error:.18E}"
    )

ck("57 maximum probe covariance error below theorem bound", max_entry_error < numeric_gamma_bound)
ck("58 maximum probe determinant error below theorem bound", max_det_error < numeric_det_bound)
ck("59 forced-failure mutation is off", os.environ.get("G9_2_FORCE_FAILURE") != "1")

passed = sum(checks)
print(f"g9.2 reconstructed transfer: {passed}/{len(checks)} checks passed")
print("SCOPE LIMIT: reconstructed replacement on 0<d<=1/2 and |u|=1;")
print("no original-carrier identity, finite-rho remainder, RP-C/RP-S closure, or promotion claim.")
if passed == len(checks):
    print("ALL_ASSERTIONS_PASS")
    raise SystemExit(0)
raise SystemExit(1)

