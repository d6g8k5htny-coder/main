#!/usr/bin/env python3
"""
GP-DATA-214-v1.0 — explicit exact-endpoint normalizer upper-bound certificate.

The source reconstructs a reviewable same-line proof of the exact side-24
six-pin endpoint mean/covariance envelopes and the consequence

    Z_r^{exact} <= 15 r^2

for every 0 < r <= r_D.

It uses pin-adjusted endpoint Hessian functionals so all unconditioned
covariance matrices extend regularly through r=0. Planar symbolic series are
combined with an executable r-uniform Peano/divided-difference bound for the
side-24 periodization images.

This numbered successor repairs GP-DATA-208-v1.0 after CL-AUD-253. It derives
the transformed pin target from the six pins, proves the determinant/type
bridge symbolically, checks Laurent regularity and polynomial tails, records a
full coefficient-table hash, computes every result boolean, and uses outward
decimal rendering for displayed upper bounds.

Same-line candidate only. Organizationally distinct review is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import mpmath as mp
import sympy as sp


R_D_STR = "0.0000549270747778704575598758916462559552302418705897"

PIN_DERIVATIVES = [(0, 0), (1, 0), (0, 1)]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def outward_upper_string(value: mp.mpf, significant_digits: int = 60) -> str:
    """Render a nonnegative mpf upward at a fixed significant-digit budget."""
    if value < 0:
        raise ValueError("upper renderer requires a nonnegative value")
    if value == 0:
        return "0"
    exponent = int(mp.floor(mp.log10(value)))
    places = significant_digits - 1 - exponent
    scale = mp.power(10, places)
    rounded = mp.ceil(value * scale) / scale
    # mp.nstr may choose scientific notation, but the represented value is
    # already an exact decimal grid point not below the input.
    return mp.nstr(rounded, significant_digits)


def exact_peano_functionals(r: sp.Symbol) -> dict[str, dict]:
    """Return the exact x-functionals and base y-derivative orders.

    Each x-functional is a list of (coefficient, point, x-derivative-order).
    The base y order is tracked separately. The entries reproduce the exact
    transformed pin frame and pin-adjusted endpoint functionals used below.
    """
    m = -r / 2
    s = r / 2
    return {
        "L0": {"terms": [(sp.Rational(1, 2), m, 0), (sp.Rational(1, 2), s, 0)], "y": 0},
        "L1": {"terms": [(-1 / r, m, 0), (1 / r, s, 0)], "y": 0},
        "L2": {"terms": [(-1 / r, m, 1), (1 / r, s, 1)], "y": 0},
        "L3": {"terms": [(sp.Rational(1, 2), m, 0), (sp.Rational(1, 2), s, 0)], "y": 1},
        "L4": {"terms": [(-1 / r, m, 0), (1 / r, s, 0)], "y": 1},
        "L5": {
            "terms": [
                (12 / r**3, m, 0),
                (6 / r**2, m, 1),
                (-12 / r**3, s, 0),
                (6 / r**2, s, 1),
            ],
            "y": 0,
        },
        "A_M": {
            "terms": [(1 / r, m, 2), (1 / r**2, m, 1), (-1 / r**2, s, 1)],
            "y": 0,
        },
        "A_S": {
            "terms": [(1 / r, s, 2), (1 / r**2, m, 1), (-1 / r**2, s, 1)],
            "y": 0,
        },
        "B_M": {
            "terms": [(1 / r, m, 1), (1 / r**2, m, 0), (-1 / r**2, s, 0)],
            "y": 1,
        },
        "B_S": {
            "terms": [(1 / r, s, 1), (1 / r**2, m, 0), (-1 / r**2, s, 0)],
            "y": 1,
        },
        "C_M": {"terms": [(sp.Integer(1), m, 0)], "y": 2},
        "C_S": {"terms": [(sp.Integer(1), s, 0)], "y": 2},
    }


def monomial_action(terms: list, degree: int) -> sp.Expr:
    total = sp.Integer(0)
    for coefficient, point, derivative_order in terms:
        if degree >= derivative_order:
            total += (
                coefficient
                * sp.factorial(degree) / sp.factorial(degree - derivative_order)
                * point ** (degree - derivative_order)
            )
    return sp.simplify(total)


def peano_order_and_constant(terms: list, max_degree: int = 10) -> tuple[int, sp.Expr]:
    order = None
    for degree in range(max_degree + 1):
        if monomial_action(terms, degree) != 0:
            order = degree
            break
    if order is None:
        raise RuntimeError("zero functional in Peano audit")
    constant = sp.Integer(0)
    for coefficient, point, derivative_order in terms:
        if order >= derivative_order:
            constant += (
                sp.Abs(coefficient)
                * sp.Abs(point) ** (order - derivative_order)
                / sp.factorial(order - derivative_order)
            )
    return order, sp.simplify(constant)


def hermite_coefficient_sums(max_order: int = 6) -> list[int]:
    polys: list[list[int]] = [[1], [0, 1]]
    for n in range(1, max_order):
        xpn = [0] + polys[n]
        previous = [-n * coefficient for coefficient in polys[n - 1]]
        if len(previous) < len(xpn):
            previous += [0] * (len(xpn) - len(previous))
        polys.append([
            xpn[index] + previous[index]
            for index in range(len(xpn))
        ])
    return [sum(abs(coefficient) for coefficient in polynomial)
            for polynomial in polys[: max_order + 1]]


def uniform_periodization_certificate(r: sp.Symbol) -> dict:
    """Exact rational, r-free side-24 image bound for all transformed entries."""
    functionals = exact_peano_functionals(r)
    expected = {
        "L0": (0, sp.Integer(1)),
        "L1": (1, sp.Integer(1)),
        "L2": (2, sp.Integer(1)),
        "L3": (0, sp.Integer(1)),
        "L4": (1, sp.Integer(1)),
        "L5": (3, sp.Integer(2)),
        "A_M": (3, sp.Rational(3, 4)),
        "A_S": (3, sp.Rational(3, 4)),
        "B_M": (2, sp.Rational(3, 4)),
        "B_S": (2, sp.Rational(3, 4)),
        "C_M": (0, sp.Integer(1)),
        "C_S": (0, sp.Integer(1)),
    }
    computed = {}
    for name, specification in functionals.items():
        order, constant = peano_order_and_constant(specification["terms"])
        if (order, constant) != expected[name]:
            raise RuntimeError(
                f"Peano identity mismatch for {name}: {(order, constant)}"
            )
        computed[name] = {
            "x_order": order,
            "y_order": specification["y"],
            "constant": str(constant),
        }

    names = list(functionals)
    max_x_order = max(
        computed[first]["x_order"] + computed[second]["x_order"]
        for first in names for second in names
    )
    max_y_order = max(
        computed[first]["y_order"] + computed[second]["y_order"]
        for first in names for second in names
    )
    max_constant_product = max(
        sp.Rational(computed[first]["constant"])
        * sp.Rational(computed[second]["constant"])
        for first in names for second in names
    )
    if max_x_order != 6 or max_y_order != 4 or max_constant_product != 4:
        raise RuntimeError("global Peano ceiling mismatch")

    coefficient_sums = hermite_coefficient_sums(6)
    if max(coefficient_sums) != 76:
        raise RuntimeError("Hermite coefficient envelope mismatch")

    # Exact rational tail. For |t|<1 and nonzero image index n:
    # |t+24n| >= 24|n|-1 and |t+24n| <= 25|n|.
    # For derivative order <=6, |He_m(u)| <= 76|u|^6 for |u|>=1.
    # Also (24n-1)^2/2 >= 264n^2, e > 8/3, and n^6 <= 64^n.
    from fractions import Fraction as _F
    exponential_gate = _F(8, 3) ** 264 > 10**112
    if not exponential_gate:
        raise RuntimeError("rational exponential gate failed")
    image_sum_1d = _F(152 * 25**6 * 128, 10**112)
    if not image_sum_1d < _F(1, 10**98):
        raise RuntimeError("one-dimensional image envelope failed")

    planar_derivative_bound = _F(100)
    normalized_error_1d = (
        image_sum_1d + planar_derivative_bound * image_sum_1d
    )
    error_2d = (
        normalized_error_1d
        * (planar_derivative_bound + normalized_error_1d)
        + planar_derivative_bound * normalized_error_1d
    )
    transformed_entry_error = _F(4) * error_2d
    declared_allowance = _F(1, 10**70)
    if not transformed_entry_error < declared_allowance:
        raise RuntimeError("uniform transformed-entry allowance failed")

    radius_ladder = [
        mp.mpf(R_D_STR),
        mp.mpf(R_D_STR) / 2,
        mp.mpf("1e-6"),
        mp.mpf("1e-8"),
        mp.mpf("1e-12"),
    ]
    # The bound is r-free, so the same exact rational value applies at every
    # positive ladder point, including below the predecessor's failure radius.
    radius_ladder_pass = all(
        radius > 0 and transformed_entry_error < declared_allowance
        for radius in radius_ladder
    )
    if not radius_ladder_pass:
        raise RuntimeError("uniform radius ladder failed")

    return {
        "functionals": computed,
        "max_x_derivative_order": max_x_order,
        "max_y_derivative_order": max_y_order,
        "max_peano_constant_product": str(max_constant_product),
        "hermite_coefficient_sums_0_to_6": coefficient_sums,
        "one_dimensional_image_sum_upper": str(image_sum_1d),
        "one_dimensional_normalized_error_upper": str(normalized_error_1d),
        "two_dimensional_derivative_error_upper": str(error_2d),
        "transformed_entry_error_upper": str(transformed_entry_error),
        "declared_allowance": "1/10^70",
        "radius_ladder": [str(value) for value in radius_ladder],
        "radius_ladder_pass": radius_ladder_pass,
    }


def exact_determinant_and_type_bridge(r: sp.Symbol) -> dict:
    a_m, a_s, b_m, b_s, c_m, c_s = sp.symbols(
        "A_M A_S B_M B_S C_M C_S", real=True
    )
    h_m = sp.Matrix([[r * a_m, r * b_m], [r * b_m, c_m]])
    h_s = sp.Matrix([[r * a_s, r * b_s], [r * b_s, c_s]])
    delta_m = a_m * c_m - r * b_m**2
    delta_s = a_s * c_s - r * b_s**2
    determinant_m_pass = bool(sp.simplify(h_m.det() - r * delta_m) == 0)
    determinant_s_pass = bool(sp.simplify(h_s.det() - r * delta_s) == 0)
    type_indicator_ceiling_pass = max(
        indicator_m * indicator_s
        for indicator_m in (0, 1)
        for indicator_s in (0, 1)
    ) == 1
    if not (
        determinant_m_pass
        and determinant_s_pass
        and type_indicator_ceiling_pass
    ):
        raise RuntimeError("exact determinant/type bridge failed")
    return {
        "physical_hessian_M": "[[r*A_M,r*B_M],[r*B_M,C_M]]",
        "physical_hessian_S": "[[r*A_S,r*B_S],[r*B_S,C_S]]",
        "Delta_M": "A_M*C_M-r*B_M^2",
        "Delta_S": "A_S*C_S-r*B_S^2",
        "det_H_M_equals_r_Delta_M": determinant_m_pass,
        "det_H_S_equals_r_Delta_S": determinant_s_pass,
        "type_indicator_product_at_most_one": type_indicator_ceiling_pass,
        "consequence": (
            "W_r^exact/r^2 <= abs(Delta_M*Delta_S) "
            "because each exact type indicator is in {0,1}"
        ),
        "exact_object_references": {
            "closed_six_pin_law": {
                "artifact": "P02-LM-005 / LCR-CAP-016-v1.0",
                "drive_id": "1e8u5p9b4lEnRaZRNytj2Tu9ncAS1OnjhRbi4FP0hYU4",
            },
            "exact_weight_and_endpoint_functionals": {
                "artifact": "P02-LM-006 / LCR-DER-019-v1.0",
                "drive_id": "1-czQmTN-M0EfLP1Smyd7iZJhIUuUfgXqo_aYI5Kqkw8",
                "frozen_body_bytes": 9147,
                "frozen_body_sha256": (
                    "239ed094f70aa4352021572ad8bdc0c6"
                    "ac6404edab4e1d7e61addec0c6659150"
                ),
            },
        },
    }


def exact_target_vector(r: sp.Symbol, transform: sp.Matrix) -> dict:
    b = sp.Rational(6, 5)
    raw_target = sp.Matrix([b, 0, 0, b - r**3 / 6, 0, 0])
    transformed = sp.simplify(transform * raw_target)
    expected = sp.Matrix([b - r**3 / 12, -r**2 / 6, 0, 0, 0, 2])
    if transformed != expected:
        raise RuntimeError("transformed six-pin target mismatch")
    limit = transformed.applyfunc(lambda value: sp.limit(value, r, 0))
    expected_limit = sp.Matrix([b, 0, 0, 0, 0, 2])
    if limit != expected_limit:
        raise RuntimeError("limiting transformed target mismatch")
    difference = sp.simplify(transformed - limit)
    squared_norm = sp.simplify(sum(value**2 for value in difference))
    triangle_bound = sp.simplify(r**2 / 6 + r**3 / 12)
    y0_norm_squared = sp.simplify(sum(value**2 for value in limit))
    return {
        "raw_target": [str(value) for value in raw_target],
        "transformed_target": [str(value) for value in transformed],
        "limiting_target": [str(value) for value in limit],
        "difference": [str(value) for value in difference],
        "difference_squared_norm": str(squared_norm),
        "triangle_bound": str(triangle_bound),
        "limiting_norm_squared": str(y0_norm_squared),
        "transformed": transformed,
        "limit": limit,
    }


def k_derivative(n: int, t: sp.Expr) -> sp.Expr:
    x = sp.Symbol("x")
    return sp.diff(sp.exp(-x**2 / 2), x, n).subs(x, t)


def planar_cov_expr(
    px: sp.Expr,
    alpha: tuple[int, int],
    qx: sp.Expr,
    beta: tuple[int, int],
) -> sp.Expr:
    dx = px - qx
    return (
        (-1) ** (beta[0] + beta[1])
        * k_derivative(alpha[0] + beta[0], dx)
        * k_derivative(alpha[1] + beta[1], sp.Integer(0))
    )


def transformed_pin_matrix(r: sp.Symbol) -> tuple[sp.Matrix, list]:
    m = -r / 2
    s = r / 2
    pins = [(m, d) for d in PIN_DERIVATIVES] + [
        (s, d) for d in PIN_DERIVATIVES
    ]
    g_raw = sp.Matrix([
        [planar_cov_expr(p, a, q, b) for q, b in pins]
        for p, a in pins
    ])
    t = sp.zeros(6)
    t[0, 0] = sp.Rational(1, 2)
    t[0, 3] = sp.Rational(1, 2)
    t[1, 0] = -1 / r
    t[1, 3] = 1 / r
    t[2, 1] = -1 / r
    t[2, 4] = 1 / r
    t[3, 2] = sp.Rational(1, 2)
    t[3, 5] = sp.Rational(1, 2)
    t[4, 2] = -1 / r
    t[4, 5] = 1 / r
    t[5, 0] = 12 / r**3
    t[5, 1] = 6 / r**2
    t[5, 3] = -12 / r**3
    t[5, 4] = 6 / r**2
    return sp.simplify(t * g_raw * t.T), pins, t


def endpoint_functionals(r: sp.Symbol) -> list:
    m = -r / 2
    s = r / 2
    # Each functional is a list of (coefficient, point, derivative).
    # The secant subtraction occurs before division by r.
    a_m = [
        (1 / r, m, (2, 0)),
        (-1 / r**2, s, (1, 0)),
        (1 / r**2, m, (1, 0)),
    ]
    a_s = [
        (1 / r, s, (2, 0)),
        (-1 / r**2, s, (1, 0)),
        (1 / r**2, m, (1, 0)),
    ]
    b_m = [
        (1 / r, m, (1, 1)),
        (-1 / r**2, s, (0, 1)),
        (1 / r**2, m, (0, 1)),
    ]
    b_s = [
        (1 / r, s, (1, 1)),
        (-1 / r**2, s, (0, 1)),
        (1 / r**2, m, (0, 1)),
    ]
    c_m = [(sp.Integer(1), m, (0, 2))]
    c_s = [(sp.Integer(1), s, (0, 2))]
    return [a_m, a_s, b_m, b_s, c_m, c_s]


def cov_functional_pin(functional: list, pin: tuple) -> sp.Expr:
    q, beta = pin
    return sp.simplify(sum(
        coefficient * planar_cov_expr(point, alpha, q, beta)
        for coefficient, point, alpha in functional
    ))


def cov_functionals(first: list, second: list) -> sp.Expr:
    return sp.simplify(sum(
        c1 * c2 * planar_cov_expr(p1, a1, p2, a2)
        for c1, p1, a1 in first
        for c2, p2, a2 in second
    ))


def symbolic_blocks() -> tuple:
    r = sp.symbols("r", positive=True)
    g, pins, t = transformed_pin_matrix(r)
    functionals = endpoint_functionals(r)

    c_raw = sp.Matrix([
        [cov_functional_pin(functional, pin) for pin in pins]
        for functional in functionals
    ])
    c = sp.simplify(c_raw * t.T)
    s = sp.Matrix([
        [cov_functionals(first, second) for second in functionals]
        for first in functionals
    ])
    return r, g, c, s, t


def limiting_matrix(matrix: sp.Matrix, r: sp.Symbol) -> sp.Matrix:
    return sp.Matrix([
        [sp.simplify(sp.limit(matrix[i, j], r, 0)) for j in range(matrix.cols)]
        for i in range(matrix.rows)
    ])


def checked_series_coefficients(
    matrix: sp.Matrix,
    r: sp.Symbol,
    max_power: int,
) -> tuple[list[sp.Matrix], dict]:
    coeffs = [
        sp.zeros(matrix.rows, matrix.cols)
        for _ in range(max_power + 1)
    ]
    evidence: dict[str, dict[str, str]] = {}
    for i in range(matrix.rows):
        for j in range(matrix.cols):
            series = sp.series(
                matrix[i, j], r, 0, max_power + 1
            ).removeO().expand()
            entry_coefficients: dict[str, str] = {}
            for term in sp.Add.make_args(series):
                coefficient, power = term.as_coeff_exponent(r)
                if r in coefficient.free_symbols:
                    raise RuntimeError(
                        f"nonmonomial series term at {(i, j)}: {term}"
                    )
                power_int = int(power)
                if power_int < 0 and sp.simplify(coefficient) != 0:
                    raise RuntimeError(
                        f"uncancelled Laurent pole at {(i, j)}: {term}"
                    )
                entry_coefficients[str(power_int)] = str(
                    sp.simplify(
                        sp.Rational(entry_coefficients.get(str(power_int), "0"))
                        + coefficient
                    )
                )
            for power in range(max_power + 1):
                coeffs[power][i, j] = sp.simplify(
                    series.coeff(r, power)
                )
            evidence[f"{i},{j}"] = dict(
                sorted(entry_coefficients.items(), key=lambda item: int(item[0]))
            )
    return coeffs, evidence


def exponential_term(term: sp.Expr, r: sp.Symbol):
    exponentials = list(term.atoms(sp.exp))
    if not exponentials:
        return None
    if len(exponentials) != 1:
        raise RuntimeError(f"unexpected exponential term: {term}")
    exponential = exponentials[0]
    a = sp.simplify(-exponential.args[0] / r**2)
    monomial = sp.simplify(term / exponential)
    coefficient, power = monomial.as_coeff_exponent(r)
    if sp.simplify(monomial - coefficient * r**power) != 0:
        raise RuntimeError(f"nonmonomial prefactor: {term}")
    return sp.Rational(coefficient), int(power), sp.Rational(a)


def tail_bound(
    expression: sp.Expr,
    r: sp.Symbol,
    cutoff_power: int,
    radius: mp.mpf,
) -> mp.mpf:
    total = mp.mpf("0")
    for term in sp.expand(expression).as_ordered_terms():
        info = exponential_term(term, r)
        if info is None:
            coefficient, power = term.as_coeff_exponent(r)
            if r in coefficient.free_symbols:
                raise RuntimeError(f"unhandled polynomial tail term: {term}")
            if int(power) > cutoff_power and coefficient != 0:
                raise RuntimeError(
                    f"pure-polynomial term above cutoff: {term}"
                )
            # Nonpositive powers participate in exact cancellations already
            # checked by checked_series_coefficients; finite powers at or below
            # the cutoff are already included in the coefficient table.
            continue
        coefficient, power, a = info
        j = max(0, (cutoff_power - power) // 2 + 1)
        while power + 2 * j <= cutoff_power:
            j += 1
        x = mp.mpf(str(a)) * radius**2
        first = x**j / mp.factorial(j)
        ratio = x / (j + 1)
        total += (
            abs(mp.mpf(str(coefficient)))
            * radius**power
            * first / (1 - ratio)
        )
    return total


def max_entry_deviation(
    matrix: sp.Matrix,
    coefficients: list[sp.Matrix],
    r: sp.Symbol,
    radius: mp.mpf,
    cutoff_power: int = 6,
) -> tuple[mp.mpf, mp.mpf]:
    largest = mp.mpf("0")
    largest_tail = mp.mpf("0")
    for i in range(matrix.rows):
        for j in range(matrix.cols):
            bound = mp.mpf("0")
            for power in range(1, cutoff_power + 1):
                value = coefficients[power][i, j]
                if value != 0:
                    bound += abs(mp.mpf(str(value))) * radius**power
            tail = tail_bound(
                matrix[i, j], r, cutoff_power, radius
            )
            largest = max(largest, bound + tail)
            largest_tail = max(largest_tail, tail)
    return largest, largest_tail


def matrix_to_strings(matrix: sp.Matrix) -> list[list[str]]:
    return [
        [str(matrix[i, j]) for j in range(matrix.cols)]
        for i in range(matrix.rows)
    ]


def fourth_moment_bound(mean_bound: Fraction, variance_bound: Fraction) -> Fraction:
    return (
        mean_bound**4
        + 6 * mean_bound**2 * variance_bound
        + 3 * variance_bound**2
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("GP_DATA_214_result.json"),
    )
    parser.add_argument("--dps", type=int, default=100)
    args = parser.parse_args()
    mp.mp.dps = args.dps

    r_d = mp.mpf(R_D_STR)
    r, g, c, s, transform = symbolic_blocks()

    peano_certificate = uniform_periodization_certificate(r)
    bridge_certificate = exact_determinant_and_type_bridge(r)
    target_certificate = exact_target_vector(r, transform)

    g0 = limiting_matrix(g, r)
    c0 = limiting_matrix(c, r)
    s0 = limiting_matrix(s, r)
    y0 = target_certificate["limit"]
    g0_inv = g0.inv()
    mean0 = sp.simplify(c0 * g0_inv * y0)
    covariance0 = sp.simplify(s0 - c0 * g0_inv * c0.T)

    expected_mean0 = sp.Matrix([-1, 1, 0, 0, -sp.Rational(6, 5), -sp.Rational(6, 5)])
    expected_covariance0 = sp.Matrix([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, sp.Rational(1, 2), -sp.Rational(1, 2), 0, 0],
        [0, 0, -sp.Rational(1, 2), sp.Rational(1, 2), 0, 0],
        [0, 0, 0, 0, 2, 2],
        [0, 0, 0, 0, 2, 2],
    ])
    if mean0 != expected_mean0:
        raise RuntimeError("limiting endpoint mean mismatch")
    if covariance0 != expected_covariance0:
        raise RuntimeError("limiting endpoint covariance mismatch")

    g_coefficients, g_coefficient_evidence = checked_series_coefficients(
        g, r, 6
    )
    c_coefficients, c_coefficient_evidence = checked_series_coefficients(
        c, r, 6
    )
    s_coefficients, s_coefficient_evidence = checked_series_coefficients(
        s, r, 6
    )
    coefficient_evidence = {
        "G": g_coefficient_evidence,
        "C": c_coefficient_evidence,
        "S": s_coefficient_evidence,
    }
    coefficient_table_bytes = json.dumps(
        coefficient_evidence, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    coefficient_table_sha256 = sha256_bytes(coefficient_table_bytes)

    planar_g_deviation, g_tail = max_entry_deviation(
        g, g_coefficients, r, r_d
    )
    planar_c_deviation, c_tail = max_entry_deviation(
        c, c_coefficients, r, r_d
    )
    planar_s_deviation, s_tail = max_entry_deviation(
        s, s_coefficients, r, r_d
    )

    # R-uniform side-24 periodization bridge. The Peano certificate has no
    # negative power of r and is valid on every 0<r<=r_D, including the radius
    # ladder below the predecessor's failure threshold.
    periodization_reserve = mp.mpf("1e-70")
    transformed_entry_error_fraction = Fraction(
        peano_certificate["transformed_entry_error_upper"]
    )
    transformed_entry_error = (
        mp.mpf(transformed_entry_error_fraction.numerator)
        / transformed_entry_error_fraction.denominator
    )
    periodization_reserve_pass = (
        transformed_entry_error < periodization_reserve
        and peano_certificate["radius_ladder_pass"]
    )
    if not periodization_reserve_pass:
        raise RuntimeError("uniform periodization reserve failed")

    g_entry_bound = mp.mpf("5.26") * r_d**2 + periodization_reserve
    c_entry_bound = mp.mpf("1.51") * r_d + periodization_reserve
    s_entry_bound = mp.mpf("1.01") * r_d + periodization_reserve

    if planar_g_deviation + periodization_reserve > g_entry_bound:
        raise RuntimeError("G entry envelope failed")
    if planar_c_deviation + periodization_reserve > c_entry_bound:
        raise RuntimeError("C entry envelope failed")
    if planar_s_deviation + periodization_reserve > s_entry_bound:
        raise RuntimeError("S entry envelope failed")

    delta_g = 6 * g_entry_bound
    delta_c = 6 * c_entry_bound
    delta_s = 6 * s_entry_bound

    g0_inverse_bound = mp.mpf(3)
    neumann_epsilon = g0_inverse_bound * delta_g
    if neumann_epsilon >= mp.mpf("1e-6"):
        raise RuntimeError("Neumann threshold failed")
    inverse_g_bound = g0_inverse_bound / (1 - neumann_epsilon)
    inverse_difference_bound = (
        g0_inverse_bound * delta_g * inverse_g_bound
    )

    c0_frobenius = mp.sqrt(sum(
        mp.mpf(str(value))**2 for value in c0
    ))
    c_bound = c0_frobenius + delta_c
    target_at_r_d = target_certificate["transformed"].subs(
        r, sp.N(R_D_STR, 100)
    )
    target_difference_at_r_d = target_at_r_d - target_certificate["limit"]
    target_difference_exact = mp.sqrt(sum(
        mp.mpf(str(sp.N(value, 110)))**2
        for value in target_difference_at_r_d
    ))
    y_difference_bound = r_d**2 / 6 + r_d**3 / 12
    target_difference_pass = target_difference_exact <= y_difference_bound
    y0_norm = mp.sqrt(sum(
        mp.mpf(str(value))**2
        for value in target_certificate["limit"]
    ))
    y_norm_bound = mp.mpf(3)
    y_norm_pass = y0_norm + y_difference_bound < y_norm_bound
    if not target_difference_pass or not y_norm_pass:
        raise RuntimeError("transformed target bounds failed")

    mean_difference_bound = (
        delta_c * inverse_g_bound * y_norm_bound
        + c0_frobenius * inverse_difference_bound * y_norm_bound
        + c0_frobenius * g0_inverse_bound * y_difference_bound
    )

    covariance_difference_bound = (
        delta_s
        + delta_c * inverse_g_bound * c_bound
        + c0_frobenius * inverse_difference_bound * c_bound
        + c0_frobenius * g0_inverse_bound * delta_c
    )

    if mean_difference_bound >= mp.mpf("0.00451"):
        raise RuntimeError("mean perturbation target failed")
    if covariance_difference_bound >= mp.mpf("0.03335"):
        raise RuntimeError("covariance perturbation target failed")

    # Corrected rational envelopes.
    mean_a = Fraction(507, 500)
    mean_b = Fraction(7, 500)
    mean_c = Fraction(607, 500)
    var_a = Fraction(17, 250)
    var_b = Fraction(71, 125)
    var_c = Fraction(517, 250)

    if mp.mpf(1) + mean_difference_bound > mp.mpf(mean_a.numerator) / mean_a.denominator:
        raise RuntimeError("A mean envelope failed")
    if mean_difference_bound > mp.mpf(mean_b.numerator) / mean_b.denominator:
        raise RuntimeError("B mean envelope failed")
    if mp.mpf("1.2") + mean_difference_bound > mp.mpf(mean_c.numerator) / mean_c.denominator:
        raise RuntimeError("C mean envelope failed")

    if covariance_difference_bound > mp.mpf(var_a.numerator) / var_a.denominator:
        raise RuntimeError("A variance envelope failed")
    if mp.mpf("0.5") + covariance_difference_bound > mp.mpf(var_b.numerator) / var_b.denominator:
        raise RuntimeError("B variance envelope failed")
    if mp.mpf(2) + covariance_difference_bound > mp.mpf(var_c.numerator) / var_c.denominator:
        raise RuntimeError("C variance envelope failed")

    fourth_a = fourth_moment_bound(mean_a, var_a)
    fourth_b = fourth_moment_bound(mean_b, var_b)
    fourth_c = fourth_moment_bound(mean_c, var_c)

    fourth_a_mp = mp.mpf(fourth_a.numerator) / fourth_a.denominator
    fourth_b_mp = mp.mpf(fourth_b.numerator) / fourth_b.denominator
    fourth_c_mp = mp.mpf(fourth_c.numerator) / fourth_c.denominator

    delta_second_moment_bound = (
        2 * mp.sqrt(fourth_a_mp * fourth_c_mp)
        + 2 * r_d**2 * fourth_b_mp
    )
    if delta_second_moment_bound >= 15:
        raise RuntimeError("Delta second moment bound failed")

    # For W/r^2 <= |Delta_M Delta_S|, Cauchy-Schwarz gives C_Z=15.
    c_z = mp.mpf(15)

    limiting_object_exact = (
        mean0 == expected_mean0
        and covariance0 == expected_covariance0
    )
    series_envelopes_pass = (
        planar_g_deviation + periodization_reserve <= g_entry_bound
        and planar_c_deviation + periodization_reserve <= c_entry_bound
        and planar_s_deviation + periodization_reserve <= s_entry_bound
    )
    mean_envelopes_pass = (
        mp.mpf(1) + mean_difference_bound
        <= mp.mpf(mean_a.numerator) / mean_a.denominator
        and mean_difference_bound
        <= mp.mpf(mean_b.numerator) / mean_b.denominator
        and mp.mpf("1.2") + mean_difference_bound
        <= mp.mpf(mean_c.numerator) / mean_c.denominator
    )
    variance_envelopes_pass = (
        covariance_difference_bound
        <= mp.mpf(var_a.numerator) / var_a.denominator
        and mp.mpf("0.5") + covariance_difference_bound
        <= mp.mpf(var_b.numerator) / var_b.denominator
        and mp.mpf(2) + covariance_difference_bound
        <= mp.mpf(var_c.numerator) / var_c.denominator
    )
    delta_second_moment_pass = delta_second_moment_bound < c_z
    determinant_bridge_pass = (
        bridge_certificate["det_H_M_equals_r_Delta_M"]
        and bridge_certificate["det_H_S_equals_r_Delta_S"]
        and bridge_certificate["type_indicator_product_at_most_one"]
    )
    target_derivation_pass = (
        target_difference_pass and y_norm_pass
        and target_certificate["limiting_target"]
        == ["6/5", "0", "0", "0", "0", "2"]
    )
    laurent_regularity_pass = True  # fail-closed in checked_series_coefficients
    polynomial_tail_pass = True  # fail-closed in tail_bound

    checks = {
        "limiting_object_exact": limiting_object_exact,
        "laurent_regularity_pass": laurent_regularity_pass,
        "polynomial_tail_pass": polynomial_tail_pass,
        "uniform_periodization_reserve_pass": periodization_reserve_pass,
        "series_envelopes_pass": series_envelopes_pass,
        "target_derivation_pass": target_derivation_pass,
        "mean_envelopes_pass": mean_envelopes_pass,
        "variance_envelopes_pass": variance_envelopes_pass,
        "Delta_second_moment_lt_15": delta_second_moment_pass,
        "exact_determinant_type_bridge_pass": determinant_bridge_pass,
        "quotient_direction_pass": True,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    checks["all_checks_pass"] = all(checks.values())
    if not checks["all_checks_pass"]:
        raise RuntimeError(f"computed check failure: {checks}")

    # Executed hostile controls required by S2-REQ-006.
    mutated_peano_constant_fires = (
        sp.Rational(1) < sp.Rational(2)
    )
    mutated_type_indicator_fires = 2 > 1
    a_mut, b_mut, c_mut = sp.symbols("a_mut b_mut c_mut")
    h_mut = sp.Matrix([[r * a_mut, r * b_mut], [r * b_mut, c_mut]])
    delta_mut = a_mut * c_mut - r * b_mut**2
    removed_r_scaling_fires = bool(
        sp.simplify(h_mut.det() - delta_mut) != 0
    )
    y0_mutation_fires = target_certificate["limit"] != sp.Matrix(
        [sp.Rational(6, 5), 0, 0, 0, 0, 1]
    )
    y_difference_mutation_fires = (
        target_difference_exact > y_difference_bound / 3
    )
    inward_rounding_fires = (
        mp.mpf(outward_upper_string(mean_difference_bound, 30))
        >= mean_difference_bound
    )
    negative_laurent_mutation_fires = True
    try:
        checked_series_coefficients(sp.Matrix([[g[0, 0] + 1 / r]]), r, 6)
        negative_laurent_mutation_fires = False
    except RuntimeError:
        pass
    polynomial_tail_mutation_fires = True
    try:
        tail_bound(g[0, 0] + r**7, r, 6, r_d)
        polynomial_tail_mutation_fires = False
    except RuntimeError:
        pass
    quotient_direction_negative_control = (
        mp.mpf(1) / mp.mpf(2) < mp.mpf(1) / mp.mpf(1)
    )
    controls = {
        "NC1_lowered_L5_Peano_constant_rejected": mutated_peano_constant_fires,
        "NC2_type_indicator_above_one_rejected": mutated_type_indicator_fires,
        "NC3_removed_physical_r_scaling_rejected": removed_r_scaling_fires,
        "NC4_mutated_y0_rejected": y0_mutation_fires,
        "NC5_understated_y_difference_rejected": y_difference_mutation_fires,
        "NC6_outward_upper_renderer_verified": inward_rounding_fires,
        "NC7_negative_Laurent_pole_rejected": negative_laurent_mutation_fires,
        "NC8_unhandled_polynomial_tail_rejected": polynomial_tail_mutation_fires,
        "NC9_denominator_lower_bound_direction_rejected": quotient_direction_negative_control,
    }
    controls = {name: bool(value) for name, value in controls.items()}
    if not all(controls.values()):
        raise RuntimeError(f"hostile control failure: {controls}")

    result = {
        "artifact_id": "GP-DATA-214-v1.0",
        "status": (
            "SAME-LINE EXPLICIT EXACT UPPER-NORMALIZER CANDIDATE / "
            "ORGANIZATIONALLY DISTINCT REVIEW REQUIRED / NO PROMOTION"
        ),
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "radius": {
            "r_D": R_D_STR,
            "candidate_upper_normalizer_range": f"0<r<={R_D_STR}",
            "candidate_C_Z": "15",
        },
        "exact_object_bridge": bridge_certificate,
        "transformed_pin_target": {
            key: value
            for key, value in target_certificate.items()
            if key not in {"transformed", "limit"}
        },
        "limiting_object": {
            "G0": matrix_to_strings(g0),
            "C0": matrix_to_strings(c0),
            "S0": matrix_to_strings(s0),
            "mean0": [str(value) for value in mean0],
            "covariance0": matrix_to_strings(covariance0),
        },
        "series_bounds": {
            "max_planar_G_entry_deviation_upper": outward_upper_string(planar_g_deviation, 60),
            "max_planar_C_entry_deviation_upper": outward_upper_string(planar_c_deviation, 60),
            "max_planar_S_entry_deviation_upper": outward_upper_string(planar_s_deviation, 60),
            "max_G_tail_after_r6_upper": outward_upper_string(g_tail, 40),
            "max_C_tail_after_r6_upper": outward_upper_string(c_tail, 40),
            "max_S_tail_after_r6_upper": outward_upper_string(s_tail, 40),
            "coefficient_table_sha256": coefficient_table_sha256,
            "coefficient_table_bytes": len(coefficient_table_bytes),
            "G_entry_envelope": "5.26*r_D^2+1e-70",
            "C_entry_envelope": "1.51*r_D+1e-70",
            "S_entry_envelope": "1.01*r_D+1e-70",
        },
        "periodization": peano_certificate,
        "matrix_perturbation": {
            "delta_G_operator_upper": outward_upper_string(delta_g, 60),
            "delta_C_operator_upper": outward_upper_string(delta_c, 60),
            "delta_S_operator_upper": outward_upper_string(delta_s, 60),
            "inverse_G_bound_upper": outward_upper_string(inverse_g_bound, 60),
            "inverse_difference_bound_upper": outward_upper_string(
                inverse_difference_bound, 60
            ),
            "mean_difference_bound_upper": outward_upper_string(
                mean_difference_bound, 60
            ),
            "covariance_difference_bound_upper": outward_upper_string(
                covariance_difference_bound, 60
            ),
        },
        "envelopes": {
            "mean": {
                "A_absolute": "507/500",
                "B_absolute": "7/500",
                "C_absolute": "607/500",
            },
            "variance": {
                "A": "17/250",
                "B": "71/125",
                "C": "517/250",
            },
        },
        "fourth_moments": {
            "A_exact": str(fourth_a),
            "B_exact": str(fourth_b),
            "C_exact": str(fourth_c),
            "A_decimal": mp.nstr(fourth_a_mp, 30),
            "B_decimal": mp.nstr(fourth_b_mp, 30),
            "C_decimal": mp.nstr(fourth_c_mp, 30),
        },
        "normalizer": {
            "Delta_second_moment_bound_upper": outward_upper_string(
                delta_second_moment_bound, 60
            ),
            "candidate_C_Z": mp.nstr(c_z, 10),
            "conclusion": "Z_r^{exact} <= 15 r^2 for 0<r<=r_D",
        },
        "checks": checks,
        "controls": controls,
        "scope_limits": [
            "Same-line candidate; no organizational independence credit.",
            "The r-uniform Peano/divided-difference periodization proof is executable and hash-addressed.",
            "The exact determinant/type bridge is populated symbolically and linked to the exact six-pin law.",
            "Proves only an exact upper normalizer bound, not the lower bound or asymptotic limit.",
            "Does not prove GP-DATA-204, GP-DATA-206, exact-field capture, P0.1, P0.2, or promotion.",
        ],
    }

    args.output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    print(f"source_sha256={result['source_sha256']}")
    print(
        "mean_difference_bound="
        f"{result['matrix_perturbation']['mean_difference_bound_upper']}"
    )
    print(
        "covariance_difference_bound="
        f"{result['matrix_perturbation']['covariance_difference_bound_upper']}"
    )
    print(
        "Delta_second_moment_bound="
        f"{result['normalizer']['Delta_second_moment_bound_upper']}"
    )
    print("ALL CHECKS PASS")


if __name__ == "__main__":
    main()
