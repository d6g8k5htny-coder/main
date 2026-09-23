#!/usr/bin/env python3
"""
C101 exact window-critical cubic frame for the global interceptor count.

Scaled geometry
---------------
Let t=|x|, eta=r/t, omega=(c,s), and

    alpha=(b-u)/(r^3/6) in [0,1].

The maximum-saddle pair is at

    M=(-eta/2,0), S=(eta/2,0)

in t-scaled coordinates. The leading cubic satisfying the pair pins is

    P = X^3/3 - eta^2 X/4 - eta^3/12
        +(a/2)(X^2-eta^2/4)Y
        +(Q/2)Y^2
        +(w/2)XY^2
        +(z/6)Y^3.

Impose a third critical point at omega with

    P(omega)=-alpha eta^3/6,
    grad P(omega)=0.

The script solves for a,Q,z and proves:

  * each of det D2P(M), det D2P(S), det D2P(omega)
    contains an exact factor eta^2;
  * their product contains eta^6;
  * the third determinant is nonpositive for alpha in [0,1], and strictly
    negative for 0<alpha<1 on the generic chart.

Together with the C099 value/gradient density scale t^-6, the physical
determinant product t^6 eta^6=r^6, and the pair-Palm denominator r^2,
the per-height intensity is bounded by C r^4 t^-6.

Outputs
-------
C101_WINDOW_SADDLE_FRAME.json
"""

from __future__ import annotations

import json
from pathlib import Path
import sympy as sp

BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "C101_WINDOW_SADDLE_FRAME.json"


def generic_chart() -> dict:
    x, y = sp.symbols("x y")
    c, s, eta, alpha = sp.symbols(
        "c s eta alpha", real=True, nonzero=True
    )
    a, Q, w, z = sp.symbols("a Q w z", real=True)

    P = (
        x**3 / 3
        - eta**2 * x / 4
        - eta**3 / 12
        + a * (x**2 - eta**2 / 4) * y / 2
        + Q * y**2 / 2
        + w * x * y**2 / 2
        + z * y**3 / 6
    )

    equations = [
        sp.diff(P, x).subs({x: c, y: s}),
        sp.diff(P, y).subs({x: c, y: s}),
        P.subs({x: c, y: s}) + alpha * eta**3 / 6,
    ]
    solution_tuple = tuple(
        next(iter(sp.linsolve(equations, (a, Q, z))))
    )
    solution = {
        a: sp.factor(solution_tuple[0]),
        Q: sp.factor(solution_tuple[1]),
        z: sp.factor(solution_tuple[2]),
    }

    hessian = sp.hessian(P, (x, y)).subs(solution)

    def det_at(X, Y):
        return sp.factor(hessian.subs({x: X, y: Y}).det())

    det_M = det_at(-eta / 2, 0)
    det_S = det_at(eta / 2, 0)
    det_X = det_at(c, s)

    reduced_M = sp.factor(det_M / eta**2)
    reduced_S = sp.factor(det_S / eta**2)
    reduced_X = sp.factor(det_X / eta**2)
    triple_product = sp.factor(det_M * det_S * det_X)
    reduced_product = sp.factor(triple_product / eta**6)

    K = -4 * c**2 - 4 * c * eta - eta**2 + 2 * s**2 * w
    det_X_sum_squares = -eta**2 * (
        (K + 8 * alpha * c * eta) ** 2
        + 64 * alpha * (1 - alpha) * c**2 * eta**2
    ) / (64 * c**2 * s**2)

    checks = {
        "station_equations": all(
            sp.simplify(equation.subs(solution)) == 0
            for equation in equations
        ),
        "det_M_eta2": sp.simplify(det_M - eta**2 * reduced_M) == 0,
        "det_S_eta2": sp.simplify(det_S - eta**2 * reduced_S) == 0,
        "det_X_eta2": sp.simplify(det_X - eta**2 * reduced_X) == 0,
        "triple_eta6": (
            sp.simplify(triple_product - eta**6 * reduced_product) == 0
        ),
        "det_X_sum_squares": (
            sp.simplify(det_X - det_X_sum_squares) == 0
        ),
    }
    if not all(checks.values()):
        raise AssertionError(checks)

    return {
        "normal_form": sp.sstr(P),
        "solution": {
            "a": sp.sstr(solution[a]),
            "Q": sp.sstr(solution[Q]),
            "z": sp.sstr(solution[z]),
        },
        "K": sp.sstr(K),
        "determinants": {
            "M": sp.sstr(det_M),
            "S": sp.sstr(det_S),
            "third": sp.sstr(det_X),
        },
        "eta2_reduced_determinants": {
            "M": sp.sstr(reduced_M),
            "S": sp.sstr(reduced_S),
            "third": sp.sstr(reduced_X),
        },
        "triple_product": sp.sstr(triple_product),
        "eta6_reduced_product": sp.sstr(reduced_product),
        "third_saddle_sum_of_squares": sp.sstr(
            sp.factor(det_X_sum_squares)
        ),
        "checks": checks,
    }


def transverse_chart() -> dict:
    eta, alpha, a = sp.symbols(
        "eta alpha a", real=True
    )

    det_X = -eta**2 * (
        (a + (1 - 2 * alpha) * eta) ** 2
        + 4 * alpha * (1 - alpha) * eta**2
    ) / 4
    reduced = sp.factor(det_X / eta**2)

    return {
        "third_determinant": sp.sstr(sp.factor(det_X)),
        "eta2_reduced": sp.sstr(reduced),
        "conclusion": (
            "For 0<=alpha<=1 the transverse third point is a saddle or "
            "degenerate at cubic order."
        ),
    }


def axis_chart() -> dict:
    X, eta = sp.symbols("X eta", real=True)
    derivative = X**2 - eta**2 / 4
    return {
        "pair_axis_derivative": sp.sstr(derivative),
        "third_station_at_unit_radius_requires": "eta=2",
        "outside_collar_domain": "eta^2<=4/15",
        "conclusion": (
            "The leading cubic has no additional axis station on the "
            "outside-collar singular chart. Axis neighborhoods are covered "
            "by the C099 divided-difference and mean-gap atlas."
        ),
    }


def power_count() -> dict:
    r, t, C = sp.symbols("r t C", positive=True)
    integrand = r**7 * t**(-5)
    antiderivative = sp.integrate(integrand, t)
    kappa = sp.sqrt(15) / 2
    definite = sp.simplify(
        sp.integrate(integrand, (t, kappa * r, sp.symbols("delta", positive=True)))
    )

    return {
        "value_gradient_density": "C t^-6",
        "physical_triple_determinant": (
            "t^6 eta^6 = r^6 times a uniformly bounded jet polynomial"
        ),
        "pair_Palm_denominator": "Z_r>=c r^2",
        "per_height_intensity": "C r^4 t^-6",
        "height_window": "r^3/6",
        "polar_integrand": "C r^7 t^-5 dt",
        "antiderivative": sp.sstr(antiderivative),
        "lower_radius": "t>=(sqrt(15)/2)r outside both radius-2r collars",
        "definite_integral": sp.sstr(definite),
        "conclusion": "The singular near contribution is bounded by C r^3.",
    }


def main() -> None:
    generic = generic_chart()
    transverse = transverse_chart()
    axis = axis_chart()
    count = power_count()

    report = {
        "cycle": "C101",
        "result_id": "WINDOW_SADDLE_ETA6_FRAME",
        "grade": "DERIVED-EXACT",
        "generic_chart": generic,
        "transverse_chart": transverse,
        "axis_chart": axis,
        "power_count": count,
        "regional_assembly": {
            "collars": (
                "C100 all-critical collar theorem dominates every collar saddle."
            ),
            "singular_near": (
                "eta^6 determinant factor plus C099 value/gradient frames "
                "gives the displayed r^3 integral."
            ),
            "fixed_annulus": (
                "divided-difference compactness and window width r^3/6"
            ),
            "exterior": (
                "fixed-distance pair-Palm compactness and window width r^3/6"
            ),
        },
        "not_claimed": [
            "No numerical global saddle coefficient is extracted.",
            "The result targets a qualitative finite cubic rate.",
        ],
    }

    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
