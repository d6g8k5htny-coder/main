#!/usr/bin/env python3
"""
C099 exact cubic type no-go for a third maximum near the fold pair.

Scaled geometry
---------------
Let t=|x|, eta=r/t, omega=(c,s), c^2+s^2=1, and let

    alpha = (b-u)/ell in [0,1],   ell=r^3/6.

Write the leading local field as

    f(tX,tY)=b+t^3 P(X,Y)+O(t^4).

The maximum-saddle pair pins are located at

    M=(-eta/2,0), S=(eta/2,0),

with

    P(M)=0, grad P(M)=0,
    P(S)=-eta^3/6, grad P(S)=0.

The most general cubic satisfying those pair constraints is

    P = X^3/3 - eta^2 X/4 - eta^3/12
        + (a/2)(X^2-eta^2/4)Y
        + (Q/2)Y^2
        + (w/2)XY^2
        + (z/6)Y^3.

Impose a third station at omega with

    P(omega)=-alpha eta^3/6, grad P(omega)=0.

For the generic chart c*s != 0, solve for a,Q,z in terms of w and define

    K=-4c^2-4c eta-eta^2+2s^2 w.

The script proves symbolically

    det H_M^(3)
      = alpha eta^4/s^2
        - eta^2 K^2/(64 c^2 s^2),

    det H_X^(3)
      = -eta^2/(64c^2s^2)
        [(K+8 alpha c eta)^2
         +64 alpha(1-alpha)c^2 eta^2] <= 0.

Thus the third point cannot be a nondegenerate maximum at cubic order.
At alpha=0 the pair maximum determinant and third determinant coincide as
the same nonpositive square.

The transverse chart c=0 has an analogous sum-of-squares identity. On the
pair axis s=0, the cubic pair restriction has only the two pair critical
points unless eta=2, excluded by the radius-2r collar geometry.

Outputs
-------
C099_CUBIC_TYPE_NOGO.json
"""

from __future__ import annotations

import json
from pathlib import Path
import sympy as sp

BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "C099_CUBIC_TYPE_NOGO.json"


def generic_chart() -> dict:
    X, Y = sp.symbols("X Y")
    c, s, eta, alpha = sp.symbols(
        "c s eta alpha", real=True, nonzero=True
    )
    a, Q, w, z = sp.symbols("a Q w z", real=True)

    p0 = X**3 / 3 - eta**2 * X / 4 - eta**3 / 12
    P = (
        p0
        + a * (X**2 - eta**2 / 4) * Y / 2
        + Q * Y**2 / 2
        + w * X * Y**2 / 2
        + z * Y**3 / 6
    )

    equations = [
        sp.diff(P, X).subs({X: c, Y: s}),
        sp.diff(P, Y).subs({X: c, Y: s}),
        P.subs({X: c, Y: s}) + alpha * eta**3 / 6,
    ]
    solution_set = sp.linsolve(equations, (a, Q, z))
    solution = tuple(next(iter(solution_set)))
    a_sol, Q_sol, z_sol = map(sp.factor, solution)

    substitutions = {a: a_sol, Q: Q_sol, z: z_sol}

    def hessian_at(x, y):
        hxx = sp.factor(sp.diff(P, X, 2).subs({X: x, Y: y, **substitutions}))
        hxy = sp.factor(sp.diff(P, X, Y).subs({X: x, Y: y, **substitutions}))
        hyy = sp.factor(sp.diff(P, Y, 2).subs({X: x, Y: y, **substitutions}))
        determinant = sp.factor(hxx * hyy - hxy**2)
        return hxx, hxy, hyy, determinant

    hM = hessian_at(-eta / 2, 0)
    hS = hessian_at(eta / 2, 0)
    hX = hessian_at(c, s)

    K = -4 * c**2 - 4 * c * eta - eta**2 + 2 * s**2 * w

    detM_expected = (
        alpha * eta**4 / s**2
        - eta**2 * K**2 / (64 * c**2 * s**2)
    )
    detX_expected = -eta**2 * (
        (K + 8 * alpha * c * eta) ** 2
        + 64 * alpha * (1 - alpha) * c**2 * eta**2
    ) / (64 * c**2 * s**2)
    detS_expected = -eta**2 * (
        K**2 + 16 * c * eta * K + 64 * alpha * c**2 * eta**2
    ) / (64 * c**2 * s**2)
    Q_expected = -eta**2 * (
        K + 8 * alpha * c * eta
    ) / (8 * c * s**2)

    checks = {
        "pair_constraints": all(
            sp.simplify(expr.subs(substitutions)) == 0
            for expr in equations
        ),
        "Q_identity": sp.simplify(Q_sol - Q_expected) == 0,
        "det_M_identity": sp.simplify(hM[3] - detM_expected) == 0,
        "det_S_identity": sp.simplify(hS[3] - detS_expected) == 0,
        "det_X_identity": sp.simplify(hX[3] - detX_expected) == 0,
        "alpha_zero_det_M_equals_det_X": (
            sp.simplify((hM[3] - hX[3]).subs(alpha, 0)) == 0
        ),
    }

    if not all(checks.values()):
        raise AssertionError(checks)

    return {
        "normal_form": sp.sstr(P),
        "solution": {
            "a": sp.sstr(a_sol),
            "Q": sp.sstr(Q_sol),
            "z": sp.sstr(z_sol),
        },
        "K": sp.sstr(K),
        "determinants": {
            "M": sp.sstr(sp.factor(detM_expected)),
            "S": sp.sstr(sp.factor(detS_expected)),
            "X": sp.sstr(sp.factor(detX_expected)),
        },
        "Q_in_K_coordinates": sp.sstr(sp.factor(Q_expected)),
        "checks": checks,
        "conclusion": (
            "For 0<=alpha<=1 and c*s!=0, det H_X^(3)<=0. "
            "For 0<alpha<1 it is strictly negative. At alpha=0 or 1, "
            "zero requires a codimension-one square to vanish."
        ),
    }


def transverse_chart() -> dict:
    eta, alpha, a = sp.symbols(
        "eta alpha a", real=True
    )
    # Take omega=(0,1). The station equation P_X(0,1)=0 forces w=eta^2/2.
    w = eta**2 / 2
    Q, z = sp.symbols("Q z", real=True)

    # Solve P_Y=0 and P=-alpha eta^3/6 at (0,1).
    equations = [
        -a * eta**2 / 8 + Q + z / 2,
        -eta**3 / 12 - a * eta**2 / 8 + Q / 2 + z / 6
        + alpha * eta**3 / 6,
    ]
    solution = sp.solve(equations, (Q, z), dict=True)[0]
    Q_sol = sp.factor(solution[Q])
    z_sol = sp.factor(solution[z])

    det_M = eta**2 * (
        4 * alpha * eta**2 - (a + eta) ** 2
    ) / 4
    det_X = -eta**2 * (
        (a + (1 - 2 * alpha) * eta) ** 2
        + 4 * alpha * (1 - alpha) * eta**2
    ) / 4

    # Direct Hessian check.
    hM = sp.Matrix([
        [-eta, -a * eta / 2],
        [-a * eta / 2, Q_sol - w * eta / 2],
    ])
    hX = sp.Matrix([
        [a, w],
        [w, Q_sol + z_sol],
    ])
    checks = {
        "station_equations": all(
            sp.simplify(expr.subs(solution)) == 0 for expr in equations
        ),
        "det_M_identity": sp.simplify(hM.det() - det_M) == 0,
        "det_X_identity": sp.simplify(hX.det() - det_X) == 0,
    }
    if not all(checks.values()):
        raise AssertionError(checks)

    return {
        "omega": "(0,1)",
        "w": sp.sstr(w),
        "solution": {
            "Q": sp.sstr(Q_sol),
            "z": sp.sstr(z_sol),
        },
        "determinants": {
            "M": sp.sstr(sp.factor(det_M)),
            "X": sp.sstr(sp.factor(det_X)),
        },
        "checks": checks,
        "conclusion": (
            "The transverse third-point cubic determinant is also a "
            "negative sum of squares for 0<=alpha<=1."
        ),
    }


def axial_chart() -> dict:
    eta = sp.symbols("eta", positive=True)
    # On Y=0, the pair-pinned cubic derivative is X^2-eta^2/4.
    derivative_at_plus_one = sp.factor(1 - eta**2 / 4)
    derivative_at_minus_one = sp.factor(1 - eta**2 / 4)

    # Outside both radius-2r collars, eta is uniformly below sqrt(4/15)<1,
    # and in particular eta!=2.
    return {
        "pair_axis_derivative": "P_X(X,0)=X^2-eta^2/4",
        "at_X_plus_or_minus_1": sp.sstr(derivative_at_plus_one),
        "third_axial_station_condition": "eta=2",
        "collar_geometry": (
            "A point at scaled radius one outside both radius-2r collars "
            "satisfies eta<sqrt(4/15)<1 on the transverse worst case and "
            "eta<2/5 on the axial worst case."
        ),
        "conclusion": (
            "No third axial critical point exists in the cubic normal form "
            "on the admissible outside-collar domain."
        ),
    }


def main() -> None:
    generic = generic_chart()
    transverse = transverse_chart()
    axial = axial_chart()

    report = {
        "cycle": "C099",
        "result_id": "PAIR_SCALED_CUBIC_TYPE_NOGO",
        "grade": "DERIVED-EXACT",
        "generic_chart": generic,
        "transverse_chart": transverse,
        "axial_chart": axial,
        "global_conclusion": [
            (
                "A third critical point in the full maximum-value window "
                "cannot be a nondegenerate maximum at leading cubic order."
            ),
            (
                "At interior height fractions 0<alpha<1, the generic and "
                "transverse third determinants are strictly negative."
            ),
            (
                "At alpha=0 or alpha=1, a third maximum can arise only through "
                "a quartic boundary layer around a vanishing square."
            ),
            (
                "The pair maximum determinant is itself forced toward zero "
                "in the same boundary layer, adding Palm-weight depletion."
            ),
        ],
        "implication_for_Kac_Rice": {
            "baseline_without_type_no_go": (
                "The three-point value/gradient density has a t^-6 scale, "
                "while untyped determinant weights alone would allow a "
                "nonintegrable-looking leading term."
            ),
            "no_go_gain": (
                "Third-maximum positivity requires quartic corrections, "
                "height-endpoint localization, and a small Gaussian jet "
                "coordinate. These supply additional powers before spatial "
                "integration."
            ),
            "remaining_work": (
                "Turn the exact no-go into a uniform weighted Gaussian "
                "boundary-layer estimate on the compactified two-scale charts."
            ),
        },
    }

    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
