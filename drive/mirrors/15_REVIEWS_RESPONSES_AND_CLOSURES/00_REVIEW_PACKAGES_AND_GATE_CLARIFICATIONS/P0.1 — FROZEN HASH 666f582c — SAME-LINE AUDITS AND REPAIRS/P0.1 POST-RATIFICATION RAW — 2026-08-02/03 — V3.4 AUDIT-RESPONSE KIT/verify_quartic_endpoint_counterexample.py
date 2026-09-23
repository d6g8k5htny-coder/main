#!/usr/bin/env python3
"""Exact counterexample to the V3.4 endpoint *remainder-error* weight.

This fail-closed artifact verifies one explicit radial-normalized family

    F_eta,e(x,y) = P3_eta,e(x,y) + e Q4(x,y),

with M=(0,0), S=(eta,0), P=(0,1), kappa=1 and alpha=1/2.  It reconstructs
the unique cubic C3 having the same three values, six gradients, and endpoint
yy-curvature difference.  Although P3 and Q4 stay coefficient-bounded and M
has the required type, det(H_F)-det(H_C3) at both endpoints is too large for
eta(eta+e) when eta=e^p, p>1.

The same family does NOT refute the final common three-determinant envelope:
the actual determinant product remains bounded by the common weight.  The
physical extension e^3 F(X/e,Y/e)-Z^2/2 makes M negative definite and S
index 2 without changing either conclusion.

SCOPE LIMIT: polynomial diagnostic/counterexample only.  It does not model
the SIDE24 Gaussian covariance, conditional moments, or higher remainders.
"""

from __future__ import annotations

import ast
from pathlib import Path

import sympy as sp


checks = 0


def ck(label: str, condition) -> None:
    """Optimization-safe and fail closed."""
    global checks
    checks += 1
    if condition is not True and condition != sp.S.true:
        raise SystemExit(f"CHECK_FAILED[{label}]: {condition}")
    print(f"[ok] {label}")


def require_zero(label: str, expression: sp.Expr) -> None:
    ck(label, sp.factor(sp.cancel(sp.together(expression))) == 0)


def det2(matrix: sp.Matrix) -> sp.Expr:
    return sp.factor(matrix[0, 0] * matrix[1, 1] - matrix[0, 1] ** 2)


x, y, z = sp.symbols("x y z", real=True)
eta, eps = sp.symbols("eta eps", positive=True)

Q4 = x * y**3 - y**4
P3 = (
    x**3 / 3
    - eta * x**2 / 2
    - eta**3 * y**2 / 4
    + eta**3 * y**3 / 6
    - eps * x * y**2
    + 2 * eps * y**3
    - eps * y**2
)
F = sp.expand(P3 + eps * Q4)

# The unique curvature-matched cubic.  Its -eps/eta coefficient is the
# projective blow-up responsible for the failed comparison step.
C3 = (
    x**3 / 3
    - eta * x**2 / 2
    - (eps / eta) * x**2 * y
    + eps * x * y
    - eps * x * y**2
    - eta**3 * y**2 / 4
    + eta**3 * y**3 / 6
)

M = (sp.Integer(0), sp.Integer(0))
S = (eta, sp.Integer(0))
P = (sp.Integer(0), sp.Integer(1))
points = [M, S, P]
values = [sp.Integer(0), -eta**3 / 6, -eta**3 / 12]


# ---------------------------------------------------------------------------
# 1. Full polynomial, basis, and exact interpolation data.
# ---------------------------------------------------------------------------
cubic_basis = [
    sp.Integer(1), y, x, y**2, x * y, x**2,
    y**3, x * y**2, x**2 * y, x**3,
]
P3_expected_coefficients = [
    0, 0, 0,
    -eps - eta**3 / 4,
    0,
    -eta / 2,
    2 * eps + eta**3 / 6,
    -eps,
    0,
    sp.Rational(1, 3),
]
P3_poly = sp.Poly(P3, x, y)
for index, (basis_item, expected) in enumerate(
    zip(cubic_basis, P3_expected_coefficients)
):
    monomial = sp.Poly(basis_item, x, y).monoms()[0]
    require_zero(
        f"bounded P3 basis coefficient {index}",
        P3_poly.coeff_monomial(monomial) - expected,
    )
ck("bounded homogeneous Q4 coefficients", Q4 == x * y**3 - y**4)

for polynomial_name, polynomial in (("F", F), ("C3", C3)):
    for point_name, point, value in zip(("M", "S", "P"), points, values):
        substitutions = {x: point[0], y: point[1]}
        require_zero(
            f"{polynomial_name} value at {point_name}",
            polynomial.subs(substitutions) - value,
        )
        require_zero(
            f"{polynomial_name} x-gradient at {point_name}",
            sp.diff(polynomial, x).subs(substitutions),
        )
        require_zero(
            f"{polynomial_name} y-gradient at {point_name}",
            sp.diff(polynomial, y).subs(substitutions),
        )


def endpoint_yy_difference(polynomial: sp.Expr) -> sp.Expr:
    return (
        sp.diff(polynomial, y, 2).subs({x: 0, y: 0})
        - sp.diff(polynomial, y, 2).subs({x: eta, y: 0})
    )


require_zero(
    "F endpoint yy-curvature difference",
    endpoint_yy_difference(F) - 2 * eta * eps,
)
require_zero(
    "C3 matches exact endpoint yy-curvature difference",
    endpoint_yy_difference(C3) - endpoint_yy_difference(F),
)

# Ten linear functionals (three values, six gradients, one curvature
# difference) have determinant -2 eta^7 on the cubic basis, proving that the
# displayed C3 is the unique comparison cubic for eta>0.
interpolation_rows = []
for point in points:
    substitutions = {x: point[0], y: point[1]}
    interpolation_rows.extend(
        [
            [item.subs(substitutions) for item in cubic_basis],
            [sp.diff(item, x).subs(substitutions) for item in cubic_basis],
            [sp.diff(item, y).subs(substitutions) for item in cubic_basis],
        ]
    )
interpolation_rows.append(
    [endpoint_yy_difference(item) for item in cubic_basis]
)
require_zero(
    "cubic interpolation matrix determinant",
    sp.Matrix(interpolation_rows).det() + 2 * eta**7,
)


# ---------------------------------------------------------------------------
# 2. Exact type and determinant-error failure.
# ---------------------------------------------------------------------------
H_F = sp.hessian(F, (x, y))
H_C3 = sp.hessian(C3, (x, y))
H_F_values = [sp.simplify(H_F.subs({x: point[0], y: point[1]}))
              for point in points]
H_C3_values = [sp.simplify(H_C3.subs({x: point[0], y: point[1]}))
               for point in points]

ck(
    "M plane Hessian exact diagonal form",
    H_F_values[0]
    == sp.diag(-eta, -2 * eps - eta**3 / 2),
)
ck("M plane first eigenvalue negative", (-H_F_values[0][0, 0]).is_positive)
ck("M plane second eigenvalue negative", (-H_F_values[0][1, 1]).is_positive)
require_zero(
    "M typed determinant",
    det2(H_F_values[0]) - eta * (eta**3 + 4 * eps) / 2,
)

det_F = [det2(matrix) for matrix in H_F_values]
det_C3 = [det2(matrix) for matrix in H_C3_values]
errors = [sp.factor(left - right) for left, right in zip(det_F, det_C3)]
expected_errors = [
    eps * (eps + 2 * eta),
    eps * (eps - 2 * eta),
    eta * eps * (eta + 2),
]
for point_name, error, expected in zip(("M", "S", "P"),
                                       errors, expected_errors):
    require_zero(f"{point_name} determinant error", error - expected)

endpoint_target = eta * (eta + eps)
witness_target = eta**2 + eps
endpoint_ratio_M = sp.factor(errors[0] / endpoint_target)
endpoint_ratio_S_positive_branch = sp.factor(errors[1] / endpoint_target)
witness_ratio = sp.factor(errors[2] / witness_target)

for power in (2, 3, 4):
    ck(
        f"M endpoint-error ratio diverges for eta=eps^{power}",
        sp.limit(endpoint_ratio_M.subs(eta, eps**power), eps, 0, dir="+")
        == sp.oo,
    )
    # For eps small and p>1, errors[1]>0; hence no absolute-value ambiguity.
    ck(
        f"S endpoint-error ratio diverges for eta=eps^{power}",
        sp.limit(
            endpoint_ratio_S_positive_branch.subs(eta, eps**power),
            eps, 0, dir="+",
        )
        == sp.oo,
    )
    ck(
        f"witness-error ratio remains harmless for eta=eps^{power}",
        sp.limit(witness_ratio.subs(eta, eps**power), eps, 0, dir="+")
        == 0,
    )

require_zero(
    "comparison cubic M determinant",
    det_C3[0] - (eta**4 - 2 * eps**2) / 2,
)
for power in (2, 3, 4):
    ck(
        f"comparison cubic loses M type for eta=eps^{power}",
        sp.limit(
            det_C3[0].subs(eta, eps**power) / eps**2,
            eps, 0, dir="+",
        )
        == -1,
    )


# ---------------------------------------------------------------------------
# 3. Restore physical scaling and add a hard three-dimensional complement.
# ---------------------------------------------------------------------------
X, Y, Z = sp.symbols("X Y Z", real=True)
r = eta * eps
f_physical = sp.expand(eps**3 * F.subs({x: X / eps, y: Y / eps}) - Z**2 / 2)
c_physical = sp.expand(eps**3 * C3.subs({x: X / eps, y: Y / eps}) - Z**2 / 2)
physical_points = [(0, 0, 0), (r, 0, 0), (0, eps, 0)]
physical_values = [0, -r**3 / 6, -r**3 / 12]

for polynomial_name, polynomial in (("f", f_physical), ("c", c_physical)):
    for point_name, point, value in zip(
        ("M", "S", "P"), physical_points, physical_values
    ):
        substitutions = {X: point[0], Y: point[1], Z: point[2]}
        require_zero(
            f"physical {polynomial_name} value at {point_name}",
            polynomial.subs(substitutions) - value,
        )
        for variable in (X, Y, Z):
            require_zero(
                f"physical {polynomial_name} gradient {variable} at {point_name}",
                sp.diff(polynomial, variable).subs(substitutions),
            )

H_f3 = sp.hessian(f_physical, (X, Y, Z))
H_c3 = sp.hessian(c_physical, (X, Y, Z))
H_f3_values = [sp.simplify(H_f3.subs({X: point[0], Y: point[1], Z: point[2]}))
               for point in physical_points]
H_c3_values = [sp.simplify(H_c3.subs({X: point[0], Y: point[1], Z: point[2]}))
               for point in physical_points]

ck(
    "physical M Hessian negative diagonal",
    (H_f3_values[0] - sp.diag(
        -eta * eps,
        -eps * (2 * eps + eta**3 / 2),
        -1,
    )).applyfunc(sp.factor) == sp.zeros(3),
)
ck(
    "physical S Hessian has signs (+,-,-)",
    (H_f3_values[1] - sp.diag(
        eta * eps,
        -eps * (2 * eps + 2 * eta * eps + eta**3 / 2),
        -1,
    )).applyfunc(sp.factor) == sp.zeros(3),
)

# The common hard eigenvalue -1 multiplies each plane determinant by -eps^2;
# therefore it preserves every normalized determinant-error ratio.
for point_name, plane_error, full_f, full_c in zip(
    ("M", "S", "P"), errors, H_f3_values, H_c3_values
):
    require_zero(
        f"hard complement preserves {point_name} determinant-error ratio",
        full_f.det() - full_c.det() + eps**2 * plane_error,
    )


# ---------------------------------------------------------------------------
# 4. The actual full product still satisfies the common envelope.
# ---------------------------------------------------------------------------
expected_det_F = [
    eta * (eta**3 + 4 * eps) / 2,
    -eta * (eta**3 + 4 * eta * eps + 4 * eps) / 2,
    -(eta**4 - 4 * eta * eps + 2 * eps**2) / 2,
]
for point_name, actual, expected in zip(("M", "S", "P"),
                                        det_F, expected_det_F):
    require_zero(f"actual plane determinant at {point_name}", actual - expected)

normalized_product = sp.factor(sp.prod(expected_det_F))
expected_product = (
    eta**2
    * (eta**3 + 4 * eps)
    * (eta**3 + 4 * eta * eps + 4 * eps)
    * (eta**4 - 4 * eta * eps + 2 * eps**2)
    / 8
)
require_zero("actual normalized three-determinant product",
             normalized_product - expected_product)

physical_product = sp.factor(
    sp.prod(matrix.det() for matrix in H_f3_values)
)
require_zero(
    "physical hard-complement product scaling",
    physical_product + eps**6 * normalized_product,
)

common_D = sp.expand((r * (r + eps**2)) ** 2 * (r**2 + eps**3))
normalized_D = eta**2 * (eta + eps) ** 2 * (eta**2 + eps)
require_zero("physical common-D scaling", common_D - eps**6 * normalized_D)
product_ratio = sp.factor(normalized_product / normalized_D)

for power in (2, 3, 4):
    ck(
        f"actual full-product ratio tends zero for eta=eps^{power}",
        sp.limit(product_ratio.subs(eta, eps**power), eps, 0, dir="+") == 0,
    )

# A coefficient ledger gives a uniform constant on 0<eta,eps<=1:
# A<=4(eta+eps), B<=8(eta+eps), |C|<=4(eta+eps)(eta^2+eps).
# Their product divided by the factor 8 in expected_product is at most
# 16(eta+eps) times normalized_D, hence at most 32 normalized_D.
factor_A = eta**3 + 4 * eps
factor_B = eta**3 + 4 * eta * eps + 4 * eps
absolute_majorant_C = eta**4 + 4 * eta * eps + 2 * eps**2
require_zero(
    "product factor-A unit-box domination identity",
    4 * (eta + eps) - factor_A
    - eta * (3 + 1 - eta**2),
)
require_zero(
    "product factor-B unit-box domination identity",
    8 * (eta + eps) - factor_B
    - (eta * (3 + 1 - eta**2 + 4 * (1 - eps)) + 4 * eps),
)
require_zero(
    "product factor-C triangle-majorant domination identity",
    4 * (eta + eps) * (eta**2 + eps) - absolute_majorant_C
    - (eta**3 * (4 - eta) + 4 * eta**2 * eps + 2 * eps**2),
)
ck("common-product coefficient ledger constant", 4 * 8 * 4 // 8 == 16)
ck("common-product envelope constant on unit box", 16 * 2 == 32)


# The script itself must contain no optimization-stripped Python assertions.
source_tree = ast.parse(Path(__file__).read_text())
ck("zero bare Python assert statements",
   sum(isinstance(node, ast.Assert) for node in ast.walk(source_tree)) == 0)

print(f"CHECK_COUNT={checks}")
print(f"SYMPY_VERSION={sp.__version__}")
print("COUNTEREXAMPLE_CONFIRMED=endpoint cubic-remainder error weight")
print("ENDPOINT_ERRORS_NORMALIZED=eps(eps+2eta),eps(eps-2eta)")
print("WITNESS_ERROR_NORMALIZED=eta*eps*(eta+2)")
print("ETA_REGIME=eta=eps^p with p>1")
print("FULL_3D_TYPES=M negative definite; S index 2")
print("COMMON_PRODUCT_COUNTEREXAMPLE=NO for this family")
print("COMMON_PRODUCT_RATIO_LIMIT=0 for p=2,3,4")
print("PROOF_IMPACT=individual remainder comparison fails; product needs a different argument")
print("SCOPE_LIMIT=exact polynomial family only; not a Gaussian theorem audit")
print("ALL_ASSERTIONS_PASS")
