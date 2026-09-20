#!/usr/bin/env python3
"""Fail-closed verification of the corrected planar G.9.1 reconstruction."""

import os
import sys

import mpmath as mp
import sympy as sp


checks = []


def ck(label, condition):
    ok = bool(condition)
    checks.append(ok)
    print(f"[{'ok' if ok else 'FAIL'}] {label}")


d, q = sp.symbols("d q", real=True)
z = sp.symbols("z", real=True)


def hermite_prob(n, value):
    h0 = sp.Integer(1)
    if n == 0:
        return h0
    h1 = value
    if n == 1:
        return h1
    hm2, hm1 = h0, h1
    for k in range(1, n):
        hm2, hm1 = hm1, sp.expand(value * hm1 - k * hm2)
    return hm1


def cov_axis(point_a, order_a, point_b, order_b):
    displacement = sp.expand(point_a - point_b)
    kernel = sp.Integer(1) if displacement == 0 else q
    return sp.expand(
        (-1) ** order_a
        * hermite_prob(order_a + order_b, displacement)
        * kernel
    )


points = [sp.Integer(0), d, sp.Integer(0), d]
orders = [0, 0, 1, 1]
sigma_x = sp.Matrix(
    [
        [cov_axis(points[i], orders[i], points[j], orders[j]) for j in range(4)]
        for i in range(4)
    ]
)
sigma_expected = sp.Matrix(
    [
        [1, q, 0, -d * q],
        [q, 1, d * q, 0],
        [0, d * q, 1, (1 - d**2) * q],
        [-d * q, 0, (1 - d**2) * q, 1],
    ]
)
ck(
    "01 axial pin matrix derived from Hermite covariance",
    sigma_x.applyfunc(sp.expand) == sigma_expected.applyfunc(sp.expand),
)

det_sigma = sp.factor(sigma_x.det())
det_expected = (-d**2 * q + q**2 - 1) * (d**2 * q + q**2 - 1)
ck("02 correct axial pin determinant factorization", sp.expand(det_sigma - det_expected) == 0)

c_perp = sp.Matrix([-1, -q, 0, d * q])
c_xx = sp.Matrix([-1, (d**2 - 1) * q, 0, (3 * d - d**3) * q])
q_identity = sp.expand((c_perp.T * sigma_x.adjugate() * c_perp)[0] - det_sigma)
ck("03 exact Q=1 adjugate identity in Z[d,q]", q_identity == 0)

mixed_identity = sp.expand((c_xx.T * sigma_x.adjugate() * c_perp)[0] - det_sigma)
ck("04 mixed axial/transverse adjugate identity", mixed_identity == 0)

sigma_inv = sigma_x.inv()
cross_xx_perp = sp.factor(1 - (c_xx.T * sigma_inv * c_perp)[0])
ck("05 conditioned Cov(f_xx,f_yy)=0 exactly", cross_xx_perp == 0)

g = sp.factor((d**2 * q**2 + q**2 - 1) / (q**2 - 1))
v = sp.factor(3 - (c_xx.T * sigma_inv * c_xx)[0])
v_expected = (
    -d**6 * q**2
    + d**4 * q**4
    + d**4 * q**2
    + 4 * d**2 * q**4
    - 4 * d**2 * q**2
    + 2 * q**4
    - 4 * q**2
    + 2
) / ((-d**2 * q + q**2 - 1) * (d**2 * q + q**2 - 1))
ck("06 f_xy Schur variance formula", sp.simplify(g - (1 - d**2 * q**2 / (1 - q**2))) == 0)
ck("07 f_xx Schur variance formula", sp.simplify(v - v_expected) == 0)

x, y, zz = sp.symbols("x y zz", real=True)
U = x**2
P = y**2 + zz**2
gamma = sp.Matrix(
    [
        [v * U + g * P, g * x * y, g * x * zz],
        [g * x * y, g * U + 2 * y**2 + zz**2, y * zz],
        [g * x * zz, y * zz, g * U + y**2 + 2 * zz**2],
    ]
)
closed = (P + g * U) * (2 * g * P**2 + 2 * v * U * P + g * v * U**2)
ck("08 corrected determinant identity", sp.simplify(gamma.det() - closed) == 0)
ck("09 transverse-face formula", sp.simplify(closed.subs(x, 0) - 2 * g * P**3) == 0)
ck("10 pin-axis formula", sp.simplify(closed.subs({y: 0, zz: 0}) - g**2 * v * x**6) == 0)

q_exp = sp.exp(-d**2 / 2)
g_series = sp.series(g.subs(q, q_exp), d, 0, 10)
v_series = sp.series(v.subs(q, q_exp), d, 0, 10)
axis_series = sp.series((g**2 * v).subs(q, q_exp), d, 0, 12)
ck("11 g(d) coalescence series", g_series == d**2 / 2 - d**4 / 12 + d**8 / 720 + sp.Order(d**10))
ck("12 v(d) coalescence series", v_series == d**4 / 6 - d**6 / 30 + d**8 / 360 + sp.Order(d**10))
ck("13 axial d^8 coalescence series", axis_series == d**8 / 24 - d**10 / 45 + sp.Order(d**12))

UU, PP = sp.symbols("U P", nonnegative=True)
closed_up = (PP + g * UU) * (2 * g * PP**2 + 2 * v * UU * PP + g * v * UU**2)
generic_series = sp.series(closed_up.subs(q, q_exp), d, 0, 8)
generic_expected = (
    d**2 * PP**3
    + d**4 * (sp.Rational(5, 6) * UU * PP**2 - sp.Rational(1, 6) * PP**3)
    + d**6 * (sp.Rational(1, 4) * UU**2 * PP - sp.Rational(7, 30) * UU * PP**2)
    + sp.Order(d**8)
)
ck("14 generic coalescence stratification", generic_series == generic_expected)
ck("15 decoupled d=infinity limit q=0", sp.simplify(closed_up.subs(q, 0) - 2 * (UU + PP) ** 3) == 0)

F1 = -d**2 * q + q**2 + 2 * q + 1
F2 = d**2 * q + q**2 - 2 * q + 1
false_pin_factor = (q + 1) ** 2 * F1 * F2
ck("16 reject historical false pin determinant factor", sp.expand(det_sigma - false_pin_factor) != 0)
mutated_c = sp.Matrix([-1, -q, 0, -d * q])
mutated_q_identity = sp.expand((mutated_c.T * sigma_x.adjugate() * mutated_c)[0] - det_sigma)
ck("17 covariance-sign mutation rejected", mutated_q_identity != 0)
ck("18 false O(1) term 4UP^2 rejected", sp.limit(closed_up.subs(q, q_exp), d, 0) != 4 * UU * PP**2)


mp.mp.dps = 80


def hermite_mp(n, value):
    if n == 0:
        return mp.mpf(1)
    if n == 1:
        return value
    hm2, hm1 = mp.mpf(1), value
    for k in range(1, n):
        hm2, hm1 = hm1, value * hm1 - k * hm2
    return hm1


def cov_mp(point_a, alpha, point_b, beta):
    delta = [point_a[i] - point_b[i] for i in range(3)]
    kernel = mp.exp(-sum(t * t for t in delta) / 2)
    value = (-1) ** sum(alpha) * kernel
    for i in range(3):
        value *= hermite_mp(alpha[i] + beta[i], delta[i])
    return value


def linear_cov(terms_a, terms_b):
    total = mp.mpf(0)
    for coefficient_a, point_a, alpha in terms_a:
        for coefficient_b, point_b, beta in terms_b:
            total += coefficient_a * coefficient_b * cov_mp(point_a, alpha, point_b, beta)
    return total


def direct_conditioned_det(distance, vector):
    origin = (mp.mpf(0), mp.mpf(0), mp.mpf(0))
    other = (distance, mp.mpf(0), mp.mpf(0))
    first_jet = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    pins = [(mp.mpf(1), point, alpha) for point in (origin, other) for alpha in first_jet]
    outputs = []
    for i in range(3):
        terms = []
        for j in range(3):
            alpha = [0, 0, 0]
            alpha[i] += 1
            alpha[j] += 1
            terms.append((vector[j], origin, tuple(alpha)))
        outputs.append(terms)
    pin_matrix = mp.matrix([[linear_cov([pins[i]], [pins[j]]) for j in range(8)] for i in range(8)])
    output_matrix = mp.matrix([[linear_cov(outputs[i], outputs[j]) for j in range(3)] for i in range(3)])
    coupling = mp.matrix([[linear_cov(outputs[i], [pins[j]]) for j in range(8)] for i in range(3)])
    conditioned = output_matrix - coupling * (pin_matrix**-1) * coupling.T
    return mp.det(conditioned)


def formula_det(distance, vector):
    qq = mp.exp(-distance**2 / 2)
    gg = (distance**2 * qq**2 + qq**2 - 1) / (qq**2 - 1)
    vv = (
        -distance**6 * qq**2
        + distance**4 * qq**4
        + distance**4 * qq**2
        + 4 * distance**2 * qq**4
        - 4 * distance**2 * qq**2
        + 2 * qq**4
        - 4 * qq**2
        + 2
    ) / ((-distance**2 * qq + qq**2 - 1) * (distance**2 * qq + qq**2 - 1))
    uu = vector[0] ** 2
    pp = vector[1] ** 2 + vector[2] ** 2
    return (pp + gg * uu) * (2 * gg * pp**2 + 2 * vv * uu * pp + gg * vv * uu**2)


fixtures = [
    ("19 fixture d=0.1, u=(1,0,0)", mp.mpf("0.1"), (mp.mpf(1), mp.mpf(0), mp.mpf(0)), mp.mpf("4.14449071554625894308856970184054374284558536881424e-10"), mp.mpf("1e-49")),
    ("20 fixture d=1, u=(1,1,1)", mp.mpf(1), (mp.mpf(1), mp.mpf(1), mp.mpf(1)), mp.mpf("9.53941731866226576797363743769172724627000946554359"), mp.mpf("1e-49")),
    ("21 fixture d=2, u=(0.9,0.1,0.4)", mp.mpf(2), (mp.mpf("0.9"), mp.mpf("0.1"), mp.mpf("0.4")), mp.mpf("0.976024399862822005117691369544350648555032721656"), mp.mpf("1e-47")),
    ("22 face d=1, u=(0,1,0)", mp.mpf(1), (mp.mpf(0), mp.mpf(1), mp.mpf(0)), mp.mpf("0.8360465862613471512299959897819768829"), mp.mpf("1e-36")),
]

force_failure = os.environ.get("G9_FORCE_FAILURE") == "1"
for index, (label, distance, vector, expected, receipt_tolerance) in enumerate(fixtures):
    direct = direct_conditioned_det(distance, vector)
    formula = formula_det(distance, vector)
    if force_failure and index == 0:
        expected += mp.mpf("1e-4")
    scale = max(mp.mpf(1), abs(direct), abs(expected))
    ck(label + " direct=formula", abs(direct - formula) <= mp.mpf("1e-65") * scale)
    ck(label + " matches AO48 receipt", abs(formula - expected) <= receipt_tolerance * scale)

passed = sum(checks)
print(f"g9.1 reconstruction: {passed}/{len(checks)} checks passed")
if passed == len(checks):
    print("ALL_ASSERTIONS_PASS")
    raise SystemExit(0)
raise SystemExit(1)
