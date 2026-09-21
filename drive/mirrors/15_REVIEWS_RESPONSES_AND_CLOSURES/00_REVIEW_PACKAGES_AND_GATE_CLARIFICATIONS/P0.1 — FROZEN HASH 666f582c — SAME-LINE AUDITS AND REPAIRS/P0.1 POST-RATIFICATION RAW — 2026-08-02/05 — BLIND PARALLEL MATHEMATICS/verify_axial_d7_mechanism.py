#!/usr/bin/env python3
"""Fail-closed independent check of the endpoint-axis d^7 mechanism.

This script uses only the Bargmann--Fock covariance
    K(x-y) = exp(-|x-y|^2/2)
and the confluent pair-pin space.  It does not import project scripts.

Notation:
  d   = pair separation;
  z   = signed witness displacement from the left endpoint;
  eta = z/d.

The requested G1 and G1' are the axial component of
  z^{-1}(grad f(M+z t)-grad f(M))
at z=0 and its z derivative.  All conditional covariances below are with
respect to the endpoint value/gradient pins.
"""

from __future__ import annotations

import os

import sympy as sp


checks = 0


def ck(label: str, condition) -> None:
    """Optimization-safe check that exits nonzero on the first failure."""
    global checks
    checks += 1
    ok = bool(condition)
    print(f"[{'ok' if ok else 'FAIL'}] {label}")
    if not ok:
        raise SystemExit(1)


def hp(n: int, x):
    """Probabilists' Hermite polynomial by its defining recurrence."""
    h0 = sp.Integer(1)
    if n == 0:
        return h0
    h1 = x
    if n == 1:
        return h1
    hm2, hm1 = h0, h1
    for k in range(1, n):
        hm2, hm1 = hm1, sp.expand(x * hm1 - k * hm2)
    return hm1


def cov0(alpha: tuple[int, int, int], beta: tuple[int, int, int]):
    """Same-point Bargmann--Fock derivative covariance."""
    value = sp.Integer((-1) ** sum(alpha))
    for left, right in zip(alpha, beta):
        value *= hp(left + right, sp.Integer(0))
    return sp.Integer(value)


def conditional_cov(outputs: list[tuple[int, int, int]],
                    pins: list[tuple[int, int, int]]) -> sp.Matrix:
    pp = sp.Matrix([[cov0(left, right) for right in pins] for left in pins])
    oo = sp.Matrix([[cov0(left, right) for right in outputs] for left in outputs])
    op = sp.Matrix([[cov0(left, right) for right in pins] for left in outputs])
    return sp.simplify(oo - op * pp.inv() * op.T)


# Contact limit of the corrected pair-pin frame.
PINS = [
    (0, 0, 0),  # f
    (1, 0, 0),  # f_t
    (0, 1, 0),  # f_v
    (0, 0, 1),  # f_w
    (2, 0, 0),  # f_tt
    (1, 1, 0),  # f_tv
    (1, 0, 1),  # f_tw
    (3, 0, 0),  # f_ttt
]

R4 = (4, 0, 0)
RTTV = (2, 1, 0)
RTTW = (2, 0, 1)
RTVV = (1, 2, 0)
FVV = (0, 2, 0)
FVW = (0, 1, 1)

contact = conditional_cov([R4, RTTV, RTTW, RTVV, FVV, FVW], PINS)
ck("01 contact pin Gram matrix is invertible",
   sp.Matrix([[cov0(a, b) for b in PINS] for a in PINS]).det() != 0)
ck("02 Var(R4 | contact pins) = 24", contact[0, 0] == 24)
ck("03 Var(R_ttv | contact pins) = 2", contact[1, 1] == 2)
ck("04 Var(R_ttw | contact pins) = 2", contact[2, 2] == 2)
ck("05 Var(R_tvv | contact pins) = 2", contact[3, 3] == 2)
ck("06 Var(f_vv | contact pins) = 2", contact[4, 4] == 2)
ck("07 Var(f_vw | contact pins) = 1", contact[5, 5] == 1)
off_diagonal_indices = [(0, 1), (0, 2), (1, 2), (1, 3), (3, 4), (3, 5)]
ck("08 parity cross-covariances used below vanish",
   all(contact[i, j] == 0 for i, j in off_diagonal_indices))


# Axis: exact fourth-order Hermite coefficients in the joint d,z blow-up.
eta, d, z = sp.symbols("eta d z", real=True)
c_ax = (eta - 1) * (2 * eta - 1) / 12
c_tr = (eta - 1) / 2
axis_profile = sp.factor(
    (24 * c_ax**2) * (2 * c_tr**2) * (2 * c_tr**2)
)
axis_expected = (eta - 1) ** 6 * (2 * eta - 1) ** 2 / 24
ck("09 universal axial leading profile", sp.expand(axis_profile - axis_expected) == 0)
ck("10 axial leading determinant at z=0 is d^8/24",
   axis_profile.subs(eta, 0) == sp.Rational(1, 24))
ck("11 axial rho coefficient is -5 d^7/12",
   sp.diff(axis_profile, eta).subs(eta, 0) == -sp.Rational(5, 12))
ck("12 axial logarithmic slope is -10/d",
   sp.simplify(sp.diff(axis_profile, eta).subs(eta, 0)
               / axis_profile.subs(eta, 0)) == -10)

# G1 = d^2 R4/12 + O(d^3), G1' = -d R4/4 + O(d^2).
g1_g1prime_lead = sp.Rational(1, 12) * (-sp.Rational(1, 4)) * contact[0, 0]
ck("13 Cov(G1,G1' | pins) leading coefficient is -1/2",
   g1_g1prime_lead == -sp.Rational(1, 2))


# Transverse witness direction.  The soft first row is
# d*(-R_ttv + eta R_tvv)/2; the two hard rows are f_vv and f_vw.
transverse_soft_var = sp.expand(
    (contact[1, 1] + eta**2 * contact[3, 3]
     - 2 * eta * contact[1, 3]) / 4
)
transverse_profile = sp.factor(transverse_soft_var * contact[4, 4] * contact[5, 5])
ck("14 transverse leading profile is 1+eta^2",
   sp.expand(transverse_profile - (1 + eta**2)) == 0)
ck("15 transverse first-order term vanishes exactly",
   sp.diff(transverse_profile, eta).subs(eta, 0) == 0)


# Exact 5% two-sided leading-profile window.
ratio = sp.expand(axis_profile / axis_profile.subs(eta, 0))
lo = sp.Rational(1, 20)
ratio_plus = sp.factor(ratio.subs(eta, lo))
ratio_minus = sp.factor(ratio.subs(eta, -lo))
ck("16 +5% endpoint of axial ratio window",
   ratio_plus == sp.Rational(3810716361, 6400000000))
ck("17 -5% endpoint of axial ratio window",
   ratio_minus == sp.Rational(10377700641, 6400000000))
log_derivative_numerator = sp.factor(sp.diff(ratio, eta))
ck("18 axial profile is decreasing on the 5% window",
   all(sp.N(log_derivative_numerator.subs(eta, point), 30) < 0
       for point in (-lo, sp.Integer(0), lo)))


# Independent exact finite-d Schur calculation at the endpoint.  This is
# not inferred from the contact rows above.  Keeping q algebraically
# independent until the final series makes the calculation fast and also
# checks the polynomial pin algebra before q=exp(-d^2/2) is inserted.
q_kernel = sp.symbols("q_kernel", real=True)
sigma = sp.Matrix([
    [1, q_kernel, 0, -d * q_kernel],
    [q_kernel, 1, d * q_kernel, 0],
    [0, d * q_kernel, 1, (1 - d**2) * q_kernel],
    [-d * q_kernel, 0, (1 - d**2) * q_kernel, 1],
])
sigma_adj = sigma.adjugate()
sigma_det = sigma.det()
c_ax0 = sp.Matrix([[-1, (d**2 - 1) * q_kernel,
                        0, (3 * d - d**3) * q_kernel]])
c_ax1 = sp.Matrix([[0, (d**3 - 3 * d) * q_kernel / 2,
                        -sp.Rational(3, 2),
                        -(d**4 - 6 * d**2 + 3) * q_kernel / 2]])
var_ax = sp.cancel(3 - (c_ax0 * sigma_adj * c_ax0.T)[0] / sigma_det)
cov_ax_prime = sp.cancel(-(c_ax0 * sigma_adj * c_ax1.T)[0] / sigma_det)

sigma_tr = sp.Matrix([[1, q_kernel], [q_kernel, 1]])
sigma_tr_adj = sigma_tr.adjugate()
sigma_tr_det = sigma_tr.det()
c_t = sp.Matrix([[0, d * q_kernel]])
c_tprime = sp.Matrix([[-sp.Rational(1, 2),
                       (d**2 - 1) * q_kernel / 2]])
var_tr = sp.cancel(1 - (c_t * sigma_tr_adj * c_t.T)[0] / sigma_tr_det)
cov_tr_prime = sp.cancel(-(c_t * sigma_tr_adj * c_tprime.T)[0] / sigma_tr_det)

q_exp = sp.exp(-d**2 / 2)
series_var_ax = sp.series(var_ax.subs(q_kernel, q_exp), d, 0, 9).removeO().expand()
series_cov_ax = sp.series(cov_ax_prime.subs(q_kernel, q_exp), d, 0, 8).removeO().expand()
series_var_tr = sp.series(var_tr.subs(q_kernel, q_exp), d, 0, 7).removeO().expand()
series_cov_tr = sp.series(cov_tr_prime.subs(q_kernel, q_exp), d, 0, 6).removeO().expand()
ck("19 direct Schur Var(G1) begins d^4/6", series_var_ax.coeff(d, 4) == sp.Rational(1, 6))
ck("20 direct Schur Cov(G1,G1') begins -d^3/2",
   series_cov_ax.coeff(d, 3) == -sp.Rational(1, 2))
ck("21 direct Schur transverse variance begins d^2/2",
   series_var_tr.coeff(d, 2) == sp.Rational(1, 2))
ck("22 direct Schur transverse cross term begins -d/2",
   series_cov_tr.coeff(d, 1) == -sp.Rational(1, 2))

det0 = sp.cancel(var_ax * var_tr**2)
det1 = sp.cancel(
    2 * cov_ax_prime * var_tr**2
    + 4 * var_ax * var_tr * cov_tr_prime
)
series_det0 = sp.series(det0.subs(q_kernel, q_exp), d, 0, 10).removeO().expand()
series_det1 = sp.series(det1.subs(q_kernel, q_exp), d, 0, 9).removeO().expand()
ck("23 direct Schur determinant begins d^8/24",
   series_det0.coeff(d, 8) == sp.Rational(1, 24))
ck("24 direct Schur determinant derivative begins -5d^7/12",
   series_det1.coeff(d, 7) == -sp.Rational(5, 12))

if os.environ.get("D7_FORCE_FAILURE") == "1":
    ck("forced fail-closed mutation", False)

print(f"CHECK_COUNT={checks}")
print("AXIAL_PROFILE=d^8*(1-eta)^6*(1-2eta)^2/24+o(d^8)")
print("AXIAL_DERIVATIVE_AT_ZERO=-5*d^7/12+o(d^7)")
print("AXIAL_LOG_SLOPE=-10/d+o(1/d)")
print("COV_G1_G1PRIME=-d^3/2+O(d^5) [Bargmann--Fock]")
print("TRANSVERSE_PROFILE=d^2*(1+eta^2)+o(d^2)")
print("SCOPE_LIMIT: exact Euclidean contact and Schur algebra; the accompanying")
print("note states the structural side-24 transfer and its explicit assumptions.")
print("ALL_ASSERTIONS_PASS")
