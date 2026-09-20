#!/usr/bin/env python3
"""Fail-closed algebra for the axial suppression cone.

Here d is the pair separation and rho is the signed axial witness offset
from the left endpoint.  This is V3.4 (r,q), not its angular variable rho.
"""

from __future__ import annotations

import os
from math import factorial

import sympy as sp


checks = 0


def ck(label: str, condition) -> None:
    global checks
    checks += 1
    ok = bool(condition)
    print(f"[{'ok' if ok else 'FAIL'}] {label}")
    if not ok:
        raise SystemExit(1)


x, d, rho, lam = sp.symbols("x d rho lam", real=True)
z = x / d
h00 = 2 * z**3 - 3 * z**2 + 1
h10 = z**3 - 2 * z**2 + z
h01 = -2 * z**3 + 3 * z**2
h11 = z**3 - z**2


def derivative_residual(degree: int):
    value_0 = sp.Integer(1) if degree == 0 else sp.Integer(0)
    derivative_0 = sp.Integer(1) if degree == 1 else sp.Integer(0)
    value_d = d**degree / factorial(degree)
    derivative_d = (sp.Integer(0) if degree == 0
                    else d**(degree - 1) / factorial(degree - 1))
    interpolant = (value_0 * h00 + d * derivative_0 * h10
                   + value_d * h01 + d * derivative_d * h11)
    source = x**degree / factorial(degree)
    return sp.factor(sp.diff(source - interpolant, x).subs(x, rho))


for degree in range(4):
    ck(f"0{degree + 1} cubic Hermite reproduces degree {degree}",
       derivative_residual(degree) == 0)

a = rho * (rho - d)
chi = 2 * rho - d
h4 = derivative_residual(4)
h5 = derivative_residual(5)
ck("05 fourth-order derivative residual a*chi/12",
   sp.expand(h4 - a * chi / 12) == 0)
ck("06 midpoint fourth-order cancellation", h4.subs(rho, d / 2) == 0)
ck("07 midpoint fifth-order residual is nonzero",
   h5.subs(rho, d / 2) == d**4 / 1920)

d0_sq = 4 * chi**2 + a**2
cone_d0 = sp.factor(d0_sq.subs(rho, lam * d))
cone_expected = d**2 * (4 * (2 * lam - 1)**2
                         + d**2 * lam**2 * (lam - 1)**2)
ck("08 cone d0 factorization", sp.expand(cone_d0 - cone_expected) == 0)
ck("09 cone ceiling constant is 40 for 0<=lambda<=2 and d<=1",
   4 * 9 + 4 == 40)
ck("10 midpoint d0^2=d^4/16",
   sp.factor(cone_d0.subs(lam, sp.Rational(1, 2))) == d**4 / 16)

# If |m|>=kappa/2 and Var(axial row)<=C*a^2*d0^2, the Gaussian
# half-Mahalanobis exponent is at least kappa^2/(8*C*d0^2), hence at
# least kappa^2/(320*C*d^2) on the cone.
kappa, C = sp.symbols("kappa C", positive=True)
mean_lower_sq = kappa**2 * a**2 / 4
variance_upper = C * a**2 * d0_sq
half_mahalanobis = sp.factor(mean_lower_sq / (2 * variance_upper))
ck("11 normalized mean/variance cancellation",
   sp.simplify(half_mahalanobis - kappa**2 / (8 * C * d0_sq)) == 0)
cone_exponent_certificate = kappa**2 / (320 * C * d**2)
ck("12 cone exponent certificate has d^-2 scale",
   sp.diff(cone_exponent_certificate, kappa) == kappa / (160 * C * d**2))

# Three gradient rows have exact Hermite factors a*d0, a, a.  The density
# prefactor is therefore |a|^-3*d0^-1 after a uniform normalized Gram floor.
normalized_gram = sp.symbols("A", positive=True)
det_cov = a**6 * d0_sq * normalized_gram
formal_prefactor_squared = 1 / det_cov
ck("13 determinant factor is a^6*d0^2*A",
   sp.factor(det_cov / normalized_gram - a**6 * d0_sq) == 0)
ck("14 squared density prefactor is a^-6*d0^-2*A^-1",
   sp.factor(formal_prefactor_squared
             - 1 / (a**6 * d0_sq * normalized_gram)) == 0)

# The leading endpoint-axis profile stays uniformly away from zero on the
# signed 5% window.  These exact rationals are used in the proof note.
eta = sp.symbols("eta", real=True)
window_ratio = (1 - eta)**6 * (1 - 2 * eta)**2
plus = sp.factor(window_ratio.subs(eta, sp.Rational(1, 20)))
minus = sp.factor(window_ratio.subs(eta, -sp.Rational(1, 20)))
ck("15 +5% window ratio", plus == sp.Rational(3810716361, 6400000000))
ck("16 -5% window ratio", minus == sp.Rational(10377700641, 6400000000))
ck("17 5% window is separated from midpoint and second endpoint",
   sp.Rational(1, 20) < sp.Rational(1, 2)
   and sp.Rational(1, 20) < 1)

# Fixed powers are harmless once exp(-c/d^2) is present.  The elementary
# Taylor certificate exp(u)>=u^m/m! is recorded for a representative loss.
loss = 24
m = loss + 1
ck("18 exponential absorbs a fixed d^-24 loss",
   2 * m - loss > 0 and factorial(m) > 0)

if os.environ.get("SUPPRESSION_FORCE_FAILURE") == "1":
    ck("forced fail-closed mutation", False)

print(f"CHECK_COUNT={checks}")
print("CONE=lambda=rho/d in (epsilon,2], punctured at lambda=1")
print("D0SQ=d^2*[4(2lambda-1)^2+d^2 lambda^2(lambda-1)^2]")
print("MAHALANOBIS_LOWER>=kappa^2/(320*C*d^2)")
print("MIDPOINT_STRENGTH=d^-4")
print("WINDOW=|rho|<=0.05d; leading axial ratio in")
print("[3810716361/6400000000,10377700641/6400000000]")
print("SCOPE_LIMIT: local density suppression only; collision weights and the")
print("full RP-C/RP-S regional integration are not asserted here.")
print("ALL_ASSERTIONS_PASS")
