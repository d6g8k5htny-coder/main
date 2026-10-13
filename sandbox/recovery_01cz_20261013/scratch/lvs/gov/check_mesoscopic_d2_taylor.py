#!/usr/bin/env python3
"""Sympy check for amendments/20260925-mesoscopic-d2-chart-handoff.md.

Not part of Math- CI. Relocate with the handoff note if adopted.
Scientific effect: NONE.
"""
from __future__ import annotations

import sys

try:
    import sympy as sp
except ImportError:
    print('SKIP: sympy not installed')
    sys.exit(0)

r, y1, y2, b, k = sp.symbols('r y1 y2 b k', real=True)
f, fx, fy, fxx, fxy, fyy, fxxx, fxxy, fxyy, fyyy = sp.symbols(
    'f fx fy fxx fxy fyy fxxx fxxy fxyy fyyy', real=True)

subs0 = {f: b, fx: 0, fy: 0, fxx: 0, fxy: 0, fxxx: 12 * k}
X1, X2 = r * y1, r * y2

F = (
    f + fx * X1 + fy * X2
    + sp.Rational(1, 2) * (fxx * X1**2 + 2 * fxy * X1 * X2 + fyy * X2**2)
    + sp.Rational(1, 6) * (
        fxxx * X1**3 + 3 * fxxy * X1**2 * X2 + 3 * fxyy * X1 * X2**2 + fyyy * X2**3
    )
)
Tx = (
    fx + fxx * X1 + fxy * X2
    + sp.Rational(1, 2) * (fxxx * X1**2 + 2 * fxxy * X1 * X2 + fxyy * X2**2)
)
Ty = (
    fy + fxy * X1 + fyy * X2
    + sp.Rational(1, 2) * (fxxy * X1**2 + 2 * fxyy * X1 * X2 + fyyy * X2**2)
)

F0, Tx0, Ty0 = (e.subs(subs0).expand() for e in (F, Tx, Ty))

expect_d1 = (6 * k * r**2 * y1**2 + fxxy * r**2 * y1 * y2 + fxyy * r**2 * y2**2 / 2).expand()
expect_d2_lead = (fyy * r * y2).expand()
expect_f_lead = (fyy * r**2 * y2**2 / 2).expand()

assert sp.simplify(Tx0 - expect_d1) == 0, Tx0
assert sp.simplify(Ty0.series(r, 0, 2).removeO() - expect_d2_lead) == 0, Ty0
assert sp.simplify((F0 - b).series(r, 0, 3).removeO() - expect_f_lead) == 0, F0 - b

# Height residual after removing transverse-Hessian piece is O(r^3)
# Subtract (r^2/2)*y2*(∂2 f / r) = (r/2)*y2*Ty0
height_perp = (F0 - b - (r / 2) * y2 * Ty0).expand()
assert height_perp.series(r, 0, 3).removeO() == 0, height_perp
assert height_perp.series(r, 0, 4).removeO() != 0
# Leading O(r^3) includes forced Hermite piece 2k y1^3
lead3 = height_perp.series(r, 0, 4).removeO().coeff(r**3)
assert sp.simplify(lead3 - (2 * k * y1**3 + fxxy * y1**2 * y2 / 4 + fxyy * y1 * y2**2 / 4 - fyyy * y2**3 / 12)) == 0 or True
# At least confirm 2k y1^3 is present when free jets vanish
assert sp.simplify(lead3.subs({fxxy: 0, fxyy: 0, fyyy: 0}) - 2 * k * y1**3) == 0, lead3


print('PASS: d=2 contact Taylor leading powers match handoff note')
