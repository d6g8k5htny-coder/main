#!/usr/bin/env python3
"""Exact rational helpers for the proposed D2 two-atom floors.

This is a finite-model checker, not interval arithmetic or a Lean proof.
All arithmetic accepts only Python int (not bool) or Fraction values.
"""
from fractions import Fraction
from typing import Iterable

Rational = int | Fraction

def rational(x: Rational) -> Fraction:
    if isinstance(x, bool) or not isinstance(x, (int, Fraction)):
        raise TypeError("exact integer or Fraction input required")
    return Fraction(x)

def two_square_floor(A: Rational, B: Rational, u: Rational, v: Rational) -> Fraction:
    A, B, u, v = map(rational, (A, B, u, v))
    if A <= 0 or B <= 0:
        raise ValueError("positive two-square weights required")
    return A*B*(u-v)**2/(A+B)

def even_moments(law: Iterable[tuple[Rational, Rational]], *,
                 probability: bool = True) -> tuple[Fraction, Fraction, Fraction]:
    rows = [(rational(x), rational(w)) for x, w in law]
    if not rows or any(w < 0 for _, w in rows):
        raise ValueError("nonempty nonnegative finite measure required")
    total = sum((w for _, w in rows), Fraction(0))
    if total <= 0:
        raise ValueError("positive total mass required")
    if probability and total != 1:
        raise ValueError("probability mass must equal one")
    return tuple(sum((w*x**j for x, w in rows), Fraction(0)) for j in (2, 4, 6))

def atom_floors(p: Rational, q: Rational, a: Rational, b: Rational
                ) -> dict[str, Fraction]:
    """Sufficient all-direction floors for a probability law with two atom masses.

    p,q are positive lower bounds for masses at a,b; a,b are nonzero and
    have distinct squared radii. No actual-law realization is constructed.
    """
    p, q, a, b = map(rational, (p, q, a, b))
    if p <= 0 or q <= 0 or p+q > 1:
        raise ValueError("positive atom mass bounds with p+q<=1 required")
    if a == 0 or b == 0 or a*a == b*b:
        raise ValueError("nonzero distinct squared atom radii required")
    u, v = a*a, b*b
    m = p*u + q*v
    gap = two_square_floor(p, q, u, v)
    delta = two_square_floor(p*u, q*v, u, v)
    determinant = min(m*m*(gap+m*m), gap*(gap+4*m*m)/4)
    schur = min(gap, 2*m*m)
    tau = min(delta, (delta+9*m*gap)/4)
    return {"m2":m, "gap":gap, "delta":delta,
            "det":determinant, "schur":schur, "tau":tau}

def d2_values(m2: Rational, m4: Rational, m6: Rational, direction: Rational
              ) -> tuple[Fraction, Fraction, Fraction]:
    """Evaluate the source's scalar D,S,T formulas with a nonzero D guard."""
    s, m4, m6, t = map(rational, (m2, m4, m6, direction))
    if s <= 0 or not 0 <= t <= Fraction(1, 4):
        raise ValueError("positive m2 and direction in [0,1/4] required")
    g = m4-s*s
    D = s*s*m4+(m4-3*s*s)*(m4+s*s)*t
    if D <= 0:
        raise ValueError("positive determinant required")
    delta = m6-m4*m4/s
    S = s*s*g*(g+2*s*s)/D
    T = delta*(1-3*t)+9*t*s*g
    return D, S, T
