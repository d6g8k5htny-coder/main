"""Exact two-axis derivatives of exp(-(x*x+y*y)/2), with a proved envelope.

This is an author-produced component, not an accepted research result. It
establishes no six-pin geometry, full 24-jet binding, band result, obligation
closure, claim promotion, or organizational independence. The evaluator uses
certified interval arithmetic. The new envelope remains flagged UNREVIEWED;
its written proof is in docs/HERMITE_GAUSSIAN_ENVELOPE.md.
"""
from __future__ import annotations

from fractions import Fraction as F
from functools import lru_cache
from math import factorial

from research.interval import Interval, exp
from research.bands.lattice import DecayEnvelope, GAUSSIAN, PlaneKernel


def _order(n: int) -> None:
    if type(n) is not int or n < 0:
        raise ValueError("derivative order must be a non-negative integer")


@lru_cache(maxsize=128, typed=True)
def hermite_coefficients(n: int) -> tuple[int, ...]:
    """Ascending coefficients of probabilists' He_n, in exact integers."""
    _order(n)
    previous, current = (1,), (0, 1)
    if n == 0:
        return previous
    for k in range(1, n):
        result = [0, *current]
        for j, coefficient in enumerate(previous):
            result[j] -= k * coefficient
        previous, current = current, tuple(result)
    return current


def monomial_majorant(n: int) -> F:
    """M_n >= |x|**n exp(-x*x/4), for every real x (proof in companion)."""
    _order(n)
    m = n // 2
    even = F(4**m * factorial(m))
    if n % 2 == 0:
        return even
    return (even + F(4**(m + 1) * factorial(m + 1))) / 2


def hermite_majorant(n: int) -> F:
    """A_n >= |He_n(x)| exp(-x*x/4) globally, by coefficient domination."""
    return sum((abs(c) * monomial_majorant(j)
                for j, c in enumerate(hermite_coefficients(n))), F(0))


def _horner(coefficients: tuple[int, ...], x: Interval) -> Interval:
    value = Interval.exact(0)
    for coefficient in reversed(coefficients):
        value = value * x + coefficient
    return value


def hermite_gaussian(a: int, b: int) -> PlaneKernel:
    """Return D_x**a D_y**b exp(-|z|²/2), with a global two-axis envelope.

    The same normalization and derivative sign as frozen d3_rn_unif.kplane
    are implemented independently. No frozen source is imported or executed.
    Envelope review is pending; downstream fully_certified stays False.
    """
    x_coefficients = hermite_coefficients(a)
    y_coefficients = hermite_coefficients(b)
    amplitude = hermite_majorant(a) * hermite_majorant(b)
    sign = (-1) ** (a + b)

    def evaluate(dx: Interval, dy: Interval, prec: int) -> Interval:
        polynomial = _horner(x_coefficients, dx) * _horner(y_coefficients, dy)
        return sign * polynomial * exp(F(-1, 2) * (dx**2 + dy**2), prec)

    proof = (
        "Let M_(2m)=4^m m! and M_(2m+1)=(M_(2m)+M_(2m+2))/2. "
        "exp(t)>=t^m/m! with t=x^2/4 proves the even monomial bound; "
        "2|x|<=1+x^2 proves the odd bound. For He_n=sum c_j x^j, "
        "A_n=sum |c_j|M_j dominates |He_n(x)|exp(-x^2/4). "
        "Multiplying the two independent axis inequalities proves "
        "|k_ab(x,y)|<=A_a A_b exp(-(x^2+y^2)/4). "
        "See docs/HERMITE_GAUSSIAN_ENVELOPE.md; author proof, review pending."
    )
    return PlaneKernel(
        name=f"hermite-gaussian ({a},{b})",
        evaluate=evaluate,
        envelope=DecayEnvelope(kind=GAUSSIAN, A=amplitude, B=F(1, 4),
                               valid_from=F(0), justification=proof,
                               certified=False),
        certified=True,
        notes=("Exact interval evaluator; author-produced global envelope. "
               "Envelope review pending; no six-pin or full 24-jet binding, "
               "no obligation closure or claim promotion; independence credit 0."),
    )
