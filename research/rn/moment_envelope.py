"""Determinant-moment envelopes for the RN near-region bound, in exact rationals.

Background. Under the nine-pin conditional law write ``A = det H_M``,
``B = det H_S``, ``C = det H_y``. Dropping the maximum/saddle indicators and
applying Hölder with exponents ``(4, 4, 2)`` gives

    E[ |ABC| · 1{M max} · 1{S saddle} · 1{y saddle} ]
        <= (E A^4 · E B^4)^(1/4) · (E C^2)^(1/2)                        (1)

The pinned ``d3_perc.py`` helper ``envelope_v`` instead returned

    (E A^4 · E B^4)^(1/4) · (E C^4)^(1/2)                        (DEFECTIVE)

and described it as "Cauchy–Schwarz twice". Because ``det_moment(mu, S, k)`` is
the k-th determinant moment, that substitutes the **fourth** moment where the
**second** is required. For small determinants the replacement can *decrease*
the bound, so the defective expression is not an upper bound at all.

RN5 (2026-09-17) established this with an exact typed Gaussian counterexample.
This module reimplements both expressions and that counterexample over
``fractions.Fraction`` so the defect stays permanently falsifiable in CI; see
``tests/test_rn_moment_envelope.py``.

Nothing here re-proves RN3 or closes any premise.

On RN3 and the affected scope, stated precisely. RN3's *far-region proof* uses
the correct second moment and is outside the affected scope. That is true of the
far-region proof and of nothing else in that object: §9's conditional arithmetic
imports the near target ``17.6804 r^3``, which the RN5 repair places inside or
immediately adjacent to the affected scope — the corrected diagnostic is about
``2.34195 r^3`` against a wrong-power ``17.67237 r^3``, a factor of roughly 7.5.
So the displayed sum ``I_near + I_far < 20.51352 r^3`` is built on a number the
erratum touches. The object itself hedges that line as "Conditional arithmetic
only" and "not presently a full remote certificate". Read the scope sentence as
covering the whole object and you carry ``20.51352 r^3`` out of scope by mistake.

Recorded by the nonauthor technical review of RV-RN3
(``reviews/records/REV-RN3-FARZONE-20260918.json``, MAJOR finding on §9), which
also observed that the object's two named zones do not exhaust the domain: far
is ``|y| >= 5`` and near is ``0.1 <= |y| <= 5``, leaving the open disc
``|y| < 0.1`` unaccounted for in the §9 sum.
"""
from __future__ import annotations

from fractions import Fraction as F

__all__ = [
    "normal_moment", "det_moment", "envelope_correct_pow4",
    "envelope_defective_pow4", "counterexample",
]


def normal_moment(mu: F, var: F, n: int) -> F:
    """E[X^n] for X ~ Normal(mu, var), exactly, for 0 <= n <= 4."""
    if n == 0:
        return F(1)
    if n == 1:
        return mu
    if n == 2:
        return mu * mu + var
    if n == 3:
        return mu**3 + 3 * mu * var
    if n == 4:
        return mu**4 + 6 * mu * mu * var + 3 * var * var
    raise ValueError("only moments up to order 4 are implemented")


def det_moment(mu_xx: F, mu_yy: F, var: F, k: int) -> F:
    """E[D^k] for D = xx*yy - xy^2 with independent coordinates.

    ``xx ~ N(mu_xx, var)``, ``yy ~ N(mu_yy, var)``, ``xy ~ N(0, var)``.
    Implemented for k = 2 and k = 4 by expanding the binomial and using the
    central Gaussian even moments E[w^(2j)] = (2j-1)!! * var^j.
    """
    m = normal_moment
    if k == 2:
        # E[(uv)^2] - 2 E[uv] E[w^2] + E[w^4]
        return (m(mu_xx, var, 2) * m(mu_yy, var, 2)
                - 2 * mu_xx * mu_yy * var
                + 3 * var**2)
    if k == 4:
        # E[(uv)^4] - 4 E[(uv)^3] E[w^2] + 6 E[(uv)^2] E[w^4]
        #   - 4 E[uv] E[w^6] + E[w^8]
        return (m(mu_xx, var, 4) * m(mu_yy, var, 4)
                - 4 * m(mu_xx, var, 3) * m(mu_yy, var, 3) * var
                + 6 * m(mu_xx, var, 2) * m(mu_yy, var, 2) * 3 * var**2
                - 4 * mu_xx * mu_yy * 15 * var**3
                + 105 * var**4)
    raise ValueError("only k = 2 and k = 4 are implemented")


def envelope_correct_pow4(ea4: F, eb4: F, ec2: F) -> F:
    """The fourth power of the correct Hölder(4, 4, 2) envelope (1).

    Returning the fourth power keeps the comparison exact: the envelope itself
    involves a fourth root and a square root.
    """
    return ea4 * eb4 * ec2**2


def envelope_defective_pow4(ea4: F, eb4: F, ec4: F) -> F:
    """The fourth power of the defective ``envelope_v`` expression."""
    return ea4 * eb4 * ec4**2


def counterexample() -> dict:
    """The exact RN5 typed, nondegenerate Gaussian counterexample.

    Nine symmetric-Hessian coordinates, mutually independent, each of variance
    1e-6. Their (xx, yy, xy) means are (-1, -1, 0), (1, -1, 0) and (1, -1/4, 0),
    so the first Hessian is negative definite and the other two are saddles.

    With probability at least 91/100 every coordinate is within 1/100 of its
    mean (union bound over nine coordinates plus Chebyshev, each coordinate
    failing with probability at most 1e-6 / (1/100)^2 = 1/100). On that event
    the determinant magnitudes are at least 98/100, 9801/10000 and 2376/10000,
    so the typed expectation is at least their product times 91/100.
    """
    var = F(1, 10**6)
    delta = F(1, 100)

    means = {"A": (F(-1), F(-1)), "B": (F(1), F(-1)), "C": (F(1), F(-1, 4))}
    ea4 = det_moment(*means["A"], var, 4)
    eb4 = det_moment(*means["B"], var, 4)
    ec2 = det_moment(*means["C"], var, 2)
    ec4 = det_moment(*means["C"], var, 4)

    # |det| lower bounds on the event, from the interval corners.
    #   A: xx, yy <= -(1 - delta)  ->  det >= (1-delta)^2 - delta^2
    #   B: xx >= 1-delta, yy <= -(1-delta)  ->  |det| >= (1-delta)^2
    #   C: xx >= 1-delta, yy <= -(1/4 - delta)  ->  |det| >= (1-delta)(1/4-delta)
    det_a = (1 - delta) ** 2 - delta**2
    det_b = (1 - delta) ** 2
    det_c = (1 - delta) * (F(1, 4) - delta)
    prob = 1 - 9 * (var / delta**2)

    return {
        "var": var,
        "delta": delta,
        "prob_event": prob,
        "det_lower": {"A": det_a, "B": det_b, "C": det_c},
        "EA4": ea4,
        "EB4": eb4,
        "EC2": ec2,
        "EC4": ec4,
        "typed_expectation_lower": prob * det_a * det_b * det_c,
        "correct_pow4": envelope_correct_pow4(ea4, eb4, ec2),
        "defective_pow4": envelope_defective_pow4(ea4, eb4, ec4),
    }


if __name__ == "__main__":
    c = counterexample()
    # NON-CERTIFYING display block. Every comparison this module makes is an
    # exact rational comparison of fourth powers; the decimals below are a
    # float rendering for reading and no bound may be taken from one.
    print("(decimals below are NON-CERTIFYING displays of exact rationals)")
    print(f"P(event)                  >= {c['prob_event']} = {float(c['prob_event'])}")
    print(f"typed expectation         >= {float(c['typed_expectation_lower']):.12f}")
    print(f"defective envelope_v       = {float(c['defective_pow4']) ** 0.25:.12f}")
    print(f"correct Hoelder(4,4,2)     = {float(c['correct_pow4']) ** 0.25:.12f}")
    ok = c["defective_pow4"] < c["typed_expectation_lower"] ** 4
    print(f"defective bound is violated: {ok}")
