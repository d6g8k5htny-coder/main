"""Certified envelopes for the probabilists' Hermite polynomials, in exact rationals.

WHY THIS IS HERE. ``docs/OPEN_PROBLEMS.md`` A5 records, among RN5's next exact
actions, "Build the whitened ``env_form`` orders 2-4 runnable smoke (currently
missing)". ``env_form`` is the moment-series envelope of the frozen RN-UNIF
engine, at
``engine/rn_engine/frozen/K3_SIDE24_LB/UPPER2D/D3_percolation/d3_rn_unif.py``
line 381. Reading it, every quantity it multiplies is one of exactly two kinds:

* ``he_abs(n, t)`` (line 140) -- an **integer-coefficient polynomial** in
  ``|t|``, built by the all-plus recurrence ``_he_abs_poly`` (line 123) and
  tabulated explicitly for ``n <= 10`` (lines 108-121); and
* ``kern(d2) = C_KERN * exp(-d2/2)`` (line 143), with ``C_KERN = 1`` -- "the
  rung kernel is the unit separable Gaussian" (line 97).

The first kind is exactly representable over ``fractions.Fraction``. The second
is the only transcendental in the envelope, and ``research/interval/``
certifies it. So the arithmetic shape of ``env_form`` admits an exact-rational
and certified-interval treatment throughout. That observation is what this
module makes usable, and ``research/rn/env_form_reference.py`` is what consumes it.

The frozen engine is ``mpmath`` at ``mp.dps = 100`` (line 41) throughout. High
precision is not certification: a 100-digit float evaluation with no interval
discipline produces a number, not an enclosure.

WHAT THIS MODULE DOES NOT ESTABLISH
-----------------------------------
* It discharges, reduces, closes, promotes and reclassifies **nothing**.
  ``D3-LEMMA-RN-UNIF`` Piece 1 and Piece 2 stay **OPEN** exactly as
  ``docs/OPEN_PROBLEMS.md`` A5 records them, and the lane receipts still carry
  ``lemma_closed: false``. ``OBL-H5-JETMOD``, ``OBL-H5-ZBAND``,
  ``OBL-H5-REMOTE-THRESHOLD``, ``OBL-D1-PROMOTE``, ``PERC-DECAY``,
  ``OBL-B1-BRANCH`` and the ``B4.loc`` wrap/remote reconciliation stand as
  recorded.
* It **re-certifies no existing number**. Nothing the frozen engine computed in
  ``mpmath`` becomes certified because a certified arithmetic for the same
  shape now exists here. Nothing here relabels anything.
* It supplies **no** moment table, **no** six-pin form, **no** jet, **no** band
  enclosure and **no** coverage certificate. In particular it does not contain
  the program's ``MOMS[k]`` or ``FORMS[k]``, which are frozen-engine geometry;
  a consumer that wants the program's actual envelope must supply them and this
  module will not invent them.
* An enclosure of a Hermite envelope is an enclosure of a Hermite envelope and
  of nothing else. It is not a bound on ``1 - q(r)``, not a cell supremum, and
  not a remote budget.
* It relates the 2D upper, 2D lower and 3D lifetime tracks in no way, and
  composes none of them. It bears on no prize problem.
* Green tests establish that the mutations enumerated in
  ``tests/test_hermite_envelope.py`` are caught. The containment arguments
  below are ordinary mathematics written out for a human to check; passing
  tests are not that check.

Standard library only (``fractions``, ``typing``). Python 3.11. Certified
enclosures come from ``research/interval/``; there is no float anywhere in this
module.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Sequence

from research.interval import Interval, exp

__all__ = [
    "he_coefficients", "he_abs_coefficients", "he", "he_abs",
    "he_interval", "he_abs_interval", "gaussian_kernel", "kernel_exponent",
    "MAX_TABULATED_ORDER",
]

#: The frozen body tabulates ``_HE_ABS`` explicitly for ``n <= 10`` and
#: generates ``11 <= n <= 19`` from ``_he_abs_poly``. The two agree; that they
#: agree is a negative control, not an assumption -- see
#: ``tests/test_hermite_envelope.py::test_frozen_table_matches_the_recurrence``.
MAX_TABULATED_ORDER = 10


def he_coefficients(n: int) -> tuple[int, ...]:
    """Exact integer coefficients of ``He_n``, lowest degree first.

    The probabilists' Hermite polynomials satisfy ``He_0 = 1``, ``He_1 = x``
    and the three-term recurrence ``He_k(x) = x He_{k-1}(x) - (k-1) He_{k-2}(x)``
    -- the frozen body's ``he`` (line 100). The coefficients are integers, so
    this is exact in ``int`` with no rounding anywhere.
    """
    if n < 0:
        raise ValueError("order must be non-negative")
    prev: list[int] = [1]
    if n == 0:
        return (1,)
    cur: list[int] = [0, 1]
    for k in range(2, n + 1):
        # x * cur  -  (k-1) * prev
        nxt = [0] + cur
        for i, v in enumerate(prev):
            nxt[i] -= (k - 1) * v
        prev, cur = cur, nxt
    return tuple(cur)


def he_abs_coefficients(n: int) -> tuple[int, ...]:
    """``|c_k|`` for ``He_n = sum_k c_k x^k``, lowest degree first.

    Every entry is non-negative. This is exactly the coefficient list of the
    polynomial the frozen ``_he_abs_poly`` evaluates; see ``he_abs``.
    """
    return tuple(abs(c) for c in he_coefficients(n))


def _horner(coeffs: Sequence, x):
    """Evaluate ``sum_k coeffs[k] x^k`` by Horner, highest degree first.

    Generic in the coefficient and argument types, so the same code path serves
    ``Fraction`` and ``Interval``. For ``Interval`` the result is an enclosure
    of the polynomial over the input interval -- sound, and for a polynomial
    with sign-mixed coefficients not necessarily tight (the dependency problem;
    ``he_abs_interval`` is the tight one, because its coefficients are all of
    one sign).
    """
    acc = None
    for c in reversed(coeffs):
        acc = c if acc is None else acc * x + c
    return acc


def he(n: int, x) -> Fraction:
    """``He_n(x)`` exactly, for ``int`` or ``Fraction`` ``x``.

    Refuses a ``float``: ``Fraction(0.1)`` is not ``1/10``, and a bound derived
    from a binary double is not a bound on the rational it was printed as. The
    same refusal as ``research/interval/``.
    """
    if isinstance(x, float):
        raise TypeError("he refuses a float argument; pass Fraction or int")
    return Fraction(_horner(he_coefficients(n), Fraction(x)))


def he_abs(n: int, t) -> Fraction:
    """The frozen engine's envelope ``sum_k |c_k| |t|^k``, exactly.

    This reimplements ``he_abs`` of the frozen body (line 140) over
    ``Fraction``. It is a reimplementation for use here, not a re-certification
    of anything the frozen body computed.

    CONTAINMENT. For every real ``x``,

        |He_n(x)|  =  |sum_k c_k x^k|  <=  sum_k |c_k| |x|^k  =  he_abs(n, |x|)

    by the triangle inequality. The bound is an equality when all the terms
    ``c_k x^k`` share a sign, which happens at ``x = 0`` and in the limit of
    large ``|x|`` where the leading term dominates; it is loose near a zero of
    ``He_n``, where the left side vanishes and the right side does not. The
    envelope is rigorous and lossy, and the frozen body's docstring calls it
    exactly that: "the rigorous envelope" (line 124).

    WHY THE ALL-PLUS RECURRENCE GIVES EXACTLY ``|c_k|``. The frozen
    ``_he_abs_poly`` runs ``h_k = t h_{k-1} + (k-1) h_{k-2}``, the He recurrence
    with the minus turned into a plus. That this produces the coefficient-wise
    absolute value -- rather than merely some upper bound -- is the identity

        sum_k |c_k| t^k  =  (-i)^n He_n(i t).

    Proof: ``He_n`` has only degrees ``n, n-2, n-4, ...``, and the coefficient
    of ``x^(n-2j)`` carries the sign ``(-1)^j``. Writing ``k = n - 2j``,

        c_k (it)^k = (-1)^j |c_k| i^(n-2j) t^k = (-1)^j |c_k| i^n (-1)^(-j) t^k
                   = |c_k| i^n t^k,

    so ``He_n(it) = i^n sum_k |c_k| t^k``. Every sign has been absorbed and
    nothing cancels, which is also why the two contributions to any one
    coefficient in the recurrence never cancel: both carry the sign
    ``(-1)^((k-m)/2)``. Hence the all-plus recurrence is the
    coefficient-wise-absolute-value polynomial on the nose.

    ``tests/test_hermite_envelope.py`` pins this three ways: against the frozen
    body's explicit table for ``n <= 10``, against ``abs`` of
    ``he_coefficients``, and against the ``(-i)^n He_n(it)`` identity.
    """
    if isinstance(t, float):
        raise TypeError("he_abs refuses a float argument; pass Fraction or int")
    return Fraction(_horner(he_abs_coefficients(n), abs(Fraction(t))))


def he_interval(n: int, x: Interval) -> Interval:
    """A certified enclosure of ``He_n`` over the interval ``x``.

    Horner in interval arithmetic. Containment is unconditional: every
    operation ``research/interval/`` performs is outward-rounded, so the result
    contains ``He_n(t)`` for every ``t`` in ``x``.

    Tightness is not claimed and is often poor: ``He_n`` has sign-mixed
    coefficients, so Horner suffers the dependency problem and the returned
    width can exceed the true range by a wide margin on a wide input. For an
    envelope, use ``he_abs_interval``, whose coefficients are all of one sign
    and which is therefore monotone on non-negative inputs.
    """
    if not isinstance(x, Interval):
        raise TypeError("he_interval needs an Interval")
    return _horner([Interval.exact(c) for c in he_coefficients(n)], x)


def he_abs_interval(n: int, x: Interval) -> Interval:
    """A certified enclosure of the envelope ``he_abs(n, .)`` over ``x``.

    The envelope has non-negative coefficients and is evaluated at ``|t|``, so
    it is non-decreasing on ``[0, inf)``. The enclosure is therefore exact at
    the endpoints: ``[he_abs(n, mig), he_abs(n, mag)]`` where ``mig`` and
    ``mag`` are the smallest and largest ``|t|`` attained on ``x``. No interval
    Horner is needed and none is used, so there is no dependency-problem
    widening here at all.

    THE THEOREM THIS MODULE EXISTS FOR. For every ``t`` in ``x``,

        he_interval(n, x)  is contained in  [-hi, hi]   where   hi = he_abs(n, x.mag())

    ``tests/test_hermite_envelope.py`` asserts exactly that containment, and a
    negative control asserts it FAILS when the envelope is evaluated at
    ``mig()`` instead of ``mag()``.
    """
    if not isinstance(x, Interval):
        raise TypeError("he_abs_interval needs an Interval")
    return Interval(he_abs(n, x.mig()), he_abs(n, x.mag()))


def kernel_exponent(d2) -> Fraction:
    """``-d2/2``, the exponent of the frozen ``kern`` (line 143), exactly."""
    if isinstance(d2, float):
        raise TypeError("kernel_exponent refuses a float; pass Fraction or int")
    return -Fraction(d2) / 2


def gaussian_kernel(d2: Interval, prec: int) -> Interval:
    """A certified enclosure of the frozen ``kern(d2) = exp(-d2/2)``.

    ``C_KERN`` is ``1`` in the frozen body -- "the rung kernel is the unit
    separable Gaussian ``e^{-d^2/2}``" (line 97) -- so the constant is carried
    here as the exact rational ``1`` and not as a fitted or measured value. If
    a consumer needs a different normalisation it must multiply by its own
    exactly-stated constant; this function will not guess one.

    ``prec`` is a RELATIVE width hint, as ``research/interval/exp`` documents;
    containment does not depend on it. Underflow past ``EXP_BIT_LIMIT``
    degrades the lower endpoint to exactly ``0``, which is sound and, for a
    lower bound, useless -- ``research/interval/README.md`` states the limit and
    the magnitudes at which it bites.
    """
    if not isinstance(d2, Interval):
        raise TypeError("gaussian_kernel needs an Interval")
    return exp(Interval(-d2.hi / 2, -d2.lo / 2), prec)
