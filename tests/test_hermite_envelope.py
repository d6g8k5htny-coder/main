"""Tests and negative controls for ``research/rn/hermite_envelope.py``.

WHAT IS UNDER TEST. The module reimplements, over exact rationals, the two
quantities the frozen RN-UNIF engine's ``env_form`` multiplies: the Hermite
envelope ``he_abs`` and the Gaussian kernel ``kern``. The frozen body is
``engine/rn_engine/frozen/K3_SIDE24_LB/UPPER2D/D3_percolation/d3_rn_unif.py``;
it is read here and never edited.

The load-bearing claim is a containment: ``|He_n(x)| <= he_abs(n, |x|)``, and
its interval form ``he_interval(n, x) subset [-he_abs(n, x.mag()), +...]``.
Five negative controls build deliberately weakened envelopes INSIDE this file
-- never by editing the library -- and assert that containment then FAILS. An
envelope that cannot be broken by dropping a term is not being tested.

None of this closes Piece 1 or Piece 2 of ``D3-LEMMA-RN-UNIF``, both of which
stay OPEN, and none of it re-certifies anything the frozen engine computed.
"""
from __future__ import annotations

import os
import sys
from fractions import Fraction as F

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.interval import Interval  # noqa: E402
from research.rn.hermite_envelope import (  # noqa: E402
    FROZEN_MAX_ORDER, MAX_TABULATED_ORDER, gaussian_kernel, he, he_abs,
    he_abs_coefficients, he_abs_interval, he_coefficients, he_interval,
    kernel_exponent,
)

# ---------------------------------------------------------------------------
# The frozen body, transcribed. Lines 108-121 (the explicit table) and lines
# 123-135 (the generator used for 11 <= n <= 19). Transcription, not import:
# the frozen body is mpmath and must not be imported into a bound path.
# ---------------------------------------------------------------------------

FROZEN_TABLE = {
    0: lambda t: F(1),
    1: lambda t: t,
    2: lambda t: t**2 + 1,
    3: lambda t: t**3 + 3 * t,
    4: lambda t: t**4 + 6 * t**2 + 3,
    5: lambda t: t**5 + 10 * t**3 + 15 * t,
    6: lambda t: t**6 + 15 * t**4 + 45 * t**2 + 15,
    7: lambda t: t**7 + 21 * t**5 + 105 * t**3 + 105 * t,
    8: lambda t: t**8 + 28 * t**6 + 210 * t**4 + 420 * t**2 + 105,
    9: lambda t: t**9 + 36 * t**7 + 378 * t**5 + 1260 * t**3 + 945 * t,
    10: lambda t: (t**10 + 45 * t**8 + 630 * t**6 + 3150 * t**4
                   + 4725 * t**2 + 945),
}


def frozen_he_abs_poly(n: int, t: F) -> F:
    """``_he_abs_poly`` of the frozen body: the all-plus recurrence."""
    t = abs(t)
    if n == 0:
        return F(1)
    if n == 1:
        return t
    h0, h1 = F(1), t
    for k in range(2, n + 1):
        h0, h1 = h1, t * h1 + (k - 1) * h0
    return h1


def frozen_he(n: int, x: F) -> F:
    """``he`` of the frozen body (line 100): the true signed ``He_n``."""
    if n == 0:
        return F(1)
    if n == 1:
        return x
    h0, h1 = F(1), x
    for k in range(2, n + 1):
        h0, h1 = h1, x * h1 - (k - 1) * h0
    return h1


SAMPLE = [F(0), F(1, 97), F(1, 3), F(1), F(7, 4), F(13, 4), F(17), F(-1, 3),
          F(-1), F(-9, 2), F(-17)]


# ---------------------------------------------------------------------------
# 1. Agreement with the frozen body
# ---------------------------------------------------------------------------

def test_frozen_table_matches_the_module():
    """The explicit table for n <= 10 is reproduced exactly."""
    for n in range(MAX_TABULATED_ORDER + 1):
        for t in SAMPLE:
            assert he_abs(n, t) == FROZEN_TABLE[n](abs(t)), (n, t)


def test_frozen_table_matches_the_recurrence():
    """The frozen body's own two branches agree with each other.

    ``_HE_ABS`` is a table for ``n <= 10`` and ``_he_abs_poly`` for
    ``11 <= n <= 19``. That the branches agree is checked, not assumed -- a
    disagreement would be a defect in the frozen body worth recording.
    """
    for n in range(MAX_TABULATED_ORDER + 1):
        for t in SAMPLE:
            assert FROZEN_TABLE[n](abs(t)) == frozen_he_abs_poly(n, t), (n, t)


def test_module_matches_the_recurrence_above_the_table():
    """Orders 11-19, the range the frozen body generates."""
    for n in range(11, 20):
        for t in SAMPLE:
            assert he_abs(n, t) == frozen_he_abs_poly(n, t), (n, t)


def test_he_matches_the_frozen_signed_recurrence():
    for n in range(20):
        for x in SAMPLE:
            assert he(n, x) == frozen_he(n, x), (n, x)


# ---------------------------------------------------------------------------
# 2. The envelope is exactly the coefficient-wise absolute value
# ---------------------------------------------------------------------------

def test_he_abs_coefficients_are_abs_of_he_coefficients():
    for n in range(20):
        assert he_abs_coefficients(n) == tuple(abs(c) for c in he_coefficients(n))


def test_the_all_plus_recurrence_is_the_absolute_value_identity():
    """``sum_k |c_k| t^k = (-i)^n He_n(i t)``, checked in exact Gaussian rationals.

    This is the proof written out in ``he_abs``'s docstring. If the identity
    held only approximately the envelope would be some upper bound rather than
    the coefficient-wise absolute value, and the docstring would be wrong.
    """
    def he_at_i(n, t):
        h0, h1 = (F(1), F(0)), (F(0), t)
        if n == 0:
            return h0
        if n == 1:
            return h1
        for k in range(2, n + 1):
            a, b = h1
            na, nb = -t * b, t * a          # (i t) * (a + b i)
            na -= (k - 1) * h0[0]
            nb -= (k - 1) * h0[1]
            h0, h1 = h1, (na, nb)
        return h1

    for n in range(16):
        for t in (F(1, 3), F(2), F(9, 2)):
            a, b = he_at_i(n, t)
            for _ in range(n % 4):
                a, b = b, -a                # multiply by (-i)
            assert b == 0, (n, t)
            assert a == he_abs(n, t), (n, t)


def test_only_the_degrees_of_one_parity_are_present():
    """``He_n`` carries degrees n, n-2, n-4, ... and no others."""
    for n in range(20):
        for k, c in enumerate(he_coefficients(n)):
            if (n - k) % 2:
                assert c == 0, (n, k, c)


# ---------------------------------------------------------------------------
# 3. The containment the module exists for
# ---------------------------------------------------------------------------

def test_envelope_dominates_he_pointwise():
    for n in range(20):
        for num in range(-80, 81):
            x = F(num, 9)
            assert abs(he(n, x)) <= he_abs(n, abs(x)), (n, x)


def test_he_interval_is_contained_in_the_envelope():
    """The theorem: ``he_interval(n, x) subset [-he_abs(n, mag), +he_abs(n, mag)]``."""
    for n in range(14):
        for lo, hi in ((F(-3), F(2)), (F(1, 5), F(1, 2)), (F(-7), F(-1)),
                       (F(0), F(6)), (F(-1, 10), F(1, 10))):
            x = Interval(lo, hi)
            env = he_abs(n, x.mag())
            got = he_interval(n, x)
            assert got.lo >= -env and got.hi <= env, (n, lo, hi)


def test_he_interval_encloses_every_point_of_the_input():
    for n in range(12):
        x = Interval(F(-2), F(3))
        got = he_interval(n, x)
        for num in range(-20, 31):
            t = F(num, 10)
            assert got.lo <= he(n, t) <= got.hi, (n, t)


def test_he_abs_interval_endpoints_are_exact():
    """The envelope is monotone on non-negatives, so no widening is needed."""
    x = Interval(F(1, 2), F(3))
    for n in range(14):
        got = he_abs_interval(n, x)
        assert got.lo == he_abs(n, F(1, 2)), n
        assert got.hi == he_abs(n, F(3)), n


def test_he_abs_interval_on_a_straddling_input_starts_at_zero():
    """``mig`` of an interval containing 0 is 0, so the envelope starts there."""
    x = Interval(F(-2), F(5))
    for n in range(1, 10):
        assert he_abs_interval(n, x).lo == he_abs(n, F(0))
        assert he_abs_interval(n, x).hi == he_abs(n, F(5))


def test_he_abs_is_non_decreasing_on_non_negatives():
    for n in range(20):
        vals = [he_abs(n, F(j, 4)) for j in range(0, 40)]
        assert vals == sorted(vals), n


# ---------------------------------------------------------------------------
# 4. NEGATIVE CONTROLS. Each weakens the envelope here, never in the library,
#    and asserts that containment then FAILS.
# ---------------------------------------------------------------------------

def test_control_signed_coefficients_are_not_an_envelope():
    """Using ``c_k`` instead of ``|c_k|`` is exactly ``He_n``, not a bound."""
    def weakened(n, t):
        t = abs(t)
        return sum(c * t**k for k, c in enumerate(he_coefficients(n)))

    broke = False
    for n in range(2, 12):
        for num in range(1, 40):
            x = F(num, 10)
            if abs(he(n, -x)) > weakened(n, x):
                broke = True
    assert broke, "signed coefficients must fail to dominate somewhere"


def test_control_dropping_the_constant_term_breaks_the_envelope():
    def weakened(n, t):
        cs = list(he_abs_coefficients(n))
        cs[0] = 0
        return sum(c * abs(t)**k for k, c in enumerate(cs))

    broke = False
    for n in range(2, 12, 2):            # even n have a non-zero constant term
        if abs(he(n, F(0))) > weakened(n, F(0)):
            broke = True
    assert broke, "dropping the constant term must break containment at t = 0"


def test_control_the_true_recurrence_is_not_an_envelope():
    """``h1 = t*h1 - (k-1)*h0`` -- the minus the frozen envelope turns to plus.

    Running the signed recurrence and calling the result an envelope is the
    single-character mutation this module is most exposed to. It must fail:
    the signed value is strictly below the envelope at every order where any
    cancellation happens at all.
    """
    strictly_below = [n for n in range(2, 14)
                      if abs(frozen_he(n, F(1))) < frozen_he_abs_poly(n, F(1))]
    assert strictly_below == list(range(2, 14)), strictly_below


def test_control_envelope_at_mig_does_not_contain():
    """The theorem needs ``mag()``. Taking ``mig()`` must fail."""
    broke = False
    for n in range(2, 12):
        x = Interval(F(1, 10), F(4))
        env_wrong = he_abs(n, x.mig())
        got = he_interval(n, x)
        if got.hi > env_wrong or got.lo < -env_wrong:
            broke = True
    assert broke, "an envelope taken at mig() must fail to contain"


def test_control_truncating_the_leading_term_breaks_the_envelope():
    def weakened(n, t):
        cs = list(he_abs_coefficients(n))
        cs[-1] = 0
        return sum(c * abs(t)**k for k, c in enumerate(cs))

    broke = False
    for n in range(2, 12):
        t = F(20)                         # large t: the leading term dominates
        if abs(he(n, t)) > weakened(n, t):
            broke = True
    assert broke, "dropping the leading term must break containment at large t"


def test_control_kernel_with_the_exponent_sign_flipped_does_not_contain():
    from research.interval import exp as _exp
    d2 = Interval(F(2), F(2))
    good = gaussian_kernel(d2, 30)
    bad = _exp(Interval(F(1), F(1)), 30)          # exp(+d2/2) instead of exp(-d2/2)
    assert good.hi < bad.lo, "a sign-flipped exponent must not enclose the kernel"


# ---------------------------------------------------------------------------
# 5. The kernel
# ---------------------------------------------------------------------------

def test_kernel_exponent_is_exact():
    assert kernel_exponent(F(9, 4)) == F(-9, 8)
    assert kernel_exponent(0) == 0


# Published decimal brackets, each straddling the true value. Any enclosure of
# that value must MEET its bracket. Asserting instead that the enclosure
# CONTAINS a truncated decimal is the wrong test and fails at high prec: a
# 16-digit truncation of exp(-1/2) lies just below the true value, and a tight
# enough enclosure correctly excludes it. Meeting is the sound relation, and
# getting this backwards cost one red run here.
EXP_HALF = (F(60653065971263342360, 10**20), F(60653065971263342361, 10**20))
EXP_TWO = (F(13533528323661269189, 10**20), F(13533528323661269190, 10**20))


def _meets(iv, bracket):
    lo, hi = bracket
    return iv.lo <= hi and lo <= iv.hi


def test_gaussian_kernel_meets_the_true_values():
    assert _meets(gaussian_kernel(Interval(F(1), F(1)), 40), EXP_HALF)
    assert _meets(gaussian_kernel(Interval(F(4), F(4)), 40), EXP_TWO)


def test_a_wide_input_spans_both_endpoint_values():
    """An enclosure over d2 in [1, 4] must meet BOTH endpoint brackets.

    Not "contain the bracket": the enclosure is tighter than a 20-digit
    bracket, so containment of the bracket is false while containment of the
    true value holds. Meeting is the relation a bracketed oracle supports.
    """
    k = gaussian_kernel(Interval(F(1), F(4)), 40)
    assert _meets(k, EXP_TWO) and _meets(k, EXP_HALF)
    assert k.width() >= EXP_HALF[0] - EXP_TWO[1]


def test_gaussian_kernel_is_decreasing_in_d2():
    near = gaussian_kernel(Interval(F(1), F(1)), 30)
    far = gaussian_kernel(Interval(F(9), F(9)), 30)
    assert far.hi < near.lo


def test_gaussian_kernel_at_zero_is_one():
    k = gaussian_kernel(Interval(F(0), F(0)), 30)
    assert k.lo <= 1 <= k.hi


def test_tighter_prec_never_breaks_containment():
    """Containment is unconditional in ``prec``; only the width may move.

    Every enclosure contains the true value, so every pair of them must meet,
    and each must meet the published bracket. Widths must not grow with prec.
    """
    encs = [gaussian_kernel(Interval(F(1), F(1)), p) for p in (5, 10, 20, 40, 80)]
    for e in encs:
        assert _meets(e, EXP_HALF)
    for a in encs:
        for b in encs:
            assert a.lo <= b.hi and b.lo <= a.hi
    widths = [e.width() for e in encs]
    assert widths == sorted(widths, reverse=True), widths


# ---------------------------------------------------------------------------
# 6. Refusals
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("call", [
    lambda: he(3, 0.5),
    lambda: he_abs(3, 0.5),
    lambda: kernel_exponent(0.5),
])
def test_float_arguments_are_refused(call):
    with pytest.raises(TypeError):
        call()


@pytest.mark.parametrize("call", [
    lambda: he_interval(3, F(1, 2)),
    lambda: he_abs_interval(3, F(1, 2)),
    lambda: gaussian_kernel(F(1, 2), 30),
])
def test_non_interval_arguments_are_refused(call):
    with pytest.raises(TypeError):
        call()


def test_negative_order_is_refused():
    with pytest.raises(ValueError):
        he_coefficients(-1)


# ---------------------------------------------------------------------------
# 7. The frozen table's order ceiling, recorded and shown not to apply here
# ---------------------------------------------------------------------------

def test_the_frozen_ceiling_is_where_the_frozen_body_puts_it():
    """`_HE_ABS` holds 0..19 and `he_abs` is a bare lookup, so 20 raises."""
    assert FROZEN_MAX_ORDER == 19
    assert MAX_TABULATED_ORDER < FROZEN_MAX_ORDER


def test_this_module_has_no_order_ceiling():
    """The coefficients come from the recurrence, so any order is available."""
    for n in (FROZEN_MAX_ORDER, FROZEN_MAX_ORDER + 1, 40, 60):
        v = he_abs(n, F(1))
        assert v > 0
        assert he_abs_coefficients(n)[-1] == 1     # monic, at every order
    assert he_abs(FROZEN_MAX_ORDER + 1, F(1)) > he_abs(FROZEN_MAX_ORDER, F(1))


def test_the_ceiling_would_bite_through_the_remainder_index():
    """`mx2` reaches index 11 + max(gamma) + qord; at gamma = 0 that is qord = 9."""
    from research.rn.env_form_reference import MX2_INDEX_SUM  # noqa: PLC0415
    assert MX2_INDEX_SUM + 0 + 9 > FROZEN_MAX_ORDER
    assert MX2_INDEX_SUM + 0 + 8 <= FROZEN_MAX_ORDER
    for qord in (0, 1, 2, 3, 4):                   # the orders built here
        assert MX2_INDEX_SUM + 4 + qord <= FROZEN_MAX_ORDER + 4
