"""Tests for the certified interval library in ``research/interval/``.

THE CONTRACT UNDER TEST. Every ``Interval`` returned by the library provably
contains the true value; containment is unconditional and does not depend on
the ``prec`` hint. Tightness is best effort and does depend on it.

These tests are organised in six groups:

1. ``Interval`` algebra, including the non-monotone traps.
2. Containment of independently-sourced high-precision decimal constants.
3. Tightness monotonicity in ``prec``.
4. **NEGATIVE CONTROLS** — eight deliberately weakened variants, each built
   locally inside this file (never by editing the library), each asserted to
   LOSE containment. These are the point of the file. If a negative control
   stops failing, the corresponding safeguard in the library has been removed
   or has stopped working.
5. A seeded, deterministic float cross-check sweep, labelled NON-CERTIFYING.
6. **AUDIT REGRESSIONS** — one test per defect reported by the four adversarial
   audits of 2026-09-18, each of which was reproduced on the unfixed code
   before anything was changed, plus negative controls for the coverage gaps
   those audits found by mutation (``__abs__``, the sign of the ``sin``
   extremum at negative odd multiples of ``pi/2``, and the endpoint selection of
   the monotone functions on WIDE inputs, where three genuine
   containment-breaking mutants survived the whole suite). Every test in group
   6 was run against a deliberately broken copy of the library and confirmed to
   fail there; each names its mutation in its docstring.

WHAT THESE TESTS DO NOT ESTABLISH
---------------------------------
* A green build is not a mathematical review. The containment arguments are
  written out as short proofs in the library docstrings; a human must read
  them. These tests check that the code behaves as those proofs say, on the
  inputs exercised here.
* Nothing here discharges, closes, promotes or reclassifies any obligation.
  ``OBL-H5-JETMOD``, ``OBL-H5-ZBAND``, ``OBL-H5-REMOTE-THRESHOLD``,
  ``OBL-D1-PROMOTE``, ``D3-LEMMA-RN-UNIF``, ``PERC-DECAY`` and
  ``OBL-B1-BRANCH`` stand exactly as ``docs/OPEN_PROBLEMS.md`` records them.
  A certified arithmetic is not a certified result.
* The float sweep in group 5 is **NON-CERTIFYING**. Agreement with
  ``math.sin`` corroborates that the library is not wildly wrong; it certifies
  nothing, and it is not evidence for any bound. Only the exact ``Fraction``
  assertions carry weight.
* The decimal constants in group 2 are typed in from independent knowledge, not
  computed by the library under test. They are treated as *truncations* (never
  roundings) of the true expansions, so the true value lies in
  ``[literal, literal + 1ulp]``; the tests assert that whole bracket lies inside
  the computed enclosure, which is what "the true value is inside" means here.
  That requires the enclosure to be WIDER than the literal's last place, so the
  ``prec`` values used in those tests are chosen accordingly. Where an
  enclosure is deliberately tighter than the literal (the normal-pdf peak), the
  assertion is stated as a two-sided inequality against the literal instead,
  and says so. If a literal were mistyped the test would fail loudly rather
  than pass silently.
* No test here promotes, closes or reclassifies anything, and no test result
  may be cited as evidence for any mathematical claim in this program.
"""
import math
import os
import random
import sys
from fractions import Fraction as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pytest  # noqa: E402

from research.interval import (  # noqa: E402
    EXP_BIT_LIMIT, Interval, Phi, cos, erf, erfc, exp, log, normal_pdf,
    normal_sf, pi, sin, sqrt,
)
from research.interval import transcendental as T  # noqa: E402


# ---------------------------------------------------------------------------
# Independently-sourced decimal constants.
#
# Each is a TRUNCATION of the true decimal expansion, typed in from knowledge
# and NOT produced by the library under test. Truncation (not rounding) is what
# makes ``true value in [literal, literal + 1ulp]`` valid.
# ---------------------------------------------------------------------------
PI_LIT = "3.1415926535897932384626433832795028841971"
E_LIT = "2.7182818284590452353602874713526624977572"
LN2_LIT = "0.6931471805599453094172321214581765680755"
SQRT2_LIT = "1.4142135623730950488016887242096980785696"
ERF1_LIT = "0.8427007929497148"
PHI1_LIT = "0.8413447460685429"
INV_SQRT_2PI_LIT = "0.3989422804014326"   # 1/sqrt(2*pi), the normal-pdf peak


def bracket(literal: str) -> Interval:
    """``[literal, literal + 1ulp]`` — provably contains the true value.

    Valid because every literal above is a truncation of the true expansion, so
    the true value is at least the literal and less than the literal plus one
    unit in its last decimal place.
    """
    lo = F(literal)
    places = len(literal.split(".")[1])
    return Interval(lo, lo + F(1, 10 ** places))


def assert_encloses(iv: Interval, literal: str, max_width: F) -> None:
    """The enclosure contains the true value, and is actually tight."""
    brk = bracket(literal)
    assert brk in iv, f"enclosure {iv!r} does not contain {literal}"
    assert iv.width() <= max_width, f"enclosure {iv!r} is wider than {max_width}"


# ===========================================================================
# 1. Interval algebra
# ===========================================================================

def test_construction_and_invariant():
    assert Interval(1).lo == Interval(1).hi == F(1)
    assert Interval("1/3", "1/2").lo == F(1, 3)
    assert Interval("0.1").lo == F(1, 10)          # exact decimal, not a float
    assert Interval.exact(7) == Interval(7, 7)
    with pytest.raises(ValueError):
        Interval(2, 1)


def test_float_endpoints_are_refused():
    """Fraction(0.1) is not 1/10; silently accepting it is the bug this
    package exists to exclude."""
    with pytest.raises(TypeError):
        Interval(0.1)
    with pytest.raises(TypeError):
        Interval.exact(1.5)


def test_arithmetic_is_exact_and_outward():
    a, b = Interval(1, 2), Interval("1/2", 3)
    assert a + b == Interval(F(3, 2), 5)
    assert a - b == Interval(-2, F(3, 2))
    assert a * b == Interval(F(1, 2), 6)
    assert -a == Interval(-2, -1)
    assert Interval(-1, 2) * Interval(-3, 4) == Interval(-6, 8)
    assert a / Interval(2, 4) == Interval(F(1, 4), 1)


def test_division_by_an_interval_containing_zero_raises():
    with pytest.raises(ZeroDivisionError):
        Interval(1, 2) / Interval(-1, 1)
    with pytest.raises(ZeroDivisionError):
        Interval(1, 2) / Interval(0, 1)
    with pytest.raises(ZeroDivisionError):
        Interval(-1, 1) ** -2


def test_width_mid_mag_mig_hull_intersect():
    iv = Interval(-2, 6)
    assert iv.width() == 8
    assert iv.mid() == 2
    assert iv.mag() == 6
    assert iv.mig() == 0          # contains zero
    assert Interval(2, 6).mig() == 2
    assert Interval(-6, -2).mig() == 2
    assert Interval(-6, -2).mag() == 6
    assert iv.contains_zero()
    assert not Interval(1, 2).contains_zero()
    assert Interval(0, 1).hull(Interval(5, 7)) == Interval(0, 7)
    assert Interval(0, 5).intersect(Interval(3, 9)) == Interval(3, 5)
    assert Interval(0, 1).intersect(Interval(3, 9)) is None


def test_membership():
    iv = Interval(-1, 2)
    assert F(1, 3) in iv
    assert 2 in iv
    assert 3 not in iv
    assert Interval(0, 1) in iv
    assert Interval(0, 3) not in iv


def test_repr_shows_exact_fractions_and_a_decimal_display():
    r = repr(Interval(F(1, 3), F(1, 2)))
    assert "1/3" in r and "1/2" in r
    assert "0.333333333333" in r and "0.500000000000" in r


# --- the non-monotone traps ------------------------------------------------

def test_even_power_over_an_interval_straddling_zero():
    """The classic place interval libraries are silently wrong.

    ``x -> x**2`` has an interior minimum at 0, so endpoint evaluation alone
    gives ``[1, 4]`` (or the non-interval ``[4, 1]``) and loses the true value
    ``0`` attained at ``x = 0``.
    """
    assert Interval(-2, 1) ** 2 == Interval(0, 4)
    assert Interval(-2, 1) ** 4 == Interval(0, 16)
    assert 0 in Interval(-2, 1) ** 2


def test_powers_away_from_zero_and_odd_powers():
    assert Interval(1, 2) ** 2 == Interval(1, 4)
    assert Interval(-2, -1) ** 2 == Interval(1, 4)
    assert Interval(-2, 1) ** 3 == Interval(-8, 1)
    assert Interval(-2, 1) ** 0 == Interval(1, 1)
    assert Interval(2, 4) ** -1 == Interval(F(1, 4), F(1, 2))
    assert Interval(-4, -2) ** -2 == Interval(F(1, 16), F(1, 4))


def test_cos_over_an_interval_straddling_zero_attains_one():
    c = cos(Interval("-1/10", "1/10"), 30)
    assert c.hi == 1
    assert c.lo < 1


def test_sin_over_an_interval_containing_half_pi_attains_one():
    s = sin(Interval("3/2", "17/10"), 30)     # pi/2 = 1.5707... is inside
    assert s.hi == 1
    assert s.lo < 1


def test_sin_wider_than_a_period_is_the_full_unit_interval():
    assert sin(Interval(0, 100), 20) == Interval(-1, 1)
    assert cos(Interval(0, 100), 20) == Interval(-1, 1)


def test_normal_pdf_over_an_interval_straddling_zero_attains_its_peak():
    """The maximum is at the interior point 0, not at either endpoint.

    The enclosure is tighter than the last place of the typed literal, so the
    assertion is stated the other way round: ``p.hi`` must be at least the
    literal (a certified lower bound for 1/sqrt(2*pi), so the enclosure really
    does reach the peak) and at most one unit in the literal's last place above
    it (so it is a tight upper bound and not a vacuous one).
    """
    p = normal_pdf(Interval(-2, 1), 25)
    peak_lo = F(INV_SQRT_2PI_LIT)
    assert p.hi >= peak_lo
    assert p.hi <= peak_lo + F(1, 10 ** 16)
    # the density at the ends of the interval is far below the peak
    assert p.lo < F(6, 100)
    # and the peak is strictly interior: neither endpoint attains it
    assert normal_pdf(Interval.exact(-2), 25).hi < F(6, 100)
    assert normal_pdf(Interval.exact(1), 25).hi < F(25, 100)


# ===========================================================================
# 2. Containment of known constants
# ===========================================================================

def test_pi_encloses_the_known_expansion():
    assert_encloses(pi(25), PI_LIT, F(1, 10 ** 20))


def test_e_encloses_the_known_expansion():
    assert_encloses(exp(Interval.exact(1), 25), E_LIT, F(1, 10 ** 20))


def test_log2_encloses_the_known_expansion():
    assert_encloses(log(Interval.exact(2), 25), LN2_LIT, F(1, 10 ** 20))


def test_sqrt2_encloses_the_known_expansion():
    assert_encloses(sqrt(Interval.exact(2), 25), SQRT2_LIT, F(1, 10 ** 20))


def test_erf_one_encloses_the_known_value():
    assert_encloses(erf(Interval.exact(1), 10), ERF1_LIT, F(1, 10 ** 9))


def test_Phi_one_encloses_the_known_value():
    assert_encloses(Phi(Interval.exact(1), 10), PHI1_LIT, F(1, 10 ** 9))


def test_Phi_zero_is_exactly_one_half():
    assert Phi(Interval.exact(0), 25) == Interval(F(1, 2), F(1, 2))


def test_sqrt_certificate_inequalities_hold_on_the_returned_endpoints():
    """The certificate itself, re-checked from outside the library."""
    for a in (F(2), F(3), F(1, 7), F(10 ** 9), F(1, 10 ** 9), F(0)):
        s = sqrt(Interval.exact(a), 30)
        assert s.lo * s.lo <= a
        assert s.hi * s.hi >= a


def test_log_and_exp_are_mutually_consistent():
    """log(exp(x)) must contain x. A cross-check between two independent
    certified constructions, not a certification of either."""
    for v in (F(1, 3), F(-2), F(5), F(-1, 7)):
        assert v in log(exp(Interval.exact(v), 30), 25)


def test_pythagorean_identity_is_contained():
    """sin^2 + cos^2 must contain 1 for every argument."""
    for v in (F(0), F(1, 3), F(-7, 2), F(100), F(-1000, 7)):
        x = Interval.exact(v)
        assert 1 in (sin(x, 25) ** 2 + cos(x, 25) ** 2)


def test_addition_formula_is_consistent_at_a_large_argument():
    """A discriminating check on the ARGUMENT REDUCTION, not on the series.

    ``sin(a+b) = sin a cos b + cos a sin b`` holds exactly, so two enclosures
    that both contain the true value must intersect.

    Reducing against a fixed rational ``p`` in place of ``pi`` returns, in
    effect, ``sin(x + j(x) * eps)`` where ``j(x)`` is the quadrant index and
    ``eps = pi/2 - p``. Those errors CANCEL in the addition formula whenever
    ``j(a+b) = j(a) + j(b)``, which is the common case — so the offset matters.
    ``b = 4/5`` is chosen because ``j(100000) = 63662``, ``j(4/5) = 1`` and
    ``j(100000.8) = 63662``, so the indices do NOT add and the errors do not
    cancel. Verified: with a hard-coded 14- or 20-digit rational for ``pi`` the
    two sides become disjoint, while the certified enclosure keeps them
    consistent.

    Honest limitation, established by running the mutants: this test does NOT
    catch a point value of ``pi`` taken at the *adaptive* precision that
    ``_pi_for`` selects (midpoint of a sufficiently tight enclosure). That
    variant is unjustified but its error stays below the returned width at
    these magnitudes, so no output test can see it; it is caught structurally
    by ``test_reduction_uncertainty_is_carried_as_width_not_discarded``.
    """
    a, b, p = F(100000), F(4, 5), 30
    lhs = sin(Interval.exact(a + b), p)
    rhs = (sin(Interval.exact(a), p) * cos(Interval.exact(b), p)
           + cos(Interval.exact(a), p) * sin(Interval.exact(b), p))
    assert lhs.intersect(rhs) is not None

    lhs_c = cos(Interval.exact(a + b), p)
    rhs_c = (cos(Interval.exact(a), p) * cos(Interval.exact(b), p)
             - sin(Interval.exact(a), p) * sin(Interval.exact(b), p))
    assert lhs_c.intersect(rhs_c) is not None


def test_reduction_uncertainty_is_carried_as_width_not_discarded():
    """The reduced argument must be an INTERVAL, not a point.

    ``pi`` is irrational, so a certified reduction of a nonzero rational
    argument can never produce a degenerate reduced interval. If it does, the
    uncertainty in ``pi`` was silently dropped somewhere.
    """
    p = 25
    half_pi = T._pi_for(Interval.exact(F(100000)), p) * F(1, 2)
    assert half_pi.width() > 0, "pi enclosure collapsed to a point"
    s = Interval.exact(F(100000)) - Interval.exact(63662) * half_pi
    assert s.width() > 0, "reduced argument collapsed to a point"


def test_erf_is_odd_and_Phi_is_symmetric():
    for v in (F(1, 4), F(1), F(3), F(7)):
        a = erf(Interval.exact(v), 20)
        b = erf(Interval.exact(-v), 20)
        assert (-b.lo) in a and (-b.hi) in a
        assert 1 in (Phi(Interval.exact(v), 20) + Phi(Interval.exact(-v), 20))


def test_erf_tail_branch_is_a_valid_bracket():
    """Above the documented crossover, erf uses the monotone tail bracket.

    This asserts the property, not a particular tightness. An earlier version
    pinned ``e.hi == 1``, which described the crude clamp the tail branch used
    at the time; tightening the branch then failed a test that was only ever
    recording the implementation. What must hold is that the bracket contains
    the true value and does not exceed 1, since ``erf(x) < 1`` for every finite
    ``x``. A tighter bracket must pass this test, and a wrong one must not.
    """
    e = erf(Interval.exact(8), 30)

    # erfc(8), typed from an independent source, NOT computed by this library.
    ERFC_8 = F(11224297172982928, 10 ** 45)          # 1.1224297172982928e-29
    true_erf_8 = 1 - ERFC_8

    assert true_erf_8 in e                            # containment, the contract
    assert e.hi <= 1                                  # erf(x) < 1 for finite x
    assert e.lo < e.hi                                # a bracket, not a point
    assert e.lo > F(9999999, 10000000)

    # Corroboration only, never certification: agreement with the float erfc.
    assert (1 - e.hi) <= F(math.erfc(8)) <= (1 - e.lo)


def test_erf_tail_bracket_rejects_a_widened_but_wrong_bound():
    """Negative control for the test above.

    A bracket that is merely wide is not thereby correct. Shift the enclosure
    off the true value while keeping it wide and containment must fail.
    """
    e = erf(Interval.exact(8), 30)
    ERFC_8 = F(11224297172982928, 10 ** 45)
    true_erf_8 = 1 - ERFC_8

    shifted = Interval(e.lo - ERFC_8 * 10, e.hi - ERFC_8 * 10)
    assert shifted.hi - shifted.lo == e.hi - e.lo      # same width
    assert true_erf_8 not in shifted                   # and yet wrong


# ===========================================================================
# 3. Tightness monotonicity in prec
# ===========================================================================

_MONOTONICITY_SAMPLE = [
    (sqrt, Interval(2, 3)),
    (sqrt, Interval("1/7", "1/3")),
    (exp, Interval(-2, 1)),
    (exp, Interval("1/5", "1/5")),
    (log, Interval(2, 10)),
    (log, Interval("1/1000", "1/999")),
    (sin, Interval("1/10", "3/10")),
    (cos, Interval(2, 3)),
    (erf, Interval("1/2", 1)),
    (erf, Interval(2, 3)),
    (Phi, Interval(-1, 1)),
    (normal_pdf, Interval("1/2", 2)),
]


@pytest.mark.parametrize("fn,x", _MONOTONICITY_SAMPLE,
                         ids=[f"{f.__name__}-{i}" for i, (f, _) in
                              enumerate(_MONOTONICITY_SAMPLE)])
def test_tightness_improves_with_prec(fn, x):
    wide = fn(x, 20).width()
    tight = fn(x, 40).width()
    assert tight <= wide, f"{fn.__name__}: width({40}) > width({20})"


def test_pi_tightness_improves_with_prec():
    assert pi(40).width() <= pi(20).width()
    assert pi(20).width() <= pi(10).width()


def test_containment_holds_at_absurdly_low_prec():
    """Containment is unconditional: prec=1 must still be correct, just wide."""
    assert bracket(PI_LIT) in pi(1)
    assert bracket(E_LIT) in exp(Interval.exact(1), 1)
    assert bracket(SQRT2_LIT) in sqrt(Interval.exact(2), 1)
    assert bracket(LN2_LIT) in log(Interval.exact(2), 1)
    assert bracket(ERF1_LIT) in erf(Interval.exact(1), 1)


# ===========================================================================
# 4. NEGATIVE CONTROLS
#
# Each builds a deliberately weakened variant here, in the test file, and
# asserts that containment FAILS. None of them edits the library. If one of
# these tests starts passing trivially (i.e. the broken variant still
# contains the true value), the corresponding safeguard has stopped mattering
# and the test must be re-examined, not deleted.
# ===========================================================================

def _certified_reference(fn, x, prec=45):
    """A certified enclosure from the (unbroken) library, used as the oracle
    against which a broken variant is shown to be disjoint."""
    return fn(x, prec)


# --- NC1: sqrt endpoints rounded INWARD ------------------------------------

def _broken_sqrt_inward(a: F, scale: int) -> Interval:
    """``sqrt`` with the upper endpoint rounded INWARD (down) instead of
    outward (up) — the single character change ``s + 1`` -> ``s``."""
    p, q = a.numerator, a.denominator
    s = T._isqrt_newton(p * q * scale * scale)
    return Interval(F(s, q * scale), F(s, q * scale))


def test_negative_control_sqrt_rounded_inward_loses_containment():
    a = F(2)
    broken = _broken_sqrt_inward(a, 10 ** 5)
    good = sqrt(Interval.exact(a), 30)

    # The broken enclosure does not contain sqrt(2) ...
    assert broken.intersect(good) is None
    assert bracket(SQRT2_LIT) not in broken
    # ... and the certificate the library keeps would have caught it:
    assert not broken.hi * broken.hi >= a
    # while the real implementation passes that same check.
    assert good.hi * good.hi >= a and good.lo * good.lo <= a


# --- NC2: exp Taylor truncated one term early, without widening -------------

def _broken_exp_one_term_early(t: F, prec: int) -> Interval:
    """The Taylor sum stops one term early but keeps the remainder bound that
    was derived for the full sum. Classic off-by-one; the dropped term is
    larger than the remainder bound it is supposedly covered by."""
    k, u = 0, t
    while abs(u) > F(1, 2):
        u /= 2
        k += 1
    target = F(1, 10 ** prec) / 2 ** k
    au = abs(u)
    terms, term, absterm, j = [], F(1), F(1), 0
    while True:
        terms.append(term)
        nxt_abs = absterm * au / (j + 1)
        rem = nxt_abs / (1 - au / (j + 2))
        if rem <= target:
            break
        j += 1
        term = term * u / j
        absterm = nxt_abs
    total = sum(terms[:-1], F(0))              # <-- one term dropped
    enc = Interval(total - rem, total + rem)
    for _ in range(k):
        enc = enc ** 2
    return enc


def test_negative_control_exp_truncated_early_loses_containment():
    broken = _broken_exp_one_term_early(F(1), 20)
    good = exp(Interval.exact(1), 25)
    assert broken.intersect(good) is None
    assert bracket(E_LIT) not in broken
    assert bracket(E_LIT) in good


# --- NC3: log with the series tail bound dropped ----------------------------

def _broken_log2_no_tail(target: F) -> Interval:
    """``log 2 = 2 atanh(1/3)`` with the positive tail simply omitted. Every
    term of the atanh series is positive, so the partial sum is a strict
    under-estimate and the "enclosure" sits entirely below log 2."""
    z, zz = F(1, 3), F(1, 9)
    total, term, j = F(0), F(1, 3), 0
    while True:
        total += term / (2 * j + 1)
        nxt = term * zz
        tail = nxt / ((2 * j + 3) * (1 - zz))
        if tail <= target:
            break
        j += 1
        term = nxt
    return Interval(2 * total, 2 * total)       # <-- tail dropped


def test_negative_control_log_without_tail_bound_loses_containment():
    broken = _broken_log2_no_tail(F(1, 10 ** 20))
    good = log(Interval.exact(2), 25)
    assert broken.hi < good.lo                  # strictly below log 2
    assert bracket(LN2_LIT) not in broken
    assert bracket(LN2_LIT) in good


# --- NC4: Machin remainder taken from the wrong index -----------------------

def _broken_atan_recip_off_by_one(n: int, target: F) -> Interval:
    """The alternating-series error after the j-th term is bounded by the
    FIRST omitted term. This variant uses the SECOND omitted term instead —
    about ``n^2`` times too small."""
    nn = n * n
    total, powv, j = F(0), F(1, n), 0
    while True:
        term = powv / (2 * j + 1)
        total = total + term if j % 2 == 0 else total - term
        nxt_pow = powv / nn
        nxt = nxt_pow / (2 * j + 3)
        if nxt <= target:
            break
        j += 1
        powv = nxt_pow
    second_omitted = (nxt_pow / nn) / (2 * j + 5)   # <-- wrong index
    return Interval(total - second_omitted, total + second_omitted)


def _broken_pi_off_by_one(prec: int) -> Interval:
    target = F(1, 10 ** prec) / 32
    a5 = _broken_atan_recip_off_by_one(5, target)
    a239 = _broken_atan_recip_off_by_one(239, target)
    return Interval.exact(4) * (Interval.exact(4) * a5 - a239)


def test_negative_control_machin_remainder_off_by_one_loses_containment():
    broken = _broken_pi_off_by_one(12)
    good = pi(25)
    assert broken.intersect(good) is None
    assert bracket(PI_LIT) not in broken
    assert bracket(PI_LIT) in good


# --- NC5: sin/cos argument reduction with a POINT value of pi ---------------

_PI_AS_A_POINT = F("3.14159265")     # 9 correct digits, used as if exact


def test_negative_control_point_pi_reduction_loses_containment():
    """Reducing a large argument against a fixed rational ``pi`` accumulates
    ``j * (pi_error)``; the certified reduction carries that error as interval
    width instead. Here ``j`` is about 63662, so the point reduction is wrong
    by about 1e-4 while claiming a width of about 1e-30."""
    v = F(100000)
    prec = 30
    target = F(1, 10 ** prec) / 4
    sig = 4 * prec + 128

    broken_sin, broken_cos = T._sin_cos_point(
        v, Interval.exact(_PI_AS_A_POINT / 2), target, sig)
    good_sin = sin(Interval.exact(v), prec)
    good_cos = cos(Interval.exact(v), prec)

    assert broken_sin.intersect(good_sin) is None
    assert broken_cos.intersect(good_cos) is None
    # the point reduction even claims to be tight while being wrong
    assert broken_sin.width() < F(1, 10 ** 25)
    assert abs(broken_sin.mid() - good_sin.mid()) > F(1, 10 ** 5)


# --- NC6: x**2 without the interior extremum --------------------------------

def _broken_square_endpoints_only(x: Interval) -> Interval:
    """``x ** 2`` evaluated at the endpoints only, ignoring the interior
    minimum at 0."""
    a, b = x.lo ** 2, x.hi ** 2
    return Interval(min(a, b), max(a, b))


def test_negative_control_square_without_interior_extremum():
    x = Interval(-2, 1)
    broken = _broken_square_endpoints_only(x)
    assert broken == Interval(1, 4)
    assert 0 in x                      # x = 0 is in the domain ...
    assert 0 not in broken             # ... but 0 = 0**2 is not in the range
    assert 0 in x ** 2                 # the library gets it right


def test_negative_control_normal_pdf_built_on_the_broken_square():
    """The same defect one level up: the density's peak at 0 disappears."""
    x = Interval(-2, 1)
    g = 25 + 20
    broken = (exp(-_broken_square_endpoints_only(x) * F(1, 2), 25)
              / sqrt(Interval.exact(2) * pi(g), g))
    good = normal_pdf(x, 25)

    peak_lo = F(INV_SQRT_2PI_LIT)      # certified lower bound for 1/sqrt(2*pi)
    assert broken.hi < peak_lo         # the true maximum is missed entirely
    assert good.hi >= peak_lo          # the library reaches it
    assert broken.hi < good.hi


# --- NC7: sin/cos hull of endpoints only, interior extrema ignored ----------

def _broken_cos_endpoints_only(x: Interval, prec: int) -> Interval:
    p = T._pi_for(x, prec)
    half_pi = p * F(1, 2)
    target = F(1, 10 ** prec) / 4
    sig = 4 * prec + 128
    _, c_lo = T._sin_cos_point(x.lo, half_pi, target, sig)
    _, c_hi = T._sin_cos_point(x.hi, half_pi, target, sig)
    return c_lo.hull(c_hi)              # <-- interior critical points ignored


def test_negative_control_cos_ignoring_interior_extrema():
    x = Interval("-1/10", "1/10")
    broken = _broken_cos_endpoints_only(x, 30)
    good = cos(x, 30)
    assert 1 not in broken              # cos(0) = 1 is missed
    assert 1 in good
    assert broken.hi < 1


# --- NC8: erf alternating bracket taken one-sided ---------------------------

def _broken_erf_one_sided(z: F, stop_at: int, prec: int) -> Interval:
    """The alternating remainder bracket ``[S - t, S + t]`` replaced by the
    one-sided ``[S, S + t]``. Stopping on an EVEN index leaves the partial sum
    above the true sum, so the true value falls out of the bracket."""
    assert stop_at % 2 == 0
    zz = z * z
    total, t, n = F(0), z, 0
    while True:
        total = total + t if n % 2 == 0 else total - t
        t_next = t * zz * (2 * n + 1) / ((n + 1) * (2 * n + 3))
        if n == stop_at:
            break
        n += 1
        t = t_next
    g = prec + 20
    series = Interval(total, total + t_next)    # <-- lower half dropped
    return series * (Interval.exact(2) / sqrt(pi(g), g))


def test_negative_control_erf_one_sided_bracket_loses_containment():
    broken = _broken_erf_one_sided(F(1), 2, 30)
    good = erf(Interval.exact(1), 10)
    assert broken.intersect(good) is None
    assert bracket(ERF1_LIT) not in broken
    assert bracket(ERF1_LIT) in good


# ===========================================================================
# 5. NON-CERTIFYING float cross-check sweep
#
# Seeded and deterministic. Float agreement is CORROBORATION ONLY. It is not a
# certification, it is not a bound, and no result in this program may cite it
# as evidence for anything. The exact Fraction assertions above are what carry
# weight; this sweep exists to catch gross implementation mistakes early.
# ===========================================================================

SWEEP_SEED = 20260918           # fixed literal: the sweep is reproducible
SWEEP_SIZE = 40
SWEEP_PREC = 20


def _float_agrees(enc: Interval, reference: float) -> bool:
    """NON-CERTIFYING. Generous tolerance; the float is the suspect party."""
    slack = 1e-11 + 1e-9 * abs(reference)
    return float(enc.lo) - slack <= reference <= float(enc.hi) + slack


def test_float_cross_check_sweep_NONCERTIFYING():
    rng = random.Random(SWEEP_SEED)
    checked = 0
    for _ in range(SWEEP_SIZE):
        big = F(rng.randint(-2000, 2000), rng.randint(1, 300))
        small = F(rng.randint(-40, 40), rng.randint(1, 20))
        mid = F(rng.randint(-800, 800), rng.randint(1, 100))

        x_big, x_small, x_mid = (Interval.exact(big), Interval.exact(small),
                                 Interval.exact(mid))

        assert _float_agrees(sin(x_big, SWEEP_PREC), math.sin(float(big)))
        assert _float_agrees(cos(x_big, SWEEP_PREC), math.cos(float(big)))
        assert _float_agrees(exp(x_small, SWEEP_PREC), math.exp(float(small)))
        assert _float_agrees(erf(x_small, SWEEP_PREC), math.erf(float(small)))
        assert _float_agrees(
            Phi(x_small, SWEEP_PREC),
            0.5 * (1.0 + math.erf(float(small) / math.sqrt(2.0))))
        assert _float_agrees(
            normal_pdf(x_small, SWEEP_PREC),
            math.exp(-float(small) ** 2 / 2.0) / math.sqrt(2.0 * math.pi))
        if mid > 0:
            assert _float_agrees(sqrt(x_mid, SWEEP_PREC), math.sqrt(float(mid)))
            assert _float_agrees(log(x_mid, SWEEP_PREC), math.log(float(mid)))
        checked += 1

    assert checked == SWEEP_SIZE


def test_float_cross_check_sweep_over_nondegenerate_intervals_NONCERTIFYING():
    """Same NON-CERTIFYING corroboration, but over genuine (wide) intervals:
    sampled interior points must land inside the enclosure."""
    rng = random.Random(SWEEP_SEED + 1)
    for _ in range(20):
        lo = F(rng.randint(-300, 300), rng.randint(1, 50))
        hi = lo + F(rng.randint(1, 40), rng.randint(1, 20))
        x = Interval(lo, hi)
        s, c = sin(x, 15), cos(x, 15)
        for _ in range(5):
            t = lo + (hi - lo) * F(rng.randint(0, 1000), 1000)
            assert _float_agrees(s, math.sin(float(t)))
            assert _float_agrees(c, math.cos(float(t)))


# ===========================================================================
# 6. AUDIT REGRESSIONS (2026-09-18)
#
# Four adversarial audits attacked this library. None found a containment
# violation. They did find one class of hard failure (unbounded memory, and
# ValueError from int->str in displays and in one bound path), one class of
# tightness failure (the Gaussian tail had no positive certified lower bound at
# any prec), and three coverage gaps where a genuine containment-breaking
# mutation survived the entire suite.
#
# Every test below was FIRST run against the unfixed library, or against a
# deliberately broken copy, and confirmed to fail there. A test that cannot
# fail is not a test.
#
# NOTHING IN THIS GROUP PROMOTES, CLOSES, DISCHARGES OR RECLASSIFIES ANYTHING.
# ===========================================================================

# --- R1: repr must never raise ----------------------------------------------

def test_repr_of_large_endpoints_does_not_raise():
    """CPython 3.11 refuses ``int`` -> ``str`` above 4300 digits.

    ``erf(Interval(100), 20)`` legitimately returns an endpoint with a
    14,721-bit denominator (about 4,431 decimal digits), and the unfixed
    ``__repr__`` interpolated the endpoints with ``{self.lo!s}``, so
    ``repr()`` of that interval raised
    ``ValueError: Exceeds the limit (4300 digits)``.

    Fires on: restoring ``f"Interval({self.lo!s}, {self.hi!s})"``.
    """
    r = repr(erf(Interval(100), 20))
    assert r.startswith("Interval(")
    assert "bit numerator" in r or "bit denominator" in r
    # the outward decimal display still carries the value
    assert "~[" in r
    # and a plain small interval is displayed exactly as before
    assert repr(Interval(1, 2)).startswith("Interval(1, 2)")


def test_guard_messages_survive_endpoints_too_large_to_print():
    """The contracted exception, not a ValueError about digit limits.

    Every guard in this package formats an interval or a Fraction into its
    message. On the unfixed code each of these three raised
    ``ValueError: Exceeds the limit (4300 digits)`` instead of the exception the
    module docstring promises, so ``except ZeroDivisionError`` around a
    certified computation did not catch the failure and the message did not say
    what failed.

    Fires on: restoring the raw ``{lo}``/``{self.lo!s}`` interpolations in
    ``core.Interval.__init__``, ``__truediv__``, ``__pow__`` and ``__repr__``.
    """
    tiny = Interval(F(-1, 10 ** 5000), F(1, 10 ** 5000))
    with pytest.raises(ZeroDivisionError):
        Interval(1, 2) / tiny
    with pytest.raises(ZeroDivisionError):
        tiny ** -2
    with pytest.raises(ValueError, match="empty interval"):
        Interval(F(10 ** 5000 + 1, 10 ** 5000), F(1))


def test_outward_decimal_display_stays_outward_when_it_falls_back():
    """The power-of-ten fallback must still bound the endpoints outwardly.

    A display that claimed a TIGHTER interval than the endpoints support would
    be a lie in the one place a reader looks first.
    """
    from research.interval.core import _decimal_string

    for x in (F(10) ** 5000, -(F(10) ** 5000), F(10) ** 5001 + 7,
              F(1, 10 ** 5000), -F(1, 10 ** 5000)):
        down = _decimal_string(x, 12, upward=False)
        up = _decimal_string(x, 12, upward=True)
        assert F(down) <= x <= F(up), (x, down, up)


# --- R2: sin/cos of an argument with more digits than str() will render ------

def test_sin_of_an_argument_beyond_the_int_to_str_limit():
    """``_pi_for`` chose the certified ``pi`` precision with ``len(str(v))``.

    That is a decimal-digit count taken by materialising the integer, so any
    ``sin``/``cos`` of an argument with at least ``sys.get_int_max_str_digits()``
    digits raised ``ValueError`` in the step that decides how much ``pi``
    precision the reduction needs — before any arithmetic happened. Measured
    threshold on the unfixed code: ``10**4299`` worked, ``10**4300`` raised.

    The limit is lowered here rather than using a 4300-digit argument so the
    test costs milliseconds instead of two seconds; the failure mode is
    identical, since it is the interpreter limit that is hit either way.

    Fires on: restoring ``len(str(v))`` in ``_pi_for``.
    """
    old = sys.get_int_max_str_digits()
    try:
        sys.set_int_max_str_digits(640)          # 640 is the interpreter minimum
        v = 10 ** 700                            # 701 digits > 640
        s = sin(Interval(v), 5)
        c = cos(Interval(v), 5)
    finally:
        sys.set_int_max_str_digits(old)
    assert s.lo >= -1 and s.hi <= 1
    assert c.lo >= -1 and c.hi <= 1
    assert s.width() < F(1, 10 ** 4)
    # and the helper agrees with the exact count it replaces, where str works
    for n in (0, 1, 9, 10, 99, 100, 12345, 2 ** 64, 10 ** 300):
        assert T._decimal_digits(n) >= len(str(abs(n)))
        assert T._decimal_digits(n) <= len(str(abs(n))) + 1


# --- R3: the exp exponent cap ------------------------------------------------

def test_exp_underflow_is_capped_and_still_contains_the_true_value():
    """Unbounded endpoint growth in ``_exp_point``, reachable from the API.

    An enclosure of ``exp(t)`` carries about ``1.4427 |t|`` bits of scale, which
    ``round_out`` cannot remove. Measured on the unfixed code:
    ``exp(Interval.exact(-10**7), 20)`` returned 14,427,176-bit endpoints, and
    through ``Phi``/``normal_pdf``/``erf`` the growth is quadratic in the
    argument: ``Phi(Interval(-10**5), 1)`` raised ``MemoryError`` under a
    512 MiB address-space cap after 14 s and was OOM-killed without one.

    Past ``EXP_BIT_LIMIT`` the enclosure degrades to ``[0, 2**-EXP_BIT_LIMIT]``,
    which CONTAINS ``exp(t)`` because ``0 < exp(t) <= 2**-EXP_BIT_LIMIT`` there.

    Fires on: removing the underflow branch (the endpoint-size assertion fails
    in about half a second at this argument).
    """
    r = exp(Interval.exact(-10 ** 7), 20)
    assert r.lo == 0
    assert r.hi == F(1, 1 << EXP_BIT_LIMIT)
    assert r.hi.denominator.bit_length() == EXP_BIT_LIMIT + 1
    # below the cap nothing changed: still a genuine two-sided enclosure
    small = exp(Interval.exact(-1000), 20)
    assert small.lo > 0 and small.hi > small.lo
    assert small.hi < F(1, 10 ** 434)


def test_exp_overflow_refuses_instead_of_exhausting_memory():
    """An exact rational upper bound on ``exp(10**8)`` needs about 17 MB, on
    ``exp(10**12)`` about 180 GB. There is no coarse honest answer, so this is
    refused with a typed error naming the limit. ``OverflowError`` is an
    ``ArithmeticError``, so ``except ArithmeticError`` around a certified
    computation catches it.

    Fires on: removing the overflow branch (the call then takes minutes and
    allocates without bound rather than raising).
    """
    with pytest.raises(OverflowError, match="EXP_BIT_LIMIT"):
        exp(Interval.exact(10 ** 8), 20)
    with pytest.raises(ArithmeticError):
        exp(Interval.exact(10 ** 8), 20)
    assert exp(Interval.exact(1000), 5).lo > 0        # below the cap, unaffected


def test_the_gaussian_api_terminates_at_large_arguments():
    """``erf``, ``Phi``, ``normal_pdf`` and ``normal_sf`` inherited the blow-up.

    Measured on the unfixed code at ``prec = 5``: ``erf(Interval(10**4))`` 15.8 s
    with a 144,269,749-bit endpoint, ``normal_pdf(Interval(10**4))`` 3.5 s,
    ``Phi(Interval(-10**5), 1)`` MemoryError. All four calls below now return in
    well under a second with endpoints bounded by the cap.

    Fires on: removing the exp cap (this test then does not terminate).
    """
    budget = EXP_BIT_LIMIT + 4 * 25 + 4096          # cap + significand + slack
    for iv in (erf(Interval(10 ** 5), 5),
               erfc(Interval(10 ** 5), 5),
               Phi(Interval(-10 ** 6), 5),
               normal_sf(Interval(10 ** 6), 5),
               normal_pdf(Interval(10 ** 6), 5)):
        for e in (iv.lo, iv.hi):
            assert e.numerator.bit_length() <= budget
            assert e.denominator.bit_length() <= budget
    # and the values are still sound where they are informative
    assert normal_pdf(Interval(10 ** 6), 5).lo == 0
    assert 0 <= Phi(Interval(-10 ** 6), 5).lo
    assert Phi(Interval(-10 ** 6), 5).hi < F(1, 10 ** 100)


# --- R4: the Gaussian tail now has a positive certified lower bound ----------

def test_gaussian_tail_has_a_positive_certified_lower_bound():
    """The defect four auditors independently reported as the most consequential.

    The old tail branch returned the ONE-SIDED bracket ``erf(z) in [1 - U, 1]``,
    so by oddness ``erf(-z).lo`` was exactly ``-1`` and ``Phi(x).lo`` was exactly
    ``0`` for every ``x <= -6 sqrt 2 = -8.4853``, at EVERY prec: relative width
    1, no improvement with prec, and ``log(Phi(Interval(-9), 30), 30)`` raised
    ``ValueError: log requires x.lo > 0``. The library could not bound a
    Gaussian tail probability from below at all.

    The Mills bracket proved in ``_erfc_mills`` fixes it. Its lower half,
    ``erfc(z) > 2 z e^(-z^2) / (sqrt(pi) (2 z^2 + 1))``, is elementary: see the
    lemma in that docstring.

    Fires on: restoring ``return _clamp_unit(Interval(1 - upper, Fraction(1)))``.
    """
    for prec in (10, 30, 60, 120):
        assert Phi(Interval(-9), prec).lo > 0
        assert erf(Interval(-7), prec).lo > -1
        assert erf(Interval(7), prec).hi < 1
        assert normal_sf(Interval(9), prec).lo > 0
    # the prec-dependent collapse below the crossover is gone too
    assert Phi(Interval(-8), 10).lo > 0
    # and the log-tail bound a consumer actually wants now exists
    lg = log(normal_sf(Interval.exact(9), 30), 30)
    assert lg.hi < -43 and lg.lo > -44           # log(1.1286e-19) = -43.6
    # far out, where the exp cap bites, the lower bound honestly degrades to 0
    assert normal_sf(Interval(10 ** 4), 5).lo == 0


def test_gaussian_tail_bracket_contains_the_true_value():
    """Independently sourced decimals, NOT computed by the library.

    ``Phi(-9)``, ``erfc(8)`` and ``erfc(10)`` are typed in as truncations of
    their true expansions, so the true value lies in ``[literal, literal+1ulp]``
    and that whole bracket must sit inside the enclosure.
    """
    cases = [
        (lambda p: Phi(Interval.exact(-9), p), "0.00000000000000000011285884059538"),
        (lambda p: erfc(Interval.exact(8), p), "0.000000000000000000000000000011224297172982"),
        (lambda p: erfc(Interval.exact(10), p), "0.00000000000000000000000000000000000000000000208848758376"),
        (lambda p: normal_sf(Interval.exact(20), p), "0.000000000000000000000000000000000000000000000000000000000000000000000000000000000000000027536241186"),
    ]
    for fn, lit in cases:
        for prec in (20, 40):
            assert bracket(lit) in fn(prec), (lit, prec, repr(fn(prec)))


def test_tail_bracket_agrees_with_the_exact_series_above_the_crossover():
    """A CERTIFIED cross-check between two independent derivations.

    The alternating Maclaurin series stays exact above ``ERF_CROSSOVER``; it is
    only its COST that makes the crossover exist. So for moderate ``z`` above
    the crossover both branches can be evaluated and must overlap, in exact
    ``Fraction`` arithmetic, with no float and no external oracle. This is the
    strongest available check on the Mills algebra.

    Fires on: any sign or coefficient error in ``_erfc_mills``.
    """
    for zs, prec in (("13/2", 70), ("7", 80), ("8", 100), ("10", 140)):
        z = F(zs)
        series = Interval(1) - T._erf_series(z, prec)     # exact, independent
        mills = T._erfc_mills(z, prec)
        assert series.intersect(mills) is not None, zs
        assert mills.lo <= series.hi and series.lo <= mills.hi
        # and below the crossover the library intersects them, so the returned
        # enclosure is inside both
        if z <= T.ERF_CROSSOVER:
            got = erfc(Interval.exact(z), prec)
            assert got in series and got in mills


def test_mills_bracket_relative_width_is_the_documented_one():
    """TIGHTNESS, not containment: the width claimed in the docstrings.

    The bracket's relative width is about ``3/(4 z^4)``, which is what makes it
    usable; it does NOT depend on prec, which is what makes the docstring say so
    out loud.
    """
    for zs in ("13/2", "8", "12"):
        z = F(zs)
        e = erfc(Interval.exact(z), 30)
        rel = e.width() / e.lo
        assert rel <= 4 * F(3, 4) / z ** 4, (zs, float(rel))
        assert rel > 0
        # prec is very nearly inert out here: the Mills gap is a floor that
        # prec cannot go below, and 60 extra digits buy less than a factor of
        # two. (Not exact equality: the exp enclosure inside the bracket and
        # the round_out significand do depend on prec, by a negligible amount
        # next to the gap itself.)
        assert erfc(Interval.exact(z), 90).width() > e.width() / 2


def test_erfc_and_normal_sf_agree_with_erf_and_Phi():
    """Consistency of the added API with the existing one, where both work."""
    for v in (F(-3), F(-1, 2), F(0), F(1), F(3)):
        x = Interval.exact(v)
        assert (Interval(1) - erf(x, 30)).intersect(erfc(x, 30)) is not None
        assert (Interval(1) - Phi(x, 30)).intersect(normal_sf(x, 30)) is not None
        assert (erfc(x, 30) + erfc(Interval.exact(-v), 30)).intersect(
            Interval(2)) is not None
    assert normal_sf(Interval.exact(0), 30) == Interval(F(1, 2))


# --- NC9: __abs__ without the interior minimum ------------------------------

def _broken_abs_endpoints_only(x: Interval) -> Interval:
    """``abs`` evaluated at the endpoints only, ignoring the interior minimum
    ``|0| = 0`` attained when the interval straddles zero."""
    a, b = abs(x.lo), abs(x.hi)
    return Interval(min(a, b), max(a, b))


def test_negative_control_abs_without_the_interior_minimum():
    """``Interval.__abs__`` has the SAME interior-minimum trap as ``** 2``,
    which NC6 guards, and before this control it had no test at all: an auditor
    mutated it to the endpoint-only form above and the full suite passed 52/52.

    Fires on: ``a, b = abs(self.lo), abs(self.hi); return
    Interval(min(a, b), max(a, b))`` in ``core.Interval.__abs__``.
    """
    x = Interval(-2, 3)
    broken = _broken_abs_endpoints_only(x)
    assert broken == Interval(2, 3)
    assert 0 in x                       # x = 0 is in the domain ...
    assert F(0) not in broken           # ... but |0| = 0 is not in the range
    assert abs(x) == Interval(0, 3)     # the library gets it right
    assert F(0) in abs(x)
    assert abs(Interval(-5, -2)) == Interval(2, 5)
    assert abs(Interval(2, 5)) == Interval(2, 5)


# --- NC10: monotone functions evaluated at one endpoint only ----------------

def _broken_monotone_lo_only(fn, x: Interval, prec: int) -> Interval:
    """The far endpoint of the input silently dropped."""
    e = fn(Interval.exact(x.lo), prec)
    return Interval(e.lo, e.hi)


def test_negative_control_monotone_functions_need_both_endpoints():
    """Every constant-containment test in this file used POINT intervals, so
    the endpoint selection in ``erf`` and ``log`` was unpinned on wide inputs:
    an auditor replaced each by the ``x.lo`` enclosure alone — containment lost
    for every non-degenerate input — and the suite passed 52/52.

    Fires on: ``def erf(x, prec): e = _erf_point(x.lo, prec); return
    Interval(e.lo, e.hi)`` and the same mutation of ``log``, ``exp``, ``sqrt``,
    ``erfc``, ``Phi``, ``normal_sf``.
    """
    probes = [
        (erf, Interval(0, 2), F(99, 100)),
        (log, Interval(1, 10), F(23, 10)),
        (exp, Interval(0, 2), F(5)),
        (sqrt, Interval(1, 4), F(3, 2)),
        (Phi, Interval(-1, 2), F(9, 10)),
        (erfc, Interval(0, 2), F(1, 2)),
        (normal_sf, Interval(-1, 1), F(1, 2)),
    ]
    for fn, x, probe in probes:
        good = fn(x, 20)
        broken = _broken_monotone_lo_only(fn, x, 20)
        assert probe in good, (fn.__name__, repr(good))
        assert probe not in broken, fn.__name__


def test_decreasing_functions_are_not_treated_as_increasing():
    """``erfc`` and ``normal_sf`` DECREASE: the enclosure at ``x.hi`` supplies
    the lower endpoint. Building them the other way round produces ``lo > hi``,
    which the ``Interval`` invariant refuses — so the direction is load-bearing
    and cannot be got wrong quietly.

    Fires on: swapping ``x.hi`` and ``x.lo`` in ``erfc`` or ``normal_sf``.
    """
    x = Interval(0, 2)
    with pytest.raises(ValueError, match="empty interval"):
        Interval(T._erfc_point(x.lo, 20).lo, T._erfc_point(x.hi, 20).hi)
    good = erfc(x, 20)
    assert good.lo < good.hi
    assert good.hi <= 1 and good.lo > 0


# --- NC11: the Mills lower bound with the +1 dropped ------------------------

def test_negative_control_mills_lower_bound_without_the_plus_one():
    """``erfc(z) > 2 z e^(-z^2) / (sqrt(pi) (2 z^2 + 1))``. Drop the ``+ 1``
    and the expression becomes ``e^(-z^2)/(z sqrt(pi))``, which is the proved
    UPPER bound — strictly above ``erfc(z)``. Using it as a lower bound is a
    containment violation, and the test shows it by placing it strictly above
    the library's certified upper endpoint.

    Fires on: ``2 * zz`` in place of ``2 * zz + 1`` in ``_erfc_mills``.
    """
    for zs in ("13/2", "8", "12"):
        z = F(zs)
        prec = 30
        g = prec + 20
        e = exp(Interval.exact(-(z * z)), g)
        sp = sqrt(pi(g), g)
        broken_lo = ((Interval.exact(2 * z) * e)
                     / (sp * Interval.exact(2 * z * z))).lo
        good = erfc(Interval.exact(z), prec)
        assert broken_lo > good.hi, zs        # not a lower bound at all
        assert good.lo < broken_lo


# --- NC12: the sin extremum sign at NEGATIVE odd multiples of pi/2 ----------

def test_sin_attains_minus_one_at_a_negative_odd_multiple_of_half_pi():
    """The sign selection in the interior-extrema loop is ``j % 4 == 1``, which
    relies on Python's ``%`` giving the mathematical residue for negative ``j``
    (``j = -1 -> 3 -> -1``, ``j = -3 -> 1 -> +1``). No exact test covered a
    negative odd ``j``: an auditor mutated it to ``abs(j) % 4 == 1`` and the only
    test that failed was the NON-CERTIFYING float sweep — the one test this
    program forbids citing as evidence.

    ``[-8/5, -3/2]`` contains ``-pi/2 = -1.5707963...``, where ``sin = -1``.

    Fires on: ``abs(j) % 4 == 1`` in ``_sin_cos``.
    """
    s = sin(Interval(F(-8, 5), F(-3, 2)), 30)
    assert s.lo == -1
    assert s.hi < 0
    # j = -3: sin(-3 pi/2) = +1, the other residue of the same branch
    s3 = sin(Interval(F(-48, 10), F(-46, 10)), 30)
    assert s3.hi == 1
    # and the positive-j cases still hold
    assert sin(Interval(F(15, 10), F(16, 10)), 30).hi == 1
    assert cos(Interval(F(-1, 10), F(1, 10)), 30).hi == 1


# --- R5: the reduced-index span, so the dead branch cannot mask a regression -

def test_reduced_index_span_stays_small():
    """``_sin_cos`` carries ``if j_hi - j_lo > _J_SPAN_GUARD: return [-1, 1]``,
    which an auditor showed is unreachable (800 randomised calls plus magnitudes
    to ``10**40``: zero hits). Returning ``[-1, 1]`` is always sound for
    ``sin``/``cos``, so the branch is kept — but it must not be the only thing
    standing between a regression in the full-period test and an unbounded loop.
    This measures the span directly, so a regression shows up here.

    Fires on: any change that lets the reduced span grow (e.g. weakening the
    full-period test in ``_sin_cos``).
    """
    rng = random.Random(20260918)
    worst = 0
    samples = [(F(rng.randint(-10 ** 6, 10 ** 6), rng.randint(1, 97)),
                F(rng.randint(0, 629), 100)) for _ in range(120)]
    samples += [(F(m), w) for m in (10 ** 3, 10 ** 9, 10 ** 18, 10 ** 40)
                for w in (F(0), F(1), F(6), F(62, 10))]
    for a, w in samples:
        x = Interval(a, a + w)
        for prec in (1, 5, 25):
            p = T._pi_for(x, prec)
            if x.width() >= 2 * p.hi:
                continue
            half_pi = p * F(1, 2)
            q = x / half_pi
            span = T._ceil_frac(q.hi) - T._floor_frac(q.lo)
            worst = max(worst, span)
    assert worst <= 8, worst
    assert worst < T._J_SPAN_GUARD


# --- R6: the two documented refusals that the audits asked to be stated -----

def test_float_membership_is_a_deliberate_refusal():
    """``0.5 in Interval(0, 1)`` raises ``TypeError``; it does not return a
    bool. Neither bool is honest — ``False`` is a wrong answer and ``True``
    admits a binary double as an exact probe — so the membership test refuses.
    This is now documented in the module docstring and in ``__contains__``;
    the test exists so the behaviour cannot drift silently into either bool.
    """
    with pytest.raises(TypeError, match="float endpoints are refused"):
        0.5 in Interval(0, 1)
    assert F(1, 2) in Interval(0, 1)
    assert F("0.5") in Interval(0, 1)
    assert Interval(F(1, 4), F(1, 2)) in Interval(0, 1)


def test_decimal_built_from_a_float_is_the_double_not_the_decimal():
    """The one hole in the float refusal, pinned so it stays visible.

    ``to_fraction`` refuses ``float`` but accepts ``Decimal`` unconditionally,
    and ``Decimal(0.1)`` IS the binary double. The conversion is exact, so no
    bound is ever wrong — but the refusal does not reach through a ``Decimal``,
    and a ``Decimal`` does not carry its provenance, so this cannot be detected
    in code. It is documented beside the float refusal instead.
    """
    from decimal import Decimal

    assert Interval(Decimal("0.1")).lo == F(1, 10)          # from a string: exact
    assert Interval(Decimal(0.1)).lo != F(1, 10)            # from a float: the double
    assert Interval(Decimal(0.1)).lo == F(3602879701896397, 36028797018963968)
    with pytest.raises(TypeError):
        Interval(0.1)


# --- R7: what prec means for exp, which the module docstring used to misstate -

def test_exp_prec_is_a_relative_width_hint_not_an_absolute_one():
    """The module docstring said ``prec`` is a decimal-digits-of-target-WIDTH
    hint, without qualification. For ``exp`` the target is RELATIVE: the
    remainder target is absolute on ``exp(u)`` with ``|u| <= 1/2``, and the ``k``
    squarings multiply the absolute width by ``exp(t)``. Measured at
    ``prec = 30``: ``exp(1)`` width 6.5e-31 but ``exp(700)`` width 3.1e+272.

    No containment is lost — every one of those intervals contains the true
    value — so this test pins the CONTRACT, which is now stated per function.
    """
    assert exp(Interval.exact(1), 30).width() < F(1, 10 ** 30)
    w700 = exp(Interval.exact(700), 30).width()
    assert w700 > 10 ** 270                     # not 1e-30
    rel = w700 / exp(Interval.exact(700), 30).lo
    assert rel < F(1, 10 ** 28)                 # relative, as documented
    assert bracket(E_LIT) in exp(Interval.exact(1), 30)


def test_erf_crossover_regime_is_visible_and_documented():
    """``prec`` is INERT above the crossover, and in the ``Phi`` coordinate the
    crossover sits at ``6 sqrt 2 = 8.4853...``, which is what a consumer works
    in. The audits found this undocumented; it is now stated in ``erf``,
    ``erfc``, ``Phi`` and ``normal_sf``. Pinned here so the documentation and
    the behaviour cannot drift apart.
    """
    below = Interval(F(8485281, 10 ** 6))
    above = Interval(F(8485282, 10 ** 6))
    # below the crossover, 40 more digits of prec buy 40 orders of magnitude
    assert Phi(below, 60).width() < Phi(below, 20).width() / 10 ** 30
    # above it, they buy essentially nothing: the Mills gap is a floor
    assert Phi(above, 60).width() > Phi(above, 20).width() / 2
    # the cliff in achievable width is now small, because the two branches are
    # intersected below the crossover rather than switched between
    assert Phi(above, 20).width() < Phi(below, 20).width() * 100
