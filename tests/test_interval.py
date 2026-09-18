"""Tests for the certified interval library in ``research/interval/``.

THE CONTRACT UNDER TEST. Every ``Interval`` returned by the library provably
contains the true value; containment is unconditional and does not depend on
the ``prec`` hint. Tightness is best effort and does depend on it.

These tests are organised in five groups:

1. ``Interval`` algebra, including the non-monotone traps.
2. Containment of independently-sourced high-precision decimal constants.
3. Tightness monotonicity in ``prec``.
4. **NEGATIVE CONTROLS** — eight deliberately weakened variants, each built
   locally inside this file (never by editing the library), each asserted to
   LOSE containment. These are the point of the file. If a negative control
   stops failing, the corresponding safeguard in the library has been removed
   or has stopped working.
5. A seeded, deterministic float cross-check sweep, labelled NON-CERTIFYING.

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
    Interval, Phi, cos, erf, exp, log, normal_pdf, pi, sin, sqrt,
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
    """Above the documented crossover, erf uses the monotone tail bracket."""
    e = erf(Interval.exact(8), 30)
    assert e.hi == 1
    assert e.lo < 1
    assert e.lo > F(9999999, 10000000)


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
