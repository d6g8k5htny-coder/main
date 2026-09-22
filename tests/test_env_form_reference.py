"""Tests and negative controls for ``research/rn/env_form_reference.py``.

WHAT IS UNDER TEST. The ``env_form`` envelope shape, assembled in exact
rationals and enclosed with ``research/interval/``. The module is deliberately
NOT the whitened artifact ``docs/OPEN_PROBLEMS.md`` A5 asks for; the sources
give two incompatible readings of "whitened ``env_form``" and this repository
does not choose between them. ``test_the_module_refuses_to_whiten`` pins that
the module stays out of it.

The strongest test here is ``test_against_an_independent_reimplementation``: a
second, deliberately naive transcription of the frozen formula, written from
the frozen source rather than from the module, evaluated with ``decimal`` at
60 digits. A shared-code bug cannot pass it.

Nothing here closes Piece 1 or Piece 2 of ``D3-LEMMA-RN-UNIF``, and nothing
here re-certifies any number the frozen engine produced.
"""
from __future__ import annotations

import decimal
import os
import sys
from fractions import Fraction as F
from math import comb, factorial

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.interval import Interval  # noqa: E402
from research.rn.env_form_reference import (  # noqa: E402
    MX2_INDEX_SUM, REFERENCE_FORMS, REFERENCE_MOMENTS, REFERENCE_R,
    REMAINDER_ORDER, TORUS_PERIOD, env_form_enclosure, env_form_parts,
)
from research.rn.hermite_envelope import he_abs  # noqa: E402

ORDERS = (0, 1, 2, 3, 4)
GAMMAS = ((0, 0), (1, 0), (0, 2), (2, 1))
MODULE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "research", "rn", "env_form_reference.py")


# ---------------------------------------------------------------------------
# An independent transcription of the frozen body, written from
# d3_rn_unif.py lines 381-421 and not from the module under test.
# ---------------------------------------------------------------------------

def exp_neg_bracket(a: F, terms: int = 400) -> tuple[F, F]:
    """An exact rational bracket for ``exp(-a)``, ``a >= 0``. No rounding at all.

    ``exp(a) = sum_{k<=N} a^k/k! + R`` with
    ``0 < R <= a^(N+1)/(N+1)! * 1/(1 - a/(N+2))`` whenever ``N + 2 > a``,
    the geometric majorant of the tail from ``(N+1+i)! >= (N+1)! (N+2)^i``.
    So ``exp(a)`` is in ``[s, s + Rb]`` and ``exp(-a)`` in ``[1/(s+Rb), 1/s]``.

    An earlier draft used ``decimal.Context.exp`` as the oracle. That is
    accurate to its stated precision and no further, and the module's
    enclosure turned out TIGHTER than a 60-digit decimal -- relative width
    5.6e-61 against a 1.1e-60 rounding error -- so the oracle, not the module,
    was failing the comparison. An exact bracket has no such ceiling.
    """
    if a < 0:
        raise ValueError("a must be non-negative")
    if terms + 2 <= a:
        raise ValueError("terms too small for the geometric majorant")
    s = F(0)
    term = F(1)
    for k in range(terms + 1):
        s += term
        term *= a / (k + 1)
    remainder = term * (1 / (1 - a / (terms + 2)))
    return 1 / (s + remainder), 1 / s


def naive_env_form_bracket(moments, forms, gamma, d, qord, R=REFERENCE_R):
    """The frozen formula, transcribed straight, as an exact rational bracket.

    Written from ``d3_rn_unif.py`` lines 381-421, not from the module under
    test. Every coefficient is non-negative -- they are built from absolute
    values -- so the bracket is monotone in the three Gaussian factors and the
    endpoints pair up directly.
    """
    rho = d - R / 2
    rimg = 24 - d - R / 2

    tot = F(0)
    for (b1, b2), mu in moments.items():
        if mu == 0:
            continue
        base = abs(mu) / (factorial(b1) * factorial(b2))
        for e1 in range(qord + 1):
            e2 = qord - e1
            tot += (base * comb(qord, e1)
                    * he_abs(b1 + gamma[0] + e1, d)
                    * he_abs(b2 + gamma[1] + e2, d))

    rem = F(0)
    for a, c in forms:
        ao = a[0] + a[1]
        inner = F(0)
        for b1 in range(10):
            b2 = 9 - b1
            if b1 < a[0] or b2 < a[1]:
                continue
            inner += ((R / 2) ** (9 - ao)
                      / (factorial(b1 - a[0]) * factorial(b2 - a[1])))
        rem += abs(c) * inner

    mx2 = F(0)
    for d1 in range(12):
        d2 = 11 - d1
        for e1 in range(qord + 1):
            e2 = qord - e1
            mx2 = max(mx2, comb(qord, e1)
                      * he_abs(d1 + gamma[0] + e1, rho)
                      * he_abs(d2 + gamma[1] + e2, rho))
    rem *= mx2

    img = sum((abs(c) for _, c in forms), F(0)) * 8 * he_abs(6 + qord, rimg)

    assert tot >= 0 and rem >= 0 and img >= 0
    lo = hi = F(0)
    for coeff, sep in ((tot, d), (rem, rho), (img, rimg)):
        e_lo, e_hi = exp_neg_bracket(sep * sep / 2)
        lo += coeff * e_lo
        hi += coeff * e_hi
    return lo, hi



# ---------------------------------------------------------------------------
# 1. It runs at orders 0-4, which is what "runnable" means
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("qord", ORDERS)
@pytest.mark.parametrize("gamma", GAMMAS)
def test_runs_at_every_order_and_gamma(qord, gamma):
    enc = env_form_enclosure(REFERENCE_MOMENTS, REFERENCE_FORMS, gamma, F(5), qord, 40)
    assert isinstance(enc, Interval)
    assert enc.lo > 0 and enc.lo <= enc.hi


@pytest.mark.parametrize("qord", ORDERS)
@pytest.mark.parametrize("gamma", GAMMAS)
def test_against_an_independent_reimplementation(qord, gamma):
    """The enclosure and an exact rational bracket of the formula must meet.

    Both contain the true value, so they must overlap. A shared-code bug
    cannot pass this: the bracket is transcribed from the frozen source and
    shares nothing with the module but ``he_abs``. It earned its keep at once
    -- it caught the module summing the binomial split inside ``mx2`` where
    the frozen body maximises over it.
    """
    lo, hi = naive_env_form_bracket(REFERENCE_MOMENTS, REFERENCE_FORMS, gamma, F(5), qord)
    enc = env_form_enclosure(REFERENCE_MOMENTS, REFERENCE_FORMS, gamma, F(5), qord, 60)
    assert enc.lo <= hi and lo <= enc.hi, (qord, gamma)
    assert (hi - lo) / lo < F(1, 10**100)          # the oracle really is exact


# ---------------------------------------------------------------------------
# 2. Structure
# ---------------------------------------------------------------------------

def test_the_three_parts_sum_to_the_enclosure():
    p = env_form_parts(REFERENCE_MOMENTS, REFERENCE_FORMS, (0, 0), F(5), 2)
    from research.rn.hermite_envelope import gaussian_kernel
    by_hand = Interval.exact(0)
    for coeff, sep in ((p.tot_coefficient, p.d),
                       (p.remainder_coefficient, p.rho),
                       (p.image_coefficient, p.rimg)):
        by_hand = by_hand + gaussian_kernel(Interval.exact(sep * sep), 40) * Interval.exact(coeff)
    enc = p.enclosure(40)
    assert enc.lo == by_hand.lo and enc.hi == by_hand.hi


def test_separations_follow_the_frozen_definitions():
    p = env_form_parts(REFERENCE_MOMENTS, REFERENCE_FORMS, (0, 0), F(5), 1)
    assert p.rho == F(5) - REFERENCE_R / 2
    assert p.rimg == TORUS_PERIOD - F(5) - REFERENCE_R / 2


def test_coefficients_are_exact_rationals():
    p = env_form_parts(REFERENCE_MOMENTS, REFERENCE_FORMS, (0, 0), F(5), 3)
    for c in (p.tot_coefficient, p.remainder_coefficient, p.image_coefficient):
        assert isinstance(c, F)


def test_the_envelope_grows_with_the_derivative_order():
    vals = [env_form_parts(REFERENCE_MOMENTS, REFERENCE_FORMS, (0, 0), F(5), q).tot_coefficient
            for q in ORDERS]
    assert vals == sorted(vals)


def test_higher_prec_nests_and_never_breaks_containment():
    encs = [env_form_enclosure(REFERENCE_MOMENTS, REFERENCE_FORMS, (0, 0), F(5), 2, p)
            for p in (10, 20, 40, 80)]
    for a in encs:
        for b in encs:
            assert a.lo <= b.hi and b.lo <= a.hi
    widths = [e.width() for e in encs]
    assert widths == sorted(widths, reverse=True)


def test_an_empty_moment_table_leaves_only_the_remainder_and_image():
    p = env_form_parts({}, REFERENCE_FORMS, (0, 0), F(5), 1)
    assert p.tot_coefficient == 0
    assert p.remainder_coefficient > 0 and p.image_coefficient > 0


def test_no_forms_leaves_only_the_moment_series():
    p = env_form_parts(REFERENCE_MOMENTS, (), (0, 0), F(5), 1)
    assert p.tot_coefficient > 0
    assert p.remainder_coefficient == 0 and p.image_coefficient == 0


# ---------------------------------------------------------------------------
# 3. NEGATIVE CONTROLS
# ---------------------------------------------------------------------------

def test_control_dropping_the_image_part_under_bounds():
    p = env_form_parts(REFERENCE_MOMENTS, REFERENCE_FORMS, (0, 0), F(5), 2)
    full = p.enclosure(60)
    from research.rn.env_form_reference import EnvFormParts
    maimed = EnvFormParts(p.tot_coefficient, p.remainder_coefficient, F(0),
                          p.d, p.rho, p.rimg, p.qord).enclosure(60)
    assert maimed.hi < full.hi


def test_control_dropping_the_binomial_split_changes_the_answer():
    """Replacing C(qord, e1) by 1 must break agreement at qord >= 2."""
    def without_binomial(gamma, d, qord):
        tot = F(0)
        for (b1, b2), mu in REFERENCE_MOMENTS.items():
            if mu == 0:
                continue
            base = abs(mu) / (factorial(b1) * factorial(b2))
            for e1 in range(qord + 1):
                e2 = qord - e1
                tot += (base * he_abs(b1 + gamma[0] + e1, d)
                        * he_abs(b2 + gamma[1] + e2, d))
        return tot

    good = env_form_parts(REFERENCE_MOMENTS, REFERENCE_FORMS, (0, 0), F(5), 2)
    assert without_binomial((0, 0), F(5), 2) != good.tot_coefficient


def test_control_the_mx2_index_sum_matters():
    """The frozen code sums its mx2 indices to 11; its comment says 9.

    This control pins that the two give DIFFERENT answers, so the discrepancy
    recorded in ``MX2_INDEX_SUM`` is a real fork and not a harmless one. It
    takes no view on which the frozen body meant; the frozen body is never
    edited, and no status turns on this.
    """
    def remainder_with(index_sum, qord):
        mx2 = F(0)
        for d1 in range(index_sum + 1):
            d2 = index_sum - d1
            acc = F(0)
            for e1 in range(qord + 1):
                e2 = qord - e1
                acc += comb(qord, e1) * he_abs(d1 + e1, F(5) - REFERENCE_R / 2) \
                    * he_abs(d2 + e2, F(5) - REFERENCE_R / 2)
            mx2 = max(mx2, acc)
        return mx2

    assert MX2_INDEX_SUM == 11
    assert remainder_with(11, 2) != remainder_with(9, 2)
    assert remainder_with(11, 2) > remainder_with(9, 2)


def test_control_an_inexact_R_changes_the_coefficients():
    """R must be the exact 1/20, not a decimal that merely prints as 0.05."""
    near = F(decimal.Decimal("0.0500000000000000027755575615628914"))
    a = env_form_parts(REFERENCE_MOMENTS, REFERENCE_FORMS, (0, 0), F(5), 2)
    b = env_form_parts(REFERENCE_MOMENTS, REFERENCE_FORMS, (0, 0), F(5), 2, R=near)
    assert a.remainder_coefficient != b.remainder_coefficient
    assert REFERENCE_R == F(1, 20)


def test_control_a_zero_moment_is_skipped_not_counted():
    padded = dict(REFERENCE_MOMENTS)
    padded[(1, 1)] = F(0)
    assert (env_form_parts(padded, REFERENCE_FORMS, (0, 0), F(5), 1).tot_coefficient
            == env_form_parts(REFERENCE_MOMENTS, REFERENCE_FORMS, (0, 0), F(5), 1).tot_coefficient)


def test_control_a_nonzero_moment_is_counted():
    padded = dict(REFERENCE_MOMENTS)
    padded[(1, 1)] = F(1, 1000)
    assert (env_form_parts(padded, REFERENCE_FORMS, (0, 0), F(5), 1).tot_coefficient
            > env_form_parts(REFERENCE_MOMENTS, REFERENCE_FORMS, (0, 0), F(5), 1).tot_coefficient)


# ---------------------------------------------------------------------------
# 4. The boundary this module must not cross
# ---------------------------------------------------------------------------

def test_the_module_refuses_to_whiten():
    """No whitening matrix, no eigenvalue scaling, no LAM0 anywhere.

    The sources give two incompatible readings of "whitened env_form" -- one
    its own author stamped PROXY only, one a later theorem refuses -- so the
    module builds the shape and stops. If a future edit adds a 1/sqrt(lambda)
    scaling, this control fails and the edit has to argue for itself.
    """
    src = open(MODULE, encoding="utf-8").read()
    body = src.split('"""', 2)[2]          # past the module docstring
    for token in ("LAM0", "sqrt", "eigval", "whiten", "V0"):
        assert token not in body, f"{token!r} appears in the module body"


def test_the_reference_data_is_labelled_and_is_not_the_programs():
    src = open(MODULE, encoding="utf-8").read()
    assert "REFERENCE" in src
    assert "MOMS[k]" in src and "FORMS[k]" in src     # named only to disclaim them
    assert "not the program's" in src


def test_reference_moments_are_the_gaussian_ones():
    assert REFERENCE_MOMENTS[(0, 0)] == 1
    assert REFERENCE_MOMENTS[(2, 0)] == 1
    assert REFERENCE_MOMENTS[(4, 0)] == 3
    assert REFERENCE_MOMENTS[(6, 0)] == 15
    assert REFERENCE_MOMENTS[(2, 2)] == 1
    assert all(k[0] % 2 == 0 and k[1] % 2 == 0 for k in REFERENCE_MOMENTS)
    assert all(k[0] + k[1] <= 8 for k in REFERENCE_MOMENTS)


def test_remainder_order_is_nine():
    assert REMAINDER_ORDER == 9


# ---------------------------------------------------------------------------
# 5. Refusals
# ---------------------------------------------------------------------------

def test_a_negative_order_is_refused():
    with pytest.raises(ValueError):
        env_form_parts(REFERENCE_MOMENTS, REFERENCE_FORMS, (0, 0), F(5), -1)


def test_a_malformed_gamma_is_refused():
    with pytest.raises(ValueError):
        env_form_parts(REFERENCE_MOMENTS, REFERENCE_FORMS, (0, 0, 0), F(5), 1)


# ---------------------------------------------------------------------------
# 6. Parts that are numerically negligible in the total need their own tests.
#
# Mutating the image allowance's shell factor from 8 to 4 left all 59 tests
# green: at d = 5 the image part is about 1e-70 of the total, far below the
# enclosure's own relative width of 5.6e-61, so no assertion on the TOTAL can
# see it. Exposing the three coefficients separately is what makes it testable
# at all, and these controls are why that shape was chosen.
# ---------------------------------------------------------------------------

def _image_coefficient_by_hand(forms, qord, rimg, shell=8):
    return sum((abs(c) for _, c in forms), F(0)) * shell * he_abs(6 + qord, rimg)


@pytest.mark.parametrize("qord", ORDERS)
def test_the_image_coefficient_is_the_first_image_shell(qord):
    """Eight images: the frozen `_IMG` is the 3x3 block without its centre."""
    p = env_form_parts(REFERENCE_MOMENTS, REFERENCE_FORMS, (0, 0), F(5), qord)
    assert p.image_coefficient == _image_coefficient_by_hand(REFERENCE_FORMS, qord, p.rimg)
    assert len([(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0)]) == 8


def test_control_a_wrong_image_shell_factor_is_visible_in_the_coefficient():
    p = env_form_parts(REFERENCE_MOMENTS, REFERENCE_FORMS, (0, 0), F(5), 2)
    for wrong in (4, 6, 9, 16):
        assert p.image_coefficient != _image_coefficient_by_hand(
            REFERENCE_FORMS, 2, p.rimg, shell=wrong)


def test_the_image_part_is_negligible_at_this_separation_and_that_is_stated():
    """Pin the reason the total cannot see the image part, so it stays known."""
    p = env_form_parts(REFERENCE_MOMENTS, REFERENCE_FORMS, (0, 0), F(5), 2)
    from research.rn.hermite_envelope import gaussian_kernel
    tot_part = gaussian_kernel(Interval.exact(p.d * p.d), 60) * Interval.exact(p.tot_coefficient)
    img_part = gaussian_kernel(Interval.exact(p.rimg * p.rimg), 60) * Interval.exact(p.image_coefficient)
    assert img_part.hi / tot_part.lo < F(1, 10**50)


@pytest.mark.parametrize("qord", ORDERS)
def test_the_remainder_coefficient_matches_an_independent_assembly(qord):
    """The remainder is also small; pin its coefficient directly too."""
    p = env_form_parts(REFERENCE_MOMENTS, REFERENCE_FORMS, (0, 0), F(5), qord)
    rho = F(5) - REFERENCE_R / 2
    inner_total = F(0)
    for a, c in REFERENCE_FORMS:
        ao = a[0] + a[1]
        inner = F(0)
        for b1 in range(10):
            b2 = 9 - b1
            if b1 < a[0] or b2 < a[1]:
                continue
            inner += ((REFERENCE_R / 2) ** (9 - ao)
                      / (factorial(b1 - a[0]) * factorial(b2 - a[1])))
        inner_total += abs(c) * inner
    mx2 = F(0)
    for d1 in range(12):
        d2 = 11 - d1
        for e1 in range(qord + 1):
            e2 = qord - e1
            mx2 = max(mx2, comb(qord, e1) * he_abs(d1 + e1, rho) * he_abs(d2 + e2, rho))
    assert p.remainder_coefficient == inner_total * mx2


@pytest.mark.parametrize("qord", ORDERS)
def test_the_tot_coefficient_matches_an_independent_assembly(qord):
    p = env_form_parts(REFERENCE_MOMENTS, REFERENCE_FORMS, (0, 0), F(5), qord)
    tot = F(0)
    for (b1, b2), mu in REFERENCE_MOMENTS.items():
        if mu == 0:
            continue
        base = abs(mu) / (factorial(b1) * factorial(b2))
        for e1 in range(qord + 1):
            e2 = qord - e1
            tot += base * comb(qord, e1) * he_abs(b1 + e1, F(5)) * he_abs(b2 + e2, F(5))
    assert p.tot_coefficient == tot


# ---------------------------------------------------------------------------
# 7. The envelope is not monotone in d, and the separations must be positive.
#
# `docs/ENGINE_RECOVERY.md` raises the "for |y| >= d" uniformity question
# qualitatively. With the shape in exact arithmetic it becomes a number: the
# image allowance sits at rimg = 24 - d - R/2, which shrinks as d grows, so
# that part grows, overtakes the moment series, and eventually runs away.
# ---------------------------------------------------------------------------

def _parts_of(d, qord=2, prec=80):
    from research.rn.hermite_envelope import gaussian_kernel
    p = env_form_parts(REFERENCE_MOMENTS, REFERENCE_FORMS, (0, 0), d, qord)
    tot = gaussian_kernel(Interval.exact(p.d * p.d), prec) * Interval.exact(p.tot_coefficient)
    img = gaussian_kernel(Interval.exact(p.rimg * p.rimg), prec) * Interval.exact(p.image_coefficient)
    return p, tot, img


def test_the_image_part_overtakes_the_moment_series():
    """Below the crossover the moment series dominates; above it, the image."""
    _, tot_lo, img_lo = _parts_of(F(11))
    assert img_lo.hi < tot_lo.lo
    _, tot_hi, img_hi = _parts_of(F(13))
    assert img_hi.lo > tot_hi.hi


def test_the_crossover_is_where_it_was_measured():
    """Between d = 12.0115565 and 12.0115566, on the reference data at q = 2."""
    _, tot_a, img_a = _parts_of(F(120115565, 10**7))
    assert img_a.hi < tot_a.lo
    _, tot_b, img_b = _parts_of(F(120115566, 10**7))
    assert img_b.lo > tot_b.hi


def test_the_envelope_is_not_monotone_in_d():
    """The headline: a larger d does not buy a smaller bound past the minimum."""
    at = {d: env_form_enclosure(REFERENCE_MOMENTS, REFERENCE_FORMS, (0, 0), F(d), 2, 80)
          for d in (5, 9, 12, 16, 20, 23)}
    assert at[9].hi < at[5].lo                     # still falling
    assert at[12].hi < at[9].lo                    # still falling
    assert at[16].lo > at[12].hi                   # risen again
    assert at[20].lo > at[16].hi
    assert at[23].lo > at[20].hi
    assert at[23].lo / at[12].hi > 10**24          # by twenty-four orders


def test_the_crossover_falls_inside_the_lanes_own_T4_region():
    """The part that bears on the lane, and the claim I first got backwards.

    An earlier draft of this module's docstring said the image allowance stays
    negligible across `d` in [5, 17]. It does not: on this reference data it
    goes from 68 orders below the moment series at d = 5 to 48 orders above it
    at d = 17, crossing at about d = 12.01, inside the RN-UNIF lane's own T4
    region. The location is a property of the reference moments and not of the
    program's, which are not here.
    """
    _, tot5, img5 = _parts_of(F(5))
    assert img5.hi / tot5.lo < F(1, 10**60)        # negligible where the push evaluated
    _, tot17, img17 = _parts_of(F(17))
    assert img17.lo / tot17.hi > 10**40            # the whole bound at the top of T4


@pytest.mark.parametrize("d", [F(24), F(25), F(959, 40), F(100)])
def test_a_non_positive_image_separation_is_refused(d):
    """`he_abs` takes abs(), so nothing else would have raised."""
    with pytest.raises(ValueError, match="rimg"):
        env_form_parts(REFERENCE_MOMENTS, REFERENCE_FORMS, (0, 0), d, 2)


@pytest.mark.parametrize("d", [F(1, 100), F(0), F(-1), REFERENCE_R / 2])
def test_a_non_positive_taylor_separation_is_refused(d):
    with pytest.raises(ValueError, match="rho"):
        env_form_parts(REFERENCE_MOMENTS, REFERENCE_FORMS, (0, 0), d, 2)


def test_the_largest_accepted_separation_is_just_below_the_collapse():
    largest = F(TORUS_PERIOD) - REFERENCE_R / 2
    ok = env_form_parts(REFERENCE_MOMENTS, REFERENCE_FORMS, (0, 0),
                        largest - F(1, 1000), 2)
    assert ok.rimg > 0
    with pytest.raises(ValueError):
        env_form_parts(REFERENCE_MOMENTS, REFERENCE_FORMS, (0, 0), largest, 2)


def test_control_without_the_guard_a_negative_separation_returns_a_number():
    """Why the guard is needed: the arithmetic itself does not object.

    `he_abs(n, t)` evaluates at `abs(t)`, so a negative separation produces a
    perfectly ordinary Fraction. The guard is the only thing standing between
    a caller and a number with no referent.
    """
    assert he_abs(8, F(-1, 40)) == he_abs(8, F(1, 40))
    assert he_abs(8, F(-1, 40)) > 0


def test_the_module_docstring_records_the_non_monotonicity():
    src = open(MODULE, encoding="utf-8").read()
    assert "NOT MONOTONE IN ``d``" in src
    assert "12.0115565" in src
