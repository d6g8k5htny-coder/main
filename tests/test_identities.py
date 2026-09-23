"""Exact tests for the corpus absolute-moment identities, with negative controls.

Pins three things:

  1. The corpus identity ``E(|xi| + |eta|)^4 = 12 + 16/pi`` is FALSE. The true
     value is ``12 + 32/pi``; the discrepancy is exactly ``16/pi``; the cause is
     one dropped cross term. Asserted by exact comparison in the ``{1, 1/pi}``
     basis -- no float participates in any verdict.
  2. ``sum_abs_power_moment(2, k)`` against values derived independently in this
     file, by hand, for ``k = 1, 2, 3, 4`` (and 0). The test file does not call
     the module to compute its own expectations.
  3. NEGATIVE CONTROLS. Each one FAILS if the module is weakened in a specific,
     named way: a cross term dropped, the ``{1, 1/pi}`` basis coefficients
     swapped, the discrepancy stated as ``32/pi`` instead of ``16/pi``, the odd
     absolute moments made rational, or the basis view made permissive enough to
     silently swallow an ``s^1`` term.

These tests verify arithmetic. They promote, close, discharge and reclassify
NOTHING. Refuting the identity does not invalidate any derivation that cited
it -- it flags every consumer for re-check, and the consumers are not
enumerated here.
"""
from fractions import Fraction as F
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "research", "identities"))

from gaussian_moments import (  # noqa: E402
    CLAIMED_FALSE_IDENTITY, FOUND_NOT_SETTLED, QSqrt2, Sym,
    abs_moment, abs_moment_sym, binomial_terms, numeric_abs_moment,
    numeric_oracle, oracle_report, rayleigh_even_moment, refutation_record,
    sibling_rayleigh_fourth_moment, sibling_records,
    sibling_truncated_fourth_moment, sibling_truncated_second_moment,
    sum_abs_power_moment, sum_abs_power_terms, truncated_normal_moment,
)

# ---------------------------------------------------------------------------
# Independently derived expectations. Written out by hand from
# E|X|^k = 2^(k/2) Gamma((k+1)/2) / sqrt(pi), NOT read back from the module.
#
#   E|X|^0 = 1        E|X|^1 = s       E|X|^2 = 1
#   E|X|^3 = 2 s      E|X|^4 = 3       s := sqrt(2/pi),  s^2 = 2/pi
#
# E(|xi|+|eta|)^k = sum_j C(k,j) E|xi|^j E|eta|^(k-j):
#   k=0: 1
#   k=1: 1*s*1 + 1*1*s                       = 2 s
#   k=2: 1 + 2*s*s + 1                       = 2 + 2 s^2      = 2 + 4/pi
#   k=3: 2s + 3*s*1 + 3*1*s + 2s             = 10 s
#   k=4: 3 + 4*s*2s + 6*1*1 + 4*2s*s + 3     = 12 + 16 s^2    = 12 + 32/pi
# ---------------------------------------------------------------------------
HAND_DERIVED = {
    0: Sym({0: F(1)}),
    1: Sym({1: F(2)}),
    2: Sym({0: F(2), 2: F(2)}),
    3: Sym({1: F(10)}),
    4: Sym({0: F(12), 2: F(16)}),
}


# ===========================================================================
# 1. Absolute moments
# ===========================================================================

def test_abs_moment_matches_hand_values():
    assert abs_moment(0) == (F(1), 0)
    assert abs_moment(1) == (F(1), 1)
    assert abs_moment(2) == (F(1), 0)
    assert abs_moment(3) == (F(2), 1)
    assert abs_moment(4) == (F(3), 0)
    assert abs_moment(5) == (F(8), 1)
    assert abs_moment(6) == (F(15), 0)


def test_abs_moment_satisfies_the_gamma_recursion_exactly():
    """E|X|^k = (k-1) E|X|^(k-2), the exact ratio of the Gamma closed forms."""
    for k in range(2, 30):
        c_k, p_k = abs_moment(k)
        c_km2, p_km2 = abs_moment(k - 2)
        assert p_k == p_km2
        assert c_k == (k - 1) * c_km2


def test_even_absolute_moments_are_rational_and_odd_ones_are_not():
    for k in range(0, 20):
        _, power = abs_moment(k)
        assert power == (k % 2)


# ===========================================================================
# 2. sum_abs_power_moment against independently derived values
# ===========================================================================

@pytest.mark.parametrize("k", [0, 1, 2, 3, 4])
def test_sum_abs_power_moment_matches_hand_derivation(k):
    assert sum_abs_power_moment(2, k) == HAND_DERIVED[k]


def test_k4_in_the_one_over_pi_basis_is_exactly_12_and_32():
    a, b = sum_abs_power_moment(2, 4).as_a_plus_b_over_pi()
    assert (a, b) == (F(12), F(32))


def test_k2_in_the_one_over_pi_basis_is_exactly_2_and_4():
    a, b = sum_abs_power_moment(2, 2).as_a_plus_b_over_pi()
    assert (a, b) == (F(2), F(4))


def test_odd_k_is_not_in_the_one_over_pi_basis():
    """k = 1, 3 land on s^1, which is not a + b/pi. The view must REFUSE."""
    for k in (1, 3, 5):
        with pytest.raises(ValueError):
            sum_abs_power_moment(2, k).as_a_plus_b_over_pi()


def test_n_equals_one_reduces_to_the_plain_absolute_moment():
    for k in range(0, 8):
        assert sum_abs_power_moment(1, k) == abs_moment_sym(k)


def test_more_summands_can_leave_the_two_element_basis():
    """n = 4, k = 4 carries an s^4 = 4/pi^2 term, so {1, 1/pi} does not span it.

    The (1,1,1,1) exponent tuple has four odd factors, each contributing one
    power of s. Its multinomial coefficient is 4!/(1!)^4 = 24.
    """
    v = sum_abs_power_moment(4, 4)
    assert v.coeffs.get(4) == F(24)
    with pytest.raises(ValueError):
        v.as_a_plus_b_over_pi()
    # n = 3 with odd k also leaves the basis: (1,1,1) gives s^3.
    assert sum_abs_power_moment(3, 3).coeffs.get(3) == F(6)


# ===========================================================================
# 3. The refutation
# ===========================================================================

def test_the_corpus_identity_is_false():
    rec = refutation_record()
    assert rec.verdict == "REFUTED"
    assert rec.claimed != rec.derived
    assert rec.claimed == Sym.from_a_plus_b_over_pi(12, 16)
    assert rec.derived == Sym.from_a_plus_b_over_pi(12, 32)


def test_discrepancy_is_exactly_16_over_pi():
    rec = refutation_record()
    a, b = rec.discrepancy.as_a_plus_b_over_pi()
    assert (a, b) == (F(0), F(16))
    assert rec.discrepancy == Sym.from_a_plus_b_over_pi(0, 16)


def test_the_claim_understates_the_truth():
    """Exact sign of the error, using 223/71 < pi < 22/7 so no float decides."""
    rec = refutation_record()
    a_c, b_c = rec.claimed.as_a_plus_b_over_pi()
    a_t, b_t = rec.derived.as_a_plus_b_over_pi()
    pi_lo, pi_hi = F(223, 71), F(22, 7)
    claimed_max = a_c + b_c / pi_lo
    true_min = a_t + b_t / pi_hi
    assert claimed_max < true_min


def test_the_two_cross_terms_are_equal_and_each_is_the_discrepancy():
    rec = refutation_record()
    terms = binomial_terms(4)
    assert len(terms) == 5
    cross = [t for t in terms if t["exponents"] in ((3, 1), (1, 3))]
    assert len(cross) == 2
    assert cross[0]["value"] == cross[1]["value"] == rec.discrepancy


def test_structural_cause_is_named_and_is_a_dropped_cross_term():
    rec = refutation_record()
    assert "cross term" in rec.structural_cause
    assert "dropped" in rec.structural_cause.lower()


def test_every_record_states_what_it_does_not_establish():
    for rec in [refutation_record()] + sibling_records():
        assert rec.does_not_establish, f"{rec.key} states no limits"
        assert all(isinstance(x, str) and x for x in rec.does_not_establish)


# ===========================================================================
# 4. NEGATIVE CONTROLS
#
# Each of these fails if the module is weakened in the named way. They are the
# deliverable, not decoration.
# ===========================================================================

def _sum_with_terms_dropped(k, drop):
    """Recompute E(|xi|+|eta|)^k omitting the exponent tuples in `drop`."""
    total = Sym()
    for t in sum_abs_power_terms(2, k):
        if t["exponents"] in drop:
            continue
        total = total + t["value"]
    return total


def test_NEGATIVE_CONTROL_dropping_one_cross_term_reproduces_the_false_claim():
    """If a cross term is dropped, the result IS the refuted value.

    This is the control that fires when someone "fixes" the expansion by
    reintroducing the original defect: the wounded sum must equal the claim, and
    must NOT equal the true value.
    """
    wounded = _sum_with_terms_dropped(4, {(3, 1)})
    assert wounded == CLAIMED_FALSE_IDENTITY
    assert wounded == Sym.from_a_plus_b_over_pi(12, 16)
    assert wounded != sum_abs_power_moment(2, 4)
    # Symmetrically for the other cross term.
    assert _sum_with_terms_dropped(4, {(1, 3)}) == CLAIMED_FALSE_IDENTITY


def test_NEGATIVE_CONTROL_dropping_any_single_term_changes_the_value():
    """No term of the k=4 expansion is inert. Silent term loss cannot pass."""
    full = sum_abs_power_moment(2, 4)
    for t in binomial_terms(4):
        assert _sum_with_terms_dropped(4, {t["exponents"]}) != full


def test_NEGATIVE_CONTROL_swapped_basis_coefficients_are_rejected():
    """(a, b) = (12, 32) must not be interchangeable with (32, 12)."""
    true_value = sum_abs_power_moment(2, 4)
    a, b = true_value.as_a_plus_b_over_pi()
    swapped = Sym.from_a_plus_b_over_pi(b, a)
    assert swapped != true_value
    assert swapped.as_a_plus_b_over_pi() == (F(32), F(12))
    # And the claim's own coefficients, swapped, are also not the truth.
    ca, cb = CLAIMED_FALSE_IDENTITY.as_a_plus_b_over_pi()
    assert Sym.from_a_plus_b_over_pi(cb, ca) != true_value


def test_NEGATIVE_CONTROL_discrepancy_is_not_32_over_pi():
    """Stating the discrepancy as 32/pi (the whole cross pair) must fail."""
    rec = refutation_record()
    assert rec.discrepancy != Sym.from_a_plus_b_over_pi(0, 32)
    assert rec.discrepancy.as_a_plus_b_over_pi() != (F(0), F(32))
    # 32/pi would put the "true" value at 12 + 48/pi, which is not the truth.
    assert rec.claimed + Sym.from_a_plus_b_over_pi(0, 32) != rec.derived


def test_NEGATIVE_CONTROL_sign_flip_of_the_discrepancy_is_rejected():
    """claimed - true is -16/pi, not +16/pi. The direction is load-bearing."""
    rec = refutation_record()
    reversed_ = rec.claimed - rec.derived
    assert reversed_ != rec.discrepancy
    assert reversed_ == Sym.from_a_plus_b_over_pi(0, -16)


def test_NEGATIVE_CONTROL_odd_moments_must_not_be_rational():
    """If E|X|^1 or E|X|^3 were stored at s-power 0, k=4 would go wrong.

    Simulate the weakening and check the value moves.
    """
    fake_m1, fake_m3 = Sym.rational(1), Sym.rational(2)
    m0, m2, m4 = abs_moment_sym(0), abs_moment_sym(2), abs_moment_sym(4)
    fake = (Sym.rational(1) * m0 * m4
            + Sym.rational(4) * fake_m1 * fake_m3
            + Sym.rational(6) * m2 * m2
            + Sym.rational(4) * fake_m3 * fake_m1
            + Sym.rational(1) * m4 * m0)
    assert fake != sum_abs_power_moment(2, 4)
    assert fake == Sym.rational(28)        # 3 + 8 + 6 + 8 + 3, all rational


def test_NEGATIVE_CONTROL_basis_view_must_not_swallow_an_s1_term():
    """as_a_plus_b_over_pi must REFUSE odd powers rather than drop them."""
    contaminated = Sym({0: F(12), 1: F(1), 2: F(16)})
    with pytest.raises(ValueError):
        contaminated.as_a_plus_b_over_pi()


def test_NEGATIVE_CONTROL_sym_equality_is_not_float_equality():
    """Two Syms with equal floats but different exact forms must compare unequal.

    12 + 32/pi and its 30-digit decimal truncation agree to every digit a float
    can see. Exact arithmetic must still separate them.
    """
    exact = Sym.from_a_plus_b_over_pi(12, 32)
    decimal_lookalike = Sym.rational(F("22.185916357881301584"))
    assert exact != decimal_lookalike
    assert abs(numeric_oracle(exact) - numeric_oracle(decimal_lookalike)) < 1e-9


# ===========================================================================
# 5. Sibling identities found in the corpus audit
# ===========================================================================

def test_rayleigh_fourth_moment_is_exactly_8():
    assert rayleigh_even_moment(2) == F(8)
    assert rayleigh_even_moment(1) == F(2)
    assert rayleigh_even_moment(3) == F(48)
    rec = sibling_rayleigh_fourth_moment()
    assert rec.verdict == "CONFIRMED_EXACT"
    assert rec.derived == F(8)
    assert rec.discrepancy == F(0)


def test_NEGATIVE_CONTROL_rayleigh_bracket_does_not_detect_the_false_value():
    """8 <= E(|xi|+|eta|)^4 <= 32 holds for BOTH 12+16/pi and 12+32/pi.

    Recorded so nobody mistakes the corpus's rho^4 comparison for a check that
    would have caught the defect. Exact, via 223/71 < pi < 22/7.
    """
    pi_lo, pi_hi = F(223, 71), F(22, 7)
    lo, hi = rayleigh_even_moment(2), 4 * rayleigh_even_moment(2)
    assert (lo, hi) == (F(8), F(32))
    for value in (CLAIMED_FALSE_IDENTITY, sum_abs_power_moment(2, 4)):
        a, b = value.as_a_plus_b_over_pi()
        assert a + b / pi_hi >= lo     # lower bound on the value clears 8
        assert a + b / pi_lo <= hi     # upper bound on the value stays under 32


def test_truncated_normal_second_moment_identity_is_confirmed_exactly():
    rec = sibling_truncated_second_moment()
    assert rec.verdict == "CONFIRMED_EXACT"
    b = F(6, 5)
    big, small = truncated_normal_moment(b, 2)
    assert big == QSqrt2(b * b + 2, F(0))          # (b^2 + 2) Phi
    assert small == QSqrt2(F(0), b)                # sqrt(2) b phi


def test_truncated_normal_moment_recursion_low_orders():
    """T_0 = Phi, T_1 = -phi, T_2 = Phi - c phi -- checked through k = 0, 1."""
    b = F(6, 5)
    assert truncated_normal_moment(b, 0) == (QSqrt2(F(1), F(0)), QSqrt2())
    big, small = truncated_normal_moment(b, 1)
    # E[Q 1{Q<0}] = -b Phi(c) - sqrt(2) phi(c)
    assert big == QSqrt2(-b, F(0))
    assert small == QSqrt2(F(0), F(-1))


def test_NEGATIVE_CONTROL_truncated_second_moment_phi_coefficient_is_not_doubled():
    """The phi coefficient is sqrt(2) b, NOT 2 sqrt(2) b.

    2 sqrt(2) b is what you get by keeping 2 sqrt(2) b phi and forgetting the
    -2 c phi contribution from T_2 -- the exact analogue of the dropped cross
    term in the refuted identity. This control fires if that happens.
    """
    b = F(6, 5)
    _, small = truncated_normal_moment(b, 2)
    assert small == QSqrt2(F(0), b)
    assert small != QSqrt2(F(0), 2 * b)


def test_truncated_fourth_moment_closed_form_shape_and_honest_verdict():
    b = F(6, 5)
    big, small = truncated_normal_moment(b, 4)
    assert big == QSqrt2(b ** 4 + 12 * b * b + 12, F(0))
    assert small == QSqrt2(F(0), b * (b * b + 10))
    rec = sibling_truncated_fourth_moment()
    # The register gives only a decimal, so the verdict must NOT claim closure.
    assert rec.verdict == "NOT_SETTLED_IDENTIFICATION_INFERRED"
    assert "INFERENCE" in " ".join(rec.notes).upper()


def test_found_not_settled_is_recorded_with_reasons():
    assert len(FOUND_NOT_SETTLED) >= 5
    for item in FOUND_NOT_SETTLED:
        assert item["expression"] and item["source"] and item["why_not_settled"]


def test_no_sibling_record_promotes_anything():
    """Verdict vocabulary is closed. Nothing may say CLOSED, DISCHARGED, etc."""
    forbidden = ("CLOSED", "DISCHARGED", "PROMOTED", "CERTIFIED", "RESOLVED")
    allowed = {"REFUTED", "CONFIRMED_EXACT", "NOT_SETTLED_IDENTIFICATION_INFERRED"}
    for rec in [refutation_record()] + sibling_records():
        assert rec.verdict in allowed
        assert not any(f in rec.verdict for f in forbidden)


# ===========================================================================
# 6. The NON-CERTIFYING oracle, labelled as such
# ===========================================================================

def test_gamma_oracle_agrees_NON_CERTIFYING():
    """Float cross-check only. Agreement here proves nothing; it catches typos."""
    for row in oracle_report(8):
        scale = max(1.0, abs(row["gamma_oracle_NON_CERTIFYING"]))
        assert row["abs_diff_NON_CERTIFYING"] / scale < 1e-12


def test_gamma_oracle_reproduces_the_true_decimal_NON_CERTIFYING():
    exact = sum_abs_power_moment(2, 4)
    assert abs(numeric_oracle(exact) - (12 + 32 / math.pi)) < 1e-12
    assert abs(numeric_oracle(exact) - 22.18591635788) < 1e-10
    # And is NOT the claimed 12 + 16/pi.
    assert abs(numeric_oracle(exact) - (12 + 16 / math.pi)) > 5.0


def test_oracle_is_labelled_non_certifying_everywhere_it_is_returned():
    keys = set(oracle_report(2)[0])
    assert {"exact_as_float_NON_CERTIFYING", "gamma_oracle_NON_CERTIFYING",
            "abs_diff_NON_CERTIFYING"} <= keys
    assert "NON-CERTIFYING" in (numeric_abs_moment.__doc__ or "")
    assert "NON-CERTIFYING" in (numeric_oracle.__doc__ or "")


def test_no_verdict_depends_on_a_float():
    """Every record's exact fields are Fraction/Sym/QSqrt2 -- never float."""
    def _clean(x):
        if isinstance(x, float):
            return False
        if isinstance(x, (tuple, list)):
            return all(_clean(i) for i in x)
        return True
    for rec in [refutation_record()] + sibling_records():
        assert _clean(rec.claimed) and _clean(rec.derived)
        assert _clean(rec.discrepancy)
