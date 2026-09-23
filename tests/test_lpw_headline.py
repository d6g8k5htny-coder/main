"""End-to-end headline tests for the LPW constant (docs/OPEN_PROBLEMS.md §C).

Every assertion below is an exact rational comparison. No float, no ``decimal``,
no tolerance: the verdicts are arithmetic, not estimates.

What is tested:

  * the exact fraction ``260/(3790446482793*2^40*10^21)`` reduces to
    ``13/208381999114677270242918400000000000000000000``, pinned against an
    independently written literal so that perturbing any named factor fails;
  * the delivered headline ``6.239e-44`` is strictly ABOVE the exact fraction
    and is simultaneously its correctly-rounded 4-significant-digit value;
  * the proposed repair ``6.238e-44`` is strictly BELOW it;
  * the admissibility rule — rounded away from the asserted inequality — gives
    opposite verdicts for the two decimals in each direction, and admits
    nothing at all under ``equality``.

NEGATIVE CONTROLS (the deliverable, not decoration):

  * ``test_negative_control_comparison_flip`` fails if the ``lower`` branch's
    comparison is flipped to ``>=`` (or the ``upper`` branch to ``<=``);
  * ``test_negative_control_perturbed_factors`` fails if any named factor of
    the exact fraction is perturbed, including by one unit in the B3 ceiling;
  * ``test_negative_control_wrong_side_rejected`` fails if ``6.238e-44`` is
    ever admitted as an upper bound or ``6.239e-44`` as a lower bound.

These tests do NOT verify the derivation that produced the exact fraction, do
NOT close or review the R05 Rayleigh amplitude lemma, the tails/profile modulus,
the conditional bounds or the actual headline mutation, and do NOT admit or
promote any certificate. Green here is not a mathematical review.
"""
from fractions import Fraction as F
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "research", "lpw"))

from headline import (  # noqa: E402
    B3_CEILING, DELIVERED_HEADLINE, DIRECTIONS, NUMERATOR, PROPOSED_REPAIR,
    TEN_POWER, TWO_POWER, admissible_as, decimal_digits, decimal_expansion,
    exact_value, has_terminating_decimal, parse_decimal, relative_offset,
    report,
)

# Written out independently of the module's factor constants, on purpose.
EXACT_REDUCED = F(13, 208381999114677270242918400000000000000000000)


# --------------------------------------------------------------------------
# the exact value
# --------------------------------------------------------------------------

def test_exact_value_is_the_named_fraction():
    assert exact_value() == F(260, 3790446482793 * 2**40 * 10**21)
    assert exact_value() == EXACT_REDUCED
    assert exact_value().numerator == 13
    assert exact_value().denominator == 208381999114677270242918400000000000000000000


def test_named_factors_are_the_registers_factors():
    assert (NUMERATOR, B3_CEILING, TWO_POWER, TEN_POWER) == (260, 3790446482793, 40, 21)


def test_exact_value_brackets_both_decimals_strictly():
    exact = exact_value()
    assert parse_decimal(PROPOSED_REPAIR) < exact < parse_decimal(DELIVERED_HEADLINE)


def test_decimal_expansion_is_not_terminating():
    assert has_terminating_decimal() is False
    # the reduced denominator carries the B3 ceiling, which is neither 2 nor 5
    assert exact_value().denominator % B3_CEILING == 0
    assert decimal_expansion(places=40).startswith("6.238542702935587792943041084518910355631")


# --------------------------------------------------------------------------
# correct rounding, and the offsets
# --------------------------------------------------------------------------

def test_four_significant_digits_is_the_delivered_headline_and_moved_up():
    rd = decimal_digits(4)
    assert rd.literal == DELIVERED_HEADLINE
    assert rd.direction == "up"
    assert rd.value == parse_decimal(DELIVERED_HEADLINE)
    assert rd.value > exact_value()


def test_the_proposed_repair_is_not_the_correct_four_digit_rounding():
    assert decimal_digits(4).literal != PROPOSED_REPAIR
    assert parse_decimal(PROPOSED_REPAIR) < exact_value()


def test_relative_offsets_match_the_published_figures():
    excess = relative_offset(DELIVERED_HEADLINE)
    deficit = relative_offset(PROPOSED_REPAIR)
    assert excess > 0 and deficit < 0
    assert F("7.3301e-5") < excess < F("7.3302e-5")       # +7.330e-05
    assert F("-8.6992e-5") < deficit < F("-8.6991e-5")    # -8.699e-05


def test_rounding_directions_alternate_as_computed():
    assert [decimal_digits(n).direction for n in range(2, 9)] == [
        "down", "up", "up", "down", "down", "up", "down"]


def test_rounding_is_correct_rounding_at_every_width():
    """Each returned decimal is the closest n-digit decimal to the exact value."""
    exact = exact_value()
    for n in range(1, 13):
        rd = decimal_digits(n)
        unit = F(10) ** (_exponent_of(rd.value, n))
        assert abs(rd.value - exact) <= unit / 2
        assert abs(rd.value + unit - exact) >= abs(rd.value - exact)
        assert abs(rd.value - unit - exact) >= abs(rd.value - exact)


def _exponent_of(value: F, digits: int) -> int:
    """The power of ten of the last significant digit of ``value``."""
    k = 0
    scaled = value
    while scaled >= F(10) ** digits:
        scaled /= 10
        k += 1
    while scaled < F(10) ** (digits - 1):
        scaled *= 10
        k -= 1
    return k


# --------------------------------------------------------------------------
# the admissibility rule
# --------------------------------------------------------------------------

def test_delivered_headline_is_admissible_only_as_an_upper_bound():
    ok_upper, why_upper = admissible_as(DELIVERED_HEADLINE, "upper")
    ok_lower, _ = admissible_as(DELIVERED_HEADLINE, "lower")
    ok_eq, _ = admissible_as(DELIVERED_HEADLINE, "equality")
    assert ok_upper is True and "above" in why_upper
    assert ok_lower is False
    assert ok_eq is False


def test_proposed_repair_is_admissible_only_as_a_lower_bound():
    ok_upper, _ = admissible_as(PROPOSED_REPAIR, "upper")
    ok_lower, why_lower = admissible_as(PROPOSED_REPAIR, "lower")
    ok_eq, _ = admissible_as(PROPOSED_REPAIR, "equality")
    assert ok_upper is False
    assert ok_lower is True and "below" in why_lower
    assert ok_eq is False


def test_equality_admits_nothing_rounded_and_says_so():
    for lit in (DELIVERED_HEADLINE, PROPOSED_REPAIR, "6.2385427e-44", "6e-44"):
        ok, reason = admissible_as(lit, "equality")
        assert ok is False
        assert "nothing rounded is admissible" in reason
        assert "does not terminate" in reason


def test_the_exact_value_itself_is_admissible_in_every_direction():
    for direction in DIRECTIONS:
        ok, _ = admissible_as(exact_value(), direction)
        assert ok is True


def test_unknown_direction_is_rejected():
    with pytest.raises(ValueError):
        admissible_as(DELIVERED_HEADLINE, "two-sided")


def test_float_input_is_rejected():
    with pytest.raises(TypeError):
        parse_decimal(6.239e-44)


def test_report_runs(capsys):
    """Smoke only. A display is not a certificate and admits nothing."""
    report()
    out = capsys.readouterr().out
    assert "NOT ESTABLISHED" in out
    assert "6.239e-44" in out and "6.238e-44" in out


# --------------------------------------------------------------------------
# NEGATIVE CONTROLS
# --------------------------------------------------------------------------

def _flipped_admissible(literal, direction):
    """The mutant: the same rule with both comparisons flipped."""
    d = parse_decimal(literal)
    exact = exact_value()
    if direction == "upper":
        return d <= exact
    if direction == "lower":
        return d >= exact
    return d == exact


def test_negative_control_comparison_flip():
    """Fails if ``admissible_as`` flips its comparison (``<=`` <-> ``>=``).

    Both decimals in play are strict witnesses: neither equals the exact value,
    so the flipped rule must disagree with the real one on every one of the
    four bound verdicts.
    """
    disagreements = 0
    for lit in (DELIVERED_HEADLINE, PROPOSED_REPAIR):
        for direction in ("upper", "lower"):
            real, _ = admissible_as(lit, direction)
            if real != _flipped_admissible(lit, direction):
                disagreements += 1
    assert disagreements == 4, (
        "the flipped-comparison mutant agrees with the module somewhere: the "
        "witnesses no longer discriminate direction")


def test_negative_control_wrong_side_rejected():
    """Fails if the wrong-side decimal is ever admitted.

    ``6.238e-44`` is below the exact fraction, so as an UPPER bound it asserts
    something stronger than the exact chain supports; ``6.239e-44`` is above it,
    so as a LOWER bound it does the same.
    """
    ok, reason = admissible_as(PROPOSED_REPAIR, "upper")
    assert ok is False
    assert "STRONGER than the exact chain supports" in reason

    ok, reason = admissible_as(DELIVERED_HEADLINE, "lower")
    assert ok is False
    assert "STRONGER than the exact chain supports" in reason


def _admissible_against(value: F, literal, direction: str) -> bool:
    """The admissibility rule evaluated against an arbitrary ``value``."""
    d = parse_decimal(literal)
    if direction == "upper":
        return d >= value
    if direction == "lower":
        return d <= value
    return d == value


# Built from the register's own factor values written out here, NOT from the
# module's constants: if the module's factors are edited, its exact_value()
# lands on one of these perturbations and the control below fires.
_N, _B3, _P2, _P10 = 260, 3790446482793, 40, 21

PERTURBATIONS = [
    ("numerator +1", F(_N + 1, _B3 * 2**_P2 * 10**_P10)),
    ("numerator -1", F(_N - 1, _B3 * 2**_P2 * 10**_P10)),
    ("B3 ceiling +1", F(_N, (_B3 + 1) * 2**_P2 * 10**_P10)),
    ("B3 ceiling -1", F(_N, (_B3 - 1) * 2**_P2 * 10**_P10)),
    ("2^39", F(_N, _B3 * 2**(_P2 - 1) * 10**_P10)),
    ("2^41", F(_N, _B3 * 2**(_P2 + 1) * 10**_P10)),
    ("10^20", F(_N, _B3 * 2**_P2 * 10**(_P10 - 1))),
    ("10^22", F(_N, _B3 * 2**_P2 * 10**(_P10 + 1))),
]


@pytest.mark.parametrize("name,perturbed", PERTURBATIONS)
def test_negative_control_perturbed_factors(name, perturbed):
    """Fails if any named factor of the exact fraction is perturbed.

    Two ways, both exact. First the value itself must differ from the module's.
    Second — so that the check is not merely a restatement of the literal — the
    midpoint between the true and the perturbed value is a decisive witness:
    it lies on one side of the true value and the other side of the perturbed
    one, so the two rules must return opposite upper-bound verdicts for it.
    """
    exact = exact_value()
    assert perturbed != exact, f"perturbation {name} did not change the value"

    witness = (exact + perturbed) / 2
    real, _ = admissible_as(witness, "upper")
    mutant = _admissible_against(perturbed, witness, "upper")
    assert real != mutant, (
        f"perturbation {name} is invisible to the admissibility rule")


def test_negative_control_perturbed_factors_change_the_headline_digits():
    """A coarse perturbation must also move the 4-significant-digit headline."""
    coarse = F(_N, _B3 * 2**_P2 * 10**(_P10 - 1))
    assert decimal_digits(4, coarse).literal != decimal_digits(4).literal
