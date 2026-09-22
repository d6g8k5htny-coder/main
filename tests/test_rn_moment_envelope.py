"""Regression test for the RN5 near-region determinant-moment defect.

The defect: the pinned ``d3_perc.py`` helper ``envelope_v`` used the fourth
determinant moment of ``C = det H_y`` where Hölder(4, 4, 2) requires the second.
RN5 (2026-09-17) disproved the claimed universal inequality with an exact typed
Gaussian counterexample.

These tests re-derive that counterexample here, from the moment formulas rather
than from RN5's reported numbers, and assert:

  * the defective expression is **not** an upper bound (it is strictly below a
    proven lower bound for the typed expectation);
  * the correct Hölder(4, 4, 2) expression is consistent with that same lower
    bound;
  * the exact rational margins RN5 published are reproduced
    (old envelope < 63/1000 < 207/1000 < typed expectation);
  * the reported decimal values match.

All comparisons are exact rational comparisons of fourth powers, so no
floating-point rounding enters the verdict.

This test verifies the counterexample. It does not close Piece 2 of
D3-LEMMA-RN-UNIF, and it says nothing about RN3's far-region proof, which uses
the correct second moment.
"""
from fractions import Fraction as F
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "research", "rn"))

from moment_envelope import (  # noqa: E402
    counterexample, det_moment, envelope_correct_pow4, envelope_defective_pow4,
    normal_moment,
)


def test_normal_moments_are_exact():
    mu, var = F(-1), F(1, 10**6)
    assert normal_moment(mu, var, 1) == mu
    assert normal_moment(mu, var, 2) == mu * mu + var
    assert normal_moment(mu, var, 3) == mu**3 + 3 * mu * var
    assert normal_moment(mu, var, 4) == mu**4 + 6 * mu * mu * var + 3 * var * var


def test_det_second_moment_matches_direct_expansion():
    """E[(uv - w^2)^2] expanded independently of det_moment."""
    a, b, var = F(1), F(-1, 4), F(1, 10**6)
    direct = ((a * a + var) * (b * b + var) - 2 * a * b * var + 3 * var**2)
    assert det_moment(a, b, var, 2) == direct


def test_defective_expression_is_not_an_upper_bound():
    c = counterexample()
    # Exact comparison of fourth powers; both quantities are positive.
    assert c["defective_pow4"] < c["typed_expectation_lower"] ** 4


def test_correct_holder_expression_is_consistent_with_the_lower_bound():
    c = counterexample()
    assert c["correct_pow4"] >= c["typed_expectation_lower"] ** 4


def test_published_rational_margins():
    """RN5: old envelope < 63/1000 < 207/1000 < typed expectation."""
    c = counterexample()
    assert c["defective_pow4"] < F(63, 1000) ** 4
    assert F(63, 1000) < F(207, 1000)
    assert c["typed_expectation_lower"] > F(207, 1000)


def test_published_decimal_values():
    c = counterexample()
    assert c["typed_expectation_lower"] == F(207675035568, 10**12)
    # NON-CERTIFYING. The verdict of this file rests on the exact rational
    # comparisons above; these two lines only check that the DECIMALS RN5
    # published round-trip, and a float agreement is not evidence of a bound.
    assert abs(float(c["defective_pow4"]) ** 0.25 - 0.0625040624) < 1e-9
    assert abs(float(c["correct_pow4"]) ** 0.25 - 0.2500046250) < 1e-9


def test_event_probability_is_the_union_chebyshev_bound():
    c = counterexample()
    # nine coordinates, each failing with probability <= var / delta^2
    assert c["prob_event"] == 1 - 9 * (c["var"] / c["delta"] ** 2)
    assert c["prob_event"] == F(91, 100)


def test_determinant_lower_bounds_on_the_event():
    c = counterexample()
    assert c["det_lower"]["A"] == F(98, 100)
    assert c["det_lower"]["B"] == F(9801, 10000)
    assert c["det_lower"]["C"] == F(2376, 10000)


def test_the_two_expressions_differ_only_in_the_C_moment():
    """Guards against a future edit that silently swaps EC2 back to EC4."""
    c = counterexample()
    assert envelope_correct_pow4(c["EA4"], c["EB4"], c["EC2"]) == c["correct_pow4"]
    assert envelope_defective_pow4(c["EA4"], c["EB4"], c["EC4"]) == c["defective_pow4"]
    assert c["EC2"] != c["EC4"]
