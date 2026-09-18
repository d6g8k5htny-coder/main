"""Tests for the bound-slack registry in ``research/slack/``.

WHAT IS UNDER TEST. ``research/slack`` records, for each bound the corpus
states both a claimed value and a true/attained value for, the exact ratio
between them and an order-of-magnitude classification of it. Its one hard rule
is that a slack ratio measures a bound's UTILITY and never its CORRECTNESS,
and that rule is enforced in code rather than asserted in prose.

Organised in five groups:

1. Exact slack-ratio arithmetic on the ``chi2_grad_bound`` headline case,
   reproducing the source's own "~1e19 slack" from the two quoted numbers.
2. The severity ladder, probed from both sides of every boundary.
3. Provenance, ratio-kind and display: no float anywhere, and the tightest
   record must not print as a flat ``1.00e+0``.
4. **NEGATIVE CONTROLS.** Five deliberately-wrong records, each asserted to be
   REFUSED at construction. These are the point of the file. Every one was run
   against a deliberately broken copy of ``research/slack/registry.py`` and
   confirmed to stop failing there; each names its mutation in its docstring.
5. Registry invariants and reproduction of the sources' own reported figures.

WHAT THESE TESTS DO NOT ESTABLISH
---------------------------------
* A green build is not a mathematical review. These tests check arithmetic and
  guards, not any bound's derivation.
* They close, discharge, promote and reclassify nothing. OBL-H5-JETMOD,
  OBL-H5-ZBAND (hi), OBL-H5-REMOTE-THRESHOLD, OBL-D1-PROMOTE (chart) and both
  pieces of D3-LEMMA-RN-UNIF stay OPEN.
* Reproducing a number a document reports is not certifying it. Most of the
  registry's inputs are mpmath floats, finite differences, candidate constants
  or quoted digit strings; an exact ratio between uncertified inputs is an
  exact ratio of uncertified inputs.
* Original prize problems solved: 0.
"""
import os
import sys
from decimal import Decimal
from fractions import Fraction as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pytest  # noqa: E402

from research.interval import Interval  # noqa: E402
from research.slack import registry as S  # noqa: E402
from research.slack.registry import (  # noqa: E402
    AttainedKind, BoundDirection, Provenance, RatioKind, Severity, SlackRecord,
    SoundnessWitness,
)


def _find(name_fragment):
    for r in S.REGISTRY:
        if name_fragment in r.bound_name:
            return r
    raise AssertionError(f"no registry record matching {name_fragment!r}")


def _sound_record(**over):
    """A minimal well-formed record, for mutation by the negative controls."""
    kw = dict(
        bound_name="test bound",
        carrier="a carrier",
        source_document="a source document",
        source_quote="a quote",
        direction=BoundDirection.UPPER,
        claimed=Interval("10"),
        claimed_provenance=Provenance.EXACT_RATIONAL,
        attained=Interval("1"),
        attained_provenance=Provenance.EXACT_RATIONAL,
        attained_kind=AttainedKind.ENCLOSURE_OF_TRUE,
    )
    kw.update(over)
    return SlackRecord(**kw)


# ---------------------------------------------------------------------------
# 1. exact slack-ratio arithmetic on the headline case
# ---------------------------------------------------------------------------

def test_chi2_grad_bound_quoted_values_are_stored_as_rounding_intervals():
    """"~1.57e14" means [1.565e14, 1.575e14], not the point 1.57e14.

    Storing a prose quotation as a point value would claim a precision the
    document does not offer. The rounding interval encloses what the document
    reported; ``QUOTED_FROM_DOCUMENT`` says that is all it encloses.
    """
    r = _find("chi2_grad_bound")
    assert r.claimed.lo == F("1.565e14")
    assert r.claimed.hi == F("1.575e14")
    assert r.attained.lo == F("1.5625e-5")
    assert r.attained.hi == F("1.5635e-5")
    assert r.claimed_provenance is Provenance.QUOTED_FROM_DOCUMENT
    assert r.attained_provenance is Provenance.QUOTED_FROM_DOCUMENT


def test_chi2_grad_bound_slack_ratio_is_exact_and_reproduces_the_source():
    """The ratio is exact rational, and it reproduces CL-RNU-001's "~1e19".

    claimed / attained over the two rounding intervals, with no float anywhere.
    """
    r = _find("chi2_grad_bound")
    ratio = r.slack_ratio()
    assert ratio.lo == F("1.565e14") / F("1.5635e-5")
    assert ratio.hi == F("1.575e14") / F("1.5625e-5")
    # exact endpoints as integers/rationals, spelled out
    assert ratio.hi == F(1008, 10) * 10**17
    assert isinstance(ratio.lo, F) and isinstance(ratio.hi, F)
    # the source says ~1e19; the whole enclosure has order of magnitude 19
    assert r.order_of_magnitude() == (19, 19)
    assert ratio.lo > 10**19 and ratio.hi < 1.1 * 10**19


def test_chi2_grad_bound_is_unusable_but_not_unsound():
    """The headline record must be UNUSABLE_AS_STATED and NOT known_unsound.

    This is the rule the whole module exists for. A bound nineteen orders of
    magnitude loose is useless; nothing in the corpus exhibits it on the wrong
    side of the quantity, so this registry says nothing about its correctness.
    """
    r = _find("chi2_grad_bound")
    assert r.severity() is Severity.UNUSABLE_AS_STATED
    assert r.severity_is_sharp()
    assert r.known_unsound is False
    assert r.unsound_witness is None


def test_chi2_grad_bound_ratio_is_indicative_only():
    """Both inputs are quoted digits, so the ratio has no certified relation to
    the true slack, and the record must say so rather than implying otherwise."""
    r = _find("chi2_grad_bound")
    assert r.attained_kind is AttainedKind.QUOTED_POINT_ESTIMATE
    assert r.ratio_kind() is RatioKind.INDICATIVE_ONLY
    assert r.certifying() is False


def test_tau_and_t4_reproduce_their_sources_reported_factors():
    """The engine's own "x5.3e3" and CL-RNU-003's 1.39e7-over-35."""
    tau = _find("tau entrywise envelope")
    # the engine header's "x5.3e3" is a two-significant-figure rendering of the
    # ratio of the two numbers in the same parenthesis; both endpoints render
    # to it, so the enclosure reproduces the reported factor exactly.
    assert S.sci(tau.slack_ratio().lo, 2) == "5.3e+3"
    assert S.sci(tau.slack_ratio().hi, 2) == "5.3e+3"
    assert tau.slack_ratio().lo == F("1.8805") / F("3.535e-4")
    assert tau.slack_ratio().hi == F("1.8815") / F("3.525e-4")
    assert tau.severity() is Severity.SEVERE

    t4 = _find("T4_kap(5)")
    lo, hi = t4.slack_ratio().lo, t4.slack_ratio().hi
    assert lo == F("1.385e7") / F("35.5")
    assert hi == F("1.395e7") / F("34.5")
    assert t4.severity() is Severity.SEVERE
    assert t4.claimed_provenance is Provenance.CANDIDATE_CONSTANT
    assert t4.attained_provenance is Provenance.FINITE_DIFFERENCE


def test_lower_direction_ratio_is_inverted():
    """For a LOWER bound, slack is attained/claimed, not claimed/attained.

    The H3 rung floor reproduces its source's "+92.1%" margin, which only comes
    out if the division goes the right way round.
    """
    z = _find("c_Z * r^2")
    assert z.direction is BoundDirection.LOWER
    num = S.rounding_interval("7.7592917375327855e-3")
    den = S.rounding_interval("4.0387231691e-3")
    assert z.slack_ratio().lo == num.lo / den.hi     # attained / claimed
    assert z.slack_ratio().hi == num.hi / den.lo
    pct = (z.slack_ratio().lo - 1) * 100
    assert F("92.0") < pct < F("92.2")          # source: "+92.1%"
    assert z.severity() is Severity.TIGHT
    assert z.ratio_kind() is RatioKind.LOWER_BOUND_ON_TRUE_SLACK


def test_rn3_far_count_ratio_is_an_upper_bound_on_the_true_slack():
    """Dividing by a certified LOWER bound overstates an UPPER bound's slack.

    ``2.22542 r^3 < I_far < 2.83312 r^3``: the true slack lies in [1, 1.2731],
    and the record must classify its ratio as an upper bound on it rather than
    as an enclosure.
    """
    r = _find("I_far upper")
    assert r.ratio_kind() is RatioKind.UPPER_BOUND_ON_TRUE_SLACK
    assert r.slack_ratio().hi < F(12731, 10000)
    assert r.slack_ratio().lo > 1
    assert r.severity() is Severity.TIGHT


# ---------------------------------------------------------------------------
# 2. the severity ladder
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("x, expected", [
    (F(0), Severity.VIOLATION_WITNESSED),
    (F(1, 2), Severity.VIOLATION_WITNESSED),
    (F(9999, 10000), Severity.VIOLATION_WITNESSED),
    (F(1), Severity.TIGHT),
    (F(19999, 10000), Severity.TIGHT),
    (F(2), Severity.ROUTINE),
    (F(99999, 10000), Severity.ROUTINE),
    (F(10), Severity.LOOSE),
    (F(999), Severity.LOOSE),
    (F(10**3), Severity.SEVERE),
    (F(10**6) - 1, Severity.SEVERE),
    (F(10**6), Severity.UNUSABLE_AS_STATED),
    (F(10**19), Severity.UNUSABLE_AS_STATED),
])
def test_severity_ladder_boundaries_from_both_sides(x, expected):
    """Bands are half-open on the right; every edge is probed at the edge and
    just below it."""
    assert S.severity_of(x) is expected


def test_severity_ladder_edges_match_the_documented_ladder():
    """The boundaries the docstring justifies are the boundaries in the code."""
    edges = [e for e, _ in S.SEVERITY_LADDER]
    assert edges == [F(0), F(1), F(2), F(10), F(10**3), F(10**6)]


def test_negative_ratio_is_refused_not_classified():
    with pytest.raises(ValueError, match="cannot be negative"):
        S.severity_of(F(-1))


def test_severity_is_classified_from_the_lower_endpoint():
    """A record straddling a boundary reports the LESS severe band, and says it
    straddles. Reporting the worse band would overstate the defect."""
    r = _sound_record(claimed=Interval("9", "11"), attained=Interval("1"))
    assert r.slack_ratio().lo == 9 and r.slack_ratio().hi == 11
    assert r.severity() is Severity.ROUTINE      # from lo = 9, not hi = 11
    assert r.severity_is_sharp() is False


def test_severity_never_reads_known_unsound():
    """Flipping soundness must not move severity. They are disjoint axes.

    Compared across the whole registry: each record's severity is exactly what
    ``severity_of`` gives for its ratio's lower endpoint, unsound or not.
    """
    for r in S.REGISTRY:
        assert r.severity() is S.severity_of(r.slack_ratio().lo)
    tight_unsound = _find("cone_slope_margin")
    loose_sound = _find("chi2_grad_bound")
    assert tight_unsound.known_unsound is True
    assert loose_sound.known_unsound is False
    # and the loose one is the one with the enormous ratio
    assert loose_sound.slack_ratio().lo > tight_unsound.slack_ratio().lo * 10**18


def test_order_of_magnitude_is_exact_integer_arithmetic():
    assert S.order_of_magnitude(F(1)) == 0
    assert S.order_of_magnitude(F(10)) == 1
    assert S.order_of_magnitude(F(1, 10)) == -1
    assert S.order_of_magnitude(F(99999, 10000)) == 0
    assert S.order_of_magnitude(F(10**19)) == 19
    with pytest.raises(ValueError):
        S.order_of_magnitude(F(0))


def test_sci_rounds_half_up_and_carries_across_the_decade():
    assert S.sci(F(1)) == "1.00e+0"
    assert S.sci(F("1.565e14")) == "1.57e+14"
    assert S.sci(F("9.999")) == "1.00e+1"        # carry
    assert S.sci(F("-2.5e-3")) == "-2.50e-3"


# ---------------------------------------------------------------------------
# 3. provenance, ratio kinds, display
# ---------------------------------------------------------------------------

def test_only_two_provenances_are_certifying():
    """mpmath at any precision, doubles, finite differences, candidates and
    quoted digits are all NON-CERTIFYING, exactly as the sources say of
    themselves."""
    cert = {p for p in Provenance if p.certifying}
    assert cert == {Provenance.CERTIFIED_INTERVAL, Provenance.EXACT_RATIONAL}
    assert Provenance.MPMATH_FLOAT.tag == "NON-CERT"
    assert Provenance.CERTIFIED_INTERVAL.tag == "CERT"


def test_ratio_kind_flips_with_direction():
    """A certified lower bound on the true value overstates an UPPER bound's
    slack and understates a LOWER bound's. Getting this backwards would turn a
    conservative statement into an overclaim."""
    up = _sound_record(direction=BoundDirection.UPPER,
                       attained_kind=AttainedKind.CERTIFIED_LOWER_BOUND_ON_TRUE)
    assert up.ratio_kind() is RatioKind.UPPER_BOUND_ON_TRUE_SLACK
    down = _sound_record(direction=BoundDirection.LOWER,
                         claimed=Interval("1"), attained=Interval("10"),
                         attained_kind=AttainedKind.CERTIFIED_LOWER_BOUND_ON_TRUE)
    assert down.ratio_kind() is RatioKind.LOWER_BOUND_ON_TRUE_SLACK


def test_cone_slope_margin_does_not_display_as_flat_one():
    """The tightest record must show its deviation from 1, with its sign.

    Three significant figures would print ``1.00e+0`` and hide the one thing
    about this record that matters: the ratio is BELOW one.
    """
    r = _find("cone_slope_margin")
    disp = r.ratio_display()
    assert disp.startswith("[1-") or disp.startswith("1-")
    assert "1.00e+0" not in disp
    assert r.slack_ratio().hi < 1


def test_cl_aud_202_overshoot_reproduces_from_the_double_not_the_digit_string():
    """CL-AUD-202's "+6.083e-20" comes out of the IEEE-754 double, not the
    printed decimal. Both readings are pinned so neither can be quietly
    dropped, and both overshoot, so the verdict does not depend on the choice.
    """
    exact = S.truncated_interval("0.00864436749429013482649377440506888676")
    stored_double = Interval(
        "0.00864436749429013488732476133691307040862739086151123046875")
    digit_string = Interval("0.0086443674942901349")

    # the audit's own figure, exactly
    d_double = stored_double.lo - exact.lo
    assert F("6.0830986931844e-20") < d_double < F("6.0830986931845e-20")
    # the digit string read as an exact decimal gives a different number
    d_digits = digit_string.lo - exact.lo
    assert F("7.3506e-20") < d_digits < F("7.3507e-20")
    # both overshoot: the verdict is unaffected by the reading
    assert d_double > 0 and d_digits > 0
    # and the registry stores the double
    assert _find("cone_slope_margin").claimed == stored_double


def test_rounding_and_truncated_intervals_differ_as_documented():
    """A rounded quotation is two-sided; a truncated expansion is one-sided."""
    assert S.rounding_interval("2.83312") == Interval("2.833115", "2.833125")
    t = S.truncated_interval("0.125")
    assert t.lo == F(1, 8) and t.hi == F(1, 8) + F(1, 1000)


# ---------------------------------------------------------------------------
# 4. NEGATIVE CONTROLS
# ---------------------------------------------------------------------------

def test_conflating_looseness_with_unsoundness_is_refused():
    """CONTROL 1. A loose bound marked unsound on the strength of its slack.

    The chi2_grad_bound numbers offered as a "witness": bound ~1.57e14,
    quantity ~1.563e-5, direction UPPER. A violation would need
    bound.hi < quantity.lo, i.e. 1.575e14 < 1.5625e-5, which is false by
    nineteen orders of magnitude -- the witness proves the OPPOSITE of a
    violation. The record must be refused.

    MUTATION THAT MAKES THIS STOP FAILING: delete the
    ``if not self.unsound_witness.violates(): raise`` branch in
    ``SlackRecord.__post_init__``. Run against that broken copy, this test
    stops raising and fails. Confirmed.
    """
    w = SoundnessWitness(
        source="fabricated",
        statement="the bound is nineteen orders of magnitude loose",
        direction=BoundDirection.UPPER,
        bound=S.rounding_interval("1.57e14"),
        quantity=S.rounding_interval("1.563e-5"),
        quantity_kind=AttainedKind.CERTIFIED_LOWER_BOUND_ON_TRUE,
    )
    assert w.violates() is False
    assert w.margin() < 0
    with pytest.raises(ValueError, match="LOOSENESS IS NOT UNSOUNDNESS"):
        _sound_record(
            bound_name="a very loose but true bound",
            claimed=S.rounding_interval("1.57e14"),
            attained=S.rounding_interval("1.563e-5"),
            attained_kind=AttainedKind.QUOTED_POINT_ESTIMATE,
            claimed_provenance=Provenance.QUOTED_FROM_DOCUMENT,
            attained_provenance=Provenance.QUOTED_FROM_DOCUMENT,
            known_unsound=True,
            unsound_witness=w,
        )


def test_unsound_without_a_witness_is_refused():
    """CONTROL 2. ``known_unsound=True`` with no witness at all.

    The only route to the unsound field is an exhibited, re-verified failure.
    A bare boolean is an opinion.

    MUTATION: delete the ``if self.unsound_witness is None: raise`` branch.
    Confirmed to make this test fail.
    """
    with pytest.raises(ValueError, match="requires a SoundnessWitness"):
        _sound_record(known_unsound=True)


def test_a_proved_violation_recorded_as_sound_is_refused():
    """CONTROL 3. The guard run the other way: a witness that DOES exhibit a
    violation, attached to a record claiming ``known_unsound=False``.

    A registry that can hide a proved violation is worse than no registry.

    MUTATION: delete the ``elif ... and self.unsound_witness.violates(): raise``
    branch. Confirmed to make this test fail.
    """
    w = SoundnessWitness(
        source="a real source",
        statement="bound 1 lies strictly below the quantity, which is >= 5",
        direction=BoundDirection.UPPER,
        bound=Interval("1"),
        quantity=Interval("5"),
        quantity_kind=AttainedKind.CERTIFIED_LOWER_BOUND_ON_TRUE,
    )
    assert w.violates() is True
    with pytest.raises(ValueError, match="may not be recorded silently"):
        _sound_record(known_unsound=False, unsound_witness=w)


def test_a_ratio_provably_below_one_must_carry_a_witness():
    """CONTROL 4. A record whose own certified numbers put the slack below 1,
    with ``known_unsound=False``.

    claimed=1, attained=5, direction UPPER, attained an ENCLOSURE of the true
    value: the ratio enclosure is [1/5, 1/5], wholly below one, which exhibits
    the bound on the wrong side. That may not be recorded as sound.

    MUTATION: delete the final ``if proves_violation and ratio.hi < 1: raise``
    block. Confirmed to make this test fail.
    """
    with pytest.raises(ValueError, match="wholly below 1"):
        _sound_record(claimed=Interval("1"), attained=Interval("5"))


def test_a_float_in_a_value_field_is_refused():
    """CONTROL 5. ``float`` in ``claimed`` or ``attained``.

    float('0.1') is not 1/10 and a bound recorded as a double is not the bound
    the document states. The refusal is this module's own, with its own
    message, over and above research.interval.to_fraction's.

    MUTATION: drop the ``isinstance(x, float)`` branch from ``_as_interval``.
    Confirmed: ``to_fraction`` then still refuses the float, so the record is
    still rejected -- a layered defence -- but with the interval library's
    message instead of this one, and ``pytest.raises(TypeError, match=...)``
    on the slack-specific wording fails. Both layers are asserted below.
    """
    with pytest.raises(TypeError, match="float is refused in a slack-record"):
        _sound_record(claimed=1.57e14)
    with pytest.raises(TypeError, match="float is refused in a slack-record"):
        _sound_record(attained=1.563e-5)
    # the layer underneath, asserted so the defence stays double
    with pytest.raises(TypeError, match="float endpoints are refused"):
        Interval(1.57e14)


def test_an_unsourced_record_is_refused():
    """CONTROL 6. Any of bound_name, carrier, source_document or source_quote
    empty or whitespace.

    A registry of other people's numbers that does not say whose they are is a
    rumour.

    MUTATION: drop the ``for fname in (...)`` emptiness loop from
    ``__post_init__``. Confirmed to make this test fail.
    """
    for fname in ("carrier", "source_document", "source_quote"):
        with pytest.raises(ValueError, match="unsourced slack record"):
            _sound_record(**{fname: ""})
        with pytest.raises(ValueError, match="unsourced slack record"):
            _sound_record(**{fname: "   "})
    with pytest.raises(ValueError, match="unsourced slack record"):
        _sound_record(bound_name="")


def test_a_witness_with_the_wrong_quantity_kind_cannot_prove_a_violation():
    """CONTROL 7. An UPPER-direction witness whose quantity is only a certified
    UPPER bound on the true value proves nothing.

    bound = 1, quantity <= 5 says nothing about whether the true value exceeds
    1. ``usable()`` must be False and ``violates()`` must be False, so the
    record is refused.

    MUTATION: make ``usable()`` return True unconditionally. Confirmed: the
    record is then accepted and this test fails.
    """
    w = SoundnessWitness(
        source="a source",
        statement="quantity is at most 5",
        direction=BoundDirection.UPPER,
        bound=Interval("1"),
        quantity=Interval("5"),
        quantity_kind=AttainedKind.CERTIFIED_UPPER_BOUND_ON_TRUE,
    )
    assert w.usable() is False
    assert w.violates() is False
    with pytest.raises(ValueError, match="LOOSENESS IS NOT UNSOUNDNESS"):
        _sound_record(
            claimed=Interval("1"), attained=Interval("5"),
            attained_kind=AttainedKind.CERTIFIED_UPPER_BOUND_ON_TRUE,
            known_unsound=True, unsound_witness=w,
        )


def test_a_witness_without_a_source_is_refused():
    """CONTROL 8. Unsoundness attributed to nobody.

    MUTATION: drop the source/statement emptiness checks from
    ``SoundnessWitness.__post_init__``. Confirmed to make this test fail.
    """
    kw = dict(direction=BoundDirection.UPPER, bound=Interval("1"),
              quantity=Interval("5"),
              quantity_kind=AttainedKind.CERTIFIED_LOWER_BOUND_ON_TRUE)
    with pytest.raises(ValueError, match="must name its source"):
        SoundnessWitness(source="  ", statement="x", **kw)
    with pytest.raises(ValueError, match="must state what it exhibits"):
        SoundnessWitness(source="x", statement="", **kw)


def test_audit_registry_catches_a_mutated_record():
    """CONTROL 9. ``audit_registry`` re-runs the guards over already-built
    records, so post-construction mutation is caught too.

    ``SlackRecord`` is a mutable dataclass by design (records carry long prose
    that is easier to amend in place), so a check that only ran at construction
    would be bypassable by assignment.

    MUTATION: make ``audit_registry`` return ``[]`` unconditionally. Confirmed
    to make this test fail.
    """
    assert S.audit_registry() == []
    r = _sound_record()
    r.known_unsound = True           # mutated after construction, no witness
    problems = S.audit_registry([r])
    assert problems and "unsound without a witness" in problems[0]

    r2 = _sound_record()
    r2.source_quote = ""
    assert any("unsourced" in p for p in S.audit_registry([r2]))


# ---------------------------------------------------------------------------
# 5. registry invariants and honest reporting
# ---------------------------------------------------------------------------

def test_registry_audits_clean():
    assert S.audit_registry() == []


def test_every_registry_value_is_an_exact_interval():
    for r in S.REGISTRY:
        for v in (r.claimed, r.attained):
            assert isinstance(v, Interval)
            assert isinstance(v.lo, F) and isinstance(v.hi, F)
            assert not isinstance(v.lo, float) and not isinstance(v.hi, float)


def test_every_registry_record_is_sourced_and_quoted():
    for r in S.REGISTRY:
        assert r.carrier.strip()
        assert r.source_document.strip()
        assert len(r.source_quote.strip()) > 20


def test_only_witnessed_records_are_marked_unsound():
    """Exactly two records are unsound, both on a named source's authority, and
    both witnesses recompute to a violation here."""
    unsound = [r for r in S.REGISTRY if r.known_unsound]
    assert len(unsound) == 2
    names = {r.bound_name for r in unsound}
    assert any("envelope_v" in n for n in names)
    assert any("cone_slope_margin" in n for n in names)
    for r in unsound:
        assert r.unsound_witness is not None
        assert r.unsound_witness.violates()
        assert r.unsound_witness.margin() > 0
        assert r.unsound_witness.source.strip()


def test_the_tightest_record_is_unsound_and_the_loosest_is_not():
    """The registry's own counterexample to reading itself as a correctness
    ranking. If this ever stops holding the worked example is gone and the
    README must be rewritten."""
    ordered = S.sorted_records()
    loosest, tightest = ordered[0], ordered[-1]
    assert "chi2_grad_bound" in loosest.bound_name
    assert loosest.known_unsound is False
    assert loosest.slack_ratio().lo > 10**19
    cone = _find("cone_slope_margin")
    assert cone.known_unsound is True
    assert abs(cone.slack_ratio().hi - 1) < F(1, 10**15)


def test_report_is_sorted_by_slack_ratio_descending():
    ordered = S.sorted_records()
    ratios = [r.slack_ratio().lo for r in ordered]
    assert ratios == sorted(ratios, reverse=True)


def test_report_carries_the_honesty_rule_and_the_disclaimer():
    text = S.render_report()
    assert "UTILITY, never its CORRECTNESS" in text
    assert "A bound loose by 1e19 is still a bound." in text
    assert "Original prize problems solved: 0." in text
    for obl in ("OBL-H5-JETMOD", "OBL-D1-PROMOTE", "D3-LEMMA-RN-UNIF"):
        assert obl in text
    assert "WHAT WAS SEARCHED" in text
    assert "registry audit: 0 problem(s)" in text


def test_report_never_says_a_sound_record_is_sound_either():
    """The unsound column reads ``-`` for unestablished, never ``ok``/``sound``.

    This registry certifies soundness no more than it refutes it.
    """
    text = S.render_report()
    assert "not established by any source; this registry claims none" in text
    assert "unsound?" in text


def test_generated_slack_columns_carry_no_correctness_vocabulary():
    """The band names, ratio displays and honesty banner must be free of
    correctness words, so that a skimmed table cannot read a big number as a
    wrong number. Quoted source prose is exempt and does contain such words --
    record 7's source says the expression "is not an upper bound at all" --
    because quoting accurately is not editorialising.
    """
    assert S.forbidden_words_in_report() == []
    for sev in Severity:
        low = sev.value.lower()
        assert not any(w in low for w in S.FORBIDDEN_CORRECTNESS_WORDS)


def test_search_log_is_present_and_answers_the_headline_question():
    joined = " ".join(S.SEARCH_LOG)
    assert "99_DO_NOT_OPEN" in joined
    assert "02_LEGACY_Q0_ARCHIVE" in joined
    assert "only one of its size" in joined
    assert "COVERAGE GAP" in joined


def test_no_record_composes_the_2d_and_3d_tracks():
    """Standing firewall. No registry record may mention the 3D lifetime track
    at all; every one of them is 2D-upper-track machinery."""
    for r in S.REGISTRY:
        blob = " ".join([r.bound_name, r.carrier, r.source_document,
                         r.source_quote, r.notes]).lower()
        for token in ("lifetime", "ao48", "3d track", "3d lifetime",
                      "side24 3d", "3-d", "three-dimensional"):
            assert token not in blob, f"{r.bound_name}: mentions {token!r}"


def test_decimal_built_from_a_float_is_the_documented_hole():
    """``to_fraction``'s hole reaches this module and is pinned, not papered
    over: ``Decimal(0.1)`` is the binary double, ``Decimal("0.1")`` is 1/10.
    A Decimal carries no provenance so this cannot be detected here."""
    assert Interval(Decimal("0.1")).lo == F(1, 10)
    assert Interval(Decimal(0.1)).lo != F(1, 10)
    assert Interval(Decimal(0.1)).lo == F(0.1)
