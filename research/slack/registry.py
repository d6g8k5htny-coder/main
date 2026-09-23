"""Bound-slack registry: how loose is each bound the corpus states, exactly.

The corpus records bound-slack failures in prose. The headline one, from
``CL-RNU-001``, as ``LANE_RN_UNIF.md`` §2 summarises it in its own words (the
record below carries CL-RNU-001's own sentences)::

    Engine `d3_rn_unif.py` located (Kimi mid-build); certifier never invoked.
    Root cause: `chi2_grad_bound` ~1.57e14 vs true |grad chi^2| ~1.563e-5
    (~1e19 slack).

A bound nineteen orders of magnitude loose cannot close anything, and until
this module nothing in the repository noticed. This module turns "that bound is
loose" from an anecdote into a tracked, measured, exactly-computed quantity.


THE CRITICAL HONESTY RULE (the reason this module exists at all)
===============================================================

**A large slack ratio is a statement about a BOUND'S UTILITY. It is never a
statement about the bound's CORRECTNESS.**

A bound loose by ``1e19`` is still, as far as this module knows, a perfectly
true bound. ``chi2_grad_bound`` returning ``1.57e14`` where the true gradient
magnitude is ``1.563e-5`` is, on that evidence alone, a *correct* upper bound
that happens to be useless. Nothing in this module may be read as saying
otherwise, and nothing it prints is allowed to imply it.

The rule is enforced in code, not merely asserted in prose:

1. :class:`SlackRecord` has a field ``known_unsound``, entirely separate from
   every slack quantity, defaulting to ``False``.

2. ``known_unsound=True`` is **not** settable from a slack ratio, however
   large. It requires a :class:`SoundnessWitness` — an explicit pair of
   certified enclosures that *exhibits* the failure — and
   :func:`SoundnessWitness.violates` **recomputes** the violation by exact
   interval comparison. A witness that does not actually exhibit a violation is
   rejected at construction. So a loose bound cannot be marked unsound: its own
   numbers, fed in as a witness, prove the opposite of a violation and the
   constructor refuses the record. See
   ``tests/test_slack.py::test_conflating_looseness_with_unsoundness_is_refused``.

3. :meth:`SlackRecord.severity` never reads ``known_unsound``. The two axes are
   computed from disjoint data and are printed in disjoint columns.

4. The severity names are utility words — ``TIGHT``, ``ROUTINE``, ``LOOSE``,
   ``SEVERE``, ``UNUSABLE_AS_STATED``. None of them is a correctness word.
   :func:`report` carries the honesty rule in its banner on every single run,
   and :func:`forbidden_words_in_report` is a test hook that pins the absence of
   correctness vocabulary from the slack columns.

The converse half of the rule matters just as much and this registry has a
worked example of it: the ``cone_slope_margin`` record is the **tightest**
entry here — its slack ratio differs from 1 by about ``7e-18`` — and it is one
of the two entries known to be **unsound**. Tightness is not soundness either.


WHAT A "SLACK RATIO" IS HERE
============================

Each record carries a ``direction``:

* ``UPPER`` — the bound claims ``true_value <= claimed``. Slack is
  ``claimed / attained``.
* ``LOWER`` — the bound claims ``true_value >= claimed``. Slack is
  ``attained / claimed``.

Either way a sound bound has slack ``>= 1`` at the attained value, ``1`` is
perfect tightness, and larger is looser. The ratio is computed in exact rational
interval arithmetic (``research/interval``) and returned as an
:class:`~research.interval.Interval`, never as a float.

The ratio is only as good as the two numbers it divides, and those are usually
not certified enclosures of the true quantity. ``attained_kind`` records which
of them it is, and :meth:`SlackRecord.ratio_kind` propagates that into a
statement about what the ratio enclosure *means*:

======================================  ================================
``attained_kind``                       meaning of the ratio enclosure
======================================  ================================
``ENCLOSURE_OF_TRUE``                   an enclosure of the true slack
``CERTIFIED_LOWER_BOUND_ON_TRUE``       (UPPER) an upper bound on the true
                                        slack; (LOWER) a lower bound on it
``CERTIFIED_UPPER_BOUND_ON_TRUE``       the other way round
``QUOTED_POINT_ESTIMATE``               INDICATIVE only — no certified
``CORRECTED_DIAGNOSTIC``                relation to the true slack
======================================  ================================

That distinction is the whole difference between "the bound is 1.27x loose" and
"the bound is at most 1.27x loose, and may be exactly tight".


THE SEVERITY LADDER, AND WHY THESE BOUNDARIES
=============================================

The boundaries are not round numbers picked for looking tidy. Each is anchored
to a quantity this program's own machinery uses, so that "SEVERE" means
something operational: *no amount of the refinement this program can actually
run will rescue this bound.*

``TIGHT``               ``[1, 2)``
    Within a factor of two. The H3 rung floor sits here (``+92.1%`` margin at
    ``r = 1/20``, ``H3_RUNG_FLOOR.md``), and so does RN3's far-zone count
    (``2.22542 r^3 < I_far < 2.83312 r^3``, a spread of ``1.273x``). A factor
    under two is the regime where the published certified enclosures live.

``ROUTINE``             ``[2, 10)``
    Inside the fudge the frozen engine already grants itself. Its cell
    sup-bound (``d3_rn_unif.py`` line 2163) is
    ``kap0 + g0*hw + SAFE_H*h0*hw^2/2 + SAFE_T*t0*hw^3/6`` with
    ``SAFE_H = 3`` and ``SAFE_T = 9`` "absorbing cell-scale variation". A
    bound loose by less than ten is loose by less than the engine's own
    second- and third-order safety factors.

``LOOSE``               ``[10, 1e3)``
    Recoverable by refinement *in principle*. A first-order term in that same
    sup-bound shrinks linearly in the cell half-width ``hw``, so three orders
    of magnitude is three orders of magnitude of subdivision — expensive, but
    not structurally impossible.

``SEVERE``              ``[1e3, 1e6)``
    At or past the refinement budget the program actually has. The polar cover
    ``d in [5, 17]`` (``LANE_RN_UNIF.md`` T4 item 2) bisects down to a
    fail-closed floor of ``hw < 4e-4`` (line 2170), i.e. at most
    ``(17-5)/(2*4e-4) = 15,000`` radial cells; with theta-halving on top,
    something on the order of ``1e6`` cells. A bound loose by ``1e3``..``1e6``
    consumes that entire budget to buy back what a better inequality would
    give for free. This is where the ``tau`` entrywise envelope (``5.3e3``)
    and the candidate ``T4_kap(5)`` (``4e5``) sit.

``UNUSABLE_AS_STATED``  ``[1e6, inf)``
    No feasible subdivision of this program's own covers recovers it. The
    bound may be perfectly true and it still cannot close a cell.
    ``chi2_grad_bound``'s ``1e19`` is thirteen orders of magnitude past the
    whole cover budget.

``VIOLATION_WITNESSED`` ``[0, 1)``
    Not a looseness class at all. The ratio has fallen below one, which means
    the "bound" is on the wrong side of the quantity at an exhibited witness.
    A record may only land here if it carries a :class:`SoundnessWitness` that
    this module re-verifies; otherwise the constructor raises.

Severity is classified from the ratio's **lower** endpoint — the least slack
consistent with the recorded data — so the classification never overstates how
bad a bound is. :meth:`SlackRecord.severity_is_sharp` says whether the upper
endpoint lands in the same band.


ARITHMETIC DISCIPLINE
=====================

* Values are ``Fraction``, exact decimal/rational strings, ``Decimal``, or
  ``Interval``. **A bare ``float`` in a value field raises.** The refusal is
  this module's own, over and above ``research.interval.to_fraction``'s.
* ``to_fraction``'s documented hole applies here too: ``Decimal(0.1)`` is the
  exact binary double, not ``1/10``. A ``Decimal`` carries no provenance so
  this cannot be detected; pass strings.
* Quoted values such as "~1.57e14" are stored as the **rounding interval of the
  quoted digits** (``[1.565e14, 1.575e14]``), which is an honest enclosure of
  *what the document reported* and not a certified enclosure of the
  mathematical quantity. ``claimed_provenance`` says which.
* Exactly one value in this registry is stored as a binary double's exact
  rational value: ``cone_slope_margin``'s ``claimed``, because the audit that reports it
  computed with the double rather than with the decimal digit string it
  prints. That is recorded in the record's ``notes`` with both readings and
  both differences, rather than quietly picking one.


WHAT THIS MODULE DOES NOT ESTABLISH
===================================

* It closes, discharges, reduces, promotes, reclassifies and repairs
  **nothing**. ``OBL-H5-JETMOD``, ``OBL-H5-ZBAND`` (hi side),
  ``OBL-H5-REMOTE-THRESHOLD``, ``OBL-D1-PROMOTE`` (chart side) and **both**
  pieces of ``D3-LEMMA-RN-UNIF`` stay exactly as OPEN as
  ``docs/OPEN_PROBLEMS.md`` records them. Measuring a defect is not fixing it.
* It refutes no bound. Two records carry ``known_unsound=True``; in both cases
  the unsoundness was established and published by a **source** (RN5 2026-09-17
  for ``envelope_v``, CL-AUD-202-v1.0 for ``cone_slope_margin``) and this
  module only re-verifies the published witness arithmetic. It has discovered
  no unsoundness of its own and claims none.
* It repairs no bound and patches no engine. The frozen engine is untouched.
* A slack ratio here is arithmetic on numbers **other documents report**. Where
  those numbers are ``mpmath`` floats, finite differences, candidate constants
  or quoted digit strings, the ratio inherits exactly that status and is
  labelled NON-CERTIFYING. An exactly-computed ratio between two uncertified
  inputs is an exact ratio of uncertified inputs.
* It never composes the 2D upper track, the 2D lower track and the 3D lifetime
  track, and relates them in no way.
* It solves no prize problem and bears on none. Original prize problems solved:
  **0**.
* Passing tests show the arithmetic does what the docstrings say. They are not
  a review of any bound's derivation.

Standard library only (``fractions``, ``decimal``, ``dataclasses``, ``enum``),
plus ``research.interval``. Python 3.11.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from fractions import Fraction
from typing import List, Optional, Sequence, Tuple, Union

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:  # pragma: no cover - import plumbing
    sys.path.insert(0, _ROOT)

from research.interval import Interval  # noqa: E402

__all__ = [
    "Provenance", "BoundDirection", "AttainedKind", "RatioKind", "Severity",
    "SoundnessWitness", "SlackRecord",
    "REGISTRY", "SEARCH_LOG", "HONESTY_RULE", "DOES_NOT_ESTABLISH",
    "FORBIDDEN_CORRECTNESS_WORDS",
    "slack_ratio", "severity_of", "order_of_magnitude", "sci",
    "report", "render_report", "audit_registry", "forbidden_words_in_report",
]

Value = Union[int, str, Fraction, Decimal, Interval]


HONESTY_RULE = (
    "A slack ratio measures a bound's UTILITY, never its CORRECTNESS. "
    "Every bound below is, as far as this registry knows, a TRUE bound unless "
    "its own separate 'unsound' column says otherwise on a named source's "
    "authority. A bound loose by 1e19 is still a bound."
)

DOES_NOT_ESTABLISH = (
    "This registry closes, discharges, promotes, reclassifies and repairs "
    "nothing. OBL-H5-JETMOD, OBL-H5-ZBAND (hi), OBL-H5-REMOTE-THRESHOLD, "
    "OBL-D1-PROMOTE (chart) and both pieces of D3-LEMMA-RN-UNIF remain OPEN. "
    "It refutes no bound: the two unsound records were established by their "
    "sources, not here. Original prize problems solved: 0."
)

#: Correctness vocabulary that must never appear in a rendered slack report.
#: Pinned by ``tests/test_slack.py`` so that no future edit can let a big
#: number start reading as a wrong number.
FORBIDDEN_CORRECTNESS_WORDS: Tuple[str, ...] = (
    "wrong", "false", "incorrect", "invalid", "refuted", "disproved",
    "bogus", "broken", "fails", "erroneous",
)


# --------------------------------------------------------------------------
# taxonomies
# --------------------------------------------------------------------------

class Provenance(Enum):
    """Where a recorded value came from, and whether that makes it certified.

    ``certifying`` is true for exactly two members. Everything else — mpmath at
    any precision, IEEE-754 doubles, finite differences, candidate constants,
    digits quoted out of a prose document — is NON-CERTIFYING, and the sources
    say so about themselves.
    """

    CERTIFIED_INTERVAL = "certified interval enclosure (exact rational endpoints)"
    EXACT_RATIONAL = "exact rational arithmetic"
    MPMATH_FLOAT = "mpmath binary floating point -- NON-CERTIFYING"
    FLOAT64 = "IEEE-754 double -- NON-CERTIFYING"
    FINITE_DIFFERENCE = "finite-difference estimate -- NON-CERTIFYING"
    CANDIDATE_CONSTANT = "candidate constant, not proved -- NON-CERTIFYING"
    QUOTED_FROM_DOCUMENT = "digits quoted from a document -- NON-CERTIFYING"

    @property
    def certifying(self) -> bool:
        return self in (Provenance.CERTIFIED_INTERVAL, Provenance.EXACT_RATIONAL)

    @property
    def tag(self) -> str:
        return "CERT" if self.certifying else "NON-CERT"


class BoundDirection(Enum):
    """Which way the bound points."""

    UPPER = "upper"   # claims true <= claimed
    LOWER = "lower"   # claims true >= claimed


class AttainedKind(Enum):
    """What the recorded ``attained`` value is, relative to the true value."""

    ENCLOSURE_OF_TRUE = "certified enclosure of the true value"
    CERTIFIED_LOWER_BOUND_ON_TRUE = "certified lower bound on the true value"
    CERTIFIED_UPPER_BOUND_ON_TRUE = "certified upper bound on the true value"
    QUOTED_POINT_ESTIMATE = "point value quoted from a document -- no certified relation"
    CORRECTED_DIAGNOSTIC = "a corrected diagnostic replacing a defective one -- not a true value"


class RatioKind(Enum):
    """What a computed slack-ratio enclosure means about the *true* slack."""

    ENCLOSURE_OF_TRUE_SLACK = "encloses the true slack"
    UPPER_BOUND_ON_TRUE_SLACK = "bounds the true slack from above"
    LOWER_BOUND_ON_TRUE_SLACK = "bounds the true slack from below"
    INDICATIVE_ONLY = "INDICATIVE ONLY -- no certified relation to the true slack"


class Severity(Enum):
    """Utility bands. See the module docstring for the justification of each
    boundary. Not one of these names is a correctness word."""

    VIOLATION_WITNESSED = "VIOLATION_WITNESSED"
    TIGHT = "TIGHT"
    ROUTINE = "ROUTINE"
    LOOSE = "LOOSE"
    SEVERE = "SEVERE"
    UNUSABLE_AS_STATED = "UNUSABLE_AS_STATED"

    @property
    def rank(self) -> int:
        return _SEVERITY_ORDER.index(self)


_SEVERITY_ORDER = [
    Severity.VIOLATION_WITNESSED,
    Severity.TIGHT,
    Severity.ROUTINE,
    Severity.LOOSE,
    Severity.SEVERE,
    Severity.UNUSABLE_AS_STATED,
]

#: ``(lower_inclusive_endpoint, severity)``, ascending. A ratio ``x`` lands in
#: the last band whose endpoint is ``<= x``. ``VIOLATION_WITNESSED`` is the
#: ``[0, 1)`` band and is reachable only with a verified witness.
SEVERITY_LADDER: Tuple[Tuple[Fraction, Severity], ...] = (
    (Fraction(0), Severity.VIOLATION_WITNESSED),
    (Fraction(1), Severity.TIGHT),
    (Fraction(2), Severity.ROUTINE),
    (Fraction(10), Severity.LOOSE),
    (Fraction(10) ** 3, Severity.SEVERE),
    (Fraction(10) ** 6, Severity.UNUSABLE_AS_STATED),
)


def severity_of(x: Fraction) -> Severity:
    """The band a single exact ratio falls in.

    Bands are half-open on the right: ``severity_of(2) is ROUTINE`` and
    ``severity_of(Fraction(19999, 10000)) is TIGHT``. A negative ratio is not a
    meaningful slack and raises rather than being silently classified.
    """
    if x < 0:
        raise ValueError(f"a slack ratio cannot be negative: {x}")
    out = SEVERITY_LADDER[0][1]
    for edge, sev in SEVERITY_LADDER:
        if x >= edge:
            out = sev
    return out


# --------------------------------------------------------------------------
# exact value handling
# --------------------------------------------------------------------------

def _as_interval(x: Value, whose: str) -> Interval:
    """Coerce to an ``Interval``, refusing ``float`` explicitly.

    ``research.interval.to_fraction`` already refuses floats, but this module
    refuses them itself and with its own message, because a float reaching a
    *value* field of a slack record is a different failure from a float
    reaching an interval endpoint: it silently converts a reported decimal into
    a nearby binary double and then reports the ratio of two things nobody
    wrote down.
    """
    if isinstance(x, float):
        raise TypeError(
            f"{whose}: float is refused in a slack-record value field. "
            f"float('0.1') is not 1/10, and a bound recorded as a double is "
            f"not the bound the document states. Pass an int, a Fraction, a "
            f"Decimal, a decimal/rational string such as '1.57e14' or "
            f"'63/1000', or an Interval. If a binary double really is the "
            f"recorded object -- see the cone_slope_margin record), pass its "
            f"exact value as a "
            f"decimal string and say so in notes."
        )
    if isinstance(x, Interval):
        return x
    return Interval(x)


def rounding_interval(digits: str) -> Interval:
    """The enclosure of a value reported only to the digits given.

    ``rounding_interval("1.57e14")`` is ``[1.565e14, 1.575e14]``: every real
    number whose correctly-rounded rendering at that many digits is the given
    string. This is the honest encoding of a prose quotation such as
    "``~1.57e14``" — it encloses *what the document reported*, which is not the
    same thing as enclosing the mathematical quantity, and
    ``Provenance.QUOTED_FROM_DOCUMENT`` is what says so.

    The half-ulp is taken in the last written digit, including trailing zeros:
    ``"2.83312"`` gives a half-ulp of ``5e-6``.
    """
    d = Decimal(digits)
    sign, dgts, exp = d.as_tuple()
    if not isinstance(exp, int):  # pragma: no cover - NaN/Inf
        raise ValueError(f"not a finite decimal: {digits}")
    half = Fraction(1, 2) * Fraction(10) ** exp
    centre = Fraction(d)
    return Interval(centre - half, centre + half)


def truncated_interval(digits: str) -> Interval:
    """The enclosure of a value quoted as a *truncated* decimal expansion.

    A document that prints ``0.00864436749429013482649377440506888676...``
    with a trailing ellipsis has told us the value lies in
    ``[d, d + 10**exp]`` — the digits shown are correct and more follow. That
    is one-sided, unlike :func:`rounding_interval`.
    """
    d = Decimal(digits)
    _, _, exp = d.as_tuple()
    if not isinstance(exp, int):  # pragma: no cover - NaN/Inf
        raise ValueError(f"not a finite decimal: {digits}")
    lo = Fraction(d)
    return Interval(lo, lo + Fraction(10) ** exp)


def order_of_magnitude(x: Fraction) -> int:
    """``floor(log10 |x|)``, computed exactly by integer comparison.

    No logarithm, no float. ``order_of_magnitude(Fraction(1)) == 0``,
    ``order_of_magnitude(Fraction(1, 10)) == -1``. Zero raises: it has no order
    of magnitude, and returning a sentinel would let it be printed as one.
    """
    if x == 0:
        raise ValueError("zero has no order of magnitude")
    v = abs(Fraction(x))
    e = 0
    ten = Fraction(10)
    while v >= ten:
        v /= ten
        e += 1
    while v < 1:
        v *= ten
        e -= 1
    return e


def sci(x: Fraction, sig: int = 3) -> str:
    """Round-half-up scientific notation, computed in exact integers.

    Used only for display. It never enters an arithmetic path, and it is not
    permitted to: every comparison and every ratio in this module is exact
    rational.
    """
    x = Fraction(x)
    if x == 0:
        return "0"
    sign = "-" if x < 0 else ""
    v = abs(x)
    e = order_of_magnitude(v)
    scaled = v / Fraction(10) ** e * 10 ** (sig - 1)
    n = (2 * scaled.numerator + scaled.denominator) // (2 * scaled.denominator)
    if n >= 10 ** sig:          # rounded up past the decade, e.g. 9.999 -> 10.0
        n //= 10
        e += 1
    s = str(n)
    mant = s[0] + ("." + s[1:] if sig > 1 else "")
    return f"{sign}{mant}e{e:+d}"


# --------------------------------------------------------------------------
# soundness: witnessed, never inferred
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SoundnessWitness:
    """An exhibited failure of a bound, which this module re-verifies.

    THIS IS THE ONLY ROUTE TO ``known_unsound=True``. It exists so that
    "unsound" can never be a synonym for "loose": a witness is checked by
    exact interval comparison, and a merely loose bound's own numbers, offered
    as a witness, prove the opposite of a violation and are refused.

    ``bound`` encloses the value the bound reports at the witness point.
    ``quantity`` encloses (or one-sidedly bounds) the true value there, and
    ``quantity_kind`` says which. The violation checks are:

    * ``direction=UPPER``  — violated iff ``bound.hi < quantity.lo``, which
      needs ``quantity.lo`` to be a certified lower bound on the true value,
      i.e. ``quantity_kind`` in ``{ENCLOSURE_OF_TRUE,
      CERTIFIED_LOWER_BOUND_ON_TRUE}``.
    * ``direction=LOWER``  — violated iff ``bound.lo > quantity.hi``, which
      needs ``quantity_kind`` in ``{ENCLOSURE_OF_TRUE,
      CERTIFIED_UPPER_BOUND_ON_TRUE}``.

    Both checks are strict. A bound that merely touches the quantity is not
    witnessed as violated here.

    ``source`` must name the document that established the violation. This
    module has never discovered a violation and does not claim to; it
    re-verifies published ones.
    """

    source: str
    statement: str
    direction: BoundDirection
    bound: Interval
    quantity: Interval
    quantity_kind: AttainedKind

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("a soundness witness must name its source document")
        if not self.statement.strip():
            raise ValueError("a soundness witness must state what it exhibits")
        if not isinstance(self.bound, Interval) or not isinstance(self.quantity, Interval):
            raise TypeError("witness bound and quantity must be Intervals")

    def usable(self) -> bool:
        """Whether ``quantity_kind`` supports a check in this direction."""
        if self.direction is BoundDirection.UPPER:
            return self.quantity_kind in (
                AttainedKind.ENCLOSURE_OF_TRUE,
                AttainedKind.CERTIFIED_LOWER_BOUND_ON_TRUE,
            )
        return self.quantity_kind in (
            AttainedKind.ENCLOSURE_OF_TRUE,
            AttainedKind.CERTIFIED_UPPER_BOUND_ON_TRUE,
        )

    def violates(self) -> bool:
        """Recompute the violation, exactly. False whenever it is not proved."""
        if not self.usable():
            return False
        if self.direction is BoundDirection.UPPER:
            return self.bound.hi < self.quantity.lo
        return self.bound.lo > self.quantity.hi

    def margin(self) -> Fraction:
        """How far the bound is on the wrong side, exactly. Negative if not."""
        if self.direction is BoundDirection.UPPER:
            return self.quantity.lo - self.bound.hi
        return self.bound.lo - self.quantity.hi


# --------------------------------------------------------------------------
# the record
# --------------------------------------------------------------------------

@dataclass
class SlackRecord:
    """One bound, the value it claims, the value actually attained, and the
    exact ratio between them.

    Construction refuses, in this order:

    * a value field holding a ``float``;
    * an unsourced record — ``bound_name``, ``carrier`` and ``source_document``
      must all be non-empty, and ``source_quote`` must be non-empty because a
      registry of other people's numbers that does not quote them is a rumour;
    * ``known_unsound=True`` without a :class:`SoundnessWitness`;
    * ``known_unsound=True`` with a witness that does not, on recomputation,
      exhibit a violation — this is the conflation guard;
    * a witness that does exhibit a violation while ``known_unsound=False`` —
      a proved violation may not be hidden;
    * ``known_unsound=False`` while the slack ratio is provably below one — if
      the record's own certified numbers show the bound on the wrong side, it
      must carry the witness that says so.
    """

    bound_name: str
    carrier: str
    source_document: str
    source_quote: str
    direction: BoundDirection

    claimed: Interval
    claimed_provenance: Provenance
    attained: Interval
    attained_provenance: Provenance
    attained_kind: AttainedKind

    known_unsound: bool = False
    unsound_witness: Optional[SoundnessWitness] = None
    obligations_untouched: Tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        self.claimed = _as_interval(self.claimed, f"{self.bound_name}.claimed")
        self.attained = _as_interval(self.attained, f"{self.bound_name}.attained")

        for fname in ("bound_name", "carrier", "source_document", "source_quote"):
            if not str(getattr(self, fname)).strip():
                raise ValueError(
                    f"unsourced slack record: {fname!r} is empty. Every record "
                    f"must name the bound, the carrier that states it, the "
                    f"source document, and quote it."
                )

        if self.known_unsound:
            if self.unsound_witness is None:
                raise ValueError(
                    f"{self.bound_name}: known_unsound=True requires a "
                    f"SoundnessWitness. Unsoundness is witnessed here, never "
                    f"inferred -- least of all from a slack ratio, however "
                    f"large. A loose bound is not a wrong bound."
                )
            if not self.unsound_witness.violates():
                raise ValueError(
                    f"{self.bound_name}: known_unsound=True but the supplied "
                    f"witness does not exhibit a violation on recomputation "
                    f"(margin {sci(self.unsound_witness.margin())}). "
                    f"LOOSENESS IS NOT UNSOUNDNESS: a bound being far from the "
                    f"true value is not evidence that it is on the wrong side "
                    f"of it."
                )
        elif self.unsound_witness is not None and self.unsound_witness.violates():
            raise ValueError(
                f"{self.bound_name}: a witness exhibits a violation but "
                f"known_unsound is False. A proved violation may not be "
                f"recorded silently."
            )

        if not self.known_unsound:
            ratio = self.slack_ratio()
            kind = self.ratio_kind()
            proves_violation = kind in (
                RatioKind.ENCLOSURE_OF_TRUE_SLACK,
                RatioKind.UPPER_BOUND_ON_TRUE_SLACK,
            )
            if proves_violation and ratio.hi < 1:
                raise ValueError(
                    f"{self.bound_name}: the record's own certified numbers "
                    f"put the slack ratio wholly below 1 "
                    f"({sci(ratio.hi)}), which exhibits the bound on the wrong "
                    f"side of the quantity, yet known_unsound is False. "
                    f"Supply the witness and the source that established it."
                )

    # ------------------------------------------------------------- arithmetic

    def slack_ratio(self) -> Interval:
        """The slack as an exact-rational interval.

        ``UPPER``: ``claimed / attained``. ``LOWER``: ``attained / claimed``.
        Both are ``>= 1`` for a bound that holds at the attained value.

        Raises ``ZeroDivisionError`` (from the interval library) if the
        denominator straddles zero, which is correct: a slack ratio against a
        quantity that might be zero is not a number, and silently returning
        something would be worse than refusing.
        """
        if self.direction is BoundDirection.UPPER:
            return self.claimed / self.attained
        return self.attained / self.claimed

    def ratio_kind(self) -> RatioKind:
        """What :meth:`slack_ratio` means about the *true* slack.

        The direction of the inequality flips with ``direction``: dividing by a
        certified lower bound on the true value makes an UPPER-bound ratio too
        big (hence an upper bound on the true slack) and a LOWER-bound ratio
        too small.
        """
        k = self.attained_kind
        if k is AttainedKind.ENCLOSURE_OF_TRUE:
            return RatioKind.ENCLOSURE_OF_TRUE_SLACK
        if k in (AttainedKind.QUOTED_POINT_ESTIMATE, AttainedKind.CORRECTED_DIAGNOSTIC):
            return RatioKind.INDICATIVE_ONLY
        if self.direction is BoundDirection.UPPER:
            if k is AttainedKind.CERTIFIED_LOWER_BOUND_ON_TRUE:
                return RatioKind.UPPER_BOUND_ON_TRUE_SLACK
            return RatioKind.LOWER_BOUND_ON_TRUE_SLACK
        if k is AttainedKind.CERTIFIED_LOWER_BOUND_ON_TRUE:
            return RatioKind.LOWER_BOUND_ON_TRUE_SLACK
        return RatioKind.UPPER_BOUND_ON_TRUE_SLACK

    def severity(self) -> Severity:
        """The utility band, from the ratio's lower endpoint.

        **This method never reads ``known_unsound``.** Severity and soundness
        are computed from disjoint data on purpose; flipping one may not move
        the other. ``tests/test_slack.py`` pins that.

        The lower endpoint is used so that the classification reports the least
        slack consistent with the record, never the most.
        """
        return severity_of(self.slack_ratio().lo)

    def severity_is_sharp(self) -> bool:
        """True when both ratio endpoints land in the same band."""
        r = self.slack_ratio()
        return severity_of(r.lo) is severity_of(r.hi)

    def order_of_magnitude(self) -> Tuple[int, int]:
        """``(floor(log10 ratio.lo), floor(log10 ratio.hi))``, exactly."""
        r = self.slack_ratio()
        return order_of_magnitude(r.lo), order_of_magnitude(r.hi)

    def certifying(self) -> bool:
        """True only when BOTH recorded values come from certifying arithmetic.

        A ratio is no more certified than its inputs. Everything else is
        NON-CERTIFYING and printed as such.
        """
        return self.claimed_provenance.certifying and self.attained_provenance.certifying

    def ratio_display(self) -> str:
        """Compact display. Never used in arithmetic.

        A ratio within ``1e-3`` of exact tightness is rendered as its deviation
        from 1 (``1-7.03e-18``) rather than as ``1.00e+0``. Three significant
        figures would otherwise print ``cone_slope_margin`` -- a bound that
        overshoots the
        quantity it is supposed to bound below — as a flat ``1.00e+0``, hiding
        the sign of the deviation, which is the one thing about it that
        matters.
        """
        r = self.slack_ratio()
        lo, hi = r.lo, r.hi
        one = Fraction(1)
        near = Fraction(1, 1000)
        if max(abs(lo - one), abs(hi - one)) < near and (lo != one or hi != one):
            def dev(x: Fraction) -> str:
                d = x - one
                if d == 0:
                    return "1"
                return ("1+" if d > 0 else "1-") + sci(abs(d))
            return dev(lo) if lo == hi else f"[{dev(lo)}, {dev(hi)}]"
        if lo == hi:
            return sci(lo)
        return f"[{sci(lo)}, {sci(hi)}]"


# --------------------------------------------------------------------------
# the registry
# --------------------------------------------------------------------------

_OBL_ALL = (
    "OBL-H5-JETMOD", "OBL-H5-ZBAND(hi)", "OBL-H5-REMOTE-THRESHOLD",
    "OBL-D1-PROMOTE(chart)", "D3-LEMMA-RN-UNIF Piece 1",
    "D3-LEMMA-RN-UNIF Piece 2",
)


# 1. The headline. CL-RNU-001's numbers, as summarised in LANE_RN_UNIF.md
#    section 2 (that memo's own sentences, not a verbatim quotation of CL-RNU-001).
_R_CHI2 = SlackRecord(
    bound_name="chi2_grad_bound",
    carrier=(
        "d3_rn_unif.py lines 1690-1720 (the frozen RN engine); recovered "
        "byte-exact at engine/rn_engine/frozen/K3_SIDE24_LB/UPPER2D/"
        "D3_percolation/d3_rn_unif.py, sha256 85d7725fab42eeb0e823226f44d17f14"
        "2a5c57e5d89084b2b6edffe4a8f0c930, 103,166 B"
    ),
    source_document=(
        "CL-RNU-001_RN-UNIF_ENGINE_STATUS_AND_CLOSURE_PLAN_2026-09-16.md "
        "(Drive 16o9u_KgysJv9lnR93PNXCqc-MdrW9QL6, 12,291 B, sha256 "
        "488b1b0c2dd270083fa0dd60a434ecf0461bae026f4f8d63ad960cb7b05906b5), "
        "mirrored byte-exact under drive/mirrors/; the sentences below are "
        "CL-RNU-001's own. LANE_RN_UNIF.md section 2 (Drive "
        "1dK4ZimCC8o9670K-9ZJS9Jtquid1CAA6, also mirrored) summarises them in "
        "its own words ('certifier never invoked … Piece 2 driver unwritten'), "
        "which until 2026-09-19 this record presented as a verbatim quotation"
    ),
    source_quote=(
        "`chi2_grad_bound` = 1.57e14 with chi^2 = 1.94e-6 [...] True |grad "
        "chi^2| at (5,0), central FD at dps 100 with h = 1e-20: 1.563e-5 (vs "
        "the engine's bound 1.57e14 -- 1e19 slack) [...] the adaptive polar "
        "certifier is defined and never invoked [...] its driver is likewise "
        "unwritten."
    ),
    direction=BoundDirection.UPPER,
    claimed=rounding_interval("1.57e14"),
    claimed_provenance=Provenance.QUOTED_FROM_DOCUMENT,
    attained=rounding_interval("1.563e-5"),
    attained_provenance=Provenance.QUOTED_FROM_DOCUMENT,
    attained_kind=AttainedKind.QUOTED_POINT_ESTIMATE,
    known_unsound=False,
    obligations_untouched=_OBL_ALL,
    notes=(
        "STILL A TRUE BOUND as far as this registry knows; nothing in the "
        "corpus exhibits it on the wrong side of |grad chi^2|. What is "
        "recorded is that it cannot close anything. "
        "Structural cause identified in docs/ENGINE_RECOVERY.md section 3.5 "
        "and NOT measured: log q is a signed sum of four terms and line 1718 "
        "replaces that signed sum by the sum of absolute values; each term is "
        "dS conjugated by S_pair^-1 with the engine's stated lambda_min = "
        "2.6e-10, and the gdetM term carries the inverse twice, so "
        "lambda_min^-2 ~ 1.5e19 lands at the reported ratio. That attribution "
        "is an inference from lines 1700 and 1709-1720, not a measurement; the "
        "one-line diagnostic that would settle it (print the four terms "
        "separately at (5,0)) was deliberately not run, because it means "
        "running the engine. "
        "SEPARATE AND MORE SERIOUS THAN SLACK: the same section reports that "
        "chi2_grad_bound calls mean_grad_exact, which is missing chain-rule "
        "terms of dTY6. A bound fed a derivative that is missing terms is not "
        "merely loose -- its certification is void. That is a soundness "
        "question about a DIFFERENT function and it is recorded in "
        "docs/ENGINE_RECOVERY.md section 3.6, not here; no source exhibits "
        "chi2_grad_bound itself on the wrong side of the quantity, so "
        "known_unsound stays False and this registry makes no claim either "
        "way about it."
    ),
)

# 2. The pair-block tau envelope. Stated in the frozen engine's own header.
_R_TAU = SlackRecord(
    bound_name="tau entrywise envelope  ||Delta||_F^2 / lambda_min",
    carrier=(
        "d3_rn_unif.py header lines 5-7 (frozen RN engine, sha256 85d7725f...); "
        "recomputed and emitted by d3_amend.py a3_unif_disposition lines "
        "386-389, sha256-pinned by the engine"
    ),
    source_document=(
        "engine/rn_engine/frozen/K3_SIDE24_LB/UPPER2D/D3_percolation/"
        "d3_rn_unif.py, recovered byte-exact and bound in "
        "engine/rn_engine/BINDING.json"
    ),
    source_quote=(
        "The pair-Hessian conditional covariance Spair is rigid (lambda_min = "
        "2.6e-10 at the rung): entrywise envelopes overbound tau by x5.3e3 "
        "(||Delta||_F^2/lambda_min = 1.881 vs true tau = 3.53e-4) because the "
        "pins EXPLAIN the pair block's rigid directions."
    ),
    direction=BoundDirection.UPPER,
    claimed=rounding_interval("1.881"),
    claimed_provenance=Provenance.MPMATH_FLOAT,
    attained=rounding_interval("3.53e-4"),
    attained_provenance=Provenance.MPMATH_FLOAT,
    attained_kind=AttainedKind.QUOTED_POINT_ESTIMATE,
    known_unsound=False,
    obligations_untouched=("D3-LEMMA-RN-UNIF Piece 1",),
    notes=(
        "The engine states this slack about itself and names the mechanism: "
        "the rigid directions of the pair block lie in the pin-determined "
        "subspace, so an entrywise kernel envelope cannot see the "
        "cancellation. This is the same mechanism as chi2_grad_bound at first "
        "order with one inverse factor; chi2_grad_bound is it at gradient level "
        "with two. "
        "d3_amend.py's own disposition text calls the resulting gap the "
        "precise thing missing for far-zone uniformity at r = 0.05. "
        "Certified status: mpmath at mp.dps = 100 throughout -- high precision "
        "is NOT a certified enclosure, and no part of the frozen engine uses "
        "exact-rational or interval arithmetic anywhere."
    ),
)

# 3. The candidate T4 constant. CL-RNU-003, as summarised in LANE_RN_UNIF.md
#    section 2 (the memo's own summary sentences, not CL-RNU-003 verbatim).
_R_T4 = SlackRecord(
    bound_name="T4_kap(5) = C_comp * T4_form(5)  (candidate)",
    carrier=(
        "rnu_t4_push.py (Drive 1is4WkbVwztK_HspOel_omNKN90oNFh61, 7,924 B, "
        "sha256 7b7cc46ba56052505d6f109525f76ff559a85a81c143b48fd435503d447d85"
        "ba); T4_form, T4_kap and C_comp appear NOWHERE in the frozen engine "
        "(docs/ENGINE_RECOVERY.md section 3.7), and rnu_t4.py is recorded "
        "CANNOT_VERIFY in docs/OPEN_PROBLEMS.md section E"
    ),
    source_document=(
        "CL-RNU-003_T4_PUSH_2026-09-16.md (Drive "
        "1lnZFn0VzUAKMN0OQk_ivTnykFEHiNSaw, 1,713 B, sha256 "
        "7743de124489ac59bb7e51ff2a83463fb7a0e63f1936e7e04af852bb124e30ea), "
        "as summarised in LANE_RN_UNIF.md section 2 (Drive "
        "1dK4ZimCC8o9670K-9ZJS9Jtquid1CAA6, mirrored byte-exact): the "
        "source_quote below is that memo's summary, not CL-RNU-003 verbatim"
    ),
    source_quote=(
        "Candidate C_comp~=3091.9059; T4_kap(5)=C_comp x T4_form(5)~=1.39e7 "
        "covers FD |T4|~=35. Explicitly: Wick/Bures/ratio 4-jets not proved "
        "under this Lipschitz. Full delicate-patch cover (~2e4 DS evals) not "
        "run."
    ),
    direction=BoundDirection.UPPER,
    claimed=rounding_interval("1.39e7"),
    claimed_provenance=Provenance.CANDIDATE_CONSTANT,
    attained=rounding_interval("35"),
    attained_provenance=Provenance.FINITE_DIFFERENCE,
    attained_kind=AttainedKind.QUOTED_POINT_ESTIMATE,
    known_unsound=False,
    obligations_untouched=("D3-LEMMA-RN-UNIF Piece 1", "D3-LEMMA-RN-UNIF Piece 2"),
    notes=(
        "WEAKEST PROVENANCE IN THE REGISTRY, ON BOTH SIDES. The claimed value "
        "rests on a CANDIDATE C_comp whose Lipschitz the source itself says is "
        "not proved for the Wick/Bures/ratio 4-jets; the attained value is a "
        "finite-difference estimate. Neither is a bound and neither is "
        "certified, so the ratio is INDICATIVE ONLY. It is registered because "
        "the source states both numbers in one sentence and the gap is the "
        "reason the T4 push is PROPOSED rather than frozen: "
        "RNU_T4_PUSH_RECEIPT.json carries status=PROPOSED and "
        "lemma_closed=false. Probe cells under this candidate T4 close at "
        "hw = 7e-4/1e-3 against cap 0.68 and OPEN at 2e-3 -- probe cells are "
        "not certified cells, and EXECUTE cell rows use measured-scale T4 "
        "(INTERNAL-NOT-ENV_FORM) which the lane memo says must not be cited "
        "as certified cells."
    ),
)

# 4. envelope_v. KNOWN UNSOUND -- by RN5's published counterexample, not here.
_ENVELOPE_WITNESS = SoundnessWitness(
    source=(
        "RN5_REPAIR_AND_ERRATUM_BUNDLE (2026-09-17), reproduced exactly in "
        "research/rn/moment_envelope.py and tests/test_rn_moment_envelope.py"
    ),
    statement=(
        "An exact typed nondegenerate Gaussian counterexample: nine "
        "independent symmetric-Hessian coordinates of variance 1e-6 with "
        "(xx, yy, xy) means (-1,-1,0), (1,-1,0), (1,-1/4,0). RN5 publishes the "
        "exact rational comparison old envelope < 63/1000 < 207/1000 < typed "
        "expectation, so the defective expression is strictly BELOW a proven "
        "lower bound for the quantity it is supposed to bound above."
    ),
    direction=BoundDirection.UPPER,
    bound=Interval("625/10000", "63/1000"),
    quantity=Interval("207/1000", "207/1000"),
    quantity_kind=AttainedKind.CERTIFIED_LOWER_BOUND_ON_TRUE,
)

_R_ENVELOPE = SlackRecord(
    bound_name="envelope_v  (d3_perc.py determinant-moment envelope)",
    carrier=(
        "d3_perc.py helper envelope_v; the payload carries scope hold "
        "Q-RN5-MOMENT-001 in drive/source_map/Payloads.csv: 'envelope_v; "
        "window_cap_env and consumers relying on the claimed polarity-safe "
        "upper bound. Other functions are outside this finding.'"
    ),
    source_document=(
        "RN5_REPAIR_AND_ERRATUM_BUNDLE.zip (Drive "
        "1g5UP_KYlvPmx7LFwjGk6qK5QZLnL1hLj), 2026-09-17; restated in "
        "docs/RESEARCH_MAP.md section 3 and research/rn/moment_envelope.py"
    ),
    source_quote=(
        ">= 0.2076750355 68 against an old envelope of ~= 0.0625040624 9, "
        "proving old envelope < 63/1000 < 207/1000 < typed expectation by "
        "exact rational comparison of fourth powers."
    ),
    direction=BoundDirection.UPPER,
    claimed=Interval("625/10000", "63/1000"),
    claimed_provenance=Provenance.EXACT_RATIONAL,
    attained=Interval("207/1000", "207/1000"),
    attained_provenance=Provenance.EXACT_RATIONAL,
    attained_kind=AttainedKind.CERTIFIED_LOWER_BOUND_ON_TRUE,
    known_unsound=True,
    unsound_witness=_ENVELOPE_WITNESS,
    obligations_untouched=("D3-LEMMA-RN-UNIF Piece 2",),
    notes=(
        "THE ONLY KIND OF ENTRY THAT MAY CARRY known_unsound=True: a source "
        "exhibited the failure and this module re-verified the published "
        "witness arithmetic. The registry did not discover it. "
        "Hoelder(4,4,2) needs E[C^2]; envelope_v used E[C^4], substituting the "
        "fourth determinant moment where the second is required, and for small "
        "determinants that replacement can DECREASE the expression -- so it is "
        "not an upper bound at all. Note the slack ratio here is about 0.30, "
        "i.e. BELOW one: that is what a violation looks like, and it is "
        "categorically different from chi2_grad_bound's 1e19. "
        "Scope: this does not touch RN3's far-region proof, which uses the "
        "correct second moment, and recovering d3_perc.py does not lift "
        "Q-RN5-MOMENT-001."
    ),
)

# 5. cone_slope_margin. KNOWN UNSOUND and the TIGHTEST record here.
_CONE_STORED_DOUBLE = (
    "0.00864436749429013488732476133691307040862739086151123046875"
)
_CONE_EXACT_TRUNCATED = "0.00864436749429013482649377440506888676"

_CONE_WITNESS = SoundnessWitness(
    source="CL-AUD-202-v1.0 (Drive 1QfBwljs_j2ehbf3e2tGW7Nh5IbPHyNx9qo1DQ7IBxEo) section 3",
    statement=(
        "The audit compared every one of the 19 stored lower bounds of the "
        "GP-DATA-106 Route-B degree-four box against the exact rational value "
        "of the identical quantity from the CL clean-room implementation "
        "(CL-DATA-200/201). 18 of 19 sit between +4.1e-18 and +3.2e-16 BELOW "
        "the exact margin and are valid. cone_slope_margin sits ABOVE it, by "
        "+6.083e-20, and 'is not a valid lower bound as recorded'."
    ),
    direction=BoundDirection.LOWER,
    bound=Interval(_CONE_STORED_DOUBLE),
    quantity=truncated_interval(_CONE_EXACT_TRUNCATED),
    quantity_kind=AttainedKind.ENCLOSURE_OF_TRUE,
)

_R_CONE = SlackRecord(
    bound_name="cone_slope_margin  (GP-DATA-106 Route-B degree-four box)",
    carrier=(
        "GP-DATA-106-v1.0_routeB_small_box_interval.py line 204, "
        "cone_slope_margin = kappa - 2*tau_upper, finalized in ordinary "
        "float arithmetic; capsule Drive "
        "1sAbtS40ioX91gxiYRpbTZVg2hmqDqQu-z5HklYMibe4, source 8,709 B sha256 "
        "2eb38f92c83dee204d248357353d02770b076fd58bc6293e8f72abfb6ddf6c2e"
    ),
    source_document=(
        "CL-AUD-202-v1.0 (Drive 1QfBwljs_j2ehbf3e2tGW7Nh5IbPHyNx9qo1DQ7IBxEo), "
        "sections 2 and 3; canonical mathematical impact NONE"
    ),
    source_quote=(
        "1 of 19 OVERSHOOTS. cone_slope_margin: stored: 0.0086443674942901349 "
        "exact : 0.00864436749429013482649377440506888676... "
        "stored - exact = +6.083e-20. The stored digit-string therefore "
        "slightly OVERSTATES the true margin and is not a valid lower bound as "
        "recorded."
    ),
    direction=BoundDirection.LOWER,
    claimed=Interval(_CONE_STORED_DOUBLE),
    claimed_provenance=Provenance.FLOAT64,
    attained=truncated_interval(_CONE_EXACT_TRUNCATED),
    attained_provenance=Provenance.EXACT_RATIONAL,
    attained_kind=AttainedKind.ENCLOSURE_OF_TRUE,
    known_unsound=True,
    unsound_witness=_CONE_WITNESS,
    obligations_untouched=("OBL-D1-PROMOTE(chart)",),
    notes=(
        "THE REGISTRY'S OWN COUNTEREXAMPLE TO ANY READING OF ITSELF AS A "
        "CORRECTNESS RANKING. This is the tightest record here -- its slack "
        "ratio differs from 1 by about 7e-18 -- and it is unsound, while "
        "chi2_grad_bound is loose by 1e19 and is not known to be unsound. Sort "
        "this "
        "registry by slack and the two sit at opposite ends. "
        "WHICH NUMBER IS 'stored': the audit prints the digit string "
        "0.0086443674942901349, but read as an exact decimal that overshoots "
        "by +7.3506e-20, not the +6.083e-20 the audit states. Read as the "
        "IEEE-754 double whose shortest repr it is -- exactly "
        + _CONE_STORED_DOUBLE + " -- it overshoots by exactly "
        "6.083098693184418e-20, reproducing the audit's own figure. This "
        "record therefore stores the double's exact rational value, and both "
        "readings are pinned in tests/test_slack.py. Either reading overshoots, "
        "so the verdict is unaffected. "
        "MATERIALITY, from the audit: NONE. The error is seventeen orders of "
        "magnitude below the 8.64e-3 margin, the smallest margin in the "
        "certificate is 7.49e-3, and every gate's SIGN is confirmed positive "
        "by exact arithmetic. An unsound bound here is not a wrong conclusion "
        "there -- which is the same distinction this module exists to keep, "
        "running the other way."
    ),
)

# 6. RN3 far-zone conditional count: a certified two-sided enclosure.
_R_IFAR = SlackRecord(
    bound_name="I_far upper  (RN3 far-zone conditional count, |y| >= 5)",
    carrier="RN3 far-region interval run (384-bit Arb, author-side, zero independence credit)",
    source_document=(
        "docs/RESEARCH_MAP.md section 3, tabulating the RN3 object's headline "
        "numbers; the object's own far-region proof, reviewed at "
        "reviews/records/REV-RN3-FARZONE-20260918.json (MAJOR finding on "
        "section 9, none on the far-region proof)"
    ),
    source_quote="Conditional far count | 2.22542 r^3 < I_far < 2.83312 r^3",
    direction=BoundDirection.UPPER,
    claimed=rounding_interval("2.83312"),
    claimed_provenance=Provenance.CERTIFIED_INTERVAL,
    attained=rounding_interval("2.22542"),
    attained_provenance=Provenance.CERTIFIED_INTERVAL,
    attained_kind=AttainedKind.CERTIFIED_LOWER_BOUND_ON_TRUE,
    known_unsound=False,
    obligations_untouched=("D3-LEMMA-RN-UNIF Piece 2",),
    notes=(
        "THE TIGHT END, AND THE ONLY RECORD WHOSE BOTH SIDES ARE CERTIFIED. "
        "The ratio is an UPPER bound on the true slack, because the attained "
        "value is the certified lower endpoint of the same two-sided "
        "enclosure: the true I_far lies somewhere in [2.22542, 2.83312] r^3, "
        "so the true slack lies in [1, 1.2731]. This is what a bound-slack "
        "record looks like when the bound is fit to close something. "
        "Scope, from the review: 'RN3's far-region proof is outside the RN5 "
        "affected scope' is accurate about the far-region proof and about "
        "nothing else in that object; section 9's conditional arithmetic "
        "imports the near target 17.6804 r^3, and the open disc |y| < 0.1 is "
        "in neither named zone. This record is about the far count only."
    ),
)

# 7. H3 rung floor: the theorem-grade uniform bound against the rung certificate.
_R_ZFLOOR = SlackRecord(
    bound_name="c_Z * r^2 uniform floor for Z_r, at the rung r = 1/20",
    carrier=(
        "H3_RUNG_FLOOR.md rung table; hash-pinned by the frozen engine and "
        "recovered at engine/rn_engine/frozen/K3_SIDE24_LB/UPPER2D/H3_closure/"
        "H3_RUNG_FLOOR.md"
    ),
    source_document=(
        "H3_RUNG_FLOOR.md, one of the six hash-pinned dependencies the frozen "
        "engine fail-closes on (d3_rn_unif.py lines 59-70); verified "
        "byte-exact against those pins by engine/rn_engine/verify_recovery.py"
    ),
    source_quote=(
        "| rung | certified lo for Z_r | c_Z*r^2 | margin |  ...  "
        "| **1/20 = 0.05** | **7.7592917375327855e-3** | **4.0387231691e-3** "
        "| **+92.1%** |"
    ),
    direction=BoundDirection.LOWER,
    claimed=rounding_interval("4.0387231691e-3"),
    claimed_provenance=Provenance.CERTIFIED_INTERVAL,
    attained=rounding_interval("7.7592917375327855e-3"),
    attained_provenance=Provenance.CERTIFIED_INTERVAL,
    attained_kind=AttainedKind.CERTIFIED_LOWER_BOUND_ON_TRUE,
    known_unsound=False,
    obligations_untouched=("OBL-H5-ZBAND(hi)", "OBL-D1-PROMOTE(chart)"),
    notes=(
        "A LOWER-direction record, included so the registry is not silently "
        "an upper-bound registry. The ratio here is a LOWER bound on the true "
        "slack (the attained value is itself only a certified lower bound on "
        "Z_r), and it reproduces the source's own '+92.1%' margin to the digit. "
        "SCOPE, verbatim from the source: 'these are pointwise certifications "
        "at rational rungs. They do NOT certify the continuum (0, 0.05]'. The "
        "same paragraph names the continuation -- enclose r itself as an "
        "interval through the same pipeline, the box margins have ~1e-2 slack "
        "at 0.05 -- and says it is NOT EXECUTED. Nothing here executes it, and "
        "OBL-H5-ZBAND's hi side stays OPEN."
    ),
)


REGISTRY: List[SlackRecord] = [
    _R_CHI2, _R_TAU, _R_T4, _R_ENVELOPE, _R_CONE, _R_IFAR, _R_ZFLOOR,
]


# --------------------------------------------------------------------------
# what was searched, and what was found
# --------------------------------------------------------------------------

SEARCH_LOG: Tuple[str, ...] = (
    "Searched 2026-09-18, over docs/, registers/json/ (42 exported tabs), "
    "claims/, research/, reviews/, engine/ and the Drive index "
    "(tools/drive_index.py find, 4,456 items).",
    "Terms: slack, vs true, versus true, orders of magnitude, loose, "
    "conservative, overestimate, overbound, overshoot, too loose, not tight, "
    "fail-close(s), collapsed, margin, factor of N, x<N>e<k>.",
    "Drive title search returned nothing usable for 'orders of magnitude', "
    "'overbound', 'not tight', 'too loose' or 'conservative'; 'slack' returned "
    "two P0.2 review artifacts about worst-case slack TABLES (CL-AUD-049, "
    "CL-RSP-050) which state margins but no claimed-bound/true-value pair; "
    "'overshoot' returned exactly one item, which became the "
    "cone_slope_margin record.",
    "SUBSTANTIATED AND REGISTERED: 7 records. Documents that state BOTH a "
    "claimed bound and a measured or true value for the same quantity in the "
    "same breath are rare in this corpus -- these seven are all that were "
    "found.",
    "HONEST ANSWER TO 'is the headline case the only well-documented one': "
    "no, but it is the only one of its size. The chi2_grad_bound ~1e19 is "
    "fourteen orders of magnitude clear of the next-largest slack in the "
    "registry (T4_kap at ~4e5) and it is the only record in the "
    "UNUSABLE_AS_STATED band. The other six are well documented; they are not "
    "comparably bad.",
    "EXAMINED AND DELIBERATELY NOT REGISTERED, with the reason: "
    "(a) RN3 section 9's '2.34195 r^3 corrected versus 17.67237 r^3 "
    "wrong-power' (factor ~7.5, docs/RESEARCH_MAP.md section 3) -- this is a "
    "corrected diagnostic against a defective one, not a bound against a true "
    "value, and the 17.67237 rides the envelope_v defect already registered as "
    "the envelope_v record; registering it would double-count and would imply "
    "2.34195 is a "
    "true value, which no source says. It would have been the registry's only "
    "ROUTINE-band entry, and padding a band is not a reason. "
    "(b) B_remote = 19.55 r^3 against I_ann 17.02 + I_far 2.5283 = 19.5483 -- "
    "a budget against its consumption, not a bound against an attained value. "
    "(c) The H5 rung ladder's same-r version spread at r = 0.05 (frozen v1 "
    "731.4311 against live v3 clean 647.8048, 12.91%) and the 4.47x "
    "disagreement between the constants two adjacent bands force -- these "
    "compare two certifications of the same quantity, and two forced "
    "constants, not a bound against a true value; research/bands/ladder.py "
    "already carries them with their stated assumption. "
    "(d) SAFE_H = 3 and SAFE_T = 9 in the engine's cell sup-bound -- declared "
    "fudge factors with no measured counterpart, so no ratio exists. They are "
    "used to justify the ROUTINE band boundary instead. "
    "(e) The engine's 1e-25 / 1e-60 / 1e-80 / 1e-90 fail-close thresholds -- "
    "absolute thresholds against floating-point residuals, not bound/true "
    "pairs. "
    "(f) 'box margins have ~1e-2 slack at 0.05' in H3_RUNG_FLOOR.md -- an "
    "absolute slack in an unnamed margin with no stated true value; the same "
    "document's rung table gave the c_Z*r^2 record instead.",
    "NOT SEARCHED, on the standing rules: 02_LEGACY_Q0_ARCHIVE (zero "
    "evidentiary authority) and the folder 99_DO_NOT_OPEN, which was neither "
    "opened nor listed.",
    "COVERAGE GAP, stated rather than filled: no record lands in the ROUTINE "
    "[2, 10) or LOOSE [10, 1e3) bands. The registry is bimodal -- two records "
    "under 2x, two violations, and three at 5e3 or worse. Whether that is the "
    "corpus or the search is not established here.",
)


# --------------------------------------------------------------------------
# auditing and reporting
# --------------------------------------------------------------------------

def audit_registry(records: Sequence[SlackRecord] = None) -> List[str]:
    """Re-check every invariant across a set of records. Returns problems.

    Construction already enforces these per record; this re-runs them over a
    whole registry so a test, or CI, can assert the list is empty without
    relying on nothing having been mutated since construction.
    """
    recs = REGISTRY if records is None else records
    problems: List[str] = []
    for r in recs:
        if r.known_unsound:
            w = r.unsound_witness
            if w is None:
                problems.append(f"{r.bound_name}: unsound without a witness")
            elif not w.violates():
                problems.append(
                    f"{r.bound_name}: unsound witness does not recompute to a "
                    f"violation -- looseness conflated with unsoundness"
                )
            elif not w.source.strip():
                problems.append(f"{r.bound_name}: unsound witness has no source")
        elif r.unsound_witness is not None and r.unsound_witness.violates():
            problems.append(f"{r.bound_name}: a proved violation is recorded as sound")
        if isinstance(r.claimed, float) or isinstance(r.attained, float):
            problems.append(f"{r.bound_name}: float in a value field")
        for fname in ("bound_name", "carrier", "source_document", "source_quote"):
            if not str(getattr(r, fname)).strip():
                problems.append(f"{r.bound_name}: unsourced ({fname} empty)")
        ratio = r.slack_ratio()
        if ratio.lo < 0:
            problems.append(f"{r.bound_name}: negative slack ratio")
        if (not r.known_unsound and ratio.hi < 1
                and r.ratio_kind() in (RatioKind.ENCLOSURE_OF_TRUE_SLACK,
                                       RatioKind.UPPER_BOUND_ON_TRUE_SLACK)):
            problems.append(f"{r.bound_name}: ratio below 1 but not marked unsound")
    return problems


def sorted_records(records: Sequence[SlackRecord] = None) -> List[SlackRecord]:
    """Descending by slack ratio (lower endpoint), then by name.

    Sorting by the lower endpoint keeps the order a statement about what is
    established, not about how wide an enclosure happens to be.
    """
    recs = list(REGISTRY if records is None else records)
    return sorted(recs, key=lambda r: (-r.slack_ratio().lo, r.bound_name))


def render_report(records: Sequence[SlackRecord] = None, width: int = 100) -> str:
    """The registry as text, sorted by slack ratio, worst utility first.

    The honesty rule is printed on every run, above the table, because a table
    of enormous numbers read without it is exactly the misreading this module
    exists to prevent. The soundness column is physically separated from the
    slack columns by a gap, and it reads ``-`` for every record whose
    unsoundness nobody has established -- not ``ok``, not ``sound``, because
    this registry does not certify soundness either.
    """
    recs = sorted_records(records)
    out: List[str] = []
    rule = "=" * width
    out.append(rule)
    out.append("BOUND-SLACK REGISTRY  (research/slack)")
    out.append(rule)
    for line in _wrap(HONESTY_RULE, width):
        out.append(line)
    out.append("")
    for line in _wrap(DOES_NOT_ESTABLISH, width):
        out.append(line)
    out.append(rule)
    out.append("")

    hdr = (f"{'#':>2}  {'bound':<44} {'slack ratio':>22} {'band':<20}"
           f"   {'arith':<9} {'unsound?':<9}")
    out.append(hdr)
    out.append("-" * len(hdr))
    for i, r in enumerate(recs, 1):
        band = r.severity().value + ("" if r.severity_is_sharp() else " (straddles)")
        arith = "CERT" if r.certifying() else "NON-CERT"
        unsound = "YES (src)" if r.known_unsound else "-"
        out.append(
            f"{i:>2}  {_clip(r.bound_name, 44):<44} {r.ratio_display():>22} "
            f"{band:<20}   {arith:<9} {unsound:<9}"
        )
    out.append("")

    out.append("PER-RECORD DETAIL")
    out.append("-" * len(hdr))
    for i, r in enumerate(recs, 1):
        rr = r.slack_ratio()
        if max(abs(rr.lo - 1), abs(rr.hi - 1)) < Fraction(1, 1000):
            # floor(log10(1 - 7e-18)) is -1, which read as "order 1e-1" would
            # suggest a ratio near a tenth. Say what is actually true instead.
            oom = None
        else:
            lo_e, hi_e = r.order_of_magnitude()
            oom = f"1e{lo_e}" if lo_e == hi_e else f"1e{lo_e}..1e{hi_e}"
        out.append(f"[{i}] {r.bound_name}")
        out.append(f"     direction      : {r.direction.value} bound")
        out.append(f"     claimed        : {r.claimed!r}")
        out.append(f"       provenance   : {r.claimed_provenance.tag} -- "
                   f"{r.claimed_provenance.value}")
        out.append(f"     attained       : {r.attained!r}")
        out.append(f"       provenance   : {r.attained_provenance.tag} -- "
                   f"{r.attained_provenance.value}")
        out.append(f"       kind         : {r.attained_kind.value}")
        out.append(f"     slack ratio    : {r.ratio_display()}   "
                   + (f"(order of magnitude {oom})" if oom
                      else "(within 1e-3 of exact tightness)"))
        out.append(f"       meaning      : {r.ratio_kind().value}")
        out.append(f"     utility band   : {r.severity().value}"
                   f"{'' if r.severity_is_sharp() else '  (endpoints straddle a band edge)'}")
        out.append(f"     unsoundness    : "
                   f"{'ESTABLISHED BY SOURCE' if r.known_unsound else 'not established by any source; this registry claims none'}")
        if r.known_unsound and r.unsound_witness is not None:
            w = r.unsound_witness
            out.append(f"       witness src  : {w.source}")
            for line in _wrap(w.statement, width - 9, indent=9):
                out.append(line)
            out.append(f"       re-verified  : violation margin {sci(w.margin())}")
        out.append(f"     carrier        :")
        for line in _wrap(r.carrier, width - 9, indent=9):
            out.append(line)
        out.append(f"     source         :")
        for line in _wrap(r.source_document, width - 9, indent=9):
            out.append(line)
        out.append(f"     quoted         :")
        for line in _wrap('"' + r.source_quote + '"', width - 9, indent=9):
            out.append(line)
        if r.obligations_untouched:
            out.append(f"     still OPEN     : {', '.join(r.obligations_untouched)}")
        if r.notes:
            out.append(f"     notes          :")
            for line in _wrap(r.notes, width - 9, indent=9):
                out.append(line)
        out.append("")

    out.append(rule)
    out.append("WHAT WAS SEARCHED")
    out.append(rule)
    for entry in SEARCH_LOG:
        for line in _wrap("* " + entry, width):
            out.append(line)
        out.append("")

    problems = audit_registry(recs)
    out.append(rule)
    out.append(f"registry audit: {len(problems)} problem(s)"
               + ("" if not problems else " -- " + "; ".join(problems)))
    out.append(rule)
    return "\n".join(out)


def report(records: Sequence[SlackRecord] = None, width: int = 100) -> None:
    """Print :func:`render_report`."""
    print(render_report(records, width))


def forbidden_words_in_report(text: str = None) -> List[str]:
    """Correctness vocabulary appearing in the report's SLACK columns.

    A test hook. Prose fields (``source_quote``, ``notes``, ``statement``) may
    and do contain correctness words -- envelope_v's source literally says the
    expression "is not an upper bound at all" -- because quoting a source
    accurately is not the same as this module editorialising. What is pinned is
    the generated table: band names, ratio displays and column headers, the
    parts a reader skims. Those must stay free of correctness vocabulary.
    """
    if text is None:
        text = render_report()
    generated: List[str] = []
    for sev in Severity:
        generated.append(sev.value)
    for rec in REGISTRY:
        generated.append(rec.ratio_display())
    generated.append(HONESTY_RULE)
    hay = " ".join(generated).lower()
    return [w for w in FORBIDDEN_CORRECTNESS_WORDS if w in hay]


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "~"


def _wrap(s: str, width: int, indent: int = 0) -> List[str]:
    pad = " " * indent
    words, lines, cur = s.split(), [], pad
    for w in words:
        if cur.strip() and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = pad + w
        else:
            cur = (cur + " " + w) if cur.strip() else (pad + w)
    if cur.strip():
        lines.append(cur)
    return lines


if __name__ == "__main__":  # pragma: no cover
    report()
