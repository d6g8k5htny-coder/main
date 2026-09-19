"""The ``OBL-H5-JETMOD`` falsifier, as an executable check.

THE RULE, AS THE SOURCE STATES IT. ``docs/OPEN_PROBLEMS.md`` A1, quoting
``CHART_SIDE_JETMOD_PLAN.md`` quoting ``H5_PROMOTE.md`` §3:

    "**Falsifier.** A band enclosure whose width exceeds the claimed modulus."

That is the whole rule, and :func:`falsifies` is exactly it: given the width of
a band enclosure and the claimed modulus for that band, the obligation is
falsified on that band when the width **exceeds** the modulus. Strictly
exceeds; equality is not a falsification. Both quantities are exact
``Fraction``s and the comparison is exact, so no floating-point margin decides a
verdict.

THE DECISION IS EXACT; THE PRINTED REPORT IS NOT. :func:`format_report` renders
a decimal view of each pair, and a decimal view is lossy: two rows differing in
the sixteenth digit print the same string and can carry opposite verdicts. That
rendering is labelled NON-CERTIFYING in the output itself and every row also
prints its exact ``Fraction``s. Quote the exact line, not the column.

INSUFFICIENT_DATA IS A FIRST-CLASS OUTCOME. A band for which either the
enclosure width or the claimed modulus is missing is **not** a pass. It has not
been checked. Treating an unevaluated band as passing is the precise failure
mode this repository's status discipline exists to prevent, and the code refuses
to do it: :func:`band_verdict` returns ``INSUFFICIENT_DATA`` and
:func:`report_is_clean` is ``False`` whenever any band is in that state.

WHAT THIS MODULE DOES **NOT** ESTABLISH
---------------------------------------
* A ``PASS`` here is **not** a discharge of ``OBL-H5-JETMOD`` and is not
  evidence for it. ``PASS`` means one width was compared with one claimed
  modulus and came out no larger. The obligation asks for certified intervals
  for the *full 24-jet set* over the program's *own* r-bands with the
  program's *own* kernel and lattice-tail constants. None of those is bound in
  this repository. A whole table of ``PASS`` rows would leave the obligation
  exactly as OPEN as it is now.
* The "claimed modulus" is an **input**. This module neither derives, verifies
  nor endorses any modulus. Supplying one is the caller's responsibility and
  its provenance is the caller's to state. In particular the constants in
  ``research/bands/ladder.py`` are what a *reading* of a *display* would force;
  they are not a source's claimed modulus and must not be passed here as one
  without saying so.
* A ``FALSIFIED`` row is likewise not a refutation of the program. It says the
  pair (width, modulus) handed to this function failed the source's own test.
  Which of the two is at fault -- a loose enclosure, a mis-stated modulus, a
  wrong band -- this module cannot and does not say.

Standard library only (``fractions``, ``dataclasses``, ``typing``). Python 3.11.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
from typing import Dict, Iterable, List, Optional, Sequence

__all__ = [
    "PASS", "FALSIFIED", "INSUFFICIENT_DATA", "OUTCOMES",
    "FALSIFIER_RULE",
    "BandCheck", "BandResult",
    "falsifies", "band_verdict", "band_report", "report_summary",
    "report_is_clean", "format_report", "RENDERING_NOTE",
]

PASS = "PASS"
FALSIFIED = "FALSIFIED"
INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

#: The three outcomes. There is no fourth, and in particular there is no
#: outcome that means "not checked, assume fine".
OUTCOMES = (PASS, FALSIFIED, INSUFFICIENT_DATA)

FALSIFIER_RULE = (
    "A band enclosure whose width exceeds the claimed modulus."
)


def falsifies(enclosure_width: F, claimed_modulus: F) -> bool:
    """``True`` iff this band falsifies the obligation, by the source's own rule.

    The rule, verbatim from the source: *"A band enclosure whose width exceeds
    the claimed modulus."* So::

        falsifies(w, m)  <=>  w > m

    Strict: a width exactly equal to the modulus does **not** exceed it and does
    not falsify. Both arguments must be non-negative exact ``Fraction``s -- a
    negative width is not a width and a negative modulus is not a modulus, and
    both raise rather than produce a verdict.

    ``None`` is refused here on purpose. "I could not evaluate this band" has no
    boolean answer: ``False`` would read as "does not falsify", which is how an
    unchecked band gets silently counted as fine. Use :func:`band_verdict`,
    which has a third outcome for exactly this.

    ``float`` is refused too, for the same reason ``Interval.__contains__``
    refuses it: a binary double is not an exact width and a verdict decided by
    one is not exact. Convert first (``Fraction("0.5")``, or
    ``Fraction(the_float)`` if the double really is what is meant).
    """
    w = _exact_nonneg(enclosure_width, "enclosure_width")
    m = _exact_nonneg(claimed_modulus, "claimed_modulus")
    return w > m


def _exact_nonneg(x, what: str) -> F:
    if x is None:
        raise TypeError(
            f"{what} is None: an unevaluated band has no boolean verdict. Use "
            "band_verdict(), whose third outcome is INSUFFICIENT_DATA."
        )
    if isinstance(x, float):
        raise TypeError(
            f"{what} was a float; this check is exact. Convert it first "
            "(Fraction(\"0.5\"), or Fraction(the_float) if the double really "
            "is what is meant)."
        )
    if isinstance(x, bool) or not isinstance(x, (int, F)):
        raise TypeError(f"{what} must be an exact Fraction or int, got {type(x).__name__}")
    v = F(x)
    if v < 0:
        raise ValueError(f"{what} must be non-negative, got {v}")
    return v


def band_verdict(
    enclosure_width: Optional[F],
    claimed_modulus: Optional[F],
) -> str:
    """One of :data:`PASS`, :data:`FALSIFIED`, :data:`INSUFFICIENT_DATA`.

    ``INSUFFICIENT_DATA`` when **either** input is ``None`` -- an enclosure that
    could not be computed, or a band for which no modulus has been claimed.
    Otherwise the verdict is :func:`falsifies`.

    An ``INSUFFICIENT_DATA`` band is never a ``PASS``. That is asserted by
    ``tests/test_bands.py`` as a negative control, because it is the assertion
    most worth protecting: everything in this program that has gone wrong has
    gone wrong by an unchecked thing being counted as a checked one.
    """
    if enclosure_width is None or claimed_modulus is None:
        return INSUFFICIENT_DATA
    return FALSIFIED if falsifies(enclosure_width, claimed_modulus) else PASS


@dataclass(frozen=True)
class BandCheck:
    """One band submitted for checking.

    ``width`` is the width of a certified band enclosure (for example
    ``research.bands.lattice.BandEnclosure.width()``). ``claimed_modulus`` is
    the modulus claimed for that band by whoever claims it; its provenance
    belongs in ``modulus_source`` and this module does not check it.

    Either may be ``None``, which is how a band that could not be evaluated is
    represented. ``reason`` records why.
    """

    band: str
    width: Optional[F] = None
    claimed_modulus: Optional[F] = None
    modulus_source: str = "UNSTATED"
    reason: str = ""


@dataclass(frozen=True)
class BandResult:
    """The verdict on one band, with everything needed to re-derive it."""

    band: str
    verdict: str
    width: Optional[F]
    claimed_modulus: Optional[F]
    modulus_source: str
    reason: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "band": self.band,
            "verdict": self.verdict,
            "width": None if self.width is None else str(self.width),
            "claimed_modulus": (
                None if self.claimed_modulus is None else str(self.claimed_modulus)
            ),
            "modulus_source": self.modulus_source,
            "reason": self.reason,
            "rule": FALSIFIER_RULE,
        }


def band_report(checks: Iterable[BandCheck]) -> List[BandResult]:
    """Run :func:`band_verdict` over a ladder of bands, one row per band.

    Every submitted band produces exactly one row. No band is skipped, dropped,
    merged or defaulted: a band that cannot be evaluated appears with
    ``INSUFFICIENT_DATA`` and its ``reason``, which is how the caller sees that
    it is still outstanding rather than never seeing it at all.
    """
    rows: List[BandResult] = []
    for c in checks:
        verdict = band_verdict(c.width, c.claimed_modulus)
        reason = c.reason
        if verdict == INSUFFICIENT_DATA and not reason:
            missing = []
            if c.width is None:
                missing.append("no enclosure width")
            if c.claimed_modulus is None:
                missing.append("no claimed modulus")
            reason = "; ".join(missing)
        rows.append(BandResult(
            band=c.band,
            verdict=verdict,
            width=c.width,
            claimed_modulus=c.claimed_modulus,
            modulus_source=c.modulus_source,
            reason=reason,
        ))
    return rows


def report_summary(rows: Sequence[BandResult]) -> Dict[str, int]:
    """Counts per outcome. Every outcome key is present, even at zero."""
    counts = {o: 0 for o in OUTCOMES}
    for r in rows:
        counts[r.verdict] += 1
    return counts


def report_is_clean(rows: Sequence[BandResult]) -> bool:
    """``True`` only when every band PASSED.

    An ``INSUFFICIENT_DATA`` row makes this ``False``. A report with unchecked
    bands is not a clean report, and a caller that wants to know "did anything
    fail" must not be allowed to conflate "nothing failed" with "nothing was
    checked".

    ``True`` here still means nothing beyond "every submitted pair (width,
    modulus) satisfied the source's inequality". It is not a discharge of
    ``OBL-H5-JETMOD``. See this module's docstring.
    """
    return bool(rows) and all(r.verdict == PASS for r in rows)


#: Printed on every rendered report. The decimal columns are a lossy view of
#: exact Fractions and are never what decides a verdict.
RENDERING_NOTE = (
    "NON-CERTIFYING RENDERING. The decimal columns below are float(Fraction) "
    "views, six significant digits, for reading only. They are LOSSY: two rows "
    "carrying opposite verdicts can print identical decimals, because the "
    "verdict is decided on the exact Fractions and the display is not. The "
    "exact values are printed underneath each row and are the only thing to "
    "quote. Nothing in this rendering is a bound."
)


def _exact_pair(r: BandResult) -> str:
    """The exact Fractions behind one row, as the string a reader should quote."""
    w = "None" if r.width is None else str(r.width)
    m = "None" if r.claimed_modulus is None else str(r.claimed_modulus)
    return f"exact: width={w} modulus={m}"


def format_report(rows: Sequence[BandResult]) -> str:
    """A plain-text table, NON-PROMOTING, with the rule and the caveat printed.

    THE DECIMAL COLUMNS ARE A NON-CERTIFYING RENDERING AND THE OUTPUT SAYS SO.
    The verdict is computed by :func:`falsifies` on exact ``Fraction``s, so no
    floating-point margin decides it -- but ``%.6g`` of two Fractions that
    differ in the 16th digit prints the same string twice, and this report is
    the artifact a human pastes into a status discussion. Every row therefore
    also prints its exact numerator/denominator pair, and the header carries
    :data:`RENDERING_NOTE`. Repository rule: every float path is labelled
    NON-CERTIFYING in code AND in output.
    """
    out = [
        "OBL-H5-JETMOD falsifier report",
        "rule: " + FALSIFIER_RULE,
        "",
    ]
    out.extend("  " + line for line in _wrap(RENDERING_NOTE, 76))
    out.append("=" * 78)
    for r in rows:
        w = "-" if r.width is None else f"{float(r.width):.6g}"
        m = "-" if r.claimed_modulus is None else f"{float(r.claimed_modulus):.6g}"
        out.append(f"  {r.band:<28} width~={w:<13} modulus~={m:<13} {r.verdict}")
        out.append(f"       {_exact_pair(r)}")
        if r.reason:
            out.append(f"       reason: {r.reason}")
        if r.modulus_source:
            out.append(f"       modulus source: {r.modulus_source}")
    counts = report_summary(rows)
    out.append("-" * 78)
    out.append("  " + "  ".join(f"{k}={counts[k]}" for k in OUTCOMES))
    out.append("")
    out.append("  The width~= and modulus~= columns are NON-CERTIFYING decimal")
    out.append("  views. Read the 'exact:' line under each row.")
    out.append("  A PASS row is NOT a discharge of OBL-H5-JETMOD and is not")
    out.append("  evidence for it. The obligation is over the full 24-jet set,")
    out.append("  the program's kernel and the program's r-bands, none of which")
    out.append("  is bound in this repository. OBL-H5-JETMOD stays OPEN.")
    out.append("  An INSUFFICIENT_DATA row is an outstanding band, not a pass.")
    return "\n".join(out)


def _wrap(text: str, width: int) -> List[str]:
    """Greedy word wrap. Display only; no arithmetic depends on it."""
    words, lines, cur = text.split(), [], ""
    for wd in words:
        if cur and len(cur) + 1 + len(wd) > width:
            lines.append(cur)
            cur = wd
        else:
            cur = f"{cur} {wd}" if cur else wd
    if cur:
        lines.append(cur)
    return lines
