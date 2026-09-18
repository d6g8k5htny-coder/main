"""The published H5 rung ladder as exact data, plus the display-vs-enclosure gap.

WHAT THIS MODULE IS. Three published `I_hi/r^3` **point** certifications, held
as exact ``Fraction`` values with their source package and, where the rung docs
quote one, their totals digest; plus two pieces of arithmetic on them that make
the display-vs-enclosure distinction numerical rather than rhetorical:

  1. the **same-r version spread** at ``r = 0.05``, where the frozen v1 line and
     the live v3 clean line disagree at the *same* separation;
  2. the **implied modulus constant** ``C`` in ``|Delta(I_hi/r^3)| <= C*delta^kappa``
     forced by each adjacent pair of published points, and the ratio between the
     two adjacent bands.

THE ASSUMPTION BEHIND (2), STATED EVERY TIME IT IS REPORTED.
``IMPLIED_MODULUS_ASSUMPTION`` below is attached to every value this module
returns and is reproduced in every formatted report. In one line: reading (2) as
a modulus of continuity ``|Delta(I_hi/r^3)| <= C*delta^kappa`` on the quantity
``I_hi/r^3``, with ``kappa`` taken from the displayed ``kappa = 1/8`` fit, is the
*natural* reading of the shipped display -- it is **not** a quotation of any
source's own modulus statement. The sources' displayed modulus is a dense
certified sampling of ``c_2`` with an explicit fit, labelled DISPLAY. If the
intended modulus is on a different quantity, or is relative rather than
absolute, or carries a different exponent, the numbers below change. That is why
:func:`implied_modulus_constant` takes ``kappa`` as an argument instead of
hard-coding ``1/8``, and why :func:`ratio_table` exists.

WHAT THIS MODULE DOES **NOT** ESTABLISH
---------------------------------------
* It is **not** a refutation of anything, and must never be presented as one.
  It does not show the displayed modulus is wrong. A modulus on a different
  quantity, or in relative form, is untouched by these numbers.
* It does **not** discharge, reduce, close, promote or reclassify
  ``OBL-H5-JETMOD``, which stays OPEN (display only), nor ``OBL-H5-ZBAND``,
  ``OBL-H5-REMOTE-THRESHOLD``, ``OBL-D1-PROMOTE`` or either Piece of
  ``D3-LEMMA-RN-UNIF``.
* The three published values remain exactly what their sources say they are:
  **point certifications at three separations**. Reproducing arithmetic on them
  neither strengthens nor weakens them.
* It supplies **no band enclosure**. A band enclosure is the thing
  ``OBL-H5-JETMOD`` asks for and this module does not compute one. What it shows
  is why the point ladder cannot stand in for one: the supremum over a band is
  not constrained by the values at its endpoints absent an independently
  certified modulus, and these three points do not themselves exhibit a single
  constant under the natural reading.
* The ladder's engineering status is untouched: ``r = 0.0177`` stands at 42/70
  cells and ``r = 0.0125`` at 21/70, and no line count anywhere promotes that.

Standard library only (``fractions``, ``dataclasses``, ``typing``), plus
``research.interval`` for the certified fractional powers. Python 3.11.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction as F
from typing import Dict, List, Optional, Sequence, Tuple

from research.interval import Interval, exp, log

__all__ = [
    "RungPoint", "AdjacentBand", "ImpliedModulus",
    "PUBLISHED_POINTS", "LINE_LIVE_V3", "LINE_FROZEN_V1",
    "DISPLAYED_KAPPA", "IMPLIED_MODULUS_ASSUMPTION", "STATUS_NOTE",
    "points_for_line", "same_r_version_spread", "adjacent_bands",
    "implied_modulus_constant", "implied_modulus_table",
    "implied_modulus_ratio", "ratio_table", "format_report",
]

LINE_LIVE_V3 = "live-v3-clean"
LINE_FROZEN_V1 = "frozen-v1"

#: The exponent shown by the shipped modulus display (dense certified sampling
#: of c_2 plus an explicit fit). Carried here only as the default argument of
#: the analysis functions. A displayed fitted exponent is NOT a certified
#: exponent and this module never treats it as one.
DISPLAYED_KAPPA = F(1, 8)

IMPLIED_MODULUS_ASSUMPTION = (
    "ASSUMPTION (state it whenever these numbers are shown). The constant C "
    "below is what the NATURAL READING of the shipped display forces: a "
    "modulus of continuity |Delta(I_hi/r^3)| <= C * delta^kappa on the "
    "quantity I_hi/r^3, with kappa taken from the displayed kappa = 1/8 fit "
    "(itself a dense certified SAMPLING of c_2 plus an explicit fit, labelled "
    "DISPLAY by its own source). It is NOT a quotation of any source's own "
    "modulus statement. If the intended modulus is on a different quantity, or "
    "is relative rather than absolute, or carries a different exponent, these "
    "numbers change. Nothing here refutes the displayed modulus, and nothing "
    "here discharges OBL-H5-JETMOD, which stays OPEN (display only)."
)

STATUS_NOTE = (
    "The three values below are POINT certifications at three separations. "
    "They are not a band enclosure and they do not become one by being "
    "differenced. OBL-H5-JETMOD, OBL-H5-ZBAND (hi side), "
    "OBL-H5-REMOTE-THRESHOLD and OBL-D1-PROMOTE (chart side) all remain OPEN."
)


# ---------------------------------------------------------------------------
# The published data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RungPoint:
    """One published ``I_hi/r^3`` point certification.

    ``r`` and ``value`` are exact ``Fraction`` transcriptions of the decimal
    strings printed by the source packages -- decimal strings, so the
    ``Fraction`` is the literal exactly, with no binary rounding anywhere.

    ``totals_digest`` is the sha256 the rung document quotes for its own totals
    JSON, or ``None`` where the document quotes none. ``None`` is not a defect
    to be filled in: the ``r = 0.05`` values are *cited* in both rung docs
    rather than certified by them, and inventing a digest for them would be
    exactly the kind of quiet upgrade this program forbids.
    """

    r: F
    value: F
    line: str
    package: str
    totals_digest: Optional[str]
    note: str = ""


PUBLISHED_POINTS: Tuple[RungPoint, ...] = (
    RungPoint(
        r=F("0.025"),
        value=F("664.3979"),
        line=LINE_LIVE_V3,
        package="H5_RUNG2_2026-09-15.md",
        totals_digest=(
            "f7697bcfa0fe32b5c87c8ef8adef5d4384ab1bda7cd4a7c4782fef8c99103fcb"
        ),
        note="totals file h5_totals_r0.025_v2.json; doc-cited, repro PASS.",
    ),
    RungPoint(
        r=F("0.035355"),
        value=F("661.4712"),
        line=LINE_LIVE_V3,
        package="H5_RUNG3_2026-09-15.md",
        totals_digest=(
            "808d6901e3254a73181aa696904ed04567d401b6e58300454d489046c36f4f64"
        ),
        note="totals file h5_totals_r0.035355_v2.json; doc-cited, repro PASS.",
    ),
    RungPoint(
        r=F("0.05"),
        value=F("647.8048"),
        line=LINE_LIVE_V3,
        package="cited in both H5_RUNG2 and H5_RUNG3",
        totals_digest=None,
        note=(
            "Cited by the rung docs, not certified by them; no totals digest "
            "is quoted, and none is invented here."
        ),
    ),
    RungPoint(
        r=F("0.05"),
        value=F("731.4311"),
        line=LINE_FROZEN_V1,
        package="cited in both H5_RUNG2 and H5_RUNG3",
        totals_digest=None,
        note=(
            "The frozen v1 line at the same r as the live v3 clean value. Two "
            "engine versions, one separation, two numbers."
        ),
    ),
)


def points_for_line(line: str = LINE_LIVE_V3) -> List[RungPoint]:
    """The published points on one engine line, sorted by increasing ``r``.

    The ``frozen-v1`` line carries a single published point (``r = 0.05``), so
    it has no adjacent pair and no implied modulus. That is data, not an
    omission.
    """
    pts = [p for p in PUBLISHED_POINTS if p.line == line]
    if not pts:
        raise KeyError(f"no published points on line {line!r}")
    return sorted(pts, key=lambda p: p.r)


# ---------------------------------------------------------------------------
# 1. The same-r version spread
# ---------------------------------------------------------------------------

def same_r_version_spread(r: F = F("0.05")) -> Dict[str, object]:
    """The disagreement between two engine lines at ONE separation, exactly.

    At ``r = 0.05`` the frozen v1 line and the live v3 clean line report
    different values of ``I_hi/r^3``. Both are published. The spread is computed
    here from the two ``Fraction`` transcriptions, not copied from any prose.

    ``relative`` is ``|spread| / value(live v3 clean)`` -- relative to the live
    line, because that is the line the ladder's other two points sit on.

    WHAT IT SHOWS AND WHAT IT DOES NOT. It shows that a point certification is
    not by itself stable across engine versions at fixed ``r``, which is a
    reason a band claim needs its own argument. It does **not** say either value
    is wrong, does not adjudicate between the two lines, and does not bear on
    any obligation's status.
    """
    at_r = [p for p in PUBLISHED_POINTS if p.r == r]
    by_line = {p.line: p for p in at_r}
    if LINE_FROZEN_V1 not in by_line or LINE_LIVE_V3 not in by_line:
        raise KeyError(
            f"r = {r} does not carry both a frozen-v1 and a live-v3-clean "
            f"published value; lines present: {sorted(by_line)}"
        )
    frozen = by_line[LINE_FROZEN_V1].value
    live = by_line[LINE_LIVE_V3].value
    spread = frozen - live
    return {
        "r": r,
        "frozen_v1": frozen,
        "live_v3_clean": live,
        "spread": spread,
        "relative": spread / live,
        "relative_percent": spread / live * 100,
        "status_note": STATUS_NOTE,
    }


# ---------------------------------------------------------------------------
# 2. The implied modulus constant
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AdjacentBand:
    """A band ``[r_lo, r_hi]`` spanned by two adjacent published points."""

    r_lo: F
    r_hi: F
    value_lo: F
    value_hi: F
    package_lo: str
    package_hi: str

    @property
    def name(self) -> str:
        """Exact rational endpoints -- the identity of the band."""
        return f"[{self.r_lo}, {self.r_hi}]"

    @property
    def label(self) -> str:
        """A decimal rendering for reading. The exact endpoints are ``name``."""
        return f"[{float(self.r_lo):.6f}, {float(self.r_hi):.6f}]"

    @property
    def delta(self) -> F:
        """The band width ``r_hi - r_lo``, exactly."""
        return self.r_hi - self.r_lo

    @property
    def abs_diff(self) -> F:
        """``|Delta(I_hi/r^3)|`` across the band, exactly."""
        d = self.value_hi - self.value_lo
        return -d if d < 0 else d


def adjacent_bands(line: str = LINE_LIVE_V3) -> List[AdjacentBand]:
    """The bands spanned by consecutive published points on one line.

    These are **not** the program's r-bands ``[r_{k+1}, r_k]``. The program's
    band endpoints are not bound in this repository; these are simply the
    intervals between the published points, used to ask what a modulus would
    have to be to connect them.
    """
    pts = points_for_line(line)
    out: List[AdjacentBand] = []
    for a, b in zip(pts, pts[1:]):
        out.append(AdjacentBand(
            r_lo=a.r, r_hi=b.r,
            value_lo=a.value, value_hi=b.value,
            package_lo=a.package, package_hi=b.package,
        ))
    return out


@dataclass(frozen=True)
class ImpliedModulus:
    """The constant one band forces, as a certified interval, with its assumption.

    ``constant`` encloses ``|Delta| / delta^kappa``. It is an enclosure and not
    a single rational because ``delta^kappa`` is irrational for the interesting
    ``kappa``; the enclosure is produced by ``research/interval`` and is
    certified in that package's sense.

    ``assumption`` is not decoration. Every constructor of this class fills it
    with :data:`IMPLIED_MODULUS_ASSUMPTION`, and every formatter prints it, so
    the number cannot travel without the reading it depends on.
    """

    band: AdjacentBand
    kappa: F
    delta_pow_kappa: Interval
    constant: Interval
    prec: int
    assumption: str = IMPLIED_MODULUS_ASSUMPTION

    def to_dict(self) -> Dict[str, object]:
        return {
            "band": self.band.name,
            "delta": str(self.band.delta),
            "abs_delta_value": str(self.band.abs_diff),
            "kappa": str(self.kappa),
            "delta_pow_kappa": [
                str(self.delta_pow_kappa.lo), str(self.delta_pow_kappa.hi)
            ],
            "forced_constant": [str(self.constant.lo), str(self.constant.hi)],
            "prec": self.prec,
            "assumption": self.assumption,
            "status_note": STATUS_NOTE,
        }


def _delta_pow_kappa(delta: F, kappa: F, prec: int) -> Interval:
    """Certified enclosure of ``delta ** kappa`` for ``delta > 0`` and rational ``kappa``.

    ``kappa = 0`` is exactly ``1``. An integer ``kappa`` takes the exact
    algebraic path. Otherwise ``delta^kappa = exp(kappa * log(delta))``, with
    both ``log`` and ``exp`` certified by ``research/interval``, so the
    composition is certified.

    For ``kappa = 1/8`` there is an independent route -- three nested certified
    square roots -- and ``tests/test_bands.py`` checks that the two agree, which
    is a genuine cross-check of this function rather than a restatement of it.
    """
    if delta <= 0:
        raise ValueError("delta must be positive to raise it to a real power")
    if kappa == 0:
        return Interval.exact(F(1))
    d = Interval.exact(delta)
    if kappa.denominator == 1:
        return (d ** int(kappa.numerator)).round_out(4 * max(prec, 1) + 64)
    return exp(log(d, prec) * kappa, prec).round_out(4 * max(prec, 1) + 64)


def implied_modulus_constant(
    band: AdjacentBand,
    kappa: F = DISPLAYED_KAPPA,
    prec: int = 40,
) -> ImpliedModulus:
    """The constant ``C`` that this band forces under the stated assumption.

    Under ``|Delta(I_hi/r^3)| <= C * delta^kappa`` a band with width ``delta``
    and endpoint difference ``|Delta|`` forces

        C >= |Delta| / delta^kappa,

    and that quotient is what is enclosed here: the exact rational ``|Delta|``
    divided by a certified enclosure of ``delta^kappa``. The enclosure contains
    the true forced value, so its lower endpoint is itself a certified lower
    bound on any admissible ``C``.

    ``kappa`` is an argument. Nothing here is hard-coded to ``1/8``; the default
    merely matches the exponent the shipped display shows. See
    :data:`IMPLIED_MODULUS_ASSUMPTION`, which is attached to the result.
    """
    if not isinstance(kappa, F):
        kappa = F(kappa)
    dpk = _delta_pow_kappa(band.delta, kappa, prec)
    if dpk.lo <= 0:
        raise ValueError(f"delta^kappa enclosure {dpk!r} is not positive; raise prec")
    return ImpliedModulus(
        band=band,
        kappa=kappa,
        delta_pow_kappa=dpk,
        constant=Interval.exact(band.abs_diff) / dpk,
        prec=prec,
    )


def implied_modulus_table(
    kappa: F = DISPLAYED_KAPPA,
    prec: int = 40,
    line: str = LINE_LIVE_V3,
) -> List[ImpliedModulus]:
    """:func:`implied_modulus_constant` over every adjacent band on one line."""
    return [implied_modulus_constant(b, kappa, prec) for b in adjacent_bands(line)]


def implied_modulus_ratio(
    kappa: F = DISPLAYED_KAPPA,
    prec: int = 40,
    line: str = LINE_LIVE_V3,
) -> Dict[str, object]:
    """The ratio of the two adjacent bands' forced constants, as an enclosure.

    With exactly two adjacent bands the ratio is
    ``C(wider band) / C(narrower band)``, computed as an interval quotient of
    the two enclosures, so the returned interval contains the true ratio.

    Closed form, useful for reading the ``kappa`` dependence off:

        ratio(kappa) = (|Delta_wide| / |Delta_narrow|)
                       * (delta_narrow / delta_wide) ** kappa.

    Since ``delta_narrow < delta_wide``, the second factor is strictly
    decreasing in ``kappa``, so ``ratio`` is strictly decreasing in ``kappa``.
    At ``kappa = 0`` it is the bare ratio of endpoint differences. That
    monotonicity is why :func:`ratio_table` is the honest way to report this:
    a single number at a single ``kappa`` hides the dependence.

    This is an observation about published numbers under a stated reading. It
    is not a refutation, not a defect report, and not a status change.
    """
    table = implied_modulus_table(kappa, prec, line)
    if len(table) != 2:
        raise ValueError(
            f"ratio is defined for exactly two adjacent bands; line {line!r} "
            f"has {len(table)}"
        )
    wide, narrow = max(table, key=lambda t: t.band.delta), min(
        table, key=lambda t: t.band.delta)
    return {
        "kappa": kappa,
        "wide_band": wide.band.name,
        "narrow_band": narrow.band.name,
        "wide_constant": wide.constant,
        "narrow_constant": narrow.constant,
        "ratio": wide.constant / narrow.constant,
        "assumption": IMPLIED_MODULUS_ASSUMPTION,
        "status_note": STATUS_NOTE,
    }


def ratio_table(
    kappas: Sequence[F] = (F(0), F(1, 8), F(1, 2), F(1), F(2), F(4), F(5)),
    prec: int = 40,
    line: str = LINE_LIVE_V3,
) -> List[Dict[str, object]]:
    """:func:`implied_modulus_ratio` across a sweep of ``kappa``.

    The point of the sweep is that the analysis is **not** hardcoded to the
    displayed ``kappa = 1/8``: the reader can see how the disagreement between
    the two bands' forced constants depends on the exponent assumed, and can
    see that a large enough ``kappa`` would equalise them. Reporting one value
    at one ``kappa`` would present a modelling choice as a fact.
    """
    return [implied_modulus_ratio(k, prec, line) for k in kappas]


# ---------------------------------------------------------------------------
# Formatting -- the assumption travels with the numbers
# ---------------------------------------------------------------------------

def _dec(x: Interval, places: int = 6) -> str:
    """An OUTWARD-rounded decimal view of an enclosure: floor the low end, ceil
    the high end. The rendering is therefore never narrower than the enclosure
    it displays, so reading a digit off this string cannot overstate tightness.
    """
    scale = 10 ** places
    lo = math.floor(x.lo * scale) / scale
    hi = math.ceil(x.hi * scale) / scale
    return f"[{lo:.{places}f}, {hi:.{places}f}]"


def format_report(kappa: F = DISPLAYED_KAPPA, prec: int = 40,
                  line: str = LINE_LIVE_V3) -> str:
    """A plain-text report of both analyses, assumption included, NON-PROMOTING.

    The decimal renderings in the report are outward-rounded views of exact
    rational or certified-interval quantities. They are for reading; the
    returned objects carry the exact values.
    """
    out: List[str] = []
    out.append("H5 rung ladder -- published point certifications (line: %s)" % line)
    out.append("=" * 72)
    for p in points_for_line(line):
        dg = p.totals_digest[:16] + "..." if p.totals_digest else "(no digest quoted)"
        out.append(f"  r = {float(p.r):<10.6f}  I_hi/r^3 = {float(p.value):<12.4f}"
                   f"  {p.package}  totals sha256 {dg}")
    out.append("")
    out.append("These are POINT certifications. They are not a band enclosure.")
    out.append("")

    sp = same_r_version_spread()
    out.append("1. Same-r version spread at r = %s" % sp["r"])
    out.append("-" * 72)
    out.append(f"   frozen v1      {float(sp['frozen_v1']):.4f}")
    out.append(f"   live v3 clean  {float(sp['live_v3_clean']):.4f}")
    out.append(f"   spread         {float(sp['spread']):.4f}  exactly {sp['spread']}")
    out.append(f"   relative       {float(sp['relative_percent']):.4f}% of the live value")
    out.append("   Two engine versions, one separation, two published numbers.")
    out.append("   This adjudicates between them in no way and changes no status.")
    out.append("")

    out.append("2. Implied modulus constant at kappa = %s" % kappa)
    out.append("-" * 72)
    for im in implied_modulus_table(kappa, prec, line):
        out.append(
            f"   band {im.band.label:<24} delta = {float(im.band.delta):.6f}  "
            f"|Delta| = {float(im.band.abs_diff):.4f}  "
            f"forced C >= {_dec(im.constant, 4)}"
        )
    rt = implied_modulus_ratio(kappa, prec, line)
    out.append(f"   ratio of the two forced constants: {_dec(rt['ratio'], 4)}")
    out.append("")
    out.append("   ratio as a function of kappa (it is strictly decreasing):")
    for row in ratio_table(prec=prec, line=line):
        out.append(f"     kappa = {str(row['kappa']):<6}  ratio = {_dec(row['ratio'], 4)}")
    out.append("")
    for chunk in (IMPLIED_MODULUS_ASSUMPTION, STATUS_NOTE):
        out.append("   " + chunk.replace(". ", ".\n   "))
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":  # pragma: no cover - a reading aid, not a check
    print(format_report())
