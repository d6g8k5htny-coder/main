"""The cover ledger: cells, dispositions, the exact partition invariant, totals.

The ledger is the **first-class output** of a cover run. RN5's next exact
action says to retain *every rejected cell* with its boundary-area bound and to
*verify no cell remains pending*; the ledger is the object that makes both
mechanically checkable, and it refuses to hand back a certified total while
either condition is unmet.

COORDINATES. A :class:`Box` is an axis-aligned rectangle in exact rational
**parameter** coordinates ``(u, v)``. What ``(u, v)`` mean is the region's
business: for :class:`~research.cover.regions.CartesianBracketRegion` they are
``(x, y)``; for :class:`~research.cover.regions.PolarRegion` they are
``(radius, turn)`` with ``turn`` measured in whole turns, so ``theta = 2*pi*v``
and the parameter rectangle stays exactly rational even though the geometry
does not. The ledger never interprets them.

THE EXACTNESS INVARIANT, AND WHY IT IS NOT AN AREA COMPARISON
--------------------------------------------------------------
Areas are the wrong instrument. ``sum(area(cell)) == area(domain)`` can hold
with a gap of area ``a`` and an overlap of area ``a`` at the same time -- the
two errors cancel in the sum and the cover is still not a cover. The cheapest
counterexample is in ``tests/test_cover.py`` and is the sharpest control in
this package.

:func:`check_exact_partition` instead decides the question **structurally**,
over ``Fraction``, by coordinate compression:

  1. every cell must lie inside the domain (no cell escapes);
  2. no cell may be degenerate (zero width in either axis);
  3. compress the ``u`` coordinates of the domain and of every cell into a
     sorted list, giving vertical strips. Because every cell's ``u`` endpoints
     are among the compression points, a cell either **covers** a strip or is
     disjoint from its interior -- there is no partial case;
  4. inside each strip, take the ``v`` intervals of the covering cells, sort
     them, and demand that they tile the domain's ``v`` range *exactly*: the
     first ``lo`` equals the domain's ``lo``, each successive ``lo`` equals the
     previous ``hi``, and the last ``hi`` equals the domain's ``hi``. A
     successive ``lo`` strictly below the previous ``hi`` is an **overlap**; one
     strictly above is a **gap**. Both are reported, with the offending
     coordinates.

That decides gaps and overlaps independently of any area, and it is exact:
every comparison is between ``Fraction`` values.

The ledger checks a second, independent invariant as well
(:meth:`Ledger.check_refinements`): every ``REFINED`` cell's children tile that
cell exactly, by the same routine. Cover-wide tiling then also follows by
induction from the roots, so the two checks corroborate each other rather than
sharing a single point of failure.

WHAT THIS MODULE DOES NOT ESTABLISH. Nothing. A partition check is a statement
about rectangles in a parameter rectangle. It is not a mathematical result, it
closes no premise, and it says nothing at all about the integrand whose values
are being summed over those rectangles. See ``research/cover/README.md``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field as dc_field
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

from research.interval import Interval

__all__ = [
    "ACCEPTED", "REFINED", "REJECTED", "PENDING", "DISPOSITIONS",
    "RejectKind", "Box", "Cell", "CellRecord", "Total", "Ledger",
    "PartitionError", "PendingCellsError", "check_exact_partition",
]

ACCEPTED = "ACCEPTED"
REFINED = "REFINED"
REJECTED = "REJECTED"
PENDING = "PENDING"
DISPOSITIONS = (ACCEPTED, REFINED, REJECTED, PENDING)


class RejectKind:
    """Why a cell was rejected. The distinction is load bearing.

    ``OUTSIDE`` -- the cell was proved disjoint from the region by an exact
    rational test. It contributes nothing to the region integral and the total
    stays an enclosure.

    ``UNRESOLVED_BOUNDARY`` -- the cell straddles the region boundary and was
    not resolved within the depth budget. Part of it *is* in the region, so the
    total is only an enclosure if the cell carries a ``residual``: a certified
    enclosure of ``integral over (cell and region)``. Without one the ledger
    reports ``covers_region=False`` and refuses the ``certified`` label --
    though ``total()`` still RETURNS, handing back an object whose field is
    called ``enclosure`` and is not one. Publish through
    :meth:`Total.certified_enclosure`, which refuses instead.

    An ``OUTSIDE`` cell's retained area is NOT boundary area: it is proved
    disjoint from the region and holds no boundary. See
    :attr:`Total.area_unresolved_boundary_bound`.

    ``EXCLUDED`` -- excluded by a stated predicate other than the region test
    (available for callers; nothing in this package uses it).
    """

    OUTSIDE = "OUTSIDE"
    UNRESOLVED_BOUNDARY = "UNRESOLVED_BOUNDARY"
    EXCLUDED = "EXCLUDED"
    ALL = (OUTSIDE, UNRESOLVED_BOUNDARY, EXCLUDED)


class PartitionError(Exception):
    """The cells do not exactly tile the domain: a gap, an overlap or an escape."""


class UncertifiedTotalError(Exception):
    """Raised by :meth:`Total.certified_enclosure` on a Total that is not one.

    The symmetric partner of :class:`PendingCellsError`. That one refuses a
    total while cells are unvisited; this one refuses the WORD "certified" on a
    total whose region is not fully accounted for, or whose arithmetic came off
    a non-certifying path. Both exist because the failure they prevent is the
    same failure: a partial result published as a whole one.
    """


class PendingCellsError(Exception):
    """A certified total was requested while cells were still PENDING.

    This is the failure mode RN5 names when it says the present ten boxes are
    **not** a coverage certificate: a partial cover reported as a total. It
    raises. It never warns and never returns a number.
    """


# --------------------------------------------------------------------- boxes


@dataclass(frozen=True)
class Box:
    """Axis-aligned rectangle ``[u0, u1] x [v0, v1]`` over exact rationals."""

    u0: Fraction
    u1: Fraction
    v0: Fraction
    v1: Fraction

    def __post_init__(self) -> None:
        for name in ("u0", "u1", "v0", "v1"):
            val = getattr(self, name)
            if not isinstance(val, Fraction):
                raise TypeError(
                    f"Box.{name} must be a Fraction (got {type(val).__name__}); "
                    "float endpoints are refused on purpose"
                )
        if self.u0 > self.u1 or self.v0 > self.v1:
            raise ValueError(f"empty box {self!r}")

    @classmethod
    def of(cls, u0, u1, v0, v1) -> "Box":
        """Build from anything ``Fraction`` accepts exactly (``int``, ``str``,
        ``Fraction``). ``float`` is refused by ``Fraction`` itself only for
        strings; here we refuse it explicitly."""
        def f(x):
            if isinstance(x, float):
                raise TypeError(
                    "float box endpoints are refused: Fraction(0.1) is not 1/10"
                )
            return Fraction(x)

        return cls(f(u0), f(u1), f(v0), f(v1))

    # ------------------------------------------------------------ inspection

    def du(self) -> Fraction:
        return self.u1 - self.u0

    def dv(self) -> Fraction:
        return self.v1 - self.v0

    def param_area(self) -> Fraction:
        """Area **in parameter coordinates**, exactly.

        For a Cartesian region this is the geometric area. For a polar region
        it is *not*: see ``PolarRegion.area``. Nothing in the ledger treats it
        as geometric area.
        """
        return self.du() * self.dv()

    def is_degenerate(self) -> bool:
        return self.du() == 0 or self.dv() == 0

    def contains_box(self, other: "Box") -> bool:
        return (self.u0 <= other.u0 and other.u1 <= self.u1
                and self.v0 <= other.v0 and other.v1 <= self.v1)

    def covers_u_strip(self, a: Fraction, b: Fraction) -> bool:
        return self.u0 <= a and b <= self.u1

    # ------------------------------------------------------------- splitting

    def split_u(self) -> Tuple["Box", "Box"]:
        m = (self.u0 + self.u1) / 2
        return (Box(self.u0, m, self.v0, self.v1),
                Box(m, self.u1, self.v0, self.v1))

    def split_v(self) -> Tuple["Box", "Box"]:
        m = (self.v0 + self.v1) / 2
        return (Box(self.u0, self.u1, self.v0, m),
                Box(self.u0, self.u1, m, self.v1))

    def quad(self) -> Tuple["Box", "Box", "Box", "Box"]:
        a, b = self.split_u()
        a0, a1 = a.split_v()
        b0, b1 = b.split_v()
        return (a0, a1, b0, b1)

    def as_json(self) -> Dict[str, str]:
        return {"u0": str(self.u0), "u1": str(self.u1),
                "v0": str(self.v0), "v1": str(self.v1)}

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (f"Box([{self.u0}, {self.u1}] x [{self.v0}, {self.v1}])")


@dataclass(frozen=True)
class Cell:
    """A box plus its identity in the subdivision tree."""

    cid: str
    box: Box
    depth: int
    parent: Optional[str] = None


# ---------------------------------------------------- the partition invariant


def check_exact_partition(domain: Box, boxes: Sequence[Box],
                          what: str = "cover") -> None:
    """Raise :class:`PartitionError` unless ``boxes`` tile ``domain`` exactly.

    Exact, structural, and independent of area -- see the module docstring for
    the argument. Complexity is ``O(S * n log n)`` with ``S`` the number of
    compressed strips.
    """
    if not boxes:
        raise PartitionError(f"{what}: empty cover cannot tile {domain!r}")
    if domain.is_degenerate():
        raise PartitionError(f"{what}: degenerate domain {domain!r}")

    for b in boxes:
        if b.is_degenerate():
            raise PartitionError(f"{what}: degenerate cell {b!r}")
        if not domain.contains_box(b):
            raise PartitionError(
                f"{what}: cell {b!r} escapes the domain {domain!r}")

    us = sorted({domain.u0, domain.u1}
                | {b.u0 for b in boxes} | {b.u1 for b in boxes})
    index = {x: i for i, x in enumerate(us)}

    # Assign each cell to the strips it covers. Every cell endpoint is a
    # compression point, so a cell covers a contiguous run of whole strips and
    # can never partially meet one; the lookups below fail loudly if that ever
    # stops being true.
    strips: List[List[Box]] = [[] for _ in range(len(us) - 1)]
    for bx in boxes:
        try:
            lo_i, hi_i = index[bx.u0], index[bx.u1]
        except KeyError:  # pragma: no cover - endpoints are compression points
            raise PartitionError(
                f"{what}: internal invariant broken -- cell {bx!r} has an "
                "endpoint that is not a compression point") from None
        for k in range(lo_i, hi_i):
            strips[k].append(bx)

    for i, strip in enumerate(strips):
        a, b = us[i], us[i + 1]
        if not strip:
            raise PartitionError(
                f"{what}: GAP -- no cell covers the strip u in [{a}, {b}]")

        ivs = sorted(((bx.v0, bx.v1) for bx in strip))
        cur = domain.v0
        for lo, hi in ivs:
            if lo < cur:
                raise PartitionError(
                    f"{what}: OVERLAP in strip u in [{a}, {b}] -- a cell starts "
                    f"at v={lo} while the run already reached v={cur}")
            if lo > cur:
                raise PartitionError(
                    f"{what}: GAP in strip u in [{a}, {b}] -- nothing covers "
                    f"v in [{cur}, {lo}]")
            cur = hi
        if cur != domain.v1:
            raise PartitionError(
                f"{what}: GAP in strip u in [{a}, {b}] -- cover stops at v={cur}, "
                f"domain ends at v={domain.v1}")


# ---------------------------------------------------------------- the ledger


@dataclass
class CellRecord:
    """One cell and everything the ledger must never lose about it."""

    cell: Cell
    disposition: str
    # ACCEPTED
    area: Optional[Interval] = None
    value_range: Optional[Interval] = None
    contribution: Optional[Interval] = None
    # REFINED
    children: Tuple[str, ...] = ()
    # REJECTED
    reject_kind: Optional[str] = None
    reason: Optional[str] = None
    boundary_area_bound: Optional[Fraction] = None
    residual: Optional[Interval] = None
    # PENDING
    pending_reason: Optional[str] = None
    # geometry, always recorded
    diameter_bound: Optional[Fraction] = None

    def is_leaf(self) -> bool:
        return self.disposition != REFINED


@dataclass(frozen=True)
class Total:
    """The result of :meth:`Ledger.total`.

    ``enclosure`` is an enclosure of the integral over the **ACCOUNTED** part
    of the domain -- and over the region itself only when ``covers_region`` is
    ``True``. ``covers_region`` says whether the accounted part provably
    contains the whole region; ``certified`` says whether every contributing
    enclosure came from a certifying path. Read both. A ``Total`` with
    ``certified=False`` is a number, not a bound; a ``Total`` with
    ``covers_region=False`` is a number about a subset, whatever its field is
    called.

    THE FIELD NAME IS NOT THE GUARANTEE. Publishing ``total.enclosure`` under a
    ``certified`` label without first reading ``certified`` and
    ``covers_region`` is the one way this package's API can be turned into a
    false claim. :meth:`certified_enclosure` exists so that a consumer which
    intends to stamp the word "certified" on the number can ask for it and be
    refused instead of shipping a caveat as free text. Any lane that publishes
    a certified interval should call that, not this field.

    ``area_rejected_by_kind`` breaks the retained rejected area down by
    :class:`RejectKind`. RN5's phrase is "retaining boundary-area bounds", and
    a cell rejected ``OUTSIDE`` is proved DISJOINT from the region: it holds no
    boundary and contributes exactly zero. Summing it into one field named
    after the boundary overstates the unresolved boundary, conservatively but
    misleadingly. The breakdown lets a reader separate the two without walking
    the cell list.
    """

    enclosure: Interval
    certified: bool
    covers_region: bool
    caveats: Tuple[str, ...]
    area_accounted: Interval
    area_rejected_bound: Fraction
    integrand: str
    area_rejected_by_kind: Dict[str, Fraction] = dc_field(default_factory=dict)

    @property
    def area_unresolved_boundary_bound(self) -> Fraction:
        """The retained area of cells that really do straddle the boundary.

        This -- not ``area_rejected_bound`` -- is the number RN5's phrase
        "boundary-area bounds" names. ``OUTSIDE`` cells are excluded because
        they are proved disjoint from the region.
        """
        return sum(
            (v for k, v in self.area_rejected_by_kind.items()
             if k != RejectKind.OUTSIDE),
            Fraction(0),
        )

    def certified_enclosure(self) -> Interval:
        """The enclosure, or a refusal. Use this wherever the word is published.

        Raises :class:`UncertifiedTotalError` unless ``certified`` and
        ``covers_region`` are both ``True``. The failure modes of this package
        are then symmetric: a PENDING cell makes :meth:`Ledger.total` raise, and
        an unaccounted region or a non-certifying path makes this raise. Neither
        can be published by accident.
        """
        if not (self.certified and self.covers_region):
            raise UncertifiedTotalError(
                "this Total is not a certified enclosure of the region "
                f"integral: certified={self.certified}, "
                f"covers_region={self.covers_region}. Caveats: "
                + (" | ".join(self.caveats) or "(none recorded)")
                + ". Read Total.enclosure directly if a number about the "
                "ACCOUNTED part is what is wanted, and do not label it "
                "certified."
            )
        return self.enclosure

    def as_json(self) -> Dict[str, object]:
        return {
            "enclosure_lo": str(self.enclosure.lo),
            "enclosure_hi": str(self.enclosure.hi),
            "enclosure_width": str(self.enclosure.width()),
            "enclosure_is_of_the_region": self.covers_region,
            "certified": self.certified,
            "covers_region": self.covers_region,
            "caveats": list(self.caveats),
            "area_accounted_lo": str(self.area_accounted.lo),
            "area_accounted_hi": str(self.area_accounted.hi),
            "area_rejected_bound": str(self.area_rejected_bound),
            "area_rejected_by_kind": {
                k: str(v) for k, v in sorted(self.area_rejected_by_kind.items())
            },
            "area_unresolved_boundary_bound": str(
                self.area_unresolved_boundary_bound),
            "integrand": self.integrand,
        }


def _by_kind(rej) -> Dict[str, Fraction]:
    """Retained rejected area, per :class:`RejectKind`. Exact rationals."""
    out: Dict[str, Fraction] = {}
    for r in rej:
        out[r.reject_kind] = (out.get(r.reject_kind, Fraction(0))
                              + (r.boundary_area_bound or Fraction(0)))
    return out


DOES_NOT_ESTABLISH = (
    "This receipt closes nothing. D3-LEMMA-RN-UNIF Piece 1 is OPEN and Piece 2 "
    "is OPEN. OBL-H5-JETMOD, OBL-H5-ZBAND (hi side), OBL-H5-REMOTE-THRESHOLD "
    "and OBL-D1-PROMOTE (chart side) are unchanged. No cell of the program's "
    "actual cover is certified here; the integrand is a REFERENCE function, not "
    "kappa_far and not the corrected RN5 envelope. The remote budget is not "
    "reassembled. The 2D upper, 2D lower and 3D lifetime tracks are composed in "
    "no way. Original prize problems solved: 0."
)


class Ledger:
    """The accept / refine / reject / pending record for one cover run."""

    def __init__(self, region_name: str, domain: Box, coords: str,
                 integrand: str = "(unset)", certifying: bool = True,
                 note: str = "") -> None:
        self.region_name = region_name
        self.domain = domain
        self.coords = coords
        self.integrand = integrand
        #: False as soon as any NON-CERTIFYING path contributes. Never reset.
        self.certifying = bool(certifying)
        self.note = note
        self.records: Dict[str, CellRecord] = {}
        self.roots: List[str] = []
        self._order: List[str] = []

    # ------------------------------------------------------------- mutation

    def _require(self, cid: str) -> CellRecord:
        try:
            return self.records[cid]
        except KeyError:
            raise KeyError(f"no such cell in the ledger: {cid!r}") from None

    def add(self, cell: Cell, diameter_bound: Optional[Fraction] = None,
            pending_reason: str = "not yet visited") -> CellRecord:
        """Register a cell as PENDING. Every cell enters the ledger this way."""
        if cell.cid in self.records:
            raise ValueError(f"duplicate cell id {cell.cid!r}")
        rec = CellRecord(cell=cell, disposition=PENDING,
                         pending_reason=pending_reason,
                         diameter_bound=diameter_bound)
        self.records[cell.cid] = rec
        self._order.append(cell.cid)
        if cell.parent is None:
            self.roots.append(cell.cid)
        return rec

    def accept(self, cid: str, area: Interval, value_range: Interval,
               contribution: Interval) -> None:
        rec = self._require(cid)
        self._assert_open(rec)
        rec.disposition = ACCEPTED
        rec.area = area
        rec.value_range = value_range
        rec.contribution = contribution
        rec.pending_reason = None

    def refine(self, cid: str, children: Sequence[Cell],
               diameters: Optional[Sequence[Fraction]] = None) -> List[Cell]:
        """Mark ``cid`` REFINED and register its children as PENDING.

        The children must tile the parent exactly; that is checked here, not
        assumed, so a bad subdivision rule fails at the point of use with the
        offending coordinates rather than later as a mysterious cover-wide gap.
        """
        rec = self._require(cid)
        self._assert_open(rec)
        check_exact_partition(rec.cell.box, [c.box for c in children],
                              what=f"children of {cid}")
        for i, ch in enumerate(children):
            d = None if diameters is None else diameters[i]
            self.add(ch, diameter_bound=d, pending_reason=f"child of {cid}")
        rec.disposition = REFINED
        rec.children = tuple(c.cid for c in children)
        rec.pending_reason = None
        return list(children)

    def reject(self, cid: str, kind: str, reason: str,
               boundary_area_bound: Fraction,
               residual: Optional[Interval] = None) -> None:
        """Reject a cell, retaining the reason and its boundary-area bound.

        ``boundary_area_bound`` is a certified **rational upper bound** on the
        cell's geometric area -- the area this rejection removes from the
        accounted part of the cover. RN5's recipe says to retain it; the ledger
        makes it a required argument so it cannot be forgotten.

        THE NAME IS RIGHT FOR ``UNRESOLVED_BOUNDARY`` AND ``EXCLUDED`` AND IS
        LOOSE FOR ``OUTSIDE``. An ``OUTSIDE`` cell is proved disjoint from the
        region: it holds no boundary and contributes exactly zero. Its area is
        still retained here -- the recipe says retain every rejected cell --
        but summing it into a single field named after the boundary OVERSTATES
        the unresolved boundary. On the showcased bracket run the single figure
        is ``28.90625`` of which ``16.40625`` is ``OUTSIDE`` (60 cells) and only
        ``12.5`` is genuinely unresolved boundary (128 cells). The direction is
        conservative, so no bound is unsound -- but the number does not mean
        what its name says, and ``Total.area_rejected_by_kind`` plus
        ``Total.area_unresolved_boundary_bound`` exist so a reader does not have
        to walk the cell list to find that out.
        """
        rec = self._require(cid)
        self._assert_open(rec)
        if kind not in RejectKind.ALL:
            raise ValueError(f"unknown reject kind {kind!r}")
        if not reason:
            raise ValueError("a rejected cell must carry a reason")
        if not isinstance(boundary_area_bound, Fraction):
            raise TypeError("boundary_area_bound must be an exact Fraction")
        if boundary_area_bound < 0:
            raise ValueError("boundary_area_bound must be non-negative")
        rec.disposition = REJECTED
        rec.reject_kind = kind
        rec.reason = reason
        rec.boundary_area_bound = boundary_area_bound
        rec.residual = residual
        rec.pending_reason = None

    def leave_pending(self, cid: str, reason: str) -> None:
        """Leave a cell PENDING, with the reason. The honest outcome when the
        depth budget is exhausted before the tolerance is met."""
        rec = self._require(cid)
        self._assert_open(rec)
        rec.pending_reason = reason

    def mark_non_certifying(self, why: str) -> None:
        self.certifying = False
        self.note = (self.note + " | " if self.note else "") + \
            f"NON-CERTIFYING: {why}"

    @staticmethod
    def _assert_open(rec: CellRecord) -> None:
        if rec.disposition != PENDING:
            raise ValueError(
                f"cell {rec.cell.cid!r} is already {rec.disposition}; a "
                "disposition is written once")

    # ----------------------------------------------------------- inspection

    def cells(self) -> List[CellRecord]:
        return [self.records[c] for c in self._order]

    def leaves(self) -> List[CellRecord]:
        return [r for r in self.cells() if r.is_leaf()]

    def by_disposition(self, disposition: str) -> List[CellRecord]:
        return [r for r in self.cells() if r.disposition == disposition]

    def pending(self) -> List[CellRecord]:
        return self.by_disposition(PENDING)

    def rejected(self) -> List[CellRecord]:
        return self.by_disposition(REJECTED)

    def max_depth(self) -> int:
        return max((r.cell.depth for r in self.cells()), default=0)

    def _leaf_diameters(self) -> List[Fraction]:
        return [r.diameter_bound for r in self.leaves()
                if r.diameter_bound is not None]

    def max_cell_width(self) -> Optional[Fraction]:
        """Largest recorded geometric diameter bound over LEAF cells.

        READ THIS AS "THE COARSEST LEAF", NOT AS "THE ACHIEVED RESOLUTION", AND
        DO NOT EXPECT IT TO MOVE WITH THE TOLERANCE. Two reasons, both real and
        both observed in this package's own showcase runs:

        1. Adaptive refinement leaves flat parts of the domain coarse ON
           PURPOSE. A cell where the integrand barely varies meets the
           tolerance at depth 1 and is never split, so it stays the widest leaf
           however far the rest of the cover is refined.
        2. A region's ``diameter_bound`` may SATURATE. ``PolarRegion`` caps its
           bound at ``2*r_max``, which is correct and is the right cap for a
           full-turn ring; but under ``split='radius'`` theta is never
           subdivided, so every leaf is a full-turn ring and every leaf reports
           the cap. On ``rn5_annulus_polar(split='radius')`` this value is
           exactly ``10`` -- the annulus's outer diameter -- at every depth and
           at every tolerance, including a 16x tightening that cuts the total
           width by 12x.

        So a monotone comparison of this number across tolerances is not a
        refinement control; on the annulus in polar coordinates it is the
        constant ``10 >= 10 >= 10``. :meth:`min_cell_width` and
        :meth:`max_depth` are the fields that do move, and the receipt carries
        all three.
        """
        ds = self._leaf_diameters()
        return max(ds) if ds else None

    def min_cell_width(self) -> Optional[Fraction]:
        """Smallest recorded geometric diameter bound over LEAF cells.

        The finest leaf. Where :meth:`max_cell_width` can saturate at a
        region's diameter cap and sit still, this one moves with the
        refinement, so the pair brackets the resolution actually achieved
        instead of advertising one end of it.
        """
        ds = self._leaf_diameters()
        return min(ds) if ds else None

    # ------------------------------------------------------------- invariants

    def check_partition(self) -> None:
        """The leaves must tile the domain exactly. Raises on any failure."""
        check_exact_partition(self.domain, [r.cell.box for r in self.leaves()],
                              what=f"{self.region_name} leaf cover")

    def check_refinements(self) -> None:
        """Every REFINED cell's children must tile it exactly.

        An independent second reading of the same invariant: cover-wide tiling
        follows from this plus the roots tiling the domain, by induction.
        """
        for rec in self.by_disposition(REFINED):
            kids = [self.records[c].cell.box for c in rec.children]
            check_exact_partition(rec.cell.box, kids,
                                  what=f"children of {rec.cell.cid}")
        check_exact_partition(self.domain,
                              [self.records[c].cell.box for c in self.roots],
                              what=f"{self.region_name} roots")

    # ------------------------------------------------------------------ total

    def total(self) -> Total:
        """The certified total, or an exception.

        REFUSES, by raising, while any cell is PENDING or the partition
        invariant fails. There is no flag to bypass it: a partial cover
        reported as a total is the exact failure mode RN5 names.
        """
        pend = self.pending()
        if pend:
            names = ", ".join(r.cell.cid for r in pend[:8])
            more = "" if len(pend) <= 8 else f" (+{len(pend) - 8} more)"
            raise PendingCellsError(
                f"{self.region_name}: {len(pend)} cell(s) still PENDING "
                f"[{names}{more}] -- this cover is NOT a coverage certificate. "
                "RN5's recipe requires verifying that no cell remains pending "
                "before any total is reported."
            )
        self.check_partition()
        self.check_refinements()

        caveats: List[str] = []
        acc = self.by_disposition(ACCEPTED)
        rej = self.rejected()

        total = Interval.exact(Fraction(0))
        area = Interval.exact(Fraction(0))
        for r in acc:
            if r.contribution is None or r.area is None:  # pragma: no cover
                raise ValueError(f"accepted cell {r.cell.cid} has no contribution")
            total = total + r.contribution
            area = area + r.area

        rejected_area = Fraction(0)
        by_kind: Dict[str, Fraction] = {}
        covers = True
        for r in rej:
            bab = r.boundary_area_bound or Fraction(0)
            rejected_area += bab
            by_kind[r.reject_kind] = by_kind.get(r.reject_kind, Fraction(0)) + bab
            if r.reject_kind == RejectKind.OUTSIDE:
                continue
            if r.residual is None:
                covers = False
                caveats.append(
                    f"cell {r.cell.cid} rejected as {r.reject_kind} with no "
                    f"residual enclosure: area <= {r.boundary_area_bound} of the "
                    "region is UNACCOUNTED, so the enclosure below is not an "
                    "enclosure of the region integral")
            else:
                total = total + r.residual

        certified = bool(self.certifying) and covers
        if not self.certifying:
            caveats.append(
                "NON-CERTIFYING run: at least one contribution came from a "
                "float / high-precision path, which is not a certified bound")

        return Total(enclosure=total, certified=certified, covers_region=covers,
                     caveats=tuple(caveats), area_accounted=area,
                     area_rejected_bound=rejected_area,
                     integrand=self.integrand,
                     area_rejected_by_kind=by_kind)

    def provisional_leaf_counts(self) -> Dict[str, int]:
        """How many leaves :meth:`provisional_enclosure` sums, and how many not.

        ``omitted`` counts leaves whose ``contribution`` and ``residual`` are
        both ``None`` -- cells the driver never evaluated, which is EVERY cell
        dropped by the ``max_cells`` guard. Read it alongside the provisional
        number: an omitted count above zero means the number is a sum over a
        strict subset of the cover and is low by an unknown amount.
        """
        summed = omitted = 0
        for r in self.leaves():
            if r.contribution is not None or r.residual is not None:
                summed += 1
            else:
                omitted += 1
        return {"summed": summed, "omitted": omitted,
                "leaves": summed + omitted}

    def provisional_enclosure(self) -> Interval:
        """Sum over the leaves THAT HAVE A NUMBER. **NOT a total, NOT a bound.**

        WHAT IT ACTUALLY SUMS, precisely -- an earlier docstring said "sum over
        ALL leaves including PENDING ones", and that is not what the body does.
        A leaf contributes ``contribution`` if it has one, else ``residual`` if
        it has one, and **is silently skipped when it has neither**. A cell
        dropped by the driver's ``max_cells`` guard is left PENDING before the
        integrand is ever evaluated, so it has neither and is skipped. Every
        such cell contributes zero to this number.

        THE RETURNED INTERVAL IS THEREFORE NOT AN ENCLOSURE, NOT A BOUND, AND
        NOT EVEN A LOWER BOUND of anything. Measured on a real run
        (``rn5_annulus_polar(split='both')``, ``max_cells=40``): 16 leaves
        omitted, all 16 contributing nothing, and the number comes back around
        ``[3.0e-5, 5.0e-5]`` for an integral whose value is ``6.2518`` -- five
        orders of magnitude low. A caller watching it for "progress" would read
        that collapse toward zero as convergence.

        Call :meth:`provisional_leaf_counts` beside it. While ``omitted`` is
        non-zero the number is a sum over a strict subset of the cover and says
        nothing about the rest. It is never certified, never a coverage
        certificate, and is deliberately given a different name and a different
        return type from :meth:`total` so the two cannot be confused at a call
        site.
        """
        out = Interval.exact(Fraction(0))
        for r in self.leaves():
            part = r.contribution if r.contribution is not None else r.residual
            if part is not None:
                out = out + part
        return out

    # ---------------------------------------------------------------- receipt

    def receipt(self) -> Dict[str, object]:
        """The receipt. A record of a computation, not evidence."""
        counts = {d: len(self.by_disposition(d)) for d in DISPOSITIONS}
        rej = self.rejected()
        try:
            total = self.total()
            total_json: object = total.as_json()
            total_error = None
        except (PendingCellsError, PartitionError) as exc:
            total_json = None
            total_error = f"{type(exc).__name__}: {exc}"

        mcw = self.max_cell_width()
        mnw = self.min_cell_width()
        prov_counts = self.provisional_leaf_counts()
        return {
            "region": self.region_name,
            "coords": self.coords,
            "integrand": self.integrand,
            "certifying": self.certifying,
            "arithmetic": ("exact rationals + certified intervals "
                           "(research/interval)") if self.certifying
                          else "NON-CERTIFYING (float / high-precision path present)",
            "note": self.note,
            "domain": self.domain.as_json(),
            "cells_total": len(self.records),
            "cells_by_disposition": counts,
            "pending_count": counts[PENDING],
            "pending_must_be_zero_for_a_certified_total": True,
            "refine_depth": self.max_depth(),
            "max_cell_width": None if mcw is None else str(mcw),
            "max_cell_width_decimal": None if mcw is None else f"{float(mcw):.9g}",
            "min_cell_width": None if mnw is None else str(mnw),
            "min_cell_width_decimal": None if mnw is None else f"{float(mnw):.9g}",
            "cell_width_note": (
                "max_cell_width is the COARSEST leaf, not the achieved "
                "resolution: adaptive refinement leaves flat regions coarse on "
                "purpose, and a region's diameter_bound may saturate at its cap "
                "(PolarRegion caps at 2*r_max, so under split='radius' every "
                "full-turn ring reports the cap at every depth). It is not "
                "monotone in the tolerance. Read refine_depth and "
                "min_cell_width beside it."),
            "provisional_leaf_counts": prov_counts,
            "area_rejected_bound": str(
                sum((r.boundary_area_bound or Fraction(0)) for r in rej)),
            "area_rejected_by_kind": {
                k: str(v) for k, v in sorted(_by_kind(rej).items())},
            "area_rejected_bound_note": (
                "area_rejected_bound sums EVERY rejected cell's retained area, "
                "including OUTSIDE cells that are proved disjoint from the "
                "region and hold no boundary at all. RN5's phrase "
                "'boundary-area bounds' names only the UNRESOLVED_BOUNDARY "
                "part; see area_rejected_by_kind for the split."),
            "rejected_cells": [
                {
                    "cid": r.cell.cid,
                    "kind": r.reject_kind,
                    "reason": r.reason,
                    "boundary_area_bound": str(r.boundary_area_bound),
                    "residual_lo": None if r.residual is None else str(r.residual.lo),
                    "residual_hi": None if r.residual is None else str(r.residual.hi),
                    "box": r.cell.box.as_json(),
                }
                for r in rej
            ],
            "pending_cells": [
                {"cid": r.cell.cid, "reason": r.pending_reason,
                 "box": r.cell.box.as_json()}
                for r in self.pending()
            ],
            "total": total_json,
            "total_error": total_error,
            "does_not_establish": DOES_NOT_ESTABLISH,
        }

    def receipt_json(self, indent: int = 2) -> str:
        return json.dumps(self.receipt(), indent=indent, sort_keys=True)
