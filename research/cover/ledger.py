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
from dataclasses import dataclass
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
    reports ``covers_region=False`` and refuses the ``certified`` label.

    ``EXCLUDED`` -- excluded by a stated predicate other than the region test
    (available for callers; nothing in this package uses it).
    """

    OUTSIDE = "OUTSIDE"
    UNRESOLVED_BOUNDARY = "UNRESOLVED_BOUNDARY"
    EXCLUDED = "EXCLUDED"
    ALL = (OUTSIDE, UNRESOLVED_BOUNDARY, EXCLUDED)


class PartitionError(Exception):
    """The cells do not exactly tile the domain: a gap, an overlap or an escape."""


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

    ``enclosure`` is a certified enclosure of the integral over the
    **accounted** part of the domain. ``covers_region`` says whether the
    accounted part provably contains the whole region; ``certified`` says
    whether every contributing enclosure came from a certifying path. Read
    both. A ``Total`` with ``certified=False`` is a number, not a bound.
    """

    enclosure: Interval
    certified: bool
    covers_region: bool
    caveats: Tuple[str, ...]
    area_accounted: Interval
    area_rejected_bound: Fraction
    integrand: str

    def as_json(self) -> Dict[str, object]:
        return {
            "enclosure_lo": str(self.enclosure.lo),
            "enclosure_hi": str(self.enclosure.hi),
            "enclosure_width": str(self.enclosure.width()),
            "certified": self.certified,
            "covers_region": self.covers_region,
            "caveats": list(self.caveats),
            "area_accounted_lo": str(self.area_accounted.lo),
            "area_accounted_hi": str(self.area_accounted.hi),
            "area_rejected_bound": str(self.area_rejected_bound),
            "integrand": self.integrand,
        }


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

    def max_cell_width(self) -> Optional[Fraction]:
        """Largest recorded geometric diameter bound over LEAF cells."""
        ds = [r.diameter_bound for r in self.leaves() if r.diameter_bound is not None]
        return max(ds) if ds else None

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
        covers = True
        for r in rej:
            rejected_area += r.boundary_area_bound or Fraction(0)
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
                     integrand=self.integrand)

    def provisional_enclosure(self) -> Interval:
        """Sum over ALL leaves including PENDING ones. **NOT a total.**

        Never certified, never a coverage certificate, and deliberately given a
        different name and a different return type from :meth:`total` so the
        two can never be confused at a call site. It exists so that refinement
        progress can be measured while cells are still pending.
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
            "area_rejected_bound": str(
                sum((r.boundary_area_bound or Fraction(0)) for r in rej)),
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
