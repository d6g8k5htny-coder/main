"""Check mutable cover records at the boundary where a total is consumed.

The historical ledger is source-pinned by existing RN candidates. This additive
adapter does not change those bytes or rewrite their certificates. Call
``checked_total(ledger).certified_enclosure()`` instead of trusting a previously
validated, but subsequently mutable, record collection.

Scope: record/accounting consistency and the ledger's existing exact partition
checks. Supplied areas, integrand bounds and OUTSIDE geometry remain premises;
this function does not prove them, authenticate a caller or provide a lock.
A nonzero-width residual containing zero can be a valid conservative enclosure.
For OUTSIDE we deliberately require the canonical None/[0,0] representation,
so a stored number is never silently discarded by the publication path.
"""
from __future__ import annotations

from fractions import Fraction
from typing import List

from research.interval import Interval
from research.cover.ledger import (
    ACCEPTED, DISPOSITIONS, REJECTED, CellRecord, Ledger, RejectKind, Total,
)


def _interval(value: object, description: str) -> Interval:
    if (type(value) is not Interval or type(value.lo) is not Fraction
            or type(value.hi) is not Fraction or value.lo > value.hi):
        raise ValueError(f"{description}: exact Interval required")
    return value


def _sum_intervals(values: List[Interval]) -> Interval:
    # A second accumulation path: sum the exact endpoints, not Interval.__add__.
    return Interval(sum((value.lo for value in values), Fraction(0)),
                    sum((value.hi for value in values), Fraction(0)))


def checked_total(ledger: Ledger) -> Total:
    """Return the legacy Total only after independently checking its accounting.

    Bad records or inconsistent totals raise ValueError. Existing pending-cell
    and partition exceptions propagate unchanged. Well-formed noncertifying or
    incomplete totals remain noncertifying/incomplete, rather than being promoted.
    The ledger and all supplied intervals are left unchanged.
    """
    if not isinstance(ledger, Ledger):
        raise TypeError("checked_total requires a Ledger")
    if type(ledger.certifying) is not bool:
        raise ValueError("ledger.certifying must be a boolean")
    order = ledger._order
    if (type(order) is not list or any(type(cid) is not str for cid in order)
            or len(set(order)) != len(order) or set(order) != set(ledger.records)):
        raise ValueError("ledger record order must enumerate every unique record")

    contributions: List[Interval] = []
    areas: List[Interval] = []
    rejected_area = Fraction(0)
    rejected_by_kind = {}
    covers = True
    for cid in order:
        record = ledger.records[cid]
        if not isinstance(record, CellRecord) or record.cell.cid != cid:
            raise ValueError(f"cell {cid!r}: record identity mismatch")
        if record.disposition not in DISPOSITIONS:
            raise ValueError(f"cell {cid!r}: unknown disposition {record.disposition!r}")
        if record.disposition == ACCEPTED:
            area = _interval(record.area, f"cell {cid!r} area")
            if area.lo < 0:
                raise ValueError(f"cell {cid!r}: area must be nonnegative")
            _interval(record.value_range, f"cell {cid!r} value range")
            contributions.append(_interval(record.contribution, f"cell {cid!r} contribution"))
            areas.append(area)
        elif record.disposition == REJECTED:
            kind = record.reject_kind
            if kind not in RejectKind.ALL:
                raise ValueError(f"cell {cid!r}: unknown rejection kind {kind!r}")
            if type(record.reason) is not str or not record.reason.strip():
                raise ValueError(f"cell {cid!r}: rejection reason required")
            bound = record.boundary_area_bound
            if type(bound) is not Fraction or bound < 0:
                raise ValueError(f"cell {cid!r}: nonnegative exact boundary area required")
            residual = record.residual
            if residual is not None:
                residual = _interval(residual, f"cell {cid!r} residual")
            rejected_area += bound
            rejected_by_kind[kind] = rejected_by_kind.get(kind, Fraction(0)) + bound
            if kind == RejectKind.OUTSIDE:
                if residual is not None and (residual.lo != 0 or residual.hi != 0):
                    raise ValueError(f"cell {cid!r}: OUTSIDE residual must be None or [0,0]; "
                                     "refusing to discard a stored interval")
            elif residual is None:
                covers = False
            else:
                contributions.append(residual)

    # Retain the original exact tiling/refinement checks and pending refusals.
    result = ledger.total()
    if type(result) is not Total:
        raise ValueError("ledger returned an unexpected total type")
    if type(result.certified) is not bool or type(result.covers_region) is not bool:
        raise ValueError("total coverage/certification flags must be booleans")
    if result.covers_region != covers or result.certified != (ledger.certifying and covers):
        raise ValueError("total coverage/certification differs from checked records")
    _interval(result.enclosure, "total enclosure")
    _interval(result.area_accounted, "total accounted area")
    if result.enclosure != _sum_intervals(contributions):
        raise ValueError("total enclosure differs from independently summed records")
    if result.area_accounted != _sum_intervals(areas):
        raise ValueError("total accounted area differs from checked records")
    if (type(result.area_rejected_bound) is not Fraction
            or type(result.area_rejected_by_kind) is not dict
            or any(type(value) is not Fraction for value in result.area_rejected_by_kind.values())
            or result.area_rejected_bound != rejected_area
            or result.area_rejected_by_kind != rejected_by_kind):
        raise ValueError("total rejected area differs from checked records")
    return result
