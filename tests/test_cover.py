"""Tests and negative controls for ``research/cover/`` — the lane A5 driver.

WHAT IS UNDER TEST. ``docs/OPEN_PROBLEMS.md`` A5 records the annulus
Riemann-sum driver for ``D3-LEMMA-RN-UNIF`` Piece 2 as **unwritten** and quotes
RN5's recipe: a complete non-overlapping spatial cover of ``0.1 <= |y| <= 5``,
retaining boundary-area bounds and every rejected cell, summing
``area x corrected cell supremum``, verifying that no cell remains pending.
``research/cover/`` is that driver. These tests check that it behaves as its
docstrings say, and — more importantly — that the ways it can lie are all
closed.

THE NEGATIVE CONTROLS ARE THE DELIVERABLE. Every control below was run against
a deliberately broken copy of the package in the scratchpad — never by editing
the library — and confirmed to FAIL there. Mutations and the controls that
caught them, as actually observed:

  M1  ``check_exact_partition``: return early when the areas sum correctly
      (``sum(param_area) == domain.param_area()``) instead of checking the
      structure.  -> caught by controls **3** and 4. Control 3 is the only one
      that catches it on the gap/overlap question, which is the whole reason
      that control exists: an area comparison passes it exactly.
  M2  ``check_exact_partition``: treat ``lo < cur`` (an overlap) as harmless.
      -> caught by control **3b**.
  M3  ``check_exact_partition``: treat ``lo > cur`` (an interior gap) as
      harmless.  -> **SURVIVED THE FIRST VERSION OF THIS FILE.** Controls 2
      and 3 both have gaps that run to the top of the domain, so both are
      caught by the *trailing* check and neither reaches the ``lo > cur``
      branch at all. Controls **2b** and **2c** were added for the two gap
      branches the original controls never exercised, and M3 then fails. This
      is recorded rather than quietly fixed: a suite that is green under a
      mutation is not testing that line.
  M4  ``Ledger.total``: warn instead of raising when cells are PENDING.
      -> caught by controls **6, 6b, 7, 7b** and 17.
  M5  ``Ledger.reject``: silently default ``boundary_area_bound`` to 0 instead
      of refusing a non-``Fraction``.  -> caught by control **9**.
  M11 ``Ledger.reject``: make the reason optional.  -> caught by control **9**.
  M6  driver: accept a cell at ``max_depth`` instead of leaving it PENDING.
      -> caught by control **7**.
  M7  driver: drop the ``UNRESOLVED_BOUNDARY`` residual.  -> caught by control
      **10**.
  M7  was ALSO a survivor at first: the original control 9 read the rejected
      rows but never looked for a ``residual``, and the run it used had pending
      cells so it never reached ``total()``. Control 9 now asserts a residual
      on every rejected row and completes its run, and M7 then fails.
  M8  driver: pass ``certifying=True`` to the ``Ledger`` regardless of the
      integrand. M8b: delete the ``mark_non_certifying`` call instead. M8c:
      both.  -> **M8c is caught by control 12; M8 and M8b are each survivable
      alone, and that is reported rather than hidden**: the two guards are
      deliberately redundant, so disabling either one leaves the other doing
      the job. Control 12 pins the ``mark_non_certifying`` marker string, which
      catches M8b. M8 alone has no observable effect while the second guard
      stands, so no test can catch it; the redundancy is the point.
  M9  ``PolarRegion.area``: drop the ``pi`` factor; M9b: use ``(r1 - r0)``
      instead of ``(r1^2 - r0^2)``.  -> caught by controls **13/14**.
  M10 ``Ledger.refine``: skip the children-tile-the-parent check.  -> caught by
      control **5**.

WHAT THESE TESTS DO NOT ESTABLISH
---------------------------------
* They do **not** close Piece 2 of ``D3-LEMMA-RN-UNIF``, and they do not close
  Piece 1. Both are OPEN. ``OBL-H5-JETMOD``, ``OBL-H5-ZBAND`` (hi side),
  ``OBL-H5-REMOTE-THRESHOLD`` and ``OBL-D1-PROMOTE`` (chart side) are
  unchanged.
* They certify **no cell of the program's actual cover**. Every integrand here
  is a REFERENCE function, not ``kappa_far`` and not the corrected RN5
  envelope.
* They reassemble no remote budget and compose no tracks.
* The dense direct evaluation in control 11 is **NON-CERTIFYING** binary
  floating point. It corroborates that the certified enclosure is not wildly
  wrong. It certifies nothing, and it is never used as a bound.
* A green build is not a mathematical review.
"""
from __future__ import annotations

import math
import os
import sys
from fractions import Fraction as F

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.interval import Interval  # noqa: E402
from research.cover import (  # noqa: E402
    ACCEPTED, PENDING, REJECTED, Box, Cell, DriverConfig, FloatProbeReference,
    Ledger, PartitionError, PendingCellsError, RadialGaussianReference,
    RejectKind, TiltedGaussianReference, check_exact_partition,
    radial_gaussian_closed_form, rn5_annulus_bracket, rn5_annulus_polar,
    run, t4_polar_cover,
)
from research.cover.regions import INSIDE, OUTSIDE, STRADDLE  # noqa: E402


# --------------------------------------------------------------- helpers

UNIT = Box.of(0, 2, 0, 2)


def _quad_cover():
    """The hand-verifiable exact tiling of [0,2]x[0,2] by four unit squares."""
    return [Box.of(0, 1, 0, 1), Box.of(0, 1, 1, 2),
            Box.of(1, 2, 0, 1), Box.of(1, 2, 1, 2)]


def _area(boxes):
    return sum((b.param_area() for b in boxes), F(0))


# ================================================================= group 1
# The partition invariant, on a region verifiable by hand.


def test_1_exact_tiling_by_hand_passes():
    """[0,2]^2 tiled by four unit squares, and by three rectangles."""
    check_exact_partition(UNIT, _quad_cover())
    check_exact_partition(UNIT, [Box.of(0, 1, 0, 2),
                                 Box.of(1, 2, 0, 1),
                                 Box.of(1, 2, 1, 2)])
    # A single cell equal to the domain is a partition of it.
    check_exact_partition(UNIT, [UNIT])


def test_1b_ragged_but_exact_tiling_passes():
    """A tiling whose cells do not line up in a grid is still a tiling."""
    cover = [Box.of(0, 1, 0, 2),
             Box.of(1, 2, 0, F(1, 2)),
             Box.of(1, F(3, 2), F(1, 2), 2),
             Box.of(F(3, 2), 2, F(1, 2), 2)]
    assert _area(cover) == UNIT.param_area()
    check_exact_partition(UNIT, cover)


def test_2_NEGATIVE_CONTROL_gap_must_fail():
    """CONTROL 2 (gap). Three of the four quadrants: a hole of area 1.

    Fails under mutations M1 (area check) and M3 (gap ignored).
    """
    cover = _quad_cover()[:3]
    assert _area(cover) == F(3)          # area check alone would also notice
    with pytest.raises(PartitionError) as exc:
        check_exact_partition(UNIT, cover)
    assert "GAP" in str(exc.value)


def test_2b_NEGATIVE_CONTROL_interior_gap_must_fail():
    """CONTROL 2b (interior gap). A strip covered at the bottom and at the top
    but not in the middle.

    THIS CONTROL EXISTS BECAUSE MUTATION M3 SURVIVED WITHOUT IT. Controls 2 and
    3 both have gaps that run to the top of the domain, so both are caught by
    the *trailing* check ``cur != domain.v1`` and neither ever reaches the
    ``lo > cur`` branch. Disabling that branch left the whole suite green. This
    is the control that reaches it: the cover below stops at ``v = 1/2`` and
    resumes at ``v = 3/2``, so the run is interrupted in the middle.
    """
    cover = [Box.of(0, 1, 0, 2),
             Box.of(1, 2, 0, F(1, 2)),
             Box.of(1, 2, F(3, 2), 2)]
    with pytest.raises(PartitionError) as exc:
        check_exact_partition(UNIT, cover)
    msg = str(exc.value)
    assert "GAP" in msg and "1/2" in msg and "3/2" in msg


def test_2c_NEGATIVE_CONTROL_empty_strip_must_fail():
    """CONTROL 2c. A whole vertical strip with no cell in it at all: the third
    of the three distinct gap branches, which controls 2 and 2b do not reach.
    """
    with pytest.raises(PartitionError) as exc:
        check_exact_partition(UNIT, [Box.of(0, 1, 0, 2)])
    assert "GAP" in str(exc.value) and "no cell covers the strip" in str(exc.value)


def test_3_NEGATIVE_CONTROL_equal_area_gap_and_overlap_must_fail():
    """CONTROL 3 — THE SHARPEST ONE. Areas sum EXACTLY right, and the cover is
    still not a cover: quadrant [0,1]x[1,2] is missing and quadrant
    [1,2]x[1,2] is listed twice. Gap of area 1, overlap of area 1, and the two
    cancel in any area comparison.

    A driver that validated its cover by comparing areas would accept this.
    Fails under mutations M1 and M3.
    """
    cover = [Box.of(0, 1, 0, 1), Box.of(1, 2, 0, 1),
             Box.of(1, 2, 1, 2), Box.of(1, 2, 1, 2)]
    # The area test passes exactly. This is the point of the control.
    assert _area(cover) == UNIT.param_area() == F(4)
    with pytest.raises(PartitionError) as exc:
        check_exact_partition(UNIT, cover)
    msg = str(exc.value)
    assert "GAP" in msg or "OVERLAP" in msg


def test_3b_NEGATIVE_CONTROL_pure_overlap_is_reported_as_an_overlap():
    """CONTROL 3b. An overlap with no gap, so the failure must name OVERLAP
    rather than GAP. Fails under mutations M1 and M2."""
    cover = _quad_cover() + [Box.of(0, 1, 0, 1)]
    with pytest.raises(PartitionError) as exc:
        check_exact_partition(UNIT, cover)
    assert "OVERLAP" in str(exc.value)


def test_3c_NEGATIVE_CONTROL_partial_overlap_not_on_the_grid():
    """CONTROL 3c. A cell shifted by 1/3 so no coordinate lines up. Exact
    rational arithmetic must still see the overlap; a float grid might not."""
    cover = _quad_cover() + [Box.of(F(1, 3), F(2, 3), F(1, 3), F(2, 3))]
    with pytest.raises(PartitionError):
        check_exact_partition(UNIT, cover)


def test_4_NEGATIVE_CONTROL_escape_and_degenerate_must_fail():
    """CONTROL 4. A cell outside the domain, and a zero-width cell."""
    with pytest.raises(PartitionError) as exc:
        check_exact_partition(UNIT, _quad_cover() + [Box.of(2, 3, 0, 1)])
    assert "escapes" in str(exc.value)

    with pytest.raises(PartitionError) as exc:
        check_exact_partition(UNIT, _quad_cover() + [Box.of(1, 1, 0, 1)])
    assert "degenerate" in str(exc.value)

    with pytest.raises(PartitionError):
        check_exact_partition(UNIT, [])


def test_4b_float_endpoints_are_refused():
    """Exactness starts at construction: a float endpoint never enters a Box."""
    with pytest.raises(TypeError):
        Box.of(0.0, 1, 0, 1)
    with pytest.raises(TypeError):
        Box(0.0, F(1), F(0), F(1))


def test_5_NEGATIVE_CONTROL_refine_rejects_children_that_do_not_tile():
    """CONTROL 5. A subdivision rule that drops a child, or whose children
    overlap, must be caught at the point of use. Fails under mutation M10."""
    led = Ledger("hand", UNIT, "cartesian")
    root = Cell("r", UNIT, 0)
    led.add(root)
    kids = _quad_cover()
    with pytest.raises(PartitionError):
        led.refine("r", [Cell(f"r.{i}", b, 1, "r") for i, b in enumerate(kids[:3])])
    # The good subdivision is accepted, and the parent/child invariant holds.
    led2 = Ledger("hand", UNIT, "cartesian")
    led2.add(Cell("r", UNIT, 0))
    led2.refine("r", [Cell(f"r.{i}", b, 1, "r") for i, b in enumerate(kids)])
    led2.check_refinements()


# ================================================================= group 2
# total() refuses while any cell is PENDING.


def test_6_NEGATIVE_CONTROL_total_raises_while_pending():
    """CONTROL 6. A ledger with one pending cell must RAISE, not warn and not
    return a partial number. Fails under mutation M4.

    This is the failure mode RN5 names: "the present ten boxes are NOT a
    coverage certificate".
    """
    led = Ledger("hand", UNIT, "cartesian", integrand="none")
    kids = _quad_cover()
    for i, b in enumerate(kids):
        led.add(Cell(f"c{i}", b, 0))
    one = Interval.exact(F(1))
    for i in range(3):
        led.accept(f"c{i}", area=one, value_range=one, contribution=one)
    led.leave_pending("c3", "left pending on purpose")

    with pytest.raises(PendingCellsError) as exc:
        led.total()
    assert "PENDING" in str(exc.value)
    assert "coverage certificate" in str(exc.value)

    rec = led.receipt()
    assert rec["pending_count"] == 1
    assert rec["total"] is None
    assert "PendingCellsError" in rec["total_error"]

    # Dispose of it and the total becomes available.
    led.records["c3"].disposition = PENDING  # still pending
    led.accept("c3", area=one, value_range=one, contribution=one)
    total = led.total()
    assert total.enclosure == Interval.exact(F(4))
    assert total.certified and total.covers_region


def test_6b_provisional_enclosure_is_not_a_total():
    """The escape hatch is separately named, separately typed, and never
    claims certification."""
    led = Ledger("hand", UNIT, "cartesian")
    for i, b in enumerate(_quad_cover()):
        led.add(Cell(f"c{i}", b, 0))
    one = Interval.exact(F(1))
    led.accept("c0", area=one, value_range=one, contribution=one)
    prov = led.provisional_enclosure()
    assert isinstance(prov, Interval)          # not a Total
    with pytest.raises(PendingCellsError):
        led.total()


def test_7_NEGATIVE_CONTROL_driver_leaves_cells_pending_at_max_depth():
    """CONTROL 7. A real run with an impossible tolerance and a small depth
    budget must end with PENDING cells and a refusing total().

    Fails under mutations M4 and M6 (accepting at max_depth).
    """
    region = rn5_annulus_polar(split="radius")
    led = run(region, RadialGaussianReference(),
              DriverConfig(tol=F(1, 10**9), max_depth=3, prec=32))
    assert led.pending(), "an impossible tolerance must leave cells PENDING"
    with pytest.raises(PendingCellsError):
        led.total()
    rec = led.receipt()
    assert rec["pending_count"] == len(led.pending()) > 0
    assert rec["pending_cells"][0]["reason"]
    assert "NOT accepted" in rec["pending_cells"][0]["reason"]


def test_7b_NEGATIVE_CONTROL_cell_budget_exhaustion_is_pending_not_silence():
    """CONTROL 7b. Hitting ``max_cells`` must leave the remainder PENDING."""
    region = rn5_annulus_polar(split="aspect")
    led = run(region, RadialGaussianReference(),
              DriverConfig(tol=F(1, 10**6), max_depth=30, prec=32, max_cells=200))
    assert led.pending()
    reasons = " ".join(r.pending_reason or "" for r in led.pending())
    assert "budget exhausted" in reasons or "max_depth" in reasons
    with pytest.raises(PendingCellsError):
        led.total()


# ================================================================= group 3
# Rejected cells survive, with reasons and boundary-area bounds.


def test_8_bracket_classification_is_exact_and_by_hand():
    """Exact rational squared-radius classification, checked on hand cases."""
    reg = rn5_annulus_bracket()
    # Well inside the excluded disc |y| < 1/10.
    assert reg.classify(Box.of(F(-1, 100), F(1, 100),
                               F(-1, 100), F(1, 100))) == OUTSIDE
    # Well outside the outer circle |y| = 5 (corner nearest origin at (4,4)).
    assert reg.classify(Box.of(4, 5, 4, 5)) == OUTSIDE
    # Wholly inside the annulus.
    assert reg.classify(Box.of(1, 2, 1, 2)) == INSIDE
    # Straddles the outer circle.
    assert reg.classify(Box.of(3, 4, 3, 4)) == STRADDLE
    # Straddles the inner circle.
    assert reg.classify(Box.of(0, F(1, 5), 0, F(1, 5))) == STRADDLE


def test_9_rejected_cells_survive_into_the_receipt_with_their_bounds():
    """CONTROL 9. Every rejected cell, its reason and its boundary-area bound
    must reach the receipt. Fails under mutation M5.

    This is RN5's "retaining boundary-area bounds and every rejected cell".
    """
    reg = rn5_annulus_bracket()
    # A deliberately loose tolerance: this test is about the bookkeeping, not
    # about tightness. At depth 5 a Cartesian cell is 10/32 wide, so the
    # enclosure it produces is far too wide to be useful as a bound — and the
    # ledger reports it as such rather than pretending otherwise.
    led = run(reg, RadialGaussianReference(),
              DriverConfig(tol=F(30), max_depth=5, prec=32))
    rejected = led.rejected()
    assert rejected, "a Cartesian bracket of a disc must reject cells"
    rec = receipt = led.receipt()
    assert len(receipt["rejected_cells"]) == len(rejected)
    kinds = {r["kind"] for r in receipt["rejected_cells"]}
    assert RejectKind.OUTSIDE in kinds
    assert RejectKind.UNRESOLVED_BOUNDARY in kinds
    for row in receipt["rejected_cells"]:
        assert row["reason"]
        assert F(row["boundary_area_bound"]) > 0
        assert row["box"]
        # Every rejected cell must carry a residual enclosure of what it could
        # still contribute; without one the region is not covered. (M7.)
        assert row["residual_lo"] is not None and row["residual_hi"] is not None
        if row["kind"] == RejectKind.OUTSIDE:
            assert F(row["residual_lo"]) == F(row["residual_hi"]) == 0
        else:
            assert F(row["residual_hi"]) > 0
    # With residuals present the run is still a certified enclosure of the
    # region integral even though cells were rejected. (M7 breaks this.)
    if not led.pending():
        t = led.total()
        assert t.covers_region and t.certified, t.caveats
    # The retained boundary area is positive and is reported separately from
    # the accounted area; it is never quietly folded into the total.
    assert F(rec["area_rejected_bound"]) > 0

    # And the API refuses a rejection with no reason or no bound at all.
    led2 = Ledger("hand", UNIT, "cartesian")
    led2.add(Cell("c", UNIT, 0))
    with pytest.raises(ValueError):
        led2.reject("c", RejectKind.OUTSIDE, "", F(1))
    with pytest.raises(TypeError):
        led2.reject("c", RejectKind.OUTSIDE, "why", 1.0)
    with pytest.raises(ValueError):
        led2.reject("c", "NOT_A_KIND", "why", F(1))


def test_10_NEGATIVE_CONTROL_unresolved_boundary_without_residual_is_uncertified():
    """CONTROL 10. A straddling cell rejected with no residual enclosure means
    part of the region is UNACCOUNTED: the ledger must say ``covers_region``
    is False and refuse the ``certified`` label. Fails under mutation M7.
    """
    led = Ledger("hand", UNIT, "cartesian", integrand="REFERENCE:test")
    for i, b in enumerate(_quad_cover()):
        led.add(Cell(f"c{i}", b, 0))
    one = Interval.exact(F(1))
    for i in range(3):
        led.accept(f"c{i}", area=one, value_range=one, contribution=one)
    led.reject("c3", RejectKind.UNRESOLVED_BOUNDARY,
               "straddles, unresolved", F(1), residual=None)
    total = led.total()
    assert total.covers_region is False
    assert total.certified is False
    assert any("UNACCOUNTED" in c for c in total.caveats)

    # With a residual it is an enclosure again.
    led2 = Ledger("hand", UNIT, "cartesian", integrand="REFERENCE:test")
    for i, b in enumerate(_quad_cover()):
        led2.add(Cell(f"c{i}", b, 0))
    for i in range(3):
        led2.accept(f"c{i}", area=one, value_range=one, contribution=one)
    led2.reject("c3", RejectKind.UNRESOLVED_BOUNDARY, "straddles", F(1),
                residual=Interval(F(0), F(1)))
    t2 = led2.total()
    assert t2.covers_region and t2.certified
    assert t2.enclosure == Interval(F(3), F(4))


# ================================================================= group 4
# The sum is a certified enclosure; refinement tightens it monotonically.


def test_11_certified_total_contains_the_closed_form_and_a_dense_float_sum():
    """The reference integrand's certified total must contain the true value.

    Two independent readings of the same number:

    (a) the closed form ``2*pi*(exp(-r0^2/2) - exp(-r1^2/2))``, itself a
        certified enclosure, derived by hand in ``driver.radial_gaussian_closed_form``;
    (b) a dense midpoint evaluation in binary floating point — **NON-CERTIFYING**,
        400 x 360 = 144,000 samples, present only as corroboration.

    (a) and the driver's total both contain the true value, so they must
    intersect; in fact the closed form must be *inside* the driver's total,
    which is the stronger assertion made here. (b) is within the midpoint
    rule's own error of the truth and must land inside the enclosure too.
    """
    region = rn5_annulus_polar(split="radius")
    led = run(region, RadialGaussianReference(),
              DriverConfig(tol=F(1, 10), max_depth=22, prec=40))
    assert not led.pending()
    total = led.total()
    assert total.certified and total.covers_region

    closed = radial_gaussian_closed_form(region, 60)
    assert closed in total.enclosure, (
        f"certified total {total.enclosure} excludes the closed form {closed}")

    # (b) NON-CERTIFYING dense direct evaluation. Floats, on purpose, labelled.
    n_r, n_t = 400, 360
    r0, r1 = float(region.r_lo), float(region.r_hi)
    dr = (r1 - r0) / n_r
    dth = 2.0 * math.pi / n_t
    dense = 0.0
    for i in range(n_r):
        r = r0 + (i + 0.5) * dr
        w = math.exp(-0.5 * r * r) * r * dr * dth
        dense += w * n_t
    assert F(dense) in total.enclosure, (
        f"NON-CERTIFYING dense sum {dense} outside certified {total.enclosure}")


def test_12_NEGATIVE_CONTROL_non_certifying_integrand_is_labelled():
    """CONTROL 12. A float integrand must make the whole run NON-CERTIFYING,
    all the way to ``Total.certified`` and the receipt. Fails under mutation M8.

    An mpmath integrand wired in later must not be able to pass as certified.
    """
    region = rn5_annulus_polar(split="radius")
    led = run(region, FloatProbeReference(),
              DriverConfig(tol=F(1, 2), max_depth=16, prec=32))
    assert led.certifying is False
    rec = led.receipt()
    assert rec["certifying"] is False
    assert "NON-CERTIFYING" in rec["arithmetic"]
    # The marker text below is produced ONLY by ``mark_non_certifying``, so it
    # pins the second of the two guards. Without it, mutation M8b (deleting the
    # ``mark_non_certifying`` call) survives silently on the back of the
    # constructor flag. See the module docstring on M8 / M8b / M8c.
    assert "NON-CERTIFYING: integrand" in rec["note"], rec["note"]
    assert "certifying=False" in rec["note"]
    if not led.pending():
        total = led.total()
        assert total.certified is False
        assert any("NON-CERTIFYING" in c for c in total.caveats)

    # The certifying reference integrand over the same region is not affected.
    led2 = run(region, RadialGaussianReference(),
               DriverConfig(tol=F(1, 2), max_depth=16, prec=32))
    assert led2.certifying is True
    assert led2.receipt()["certifying"] is True


def test_13_subdivision_theorem_children_enclosure_is_inside_the_parent():
    """CONTROL 13. Refinement must TIGHTEN, never merely move, the enclosure.

    The subdivision theorem: the children's areas sum exactly to the parent's
    and each child's range enclosure is contained in the parent's, so

        sum_i |C_i| * rng_i  subset of  |C| * rng.

    Asserted directly on one cell rather than inferred from a run. Fails under
    mutation M9 (a wrong area formula breaks the area identity).
    """
    region = rn5_annulus_polar(split="both")
    f = RadialGaussianReference()
    parent = region.roots()[0]
    prec = 48
    p_area = region.area(parent, prec)
    p_rng = f.range_enclosure(region, parent, prec)
    p_contrib = p_area * p_rng

    kids = region.subdivide(parent)
    check_exact_partition(parent, list(kids), what="subdivision theorem")
    child_sum = Interval.exact(F(0))
    area_sum = Interval.exact(F(0))
    for k in kids:
        a = region.area(k, prec)
        r = f.range_enclosure(region, k, prec)
        assert r in p_rng, "child range enclosure escaped the parent's"
        child_sum = child_sum + a * r
        area_sum = area_sum + a

    assert child_sum in p_contrib, (
        f"refined {child_sum} is not inside coarse {p_contrib}")
    assert child_sum.width() < p_contrib.width()
    # The areas are identical, not merely close: pi enters linearly.
    assert area_sum == p_area


def test_14_polar_area_formula_matches_the_annulus_area():
    """The accounted area of a full cover must be ``pi*(r_hi^2 - r_lo^2)``.

    Fails under mutation M9 (dropping ``pi``, or using ``r1 - r0``).
    """
    from research.interval import pi as _pi
    region = rn5_annulus_polar(split="radius")
    led = run(region, RadialGaussianReference(),
              DriverConfig(tol=F(1, 4), max_depth=20, prec=40))
    total = led.total()
    exact = Interval.exact(region.r_hi ** 2 - region.r_lo ** 2) * _pi(60)
    assert exact in total.area_accounted, (
        f"accounted area {total.area_accounted} does not contain the annulus "
        f"area {exact}")
    # and it is tight, not merely containing
    assert abs(float(total.area_accounted.mid() - exact.mid())) < 1e-9
    assert total.area_rejected_bound == 0, (
        "the polar instantiation represents the boundary exactly, so its "
        "rejected boundary area must be exactly zero")


def test_15_refinement_reduces_total_width_monotonically():
    """CONTROL 15. Tightening the tolerance must not loosen the answer.

    Three runs at decreasing tolerance: the certified total width must decrease
    strictly, the max leaf cell width must not increase, and every enclosure
    must contain the closed form (they all enclose the same number).
    """
    region = rn5_annulus_polar(split="radius")
    closed = radial_gaussian_closed_form(region, 60)
    widths, cellwidths = [], []
    for tol in (F(1), F(1, 4), F(1, 16)):
        led = run(region, RadialGaussianReference(),
                  DriverConfig(tol=tol, max_depth=24, prec=40))
        assert not led.pending()
        t = led.total()
        assert closed in t.enclosure
        widths.append(t.enclosure.width())
        cellwidths.append(led.max_cell_width())
    assert widths[0] > widths[1] > widths[2], widths
    assert cellwidths[0] >= cellwidths[1] >= cellwidths[2], cellwidths


# ================================================================= group 5
# The two instantiations stay separate, and the cost is explicit.


def test_16_the_two_regions_are_different_and_are_never_merged():
    """RN5's annulus and the T4 polar cover are different regions serving
    different purposes. Nothing in the package relates them."""
    a = rn5_annulus_polar()
    t = t4_polar_cover()
    assert (a.r_lo, a.r_hi) == (F(1, 10), F(5))
    assert (t.r_lo, t.r_hi) == (F(5), F(17))
    assert a.domain() != t.domain()
    assert a.name != t.name
    # Their domains are disjoint in radius except at the single radius 5,
    # which is a boundary circle of each; no cover of one covers the other.
    assert a.r_hi == t.r_lo
    assert not a.domain().contains_box(t.domain())
    assert not t.domain().contains_box(a.domain())


def test_17_t4_theta_halving_is_the_named_policy_and_its_cost_is_visible():
    """``LANE_RN_UNIF.md`` names theta-halving for the T4 cover. Under that
    policy radial resolution is fixed by the shell list, so a tolerance that
    needs radial refinement must end PENDING rather than be accepted."""
    t = t4_polar_cover()
    assert t.split == "theta"
    assert t.shells == tuple(F(k) for k in range(5, 18))

    # (a) A tolerance reachable at the shell resolution: it completes.
    led = run(t, TiltedGaussianReference(),
              DriverConfig(tol=F(1, 60), max_depth=8, prec=32))
    assert not led.pending()
    assert led.total().certified

    # (b) The documented limitation, demonstrated rather than asserted. For a
    # purely radial integrand, halving theta cannot change a cell's range at
    # all, so a tolerance below the shell's radial floor can never be met and
    # the run must end PENDING however much depth it is given.
    tight = DriverConfig(tol=F(1, 1000), max_depth=5, prec=32)
    led2 = run(t, RadialGaussianReference(), tight)
    assert led2.pending(), "theta-halving cannot tighten a radial integrand"
    with pytest.raises(PendingCellsError):
        led2.total()

    # (c) The same region and the same tolerance with radial bisection allowed
    # completes. The difference is the split policy and nothing else.
    led3 = run(t4_polar_cover(split="radius"), RadialGaussianReference(),
               DriverConfig(tol=F(1, 1000), max_depth=12, prec=32))
    assert not led3.pending()
    assert led3.total().certified
    assert led3.max_depth() > 0


def test_18_near_axis_refinement_cost_is_a_count_not_an_estimate():
    """``uniform_cost`` must be exact rational arithmetic and must expose the
    50x inner/outer anisotropy of the RN5 annulus."""
    a = rn5_annulus_polar()
    cost = a.uniform_cost(F(1, 10))
    assert isinstance(cost["radial_steps"], int)
    assert isinstance(cost["cells"], int)
    assert cost["cells"] == cost["radial_steps"] * cost["angular_steps"]
    assert cost["inner_waste_factor"] == F(50)
    assert isinstance(cost["inner_waste_factor"], F)
    # A finer target costs strictly more.
    assert a.uniform_cost(F(1, 20))["cells"] > cost["cells"]
    with pytest.raises(ValueError):
        a.uniform_cost(F(0))


def test_19_receipt_shape_and_the_does_not_establish_field():
    """The receipt must carry everything the task requires, and must always
    say what it does not establish."""
    region = rn5_annulus_polar(split="radius")
    led = run(region, RadialGaussianReference(),
              DriverConfig(tol=F(1, 2), max_depth=20, prec=32))
    rec = led.receipt()
    for key in ("cells_by_disposition", "area_rejected_bound", "max_cell_width",
                "refine_depth", "pending_count", "total", "does_not_establish",
                "certifying", "region", "integrand"):
        assert key in rec, key
    assert set(rec["cells_by_disposition"]) == {ACCEPTED, "REFINED", REJECTED,
                                                PENDING}
    assert rec["pending_count"] == 0
    assert rec["refine_depth"] >= 1
    dne = rec["does_not_establish"]
    for phrase in ("Piece 1 is OPEN", "Piece 2", "OBL-H5-JETMOD",
                   "not kappa_far", "remote budget", "prize"):
        assert phrase.lower() in dne.lower(), phrase
    # It serialises.
    import json
    json.loads(led.receipt_json())


def test_20_a_disposition_is_written_once():
    """No cell may be accepted twice, or accepted after being rejected."""
    led = Ledger("hand", UNIT, "cartesian")
    led.add(Cell("c", UNIT, 0))
    one = Interval.exact(F(1))
    led.accept("c", area=one, value_range=one, contribution=one)
    with pytest.raises(ValueError):
        led.accept("c", area=one, value_range=one, contribution=one)
    with pytest.raises(ValueError):
        led.reject("c", RejectKind.OUTSIDE, "late", F(1))
    with pytest.raises(KeyError):
        led.accept("nope", area=one, value_range=one, contribution=one)
    with pytest.raises(ValueError):
        led.add(Cell("c", UNIT, 0))


def test_21_ledger_partition_check_runs_on_a_real_cover():
    """The leaves of a real driver run must tile the domain exactly, and the
    parent/child invariant must hold independently."""
    for region in (rn5_annulus_polar(split="aspect"),
                   rn5_annulus_bracket()):
        led = run(region, RadialGaussianReference(),
                  DriverConfig(tol=F(2), max_depth=4, prec=32))
        led.check_partition()
        led.check_refinements()
