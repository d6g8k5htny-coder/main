"""Region-generic spatial cover ledger and adaptive driver (lane A5 tooling).

``docs/OPEN_PROBLEMS.md`` A5 records **Piece 2 of ``D3-LEMMA-RN-UNIF`` as OPEN,
with the annulus Riemann-sum driver UNWRITTEN**, and quotes RN5's next exact
action:

    build a complete non-overlapping spatial cover of ``0.1 <= |y| <= 5``,
    retaining boundary-area bounds and every rejected cell; sum
    ``area x corrected cell supremum``; verify no cell remains pending; then
    reassemble the remote budget. Treat the near-axis refinement cost
    explicitly -- the present ten boxes are **not** a coverage certificate.

This package is that driver, region-generic, with the accept / refine / reject
ledger as the first-class output. It ships with REFERENCE integrands that it
can certify end to end.

WHAT THIS PACKAGE DOES NOT ESTABLISH
------------------------------------
* It does **not** close Piece 2 of ``D3-LEMMA-RN-UNIF``. It does not close
  Piece 1. Both stay **OPEN** exactly as ``docs/OPEN_PROBLEMS.md`` records
  them, and the lane receipts still carry ``lemma_closed: false``.
* It discharges, reduces, promotes and reclassifies **nothing**.
  ``OBL-H5-JETMOD``, ``OBL-H5-ZBAND`` (hi side), ``OBL-H5-REMOTE-THRESHOLD``
  and ``OBL-D1-PROMOTE`` (chart side) stand exactly as recorded.
* It certifies **no cell of the program's actual cover**. The integrands here
  are REFERENCE functions chosen because they can be certified; none of them is
  the program's ``kappa_far``, and none of them is the corrected RN5 envelope.
* It does **not** reassemble the remote budget, and it composes the 2D upper,
  2D lower and 3D lifetime tracks in no way whatsoever.
* It solves no prize problem and bears on none.
* A run of this driver is a run. A green receipt is a record of a computation,
  not evidence for any mathematical claim.

Standard library only (``fractions``, ``dataclasses``, ``json``). Python 3.11.
Certified enclosures come from ``research/interval/``; every float path in this
package is labelled NON-CERTIFYING in the code and in the receipt.
"""
from .ledger import (
    ACCEPTED, DISPOSITIONS, PENDING, REFINED, REJECTED, Box, Cell, Ledger,
    PartitionError, PendingCellsError, RejectKind, Total,
    check_exact_partition,
)
from .regions import (
    INSIDE, OUTSIDE, STRADDLE, CartesianBracketRegion, PolarRegion,
    rn5_annulus_bracket, rn5_annulus_polar, t4_polar_cover,
)
from .driver import (
    DriverConfig, FloatProbeReference, Integrand, RadialGaussianReference,
    TiltedGaussianReference, radial_gaussian_closed_form, run,
)

__all__ = [
    # ledger
    "ACCEPTED", "REFINED", "REJECTED", "PENDING", "DISPOSITIONS",
    "Box", "Cell", "Ledger", "Total", "RejectKind",
    "PartitionError", "PendingCellsError", "check_exact_partition",
    # regions
    "INSIDE", "OUTSIDE", "STRADDLE",
    "PolarRegion", "CartesianBracketRegion",
    "rn5_annulus_polar", "rn5_annulus_bracket", "t4_polar_cover",
    # driver
    "Integrand", "DriverConfig", "run",
    "RadialGaussianReference", "TiltedGaussianReference",
    "FloatProbeReference", "radial_gaussian_closed_form",
]
