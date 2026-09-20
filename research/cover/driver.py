"""The adaptive cover driver, and the REFERENCE integrands it can certify.

RN5's recipe, as ``docs/OPEN_PROBLEMS.md`` A5 records it: build a complete
non-overlapping spatial cover, retain boundary-area bounds and every rejected
cell, sum ``area x corrected cell supremum``, verify no cell remains pending,
then reassemble the remote budget. This module does the first four. It does
**not** do the fifth, and it supplies none of the mathematics the fourth would
need to bear on the lemma; see ``research/cover/README.md``.

THE SUM, AND WHY IT IS AN ENCLOSURE RATHER THAN AN UPPER BOUND
---------------------------------------------------------------
For a cell ``C`` of area ``|C|`` and an integrand ``f``,

    integral over C of f  =  |C| * (mean of f over C),

and the mean lies between the infimum and the supremum of ``f`` on ``C``.
So if ``rng`` is any certified enclosure of the **range** of ``f`` over ``C``,

    integral over C of f  is in  |C| * rng                                 (*)

as an interval product. Summing (*) over a cover whose cells tile the region
with disjoint interiors gives a certified enclosure of the region integral --
two-sided, not merely an upper bound. The "certified cell supremum" RN5 asks
for is exactly ``rng.hi``; keeping the lower endpoint as well costs nothing and
makes the result falsifiable from both sides.

For a cell that only partly meets the region -- the ``STRADDLE`` case of a
Cartesian bracket -- the same identity with ``|S| in [0, |C|]`` gives

    integral over (C and region) of f  is in  Interval(0, |C|) * rng,

which is what the driver records as that rejected cell's ``residual``. That is
how a rejected boundary cell is *bounded* rather than ignored.

ACCEPT / REFINE / REJECT / PENDING
-----------------------------------
* ``OUTSIDE``   -> REJECTED immediately, with the exact-test reason and the
  cell's area as the retained boundary-area bound. Residual is exactly zero.
* ``STRADDLE``  -> REFINED while depth remains; at the depth limit REJECTED as
  ``UNRESOLVED_BOUNDARY`` carrying the residual above. Never silently dropped.
* ``INSIDE``    -> contribution ``|C| * rng``. ACCEPTED when its width is
  within the cell's share of the tolerance; otherwise REFINED; and when the
  depth or cell budget is exhausted first, left **PENDING**.
* PENDING is a real outcome and it is fatal to a total. ``Ledger.total()``
  raises while any cell is pending. The driver never lowers the tolerance,
  never accepts on "close enough", and never reports a partial cover as a
  total.

The per-cell tolerance is ``tol * |C| / |domain|``, so the accepted cells'
widths sum to at most ``tol``: the tolerance is a target on the **total**
width, distributed by area. Budgeting uses rational upper bounds only and
cannot affect certification -- it decides when to stop refining, not what the
bound is.

OPT-IN UPPER TARGET. ``DriverConfig(upper_budget=B)`` replaces the width rule
with a sufficient rule for nonnegative ranges: each contribution upper,
including a straddling boundary residual, must fit ``B`` times its exact
parameter-area share. ``tol`` is inactive in this mode. The driver preserves
the supplied range (including an honest zero lower endpoint), multiplies by
geometric area once and sums all leaf contributions without another rounding.
No pending cell is waived. Uniform shares need not converge even when the
actual integral is below B: a locally high integrand may permanently exceed
its allocated share. This policy proves no completeness of the search.

RECOVERABLE ENCLOSURE FAILURE. Only ``RecoverableEnclosureError`` requests
subdivision without supplying a range. Its reason is retained; an unresolved
resource limit leaves PENDING, including at a straddling boundary. Ordinary
input, source and programming errors propagate. Subdivision guarantees no
future enclosure success and a failed attempt contributes no fabricated zero.

NON-CERTIFYING PATHS. An integrand declares ``certifying``. A single
non-certifying integrand makes the whole run non-certifying: the ledger is
flagged at construction, the receipt says so in ``certifying`` and
``arithmetic``, and ``Total.certified`` is False. ``FloatProbeReference`` is
included precisely so that this labelling is exercised by a test. It is not a
bound and must never be used as one.

WHAT THIS MODULE DOES NOT ESTABLISH. The built-in integrands are REFERENCE
functions, chosen because they can be certified end to end. **None of them is
the program's ``kappa_far``**, none is the corrected RN5 envelope, and a run
over one of them certifies no cell of the program's actual cover. A caller may
supply a source-derived adapter; its field-law proof, retained hypotheses and
domain scope remain separate from the generic driver's bookkeeping. The
driver supplies no full RN cover or remote budget. Piece 1 and Piece 2 of
``D3-LEMMA-RN-UNIF`` are OPEN.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import List, Optional, Protocol, Tuple

from research.interval import Interval, exp, pi

from .ledger import Cell, Ledger, RejectKind
from .regions import INSIDE, OUTSIDE, STRADDLE

__all__ = [
    "Integrand", "DriverConfig", "RecoverableEnclosureError", "run",
    "RadialGaussianReference", "TiltedGaussianReference", "FloatProbeReference",
    "radial_gaussian_closed_form",
]


class RecoverableEnclosureError(Exception):
    """A valid cell could not be enclosed; subdivision may resolve it.

    Adapters raise this only for an explicitly recognized interval admission
    failure, never for malformed inputs, changed sources or programming errors.
    No range or contribution is implied by this exception. The driver records
    it and refines, or leaves the cell PENDING when resources are exhausted.
    """


class Integrand(Protocol):
    """What the driver needs from an integrand. Four members, no more.

    ``name``       short identifier, copied into the receipt.
    ``label``      one line saying what it is and what it is not.
    ``certifying`` True only if every path inside ``range_enclosure`` is exact
                   rational or certified-interval arithmetic. An ``mpmath`` or
                   ``numpy`` integrand sets this False and the whole run is
                   labelled NON-CERTIFYING.
    ``range_enclosure(region, box, prec)`` a certified enclosure of
                   ``{f(y) : y in cell}``. Containment is the contract;
                   tightness is best effort. May raise
                   :class:`RecoverableEnclosureError` for a recognized
                   enclosure admission failure that subdivision may resolve.
    """

    name: str
    label: str
    certifying: bool

    def range_enclosure(self, region, box, prec: int) -> Interval:  # pragma: no cover
        ...


@dataclass(frozen=True)
class DriverConfig:
    """Knobs. None of them can weaken a bound -- only how far refinement goes.

    ``tol``       target width of the TOTAL enclosure, distributed over cells
                  by area. Exact ``Fraction``.
    ``max_depth`` subdivision depth limit. Hitting it leaves cells PENDING.
    ``prec``      precision hint passed to ``research/interval``; affects
                  tightness only, never containment.
    ``max_cells`` hard cap on cells visited. Hitting it leaves the remainder
                  PENDING, which is fatal to a total, by design.
    ``sig_bits``  outward rounding applied to stored enclosures to keep
                  endpoint denominators bounded. Widens, never narrows.
    ``upper_budget`` opt-in total upper target for nonnegative integrands.
                  None preserves the width/tol rule. Otherwise a cell's
                  contribution upper must fit its exact parameter-area share
                  of this budget. Boundary residuals consume the same budget.
                  This changes acceptance, never a supplied range or area.
    """

    tol: Fraction = Fraction(1, 100)
    max_depth: int = 16
    prec: int = 48
    max_cells: int = 200_000
    sig_bits: int = 80
    upper_budget: Optional[Fraction] = None

    def __post_init__(self) -> None:
        if not isinstance(self.tol, Fraction):
            raise TypeError("tol must be an exact Fraction")
        if self.tol < 0:
            raise ValueError("tol must be non-negative")
        if self.max_depth < 0 or self.max_cells < 1:
            raise ValueError("max_depth >= 0 and max_cells >= 1 required")
        if self.upper_budget is not None:
            if not isinstance(self.upper_budget, Fraction):
                raise TypeError("upper_budget must be an exact Fraction or None")
            if self.upper_budget < 0:
                raise ValueError("upper_budget must be non-negative")


def run(region, integrand: Integrand,
        config: Optional[DriverConfig] = None) -> Ledger:
    """Cover ``region``, evaluating ``integrand``, and return the ledger.

    Returns the **ledger**, not a number. Getting a number out of it means
    calling :meth:`~research.cover.ledger.Ledger.total`, which refuses while
    any cell is PENDING or the partition invariant fails. That ordering is the
    whole point of the module.
    """
    cfg = config or DriverConfig()
    ledger = Ledger(region_name=region.name, domain=region.domain(),
                    coords=region.coords, integrand=integrand.name,
                    certifying=bool(integrand.certifying),
                    note=integrand.label)
    if not integrand.certifying:
        ledger.mark_non_certifying(
            f"integrand {integrand.name!r} declares certifying=False")
    if cfg.upper_budget is not None:
        ledger.acceptance_policy = {
            "criterion": "NONNEGATIVE_CONTRIBUTION_UPPER",
            "upper_budget": str(cfg.upper_budget),
            "allocation": "exact cell parameter area / domain parameter area",
            "tol_applies": False,
            "includes_boundary_residuals": True,
        }

    roots = region.roots()
    domain_area_upper = sum((region.area_rational_upper(b) for b in roots),
                            Fraction(0))
    if domain_area_upper <= 0:  # pragma: no cover - guarded by region invariants
        raise ValueError("region has non-positive area upper bound")

    stack: List[Cell] = []
    for i, b in enumerate(roots):
        c = Cell(cid=f"c{i}", box=b, depth=0, parent=None)
        ledger.add(c, diameter_bound=region.diameter_bound(b),
                   pending_reason="root cell, not yet visited")
        stack.append(c)

    visited = 0
    while stack:
        cell = stack.pop()
        if visited >= cfg.max_cells:
            ledger.leave_pending(
                cell.cid,
                f"cell budget exhausted (max_cells={cfg.max_cells}); this cover "
                "is incomplete and total() will refuse")
            continue
        visited += 1

        box = cell.box
        cls = region.classify(box)
        area_up = region.area_rational_upper(box)

        if cls == OUTSIDE:
            ledger.reject(
                cell.cid, RejectKind.OUTSIDE,
                "provably disjoint from the region by the exact rational "
                "squared-radius test; contributes nothing to the region integral",
                boundary_area_bound=area_up,
                residual=Interval.exact(Fraction(0)))
            continue

        try:
            rng = integrand.range_enclosure(region, box, cfg.prec)
        except RecoverableEnclosureError as exc:
            reason = str(exc) or type(exc).__name__
            ledger.records[cell.cid].enclosure_failure = reason
            if cell.depth < cfg.max_depth:
                _refine(ledger, region, cell, stack)
            else:
                ledger.leave_pending(
                    cell.cid, f"recoverable enclosure failure at max_depth="
                    f"{cfg.max_depth}: {reason}; no contribution established")
            continue
        rng = rng.round_out(cfg.sig_bits)
        upper_share = None
        if cfg.upper_budget is not None:
            if rng.lo < 0:
                raise ValueError("upper-budget mode requires a nonnegative range enclosure")
            # Parameter areas are exactly additive for every valid partition.
            # Geometric area upper bounds need not be additive, so using them
            # for these shares would not imply a bound on the total upper.
            upper_share = (cfg.upper_budget * box.param_area()
                           / ledger.domain.param_area())

        if cls == STRADDLE:
            if upper_share is not None:
                residual = (Interval(Fraction(0), area_up) * rng
                            ).round_out(cfg.sig_bits)
                if residual.hi <= upper_share:
                    ledger.reject(
                        cell.cid, RejectKind.UNRESOLVED_BOUNDARY,
                        "straddles the region boundary; its retained residual "
                        "fits the exact parameter-area share of the upper budget",
                        boundary_area_bound=area_up, residual=residual)
                elif cell.depth < cfg.max_depth:
                    _refine(ledger, region, cell, stack)
                else:
                    rec = ledger.records[cell.cid]
                    rec.value_range, rec.residual = rng, residual
                    ledger.leave_pending(
                        cell.cid, f"max_depth={cfg.max_depth} reached with "
                        f"boundary residual upper {residual.hi} above the cell "
                        f"upper budget {upper_share}; NOT accepted or dropped")
                continue
            if cell.depth < cfg.max_depth:
                _refine(ledger, region, cell, stack)
            else:
                residual = (Interval(Fraction(0), area_up) * rng
                            ).round_out(cfg.sig_bits)
                ledger.reject(
                    cell.cid, RejectKind.UNRESOLVED_BOUNDARY,
                    f"straddles the region boundary and was not resolved within "
                    f"max_depth={cfg.max_depth}; its area is retained as a "
                    f"boundary-area bound and its possible contribution is "
                    f"carried as a residual enclosure",
                    boundary_area_bound=area_up, residual=residual)
            continue

        if cls != INSIDE:  # pragma: no cover - regions return one of three
            raise ValueError(f"region returned unknown classification {cls!r}")

        area_iv = region.area(box, cfg.prec).round_out(cfg.sig_bits)
        contribution = (area_iv * rng).round_out(cfg.sig_bits)
        budget = (cfg.tol * area_up / domain_area_upper
                  if upper_share is None else upper_share)
        measured = contribution.width() if upper_share is None else contribution.hi

        if measured <= budget:
            ledger.accept(cell.cid, area=area_iv, value_range=rng,
                          contribution=contribution)
        elif cell.depth < cfg.max_depth:
            _refine(ledger, region, cell, stack,
                    provisional=(area_iv, rng, contribution))
        else:
            ledger.records[cell.cid].contribution = contribution
            ledger.records[cell.cid].area = area_iv
            ledger.records[cell.cid].value_range = rng
            if upper_share is None:
                ledger.leave_pending(
                    cell.cid,
                    f"max_depth={cfg.max_depth} reached with contribution width "
                    f"{float(contribution.width()):.6g} above the cell budget "
                    f"{float(budget):.6g}; NOT accepted, NOT silently dropped")
            else:
                ledger.leave_pending(
                    cell.cid, f"max_depth={cfg.max_depth} reached with contribution "
                    f"upper {contribution.hi} above the cell upper budget "
                    f"{budget}; NOT accepted, NOT silently dropped")

    if cfg.upper_budget is not None and not ledger.pending():
        # Independently check the complete sum, including boundary residuals,
        # instead of treating successful individual comparisons as a total.
        if ledger.total().enclosure.hi > cfg.upper_budget:
            raise ValueError("complete enclosure exceeds the requested upper budget")
    return ledger


def _refine(ledger: Ledger, region, cell: Cell, stack: List[Cell],
            provisional: Optional[Tuple[Interval, Interval, Interval]] = None
            ) -> None:
    """Subdivide, record the children, and push them onto the work stack."""
    kids = region.subdivide(cell.box)
    children = [Cell(cid=f"{cell.cid}.{k}", box=b, depth=cell.depth + 1,
                     parent=cell.cid)
                for k, b in enumerate(kids)]
    if provisional is not None:
        # Keep the parent's own numbers for diagnostics; they are never summed,
        # because total() sums leaves only and a REFINED cell is not a leaf.
        rec = ledger.records[cell.cid]
        rec.area, rec.value_range, rec.contribution = provisional
    ledger.refine(cell.cid, children,
                  diameters=[region.diameter_bound(b) for b in kids])
    stack.extend(children)


# ------------------------------------------------------- REFERENCE integrands


class RadialGaussianReference:
    """REFERENCE integrand ``f(y) = exp(-|y|^2 / 2)``.

    Chosen because every step of its enclosure is certified and because its
    integral over an annulus has a closed form
    (:func:`radial_gaussian_closed_form`), so the driver can be checked against
    an independent evaluation of the same quantity.

    The enclosure is exact-range: ``|y|^2`` ranges over ``[r0^2, r1^2]`` (a
    rational interval supplied by the region), ``t -> exp(-t/2)`` is decreasing,
    and ``research/interval.exp`` returns a certified enclosure of the image of
    an interval. No dependency widening enters.

    THIS IS NOT THE PROGRAM'S ``kappa_far``, not the corrected RN5 envelope,
    and not any quantity appearing in any claim in this repository. It is a
    reference function for exercising and testing the driver.
    """

    name = "REFERENCE:radial_gaussian"
    label = ("REFERENCE integrand exp(-|y|^2/2). Fully certified. NOT kappa_far, "
             "NOT the corrected RN5 envelope, NOT any quantity in any claim.")
    certifying = True

    def range_enclosure(self, region, box, prec: int) -> Interval:
        r2 = region.radius2_enclosure(box)
        return exp((-r2) * Fraction(1, 2), prec)


class TiltedGaussianReference:
    """REFERENCE integrand ``f(y) = exp(-|y|^2 / 2) * (2 + y1/10)``.

    Same certified machinery with an angular dependence, so that theta
    refinement has something to do and the certified trigonometry in
    ``PolarRegion.cartesian_enclosure`` is exercised. On ``|y| <= 5`` the second
    factor lies in ``[3/2, 5/2]``, so ``f`` is positive there; nothing in the
    driver needs that, since the enclosure is two-sided.

    No closed form is claimed for it. THIS IS NOT ``kappa_far`` either.
    """

    name = "REFERENCE:tilted_gaussian"
    label = ("REFERENCE integrand exp(-|y|^2/2)*(2 + y1/10). Fully certified. "
             "NOT kappa_far, NOT the corrected RN5 envelope.")
    certifying = True

    def range_enclosure(self, region, box, prec: int) -> Interval:
        r2 = region.radius2_enclosure(box)
        g = exp((-r2) * Fraction(1, 2), prec)
        y1, _y2 = region.cartesian_enclosure(box, prec)
        return g * (Interval.exact(Fraction(2)) + y1 * Fraction(1, 10))


class FloatProbeReference:
    """NON-CERTIFYING float probe. **Not a bound. Never use it as one.**

    It exists for exactly one reason: to prove by test that a non-certifying
    integrand propagates its label all the way to ``Total.certified`` and to
    the receipt, so that an ``mpmath`` integrand wired in later cannot be
    mistaken for a certified one. The pad below is arbitrary and certifies
    nothing; binary floating point has no containment contract.
    """

    name = "NON-CERTIFYING-PROBE:radial_gaussian_float"
    label = ("NON-CERTIFYING float probe. A high-precision or float evaluation "
             "is not a certified bound. Present only to exercise the label.")
    certifying = False

    def range_enclosure(self, region, box, prec: int) -> Interval:
        import math  # local: the only float path in this package
        r2 = region.radius2_enclosure(box)
        lo = math.exp(-float(r2.hi) / 2.0)
        hi = math.exp(-float(r2.lo) / 2.0)
        pad = Fraction(1, 1000)
        return Interval(Fraction(lo) * (1 - pad), Fraction(hi) * (1 + pad))


def radial_gaussian_closed_form(region, prec: int = 64) -> Interval:
    """Certified enclosure of ``integral of exp(-|y|^2/2)`` over a polar annulus.

    In polar coordinates the integral separates:

        int_0^{2pi} int_{r0}^{r1} exp(-r^2/2) r dr dtheta
            = 2*pi * [ -exp(-r^2/2) ]_{r0}^{r1}
            = 2*pi * ( exp(-r0^2/2) - exp(-r1^2/2) ).

    Both exponentials are evaluated at exact rational points and ``pi`` comes
    from ``research/interval``, so the result is a certified enclosure of the
    same number the driver encloses -- an independent check of the driver, not
    an input to it.

    Only valid for a full-turn annulus, which is what
    :func:`~research.cover.regions.rn5_annulus_polar` and
    :func:`~research.cover.regions.t4_polar_cover` produce.
    """
    a = exp(Interval.exact(-(region.r_lo ** 2) / 2), prec)
    b = exp(Interval.exact(-(region.r_hi ** 2) / 2), prec)
    # round_out widens outward, so containment is preserved; it only keeps the
    # endpoint denominators printable.
    return (2 * pi(prec) * (a - b)).round_out(2 * prec)
