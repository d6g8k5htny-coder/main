"""Regions to cover, with exact rational boundaries, and two instantiations.

A region supplies six things to the driver, and nothing else:

===========================  =================================================
``roots()``                  the root boxes, which tile ``domain()`` exactly
``classify(box)``            ``INSIDE`` / ``OUTSIDE`` / ``STRADDLE``, decided
                             by an **exact rational** test -- never by a float
                             comparison and never by sampling
``area(box, prec)``          certified enclosure of the cell's geometric area
``area_rational_upper(box)`` a rational upper bound on the same, for budgeting
``diameter_bound(box)``      certified rational upper bound on the Euclidean
                             diameter of the cell's image in the plane
``subdivide(box)``           children that tile ``box`` exactly
===========================  =================================================

plus two enclosures an integrand may ask for: ``radius2_enclosure(box)``, the
range of ``|y|^2`` over the cell, and ``cartesian_enclosure(box, prec)``, the
range of the two Cartesian coordinates.

THE TWO INSTANTIATIONS ARE DIFFERENT REGIONS SERVING DIFFERENT PURPOSES AND ARE
NEVER MERGED OR SUMMED TOGETHER:

* :func:`rn5_annulus_polar` -- ``0.1 <= |y| <= 5``, from
  ``docs/OPEN_PROBLEMS.md`` A5 quoting RN5. This is the Piece 2 cover region.
* :func:`t4_polar_cover` -- ``d in [5, 17]`` with theta-halving, from
  ``LANE_RN_UNIF.md``'s T4 push freeze list.

``GROUNDING_B.md`` states the separation outright: "These are different regions
serving different purposes. A cover engine should be region-generic and
instantiate both, never merge them." Nothing in this module adds one to the
other, and the ledgers they produce are separate objects.


WHY POLAR, FOR AN ANNULUS
-------------------------
The annulus boundary is ``|y| = 1/10`` and ``|y| = 5``. Those circles are not
rational-polygon-friendly: no finite union of rational axis-aligned boxes has
them as its boundary, so a Cartesian cover must either exclude a sliver or
include one, and either way the sliver has to be carried as a bounded error.

In polar parameters ``(radius, turn)`` with ``theta = 2*pi*turn`` the boundary
is exactly ``radius = 1/10`` and ``radius = 5``, both **rational**. The
difficulty does not disappear -- it moves, and it is honest to say where to:

* the cell **area** ``pi * (t1 - t0) * (r1^2 - r0^2)`` now carries ``pi``, so it
  is a certified *enclosure* rather than an exact rational. ``research/interval``
  supplies it, and the width is under the caller's ``prec`` control;
* a cell's **Cartesian extent** needs ``sin`` and ``cos`` of ``2*pi*t``, again
  certified enclosures from ``research/interval``, widened by the dependency
  problem (``r*cos`` is evaluated as an interval product);
* the parameter rectangle ``[r_in, r_out] x [0, 1]`` maps **injectively** onto
  the annulus except that ``turn = 0`` and ``turn = 1`` name the same ray.
  Cells sharing that seam share a boundary segment of area zero, exactly as two
  side-by-side cells anywhere else in the cover do. "Non-overlapping" means
  disjoint interiors throughout this package, which is what a Riemann sum needs.
  Because ``r_in > 0`` there is no origin degeneracy.

So: **the rejected boundary area of the polar instantiation is exactly zero**,
by construction, and that is reported rather than assumed.
:func:`rn5_annulus_bracket` is the *other* honest option applied to the *same*
region -- a rational Cartesian bracket whose straddling cells are REJECTED with
their areas retained as an explicit boundary-area bound. It exists so the
sliver accounting is exercised and tested, not so its number can be added to
anything.


THE NEAR-AXIS REFINEMENT COST, FACED EXPLICITLY
-----------------------------------------------
RN5 requires that the near-axis refinement cost be treated explicitly and says
the ten boxes are not a coverage certificate. Stating plainly what this module
can and cannot know: the sources quoted in ``GROUNDING_B.md`` and
``docs/OPEN_PROBLEMS.md`` name the cost without defining "near-axis" in terms
this repository can resolve, so **the reading used here is stated as a reading,
not as a quotation**: the expensive part of ``0.1 <= |y| <= 5`` is the inner
edge, where the region approaches the excluded disc ``|y| < 0.1`` that
``reviews/records/REV-RN3-FARZONE-20260918.json`` separately observes is
unaccounted for in RN3 section 9's sum.

What this module does about it, concretely:

1. :meth:`PolarRegion.uniform_cost` computes, in exact rationals, how many
   cells a *uniform* cover needs to reach a target Cartesian cell diameter. It
   is not an estimate and not a fit; it is a count.
2. That count exposes the anisotropy: at a fixed angular step the arc extent of
   a cell is ``r * 2*pi*dt``, which is ``r_out / r_in = 50`` times larger at the
   outer edge of the RN5 annulus than at the inner edge. A uniform angular grid
   fine enough for the outer edge over-refines the inner edge by that factor,
   and one fine enough for the inner edge leaves the outer edge coarse.
3. The default split policy for the RN5 annulus is therefore ``"aspect"``:
   split whichever direction currently contributes more to the cell's diameter
   bound. That spends refinement where the geometry needs it instead of
   uniformly. The T4 instantiation keeps ``"theta"`` because ``LANE_RN_UNIF.md``
   names theta-halving; the consequence -- that its radial resolution is then
   fixed by its shell list and cells can go PENDING on radial width alone -- is
   documented on that factory and reported in the receipt rather than hidden.
4. The receipt carries ``max_cell_width`` and ``refine_depth`` so the achieved
   resolution is always visible next to any total.

WHAT THIS MODULE DOES NOT ESTABLISH. It defines regions. It closes nothing,
certifies no cell of the program's actual cover, and reassembles no budget.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from fractions import Fraction
from typing import List, Sequence, Tuple

from research.interval import Interval, cos, sin
from research.interval import pi as _pi_uncached

from .ledger import Box

__all__ = [
    "INSIDE", "OUTSIDE", "STRADDLE",
    "PolarRegion", "CartesianBracketRegion",
    "rn5_annulus_polar", "rn5_annulus_bracket", "t4_polar_cover",
    "two_pi_upper", "pi_upper",
]

INSIDE = "INSIDE"
OUTSIDE = "OUTSIDE"
STRADDLE = "STRADDLE"

#: Cached certified rational bounds on pi. ``pi(prec)`` is the certified source;
#: these are its endpoints, used where a *rational* bound is wanted (budgets,
#: diameter bounds). They are upper bounds on the true pi, never approximations.
_PI_BOUND_PREC = 64


@functools.lru_cache(maxsize=None)
def pi(prec: int) -> Interval:
    """``research.interval.pi``, memoised per precision.

    Memoising a pure function of one integer changes no value: the certified
    enclosure returned is the same object the library computes, and
    ``Interval`` is immutable. It exists because a cover run asks for ``pi`` a
    few times per cell and the series is not free.
    """
    return _pi_uncached(prec)


def pi_upper() -> Fraction:
    """A certified rational **upper** bound on ``pi``."""
    return pi(_PI_BOUND_PREC).hi


def two_pi_upper() -> Fraction:
    """A certified rational **upper** bound on ``2*pi``."""
    return 2 * pi(_PI_BOUND_PREC).hi


# ------------------------------------------------------------------ polar


@dataclass(frozen=True)
class PolarRegion:
    """An annulus ``r_lo <= |y| <= r_hi`` covered by polar cells.

    Parameter coordinates are ``(u, v) = (radius, turn)`` with
    ``theta = 2*pi*turn`` and ``turn`` ranging over ``[0, 1]``. Both radii are
    exact rationals, so the region boundary is represented exactly and the
    rejected boundary area is exactly zero.

    ``shells`` are the radial breakpoints of the root cells, in increasing
    order, beginning at ``r_lo`` and ending at ``r_hi``.

    ``split`` is the subdivision policy:

    ``"aspect"``
        split whichever direction contributes more to the diameter bound
        (radial extent ``r1 - r0`` against arc extent ``r1 * 2*pi * dt``);
        on a tie, split both. This is the answer to the near-axis refinement
        cost -- see the module docstring.
    ``"theta"``
        halve the turn only. Radial resolution is then fixed by ``shells``.
        ``LANE_RN_UNIF.md`` names theta-halving for the T4 cover.
    ``"radius"``
        bisect the radius only.
    ``"both"``
        quadrisect.
    """

    name: str
    r_lo: Fraction
    r_hi: Fraction
    shells: Tuple[Fraction, ...]
    split: str = "aspect"
    coords: str = "polar(radius, turn); theta = 2*pi*turn"

    def __post_init__(self) -> None:
        if not (0 < self.r_lo < self.r_hi):
            raise ValueError("a polar cover needs 0 < r_lo < r_hi "
                             "(r_lo > 0 keeps the origin out of the region)")
        if self.shells[0] != self.r_lo or self.shells[-1] != self.r_hi:
            raise ValueError("shells must start at r_lo and end at r_hi")
        if any(b <= a for a, b in zip(self.shells, self.shells[1:])):
            raise ValueError("shells must be strictly increasing")
        if self.split not in ("aspect", "theta", "radius", "both"):
            raise ValueError(f"unknown split policy {self.split!r}")

    # ------------------------------------------------------------- geometry

    def domain(self) -> Box:
        return Box(self.r_lo, self.r_hi, Fraction(0), Fraction(1))

    def roots(self) -> List[Box]:
        return [Box(a, b, Fraction(0), Fraction(1))
                for a, b in zip(self.shells, self.shells[1:])]

    def classify(self, box: Box) -> str:
        """Always ``INSIDE``: the parameter rectangle *is* the region.

        Exact, and that is the point of choosing polar coordinates. A box that
        escaped the domain would be caught by the ledger's partition check
        before it ever reached here.
        """
        d = self.domain()
        if not d.contains_box(box):
            raise ValueError(f"{box!r} is outside the polar domain {d!r}")
        return INSIDE

    def area(self, box: Box, prec: int) -> Interval:
        """Certified enclosure of the cell area.

        ``area = int_{theta0}^{theta1} int_{r0}^{r1} r dr dtheta
               = (theta1 - theta0) * (r1^2 - r0^2) / 2
               = pi * (t1 - t0) * (r1^2 - r0^2)``

        since ``theta = 2*pi*t``. The rational factor is exact; ``pi`` comes
        from ``research/interval`` as a certified enclosure.
        """
        q = box.dv() * (box.u1 ** 2 - box.u0 ** 2)
        return Interval.exact(q) * pi(prec)

    def area_rational_upper(self, box: Box) -> Fraction:
        return box.dv() * (box.u1 ** 2 - box.u0 ** 2) * pi_upper()

    def diameter_bound(self, box: Box) -> Fraction:
        """Certified rational upper bound on the cell's Euclidean diameter.

        For ``p = r_p e(th_p)`` and ``q = r_q e(th_q)`` in the cell,

            |p - q| <= |r_p e(th_p) - r_p e(th_q)| + |r_p - r_q| * |e(th_q)|
                     = r_p * |e(th_p) - e(th_q)| + |r_p - r_q|
                    <= r_max * dtheta + dr

        using chord <= arc for the unit-circle difference. ``dtheta`` is bounded
        above by ``2*pi_upper * dv``.
        """
        return box.du() + box.u1 * two_pi_upper() * box.dv()

    def radius2_enclosure(self, box: Box) -> Interval:
        """Exact range of ``|y|^2`` over the cell: ``[r0^2, r1^2]``, rational."""
        return Interval(box.u0 ** 2, box.u1 ** 2)

    def cartesian_enclosure(self, box: Box, prec: int) -> Tuple[Interval, Interval]:
        """Certified enclosure of ``(y1, y2)`` over the cell.

        ``y1 = r cos(2*pi*t)``, ``y2 = r sin(2*pi*t)``. Both are evaluated as
        interval products of ``[r0, r1]`` with a certified trig enclosure over
        the turn interval, so the result is widened by the dependency problem
        (``r`` and the angle are treated as independent) but always contains the
        true range. Containment is what a bound needs; tightness is spent on
        refinement.
        """
        two_pi = 2 * pi(prec)
        ang = two_pi * Interval(box.v0, box.v1)
        r = Interval(box.u0, box.u1)
        return (r * cos(ang, prec), r * sin(ang, prec))

    # ------------------------------------------------------------ refinement

    def subdivide(self, box: Box) -> Tuple[Box, ...]:
        if self.split == "theta":
            return box.split_v()
        if self.split == "radius":
            return box.split_u()
        if self.split == "both":
            return box.quad()
        # "aspect": split the direction that dominates the diameter bound.
        radial = box.du()
        arc = box.u1 * two_pi_upper() * box.dv()
        if radial > arc:
            return box.split_u()
        if arc > radial:
            return box.split_v()
        return box.quad()

    # ------------------------------------------------- the refinement cost

    def uniform_cost(self, target_diameter: Fraction) -> dict:
        """Exact cell count for a UNIFORM cover at a target cell diameter.

        Not an estimate, not a fit, not a measurement: a count, in exact
        rationals, of what a uniform refinement would cost. Reported so the
        near-axis refinement cost is a number rather than a worry.

        A uniform grid of ``N_r`` radial steps and ``N_t`` angular steps gives a
        cell diameter bound ``dr + r_hi * 2*pi * dt``. Splitting the budget
        evenly between the two terms and solving gives the counts below.

        The ``inner_waste`` field is the anisotropy: at the *inner* radius the
        same angular step gives an arc extent smaller by ``r_lo / r_hi``, so a
        uniform angular grid is finer than it needs to be there by the
        reciprocal factor. For the RN5 annulus that factor is 50.
        """
        if target_diameter <= 0:
            raise ValueError("target diameter must be positive")
        half = target_diameter / 2
        n_r = -((-(self.r_hi - self.r_lo)) // half)          # ceil division
        n_t = -((-(self.r_hi * two_pi_upper())) // half)
        n_r, n_t = int(n_r), int(n_t)
        return {
            "target_diameter": target_diameter,
            "radial_steps": n_r,
            "angular_steps": n_t,
            "cells": n_r * n_t,
            "inner_arc_extent": self.r_lo * two_pi_upper() / n_t,
            "outer_arc_extent": self.r_hi * two_pi_upper() / n_t,
            "inner_waste_factor": self.r_hi / self.r_lo,
            "note": ("uniform cost only; the driver's 'aspect' policy refines "
                     "adaptively and will not match this count. NON-BINDING: "
                     "a cost model is not a certificate."),
        }


# -------------------------------------------------------------- cartesian


@dataclass(frozen=True)
class CartesianBracketRegion:
    """An annulus covered by rational axis-aligned boxes: the bracket option.

    The region is ``r_lo <= |y| <= r_hi`` again, but the cover lives in
    Cartesian ``(x, y)`` over a rational bounding box. Because the boundary
    circles are not rational polygons, some cells necessarily straddle the
    boundary. Those are REFINED while the depth budget allows and then
    REJECTED as ``UNRESOLVED_BOUNDARY`` with their area retained as an explicit
    **boundary-area bound** -- exactly RN5's "retaining boundary-area bounds and
    every rejected cell".

    Classification is exact rational arithmetic on squared radii; no square
    root and no trigonometry are needed, and no float is ever compared:

    * ``min |y|^2`` over a box is ``dx^2 + dy^2`` where ``dx`` is 0 if the box
      straddles ``x = 0`` and ``min(|x0|, |x1|)`` otherwise (likewise ``dy``);
    * ``max |y|^2`` is ``Dx^2 + Dy^2`` with ``Dx = max(|x0|, |x1|)``.

    Then the box is OUTSIDE if ``max |y|^2 < r_lo^2`` (entirely inside the
    excluded disc) or ``min |y|^2 > r_hi^2`` (entirely beyond the outer
    circle), INSIDE if ``min |y|^2 >= r_lo^2`` and ``max |y|^2 <= r_hi^2``, and
    STRADDLE otherwise. All four comparisons are between exact rationals.

    This region covers the SAME set as :func:`rn5_annulus_polar`. Its total and
    the polar total are two computations of one quantity and are never added
    together.
    """

    name: str
    r_lo: Fraction
    r_hi: Fraction
    half_width: Fraction
    split: str = "both"
    coords: str = "cartesian(y1, y2)"

    def __post_init__(self) -> None:
        if not (0 <= self.r_lo < self.r_hi):
            raise ValueError("need 0 <= r_lo < r_hi")
        if self.half_width < self.r_hi:
            raise ValueError(
                "the bracket box must contain the outer circle, else the cover "
                "misses part of the region and no bound is possible")

    def domain(self) -> Box:
        h = self.half_width
        return Box(-h, h, -h, h)

    def roots(self) -> List[Box]:
        return [self.domain()]

    # --------------------------------------------------- exact classification

    @staticmethod
    def _axis_min_max(a: Fraction, b: Fraction) -> Tuple[Fraction, Fraction]:
        lo = Fraction(0) if (a <= 0 <= b) else min(abs(a), abs(b))
        hi = max(abs(a), abs(b))
        return lo, hi

    def radius2_range(self, box: Box) -> Tuple[Fraction, Fraction]:
        """Exact range of ``|y|^2`` over the box, as rationals."""
        xl, xh = self._axis_min_max(box.u0, box.u1)
        yl, yh = self._axis_min_max(box.v0, box.v1)
        return (xl * xl + yl * yl, xh * xh + yh * yh)

    def classify(self, box: Box) -> str:
        lo2, hi2 = self.radius2_range(box)
        rlo2, rhi2 = self.r_lo ** 2, self.r_hi ** 2
        if hi2 < rlo2 or lo2 > rhi2:
            return OUTSIDE
        if lo2 >= rlo2 and hi2 <= rhi2:
            return INSIDE
        return STRADDLE

    # ------------------------------------------------------------- geometry

    def area(self, box: Box, prec: int) -> Interval:
        """Exact: a rational rectangle's area is rational."""
        return Interval.exact(box.param_area())

    def area_rational_upper(self, box: Box) -> Fraction:
        return box.param_area()

    def diameter_bound(self, box: Box) -> Fraction:
        """``sqrt(du^2 + dv^2) <= du + dv``, rational and certified."""
        return box.du() + box.dv()

    def radius2_enclosure(self, box: Box) -> Interval:
        lo2, hi2 = self.radius2_range(box)
        return Interval(lo2, hi2)

    def cartesian_enclosure(self, box: Box, prec: int) -> Tuple[Interval, Interval]:
        return (Interval(box.u0, box.u1), Interval(box.v0, box.v1))

    def subdivide(self, box: Box) -> Tuple[Box, ...]:
        if self.split == "u":
            return box.split_u()
        if self.split == "v":
            return box.split_v()
        return box.quad()


# ------------------------------------------------------------ instantiations


def rn5_annulus_polar(split: str = "aspect",
                      shells: Sequence[Fraction] | None = None) -> PolarRegion:
    """The RN5 Piece 2 region ``0.1 <= |y| <= 5``, in polar cells.

    From ``docs/OPEN_PROBLEMS.md`` A5, quoting RN5's next exact action. The
    radii ``1/10`` and ``5`` are exact rationals, so the boundary is exact and
    the rejected boundary area is zero.

    Default ``shells`` are ``1/10, 1/2, 1, 2, 5`` -- geometric-ish, so the root
    cells already reflect the 50x anisotropy between the inner and outer edges
    instead of handing the whole job to the adaptive policy.

    This region is NOT the T4 region. Do not merge them.
    """
    r_lo, r_hi = Fraction(1, 10), Fraction(5)
    sh = tuple(shells) if shells is not None else (
        r_lo, Fraction(1, 2), Fraction(1), Fraction(2), r_hi)
    return PolarRegion(name="RN5-annulus 0.1<=|y|<=5 (polar)",
                       r_lo=r_lo, r_hi=r_hi, shells=sh, split=split)


def rn5_annulus_bracket(half_width: Fraction = Fraction(5),
                        split: str = "both") -> CartesianBracketRegion:
    """The SAME region ``0.1 <= |y| <= 5``, as a rational Cartesian bracket.

    The alternative honest handling of a boundary that is not a rational
    polygon: straddling cells are rejected with their areas retained as
    explicit boundary-area bounds. Present so that the sliver accounting is
    exercised and tested; its number is never added to the polar total.
    """
    return CartesianBracketRegion(name="RN5-annulus 0.1<=|y|<=5 (cartesian bracket)",
                                  r_lo=Fraction(1, 10), r_hi=Fraction(5),
                                  half_width=half_width, split=split)


def t4_polar_cover(split: str = "theta",
                   shells: Sequence[Fraction] | None = None) -> PolarRegion:
    """The T4 push region ``d in [5, 17]`` with theta-halving.

    From ``LANE_RN_UNIF.md``'s T4 freeze list, item 2: "Polar cover d in
    [5, 17] with theta-halving", repeated as EXECUTE item 4, "full polar cover
    d in [5, 17] with theta-halving".

    Default ``shells`` are the unit shells ``5, 6, ..., 17``. With
    ``split="theta"`` -- the policy the lane names -- **radial resolution is
    fixed by this shell list**: theta-halving never shrinks a cell's radial
    extent, so a cell whose radial width alone keeps it over tolerance will end
    the run PENDING, and ``total()`` will refuse. That is the intended
    behaviour, not a defect: it is visible in the receipt rather than papered
    over. Pass ``split="aspect"`` to allow radial bisection as well.

    Two things this factory does NOT do, and cannot:

    * it does not supply T4. ``T4_form``, ``T4_kap`` and ``C_comp`` are not in
      the frozen engine at all (``docs/ENGINE_RECOVERY.md``), and ``rnu_t4.py``
      is listed CANNOT_VERIFY in ``docs/OPEN_PROBLEMS.md``. This is the cover
      geometry the lane names and nothing else;
    * it does not touch ``env_tau``, which ``LANE_RN_UNIF.md`` records as
      fail-closing at ``d = 5`` (lambda-0 floor collapsed). Do not use this
      region to evaluate anything on that zone boundary and call it certified.

    This region is NOT the RN5 annulus. Do not merge them.
    """
    r_lo, r_hi = Fraction(5), Fraction(17)
    sh = tuple(shells) if shells is not None else tuple(
        Fraction(k) for k in range(5, 18))
    return PolarRegion(name="T4 polar cover d in [5,17] (theta-halving)",
                       r_lo=r_lo, r_hi=r_hi, shells=sh, split=split)
