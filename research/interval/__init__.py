"""Certified interval arithmetic over exact rational endpoints.

THE CONTRACT. Every ``Interval`` returned by this package provably contains the
true value. Tightness is best effort and depends on the ``prec`` hint;
CONTAINMENT IS UNCONDITIONAL and never depends on it.

Usage::

    from fractions import Fraction
    from research.interval import Interval, pi, sin, Phi

    x = Interval("1/10", "1/5")      # exact rational endpoints
    sin(x, 30)                       # enclosure of sin over [1/10, 1/5]
    Phi(Interval.exact(1), 25)       # enclosure of the standard normal CDF at 1

WHAT THIS PACKAGE DOES NOT ESTABLISH
------------------------------------
* It discharges, reduces, closes, promotes and reclassifies **nothing**. Every
  obligation in ``docs/OPEN_PROBLEMS.md`` — ``OBL-H5-JETMOD``,
  ``OBL-H5-ZBAND``, ``OBL-H5-REMOTE-THRESHOLD``, ``OBL-D1-PROMOTE``,
  ``D3-LEMMA-RN-UNIF``, ``PERC-DECAY``, ``OBL-B1-BRANCH``, the ``B4.loc``
  wrap/remote reconciliation — stands exactly as recorded there. Having a
  certified arithmetic is not having a certified result.
* It re-certifies no existing number. Quantities in the corpus that were
  produced in ``mpmath`` or ``numpy`` remain as certified (or as uncertified) as
  their own sources say they are. Nothing here relabels them.
* It supplies no band enclosure, no lattice sum, no jet, no rung, no moment and
  no coverage certificate. ``OBL-H5-JETMOD`` asks for a finite per-band
  computation; this package contains none of it.
* It relates the 2D upper track, the 2D lower track and the 3D lifetime track
  in no way whatsoever, and composes none of them.
* It solves no prize problem and bears on none.
* Passing tests are evidence that the code does what its docstrings say. They
  are not a mathematical review of the enclosure arguments, which are ordinary
  mathematics written out in the docstrings for a human to check.

Standard library only (``fractions``, ``decimal``, ``typing``). Python 3.11.
"""
from .core import Interval, to_fraction
from .transcendental import (
    ERF_CROSSOVER, Phi, cos, erf, exp, log, normal_pdf, pi, sin, sqrt,
)

__all__ = [
    "Interval", "to_fraction",
    "sqrt", "exp", "log", "sin", "cos", "pi", "erf", "Phi", "normal_pdf",
    "ERF_CROSSOVER",
]
