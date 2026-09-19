"""Band-certification machinery for the chart-side blocking proof step (lane A1).

``OBL-H5-JETMOD`` asks for *certified interval bounds for the full 24-jet set
(not just the displayed c2) over the r-bands [r_{k+1}, r_k], with lattice-tail
constants re-certified uniformly in the band*. The proof step it names is
*evaluating those sums with r as an interval over the band* -- a finite
computation per band, never a fitted exponent.

This package contains three things:

``ladder``
    The three published ``I_hi/r^3`` **point** certifications as exact
    ``Fraction`` data with their source packages and totals digests, plus the
    display-vs-enclosure gap made numerical: the same-``r`` version spread at
    ``r = 0.05`` and the modulus constant each adjacent band forces under a
    **stated assumption** about how to read the displayed ``kappa = 1/8`` fit.

``lattice``
    A certified periodized lattice-sum evaluator: the truncated sum evaluated
    with the displacement as an interval box, plus a **proved** tail bound for
    the omitted lattice points that is **uniform over that box**, with a
    pluggable kernel protocol and reference kernels.

``falsifier``
    The obligation's own falsifier as an executable check, with
    ``INSUFFICIENT_DATA`` as a first-class outcome.

WHAT THIS PACKAGE DOES NOT ESTABLISH
------------------------------------
* It does **not** discharge, reduce, close, promote or reclassify
  ``OBL-H5-JETMOD``. That obligation stays **OPEN (display only)**, exactly as
  ``docs/OPEN_PROBLEMS.md`` records it. So do ``OBL-H5-ZBAND`` (hi side),
  ``OBL-H5-REMOTE-THRESHOLD``, ``OBL-D1-PROMOTE`` (chart side) and both Pieces
  of ``D3-LEMMA-RN-UNIF``.
* Three bindings are missing before anything computed here could bear on the
  obligation, and none of the three exists in this repository: the program's
  own ``kplane`` **with a certified decay envelope for it**; the 24-jet
  definitions with their powers ``p_J``; and the actual band endpoints ``r_k``.
  What ships here is machinery plus reference kernels.
* The point certifications in ``ladder`` remain **point** certifications. No
  arithmetic performed on them turns them into a band enclosure.
* The implied-modulus analysis is an observation under a **stated reading** of
  a **display**. It is not a refutation of any source's modulus and must never
  be presented as one.
* Nothing here composes the 2D upper track, the 2D lower track or the 3D
  lifetime track in any way, and nothing here bears on any prize problem.

Standard library only, plus ``research/interval``. Python 3.11.
"""
from .falsifier import (
    FALSIFIED, FALSIFIER_RULE, INSUFFICIENT_DATA, OUTCOMES, PASS, BandCheck,
    BandResult, band_report, band_verdict, falsifies, report_is_clean,
    report_summary,
)
from .ladder import (
    DISPLAYED_KAPPA, IMPLIED_MODULUS_ASSUMPTION, PUBLISHED_POINTS,
    AdjacentBand, ImpliedModulus, RungPoint, adjacent_bands,
    implied_modulus_constant, implied_modulus_ratio, implied_modulus_table,
    ratio_table, same_r_version_spread,
)
from .lattice import (
    ENGINE_TRUNCATION, GAUSSIAN, POWER, SIDE24_PERIOD, BandEnclosure,
    DecayEnvelope, PlaneKernel, axial_displacement, band_enclosure,
    diagonal_displacement, gaussian_reference, inverse_power_reference,
    normalized_band_enclosure, tail_bound, truncated_sum,
)

__all__ = [
    # ladder
    "PUBLISHED_POINTS", "RungPoint", "AdjacentBand", "ImpliedModulus",
    "DISPLAYED_KAPPA", "IMPLIED_MODULUS_ASSUMPTION",
    "adjacent_bands", "same_r_version_spread", "implied_modulus_constant",
    "implied_modulus_table", "implied_modulus_ratio", "ratio_table",
    # lattice
    "DecayEnvelope", "PlaneKernel", "BandEnclosure", "GAUSSIAN", "POWER",
    "SIDE24_PERIOD", "ENGINE_TRUNCATION",
    "gaussian_reference", "inverse_power_reference",
    "axial_displacement", "diagonal_displacement",
    "truncated_sum", "tail_bound", "band_enclosure",
    "normalized_band_enclosure",
    # falsifier
    "PASS", "FALSIFIED", "INSUFFICIENT_DATA", "OUTCOMES", "FALSIFIER_RULE",
    "BandCheck", "BandResult", "falsifies", "band_verdict", "band_report",
    "report_summary", "report_is_clean",
]
