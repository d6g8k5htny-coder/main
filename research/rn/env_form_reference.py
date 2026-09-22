"""The ``env_form`` envelope shape, over certified intervals, on reference data.

WHAT THIS IS. ``docs/OPEN_PROBLEMS.md`` A5 carries, from RN5, "Build the
whitened ``env_form`` orders 2-4 runnable smoke (currently missing); order-1
chi-squared white is a nearest neighbour only." This module is the part of that
directive which the sources determine. It is **not** the whitened artifact, and
``WHY THERE IS NO WHITENING HERE`` below says why not, in the sources' own
words.

``env_form(k, gamma, d, qord)`` of the frozen engine
(``engine/rn_engine/frozen/K3_SIDE24_LB/UPPER2D/D3_percolation/d3_rn_unif.py``
line 381) returns a sum of three parts, each an exact rational coefficient
multiplying one Gaussian factor:

    tot = C_tot(moments, gamma, d, q) * exp(-d^2/2)
    rem = C_rem(forms, gamma, rho, q) * exp(-rho^2/2)      rho  = d - R/2
    img = C_img(forms, q, rimg)       * exp(-rimg^2/2)     rimg = 24 - d - R/2

Every ``C`` is built from ``he_abs`` -- an integer-coefficient polynomial --
times factorials, binomials and the moment or coefficient magnitudes. So each
``C`` is exact in ``fractions.Fraction``, and each part is ONE exact rational
times ONE certified exponential. There is no dependency problem to fight: the
enclosure of each part is as tight as ``research/interval/exp`` allows, and the
total is the sum of three such.

The frozen engine computes the same shape in ``mpmath`` at ``mp.dps = 100``
(line 41). High precision is not certification. That this module encloses where
the frozen body approximates is a difference in kind, and it certifies nothing
about the frozen body's numbers, which remain exactly as certified or as
uncertified as their own sources say.

THE ENVELOPE IS NOT MONOTONE IN ``d``
-------------------------------------
The frozen docstring calls ``env_form`` a "certified bound of
|d^q F_{R_k,gamma}(y)| for |y| >= d". Enlarging ``d`` shrinks the set the bound
covers, so nothing forces the bound itself to shrink with it -- and it does
not. Two of the three parts move in opposite directions, and this is a property
of the SHAPE, not of any particular data:

* the moment series carries ``exp(-d^2/2)`` against a polynomial in ``d``, so
  it decays;
* the image allowance sits at ``rimg = 24 - d - R/2``, which DECREASES as ``d``
  grows, so ``exp(-rimg^2/2)`` GROWS.

A crossover therefore exists for every moment table. WHERE it sits depends on
the table, and the numbers below are for the REFERENCE data in this module at
``qord = 2`` -- they are not the program's and must not be read as such:

| ``d`` | image / moment series |
|---|---|
| 5 | ``1.0e-68`` |
| 10 | ``2.2e-20`` |
| 12 | ``0.77`` |
| 13 | ``4.7e+09`` |
| 17 | ``5.5e+48`` |

On this data the crossover is between ``d = 12.0115565`` and ``12.0115566``,
the total bottoms out near ``d = 12.0111`` at about ``1.18e-21``, and by
``d = 23`` it is some ``5.3e+24`` times that minimum. Past
``d = 24 - R/2 = 23.975`` the image separation is negative and the construction
has no referent at all, which is why ``env_form_parts`` refuses it.

Worth stating plainly, because it is the part that bears on the lane: the
crossover on this reference data falls INSIDE the RN-UNIF lane's own T4 region
``d`` in ``[5, 17]``. At ``d = 5``, where the push evaluated, the image
allowance is 68 orders of magnitude below the moment series and costs nothing.
At ``d = 17`` it is 48 orders above it and is the whole bound.

None of this says the frozen body is wrong, and none of it is a statement about
the program's envelope: the program's ``MOMS`` and ``FORMS`` are not here, and
a different moment table moves the crossover. What is data-independent is that
a crossover exists, that "take ``d`` larger to get a smaller bound" therefore
stops working somewhere, and that where it stops is computable once the moments
are supplied. It is an observation about a shape. No status turns on it.

WHY THERE IS NO WHITENING HERE`` below says why not, in the sources' own
words.

``env_form(k, gamma, d, qord)`` of the frozen engine
(``engine/rn_engine/frozen/K3_SIDE24_LB/UPPER2D/D3_percolation/d3_rn_unif.py``
line 381) returns a sum of three parts, each an exact rational coefficient
multiplying one Gaussian factor:

    tot = C_tot(moments, gamma, d, q) * exp(-d^2/2)
    rem = C_rem(forms, gamma, rho, q) * exp(-rho^2/2)      rho  = d - R/2
    img = C_img(forms, q, rimg)       * exp(-rimg^2/2)     rimg = 24 - d - R/2

Every ``C`` is built from ``he_abs`` -- an integer-coefficient polynomial --
times factorials, binomials and the moment or coefficient magnitudes. So each
``C`` is exact in ``fractions.Fraction``, and each part is ONE exact rational
times ONE certified exponential. There is no dependency problem to fight: the
enclosure of each part is as tight as ``research/interval/exp`` allows, and the
total is the sum of three such.

The frozen engine computes the same shape in ``mpmath`` at ``mp.dps = 100``
(line 41). High precision is not certification. That this module encloses where
the frozen body approximates is a difference in kind, and it certifies nothing
about the frozen body's numbers, which remain exactly as certified or as
uncertified as their own sources say.

THE ENVELOPE IS NOT MONOTONE IN ``d``
-------------------------------------
The frozen docstring calls ``env_form`` a "certified bound of
|d^q F_{R_k,gamma}(y)| for |y| >= d". Enlarging ``d`` shrinks the set the bound
covers, so nothing forces the bound itself to shrink with it -- and it does
not. The image allowance sits at ``rimg = 24 - d - R/2``, which *decreases* as
``d`` grows, so that part *grows*. Computed here on the reference data at
``qord = 2``, with every value a certified enclosure:

* the moment series falls and the image allowance rises, and they cross at
  ``d`` between 12.0115565 and 12.0115566;
* the total bottoms out near ``d = 12.0111`` at about ``1.18e-21``;
* by ``d = 23`` it is about ``6.26e+03`` -- some ``5.3e+24`` times its minimum;
* past ``d = 24 - R/2 = 23.975`` the image separation is negative and the
  construction has no referent at all, which is why ``env_form_parts`` refuses
  it.

None of this says the frozen body is wrong. At the separations the engine
actually uses -- ``d = 5`` in the push, ``d`` in ``[5, 17]`` for T4 -- the
image allowance is 24 to 39 orders of magnitude below the moment series and the
envelope is falling steeply. The observation is that "take ``d`` larger to get
a smaller bound" stops working at a computable place, and that the best
available bound of this shape is the one at the minimum. It is an observation
about a shape, on reference data; it is not a bound on anything of the
program's, and no status turns on it.

WHY THERE IS NO WHITENING HERE
------------------------------
The directive names a *whitened* ``env_form``. Whitening itself is defined
without ambiguity -- a y-independent frame change ``W`` with ``W S0 W^T = I``,
built in the order-1 carrier as ``W[k, j] = V0[j, k] / sqrt(LAM0[k])``
(``engine/carriers/blobs/6b61af7b549c46b0__rnu_chi2_white_v2.py`` lines 13-17).
What a whitened ``env_form`` is, is NOT determined, and the sources disagree:

* the one build that ever realised it computed ``env_form(k, g, d, q)/sqrt(LAM0[k])``
  for ``q`` in ``{2, 3, 4}`` at ``d = 5`` and stamped its own result
  "**PROXY only**", adding "Native ``env_form`` already acts on V0-basis
  residual FORMS; the stub applies the CL-RNU-001-cited 1/sqrt(lambda) scaling.
  This is **not** the missing whitened residual-covariance construction for
  log q(A, m')." (``drive/mirrors/2026-09-16 - HOLD_NOT_FOR_SUBMISSION/
  MATH_PUSH_WHITENED_ENV_FORM_2_4.md`` lines 30 and 34);
* and the author-side jet theorem refuses exactly that reading: "Norms of all
  covariance derivatives refer to their actual ordered Schur/inverse
  construction, not to a residual form divided by sqrt(lambda)."
  (``...MATH-20260917-b9c2_RN_WHITENED_JET_THEOREM.md`` line 150).

Choosing between a reading a source calls a proxy and a reading a later source
refuses is a mathematical decision, and this repository does not make those. So
this module builds the shape and stops at the whitening, and the lane records
the conflict rather than resolving it. Both documents are in the Drive lane
whose own name is its status word -- HOLD, not for submission -- and are quoted
here as evidence of what the sources say, never as authority for what is true.

WHAT THIS MODULE DOES NOT ESTABLISH
-----------------------------------
* It is **not** the missing whitened smoke, and it does not close, discharge,
  promote or reclassify anything. ``D3-LEMMA-RN-UNIF`` Piece 1 and Piece 2 stay
  **OPEN**; the lane receipts still read ``lemma_closed: false``.
* It carries **no moment table and no form** of the program. ``MOMS[k]`` and
  ``FORMS[k]`` are frozen-engine six-pin geometry. The caller supplies
  reference data, and the reference data shipped here is labelled REFERENCE and
  is not the program's. **No envelope of the program's actual residual forms is
  computed here, at any order.**
* It re-certifies nothing the frozen engine computed, supplies no cell
  supremum, no band enclosure, no rung and no remote budget, composes no two of
  the three tracks, and bears on no prize problem.
* A run of this module is a run.

Standard library only (``fractions``, ``math``, ``dataclasses``, ``typing``);
``math.comb`` and ``math.factorial`` return exact integers and no float is
formed anywhere. Python 3.11. Certified enclosures come from
``research/interval/``.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import comb, factorial
from typing import Iterable, Mapping, Sequence

from research.interval import Interval
from research.rn.hermite_envelope import gaussian_kernel, he_abs

__all__ = [
    "EnvFormParts", "env_form_parts", "env_form_enclosure",
    "REFERENCE_R", "TORUS_PERIOD", "REMAINDER_ORDER", "MX2_INDEX_SUM",
    "REFERENCE_MOMENTS", "REFERENCE_FORMS",
]

#: ``R = mpf('0.05')`` in the frozen body (line 87). Carried here as the exact
#: rational 1/20. The frozen value is the binary double nearest 0.05 and is
#: therefore NOT exactly 1/20; the two agree to about 3e-18 and the difference
#: is stated rather than hidden. Nothing here re-certifies the frozen value.
REFERENCE_R = Fraction(1, 20)

#: ``_LT = 24`` (line 98): the torus period.
TORUS_PERIOD = 24

#: The Taylor remainder is taken at order 9; moments are carried to order 8.
REMAINDER_ORDER = 9

#: The ``mx2`` loop of the frozen body runs ``for d1 in range(12): d2 = 11 - d1``
#: (lines 409-410), so its index pairs sum to 11 -- while the comment on line
#: 408 reads ``|beta|=9``. The code and its comment disagree. This module
#: reproduces the CODE, because the code is what produced the frozen numbers,
#: and records the disagreement here rather than silently choosing. It is an
#: observation about a frozen body, which is never edited in place; it is not a
#: defect report against any claim, and no status turns on it.
MX2_INDEX_SUM = 11


@dataclass(frozen=True)
class EnvFormParts:
    """The three parts, each as an exact coefficient and its Gaussian factor.

    Keeping the parts separate is the point: a consumer can see which part
    dominates at a given ``d`` and ``qord`` without re-deriving it, and a test
    can pin one part while another changes.
    """

    tot_coefficient: Fraction
    remainder_coefficient: Fraction
    image_coefficient: Fraction
    d: Fraction
    rho: Fraction
    rimg: Fraction
    qord: int

    def enclosure(self, prec: int) -> Interval:
        """A certified enclosure of ``tot + rem + img``.

        Each part is one exact rational times one certified exponential, so
        each is enclosed with no dependency widening; the sum of three
        enclosures is an enclosure of the sum.
        """
        total = Interval.exact(0)
        for coeff, sep in ((self.tot_coefficient, self.d),
                           (self.remainder_coefficient, self.rho),
                           (self.image_coefficient, self.rimg)):
            if coeff == 0:
                continue
            k = gaussian_kernel(Interval.exact(sep * sep), prec)
            total = total + k * Interval.exact(coeff)
        return total


def _split_sum(gamma: Sequence[int], qord: int, n1: int, n2: int,
               at: Fraction) -> Fraction:
    """``sum_{e1+e2=qord} C(qord, e1) he_abs(n1+g0+e1, at) he_abs(n2+g1+e2, at)``.

    The binomial split over the ``qord`` derivatives, exactly as the frozen
    body performs it in all three parts.
    """
    acc = Fraction(0)
    for e1 in range(qord + 1):
        e2 = qord - e1
        acc += (comb(qord, e1)
                * he_abs(n1 + gamma[0] + e1, at)
                * he_abs(n2 + gamma[1] + e2, at))
    return acc


def env_form_parts(moments: Mapping[tuple[int, int], Fraction],
                   forms: Iterable[tuple[tuple[int, int], Fraction]],
                   gamma: Sequence[int], d: Fraction, qord: int,
                   *, R: Fraction = REFERENCE_R) -> EnvFormParts:
    """The exact coefficients of the three parts, for the supplied data.

    ``moments`` maps a multi-index ``(b1, b2)`` to the monomial moment
    ``mu_beta`` of a residual form; ``forms`` is a sequence of
    ``((a1, a2), c)`` pairs, a derivative multi-index and its coefficient. Both
    are REFERENCE data supplied by the caller. This module holds none of the
    program's, and will not invent any: an envelope is only as meaningful as
    the moments it is taken over, and inventing them would produce a number
    that looks like the program's and is not.

    The frozen body skips moments with ``abs(mu) < 1e-80``, a float cutoff
    standing in for a parity certificate. Here a moment is skipped when it is
    exactly zero, which is the exact statement of the same intent; a caller
    wanting the float cutoff's behaviour must apply it to its own data, and
    then it is that caller's approximation and not this module's.
    """
    if qord < 0:
        raise ValueError("qord must be non-negative")
    if len(gamma) != 2:
        raise ValueError("gamma must be a pair")
    d = Fraction(d)
    rho = d - R / 2
    rimg = TORUS_PERIOD - d - R / 2
    # Both separations must be positive for the construction to mean anything.
    # `he_abs` takes `abs(t)`, so a negative separation does not raise; it
    # silently returns a number built from a distance that does not exist. At
    # `d = 24` the image separation is -1/40 and the coefficient comes back
    # larger than at `d = 23.975`, which is the shape of a quantity with no
    # referent. The frozen body has no such guard because it is never called
    # near there; this module refuses rather than inherit the hole.
    if rho <= 0:
        raise ValueError(
            f"rho = d - R/2 = {rho} is not positive; the Taylor remainder is "
            f"taken at that separation and has no meaning at or below zero")
    if rimg <= 0:
        raise ValueError(
            f"rimg = {TORUS_PERIOD} - d - R/2 = {rimg} is not positive; the "
            f"torus-image allowance is an allowance at that separation, and "
            f"beyond d = {TORUS_PERIOD} - R/2 there is no image shell to allow for")

    tot = Fraction(0)
    for (b1, b2), mu in moments.items():
        mu = Fraction(mu)
        if mu == 0:
            continue
        base = abs(mu) / (factorial(b1) * factorial(b2))
        tot += base * _split_sum(gamma, qord, b1, b2, d)

    forms = list(forms)
    coeff_l1 = sum((abs(Fraction(c)) for _, c in forms), Fraction(0))

    inner_total = Fraction(0)
    for a, c in forms:
        ao = a[0] + a[1]
        inner = Fraction(0)
        for b1 in range(REMAINDER_ORDER + 1):
            b2 = REMAINDER_ORDER - b1
            if b1 < a[0] or b2 < a[1]:
                continue
            inner += ((R / 2) ** (REMAINDER_ORDER - ao)
                      / (factorial(b1 - a[0]) * factorial(b2 - a[1])))
        inner_total += abs(Fraction(c)) * inner

    # The frozen body takes the max over each (d1, e1) TERM, not over the
    # binomial sum: lines 411-415 run the `e1` loop inside the `max` call. Part
    # 1 sums the split and part 2 maximises over it, and the two are not the
    # same number. Reusing `_split_sum` here was a real bug in an earlier draft
    # of this module; `test_against_an_independent_reimplementation` caught it,
    # which is what that test is for.
    mx2 = Fraction(0)
    for d1 in range(MX2_INDEX_SUM + 1):
        d2 = MX2_INDEX_SUM - d1
        for e1 in range(qord + 1):
            e2 = qord - e1
            mx2 = max(mx2, comb(qord, e1)
                      * he_abs(d1 + gamma[0] + e1, rho)
                      * he_abs(d2 + gamma[1] + e2, rho))

    img = coeff_l1 * 8 * he_abs(6 + qord, rimg)

    return EnvFormParts(tot_coefficient=tot,
                        remainder_coefficient=inner_total * mx2,
                        image_coefficient=img,
                        d=d, rho=rho, rimg=rimg, qord=qord)


def env_form_enclosure(moments, forms, gamma, d, qord, prec: int,
                       *, R: Fraction = REFERENCE_R) -> Interval:
    """A certified enclosure of the ``env_form`` shape on the supplied data."""
    return env_form_parts(moments, forms, gamma, d, qord, R=R).enclosure(prec)


# ---------------------------------------------------------------------------
# REFERENCE data. Chosen because it is simple enough to check by hand, and
# labelled so nobody mistakes it for the program's. It is NOT MOMS[k] and NOT
# FORMS[k]; those are frozen-engine six-pin geometry and are not in this
# repository's research layer at all.
# ---------------------------------------------------------------------------

#: A reference moment table: the monomial moments of a standard 2D Gaussian
#: truncated to total order 8, so ``mu_(b1, b2)`` is ``(b1-1)!! (b2-1)!!`` for
#: even indices and 0 otherwise. Nothing in the program has these moments.
REFERENCE_MOMENTS: dict[tuple[int, int], Fraction] = {}
for _b1 in range(9):
    for _b2 in range(9 - _b1):
        if _b1 % 2 or _b2 % 2:
            continue
        _v = Fraction(1)
        for _j in range(1, _b1, 2):
            _v *= _j
        for _j in range(1, _b2, 2):
            _v *= _j
        REFERENCE_MOMENTS[(_b1, _b2)] = _v

#: A reference form: three terms of small mixed order with unit coefficients.
REFERENCE_FORMS: tuple[tuple[tuple[int, int], Fraction], ...] = (
    ((0, 0), Fraction(1)),
    ((1, 0), Fraction(-1, 2)),
    ((1, 1), Fraction(1, 4)),
)
