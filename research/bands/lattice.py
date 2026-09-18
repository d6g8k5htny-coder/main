"""Certified periodized lattice sums on the side-24 torus, with a PROVED tail.

WHAT THIS MODULE IS. A periodized kernel on a torus of period ``L`` is the
lattice sum

    S(d) = sum_{n in Z^2} kplane(d + L*n),                                 (1)

which no machine evaluates: it is infinite. Every implementation truncates it.
The frozen engine ``d3_rn_unif.py`` truncates at the **first image shell only**
-- ``_IMG = [(i, j) for i in (-1,0,1) for j in (-1,0,1) if (i,j) != (0,0)]``
with ``_LT = 24`` -- so ``kdcov`` sums exactly nine plane-kernel evaluations and
**returns that nine-point sum**. Its ``tail_bound(order)`` is asserted at import
time to be below ``1e-60``, but the value it bounds is never added to anything:
no enclosure widens by it. That tail bound is also (a) computed at a *point*
separation, via the hard-coded ``rho = m*_LT - 17``, (b) itself truncated, via
``for m in range(2, 8)``, and (c) mpmath floating point throughout. Points (a)
and (b) are exactly what ``OBL-H5-JETMOD`` complains of: "lattice-tail constants
re-certified **uniformly in the band** (the current LAT tail bound certifies at
point separations only)".

This module builds the other version:

  * the truncated sum is evaluated with the displacement given as an **interval
    box**, through ``research/interval``, so one enclosure is valid for *every*
    displacement in the box simultaneously;
  * the omitted lattice points are covered by a **proved** tail bound that is
    uniform over that box, with no second truncation anywhere in it;
  * the kernel is **pluggable**: a caller supplies a callable together with a
    certified decay envelope, because the program's own ``kplane`` is not bound
    here.

WHAT THIS MODULE DOES **NOT** ESTABLISH
---------------------------------------
* It does **not** discharge, reduce, close, promote or reclassify
  ``OBL-H5-JETMOD`` or anything else. That obligation is over the program's own
  24-jet set, with the program's kernel, over the program's r-bands. This file
  ships machinery and a reference kernel. Machinery is not a result.
* The reference kernels here are **REFERENCE KERNELS**, for exercising and
  testing the machinery. The unit Gaussian ``exp(-|z|^2/2)`` is the engine's
  ``C_KERN * kern(d2)`` at derivative multi-index ``(0, 0)`` and nothing more:
  the program's ``kplane(a1, a2, dx, dy)`` carries Hermite factors
  ``(-1)^(a1+a2) He_{a1}(dx) He_{a2}(dy)`` that are **not** implemented here and
  whose certified decay envelope is **not** established here. Swapping in the
  program's kernel, with a certified envelope for it, is what would make any
  output of this module bear on ``OBL-H5-JETMOD``.
* The default displacement map (see ``axial_displacement``) is a stand-in. The
  program's map from a pin separation ``r`` to the covariance displacements of
  its six-pin configuration is not bound here.
* Nothing here is a jet. The 24-jet set and its powers ``p_J`` are not defined
  in this repository.
* No number computed here is a rung, a certificate, a promotion or a
  measurement of anything in the corpus.

Standard library only (``fractions``, ``dataclasses``, ``typing``), plus
``research.interval``. Python 3.11.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction as F
from typing import Callable, Dict, Optional, Tuple

from research.interval import Interval, exp, log, sqrt

__all__ = [
    "GAUSSIAN", "POWER",
    "DecayEnvelope", "PlaneKernel", "BandEnclosure",
    "gaussian_reference", "inverse_power_reference",
    "axial_displacement", "diagonal_displacement",
    "truncated_sum", "tail_bound", "band_enclosure",
    "normalized_band_enclosure", "NormalizedBandEnclosure",
    "SIDE24_PERIOD", "ENGINE_TRUNCATION",
]

#: The side-24 torus period, matching the frozen engine's ``_LT = 24``.
SIDE24_PERIOD = F(24)

#: The frozen engine's truncation radius in the sup norm: ``_IMG`` is the first
#: image shell only, so ``|n|_inf <= 1`` and nine plane-kernel values are summed
#: per covariance entry. This module's default ``n_trunc`` matches it so that a
#: comparison is like for like.
ENGINE_TRUNCATION = 1

GAUSSIAN = "gaussian"
POWER = "power"


# ---------------------------------------------------------------------------
# The kernel protocol
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DecayEnvelope:
    """A **certified** decay envelope for a plane kernel, plus its proof.

    Two forms are supported, both understood as bounds on the *absolute value*
    of the plane kernel at Euclidean distance ``|z|``:

    ``kind == GAUSSIAN``
        ``|kplane(z)| <= A * exp(-B * |z|^2)``  for all ``|z| >= valid_from``,
        with ``A > 0`` and ``B > 0``.

    ``kind == POWER``
        ``|kplane(z)| <= A * |z|^(-p)``         for all ``|z| >= valid_from``,
        with ``A > 0`` and ``p > 2``. The restriction ``p > 2`` is not
        cosmetic: ``sum_m m * m^(-p)`` diverges at ``p <= 2``, so the lattice
        sum need not converge and no tail bound exists.

    ``valid_from`` exists so an envelope may be honest about a kernel it does
    not dominate near the origin. :func:`tail_bound` **checks** that every
    omitted lattice point lies at distance at least ``valid_from`` and raises
    rather than proceed if it does not.

    ``justification`` must be a non-empty human-readable proof of the envelope.
    An envelope with no argument behind it is a guess, and this dataclass
    refuses to carry one: the constructor raises on an empty justification.
    That is a discipline against the failure mode this program exists to avoid
    -- an uncertified constant travelling inside a structure whose name says
    "certified".

    ``certified`` -- DEFAULT ``False``, AND THAT DEFAULT IS THE POINT. A
    non-empty ``justification`` string is a *claim* that the envelope holds; it
    is not a check that it does, and this module cannot check it: ``A``, ``B``
    and ``p`` are constants about a function whose global behaviour no finite
    computation here inspects. ``certified`` is the caller's explicit assertion
    that the written justification is a proof a human has read. Nothing in this
    package sets it to ``True`` except the two reference kernels, whose
    envelopes are one-line consequences of their own definitions.

    Why this flag exists at all: the module's own negative control
    ``test_control_a_false_decay_envelope_loses_domination`` demonstrates that
    the envelope is the load-bearing assumption of the entire tail bound. An
    envelope claiming ``B = 7`` for a kernel that decays at ``B = 1/2`` yields a
    "tail bound" 2.4e24 times too small, and every arithmetic step downstream of
    it is still exact. The arithmetic cannot save a false premise, so the
    premise gets its own flag and :func:`band_enclosure` refuses to call a
    record certified unless that flag is ``True``.
    """

    kind: str
    A: F
    valid_from: F
    justification: str
    B: Optional[F] = None
    p: Optional[F] = None
    certified: bool = False

    def __post_init__(self) -> None:
        if self.kind not in (GAUSSIAN, POWER):
            raise ValueError(f"unknown envelope kind {self.kind!r}")
        if not isinstance(self.A, F) or self.A <= 0:
            raise ValueError("envelope amplitude A must be a positive Fraction")
        if not isinstance(self.valid_from, F) or self.valid_from < 0:
            raise ValueError("valid_from must be a non-negative Fraction")
        if not self.justification.strip():
            raise ValueError(
                "a DecayEnvelope must carry a non-empty justification: an "
                "envelope constant with no proof behind it is not certified"
            )
        if self.kind == GAUSSIAN:
            if not isinstance(self.B, F) or self.B <= 0:
                raise ValueError("gaussian envelope needs a positive Fraction B")
            if self.p is not None:
                raise ValueError("gaussian envelope must not carry p")
        else:
            if not isinstance(self.p, F) or self.p <= 2:
                raise ValueError(
                    "power envelope needs a Fraction p > 2; at p <= 2 the "
                    "lattice sum need not converge and no tail bound exists"
                )
            if self.B is not None:
                raise ValueError("power envelope must not carry B")
        if not isinstance(self.certified, bool):
            raise TypeError("DecayEnvelope.certified must be a bool")

    def to_dict(self) -> Dict[str, object]:
        """A JSON-ready view, with the certification flag in it."""
        return {
            "kind": self.kind,
            "A": str(self.A),
            "B": None if self.B is None else str(self.B),
            "p": None if self.p is None else str(self.p),
            "valid_from": str(self.valid_from),
            "certified": self.certified,
            "justification": self.justification,
        }


@dataclass(frozen=True)
class PlaneKernel:
    """A pluggable plane kernel: a callable plus its certified decay envelope.

    ``evaluate(dx, dy, prec)`` takes two :class:`~research.interval.Interval`
    displacements and a precision hint and returns an ``Interval`` that
    **contains** ``kplane(dx', dy')`` for every ``dx' in dx`` and ``dy' in dy``.

    ``certified`` records whether ``evaluate`` really meets that contract, and
    **it covers the evaluator only**. It exists so that an uncertified
    evaluator (a float or ``mpmath`` stand-in) can be plugged in for
    exploration without silently acquiring the word "certified" downstream:
    :func:`band_enclosure` propagates the flag into its result and every
    consumer must look at it. A ``False`` here makes the whole output
    NON-CERTIFYING.

    IT DOES NOT COVER THE ENVELOPE. ``envelope.certified`` is a separate flag
    for a separate assumption, and :func:`band_enclosure` requires **both**
    before it will write ``certified: True`` on a record. A correct evaluator
    combined with a false envelope produces a tail bound that is wrong by any
    factor you like; see :class:`DecayEnvelope`.

    ``certified`` DEFAULTS TO ``False``. The safe default for a flag that means
    "this was checked" is the one that says it was not. Passing ``True`` is an
    assertion the caller makes and must be able to defend.
    """

    name: str
    evaluate: Callable[[Interval, Interval, int], Interval]
    envelope: DecayEnvelope
    certified: bool = False
    notes: str = ""

    @property
    def envelope_certified(self) -> bool:
        """The envelope's own flag, surfaced so a caller cannot read past it."""
        return bool(self.envelope.certified)

    @property
    def fully_certified(self) -> bool:
        """``True`` only when the evaluator AND the envelope are both flagged.

        This is the conjunction :func:`band_enclosure` uses. Neither half
        implies the other and neither alone is enough.
        """
        return bool(self.certified) and self.envelope_certified


# ---------------------------------------------------------------------------
# Reference kernels -- machinery exercisers, NOT the program's kernel
# ---------------------------------------------------------------------------

_GAUSSIAN_NOTE = (
    "REFERENCE KERNEL, NOT THE PROGRAM'S KERNEL. This is the unit separable "
    "Gaussian exp(-|z|^2/2), which is the frozen engine's C_KERN * kern(d2) at "
    "derivative multi-index (0, 0). The engine's kplane(a1, a2, dx, dy) is "
    "(-1)^(a1+a2) * He_{a1}(dx) * He_{a2}(dy) * exp(-|z|^2/2); the Hermite "
    "factors are NOT implemented here and no certified decay envelope for them "
    "is established here. Use this kernel to exercise and test the band "
    "machinery. Its numbers bear on OBL-H5-JETMOD in no way whatsoever."
)


def gaussian_reference() -> PlaneKernel:
    """The unit separable Gaussian ``exp(-(dx^2 + dy^2)/2)``.

    Envelope: ``|k(z)| = exp(-|z|^2/2)``, so ``A = 1``, ``B = 1/2`` and the
    envelope is an identity rather than an inequality, valid everywhere. That
    is why this is the honest reference: the envelope needs no argument beyond
    reading the definition, so a failure of the machinery cannot hide behind a
    doubtful envelope constant.

    The evaluator squares each displacement interval with ``Interval.__pow__``,
    which handles the non-monotone even power over an interval straddling zero
    correctly, adds them, negates, halves, and calls the certified ``exp``.
    """

    def evaluate(dx: Interval, dy: Interval, prec: int) -> Interval:
        d2 = dx ** 2 + dy ** 2
        return exp(d2 * F(-1, 2), prec)

    return PlaneKernel(
        name="gaussian-reference exp(-|z|^2/2)",
        evaluate=evaluate,
        envelope=DecayEnvelope(
            kind=GAUSSIAN,
            A=F(1),
            B=F(1, 2),
            valid_from=F(0),
            justification=(
                "|exp(-|z|^2/2)| = 1 * exp(-(1/2)|z|^2) for every z in R^2. "
                "The envelope holds with equality, so A = 1 and B = 1/2 are "
                "exact and valid_from = 0."
            ),
            # The justification above is a one-line identity, readable in full
            # from the kernel's own definition, so this flag is defensible.
            certified=True,
        ),
        certified=True,
        notes=_GAUSSIAN_NOTE,
    )


def inverse_power_reference(m: int) -> PlaneKernel:
    """``k(z) = (1 + |z|^2)^(-m)`` for an integer ``m >= 2``.

    Envelope: for ``z != 0``, ``1 + |z|^2 > |z|^2``, hence
    ``(1 + |z|^2)^(-m) < |z|^(-2m)``. So ``A = 1``, ``p = 2m >= 4 > 2``, and the
    envelope is valid for every ``|z| > 0``; ``valid_from`` is set to ``0``
    because :func:`tail_bound` independently requires the omitted points to sit
    at strictly positive distance.

    This kernel exists to exercise the POWER branch of the tail bound on a
    kernel whose evaluation is **exactly rational** -- no transcendental call,
    so the enclosure is the exact range up to the interval dependency problem
    and no ``prec`` enters. Same standing: a REFERENCE KERNEL, not the
    program's kernel.
    """
    if not isinstance(m, int) or isinstance(m, bool) or m < 2:
        raise ValueError("inverse_power_reference needs an integer m >= 2")

    def evaluate(dx: Interval, dy: Interval, prec: int) -> Interval:
        base = Interval.exact(F(1)) + dx ** 2 + dy ** 2
        return base ** (-m)

    return PlaneKernel(
        name=f"inverse-power-reference (1+|z|^2)^(-{m})",
        evaluate=evaluate,
        envelope=DecayEnvelope(
            kind=POWER,
            A=F(1),
            p=F(2 * m),
            valid_from=F(0),
            justification=(
                f"For z != 0, 1 + |z|^2 > |z|^2 > 0, so raising the strict "
                f"inequality to the negative power -{m} reverses it: "
                f"(1 + |z|^2)^(-{m}) < (|z|^2)^(-{m}) = |z|^(-{2 * m}). "
                f"Hence A = 1 and p = {2 * m} > 2."
            ),
            # A two-step inequality between elementary functions; flagged for
            # the same reason as the Gaussian reference envelope.
            certified=True,
        ),
        certified=True,
        notes=(
            "REFERENCE KERNEL, NOT THE PROGRAM'S KERNEL. Exactly rational; it "
            "exercises the power-law branch of the tail bound. It is not the "
            "side-24 rung kernel and bears on no obligation."
        ),
    )


# ---------------------------------------------------------------------------
# Displacement maps: band in r  ->  displacement box
# ---------------------------------------------------------------------------

def axial_displacement(r: Interval) -> Tuple[Interval, Interval]:
    """``(r, 0)`` -- the pin pair separated along the x axis.

    PLACEHOLDER, SAY SO WHEN YOU USE IT. The program's own map from a pin
    separation ``r`` to the covariance displacements of its six-pin
    configuration is not bound in this repository. This map is a defensible
    stand-in for exercising the machinery and nothing more; swapping in the
    real geometry is one of the three bindings named in ``README.md`` that
    would have to happen before any output here touched ``OBL-H5-JETMOD``.
    """
    return r, Interval.exact(F(0))


def diagonal_displacement(r: Interval) -> Tuple[Interval, Interval]:
    """``(r, r)`` -- a second stand-in, same standing as :func:`axial_displacement`."""
    return r, r


# ---------------------------------------------------------------------------
# Small certified helpers
# ---------------------------------------------------------------------------

def _round_bits(prec: int) -> int:
    """Significand budget used to keep endpoint integers from growing without bound.

    ``Interval.round_out`` rounds outward, so applying it can only widen an
    enclosure: containment is preserved and tightness is spent. It is never
    required for correctness here; it exists because a sum of certified ``exp``
    endpoints accumulates exact rationals with thousands of bits, which makes
    the records unreadable and the comparisons slow.

    The budget is generous relative to the width hint ``prec`` (which is a
    decimal tolerance), so rounding never dominates the reported width. As the
    library documents, ``round_out`` bounds the *significand* and not the
    *scale*: an enclosure of ``exp(-1149)`` keeps its ~1660-bit endpoints
    whatever budget is passed, because that is the value's information content.
    """
    return 4 * max(int(prec), 1) + 64


def _pow_fraction(base: Interval, e: F, prec: int) -> Interval:
    """Certified ``base ** e`` for a positive interval ``base`` and rational ``e``.

    Integer exponents take the exact algebraic path ``Interval.__pow__``.
    Otherwise ``base ** e = exp(e * log(base))``, both factors certified, so the
    composite is certified.
    """
    if base.lo <= 0:
        raise ValueError(f"fractional power needs a strictly positive base, got {base!r}")
    if e.denominator == 1:
        return base ** int(e.numerator)
    return exp(log(base, prec) * e, prec)


def _box_radius(dx: Interval, dy: Interval, prec: int) -> F:
    """A certified upper bound ``R`` on ``|d|`` for every ``d`` in the box.

    ``|d|^2 = d_x^2 + d_y^2 <= mag(dx)^2 + mag(dy)^2`` because ``mag`` is the
    largest absolute value attained on each factor, so the square of the
    Euclidean norm is bounded corner-wise. The certified ``sqrt`` then gives an
    enclosure of the root and its upper endpoint is the ``R`` returned.
    """
    m2 = dx.mag() ** 2 + dy.mag() ** 2
    return sqrt(Interval.exact(m2), prec).hi


# ---------------------------------------------------------------------------
# The truncated sum
# ---------------------------------------------------------------------------

def truncated_sum(
    kernel: PlaneKernel,
    dx: Interval,
    dy: Interval,
    *,
    n_trunc: int = ENGINE_TRUNCATION,
    period: F = SIDE24_PERIOD,
    prec: int = 40,
) -> Interval:
    """Enclosure of ``sum_{|n|_inf <= n_trunc} kplane(d + L*n)`` over the box.

    The returned interval contains that finite sum for **every** ``d`` in the
    box ``dx x dy`` simultaneously. It does not contain the infinite sum (1):
    :func:`tail_bound` supplies what is missing and :func:`band_enclosure`
    combines them.

    Interval arithmetic is used term by term, so the result is an *outward*
    bound on the range rather than the exact range: the same ``dx`` occurs in
    every term (the classic dependency problem) and summing independent
    enclosures cannot see the correlation. That widens the result and never
    breaks containment, which is the trade this program wants.

    ``(2*n_trunc+1)^2`` kernel evaluations are made -- 9 at the default
    ``n_trunc = 1``, matching the frozen engine's nine-point ``kdcov``.
    """
    if not isinstance(n_trunc, int) or isinstance(n_trunc, bool) or n_trunc < 0:
        raise ValueError("n_trunc must be a non-negative int")
    if period <= 0:
        raise ValueError("period must be positive")
    L = F(period)
    total = Interval.exact(F(0))
    for i in range(-n_trunc, n_trunc + 1):
        for j in range(-n_trunc, n_trunc + 1):
            shifted_x = dx + Interval.exact(L * i)
            shifted_y = dy + Interval.exact(L * j)
            total = total + kernel.evaluate(shifted_x, shifted_y, prec)
    return total.round_out(_round_bits(prec))


# ---------------------------------------------------------------------------
# The tail bound -- the mathematical core
# ---------------------------------------------------------------------------

def tail_bound(
    envelope: DecayEnvelope,
    dx: Interval,
    dy: Interval,
    *,
    n_trunc: int = ENGINE_TRUNCATION,
    period: F = SIDE24_PERIOD,
    prec: int = 40,
) -> F:
    """A certified upper bound on the OMITTED lattice mass, UNIFORM over the box.

    Returns an exact ``Fraction`` ``T >= 0`` with

        sum_{n in Z^2, |n|_inf > n_trunc} |kplane(d + L*n)|  <=  T

    for **every** ``d`` in the box ``dx x dy``. Nothing in ``T`` depends on
    ``d`` except through the single radius ``R`` of the box. That uniformity is
    the content of ``OBL-H5-JETMOD``'s phrase "lattice-tail constants
    re-certified uniformly in the band", and it is the one thing the frozen
    engine's ``tail_bound`` does not have: that function hard-codes
    ``rho = m*_LT - 17``, a POINT separation.

    THE ENVELOPE IS A PREMISE, NOT A RESULT. Every line below is conditional on
    ``envelope`` actually dominating ``|kplane|``. This function does not and
    cannot check that: it never sees ``kplane``. Hand it a false envelope and
    it returns, exactly and in certified interval arithmetic, a bound on a
    kernel that is not yours. ``DecayEnvelope.certified`` is where a caller
    records that the justification has been read, and
    :func:`band_enclosure` refuses the word "certified" without it. The return
    value here is a bare ``Fraction`` and carries no flag, so a caller reading
    it directly must read ``envelope.certified`` too.

    ------------------------------------------------------------------ setup
    Let ``D = dx x dy``, let ``R`` be a certified upper bound on ``|d|`` for
    ``d in D`` (Euclidean norm; see ``_box_radius``), let ``L > 0`` be the
    period, ``N = n_trunc >= 0``, ``M = N + 1``, and

        a := L*M - R.

    The function REFUSES (raises ``ValueError``) unless ``a > 0`` and
    ``a >= envelope.valid_from``. Refusing is the right behaviour: with
    ``a <= 0`` the box is large enough that an omitted image can sit on top of
    the evaluation point and no decay argument applies at all.

    -------------------------------------------------------- common geometry
    STEP 1 (shell separation). For ``n in Z^2`` with ``|n|_inf = m``,
    ``|L*n| >= L*|n|_inf = L*m`` since the Euclidean norm dominates the sup
    norm. Hence for every ``d in D``, by the reverse triangle inequality,

        |d + L*n| >= |L*n| - |d| >= L*m - R,

    which for ``m >= M`` is ``>= a > 0``. **This is the uniformity**: the bound
    holds for all ``d in D`` at once.

    STEP 2 (shell counts). ``#{n : |n|_inf = m} = (2m+1)^2 - (2m-1)^2 = 8m``.

    ------------------------------------------------- GAUSSIAN envelope branch
    STEP 3. ``|k(z)| <= A*exp(-B*|z|^2)`` for ``|z| >= valid_from``, and
    ``t -> A*exp(-B*t^2)`` is decreasing for ``t >= 0``, so by STEP 1

        |k(d + L*n)| <= A*exp(-B*(L*m - R)^2)   whenever |n|_inf = m >= M.

    STEP 4 (summation with NO second truncation). Put ``m = M + k``, ``k >= 0``,
    so ``L*m - R = a + L*k``. For every integer ``k >= 0``,

        (a + L*k)^2 = a^2 + 2*a*L*k + L^2*k^2 >= a^2 + (2*a*L + L^2)*k,

    because ``L^2*k^2 >= L^2*k`` for integer ``k >= 0`` (equality at k = 0, 1).
    Therefore with ``q := exp(-B*(2*a*L + L^2))``, which satisfies
    ``0 < q < 1`` since ``B > 0`` and ``L > 0``,

        exp(-B*(a + L*k)^2) <= exp(-B*a^2) * q^k,

    and summing the two convergent geometric series
    ``sum_k q^k = 1/(1-q)`` and ``sum_k k*q^k = q/(1-q)^2``:

        sum_{|n|_inf > N} |k(d + L*n)|
            <= sum_{k>=0} 8*(M+k)*A*exp(-B*(a + L*k)^2)
            <= 8*A*exp(-B*a^2) * [ M/(1-q) + q/(1-q)^2 ].               (G)

    (G) is a CLOSED FORM. Contrast the frozen engine, whose ``tail_bound``
    sums ``for m in range(2, 8)`` and simply drops every ``m >= 8`` with no
    remainder term, so what it prints is formally a partial sum of the object
    it claims to bound.

    ---------------------------------------------------- POWER envelope branch
    STEP 3'. ``|k(z)| <= A*|z|^(-p)`` for ``|z| >= valid_from``, ``p > 2``. For
    ``m >= M``, ``R/(L*m) <= R/(L*M)``, so with ``c := 1 - R/(L*M)``, which lies
    in ``(0, 1]`` exactly when ``a > 0``,

        L*m - R = L*m*(1 - R/(L*m)) >= c*L*m > 0,

    and since ``t -> A*t^(-p)`` is decreasing on ``t > 0``,

        |k(d + L*n)| <= A*(c*L*m)^(-p)         whenever |n|_inf = m >= M.

    STEP 4' (integral comparison). ``t -> t^(1-p)`` is positive and decreasing
    on ``t > 0``, so ``m^(1-p) <= int_{m-1}^{m} t^(1-p) dt`` for each
    ``m >= M+1``; summing over ``m >= M+1`` telescopes the integrals into
    ``int_{M}^{inf} t^(1-p) dt = M^(2-p)/(p-2)``, finite because ``p > 2``.
    Adding back the ``m = M`` term,

        sum_{m >= M} m^(1-p) <= M^(1-p) + M^(2-p)/(p-2),

    hence

        sum_{|n|_inf > N} |k(d + L*n)|
            <= 8*A*(c*L)^(-p) * [ M^(1-p) + M^(2-p)/(p-2) ].            (P)

    ------------------------------------------------------------- arithmetic
    Every quantity above is evaluated in certified interval arithmetic and the
    UPPER endpoint of the final enclosure is returned as an exact ``Fraction``.
    Rounding therefore only ever loosens ``T``, never tightens it. This is not
    ``mpmath``, and no step is a floating-point comparison against a hard-coded
    threshold.
    """
    if not isinstance(n_trunc, int) or isinstance(n_trunc, bool) or n_trunc < 0:
        raise ValueError("n_trunc must be a non-negative int")
    L = F(period)
    if L <= 0:
        raise ValueError("period must be positive")
    M = n_trunc + 1
    R = _box_radius(dx, dy, prec)
    a = L * M - R
    if a <= 0:
        raise ValueError(
            "tail bound refused: the displacement box has radius "
            f"R = {float(R):.6g} >= L*(n_trunc+1) = {float(L * M):.6g}, so an "
            "omitted image can reach the evaluation point and no decay "
            "argument applies. Increase n_trunc or shrink the band."
        )
    if a < envelope.valid_from:
        raise ValueError(
            "tail bound refused: the closest omitted image sits at separation "
            f">= {float(a):.6g}, below the envelope's valid_from = "
            f"{float(envelope.valid_from):.6g}. The envelope is not claimed "
            "there, so using it would be an unproved extrapolation."
        )

    eight_A = Interval.exact(8 * envelope.A)

    if envelope.kind == GAUSSIAN:
        B = envelope.B
        head = exp(Interval.exact(-B * a * a), prec)                 # exp(-B a^2)
        q = exp(Interval.exact(-B * (2 * a * L + L * L)), prec)      # q
        if q.hi >= 1:
            # REACHABLE, and exercised by a test. The true q = exp(-B(2aL+L^2))
            # is < 1 for every B, a, L > 0, but a certified enclosure of it need
            # not prove that: for a tiny B*(2aL+L^2) the outward-rounded upper
            # endpoint can land at or above 1, and the two geometric series
            # 1/(1-q) and q/(1-q)^2 are then not bounded by anything this code
            # computes. Refusing is the only honest exit; it is not a claim
            # that q >= 1.
            raise ValueError(
                f"tail bound refused: the geometric ratio enclosure {q!r} does "
                "not certify q < 1; raise prec"
            )
        one_minus_q = Interval.exact(F(1)) - q
        series = (Interval.exact(F(M)) / one_minus_q
                  + q / (one_minus_q ** 2))
        return (eight_A * head * series).round_out(_round_bits(prec)).hi

    # POWER
    p = envelope.p
    c = Interval.exact(F(1)) - Interval.exact(R) / Interval.exact(L * M)
    if c.lo <= 0:
        # DEFENSIVE AND, AS THE CODE STANDS, UNREACHABLE -- said plainly rather
        # than covered by a test that cannot fire. ``c`` is built from two
        # exact rational points, so interval division is exact and
        # c.lo = 1 - R/(L*M) = a/(L*M), which the ``a > 0`` check above has
        # already forced positive. The guard stays because it would become
        # reachable the moment R or L*M arrived as a non-degenerate interval,
        # and a test in tests/test_bands.py pins the implication a > 0 => c.lo
        # > 0 so that a change making it inexact is caught there.
        raise ValueError(
            f"tail bound refused: the shrink factor enclosure {c!r} does not "
            "certify c > 0; raise prec"
        )
    cl_pow = _pow_fraction(c * Interval.exact(L), -p, prec)          # (cL)^(-p)
    m_int = Interval.exact(F(M))
    term_head = _pow_fraction(m_int, F(1) - p, prec)                 # M^(1-p)
    term_int = _pow_fraction(m_int, F(2) - p, prec) / Interval.exact(p - 2)
    return (eight_A * cl_pow * (term_head + term_int)).round_out(
        _round_bits(prec)).hi


# ---------------------------------------------------------------------------
# Band enclosure
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BandEnclosure:
    """The result of :func:`band_enclosure`: the enclosure plus its breakdown.

    ``total`` is the certified enclosure of the infinite periodized sum (1)
    **valid for every displacement in the box** ``dx x dy``. That much is
    established by this module and by nothing else.

    "HENCE FOR EVERY ``r`` IN THE BAND" IS A SEPARATE STEP AND IS NOT
    ESTABLISHED HERE. It holds exactly as far as the ``displacement`` map
    handed to :func:`band_enclosure` is the real one. With the stand-ins this
    package ships -- :func:`axial_displacement`, :func:`diagonal_displacement`,
    both documented PLACEHOLDER -- the implication runs through a geometry that
    is not the program's, so what the record states is: certified over this
    box, for the stand-in map from ``r`` to that box. Bind the program's six-pin
    geometry and the second half becomes a statement about the program; until
    then it does not.

    ``truncated`` and ``tail`` are its two pieces, reported separately because
    the whole point of this module is that the frozen engine reports only the
    first and asserts about the second.

    ``certified`` is the CONJUNCTION of two independent flags, both of which
    must be ``True``:

    * ``evaluator_certified`` -- the plugged-in ``PlaneKernel.evaluate`` meets
      its enclosure contract;
    * ``envelope_certified`` -- the ``DecayEnvelope`` handed to the tail bound
      has a justification a human has checked.

    A ``False`` in either makes the entire record NON-CERTIFYING and every
    consumer must say so. Splitting them is not pedantry: exact arithmetic on
    top of a false envelope is exactly as wrong as the envelope, and the
    evaluator flag says nothing about the envelope.
    """

    kernel_name: str
    certified: bool
    evaluator_certified: bool
    envelope_certified: bool
    r_lo: F
    r_hi: F
    dx: Interval
    dy: Interval
    period: F
    n_trunc: int
    prec: int
    terms: int
    truncated: Interval
    tail: F
    total: Interval
    envelope: Optional[DecayEnvelope] = None
    notes: str = ""
    caveats: Tuple[str, ...] = field(default_factory=tuple)

    def width(self) -> F:
        """Exact width of ``total``. This is the quantity the falsifier tests."""
        return self.total.width()

    def truncated_width(self) -> F:
        return self.truncated.width()

    def to_dict(self) -> Dict[str, object]:
        """A JSON-ready breakdown. Rationals render as ``"num/den"`` strings."""
        return {
            "kernel": self.kernel_name,
            "certified": self.certified,
            "evaluator_certified": self.evaluator_certified,
            "envelope_certified": self.envelope_certified,
            "envelope": self.envelope.to_dict() if self.envelope is not None else None,
            "band": {"r_lo": str(self.r_lo), "r_hi": str(self.r_hi)},
            "displacement_box": {
                "dx": [str(self.dx.lo), str(self.dx.hi)],
                "dy": [str(self.dy.lo), str(self.dy.hi)],
            },
            "period": str(self.period),
            "n_trunc": self.n_trunc,
            "terms_summed": self.terms,
            "prec": self.prec,
            "truncated_sum": [str(self.truncated.lo), str(self.truncated.hi)],
            "truncated_width": str(self.truncated_width()),
            "tail_bound": str(self.tail),
            "total": [str(self.total.lo), str(self.total.hi)],
            "total_width": str(self.width()),
            "notes": self.notes,
            "caveats": list(self.caveats),
        }


_BAND_CAVEATS = (
    "Does NOT discharge, reduce or reclassify OBL-H5-JETMOD, which stays OPEN.",
    "The kernel here is a REFERENCE KERNEL unless the program's own kplane has "
    "been bound, with a certified decay envelope for it.",
    "The DECAY ENVELOPE IS AN INPUT, NOT A CHECKED FACT. Nothing in this "
    "package verifies that |kplane(z)| really obeys the A, B (or A, p) handed "
    "to it; the tail bound is only as true as that premise. A false envelope "
    "produces an exact computation of a wrong number -- the package's own "
    "negative control exhibits one 2.4e24 times too small. Read "
    "envelope_certified, and read the envelope's justification.",
    "The displacement map from r to the covariance displacement box is a "
    "stand-in unless the program's six-pin geometry has been bound. Until it "
    "is, 'for every r in the band' is a statement about the stand-in geometry "
    "and not about the program's.",
    "This is not a jet. The 24-jet set and its powers p_J are not defined here.",
    "A certified enclosure of a reference kernel is machinery, not a result.",
)

_UNCERTIFIED_EVALUATOR_CAVEAT = (
    "NON-CERTIFYING: PlaneKernel.certified is False, so the evaluator is not "
    "claimed to meet its enclosure contract. Every number in this record is a "
    "computation, not a bound."
)

_UNCERTIFIED_ENVELOPE_CAVEAT = (
    "NON-CERTIFYING: DecayEnvelope.certified is False, so the decay constants "
    "behind the tail bound are an unchecked claim. The tail figure in this "
    "record bounds nothing until that claim is proved."
)


def band_enclosure(
    kernel: PlaneKernel,
    r_band: Interval,
    *,
    displacement: Callable[[Interval], Tuple[Interval, Interval]] = axial_displacement,
    n_trunc: int = ENGINE_TRUNCATION,
    period: F = SIDE24_PERIOD,
    prec: int = 40,
) -> BandEnclosure:
    """Certified enclosure of the periodized sum over a whole r-band.

    THE SHAPE OF THE SOURCE'S PROOF STEP, RUN ON A REFERENCE KERNEL. The
    source's step is *"evaluating those sums with r as an interval over the
    band yields G12-band enclosures ... a FINITE computation per band, never a
    fitted exponent"* -- and "those sums" are the PROGRAM'S G12 lattice sums,
    over the program's bands, with the program's kernel and the program's
    six-pin geometry. This function runs that shape: ``(2*n_trunc+1)^2``
    interval kernel evaluations plus one closed-form tail bound, finite, per
    band, with no exponent fitted and no sampling taken, on a kernel and a
    displacement map the caller supplies. With the reference kernels and the
    stand-in displacement maps this package ships, what comes out is the shape
    and not the step. Calling it "the source's proof step, executed" would
    claim the bindings named below, which do not exist here.

    ``total = [truncated.lo - tail, truncated.hi + tail]``: the truncated sum
    encloses the finite part over the whole box, and the omitted part is
    bounded in absolute value by ``tail`` uniformly over the same box, so their
    combination encloses (1) over the box. The uniformity over the *box* is
    genuine and is the thing this module adds; carrying it to "every r in the
    band" is the ``displacement`` map's job and the stand-ins are not the
    program's map.

    WHAT IT IS NOT. It is not a band bound for any jet of this program, because
    neither the program's kernel nor its jets nor its band endpoints are bound
    in this repository. See ``research/bands/README.md`` for the three bindings
    that would be required.

    ``certified`` on the returned record is ``kernel.certified and
    kernel.envelope.certified``. Both default to ``False``.
    """
    if r_band.lo <= 0:
        raise ValueError(
            f"band {r_band!r} is not a positive separation band; "
            "r must be strictly positive"
        )
    dx, dy = displacement(r_band)
    trunc = truncated_sum(
        kernel, dx, dy, n_trunc=n_trunc, period=period, prec=prec
    )
    tail = tail_bound(
        kernel.envelope, dx, dy, n_trunc=n_trunc, period=period, prec=prec
    )
    total = Interval(trunc.lo - tail, trunc.hi + tail)
    caveats = list(_BAND_CAVEATS)
    if not kernel.certified:
        caveats.insert(0, _UNCERTIFIED_EVALUATOR_CAVEAT)
    if not kernel.envelope_certified:
        caveats.insert(0, _UNCERTIFIED_ENVELOPE_CAVEAT)
    return BandEnclosure(
        kernel_name=kernel.name,
        certified=kernel.fully_certified,
        evaluator_certified=bool(kernel.certified),
        envelope_certified=kernel.envelope_certified,
        r_lo=r_band.lo,
        r_hi=r_band.hi,
        dx=dx,
        dy=dy,
        period=F(period),
        n_trunc=n_trunc,
        prec=prec,
        terms=(2 * n_trunc + 1) ** 2,
        truncated=trunc,
        tail=tail,
        total=total,
        envelope=kernel.envelope,
        notes=kernel.notes,
        caveats=tuple(caveats),
    )


@dataclass(frozen=True)
class NormalizedBandEnclosure:
    """``S(B)/r^power`` over a band, WITH its flags and caveats attached.

    This is the one quantity in this package whose *shape* matches the
    obligation's content line -- ``J(B)/r^{p_J} in a certified interval`` --
    and so the one most likely to be quoted out of context. It therefore does
    not travel as a bare :class:`~research.interval.Interval`. Every number
    this package hands out carries its caveats by construction, and this one
    is no exception: ``certified``, ``notes`` and ``caveats`` are copied from
    the underlying :class:`BandEnclosure` and mean exactly what they mean
    there.

    It unpacks as the pair ``(ratio, enclosure)`` for callers written against
    the older tuple return, so the companion breakdown stays reachable; but
    the first element of that pair is this object, not a bare ``Interval``,
    and it still carries the flags.
    """

    ratio: Interval
    power: int
    certified: bool
    evaluator_certified: bool
    envelope_certified: bool
    enclosure: "BandEnclosure"

    @property
    def lo(self) -> F:
        return self.ratio.lo

    @property
    def hi(self) -> F:
        return self.ratio.hi

    @property
    def notes(self) -> str:
        return self.enclosure.notes

    @property
    def caveats(self) -> Tuple[str, ...]:
        return self.enclosure.caveats

    def width(self) -> F:
        return self.ratio.width()

    def __iter__(self):
        """``ratio_record, enc = normalized_band_enclosure(...)`` still works."""
        yield self
        yield self.enclosure

    def __contains__(self, other) -> bool:
        """``x in record`` delegates to the ratio interval, as before.

        Defined explicitly so that ``__iter__`` (which exists only for the
        tuple-unpacking call site) cannot silently turn a containment question
        into an equality scan over two elements and answer ``False``.
        """
        return other in self.ratio

    def to_dict(self) -> Dict[str, object]:
        return {
            "ratio": [str(self.ratio.lo), str(self.ratio.hi)],
            "ratio_width": str(self.ratio.width()),
            "power": self.power,
            "certified": self.certified,
            "evaluator_certified": self.evaluator_certified,
            "envelope_certified": self.envelope_certified,
            "notes": self.notes,
            "caveats": list(self.caveats),
            "band_enclosure": self.enclosure.to_dict(),
        }


def normalized_band_enclosure(
    kernel: PlaneKernel,
    r_band: Interval,
    power: int,
    **kwargs,
) -> NormalizedBandEnclosure:
    """``S(B) / r^power`` over the band, as a flagged record.

    The obligation's content line is *"for each jet J and band B,
    J(B)/r^{p_J} in a certified interval"*. This is the shape of that
    normalisation for a pluggable kernel. It is NOT that statement: no jet
    ``J`` and no power ``p_J`` of this program is bound here, so ``power`` is
    whatever the caller passes.

    The division is a genuine interval division: ``r_band ** power`` is an
    exact rational interval bounded away from zero (``r_band.lo > 0`` is
    checked by :func:`band_enclosure`), and dividing the enclosure of ``S`` by
    the enclosure of ``r^power`` encloses the ratio over the band. It is an
    outward bound and generally not the exact range, since the numerator and
    the denominator move together with ``r`` and interval division cannot see
    that.

    Returns a :class:`NormalizedBandEnclosure` rather than a bare ``Interval``
    so that the ``certified`` flags, the REFERENCE-KERNEL note and
    ``_BAND_CAVEATS`` cannot be separated from the number. It unpacks as
    ``(record, enclosure)``.
    """
    if not isinstance(power, int) or isinstance(power, bool):
        raise TypeError("power must be a plain int")
    enc = band_enclosure(kernel, r_band, **kwargs)
    return NormalizedBandEnclosure(
        ratio=enc.total / (r_band ** power),
        power=power,
        certified=enc.certified,
        evaluator_certified=enc.evaluator_certified,
        envelope_certified=enc.envelope_certified,
        enclosure=enc,
    )
