"""Certified enclosures of elementary and Gaussian functions, exact endpoints.

Every function here takes an :class:`~research.interval.core.Interval` and a
precision hint ``prec`` and returns an ``Interval`` that **provably contains**
the exact range of the function over the input interval.

``prec`` is a *decimal-digits-of-target-width hint*. It is not a correctness
parameter. Every enclosure below carries an explicit, exactly-evaluated
remainder or truncation bound; shrinking ``prec`` makes the returned interval
wider and never makes it wrong, and enlarging it makes the interval narrower
and never makes it right that it was not already. Where a function cannot do
better than a coarse bound (``sin`` of an interval wider than a period, ``erf``
far out in the tail) it returns the coarse bound rather than a tighter number
it cannot justify.

EXACTLY WHAT ``prec`` TARGETS, PER FUNCTION. The hint is an **absolute** width
target for ``sqrt``, ``log``, ``sin``, ``cos``, ``pi``, ``erf`` and ``Phi``. It
is a **relative** one for ``exp``, ``normal_pdf``, ``erfc`` and ``normal_sf``:

* ``exp`` reduces to ``|u| <= 1/2``, where an absolute remainder target on
  ``exp(u) = O(1)`` is a relative one, and then squares ``k`` times; squaring
  preserves relative width and multiplies absolute width by ``exp(t)``. So
  ``exp(t, prec)`` has width of order ``exp(t) * 10**-prec``, which for
  ``t = 700`` at ``prec = 30`` is about ``3e+272``, not ``1e-30``. A caller who
  budgets an absolute width for a factor inside a sum must convert. This is a
  fact about the contract, not a defect in the enclosure: every interval
  returned does contain the true value.
* ``normal_pdf`` inherits that from ``exp``.
* ``erfc`` and ``normal_sf`` are relative in their far tail by construction:
  beyond ``ERF_CROSSOVER`` they use the Mills bracket, whose **relative** width
  is about ``3/(4 z**4)`` and which does **not** tighten with ``prec`` at all.
  See ``erfc``. ``erf`` and ``Phi`` inherit the same prec-insensitive regime on
  the side where they approach ``+-1`` / ``0`` / ``1``.

There are two hard resource caps, both documented where they live and both
fail-closed rather than fail-quiet: ``EXP_BIT_LIMIT`` (below) bounds the binary
exponent of every ``exp``-derived enclosure, and ``_sin_cos`` returns
``[-1, 1]`` for an input at least a full period wide.

NON-CERTIFYING comparisons. Nothing in this module calls ``math``, ``mpmath``
or ``numpy``. Float agreement is used in ``tests/test_interval.py`` only as a
cross-check oracle and is labelled NON-CERTIFYING there.

WHAT THIS MODULE DOES NOT ESTABLISH
-----------------------------------
* No named obligation is discharged, reduced, closed, promoted or reclassified
  by the existence of this code. ``OBL-H5-JETMOD``, ``OBL-H5-ZBAND``,
  ``OBL-H5-REMOTE-THRESHOLD``, ``OBL-D1-PROMOTE``, ``D3-LEMMA-RN-UNIF``,
  ``PERC-DECAY`` and ``OBL-B1-BRANCH`` stand exactly as ``docs/OPEN_PROBLEMS.md``
  records them.
* No quantity in the q0 / SIDE24 corpus is recomputed, re-certified or
  contradicted here. In particular no lattice sum, no jet, no band enclosure,
  no rung and no moment appears in this file.
* A certified enclosure of ``Phi`` or ``erf`` is not a certified enclosure of
  anything those functions are used inside. Composition is the caller's
  obligation, and dependency widening in naive composition is real.
* The 2D upper / lower tracks and the 3D lifetime track are not touched, not
  related and certainly not composed by anything here.

Standard-library only: ``fractions`` and ``typing``. Python 3.11.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Dict, Tuple

from .core import Interval, _exact_fraction_string

__all__ = [
    "sqrt", "exp", "log", "sin", "cos", "pi", "erf", "erfc", "Phi",
    "normal_pdf", "normal_sf", "EXP_BIT_LIMIT", "ERF_CROSSOVER", "MILLS_MIN",
]

_HALF = Fraction(1, 2)
_UNIT = Interval(-1, 1)

#: Width, in reduced-index units, past which ``_sin_cos`` gives up and returns
#: ``[-1, 1]``. Documented at its use site; believed unreachable, and pinned by
#: a test that measures the span directly.
_J_SPAN_GUARD = 32

#: A certified strict LOWER bound on ``log2(e) = 1.4426950408889634...``.
#: Used only to decide whether an ``exp`` result is past ``EXP_BIT_LIMIT``;
#: because it is a lower bound, the test it drives is conservative in the
#: direction that matters (see ``_exp_point``).
_LOG2E_LO = Fraction(14426950408, 10 ** 10)

#: Cap on the binary exponent of any enclosure produced by ``_exp_point``, in
#: bits. Raising it raises the cost ceiling; it never affects containment.
#:
#: WHY A CAP EXISTS. An endpoint here is an exact ``Fraction``. An enclosure of
#: ``exp(t)`` therefore carries a numerator or denominator of about
#: ``|t| * log2(e) = 1.4427 * |t|`` bits — that is the information content of
#: the value and ``Interval.round_out`` cannot remove it, since ``round_out``
#: bounds the significand and not the scale. Measured on the unfixed code:
#: ``exp(Interval.exact(-10**6), 20)`` gives 1,442,915-bit endpoints and
#: ``exp(Interval.exact(-10**7), 20)`` gives 14,427,176-bit endpoints, linear in
#: ``|t|`` exactly as predicted. Composed into the Gaussian API the growth is
#: quadratic in the argument, because ``erf`` evaluates ``exp(-z**2)`` and
#: ``normal_pdf`` evaluates ``exp(-x**2/2)``: at ``z = 10**5`` that is
#: ``exp(-5e9)``, about 7.2e9-bit (900 MB) endpoints with several live at once
#: in the squaring loop, and at ``z = 10**6`` about 90 GB. Those calls cannot
#: complete on any machine; unfixed, ``Phi(Interval(-10**5), 1)`` raised
#: ``MemoryError`` under a 512 MiB address-space cap after 14 s and was
#: OOM-killed without one. That is an unconditional failure, not slowness.
#:
#: WHAT THE CAP DOES. ``2**21 = 2097152`` bits is about 631,305 decimal digits,
#: reached at ``|t| = 1453635``, at ``|z| = 1205.6`` in ``erf``/``erfc`` and at
#: ``|x| = 1705.1`` in ``normal_pdf``/``Phi``/``normal_sf``. Below the cap
#: nothing changes. Past it, an underflowing ``exp`` degrades to the coarse but
#: honest ``[0, 2**-EXP_BIT_LIMIT]`` and an overflowing one raises
#: ``OverflowError`` rather than consuming memory without bound. Both are stated
#: in ``_exp_point``.
EXP_BIT_LIMIT = 1 << 21

_UNDERFLOW_CACHE: Dict[int, Fraction] = {}


def _two_pow_neg(bits: int) -> Fraction:
    """``2**-bits`` as an exact ``Fraction``, cached (the integer is large)."""
    hit = _UNDERFLOW_CACHE.get(bits)
    if hit is None:
        hit = Fraction(1, 1 << bits)
        _UNDERFLOW_CACHE[bits] = hit
    return hit


# --------------------------------------------------------------------------
# small exact helpers
# --------------------------------------------------------------------------

def _tol(prec: int) -> Fraction:
    """The width hint as an exact rational. ``prec`` below 1 is treated as 1."""
    return Fraction(1, 10 ** max(int(prec), 1))


def _floor_frac(x: Fraction) -> int:
    return x.numerator // x.denominator


def _ceil_frac(x: Fraction) -> int:
    return -((-x.numerator) // x.denominator)


def _round_frac(x: Fraction) -> int:
    """``floor(x + 1/2)``, exactly."""
    return (2 * x.numerator + x.denominator) // (2 * x.denominator)


def _clamp_unit(iv: Interval) -> Interval:
    """Intersect with ``[-1, 1]``.

    Sound for ``sin``, ``cos`` and ``erf`` because each is bounded by 1 in
    absolute value everywhere on the reals, so the intersection still contains
    the true value. An empty intersection would mean the enclosure had already
    lost containment, so it is raised, never swallowed.
    """
    out = iv.intersect(_UNIT)
    if out is None:
        raise ArithmeticError(
            f"enclosure {iv!r} is disjoint from [-1, 1]; containment was lost"
        )
    return out


def _isqrt_newton(n: int) -> int:
    """``floor(sqrt(n))`` for ``n >= 0`` by integer Newton iteration.

    Start from ``x0 = 2**ceil(bitlen(n)/2) >= sqrt(n)``. The Newton map
    ``x -> (x + n//x)//2`` is, on integers at or above ``floor(sqrt(n))``,
    non-increasing and stays at or above ``floor(sqrt(n))`` (AM-GM gives
    ``(x + n/x)/2 >= sqrt(n)``, and integer flooring cannot drop below
    ``floor(sqrt(n))``). So the iteration decreases until it stops, and it stops
    exactly at ``floor(sqrt(n))``. The caller must still check the two
    inequalities; see ``_sqrt_bounds``.
    """
    if n < 0:
        raise ValueError("negative radicand")
    if n < 2:
        return n
    x = 1 << ((n.bit_length() + 1) // 2)
    while True:
        y = (x + n // x) // 2
        if y >= x:
            return x
        x = y


# --------------------------------------------------------------------------
# sqrt
# --------------------------------------------------------------------------

def _sqrt_bounds(a: Fraction, scale: int) -> Tuple[Fraction, Fraction]:
    """Rationals ``lo <= sqrt(a) <= hi`` with the two inequalities verified.

    Write ``a = p/q`` in lowest terms with ``p >= 0`` and ``q > 0``. Then
    ``sqrt(a) = sqrt(p*q)/q``. With ``N = scale`` and
    ``s = floor(sqrt(p*q*N^2))`` we have ``s^2 <= p*q*N^2 < (s+1)^2``, hence

        (s/(qN))^2 = s^2/(q^2 N^2) <= p*q/(q^2)      = a
        ((s+1)/(qN))^2 = (s+1)^2/(q^2 N^2) >= p*q/(q^2) = a

    so ``lo = s/(qN)`` and ``hi = (s+1)/(qN)`` bracket ``sqrt(a)`` and the
    bracket has width at most ``1/N``.

    The two inequalities are then re-checked here with exact ``Fraction``
    comparisons. That check is the certificate: it does not trust the Newton
    iteration, the bit-length estimate, or this docstring. Do not remove it.
    """
    if a < 0:
        raise ValueError("sqrt of a negative rational")
    if a == 0:
        return Fraction(0), Fraction(0)
    p, q = a.numerator, a.denominator
    s = _isqrt_newton(p * q * scale * scale)
    lo = Fraction(s, q * scale)
    hi = Fraction(s + 1, q * scale)
    if not lo * lo <= a:
        raise ArithmeticError(
            f"sqrt certificate failed: {_exact_fraction_string(lo)}^2 > "
            f"{_exact_fraction_string(a)}"
        )
    if not hi * hi >= a:
        raise ArithmeticError(
            f"sqrt certificate failed: {_exact_fraction_string(hi)}^2 < "
            f"{_exact_fraction_string(a)}"
        )
    return lo, hi


def sqrt(x: Interval, prec: int) -> Interval:
    """Certified enclosure of ``sqrt`` over ``x``. Requires ``x.lo >= 0``.

    ``t -> sqrt(t)`` is increasing on ``[0, inf)``, so the range over
    ``[a, b]`` is ``[sqrt(a), sqrt(b)]``; the endpoints are enclosed outward by
    ``_sqrt_bounds``, whose certificate (``lo^2 <= a`` and ``hi^2 >= b``,
    checked exactly in ``Fraction``) is what makes this certified rather than
    merely accurate.
    """
    if x.lo < 0:
        raise ValueError(f"sqrt requires x.lo >= 0, got {x!r}")
    p = max(int(prec), 1)
    scale = 10 ** (p + 2)
    lo, _ = _sqrt_bounds(x.lo, scale)
    _, hi = _sqrt_bounds(x.hi, scale)
    out = Interval(lo, hi).round_out(4 * p + 96)
    # Re-verify the certificate on the endpoints actually returned. Rounding
    # outward lowers a non-negative lo (so lo^2 can only fall) and raises hi
    # (so hi^2 can only rise), hence both inequalities survive -- but the point
    # of a certificate is that it is checked, not argued. Do not remove.
    if not out.lo * out.lo <= x.lo:
        raise ArithmeticError(
            f"sqrt certificate failed: {_exact_fraction_string(out.lo)}^2 > "
            f"{_exact_fraction_string(x.lo)}"
        )
    if not out.hi * out.hi >= x.hi:
        raise ArithmeticError(
            f"sqrt certificate failed: {_exact_fraction_string(out.hi)}^2 < "
            f"{_exact_fraction_string(x.hi)}"
        )
    return out


# --------------------------------------------------------------------------
# exp
# --------------------------------------------------------------------------

def _exp_point(t: Fraction, prec: int) -> Interval:
    """Certified enclosure of ``exp(t)`` for an exact rational ``t``.

    *Argument reduction.* Choose the least ``k >= 0`` with ``|t| / 2^k <= 1/2``
    and set ``u = t / 2^k``. Then ``exp(t) = (exp(u))^(2^k)``, obtained by
    squaring ``k`` times.

    *Taylor remainder.* With ``S_n = sum_{j=0}^{n} u^j / j!``,

        |R_n| = |sum_{j>n} u^j/j!|
              <= sum_{i>=0} |u|^(n+1+i)/(n+1+i)!
              <= (|u|^(n+1)/(n+1)!) * sum_{i>=0} (|u|/(n+2))^i
              =  (|u|^(n+1)/(n+1)!) * 1/(1 - |u|/(n+2))

    using ``(n+1+i)! >= (n+1)! * (n+2)^i`` term by term, and the geometric sum
    converges because ``|u| <= 1/2 < n+2``. That bound is evaluated exactly in
    ``Fraction`` and added to both sides, so ``exp(u) in [S_n - R, S_n + R]``.

    *Squaring.* ``exp(u) > 0`` and ``y -> y^2`` is increasing on ``[0, inf)``,
    so squaring an enclosure whose lower endpoint is non-negative yields an
    enclosure of the square. ``Interval.__pow__`` handles the non-negative case
    by endpoints in order, so containment is preserved at each of the ``k``
    steps. The lower endpoint is clamped at ``0`` first, which is sound because
    ``exp`` is positive.

    ``round_out`` between squarings widens outward only; it trades tightness
    for endpoint size and cannot break containment.

    *What ``prec`` targets here.* A RELATIVE width, not an absolute one. The
    remainder target is absolute on ``exp(u)`` with ``|u| <= 1/2``, where
    ``exp(u)`` is of order 1; the ``k`` squarings then multiply the absolute
    width by ``exp(t)`` while leaving the relative width at about ``10**-prec``.
    So ``exp(Interval.exact(700), 30)`` has width about ``3.1e+272``. Every such
    interval still contains the true value — this is the contract, stated,
    not a containment defect.

    *The exponent cap.* ``exp(t) = 2**(t * log2 e)``, so an exact rational
    enclosure of it carries about ``1.4427 * |t|`` bits of scale, which
    ``round_out`` cannot remove (it bounds the significand, not the scale).
    Unbounded ``|t|`` therefore means unbounded memory, and through
    ``erf``/``Phi``/``normal_pdf`` — which evaluate ``exp(-z**2)`` and
    ``exp(-x**2/2)`` — unbounded **quadratically** in an ordinary-looking
    Gaussian argument. ``EXP_BIT_LIMIT`` bounds it, in the only two ways that
    keep the contract:

    * UNDERFLOW, ``t < 0`` with ``(-t) * L2 >= EXP_BIT_LIMIT`` where
      ``L2 = _LOG2E_LO`` is a strict lower bound on ``log2 e``. Then
      ``(-t) * log2(e) >= (-t) * L2 >= EXP_BIT_LIMIT``, so
      ``0 < exp(t) <= 2**-EXP_BIT_LIMIT`` and ``[0, 2**-EXP_BIT_LIMIT]``
      contains it. Coarse, cheap, and sound; it is the same policy ``_sin_cos``
      already applies when it returns ``[-1, 1]``. Note the consequence: past
      this point no positive lower bound on ``exp(t)`` is available, and any
      ``erfc`` / ``normal_sf`` lower bound built on it degrades to 0. That
      threshold is ``|z| > 1205.6`` and ``|x| > 1705.1`` respectively, and it is
      a limit of the representation, not of the argument: a positive ``Fraction``
      lower bound on ``exp(-5e9)`` would need 900 MB to write down.
    * OVERFLOW, ``t > 0`` with ``t * L2 >= EXP_BIT_LIMIT``. There is no coarse
      honest answer available: a finite upper bound on ``exp(t)`` must have
      about ``1.4427 * t`` bits, and for ``t = 10**12`` that is 180 GB. This
      raises ``OverflowError`` (an ``ArithmeticError``, so an
      ``except ArithmeticError`` around a certified computation catches it)
      naming the limit. Refusing is fail-closed; returning something smaller
      would be a wrong bound and returning nothing would be a crash with no
      explanation.

    Neither branch can affect containment: one returns a proved superset and the
    other returns nothing at all.
    """
    if t == 0:
        return Interval(1)
    if t < 0:
        if (-t) * _LOG2E_LO >= EXP_BIT_LIMIT:
            return Interval(Fraction(0), _two_pow_neg(EXP_BIT_LIMIT))
    elif t * _LOG2E_LO >= EXP_BIT_LIMIT:
        raise OverflowError(
            f"exp({_exact_fraction_string(t)}) exceeds EXP_BIT_LIMIT = "
            f"{EXP_BIT_LIMIT} bits of binary exponent: an exact rational upper "
            f"bound would need about {int(t * _LOG2E_LO)} bits to write down. "
            "Raise research.interval.transcendental.EXP_BIT_LIMIT if the "
            "memory is genuinely available, or rescale the computation. "
            "Refused rather than bounded wrongly."
        )
    k = 0
    u = t
    while abs(u) > _HALF:
        u = u / 2
        k += 1
    target = _tol(prec) / (2 ** k)
    au = abs(u)

    total = Fraction(0)
    term = Fraction(1)      # u^j / j!
    absterm = Fraction(1)   # |u|^j / j!
    j = 0
    while True:
        total += term
        nxt_abs = absterm * au / (j + 1)              # |u|^(j+1)/(j+1)!
        rem = nxt_abs / (1 - au / (j + 2))            # the bound derived above
        if rem <= target:
            break
        j += 1
        term = term * u / j
        absterm = nxt_abs

    lo = total - rem
    if lo < 0:
        lo = Fraction(0)                              # exp is positive
    enc = Interval(lo, total + rem)
    sig = 4 * max(int(prec), 1) + 96 + 2 * k
    for _ in range(k):
        enc = (enc ** 2).round_out(sig)
    return enc


def exp(x: Interval, prec: int) -> Interval:
    """Certified enclosure of ``exp`` over ``x``.

    ``exp`` is strictly increasing on the reals, so the range over ``[a, b]``
    is ``[exp(a), exp(b)]`` and the endpoints are enclosed independently by
    ``_exp_point``; the lower endpoint of the enclosure of ``exp(a)`` and the
    upper endpoint of the enclosure of ``exp(b)`` bracket the whole range.
    """
    return Interval(_exp_point(x.lo, prec).lo, _exp_point(x.hi, prec).hi)


# --------------------------------------------------------------------------
# log
# --------------------------------------------------------------------------

def _atanh_series(z: Fraction, target: Fraction) -> Interval:
    """Certified enclosure of ``atanh(z)`` for ``0 <= z < 1``.

    ``atanh(z) = sum_{j>=0} z^(2j+1)/(2j+1)`` has all terms non-negative, so
    every partial sum ``S_j`` is a lower bound. For the upper bound, since
    ``1/(2i+1) <= 1/(2j+3)`` for ``i >= j+1``,

        T_j = sum_{i>j} z^(2i+1)/(2i+1)
            <= (1/(2j+3)) * sum_{i>j} z^(2i+1)
            =  z^(2j+3) / ((2j+3) * (1 - z^2))

    so ``atanh(z) in [S_j, S_j + T_j]``. Both endpoints are exact rationals.
    """
    if not 0 <= z < 1:
        raise ValueError(
            f"atanh series needs 0 <= z < 1, got {_exact_fraction_string(z)}"
        )
    if z == 0:
        return Interval(0)
    zz = z * z
    one_minus = 1 - zz
    total = Fraction(0)
    term = z            # z^(2j+1)
    j = 0
    while True:
        total += term / (2 * j + 1)
        nxt = term * zz                                   # z^(2j+3)
        tail = nxt / ((2 * j + 3) * one_minus)
        if tail <= target:
            break
        j += 1
        term = nxt
    return Interval(total, total + tail)


_LOG2_CACHE: Dict[Fraction, Interval] = {}


def _log2_enc(target: Fraction) -> Interval:
    """Certified enclosure of ``log 2 = 2 * atanh(1/3)``.

    ``log(m) = 2 atanh((m-1)/(m+1))``; at ``m = 2`` that is ``2 atanh(1/3)``,
    and ``1/3 < 1`` so ``_atanh_series`` applies with its rigorous tail bound.
    """
    hit = _LOG2_CACHE.get(target)
    if hit is None:
        a = _atanh_series(Fraction(1, 3), target / 2)
        hit = Interval(2 * a.lo, 2 * a.hi)
        _LOG2_CACHE[target] = hit
    return hit


def _log_point(a: Fraction, prec: int) -> Interval:
    """Certified enclosure of ``log(a)`` for an exact rational ``a > 0``.

    *Reduction.* Write ``a = m * 2^k`` with ``m in [1, 2)`` and ``k`` an
    integer, obtained from the bit lengths of numerator and denominator plus at
    most one correction step. Then ``log(a) = log(m) + k * log 2``.

    *Series.* ``log(m) = 2 atanh(z)`` with ``z = (m-1)/(m+1) in [0, 1/3)``,
    enclosed by ``_atanh_series`` with the tail bound proved there.

    *Assembly.* ``k * log 2`` is formed in interval arithmetic from the
    certified ``log 2`` enclosure, so the sign of ``k`` is handled by the
    interval multiplication and never by an assumption.
    """
    if a <= 0:
        raise ValueError(
            "log requires a positive argument, got "
            f"{_exact_fraction_string(a)}"
        )
    k = a.numerator.bit_length() - a.denominator.bit_length()
    m = a / Fraction(2) ** k
    while m >= 2:
        m /= 2
        k += 1
    while m < 1:
        m *= 2
        k -= 1

    tgt = _tol(prec) / 8
    z = (m - 1) / (m + 1)
    at = _atanh_series(z, tgt / 2)
    log_m = Interval(2 * at.lo, 2 * at.hi)
    if k == 0:
        return log_m
    return log_m + Interval.exact(k) * _log2_enc(tgt / (2 * (abs(k) + 1)))


def log(x: Interval, prec: int) -> Interval:
    """Certified enclosure of ``log`` over ``x``. Requires ``x.lo > 0``.

    ``log`` is strictly increasing on ``(0, inf)``, so the range over ``[a, b]``
    is ``[log(a), log(b)]`` and the endpoints are enclosed independently.
    """
    if x.lo <= 0:
        raise ValueError(f"log requires x.lo > 0, got {x!r}")
    return Interval(_log_point(x.lo, prec).lo, _log_point(x.hi, prec).hi)


# --------------------------------------------------------------------------
# pi
# --------------------------------------------------------------------------

def _atan_recip(n: int, target: Fraction) -> Interval:
    """Certified enclosure of ``atan(1/n)`` for an integer ``n >= 2``.

    ``atan(1/n) = sum_{j>=0} (-1)^j / ((2j+1) n^(2j+1))``. The term magnitudes
    ``a_j = 1/((2j+1) n^(2j+1))`` are strictly decreasing (both factors grow),
    so this is an alternating series with decreasing terms and the truncation
    error after the ``j``-th term is bounded in absolute value by the first
    omitted term ``a_(j+1)``. That gives the enclosure
    ``[S_j - a_(j+1), S_j + a_(j+1)]`` with exact rational endpoints.
    """
    if n < 2:
        raise ValueError("atan(1/n) series requires n >= 2")
    nn = n * n
    total = Fraction(0)
    powv = Fraction(1, n)          # 1/n^(2j+1)
    j = 0
    while True:
        term = powv / (2 * j + 1)
        total = total + term if j % 2 == 0 else total - term
        nxt_pow = powv / nn        # 1/n^(2j+3)
        nxt = nxt_pow / (2 * j + 3)
        if nxt <= target:
            break
        j += 1
        powv = nxt_pow
    return Interval(total - nxt, total + nxt)


_PI_CACHE: Dict[int, Interval] = {}


def pi(prec: int) -> Interval:
    """Certified enclosure of ``pi`` via Machin's formula.

    ``pi/4 = 4 atan(1/5) - atan(1/239)`` (Machin, 1706). Each ``atan(1/n)`` is
    enclosed by ``_atan_recip`` with the alternating-series bound proved there,
    and the combination ``pi = 4*(4*A5 - A239)`` is formed in interval
    arithmetic, so the subtraction widens outward and cannot lose containment.
    """
    key = max(int(prec), 1)
    hit = _PI_CACHE.get(key)
    if hit is None:
        target = _tol(key) / 32
        a5 = _atan_recip(5, target)
        a239 = _atan_recip(239, target)
        hit = (Interval.exact(4) * (Interval.exact(4) * a5 - a239)
               ).round_out(4 * key + 96)
        _PI_CACHE[key] = hit
    return hit


# --------------------------------------------------------------------------
# sin / cos
# --------------------------------------------------------------------------

def _sin_cos_reduced(s: Interval, target: Fraction, sig: int
                     ) -> Tuple[Interval, Interval]:
    """Enclosures of ``sin`` and ``cos`` over a reduced interval, ``mag(s) <= 1``.

    Both Maclaurin series are alternating. For ``sin``, the term magnitudes
    ``a_k = |s|^(2k+1)/(2k+1)!`` satisfy ``a_(k+1)/a_k = s^2/((2k+2)(2k+3))
    <= 1/6 < 1`` whenever ``|s| <= 1``, so the terms are strictly decreasing
    from the very first one and **the truncation error is bounded by the first
    omitted term**. The same argument with ``a_k = |s|^(2k)/(2k)!`` and ratio
    ``s^2/((2k+1)(2k+2)) <= 1/2`` covers ``cos``.

    The partial sum is evaluated in interval arithmetic over the whole of ``s``,
    which encloses the polynomial's range on ``s``; adding the symmetric
    remainder interval then encloses the function's range. ``mag(s)`` is used
    in the remainder bound, which is the largest the first omitted term can be
    anywhere on ``s``.

    Intermediate powers are rounded outward between steps, which only widens.
    """
    m = s.mag()
    if m > 1:
        raise ArithmeticError(
            f"reduced argument {s!r} has magnitude "
            f"{_exact_fraction_string(m)} > 1; the "
            "alternating-series remainder bound used here is not justified"
        )

    s2 = s ** 2

    # sin: sum_k (-1)^k s^(2k+1)/(2k+1)!
    total = Interval(0)
    powi = s
    fact = 1
    k = 0
    while True:
        term = powi * Fraction(1, fact)
        total = total + term if k % 2 == 0 else total - term
        nfact = fact * (2 * k + 2) * (2 * k + 3)
        bound = m ** (2 * k + 3) / nfact
        if bound <= target:
            break
        k += 1
        powi = (powi * s2).round_out(sig)
        fact = nfact
    sin_enc = _clamp_unit((total + Interval(-bound, bound)).round_out(sig))

    # cos: sum_k (-1)^k s^(2k)/(2k)!
    total = Interval(0)
    powi = Interval(1)
    fact = 1
    k = 0
    while True:
        term = powi * Fraction(1, fact)
        total = total + term if k % 2 == 0 else total - term
        nfact = fact * (2 * k + 1) * (2 * k + 2)
        bound = m ** (2 * k + 2) / nfact
        if bound <= target:
            break
        k += 1
        powi = (powi * s2).round_out(sig)
        fact = nfact
    cos_enc = _clamp_unit((total + Interval(-bound, bound)).round_out(sig))

    return sin_enc, cos_enc


def _decimal_digits(n: int) -> int:
    """An upper bound on ``len(str(abs(n)))``, built without the string.

    ``|n| < 2**b`` with ``b = bit_length(|n|)`` and ``log10(2) < 0.30103``, so
    the digit count is at most ``floor(b * 30103 / 100000) + 1``. This must not
    be ``len(str(n))``: CPython 3.11 refuses ``int`` -> ``str`` above
    ``sys.get_int_max_str_digits()`` (default 4300), so ``sin`` and ``cos`` of an
    argument with 4300 or more digits used to raise ``ValueError`` *before any
    arithmetic happened* — in the step that decides how much certified ``pi`` the
    reduction needs. An upper bound is the safe direction: it can only ask for
    more precision than the exact digit count would.
    """
    b = abs(n).bit_length()
    if b == 0:
        return 1
    return b * 30103 // 100000 + 1


def _pi_for(x: Interval, prec: int) -> Interval:
    """``pi`` at enough extra digits to absorb the reduction of ``x``.

    Reducing an argument of size ``V`` subtracts about ``2V/pi`` copies of
    ``pi/2``, which multiplies the uncertainty in ``pi`` by that factor. So the
    enclosure of ``pi`` is taken with ``digits(V)`` extra decimal digits plus a
    fixed guard of 14. The inequality this has to support is written out in
    :func:`_sin_cos_point`, and the margin it leaves is about ``10**-15``
    against a requirement of ``1 - pi/4 = 0.2146``.
    """
    v = int(x.mag()) + 1
    return pi(max(int(prec), 1) + _decimal_digits(v) + 14)


def _sin_cos_point(v: Fraction, half_pi: Interval, target: Fraction, sig: int
                   ) -> Tuple[Interval, Interval]:
    """Enclosures of ``sin(v)`` and ``cos(v)`` for an exact rational ``v``.

    Choose the integer ``j`` nearest to ``v / (pi/2)`` — computed from the
    *enclosure*, so ``j`` is only ever a choice of quadrant and never a source
    of error — and set ``S = [v] - j * HP`` where ``HP`` is the certified
    ``pi/2`` enclosure. **``S`` is interval-valued**: it is widened by
    ``j * width(HP)``, which is exactly the uncertainty in ``pi`` propagated
    through the reduction. This is the step where a naive implementation, using
    a fixed rational for ``pi``, silently loses containment for large ``v``;
    ``tests/test_interval.py`` contains a negative control that exhibits that
    loss.

    THE REDUCTION INEQUALITY, WRITTEN OUT. Let ``w = width(HP)``, let
    ``rho = v/(pi/2)`` be the true ratio and ``q`` its enclosure, so
    ``rho in q``. ``j = round(mid(q))`` is the nearest integer to the MIDPOINT of
    ``q``, not to ``rho``, so all that follows is

        |rho - j| <= 1/2 + width(q)/2,
        |s| = |rho - j| * (pi/2) <= pi/4 + (pi/4) * width(q),

    which is weaker than the bare ``|s| <= pi/4`` an exact ``j`` would give.
    Since ``S = [v] - j*HP`` is an interval of width ``|j| w`` containing ``s``,

        mag(S) <= |s| + |j| w <= pi/4 + (pi/4) width(q) + |j| w,

    with ``width(q) = |v| w / (hp_lo * hp_hi) <= 1.1 |v| w`` (using
    ``hp_lo, hp_hi >= 1.57``) and ``|j| <= |rho| + 1/2 + width(q)/2 <=
    0.64 |v| + 1`` for any ``|v|`` of interest. So

        mag(S) <= pi/4 + (0.87 + 0.64) |v| w + w <= pi/4 + 1.6 |v| w + w.

    ``_pi_for`` takes ``pi`` at ``prec + digits(V) + 14`` decimal digits with
    ``V = |v| + 1``, and ``pi(P)`` has width below ``10**-P``, so
    ``w <= 10**-P / 2 <= 10**-(digits(V) + 14)/2 <= 10**-14 / (2 V)`` and
    ``|v| w <= 10**-14 / 2``. Hence ``mag(S) <= pi/4 + 10**-14 < 1``, against a
    requirement of ``pi/4 = 0.7854 < 1``: a margin of ``0.2146``, exceeded only
    if ``pi`` were certified about ``10**13`` times more loosely than it is.

    Note ``|j|``, not ``j``: for negative ``v`` the index is negative and the
    bound must be taken in absolute value.

    THE CONSTRUCTION IS FAIL-CLOSED INDEPENDENTLY OF THAT ARGUMENT.
    ``_sin_cos_reduced`` re-tests ``mag(s) <= 1`` on the interval it is actually
    given and raises ``ArithmeticError`` otherwise, so if the inequality above
    were ever violated the result would be a refusal, never a wrong enclosure.
    A reviewer checking this function has to check the inequality for TIGHTNESS;
    containment does not rest on it. Finally

        sin(j*(pi/2) + s) = sin s, cos s, -sin s, -cos s   for j = 0,1,2,3 mod 4
        cos(j*(pi/2) + s) = cos s, -sin s, -cos s, sin s   for j = 0,1,2,3 mod 4
    """
    q = Interval.exact(v) / half_pi
    j = _round_frac(q.mid())
    s = Interval.exact(v) - Interval.exact(j) * half_pi
    sin_s, cos_s = _sin_cos_reduced(s, target, sig)
    r = j % 4
    if r == 0:
        return sin_s, cos_s
    if r == 1:
        return cos_s, -sin_s
    if r == 2:
        return -sin_s, -cos_s
    return -cos_s, sin_s


def _sin_cos(x: Interval, prec: int, which: str) -> Interval:
    """Shared driver for ``sin`` and ``cos``; see their docstrings."""
    p = _pi_for(x, prec)
    if x.width() >= 2 * p.hi:
        # The interval is at least one full period wide, measured with an
        # UPPER bound on 2*pi so the conclusion is sound. The range is then
        # all of [-1, 1]: coarse, correct, and honest.
        return Interval(-1, 1)

    half_pi = p * _HALF
    target = _tol(prec) / 4
    sig = 4 * max(int(prec), 1) + 128

    lo_s, lo_c = _sin_cos_point(x.lo, half_pi, target, sig)
    hi_s, hi_c = _sin_cos_point(x.hi, half_pi, target, sig)
    out = (lo_s.hull(hi_s) if which == "sin" else lo_c.hull(hi_c))

    # Interior extrema. The critical points of both sin and cos are exactly the
    # integer multiples c_j = j*(pi/2); sin has its extrema at odd j and cos at
    # even j. c_j lies in x iff j lies in x/(pi/2). We test j against the
    # *enclosure* A = x / HP, which contains {t/(pi/2) : t in x}; that test can
    # only fire more often than the exact one, and firing spuriously merely adds
    # a +-1 to the hull, which widens. Missing one would break containment, so
    # the conservative direction is the safe one.
    a = x / half_pi
    j_lo, j_hi = _floor_frac(a.lo), _ceil_frac(a.hi)
    if j_hi - j_lo > _J_SPAN_GUARD:
        # DEFENSIVE, AND BELIEVED UNREACHABLE. The full-period test above has
        # already established width(x) < 2*pi_hi, so
        #   width(a) <= width(x)/hp_lo + mag(x)*width(HP)/hp_lo**2
        #            <= 2*pi_hi/hp_lo + (a tiny reduction term, see
        #               _sin_cos_point: mag(x)*w stays below 1e-14)
        #            <  4.1,
        # and j_hi - j_lo <= width(a) + 2 <= 7. Measured over 1,664
        # (interval, prec) pairs: worst span 5, against this guard at 32. So
        # this branch should never fire.
        #
        # It is kept anyway, and what it buys is stated precisely: not
        # containment -- returning [-1, 1] is sound for sin and cos at any
        # width, and so is the extrema loop, which adds every interior extremum
        # it finds -- but a BOUNDED LOOP. Without it, an input that reached
        # here a full period wide would iterate width/(pi/2) times. Weakening
        # the full-period test above is in fact an equivalent mutation for
        # containment (verified: the whole suite still passes), which is
        # exactly why this line must stay and why the span is pinned directly
        # by ``test_reduced_index_span_stays_small`` instead of being left to
        # this branch to catch.
        return Interval(-1, 1)
    for j in range(j_lo, j_hi + 1):
        if not a.lo <= j <= a.hi:
            continue
        if which == "sin" and j % 2 == 1:
            out = out.hull(Interval(1) if j % 4 == 1 else Interval(-1))
        elif which == "cos" and j % 2 == 0:
            out = out.hull(Interval(1) if j % 4 == 0 else Interval(-1))
    return _clamp_unit(out.round_out(sig))


def sin(x: Interval, prec: int) -> Interval:
    """Certified enclosure of ``sin`` over ``x``.

    Argument reduction into ``[-pi/4, pi/4]`` uses the certified ``pi``
    enclosure and is itself interval-valued, so the reduced interval carries the
    uncertainty in ``pi``. The reduced evaluation uses the **alternating**
    Maclaurin series, whose remainder is bounded by the first omitted term
    (justified in ``_sin_cos_reduced``). Interior extrema at the odd multiples
    of ``pi/2`` are added to the hull whenever the certified enclosure of such a
    multiple meets ``x``. If ``x`` is at least one full period wide, ``[-1, 1]``
    is returned: correct and honest, not a tightness failure to be hidden.
    """
    return _sin_cos(x, prec, "sin")


def cos(x: Interval, prec: int) -> Interval:
    """Certified enclosure of ``cos`` over ``x``.

    Same construction as :func:`sin`; the interior extrema of ``cos`` sit at the
    *even* multiples of ``pi/2``, i.e. at the integer multiples of ``pi``.
    """
    return _sin_cos(x, prec, "cos")


# --------------------------------------------------------------------------
# erf, Phi, normal_pdf
# --------------------------------------------------------------------------

#: Crossover between the Maclaurin branch and the tail-bracket branch of
#: :func:`erf` and :func:`erfc`. Below it the alternating series is used; at and
#: above it only the Mills bracket is used. This is a COST boundary, not an
#: accuracy one: the series stays exact above it but needs about ``z**2`` terms
#: carrying intermediate rationals of size ``e**(z**2)``. Its consequences for
#: achievable width are spelled out in :func:`erfc`.
ERF_CROSSOVER = Fraction(6)

#: Smallest ``z`` at which the Mills bracket of :func:`_erfc_mills` is proved.
#: The lower half holds for every ``z > 0``; the upper half needs ``z >= 1``.
MILLS_MIN = Fraction(1)


def _erfc_mills(z: Fraction, prec: int) -> Interval:
    """Certified TWO-SIDED enclosure of ``erfc(z) = 1 - erf(z)`` for ``z >= 1``.

    THE LEMMA, with its proof. For real ``b`` with ``2z**2 + b > 0`` put

        E(z) = int_z^inf e^(-t^2) dt,   c_b(z) = z e^(-z^2) / (2 z^2 + b),
        g_b  = E - c_b.

    Differentiating ``c_b`` and using ``E'(z) = -e^(-z^2)``,

        c_b'(z) = e^(-z^2) [ (b - 2z^2) - 2z^2(2z^2 + b) ] / (2z^2 + b)^2
        g_b'(z) = -e^(-z^2) [ (2z^2 + b)^2 + b - 2z^2 - 4z^4 - 2b z^2 ]
                            / (2z^2 + b)^2
                = -e^(-z^2) [ 2z^2 (b - 1) + b(b + 1) ] / (2z^2 + b)^2

    since ``(2z^2+b)^2 = 4z^4 + 4b z^2 + b^2``. Both ``E`` and ``c_b`` tend to 0
    at ``+inf``, so ``g_b(t) -> 0``. Hence, writing
    ``P_b(t) = 2t^2(b-1) + b(b+1)``:

      (i)  if ``P_b(t) > 0`` for all ``t >= z`` then ``g_b`` is strictly
           decreasing on ``[z, inf)``, so ``g_b(z) > 0`` and ``E(z) > c_b(z)``;
      (ii) if ``P_b(t) <= 0`` for all ``t >= z`` then ``g_b`` is non-decreasing,
           so ``g_b(z) <= 0`` and ``E(z) <= c_b(z)``.

    LOWER BOUND, ``b = 1``. ``P_1(t) = 2 > 0`` for every ``t``, so (i) gives
    ``E(z) > z e^(-z^2)/(2z^2+1)`` for every ``z > 0``.

    UPPER BOUND, ``b = 1 - 3/(2 z^2)`` with ``z >= 1``. Write ``d = 3/(2z^2)``,
    so ``0 < d <= 3/2``. ``P_b`` is non-increasing in ``t`` because ``b < 1``, so
    it is enough to check ``t = z``:

        P_b(z) = -2 z^2 d + (1 - d)(2 - d) = -3 + 2 - 3d + d^2 = d^2 - 3d - 1,

    which is ``<= 0`` for ``0 <= d <= 3``, hence for every ``z >= 1``. Also
    ``2z^2 + b = 2z^2 + 1 - d >= 2 + 1 - 3/2 > 0``, so (ii) applies and
    ``E(z) <= z e^(-z^2) / (2z^2 + 1 - 3/(2z^2))``, i.e. after clearing the
    inner fraction ``E(z) <= 2 z^3 e^(-z^2) / (4z^4 + 2z^2 - 3)``.

    Multiplying by ``2/sqrt(pi)``, for every ``z >= 1``

        2 z e^(-z^2) / (sqrt(pi) (2 z^2 + 1))
            <  erfc(z)  <=
        4 z^3 e^(-z^2) / (sqrt(pi) (4 z^4 + 2 z^2 - 3)).

    The ratio of the two is ``1 + (3/(2z^2)) / (2z^2 + 1 - 3/(2z^2))``, about
    ``1 + 3/(4 z^4)``: a RELATIVE width of about ``3/(4 z^4)``, which shrinks
    fast in ``z`` and does NOT depend on ``prec``.

    IMPLEMENTATION. ``e^(-z^2)`` comes from the certified :func:`exp`,
    ``sqrt(pi)`` from the certified :func:`sqrt` and :func:`pi`, and the two
    quotients are formed in interval arithmetic; the returned endpoints are the
    ``.lo`` of the lower quotient and the ``.hi`` of the upper one, each rounded
    outward, so both inequalities survive the arithmetic.

    WHAT THIS DOES NOT GIVE. Past ``EXP_BIT_LIMIT`` — ``|z| > 1205.6`` — the
    ``exp`` enclosure is ``[0, 2**-EXP_BIT_LIMIT]`` and the lower endpoint here
    collapses to exactly 0. The enclosure stays correct and the upper bound
    stays strong, but there is then no positive certified lower bound on the
    tail mass, for the representation reason given in ``_exp_point``.
    """
    if z < MILLS_MIN:
        raise ValueError(
            f"Mills bracket requires z >= {MILLS_MIN}, got "
            f"{_exact_fraction_string(z)}"
        )
    g = max(int(prec), 1) + 20
    sig = 4 * g + 32
    zz = z * z
    # _exp_point, not exp: the argument is a point, and ``exp`` would evaluate
    # the same point twice. It is never positive here, so the OverflowError
    # branch of _exp_point is unreachable from this call.
    e = _exp_point(-zz, g)
    sp = sqrt(pi(g), g)
    lower = ((Interval.exact(2 * z) * e)
             / (sp * Interval.exact(2 * zz + 1))).round_out(sig).lo
    upper = ((Interval.exact(4 * zz * z) * e)
             / (sp * Interval.exact(4 * zz * zz + 2 * zz - 3))).round_out(sig).hi
    return Interval(lower, upper)


def _erf_series(z: Fraction, prec: int) -> Interval:
    """Certified enclosure of ``erf(z)`` by its Maclaurin series, ``0 <= z``.

    ``erf(z) = (2/sqrt(pi)) * sum_n (-1)^n t_n`` with
    ``t_n = z^(2n+1)/(n!(2n+1)) > 0``. The ratio
    ``t_(n+1)/t_n = z^2 (2n+1)/((n+1)(2n+3))`` is **not** below 1 for small
    ``n`` when ``z`` is not small, so the alternating bound is not available
    from the start. It becomes available once ``n >= z^2``: then
    ``z^2 (2n+1) <= n(2n+1) = 2n^2 + n <= (n+1)(2n+3) = 2n^2 + 5n + 3``.
    So truncation is only permitted at an index ``n >= floor(z^2) + 1``, which
    this function enforces, and from there the terms decrease and the error is
    bounded by the first omitted term ``t_(n+1)``. Exact rational arithmetic
    means the large intermediate cancellation for ``z`` near 6 costs size, not
    accuracy.

    The error controlled here is ABSOLUTE. Near ``z = 6``, ``erf(z)`` is within
    ``2.2e-17`` of 1, so at ``prec`` below about 17 this branch alone cannot
    show ``erf(z) < 1``; that is what the Mills bracket is intersected in for
    (see :func:`_erfc_point`). Cost grows with ``z``: about ``z**2`` terms
    carrying intermediates of size ``e^(z^2)``, which is why
    :data:`ERF_CROSSOVER` exists.
    """
    g = max(int(prec), 1) + 20
    two_over_sqrt_pi = Interval.exact(2) / sqrt(pi(g), g)
    target = _tol(prec) / 4
    n_min = _floor_frac(z * z) + 1
    zz = z * z
    total = Fraction(0)
    t = z                    # t_0 = z
    n = 0
    while True:
        total = total + t if n % 2 == 0 else total - t
        t_next = t * zz * (2 * n + 1) / ((n + 1) * (2 * n + 3))
        if n >= n_min and t_next <= target:
            break
        n += 1
        t = t_next
    series = Interval(total - t_next, total + t_next)
    return _clamp_unit((series * two_over_sqrt_pi).round_out(4 * g + 32))


def _erfc_point(z: Fraction, prec: int) -> Interval:
    """Certified enclosure of ``erfc(z) = 1 - erf(z)`` for an exact rational.

    Three regimes, and one consistency check:

    * ``z <= 0``: ``erfc(z) = 2 - erfc(-z)``, exact in ``Fraction``.
    * ``0 < z <= ERF_CROSSOVER``: ``1 - S`` where ``S`` is the series enclosure
      of :func:`_erf_series`. For ``z >= MILLS_MIN`` this is INTERSECTED with
      the Mills bracket of :func:`_erfc_mills`. Both are certified enclosures of
      the same number, so the intersection is one too, and it is the tighter of
      the two everywhere: the series controls absolute error (good for small
      ``z``) and Mills controls relative error (good for large ``z``). An empty
      intersection is impossible unless containment has already been lost in one
      of them, so it raises rather than being swallowed — this is a live
      cross-check between two independently derived bounds.
    * ``z > ERF_CROSSOVER``: the Mills bracket alone. The series is still exact
      there but its cost is not bounded; see :data:`ERF_CROSSOVER`.

    The intersection removes what would otherwise be a discontinuity in
    achievable width at the crossover, and it removes the collapse of the lower
    bound at modest ``prec`` below it. What it does NOT remove: above the
    crossover the width is the Mills relative width ``~3/(4 z^4)`` and ``prec``
    has no effect on it at all.
    """
    if z == 0:
        return Interval(1)
    if z < 0:
        return Interval(2) - _erfc_point(-z, prec)
    if z > ERF_CROSSOVER:
        return _erfc_mills(z, prec)
    series = Interval(1) - _erf_series(z, prec)
    if z < MILLS_MIN:
        return series
    mills = _erfc_mills(z, prec)
    out = series.intersect(mills)
    if out is None:
        raise ArithmeticError(
            f"erfc brackets are disjoint at z={_exact_fraction_string(z)}: "
            f"series {series!r} vs Mills {mills!r}; containment was lost"
        )
    return out


def _erf_point(z: Fraction, prec: int) -> Interval:
    """Certified enclosure of ``erf(z)`` for an exact rational ``z``.

    ``erf(z) = 1 - erfc(z)`` and the subtraction is exact in ``Fraction``, so
    this loses nothing relative to :func:`_erfc_point` and inherits its
    certificates. ``erf`` is odd, so negative arguments are reflected; the
    result is clamped into ``[-1, 1]``, which is sound because ``|erf| < 1``.

    FOR A TAIL BOUND, CALL :func:`erfc` OR :func:`normal_sf` INSTEAD. ``erf(z)``
    for large ``z`` is a number just below 1 and its endpoints carry about
    ``1.4427 z^2`` bits for that reason; the tail mass itself is what ``erfc``
    returns directly, at bounded size and with the relative accuracy above.
    """
    if z < 0:
        return -_erf_point(-z, prec)
    if z == 0:
        return Interval(0)
    return _clamp_unit(Interval(1) - _erfc_point(z, prec))


def erf(x: Interval, prec: int) -> Interval:
    """Certified enclosure of ``erf`` over ``x``.

    ``erf`` is strictly increasing on the reals (its derivative
    ``2 e^(-t^2)/sqrt(pi)`` is positive everywhere), so the range over
    ``[a, b]`` is ``[erf(a), erf(b)]``: the LOWER endpoint of the enclosure at
    ``x.lo`` and the UPPER endpoint of the enclosure at ``x.hi``. Both endpoints
    of the input are used; a regression test pins the direction.

    TIGHTNESS, STATED. For ``|z| <= ERF_CROSSOVER = 6`` the width follows
    ``prec``. Above that the enclosure comes from the Mills bracket of
    :func:`_erfc_mills`, whose width is about ``3 erfc(z) / (4 z^4)`` and is
    INSENSITIVE TO ``prec``: raising ``prec`` will not meet a width target out
    there. In the ``Phi`` coordinate that regime begins at
    ``|x| = 6*sqrt(2) = 8.4853``. Past ``|z| = 1205.6`` the ``exp`` exponent cap
    makes the lower endpoint of the tail exactly 0 (equivalently
    ``erf(z).hi == 1``); see ``EXP_BIT_LIMIT``.
    """
    return Interval(_erf_point(x.lo, prec).lo, _erf_point(x.hi, prec).hi)


def erfc(x: Interval, prec: int) -> Interval:
    """Certified enclosure of ``erfc = 1 - erf`` over ``x``.

    ``erfc`` is strictly DECREASING, so the range over ``[a, b]`` is
    ``[erfc(b), erfc(a)]``: the lower endpoint comes from ``x.hi`` and the upper
    from ``x.lo``. A regression test pins that direction, because getting it
    backwards produces an interval that looks perfectly reasonable.

    WHY THIS EXISTS SEPARATELY FROM ``erf``. ``1 - erf(x)`` and ``1 - Phi(x)``
    are computed in exact ``Fraction`` arithmetic and so lose nothing *at the
    subtraction*, but the enclosures they subtract are rounded to a fixed number
    of significant bits, and for ``x`` beyond about 26 an endpoint of ``Phi``
    rounds up to exactly 1 — after which the difference is ``[0, something]``
    and the tail bound is gone. ``erfc`` and :func:`normal_sf` carry the tail
    mass as the primary quantity and never subtract it from 1, so they keep
    relative accuracy out to the ``EXP_BIT_LIMIT`` floor.

    TIGHTNESS, STATED. Below ``ERF_CROSSOVER`` the width follows ``prec``
    (absolute) intersected with the Mills bracket (relative). At and above it
    the Mills bracket alone applies, relative width about ``3/(4 z^4)``, with NO
    dependence on ``prec``. Past ``|z| = 1205.6`` the lower endpoint is exactly
    0 and only the upper bound carries information.

    This is an enclosure of ``erfc`` and of nothing else. It is not a bound on
    any quantity in the q0 / SIDE24 corpus and it discharges no obligation.
    """
    return Interval(_erfc_point(x.hi, prec).lo, _erfc_point(x.lo, prec).hi)


def Phi(x: Interval, prec: int) -> Interval:
    """Certified enclosure of the standard normal CDF over ``x``.

    ``Phi(x) = (1 + erf(x/sqrt 2)) / 2``. ``sqrt 2`` is the certified enclosure
    from :func:`sqrt`, the division is interval division by an interval that
    does not contain zero, and ``erf`` is monotone, so the composition is an
    enclosure of the range. ``Phi(0) = 1/2`` comes out exactly, because
    ``[0,0] / [s_lo, s_hi] = [0,0]`` and ``erf([0,0]) = [0,0]``.

    TIGHTNESS, STATED — read this before choosing ``prec``. ``Phi`` inherits the
    regime change of :func:`erf` at ``z = ERF_CROSSOVER``, which in THIS
    function's coordinate sits at ``|x| = 6 sqrt 2 = 8.485281...``. Below it the
    width follows ``prec``. Above it the width comes from the Mills bracket and
    is about ``3 Phi(-|x|) / (x^4)``, with NO dependence on ``prec``: raising
    ``prec`` there will not meet a width target, and nothing else signals that.
    Past ``|x| = 1705.1`` the ``exp`` exponent cap leaves ``Phi(x).lo == 0`` for
    negative ``x`` (see ``EXP_BIT_LIMIT``).

    FOR AN UPPER TAIL USE :func:`normal_sf`, NOT ``1 - Phi(x)``. The ``round_out``
    below keeps a fixed number of significant bits, so for ``x`` beyond about 26
    the upper endpoint here rounds up to exactly 1 and ``1 - Phi(x)`` is then
    ``[0, ...]`` — sound, and useless as a tail bound. ``normal_sf`` carries the
    tail mass as the primary quantity instead.
    """
    g = max(int(prec), 1) + 20
    arg = x / sqrt(Interval.exact(2), g)
    return ((Interval(1) + erf(arg, prec)) * _HALF).round_out(4 * g + 32)


def normal_sf(x: Interval, prec: int) -> Interval:
    """Certified enclosure of the standard normal survival function over ``x``.

    ``normal_sf(x) = P(X > x) = 1 - Phi(x) = erfc(x / sqrt 2) / 2``, computed
    through :func:`erfc` so the tail mass is never obtained by subtracting a
    number near 1 from 1. ``normal_sf`` is strictly DECREASING; the direction is
    carried by ``erfc`` and pinned by a regression test.

    This is the function to call for a two-sided bound on a Gaussian tail
    probability, and it is the reason :func:`erfc` exists. Worked example, the
    9-sigma tail::

        normal_sf(Interval.exact(9), 30)   # both endpoints positive
        log(normal_sf(Interval.exact(9), 30), 30)   # a certified log-tail bound

    Before the Mills bracket was added, the corresponding lower bound was
    exactly 0 at every ``prec`` and ``log`` of it raised ``ValueError``.

    TIGHTNESS, STATED. Relative width about ``3/(x^4)`` beyond
    ``|x| = 8.4853``, with NO dependence on ``prec`` (see :func:`erfc`). Past
    ``|x| = 1705.1`` the lower endpoint is exactly 0 and only the upper bound
    carries information; that is the ``EXP_BIT_LIMIT`` floor, not a property of
    the normal distribution.

    WHAT THIS IS NOT. It is an enclosure of a standard normal tail probability
    and nothing else. It is not a bound on ``1 - q(r)``, not a band enclosure,
    and it discharges no obligation in ``docs/OPEN_PROBLEMS.md``.
    """
    g = max(int(prec), 1) + 20
    arg = x / sqrt(Interval.exact(2), g)
    return (erfc(arg, prec) * _HALF).round_out(4 * g + 32)


def normal_pdf(x: Interval, prec: int) -> Interval:
    """Certified enclosure of the standard normal density over ``x``.

    ``normal_pdf(x) = exp(-x^2/2) / sqrt(2 pi)``.

    This function is **not monotone**: it rises on ``(-inf, 0]`` and falls on
    ``[0, inf)``, with its maximum at ``x = 0``. The interior extremum is
    handled where it belongs, inside ``Interval.__pow__``: for an interval
    straddling zero, ``x ** 2`` is ``[0, mag(x)^2]``, whose lower endpoint is
    the interior minimum of the square. Because ``exp`` is increasing, that
    lower endpoint propagates to the density's maximum ``1/sqrt(2 pi)``.
    Evaluating ``x**2`` at the endpoints alone would omit it and the enclosure
    would miss the peak; ``tests/test_interval.py`` has a negative control that
    demonstrates exactly that failure.

    ``prec`` is a RELATIVE width hint here, inherited from :func:`exp`; see the
    module docstring. Past ``|x| = 1705.1`` the ``exp`` exponent cap makes the
    enclosure ``[0, tiny]`` — coarse, sound, and cheap, where before the cap the
    same call consumed memory without bound (``EXP_BIT_LIMIT``).
    """
    g = max(int(prec), 1) + 20
    numerator = exp(-(x ** 2) * _HALF, prec)
    denominator = sqrt(Interval.exact(2) * pi(g), g)
    return (numerator / denominator).round_out(4 * g + 32)
