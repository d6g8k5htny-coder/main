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

from .core import Interval

__all__ = [
    "sqrt", "exp", "log", "sin", "cos", "pi", "erf", "Phi", "normal_pdf",
]

_HALF = Fraction(1, 2)
_UNIT = Interval(-1, 1)


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
        raise ArithmeticError(f"sqrt certificate failed: {lo}^2 > {a}")
    if not hi * hi >= a:
        raise ArithmeticError(f"sqrt certificate failed: {hi}^2 < {a}")
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
        raise ArithmeticError(f"sqrt certificate failed: {out.lo}^2 > {x.lo}")
    if not out.hi * out.hi >= x.hi:
        raise ArithmeticError(f"sqrt certificate failed: {out.hi}^2 < {x.hi}")
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
    """
    if t == 0:
        return Interval(1)
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
        raise ValueError(f"atanh series needs 0 <= z < 1, got {z}")
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
        raise ValueError(f"log requires a positive argument, got {a}")
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
            f"reduced argument {s!r} has magnitude {m} > 1; the "
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
    sin_enc = _clamp_unit(total + Interval(-bound, bound))

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
    cos_enc = _clamp_unit(total + Interval(-bound, bound))

    return sin_enc, cos_enc


def _pi_for(x: Interval, prec: int) -> Interval:
    """``pi`` at enough extra digits to absorb the reduction of ``x``.

    Reducing an argument of size ``V`` subtracts about ``2V/pi`` copies of
    ``pi/2``, which multiplies the uncertainty in ``pi`` by that factor. So the
    enclosure of ``pi`` is taken with ``digits(V)`` extra decimal digits plus a
    fixed guard.
    """
    v = int(x.mag()) + 1
    return pi(max(int(prec), 1) + len(str(v)) + 14)


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

    The true reduced argument ``s = v - j*(pi/2)`` lies in ``S`` and satisfies
    ``|s| <= pi/4``, so ``mag(S) <= pi/4 + j*width(HP) < 1`` and
    ``_sin_cos_reduced`` applies. Finally

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
    if j_hi - j_lo > 32:
        return Interval(-1, 1)
    for j in range(j_lo, j_hi + 1):
        if not a.lo <= j <= a.hi:
            continue
        if which == "sin" and j % 2 == 1:
            out = out.hull(Interval(1) if j % 4 == 1 else Interval(-1))
        elif which == "cos" and j % 2 == 0:
            out = out.hull(Interval(1) if j % 4 == 0 else Interval(-1))
    return _clamp_unit(out)


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
#: :func:`erf`. Below it the alternating series is used; at and above it the
#: monotone tail bracket is used. The value is documented in ``erf``.
ERF_CROSSOVER = Fraction(6)


def _erf_point(z: Fraction, prec: int) -> Interval:
    """Certified enclosure of ``erf(z)`` for an exact rational ``z``.

    ``erf`` is odd, so negative arguments are reflected.

    **Series branch, ``0 < z <= 6``.**
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

    **Tail branch, ``z > 6``.** For ``z > 0``, using ``t/z >= 1`` on ``t >= z``,

        1 - erf(z) = (2/sqrt(pi)) * int_z^inf e^(-t^2) dt
                  <= (2/sqrt(pi)) * int_z^inf (t/z) e^(-t^2) dt
                  =  e^(-z^2) / (z * sqrt(pi))

    and ``1 - erf(z) > 0``. So ``erf(z) in [1 - U, 1]`` where ``U`` is a
    certified upper bound for ``e^(-z^2)/(z sqrt(pi))``, obtained from the
    certified ``exp``, ``pi`` and ``sqrt`` above. At the crossover ``z = 6``
    this bracket has width about ``2e-17``; below the crossover it would be too
    weak, which is why the crossover sits there and not lower. The bracket does
    not improve with ``prec``, and that is stated rather than papered over.
    """
    if z < 0:
        return -_erf_point(-z, prec)
    if z == 0:
        return Interval(0)

    g = max(int(prec), 1) + 20
    two_over_sqrt_pi = Interval.exact(2) / sqrt(pi(g), g)

    if z <= ERF_CROSSOVER:
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

    upper = (exp(Interval.exact(-(z * z)), g)
             / (Interval.exact(z) * sqrt(pi(g), g))).hi
    return _clamp_unit(Interval(1 - upper, Fraction(1)))


def erf(x: Interval, prec: int) -> Interval:
    """Certified enclosure of ``erf`` over ``x``.

    ``erf`` is strictly increasing on the reals (its derivative
    ``2 e^(-t^2)/sqrt(pi)`` is positive everywhere), so the range over
    ``[a, b]`` is ``[erf(a), erf(b)]`` and the endpoints are enclosed
    independently by ``_erf_point``.
    """
    return Interval(_erf_point(x.lo, prec).lo, _erf_point(x.hi, prec).hi)


def Phi(x: Interval, prec: int) -> Interval:
    """Certified enclosure of the standard normal CDF over ``x``.

    ``Phi(x) = (1 + erf(x/sqrt 2)) / 2``. ``sqrt 2`` is the certified enclosure
    from :func:`sqrt`, the division is interval division by an interval that
    does not contain zero, and ``erf`` is monotone, so the composition is an
    enclosure of the range. ``Phi(0) = 1/2`` comes out exactly, because
    ``[0,0] / [s_lo, s_hi] = [0,0]`` and ``erf([0,0]) = [0,0]``.
    """
    g = max(int(prec), 1) + 20
    arg = x / sqrt(Interval.exact(2), g)
    return ((Interval(1) + erf(arg, prec)) * _HALF).round_out(4 * g + 32)


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
    """
    g = max(int(prec), 1) + 20
    numerator = exp(-(x ** 2) * _HALF, prec)
    denominator = sqrt(Interval.exact(2) * pi(g), g)
    return (numerator / denominator).round_out(4 * g + 32)
