"""Certified interval arithmetic over exact rational endpoints.

THE CONTRACT
------------
Every ``Interval`` returned by this package **provably contains** the true value
of the expression it was built from. Containment is unconditional: it does not
depend on any precision parameter, on the order of operations, or on how large
the operands are. Tightness is best effort and *does* depend on a precision
hint.

Endpoints are ``fractions.Fraction``. No binary floating point enters any
endpoint at any point, which is why the containment claim is a claim about
exact rational inequalities that the reader can re-derive by hand.

Why this exists
---------------
``docs/OPEN_PROBLEMS.md`` records three obligations that are phrased in terms of
*certified* interval bounds and *certified* enclosures — ``OBL-H5-JETMOD``
(A1), ``OBL-H5-ZBAND`` (A3) and ``D3-LEMMA-RN-UNIF`` (A5) — while the
program's computational carriers evaluate the corresponding quantities in
``mpmath`` at high precision. High precision is not certification. This module
is the arithmetic layer that a future certified carrier would need.

WHAT THIS MODULE DOES NOT ESTABLISH
-----------------------------------
* It does not discharge, reduce, close, promote or reclassify ``OBL-H5-JETMOD``,
  ``OBL-H5-ZBAND``, ``OBL-H5-REMOTE-THRESHOLD``, ``OBL-D1-PROMOTE``,
  ``D3-LEMMA-RN-UNIF``, ``PERC-DECAY``, ``OBL-B1-BRANCH`` or any other named
  obligation. It computes nothing about ``q(r)``, about jets, about bands or
  about the RN lemma. It is arithmetic infrastructure and nothing else.
* It does not certify any *existing* number in the corpus. Numbers previously
  produced in floating point stay exactly as certified (or uncertified) as they
  were; re-running them through this library would be new work, not a
  relabelling of old work.
* Availability of a certified arithmetic does not make a band enclosure exist.
  ``OBL-H5-JETMOD`` asks for a finite per-band computation; this module supplies
  none of it.
* It is not a proof assistant. The containment arguments live in the
  docstrings and in the negative controls under ``tests/test_interval.py``;
  they are ordinary mathematics that a human reviewer must read.

Conventions that are deliberate, and are part of the contract
-------------------------------------------------------------
* ``float`` is rejected on construction. ``Fraction(0.1)`` is not ``1/10``, and
  silently accepting it is exactly the kind of quiet inexactness this package
  exists to exclude. Pass ``int``, ``Fraction``, ``Decimal`` or a ``str``
  (``"0.1"``, ``"1/3"`` both parse exactly).
* Division by an interval containing zero raises ``ZeroDivisionError``. There is
  no extended-interval fallback: a bound that silently becomes infinite is a
  bound nobody checked.
* ``x ** 0`` is ``[1, 1]`` for every ``x``, including intervals containing zero.
  This is the ``0 ** 0 == 1`` convention of ``int``; it is stated so that no
  caller has to guess.
"""
from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
from typing import Optional, Union

__all__ = ["Interval", "to_fraction"]

Scalar = Union[int, Fraction, str, Decimal]


def to_fraction(x: Scalar) -> Fraction:
    """Exact conversion to ``Fraction``; ``float`` is refused on purpose.

    ``bool`` arrives here as ``int`` and converts to 0/1, which is harmless.
    ``float`` is refused because ``Fraction(0.1) == 3602879701896397/36028797018963968``
    and no caller ever means that. Refusing it is the difference between an
    exact endpoint and an endpoint that merely looks exact.
    """
    if isinstance(x, Fraction):
        return x
    if isinstance(x, bool):
        return Fraction(int(x))
    if isinstance(x, int):
        return Fraction(x)
    if isinstance(x, str):
        return Fraction(x)
    if isinstance(x, Decimal):
        return Fraction(x)
    if isinstance(x, float):
        raise TypeError(
            "float endpoints are refused: Fraction(0.1) is not 1/10. "
            "Pass an int, a Fraction, a Decimal, or a string such as '0.1' "
            "or '1/3'. If you really mean the binary double, pass "
            "Fraction(the_float) explicitly and accept that it is not the "
            "decimal you typed."
        )
    raise TypeError(f"cannot convert {type(x).__name__} to an exact Fraction")


def _shift_floor(x: Fraction, s: int) -> Fraction:
    """Largest ``k / 2**s`` (``k`` integer) that is ``<= x``. Exact."""
    if s >= 0:
        k = (x.numerator << s) // x.denominator
        return Fraction(k, 1 << s)
    k = x.numerator // (x.denominator << (-s))
    return Fraction(k << (-s), 1)


def _shift_ceil(x: Fraction, s: int) -> Fraction:
    """Smallest ``k / 2**s`` (``k`` integer) that is ``>= x``. Exact."""
    return -_shift_floor(-x, s)


def _sig_shift(x: Fraction, bits: int) -> int:
    """A shift ``s`` with ``|x| * 2**s`` of roughly ``bits`` binary digits."""
    e = x.numerator.bit_length() - x.denominator.bit_length()
    return bits - e


def _decimal_string(x: Fraction, places: int, upward: bool) -> str:
    """``x`` as a decimal string, rounded outward in the requested direction.

    ``upward=False`` rounds toward ``-inf`` and ``upward=True`` toward ``+inf``,
    so a display built from ``(_decimal_string(lo, p, False),
    _decimal_string(hi, p, True))`` is itself a valid enclosure: the display can
    never claim a tighter interval than the exact endpoints support.
    """
    scale = 10 ** places
    n, d = x.numerator * scale, x.denominator
    q = -((-n) // d) if upward else n // d
    neg = q < 0
    s = str(abs(q)).rjust(places + 1, "0")
    body = s if places == 0 else f"{s[:-places]}.{s[-places:]}"
    return ("-" if neg else "") + body


class Interval:
    """A closed real interval ``[lo, hi]`` with exact rational endpoints.

    Invariant: ``lo <= hi`` and both are ``Fraction``. A degenerate interval
    ``lo == hi`` is a point.

    Every operation below is an *outward* operation: the returned interval
    contains the exact set ``{f(u, v) : u in self, v in other}``. For the
    elementary operations it is in fact equal to the exact range, except where
    the same variable occurs more than once in an expression the caller writes
    (the classic dependency problem) — that widens the result but never breaks
    containment.
    """

    __slots__ = ("lo", "hi")

    def __init__(self, lo: Scalar, hi: Optional[Scalar] = None) -> None:
        a = to_fraction(lo)
        b = a if hi is None else to_fraction(hi)
        if a > b:
            raise ValueError(f"empty interval: lo={a} > hi={b}")
        self.lo = a
        self.hi = b

    # ---------------------------------------------------------------- builders

    @classmethod
    def exact(cls, x: Scalar) -> "Interval":
        """The degenerate (point) interval ``[x, x]``."""
        return cls(x)

    @classmethod
    def _coerce(cls, x) -> "Interval":
        if isinstance(x, Interval):
            return x
        return cls(to_fraction(x))

    # ------------------------------------------------------------- inspection

    def width(self) -> Fraction:
        """``hi - lo``, exactly. Zero for a point interval."""
        return self.hi - self.lo

    def mid(self) -> Fraction:
        """``(lo + hi) / 2``, exactly. Always an element of the interval."""
        return (self.lo + self.hi) / 2

    def mag(self) -> Fraction:
        """``max{|x| : x in self}`` — the magnitude."""
        a, b = abs(self.lo), abs(self.hi)
        return a if a > b else b

    def mig(self) -> Fraction:
        """``min{|x| : x in self}`` — the mignitude. Zero iff zero is inside."""
        if self.contains_zero():
            return Fraction(0)
        a, b = abs(self.lo), abs(self.hi)
        return a if a < b else b

    def contains_zero(self) -> bool:
        return self.lo <= 0 <= self.hi

    def is_point(self) -> bool:
        return self.lo == self.hi

    def __contains__(self, other) -> bool:
        """``x in iv`` for a scalar; ``jv in iv`` for set containment."""
        if isinstance(other, Interval):
            return self.lo <= other.lo and other.hi <= self.hi
        return self.lo <= to_fraction(other) <= self.hi

    # ------------------------------------------------------------ set algebra

    def hull(self, other: "Interval") -> "Interval":
        """Smallest interval containing both."""
        o = Interval._coerce(other)
        return Interval(min(self.lo, o.lo), max(self.hi, o.hi))

    def intersect(self, other: "Interval") -> Optional["Interval"]:
        """Intersection, or ``None`` when the two are disjoint."""
        o = Interval._coerce(other)
        lo, hi = max(self.lo, o.lo), min(self.hi, o.hi)
        return None if lo > hi else Interval(lo, hi)

    def round_out(self, sig_bits: int) -> "Interval":
        """Widen to endpoints with about ``sig_bits`` significant binary digits.

        Rounds ``lo`` toward ``-inf`` and ``hi`` toward ``+inf``, so the result
        contains ``self``: containment is preserved, tightness is spent. This
        exists only to stop exact rational endpoints from doubling in size under
        repeated squaring; it is never required for correctness.
        """
        lo = self.lo if self.lo == 0 else _shift_floor(self.lo, _sig_shift(self.lo, sig_bits))
        hi = self.hi if self.hi == 0 else _shift_ceil(self.hi, _sig_shift(self.hi, sig_bits))
        return Interval(lo, hi)

    # ------------------------------------------------------------- arithmetic

    def __add__(self, other) -> "Interval":
        o = Interval._coerce(other)
        return Interval(self.lo + o.lo, self.hi + o.hi)

    __radd__ = __add__

    def __sub__(self, other) -> "Interval":
        o = Interval._coerce(other)
        return Interval(self.lo - o.hi, self.hi - o.lo)

    def __rsub__(self, other) -> "Interval":
        return Interval._coerce(other) - self

    def __neg__(self) -> "Interval":
        return Interval(-self.hi, -self.lo)

    def __pos__(self) -> "Interval":
        return self

    def __abs__(self) -> "Interval":
        """``{|x| : x in self}`` = ``[mig, mag]``."""
        return Interval(self.mig(), self.mag())

    def __mul__(self, other) -> "Interval":
        """All four corner products; the range of a bilinear map on a box is
        attained at a corner, so the min/max of the four is exact."""
        o = Interval._coerce(other)
        p = (self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi)
        return Interval(min(p), max(p))

    __rmul__ = __mul__

    def __truediv__(self, other) -> "Interval":
        """``self * (1 / other)``; refuses a divisor straddling zero."""
        o = Interval._coerce(other)
        if o.contains_zero():
            raise ZeroDivisionError(
                f"interval division by {o!r}, which contains zero"
            )
        return self * Interval(1 / o.hi, 1 / o.lo)

    def __rtruediv__(self, other) -> "Interval":
        return Interval._coerce(other) / self

    def __pow__(self, n: int) -> "Interval":
        """Exact range of ``x ** n`` over the interval, for integer ``n``.

        The only subtle case is an even positive exponent over an interval
        straddling zero. ``x -> x**n`` is then *not* monotone: it decreases to
        an interior minimum at ``x = 0`` and increases after it. Evaluating at
        the endpoints alone gives ``[min(lo**n, hi**n), max(lo**n, hi**n)]``,
        which omits the interior minimum ``0``. The correct range is
        ``[0, mag**n]``. For example ``Interval(-2, 1) ** 2`` is ``[0, 4]`` —
        it is neither ``[4, 1]`` (not even an interval) nor ``[1, 4]`` (which
        excludes the true value 0 attained at x = 0).

        Odd exponents are monotone increasing, so the endpoints suffice in
        order. Negative exponents reduce to the reciprocal interval, which
        requires zero to be outside.
        """
        if not isinstance(n, int) or isinstance(n, bool):
            raise TypeError("interval exponent must be a plain int")
        if n == 0:
            return Interval(1)
        if n < 0:
            if self.contains_zero():
                raise ZeroDivisionError(
                    f"negative power of {self!r}, which contains zero"
                )
            return Interval(1 / self.hi, 1 / self.lo) ** (-n)
        if n % 2 == 1:
            return Interval(self.lo ** n, self.hi ** n)
        if self.lo >= 0:
            return Interval(self.lo ** n, self.hi ** n)
        if self.hi <= 0:
            return Interval(self.hi ** n, self.lo ** n)
        # straddles zero: interior minimum at 0
        return Interval(Fraction(0), self.mag() ** n)

    # ------------------------------------------------------------- comparison

    def __eq__(self, other) -> bool:
        if not isinstance(other, Interval):
            return NotImplemented
        return self.lo == other.lo and self.hi == other.hi

    def __hash__(self) -> int:
        return hash((self.lo, self.hi))

    # ------------------------------------------------------------------ display

    def __repr__(self) -> str:
        lo_d = _decimal_string(self.lo, 12, upward=False)
        hi_d = _decimal_string(self.hi, 12, upward=True)
        return (
            f"Interval({self.lo!s}, {self.hi!s})"
            f"  ~[{lo_d}, {hi_d}]"
        )

    def __str__(self) -> str:
        return repr(self)
