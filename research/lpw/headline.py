"""Which rounded decimals may stand in the LPW headline, and in which direction.

Standard library only (``fractions``, ``typing``). **No float,
no ``decimal``, no ``math``, no ``mpmath``, no ``numpy`` — every number in this
module is an exact ``fractions.Fraction`` or an exact integer, and every verdict
is an exact rational comparison.** Nothing here is a high-precision evaluation,
so nothing here needs a NON-CERTIFYING float label; it also means nothing here
is a certificate of anything but arithmetic.

The object
----------
``docs/OPEN_PROBLEMS.md`` section C ("LPW constant repair") and the
``lpw_fold_dispositions`` register tab name one exact fraction:

    LPW_CONSTANT = 260 / (3790446482793 * 2^40 * 10^21)

carried here from its **named factors**, never as a decimal literal. The
register describes 3790446482793 as the "B3 ceiling" retained by the R05
directed-interval companion. Its exact value is

    13 / 208381999114677270242918400000000000000000000
      = 6.2385427029355877929430410845189...e-44   (expansion truncated)

and its decimal expansion does not terminate, because the reduced denominator
has a prime factor other than 2 and 5. See ``has_terminating_decimal()``.

Two decimals are in play in the sources. The **delivered** headline is
``6.239e-44``; the register says it "exceeds that chain" and that the original
certifier and falsifier "do not catch these mathematical/report defects", with
the operator disposition "Do not admit the delivered 6.239e-44 certificate as
unqualified". The **proposed R05 repair** is ``6.238e-44``, an author-side
repair that "does not inherit Kimi approval automatically", whose companion
"rejected" the "oversized 6.239e-44 headline ... both modes".

The finding this module makes permanent and machine-checked
----------------------------------------------------------
**Direction decides admissibility.** By exact rational comparison:

* ``6.239e-44`` is **above** the exact fraction (relative excess +7.330e-05);
  it is also the correctly-rounded 4-significant-digit value of the exact
  fraction, so "it is the correct rounding" and "it is above the exact chain"
  are both true at once and neither settles anything by itself.
* ``6.238e-44`` is **below** the exact fraction (relative deficit -8.699e-05);
  it is *not* the correctly-rounded 4-digit value.

Therefore a rounded decimal is admissible **only when it is rounded away from
the asserted inequality**:

* headline asserted as an UPPER bound (``quantity <= D``): ``6.239e-44`` is
  sound, and the proposed repair ``6.238e-44`` is UNSOUND — being below the
  exact fraction, it asserts something stronger than the exact chain supports;
* headline asserted as a LOWER bound (``quantity >= D``): the directions
  reverse exactly — ``6.238e-44`` is sound and ``6.239e-44`` is UNSOUND;
* headline asserted as an EQUALITY: nothing rounded is admissible at all,
  and here no finite decimal whatsoever is, since the expansion is infinite.

Which of those three the LPW headline actually asserts is **not decided here**.
It is transcribed from the sources by whoever calls ``admissible_as``. This
module supplies the rule and the exact comparisons, and no more.

What this module does NOT establish
-----------------------------------
* It does **not verify the derivation** that produced the exact fraction
  ``260/(3790446482793*2^40*10^21)``. The fraction is transcribed from the
  register, and every factor in it — the 260, the B3 ceiling 3790446482793, the
  ``2^40``, the ``10^21`` — is taken on the source's word. If the derivation is
  wrong, every comparison here is a correct comparison against a wrong number.
* It does **not close, discharge, reduce or review** the R05 Rayleigh amplitude
  lemma, the ``E rho^4 = 8`` amplitude claim, the full tails / profile modulus,
  the conditional bounds, or the actual headline mutation. Those are exactly the
  five things the register says Kimi must review, and all five remain as the
  register writes them.
* It does **not promote or admit any certificate**. The delivered ``6.239e-44``
  certificate remains "not to be admitted unqualified"; the R05 repair remains
  author-side and does not inherit any external approval. This module admits
  nothing: ``admissible_as`` returns a verdict about *a decimal's direction of
  rounding*, which is not a review, not an approval and not a promotion.
* It does **not freeze** anything. ``docs/OPEN_PROBLEMS.md`` section C asks an
  operator to freeze either the corrected exact fraction or ``6.238e-44``. That
  choice is an operator act and has not been made here.
* Admissibility as computed here is **direction-soundness only**. It says
  nothing about tightness, and a grossly weak decimal on the sound side passes.
* It does **not touch qualitative LPW**, which the register records as
  unchanged by any of this, and it relates the 2D lower track to the 2D upper
  track and the 3D lifetime track in no way. No prize problem is involved.
* Green tests are not a mathematical review.

It settles exactly one question: which rounded decimals are admissible in which
direction.
"""
from __future__ import annotations

from fractions import Fraction
from typing import NamedTuple

__all__ = [
    "NUMERATOR",
    "B3_CEILING",
    "TWO_POWER",
    "TEN_POWER",
    "DELIVERED_HEADLINE",
    "PROPOSED_REPAIR",
    "DIRECTIONS",
    "RoundedDecimal",
    "exact_value",
    "has_terminating_decimal",
    "parse_decimal",
    "decimal_digits",
    "decimal_expansion",
    "relative_offset",
    "admissible_as",
    "report",
]

# --- the exact fraction, from its named factors (never a decimal literal) ---

NUMERATOR = 260
B3_CEILING = 3790446482793       # "B3 ceiling" retained by the R05 companion
TWO_POWER = 40                   # the 2^40 factor
TEN_POWER = 21                   # the 10^21 factor

#: The two decimals the sources put in play, as exact strings.
DELIVERED_HEADLINE = "6.239e-44"
PROPOSED_REPAIR = "6.238e-44"

#: The three ways a headline can assert its constant.
DIRECTIONS = ("upper", "lower", "equality")


def exact_value() -> Fraction:
    """``260 / (3790446482793 * 2^40 * 10^21)`` as an exact ``Fraction``.

    Built from the named factors above. Transcribed from the
    ``lpw_fold_dispositions`` register tab; not derived or checked here.
    """
    return Fraction(NUMERATOR, B3_CEILING * 2**TWO_POWER * 10**TEN_POWER)


def has_terminating_decimal(x: Fraction | None = None) -> bool:
    """True iff ``x`` has a finite decimal expansion (default: the constant).

    A reduced fraction terminates in base 10 exactly when its denominator has
    no prime factor other than 2 and 5. Decided here by exact integer division,
    so the ``equality`` branch of :func:`admissible_as` rests on a proof rather
    than on an assertion.
    """
    if x is None:
        x = exact_value()
    den = Fraction(x).denominator
    for p in (2, 5):
        while den % p == 0:
            den //= p
    return den == 1


def parse_decimal(literal: str | Fraction | int) -> Fraction:
    """Exactly parse a decimal literal such as ``"6.239e-44"``.

    ``fractions.Fraction`` parses decimal and scientific strings exactly, with
    no float ever constructed. A ``Fraction`` or ``int`` passes through. A
    ``float`` is rejected: admitting one would let binary rounding decide a
    verdict about decimal rounding.
    """
    if isinstance(literal, float):
        raise TypeError(
            "float input is rejected: this module decides decimal-rounding "
            "direction and must not inherit binary rounding. Pass a string "
            "such as '6.239e-44', or a Fraction."
        )
    return Fraction(literal)


def _decimal_exponent(x: Fraction) -> int:
    """The unique ``k`` with ``10^k <= x < 10^(k+1)``, for ``x > 0``.

    Exact integer/rational arithmetic; no logarithm and no float.
    """
    if x <= 0:
        raise ValueError("decimal exponent is defined here for x > 0 only")
    k = len(str(x.numerator)) - len(str(x.denominator))
    while Fraction(10) ** k > x:
        k -= 1
    while Fraction(10) ** (k + 1) <= x:
        k += 1
    return k


def _format_scientific(mantissa_digits: int, digits: int, exponent: int) -> str:
    """Render ``mantissa_digits * 10^exponent`` as ``d.ddd e<exp>``."""
    s = str(mantissa_digits)
    assert len(s) == digits, (s, digits)
    head, tail = s[0], s[1:]
    exp10 = exponent + digits - 1
    body = head if not tail else f"{head}.{tail}"
    return f"{body}e{'+' if exp10 >= 0 else '-'}{abs(exp10):02d}"


class RoundedDecimal(NamedTuple):
    """A correctly-rounded significant-digit decimal and the way it moved.

    ``direction`` is ``"up"`` when the rounded value is strictly greater than
    the exact fraction, ``"down"`` when strictly less, ``"exact"`` when equal.
    ``offset`` and ``rel_offset`` are exact rationals (rounded value minus
    exact, absolute and relative).
    """

    digits: int
    value: Fraction
    literal: str
    direction: str
    offset: Fraction
    rel_offset: Fraction


def decimal_digits(n: int, x: Fraction | None = None) -> RoundedDecimal:
    """The correctly-rounded ``n``-significant-digit decimal, and its direction.

    Rounding is round-half-to-even, performed on exact integers; ties are
    detected exactly rather than assumed absent. ``x`` defaults to the LPW
    constant.
    """
    if n < 1:
        raise ValueError("n must be at least 1 significant digit")
    if x is None:
        x = exact_value()
    x = Fraction(x)
    if x <= 0:
        raise ValueError("significant-digit rounding is implemented for x > 0")

    k = _decimal_exponent(x)
    exponent = k - (n - 1)
    scaled = x / Fraction(10) ** exponent          # in [10^(n-1), 10^n)
    floor_m = scaled.numerator // scaled.denominator
    frac = scaled - floor_m
    if frac > Fraction(1, 2):
        m = floor_m + 1
    elif frac < Fraction(1, 2):
        m = floor_m
    else:                                           # exact tie -> half to even
        m = floor_m if floor_m % 2 == 0 else floor_m + 1
    if m == 10**n:                                  # carry, e.g. 9.99 -> 10.0
        m //= 10
        exponent += 1

    value = Fraction(m) * Fraction(10) ** exponent
    offset = value - x
    direction = "exact" if offset == 0 else ("up" if offset > 0 else "down")
    return RoundedDecimal(
        digits=n,
        value=value,
        literal=_format_scientific(m, n, exponent),
        direction=direction,
        offset=offset,
        rel_offset=offset / x,
    )


def relative_offset(literal: str | Fraction | int) -> Fraction:
    """``(D - exact) / exact`` as an exact ``Fraction``. Positive means above."""
    d = parse_decimal(literal)
    exact = exact_value()
    return (d - exact) / exact


def admissible_as(literal: str | Fraction | int, direction: str) -> tuple[bool, str]:
    """Is decimal ``literal`` admissible in a headline asserted as ``direction``?

    The rule, and the whole of the rule: **a rounded decimal is admissible only
    when it is rounded AWAY from the asserted inequality.**

    ===========  =========================  ==============================
    direction    the headline asserts       admissible iff
    ===========  =========================  ==============================
    ``upper``    ``quantity <= D``          ``D >= exact``
    ``lower``    ``quantity >= D``          ``D <= exact``
    ``equality`` ``quantity == D``          ``D == exact`` (nothing rounded)
    ===========  =========================  ==============================

    Returns ``(verdict, reason)``. The verdict is direction-soundness only: it
    is not a review, not an approval, not a promotion, and it says nothing about
    whether ``D`` is tight or about whether the exact fraction is right.
    """
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    d = parse_decimal(literal)
    exact = exact_value()
    shown = literal if isinstance(literal, str) else str(d)
    rel = (d - exact) / exact if exact != 0 else Fraction(0)
    side = "above" if d > exact else ("below" if d < exact else "equal to")
    where = f"{shown} is {side} the exact fraction (relative offset {_rel_str(rel)})"

    if direction == "equality":
        if d == exact:
            return True, (
                f"{shown} equals the exact fraction, so it is not a rounded "
                "decimal at all. Under an equality headline only the exact "
                "value is admissible."
            )
        terminates = has_terminating_decimal(exact)
        why = (
            "the exact fraction's decimal expansion does not terminate "
            "(its reduced denominator has a prime factor other than 2 and 5), "
            "so no finite decimal equals it"
            if not terminates else
            "the exact fraction has a finite decimal expansion, but this "
            "literal is not it"
        )
        return False, (
            f"NOT ADMISSIBLE. Under an equality headline nothing rounded is "
            f"admissible: {why}. {where}."
        )

    if direction == "upper":
        ok = d >= exact
        if ok:
            return True, (
                f"ADMISSIBLE as an upper bound: {where}, so asserting "
                f"quantity <= {shown} is implied by the exact chain. This is "
                "direction-soundness only, not tightness, and not a review of "
                "the derivation behind the exact fraction."
            )
        return False, (
            f"NOT ADMISSIBLE as an upper bound: {where}. A decimal below the "
            f"exact value asserts quantity <= {shown}, which is STRONGER than "
            "the exact chain supports. An upper-bound headline must be rounded "
            "UP, away from the asserted inequality."
        )

    ok = d <= exact
    if ok:
        return True, (
            f"ADMISSIBLE as a lower bound: {where}, so asserting "
            f"quantity >= {shown} is implied by the exact chain. This is "
            "direction-soundness only, not tightness, and not a review of the "
            "derivation behind the exact fraction."
        )
    return False, (
        f"NOT ADMISSIBLE as a lower bound: {where}. A decimal above the exact "
        f"value asserts quantity >= {shown}, which is STRONGER than the exact "
        "chain supports. A lower-bound headline must be rounded DOWN, away "
        "from the asserted inequality."
    )


def _rel_str(rel: Fraction, places: int = 4) -> str:
    """Render an exact relative offset in signed scientific form, truncated.

    Truncation only, by exact integer division; no float is constructed and the
    rendered digits are a display, never a bound.
    """
    if rel == 0:
        return "0"
    sign = "+" if rel > 0 else "-"
    a = abs(rel)
    k = _decimal_exponent(a)
    scaled = a / Fraction(10) ** (k - (places - 1))
    m = scaled.numerator // scaled.denominator          # truncate, not round
    return sign + _format_scientific(m, places, k - (places - 1))


def decimal_expansion(x: Fraction | None = None, places: int = 40) -> str:
    """Truncated decimal expansion of ``x`` (default: the constant), for display.

    Exact integer division, truncated toward zero, with a trailing ``...`` when
    more digits exist. A display, never a bound.
    """
    if x is None:
        x = exact_value()
    x = Fraction(x)
    k = _decimal_exponent(x)
    scaled = x / Fraction(10) ** (k - (places - 1))
    m = scaled.numerator // scaled.denominator
    more = "..." if Fraction(m) != scaled else ""
    s = str(m)
    return f"{s[0]}.{s[1:]}{more}e{'+' if k >= 0 else '-'}{abs(k):02d}"


def report() -> None:
    """Print the full comparison table. A display; it admits nothing."""
    exact = exact_value()
    print("LPW headline constant — admissibility of rounded decimals")
    print("=" * 72)
    print(f"factors           : {NUMERATOR} / ({B3_CEILING} * 2^{TWO_POWER} * 10^{TEN_POWER})")
    print(f"exact (reduced)   : {exact.numerator} / {exact.denominator}")
    print(f"exact (expansion) : {decimal_expansion(exact, 40)}  (truncated display)")
    print(f"terminates in b10 : {has_terminating_decimal(exact)}")
    print()
    print("correctly-rounded significant-digit values")
    print("-" * 72)
    print(f"{'digits':>6}  {'literal':>16}  {'moved':>6}  {'relative offset':>16}")
    for n in range(2, 9):
        rd = decimal_digits(n)
        print(f"{rd.digits:>6}  {rd.literal:>16}  {rd.direction:>6}  "
              f"{_rel_str(rd.rel_offset):>16}")
    print()
    print("the two decimals the sources put in play")
    print("-" * 72)
    four = decimal_digits(4)
    for name, lit in (("delivered headline", DELIVERED_HEADLINE),
                      ("R05 proposed repair", PROPOSED_REPAIR)):
        d = parse_decimal(lit)
        side = "ABOVE" if d > exact else ("BELOW" if d < exact else "EQUAL")
        print(f"{name:<22} {lit:>12}  {side:>5} exact  "
              f"rel {_rel_str(relative_offset(lit)):>12}  "
              f"correctly-rounded-4sf: {lit == four.literal}")
    print()
    print("admissibility by asserted direction")
    print("-" * 72)
    print(f"{'literal':>12}  {'direction':>9}  {'verdict':>14}")
    for lit in (DELIVERED_HEADLINE, PROPOSED_REPAIR):
        for direction in DIRECTIONS:
            ok, _ = admissible_as(lit, direction)
            print(f"{lit:>12}  {direction:>9}  "
                  f"{'ADMISSIBLE' if ok else 'NOT ADMISSIBLE':>14}")
    print()
    print("reasons")
    print("-" * 72)
    for lit in (DELIVERED_HEADLINE, PROPOSED_REPAIR):
        for direction in DIRECTIONS:
            _, reason = admissible_as(lit, direction)
            print(f"  [{lit} as {direction}] {reason}")
    print()
    print("NOT ESTABLISHED by anything above: the derivation of the exact")
    print("fraction is not verified here; the R05 Rayleigh amplitude lemma, the")
    print("tails/profile modulus, the conditional bounds and the actual headline")
    print("mutation all remain open exactly as docs/OPEN_PROBLEMS.md section C")
    print("records them; no certificate is admitted or promoted; the delivered")
    print("6.239e-44 certificate is still not to be admitted unqualified; and")
    print("qualitative LPW is unchanged. Which direction the LPW headline")
    print("asserts is transcribed from the sources, never decided here.")


if __name__ == "__main__":
    report()
