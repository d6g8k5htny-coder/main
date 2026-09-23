"""Exact absolute-moment identities for standard normals, and the corpus audit.

Purpose. `docs/CONTRIBUTION_PLAN.md` section 2 records that the corpus carries a
**false identity**::

    E(|xi| + |eta|)^4  =  12 + 16/pi            (CLAIMED — FALSE)

for ``xi, eta`` iid standard normal. It is asserted in
``registers/json/lpw_fold_dispositions.json`` (LPW_CONSTANT disposition, R05
finding) and echoed in ``claims/graph.json`` and ``docs/RESEARCH_MAP.md``.

This module derives the truth from the absolute-moment formula rather than
copying any reported number, and pins the refutation as exact arithmetic.

The derivation, written out.
    For ``X ~ N(0, 1)``, ``E|X|^k = 2^(k/2) * Gamma((k+1)/2) / sqrt(pi)``.
    Split on parity, using ``Gamma(m + 1/2) = (2m)! / (4^m m!) * sqrt(pi)`` and
    ``Gamma(m + 1) = m!``:

        k = 2m       E|X|^(2m)   = 2^m * (2m)!/(4^m m!)  = (2m)!/(2^m m!) = (2m-1)!!
        k = 2m + 1   E|X|^(2m+1) = 2^m * m! * sqrt(2/pi)

    So every absolute moment is ``rational * s^e`` with ``s = sqrt(2/pi)`` and
    ``e = k mod 2``. First few: ``1, s, 1, 2s, 3, 8s, 15``.

    Independence plus the binomial theorem give

        E(|xi| + |eta|)^4 = sum_{j=0..4} C(4, j) E|xi|^j E|eta|^(4-j)
                          = 3  +  4*(s)(2s)  +  6*(1)(1)  +  4*(2s)(s)  +  3
                          = 3  +  8 s^2      +  6         +  8 s^2      +  3
                          = 12 + 16 s^2      =  12 + 32/pi
                          = 22.18591635788...                  (NON-CERTIFYING decimal)

    The claimed ``12 + 16/pi`` is what you get if you keep exactly one of the two
    equal cross terms ``4 E|xi|^3 E|eta|`` and ``4 E|xi| E|eta|^3``. Each is
    worth ``8 s^2 = 16/pi``. **The structural cause of the defect is one dropped
    cross term**, not a rounding error and not a different convention.

Exactness. Values live in ``Q[s]``, ``s = sqrt(2/pi)``, represented by
:class:`Sym` as a map from the power of ``s`` to a :class:`fractions.Fraction`.
Since ``s^2 = 2/pi``, any element whose powers lie in ``{0, 2}`` is exactly
``a + b/pi`` with ``a, b`` rational -- the ``{1, 1/pi}`` basis the task asks
for, recovered by :meth:`Sym.as_a_plus_b_over_pi`. For ``n = 2`` summands every
even ``k`` lands in that basis, because a term can carry at most two odd factors.
Odd ``k`` lands on ``s^1``, which is *not* in the span of ``{1, 1/pi}``; the
representation covers it, the two-element basis does not.

No float enters any comparison. :func:`numeric_oracle` exists only to
cross-check, and everything it returns is labelled **NON-CERTIFYING**.

WHAT THIS MODULE DOES NOT ESTABLISH
    * Refuting ``12 + 16/pi`` does **not** invalidate any derivation that cited
      it. It flags every consumer for re-check. The consumers are not enumerated
      here and must be enumerated separately.
    * No status label is promoted, closed, discharged or reclassified. The LPW
      ``AMEND REQUIRED`` disposition, the R05 review requirement and all five
      D1 validity premises stand exactly as their registers record them.
    * Confirming a sibling identity certifies nothing about the object that used
      it. A true lemma inside an unreviewed derivation leaves the derivation
      unreviewed.
    * Nothing here bears on the 2D upper track, the 2D lower track or the 3D
      lifetime track, and nothing here composes any of them.
    * No prize problem is touched. None is solved.
    * Every decimal printed anywhere in this module is a NON-CERTIFYING display.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction as F
from typing import Dict, Iterable, List, Tuple

__all__ = [
    "Sym", "S_SYM", "ONE_SYM",
    "abs_moment", "abs_moment_sym", "sum_abs_power_moment",
    "sum_abs_power_terms", "binomial_terms",
    "CLAIMED_FALSE_IDENTITY", "refutation_record",
    "QSqrt2", "truncated_normal_moment", "rayleigh_even_moment",
    "sibling_rayleigh_fourth_moment", "sibling_truncated_second_moment",
    "sibling_truncated_fourth_moment", "sibling_records",
    "FOUND_NOT_SETTLED",
    "numeric_oracle", "numeric_abs_moment", "oracle_report",
]


# --------------------------------------------------------------------------
# Exact arithmetic in Q[s], s = sqrt(2/pi)
# --------------------------------------------------------------------------

class Sym:
    """An exact element of ``Q[s]`` where ``s = sqrt(2/pi)``.

    Stored as ``{power_of_s: Fraction}`` with zero coefficients stripped, so
    ``==`` is exact structural equality. ``s`` is transcendental, so distinct
    powers are linearly independent over ``Q`` and this representation is
    faithful: two Sym objects are equal as real numbers iff their coefficient
    maps agree.
    """

    __slots__ = ("coeffs",)

    def __init__(self, coeffs: Dict[int, F] | None = None) -> None:
        c: Dict[int, F] = {}
        for power, coeff in (coeffs or {}).items():
            if int(power) < 0:
                raise ValueError("negative powers of s are not represented")
            coeff = F(coeff)
            if coeff:
                c[int(power)] = c.get(int(power), F(0)) + coeff
        self.coeffs = {p: v for p, v in sorted(c.items()) if v}

    # -- constructors ------------------------------------------------------
    @classmethod
    def rational(cls, value) -> "Sym":
        return cls({0: F(value)})

    @classmethod
    def term(cls, coeff, power: int) -> "Sym":
        return cls({power: F(coeff)})

    @classmethod
    def from_a_plus_b_over_pi(cls, a, b) -> "Sym":
        """``a + b/pi``.  Since ``s^2 = 2/pi``, ``b/pi = (b/2) s^2``."""
        return cls({0: F(a), 2: F(b) / 2})

    # -- ring operations ---------------------------------------------------
    def __add__(self, other: "Sym") -> "Sym":
        out = dict(self.coeffs)
        for p, v in other.coeffs.items():
            out[p] = out.get(p, F(0)) + v
        return Sym(out)

    def __sub__(self, other: "Sym") -> "Sym":
        out = dict(self.coeffs)
        for p, v in other.coeffs.items():
            out[p] = out.get(p, F(0)) - v
        return Sym(out)

    def __neg__(self) -> "Sym":
        return Sym({p: -v for p, v in self.coeffs.items()})

    def __mul__(self, other) -> "Sym":
        if not isinstance(other, Sym):
            return Sym({p: v * F(other) for p, v in self.coeffs.items()})
        out: Dict[int, F] = {}
        for p1, v1 in self.coeffs.items():
            for p2, v2 in other.coeffs.items():
                out[p1 + p2] = out.get(p1 + p2, F(0)) + v1 * v2
        return Sym(out)

    __rmul__ = __mul__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Sym) and self.coeffs == other.coeffs

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.coeffs.items())))

    def __bool__(self) -> bool:
        return bool(self.coeffs)

    # -- views -------------------------------------------------------------
    def as_a_plus_b_over_pi(self) -> Tuple[F, F]:
        """Return ``(a, b)`` with ``self == a + b/pi``, exactly.

        Raises ``ValueError`` when the element does not lie in the two-element
        basis ``{1, 1/pi}`` -- for instance any odd power of ``s``, which is
        what every odd ``k`` produces. Refusing is the point: silently
        projecting an ``s^1`` term onto ``{1, 1/pi}`` is the class of error this
        module exists to catch.
        """
        bad = [p for p in self.coeffs if p not in (0, 2)]
        if bad:
            raise ValueError(
                f"not in the {{1, 1/pi}} basis: s-powers {sorted(bad)} present"
            )
        return self.coeffs.get(0, F(0)), 2 * self.coeffs.get(2, F(0))

    def __repr__(self) -> str:
        if not self.coeffs:
            return "Sym(0)"
        parts = []
        for p, v in self.coeffs.items():
            if p == 0:
                parts.append(f"{v}")
            elif p == 1:
                parts.append(f"{v}*sqrt(2/pi)")
            elif p % 2 == 0:
                parts.append(f"{v * 2 ** (p // 2)}/pi^{p // 2}" if p > 2
                             else f"{v * 2}/pi")
            else:
                parts.append(f"{v}*s^{p}")
        return "Sym(" + " + ".join(parts) + ")"


ONE_SYM = Sym.rational(1)
S_SYM = Sym.term(1, 1)          # s = sqrt(2/pi)


# --------------------------------------------------------------------------
# Absolute moments of a standard normal
# --------------------------------------------------------------------------

def abs_moment(k: int) -> Tuple[F, int]:
    """``E|X|^k`` for ``X ~ N(0, 1)``, exactly, as ``(rational_part, power_of_s)``.

    ``s = sqrt(2/pi)``; the power is ``k mod 2``. Derived directly from
    ``E|X|^k = 2^(k/2) Gamma((k+1)/2) / sqrt(pi)`` by splitting on parity:

        k = 2m      -> (2m)!/(2^m m!) = (2m-1)!!,  power 0
        k = 2m + 1  -> 2^m * m!,                   power 1

    Exact for every ``k >= 0``; no float, no ``math.gamma``.

    >>> abs_moment(0), abs_moment(1), abs_moment(2)
    ((Fraction(1, 1), 0), (Fraction(1, 1), 1), (Fraction(1, 1), 0))
    >>> abs_moment(3), abs_moment(4)
    ((Fraction(2, 1), 1), (Fraction(3, 1), 0))
    """
    if k < 0:
        raise ValueError("k must be a nonnegative integer")
    m, parity = divmod(int(k), 2)
    if parity == 0:
        # (2m)! / (2^m m!)  ==  (2m-1)!!
        return F(math.factorial(2 * m), 2 ** m * math.factorial(m)), 0
    return F(2 ** m * math.factorial(m)), 1


def abs_moment_sym(k: int) -> Sym:
    """``E|X|^k`` as a :class:`Sym`."""
    coeff, power = abs_moment(k)
    return Sym.term(coeff, power)


def _compositions(k: int, n: int) -> Iterable[Tuple[int, ...]]:
    """All ``n``-tuples of nonnegative integers summing to ``k``."""
    if n == 1:
        yield (k,)
        return
    for first in range(k + 1):
        for rest in _compositions(k - first, n - 1):
            yield (first,) + rest


def _multinomial(k: int, parts: Tuple[int, ...]) -> int:
    out = math.factorial(k)
    for p in parts:
        out //= math.factorial(p)
    return out


def sum_abs_power_terms(n: int, k: int) -> List[dict]:
    """The individual expansion terms of ``E(|X_1| + ... + |X_n|)^k``.

    Each record carries the exponent tuple, the multinomial coefficient, the
    per-factor absolute moments and the term's exact value. Exposed so the
    dropped cross term in the false identity is a nameable object rather than a
    remark in prose.
    """
    if n < 1 or k < 0:
        raise ValueError("need n >= 1 and k >= 0")
    terms: List[dict] = []
    for parts in _compositions(k, n):
        coeff = _multinomial(k, parts)
        value = Sym.rational(coeff)
        for e in parts:
            value = value * abs_moment_sym(e)
        terms.append({
            "exponents": parts,
            "multinomial": coeff,
            "factor_moments": tuple(abs_moment(e) for e in parts),
            "value": value,
        })
    return terms


def binomial_terms(k: int) -> List[dict]:
    """``sum_abs_power_terms(2, k)`` -- the two-summand binomial expansion."""
    return sum_abs_power_terms(2, k)


def sum_abs_power_moment(n: int, k: int) -> Sym:
    """``E(|X_1| + ... + |X_n|)^k`` for iid standard normals, exactly.

    Computed by the multinomial expansion together with independence:

        E(sum_i |X_i|)^k = sum_{e_1+...+e_n = k} (k choose e_1,...,e_n)
                           prod_i E|X_i|^(e_i)

    ``n = 2`` is the binomial case that the corpus identity concerns. General
    ``n`` is supported; note that for ``n >= 3`` the result can carry powers of
    ``s`` above 2 and then does **not** lie in the ``{1, 1/pi}`` basis.

    >>> sum_abs_power_moment(2, 4).as_a_plus_b_over_pi()
    (Fraction(12, 1), Fraction(32, 1))
    >>> sum_abs_power_moment(2, 2).as_a_plus_b_over_pi()
    (Fraction(2, 1), Fraction(2, 1))
    """
    total = Sym()
    for term in sum_abs_power_terms(n, k):
        total = total + term["value"]
    return total


# --------------------------------------------------------------------------
# The refutation record
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class IdentityRecord:
    """A corpus-asserted identity, its verdict, and what the verdict is not."""

    key: str
    statement: str
    source: str
    verdict: str                      # REFUTED | CONFIRMED_EXACT | NOT_SETTLED ...
    claimed: object = None
    derived: object = None
    discrepancy: object = None
    structural_cause: str = ""
    notes: Tuple[str, ...] = field(default_factory=tuple)
    does_not_establish: Tuple[str, ...] = field(default_factory=tuple)


#: The claimed value ``12 + 16/pi``, transcribed from the register, not endorsed.
CLAIMED_FALSE_IDENTITY = Sym.from_a_plus_b_over_pi(12, 16)


def refutation_record() -> IdentityRecord:
    """The exact refutation of ``E(|xi|+|eta|)^4 = 12 + 16/pi``.

    The true value is recomputed here from :func:`sum_abs_power_moment`; no
    reported number is copied. The discrepancy is exactly ``16/pi``, and the
    named structural cause is one dropped cross term.
    """
    true_value = sum_abs_power_moment(2, 4)
    discrepancy = true_value - CLAIMED_FALSE_IDENTITY

    terms = binomial_terms(4)
    cross = [t for t in terms if t["exponents"] in ((3, 1), (1, 3))]
    assert len(cross) == 2, "the k=4 binomial expansion must have two cross terms"
    assert cross[0]["value"] == cross[1]["value"], "the two cross terms are equal"
    # Each cross term is exactly the discrepancy: dropping one loses 16/pi.
    assert cross[0]["value"] == discrepancy

    return IdentityRecord(
        key="LPW-R05-ABS4",
        statement="E(|xi| + |eta|)^4 = 12 + 16/pi  for xi, eta iid N(0,1)",
        source=("registers/json/lpw_fold_dispositions.json (LPW_CONSTANT "
                "disposition, R05 finding); echoed in claims/graph.json and "
                "docs/RESEARCH_MAP.md. Named in docs/CONTRIBUTION_PLAN.md s2."),
        verdict="REFUTED",
        claimed=CLAIMED_FALSE_IDENTITY,
        derived=true_value,
        discrepancy=discrepancy,
        structural_cause=(
            "One of the two equal cross terms was dropped. The k=4 binomial "
            "expansion has five terms; 4*E|xi|^3*E|eta| and 4*E|xi|*E|eta|^3 "
            "are each exactly 8*s^2 = 16/pi. Keeping one instead of both gives "
            "12 + 16/pi. This is a dropped-term defect, not rounding and not a "
            "variance or normalisation convention."
        ),
        notes=(
            "True value 12 + 32/pi; NON-CERTIFYING decimal 22.18591635788...",
            "Claimed value 12 + 16/pi; NON-CERTIFYING decimal 17.09295817894...",
            "Discrepancy exactly 16/pi; the claim UNDERSTATES the true value. "
            "Direction, for re-checking consumers: using 12 + 16/pi as an UPPER "
            "budget for E(|xi|+|eta|)^4 is a bound the true value violates; "
            "using it as a LOWER bound is true but non-sharp. Which direction "
            "any consumer needs is NOT decided here.",
            "The corpus itself already records this identity as false (R05); "
            "this module supplies the exact derivation, not the discovery.",
        ),
        does_not_establish=(
            "Does NOT invalidate any derivation that cited the identity. It "
            "flags every consumer for re-check; consumers are not enumerated "
            "here and must be enumerated separately.",
            "Does NOT resolve the LPW_CONSTANT disposition, which stands at "
            "AMEND REQUIRED, nor discharge the R05 review requirement.",
            "Does NOT bear on qualitative LPW, which its register records as "
            "unchanged by the constant defect.",
            "Does NOT change any status label anywhere in the corpus.",
        ),
    )


# --------------------------------------------------------------------------
# Sibling audit: exact arithmetic in Q[sqrt 2], for truncated normal moments
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class QSqrt2:
    """Exact element ``a + b*sqrt(2)`` of ``Q[sqrt 2]``, with rational ``a, b``.

    ``sqrt(2)`` is irrational, so ``(a, b)`` is a faithful representation and
    ``==`` is exact equality of real numbers.
    """

    a: F = F(0)
    b: F = F(0)

    @classmethod
    def of(cls, value) -> "QSqrt2":
        return value if isinstance(value, QSqrt2) else cls(F(value), F(0))

    @classmethod
    def sqrt2(cls) -> "QSqrt2":
        return cls(F(0), F(1))

    def __add__(self, other) -> "QSqrt2":
        o = QSqrt2.of(other)
        return QSqrt2(self.a + o.a, self.b + o.b)

    __radd__ = __add__

    def __sub__(self, other) -> "QSqrt2":
        o = QSqrt2.of(other)
        return QSqrt2(self.a - o.a, self.b - o.b)

    def __rsub__(self, other) -> "QSqrt2":
        return QSqrt2.of(other) - self

    def __neg__(self) -> "QSqrt2":
        return QSqrt2(-self.a, -self.b)

    def __mul__(self, other) -> "QSqrt2":
        o = QSqrt2.of(other)
        return QSqrt2(self.a * o.a + 2 * self.b * o.b, self.a * o.b + self.b * o.a)

    __rmul__ = __mul__

    def __pow__(self, n: int) -> "QSqrt2":
        if n < 0:
            raise ValueError("only nonnegative powers")
        out = QSqrt2(F(1), F(0))
        for _ in range(n):
            out = out * self
        return out

    def to_float(self) -> float:
        """NON-CERTIFYING display only."""
        return float(self.a) + float(self.b) * math.sqrt(2.0)


def truncated_normal_moment(b: F, k: int) -> Tuple[QSqrt2, QSqrt2]:
    """Exact ``(coeff_of_Phi, coeff_of_phi)`` for ``E[Q^k 1{Q < 0}]``.

    ``Q ~ N(-b, 2)`` -- the planar law the corpus writes as ``Q ~ N(-b, 2)``
    with ``z_0(b) = E[Q^2 1{Q<0}]``. Substituting ``Q = -b + sqrt(2) Z`` with
    ``Z ~ N(0, 1)`` turns ``{Q < 0}`` into ``{Z < c}``, ``c = b/sqrt(2)``, so

        E[Q^k 1{Q<0}] = sum_{j=0..k} C(k, j) (-b)^(k-j) 2^(j/2) T_j,
        T_j := E[Z^j 1{Z < c}].

    The truncated standard-normal moments obey the exact recursion

        T_0 = Phi(c),   T_1 = -phi(c),   T_j = (j-1) T_(j-2) - c^(j-1) phi(c),

    which follows from ``d/dz phi = -z phi`` and integration by parts. Every
    ``T_j`` is therefore ``alpha_j Phi(c) + beta_j phi(c)`` with ``alpha_j,
    beta_j`` in ``Q[sqrt 2]`` once ``c = b/sqrt(2)`` is substituted.

    The result is returned in the basis ``{Phi(b/sqrt 2), phi(b/sqrt 2)}``.
    ``Phi`` and ``phi`` are not rational, but the *coefficients* are, so an
    asserted closed form can be settled exactly by comparing coefficients --
    no float, no quadrature.
    """
    if k < 0:
        raise ValueError("k must be a nonnegative integer")
    b = F(b)
    root2 = QSqrt2.sqrt2()
    c = QSqrt2(F(0), b / 2)                 # b/sqrt(2) == b*sqrt(2)/2

    # T_j = alpha_j * Phi(c) + beta_j * phi(c)
    alpha: List[QSqrt2] = [QSqrt2(F(1), F(0))]
    beta: List[QSqrt2] = [QSqrt2(F(0), F(0))]
    if k >= 1:
        alpha.append(QSqrt2(F(0), F(0)))
        beta.append(QSqrt2(F(-1), F(0)))
    for j in range(2, k + 1):
        alpha.append(QSqrt2.of(j - 1) * alpha[j - 2])
        beta.append(QSqrt2.of(j - 1) * beta[j - 2] - c ** (j - 1))

    a_tot = QSqrt2()
    b_tot = QSqrt2()
    for j in range(k + 1):
        # C(k, j) * (-b)^(k-j) * (sqrt 2)^j
        w = QSqrt2.of(math.comb(k, j)) * QSqrt2.of(-b) ** (k - j) * root2 ** j
        a_tot = a_tot + w * alpha[j]
        b_tot = b_tot + w * beta[j]
    return a_tot, b_tot


def rayleigh_even_moment(p: int) -> F:
    """``E[rho^(2p)]`` for ``rho = sqrt(xi^2 + eta^2)``, ``xi, eta`` iid N(0,1).

    ``rho^2 = xi^2 + eta^2``, so ``E[rho^(2p)] = E[(xi^2 + eta^2)^p]`` expands
    binomially into products of even absolute moments, all rational. Exact.

    >>> rayleigh_even_moment(1), rayleigh_even_moment(2)
    (Fraction(2, 1), Fraction(8, 1))
    """
    if p < 0:
        raise ValueError("p must be a nonnegative integer")
    total = F(0)
    for j in range(p + 1):
        mj, pj = abs_moment(2 * j)
        mk, pk = abs_moment(2 * (p - j))
        assert pj == 0 and pk == 0, "even absolute moments are rational"
        total += math.comb(p, j) * mj * mk
    return total


def sibling_rayleigh_fourth_moment() -> IdentityRecord:
    """``E rho^4 = 8`` -- asserted in the same register row. CONFIRMED exactly."""
    derived = rayleigh_even_moment(2)      # E|xi|^4 + 2 E|xi|^2 E|eta|^2 + E|eta|^4
    # Exact pointwise bracket rho^4 <= (|xi|+|eta|)^4 <= 4 rho^4 (from
    # rho <= |xi|+|eta| <= sqrt(2) rho) gives, in mean, 8 <= E(|xi|+|eta|)^4 <= 32.
    # Does that bracket separate the claimed value from the true one? Evaluate
    # both sides of the claim exactly using Archimedes' strict rational bounds
    # 223/71 < pi < 22/7, so no float decides the answer.
    pi_lo, pi_hi = F(223, 71), F(22, 7)
    a_claim, b_claim = CLAIMED_FALSE_IDENTITY.as_a_plus_b_over_pi()
    claim_lower_bound = a_claim + b_claim / pi_hi     # <= claimed value
    claim_upper_bound = a_claim + b_claim / pi_lo     # >= claimed value
    # The bracket "detects" the defect only if it can EXCLUDE the claimed value.
    detects = (claim_upper_bound < derived) or (claim_lower_bound > 4 * derived)
    return IdentityRecord(
        key="LPW-R05-RHO4",
        statement="E rho^4 = 8 for rho = sqrt(xi^2 + eta^2), xi, eta iid N(0,1)",
        source=("registers/json/lpw_fold_dispositions.json, 'R05 Rayleigh "
                "quantitative repair' row; echoed at docs/RESEARCH_MAP.md L221."),
        verdict="CONFIRMED_EXACT",
        claimed=F(8),
        derived=derived,
        discrepancy=derived - F(8),
        structural_cause="",
        notes=(
            "E[(xi^2+eta^2)^2] = E xi^4 + 2 E xi^2 E eta^2 + E eta^4 = 3 + 2 + 3 = 8.",
            "The same register row says 'E rho^4 = 8 is below old 12 + 16/pi "
            "budget'. That comparison is TRUE under the false value (8 < 17.09) "
            "and remains TRUE under the corrected value (8 < 22.19). The "
            "comparison therefore survives the correction.",
            "Surviving is not discharging: the row's disposition is unchanged "
            "by this module, and the R05 repair remains author-side with "
            "targeted review required.",
            "The exact pointwise bracket rho^4 <= (|xi|+|eta|)^4 <= 4 rho^4 "
            "gives 8 <= E(|xi|+|eta|)^4 <= 32. The FALSE value 12 + 16/pi also "
            "satisfies that bracket, so the bracket could not have detected the "
            f"defect. (bracket separates claimed from true: {detects})",
        ),
        does_not_establish=(
            "Does NOT validate the 6.238e-44 / 6.239e-44 headline chain, the "
            "Rayleigh amplitude lemma, the tails/profile modulus, or any "
            "conditional bound in the R05 repair.",
            "Does NOT grant the R05 repair any external approval; the register "
            "records it as author-side and not inheriting the prior PASS.",
        ),
    )


def sibling_truncated_second_moment() -> IdentityRecord:
    """``z0(b) = E[Q^2 1{Q<0}] = (b^2+2) Phi(b/sqrt2) + sqrt(2) b phi(b/sqrt2)``.

    Asserted in ``registers/json/closure_log.json`` (GP-CLS-017-B, EC-006
    planar ``z0(6/5)`` normalizer). Settled exactly by comparing coefficients
    in the basis ``{Phi(b/sqrt 2), phi(b/sqrt 2)}``.
    """
    b = F(6, 5)
    got_phi_big, got_phi_small = truncated_normal_moment(b, 2)
    want_phi_big = QSqrt2.of(b * b + 2)
    want_phi_small = QSqrt2.sqrt2() * QSqrt2.of(b)
    ok = (got_phi_big == want_phi_big) and (got_phi_small == want_phi_small)
    return IdentityRecord(
        key="GP-CLS-017-B-Z0",
        statement=("For Q ~ N(-b, 2):  z0(b) = E[Q^2 1{Q<0}] "
                   "= (b^2 + 2) Phi(b/sqrt2) + sqrt(2) b phi(b/sqrt2); "
                   "z0(6/5) ~ 3.230978535287005"),
        source="registers/json/closure_log.json, row GP-CLS-017-B (EC-006).",
        verdict="CONFIRMED_EXACT" if ok else "REFUTED",
        claimed=(want_phi_big, want_phi_small),
        derived=(got_phi_big, got_phi_small),
        discrepancy=(got_phi_big - want_phi_big, got_phi_small - want_phi_small),
        structural_cause="",
        notes=(
            "Derivation: Q = -b + sqrt(2) Z gives "
            "b^2 T_0 - 2 sqrt(2) b T_1 + 2 T_2 with T_1 = -phi(c), "
            "T_2 = Phi(c) - c phi(c), c = b/sqrt2. The phi coefficient is "
            "2 sqrt(2) b - 2c = 2 sqrt(2) b - sqrt(2) b = sqrt(2) b.",
            "Same shape as the refuted identity: the value is a sum in which "
            "two contributions to one basis coefficient must BOTH be carried. "
            "Here the corpus carried both. There it carried one.",
            "The decimal 3.230978535287005 is reproduced by the NON-CERTIFYING "
            "float oracle only; the exact verdict rests on the coefficients.",
        ),
        does_not_establish=(
            "Does NOT re-open, re-close or re-grade GP-CLS-017-B. Its status "
            "label is transcribed, not decided here.",
            "Does NOT certify the decimal 3.230978535287005 to any digit. "
            "Certifying that decimal needs enclosures for Phi and phi, which "
            "this module does not supply.",
            "Does NOT bear on the exact-torus object z0,L, which the corpus "
            "explicitly distinguishes from this planar constant.",
        ),
    )


def sibling_truncated_fourth_moment() -> IdentityRecord:
    """The ``30.5469700802916329...`` planar fourth-moment decimal at ``b = 6/5``.

    ``registers/json/transition_log.json`` (TR-P02-003 / GP-DER-071) asserts a
    "planar closed form 30.5469700802916329... at b = 6/5" and elsewhere names
    a "truncated-normal fourth-moment formula". The closed form itself is NOT
    written out in the register, so what is available to settle is the decimal
    plus an INFERRED identification of the object.
    """
    b = F(6, 5)
    big, small = truncated_normal_moment(b, 4)
    want_big = QSqrt2.of(b ** 4 + 12 * b * b + 12)
    want_small = QSqrt2.sqrt2() * QSqrt2.of(b) * QSqrt2.of(b * b + 10)
    form_ok = (big == want_big) and (small == want_small)
    return IdentityRecord(
        key="GP-DER-071-Z4",
        statement=("planar closed form 30.5469700802916329... at b = 6/5, "
                   "described as a truncated-normal fourth-moment value"),
        source=("registers/json/transition_log.json, TR-P02-003 "
                "(CL-AUD-047 / GP-DER-071)."),
        verdict="NOT_SETTLED_IDENTIFICATION_INFERRED",
        claimed="30.5469700802916329... (decimal only; no closed form in the register)",
        derived=(big, small),
        discrepancy=None,
        structural_cause="",
        notes=(
            "Derived here from the same recursion: for Q ~ N(-b, 2), "
            "E[Q^4 1{Q<0}] = (b^4 + 12 b^2 + 12) Phi(b/sqrt2) "
            "+ sqrt(2) b (b^2 + 10) phi(b/sqrt2). "
            f"Matches the closed-form shape: {form_ok}.",
            "At b = 6/5 the NON-CERTIFYING float oracle gives "
            "30.54697008029163, agreeing with the register decimal to the "
            "digits double precision resolves. THAT AGREEMENT IS NOT A PROOF "
            "and it is not a certified bound.",
            "The identification of the register's object with E[Q^4 1{Q<0}] "
            "under Q ~ N(-b, 2) is MY INFERENCE from the row's own wording "
            "('truncated-normal fourth-moment formula', 'planar', 'b = 6/5') "
            "plus the numeric agreement. The register does not state it. "
            "Verdict stays NOT_SETTLED for that reason alone.",
            "Settling it exactly requires the GP-DER-071 body, which is not in "
            "this repository; only the register row is.",
        ),
        does_not_establish=(
            "Does NOT confirm or refute GP-DER-071. No verdict is issued on it.",
            "Does NOT certify the decimal 30.5469700802916329 to any digit.",
            "Does NOT touch TR-P02-003's recorded status "
            "('CANDIDATE — V2 SUBSTANTIALLY ADVANCED / OUTSIDE-LINEAGE REVIEW "
            "OPEN'), which stands as written.",
        ),
    )


def sibling_records() -> List[IdentityRecord]:
    """Every sibling identity this audit could reach, with its verdict."""
    return [
        sibling_rayleigh_fourth_moment(),
        sibling_truncated_second_moment(),
        sibling_truncated_fourth_moment(),
    ]


#: Corpus items of the right shape that this module could NOT settle, and why.
#: Recorded so the negative result is explicit rather than an absence.
FOUND_NOT_SETTLED: Tuple[dict, ...] = (
    {
        "expression": "c_infinity = 2^(2/3) 3^(5/6) Gamma(1/6) / (54 pi^(3/2))",
        "source": "registers/json/dispatch_queue.json (frozen body 7,073 bytes, "
                  "SHA-256 54cedb1e...)",
        "why_not_settled": "A DEFINITION of a constant, not an asserted equality "
                           "between two independently computable quantities. "
                           "There is nothing to refute, and Gamma(1/6) is not "
                           "expressible in exact rational arithmetic.",
    },
    {
        "expression": "E[Delta^2] = 14.0881601505150626086629336253776145062199350569... < 15",
        "source": "registers/source/GP-REG-032_v1.2_export_2026-09-17.md L203",
        "why_not_settled": "A decimal with no closed form and no stated law for "
                           "Delta in the register row. Nothing exact to compare.",
    },
    {
        "expression": "sigma5^2 = 945, L6^2 = 11340",
        "source": "registers/json/transition_log.json, TR-P02-003 (CL-AUD-047)",
        "why_not_settled": "Values attached to a conditioned fifth-derivative "
                           "tail of a specific periodized field. Reproducing "
                           "them needs the field's covariance, which is in "
                           "GP-DER-046/071, not in this repository. "
                           "(945 = 9!! is suggestive of a Gaussian even moment, "
                           "but suggestive is not settled.)",
    },
    {
        "expression": "E[J|P] = (-b, 0, 0, 0), Cov = diag(2, 2, 2, 6); and the "
                      "nine-jet form E[J|P] = (-b, 0, 0, 0, -3b, 0, 0, 0, 3b)",
        "source": "registers/source/GP-REG-032_v1.2_export_2026-09-17.md L1560, L1595",
        "why_not_settled": "A specification of a conditional law, not an "
                           "asserted closed-form moment identity. Its first "
                           "coordinate is consistent with the Q ~ N(-b, 2) used "
                           "above, which is a coherence observation only.",
    },
    {
        "expression": "E[R^8] < infinity;  E[W_r^2] <= 16 M8 r^4;  "
                      "E[W_r] >= c_Z r^2;  E[W_r^2] <= C_W r^4",
        "source": "registers/json/closure_log.json, easy_closure_queue.json",
        "why_not_settled": "Inequalities with unspecified constants, not "
                           "closed-form identities. No exact comparison exists.",
    },
    {
        "expression": "E[Q_L^2 1{Q_L < 0}] in (0, infinity)  (exact-torus)",
        "source": "registers/json/closure_log.json, GP-CLS-P01G-20260730",
        "why_not_settled": "A positivity/finiteness statement about the "
                           "exact-torus object, which the corpus explicitly "
                           "distinguishes from the planar Q ~ N(-b, 2). No "
                           "closed form is asserted for it.",
    },
)


# --------------------------------------------------------------------------
# NON-CERTIFYING numeric oracle
# --------------------------------------------------------------------------

def numeric_abs_moment(k: int) -> float:
    """NON-CERTIFYING. ``E|X|^k`` via ``math.gamma``, in binary floating point.

    This is a cross-check oracle ONLY. It is not a bound, not an enclosure and
    not a certificate; it carries unquantified rounding error. Every exact
    verdict in this module is decided in :class:`Sym` or :class:`QSqrt2`
    arithmetic and never by this function.
    """
    return 2.0 ** (k / 2.0) * math.gamma((k + 1) / 2.0) / math.sqrt(math.pi)


def numeric_oracle(value: Sym) -> float:
    """NON-CERTIFYING. Float value of a :class:`Sym`, for display and cross-check.

    Not a certified bound. See :func:`numeric_abs_moment`.
    """
    s = math.sqrt(2.0 / math.pi)
    return sum(float(coeff) * s ** power for power, coeff in value.coeffs.items())


def oracle_report(max_k: int = 8) -> List[dict]:
    """NON-CERTIFYING cross-check of every exact value against ``math.gamma``.

    Returns per-``k`` records with the exact value, the independent float
    computed from ``math.gamma`` by direct binomial summation, and the absolute
    difference. Agreement here is evidence of no transcription slip. It is NOT
    a proof and NOT a certified bound; the proofs are the exact comparisons.
    """
    out: List[dict] = []
    for k in range(max_k + 1):
        exact = sum_abs_power_moment(2, k)
        oracle = sum(math.comb(k, j) * numeric_abs_moment(j) * numeric_abs_moment(k - j)
                     for j in range(k + 1))
        out.append({
            "k": k,
            "exact": exact,
            "exact_as_float_NON_CERTIFYING": numeric_oracle(exact),
            "gamma_oracle_NON_CERTIFYING": oracle,
            "abs_diff_NON_CERTIFYING": abs(numeric_oracle(exact) - oracle),
        })
    return out


if __name__ == "__main__":
    rec = refutation_record()
    print("REFUTATION  ", rec.key, "--", rec.verdict)
    print("  claimed    ", rec.claimed, " = a + b/pi with (a, b) =",
          rec.claimed.as_a_plus_b_over_pi())
    print("  true       ", rec.derived, " = a + b/pi with (a, b) =",
          rec.derived.as_a_plus_b_over_pi())
    print("  discrepancy", rec.discrepancy, " = a + b/pi with (a, b) =",
          rec.discrepancy.as_a_plus_b_over_pi())
    print("  cause      ", rec.structural_cause)
    print()
    print("  binomial expansion of E(|xi| + |eta|)^4 (five terms):")
    for t in binomial_terms(4):
        print(f"    e={t['exponents']}  C={t['multinomial']}  -> {t['value']}")
    print()
    print("SIBLINGS")
    for s in sibling_records():
        print(f"  {s.key:18s} {s.verdict}")
    print()
    print("FOUND BUT NOT SETTLED:", len(FOUND_NOT_SETTLED), "items")
    print()
    print("NON-CERTIFYING oracle (math.gamma) -- not a bound:")
    for row in oracle_report(6):
        print(f"    k={row['k']}  exact~{row['exact_as_float_NON_CERTIFYING']:.12f}"
              f"  gamma~{row['gamma_oracle_NON_CERTIFYING']:.12f}"
              f"  |diff|={row['abs_diff_NON_CERTIFYING']:.3e}")
