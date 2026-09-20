"""Exact interval Gaussian conditioning through positive-pivot LDL solves.

Every result encloses each fixed joint PSD Gaussian member of the supplied
blocks, whose covariance is constant in the affine mark. Positive interval
pivots certify the conditioning block for every symmetric box member. No
midpoint, matrix inverse or floating-point arithmetic is used.

The joint PSD premise and field/source identity are not established here.
Matching law/order labels are consistency checks, not provenance authentication.
This candidate closes no RN obligation, changes no scientific status, supplies
no spatial cover and earns zero organizational independence credit.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
import re

from research.interval import Interval

MAX_DIMENSION = 18
MAX_BITS = 16384
MAX_OPERATIONS = 200000
SCOPE = ("FOR_EVERY_FIXED_JOINT_PSD_MEMBER; covariance fixed in the mark; "
         "joint PSD nonemptiness and actual field applicability not established; "
         "no spatial cover, RN closure, scientific promotion or independence credit")


class ConditioningInconclusive(ValueError):
    """The bounded interval calculation cannot establish admissible pivots."""


class ConditioningResourceLimit(ConditioningInconclusive):
    """An arithmetic size/operation limit stopped the calculation."""


def _exact(value):
    if type(value) not in (int, F, str) or (isinstance(value, str) and len(value) > 4096):
        raise ValueError("exact int, Fraction or rational string required; floats/bools refused")
    if isinstance(value, str) and not re.fullmatch(r"-?[0-9]+(?:/[0-9]+)?", value):
        raise ValueError("rational strings must be integers or integer ratios, without exponent notation")
    result = F(value)
    if max(abs(result.numerator).bit_length(), result.denominator.bit_length()) > MAX_BITS:
        raise ConditioningResourceLimit("rational input bit limit exceeded")
    return result


class _Arithmetic:
    def __init__(self, round_bits):
        if type(round_bits) is not int or not 128 <= round_bits <= 1024:
            raise ValueError("round_bits must be an integer in 128..1024")
        self.round_bits = round_bits
        self.operations = 0

    def value(self, value):
        if isinstance(value, Interval):
            if type(value.lo) is not F or type(value.hi) is not F:
                raise ValueError("Interval endpoints must remain exact Fractions")
            result = Interval(value.lo, value.hi)
        else:
            result = Interval.exact(_exact(value))
        bits = max(max(abs(x.numerator).bit_length(), x.denominator.bit_length())
                   for x in (result.lo, result.hi))
        if bits > MAX_BITS:
            raise ConditioningResourceLimit("intermediate rational bit limit exceeded")
        # Preserve inexpensive exact rational calculations; otherwise spend
        # tightness only through the public, outward dyadic-rounding operation.
        if bits > self.round_bits:
            result = result.round_out(self.round_bits)
        if any(max(abs(x.numerator).bit_length(), x.denominator.bit_length()) > MAX_BITS
               for x in (result.lo, result.hi)):
            raise ConditioningResourceLimit("rounded rational bit limit exceeded")
        return result

    def operation(self, op, left, right):
        self.operations += 1
        if self.operations > MAX_OPERATIONS:
            raise ConditioningResourceLimit("conditioning operation limit exceeded")
        if op == "add":
            result = left + right
        elif op == "sub":
            result = left - right
        elif op == "mul":
            result = left * right
        else:
            result = left / right
        return self.value(result)

    def add(self, left, right):
        return self.operation("add", left, right)

    def sub(self, left, right):
        return self.operation("sub", left, right)

    def mul(self, left, right):
        return self.operation("mul", left, right)

    def div(self, left, right):
        return self.operation("div", left, right)

    def dot(self, left, right):
        if len(left) != len(right):
            raise ValueError("dot-product dimensions disagree")
        result = Interval.exact(0)
        for a, b in zip(left, right):
            result = self.add(result, self.mul(a, b))
        return result


def _vector(values, length, arithmetic, *, default_zero=False):
    if values is None and default_zero:
        return tuple(Interval.exact(0) for _ in range(length))
    if not isinstance(values, (list, tuple)) or len(values) != length:
        raise ValueError("vector dimensions disagree")
    return tuple(arithmetic.value(value) for value in values)


def _matrix(values, rows, columns, arithmetic, *, symmetric=False):
    if not isinstance(values, (list, tuple)) or len(values) != rows:
        raise ValueError("matrix dimensions disagree")
    result = tuple(_vector(row, columns, arithmetic) for row in values)
    if symmetric and any(result[i][j] != result[j][i] for i in range(rows) for j in range(i)):
        raise ValueError("symmetric interval entries must be identical")
    return result


def _dimension(matrix):
    if not isinstance(matrix, (list, tuple)) or not 1 <= len(matrix) <= MAX_DIMENSION:
        raise ValueError("matrix dimension must be in 1..18")
    return len(matrix)


@dataclass(frozen=True)
class LDLFactor:
    """Containing unit-lower L and strictly positive D; not an inverse."""
    lower: tuple
    diagonal: tuple
    round_bits: int

    @property
    def pivots(self):
        return self.diagonal

    def solve(self, rhs):
        arithmetic = _Arithmetic(self.round_bits)
        return _solve(self, _vector(rhs, len(self.diagonal), arithmetic), arithmetic)


def _factor(matrix, arithmetic, pivot_floor):
    n = len(matrix)
    lower = [[Interval.exact(int(i == j)) for j in range(n)] for i in range(n)]
    diagonal = []
    for j in range(n):
        pivot = matrix[j][j]
        for k in range(j):
            # Multiplication is an inclusion extension even when occurrences
            # share a value; no independence between repeated entries is used.
            term = arithmetic.mul(arithmetic.mul(lower[j][k], lower[j][k]), diagonal[k])
            pivot = arithmetic.sub(pivot, term)
        if pivot.lo <= pivot_floor:
            raise ConditioningInconclusive(f"pivot {j} does not have a certified lower bound above the pivot floor")
        diagonal.append(pivot)
        for i in range(j + 1, n):
            numerator = matrix[i][j]
            for k in range(j):
                term = arithmetic.mul(arithmetic.mul(lower[i][k], lower[j][k]), diagonal[k])
                numerator = arithmetic.sub(numerator, term)
            lower[i][j] = arithmetic.div(numerator, pivot)
    return LDLFactor(tuple(tuple(row) for row in lower), tuple(diagonal), arithmetic.round_bits)


def interval_ldlt(matrix, *, round_bits=192, pivot_floor=0):
    """Certify every symmetric member SPD, or refuse without a conclusion.

    Induction encloses each member's exact LDL recurrences. Every lower pivot
    being positive proves SPD by congruence with a positive diagonal matrix.
    A failed interval pivot may reflect excess width, not an actual singularity.
    ``pivot_floor`` is an optional additional exact, nonnegative admission floor.
    """
    arithmetic = _Arithmetic(round_bits)
    n = _dimension(matrix)
    floor = _exact(pivot_floor)
    if floor < 0:
        raise ValueError("pivot_floor must be nonnegative")
    return _factor(_matrix(matrix, n, n, arithmetic, symmetric=True), arithmetic, floor)


def _solve(factor, rhs, arithmetic):
    n = len(factor.diagonal)
    lower = factor.lower
    y = []
    for i in range(n):
        y.append(arithmetic.sub(rhs[i], arithmetic.dot(lower[i][:i], y)))
    z = [arithmetic.div(y[i], factor.diagonal[i]) for i in range(n)]
    result = [Interval.exact(0) for _ in range(n)]
    for i in reversed(range(n)):
        result[i] = arithmetic.sub(z[i], arithmetic.dot(
            tuple(lower[j][i] for j in range(i + 1, n)), result[i + 1:]))
    return tuple(result)


def _order(order, length):
    if (not isinstance(order, (list, tuple)) or len(order) != length or
            any(type(name) is not str or not name.strip() or len(name) > 128 for name in order)
            or len(set(order)) != length):
        raise ValueError("explicit unique coordinate labels of matching length required")
    return tuple(order)


@dataclass(frozen=True)
class ConditionedGaussian:
    intercept: tuple
    slope: tuple
    covariance: tuple
    gain: tuple
    factorization: LDLFactor
    conditioning_order: tuple
    target_order: tuple
    law_id: str
    mark_domain: Interval
    operations: int
    quantifier: str = "FIXED_JOINT_PSD_MEMBERS_ONLY"
    all_conditioning_matrices_spd: bool = True
    joint_psd_established: bool = False
    field_certified: bool = False
    scientific_status_changed: bool = False
    independence_credit: int = 0
    authority: str = "NONE"
    does_not_establish: str = SCOPE

    @property
    def pivots(self):
        return self.factorization.diagonal

    def mean(self, mark):
        arithmetic = _Arithmetic(self.factorization.round_bits)
        original = Interval(mark.lo, mark.hi) if isinstance(mark, Interval) else Interval.exact(_exact(mark))
        if original not in self.mark_domain:
            raise ValueError("mark lies outside the declared domain")
        mark = arithmetic.value(original)
        return tuple(arithmetic.add(a, arithmetic.mul(b, mark))
                     for a, b in zip(self.intercept, self.slope))


def condition_gaussian(conditioning_covariance, target_conditioning_covariance,
                       target_covariance, conditioning_intercept, conditioning_slope,
                       *, conditioning_order, target_order, covariance_law, mean_law,
                       conditioning_law, mark_domain, prior_conditioning_intercept=None,
                       prior_conditioning_slope=None, prior_target_intercept=None,
                       prior_target_slope=None, round_bits=192, pivot_floor=0):
    """Enclose Y | X=c0+c1*t for joint Gaussian (X,Y), n(X)+n(Y)<=18.

    A=Cov(X,X), C=Cov(Y,X), B=Cov(Y,Y). Rows/columns use the explicit
    conditioning_order and target_order. The three law labels must match.
    Prior means default to zero; optional means are affine in the same mark.
    Each fixed joint PSD member has covariance independent of t. The result is
    B-C A^-1 C^T and mY(t)+C A^-1(c(t)-mX(t)), computed using solves only.
    Labels do not establish that blocks came from the claimed physical field.
    """
    n, m = _dimension(conditioning_covariance), _dimension(target_covariance)
    if n + m > MAX_DIMENSION:
        raise ValueError("joint dimension must not exceed 18")
    conditioning_order = _order(conditioning_order, n)
    target_order = _order(target_order, m)
    if set(conditioning_order) & set(target_order):
        raise ValueError("conditioning and target coordinate labels must be disjoint")
    if (type(covariance_law) is not str or not covariance_law.strip() or len(covariance_law) > 1024
            or mean_law != covariance_law or conditioning_law != covariance_law):
        raise ValueError("mean, covariance and conditioning must identify one explicit common law")
    arithmetic = _Arithmetic(round_bits)
    if isinstance(mark_domain, Interval):
        domain = Interval(mark_domain.lo, mark_domain.hi)
    elif isinstance(mark_domain, (list, tuple)) and len(mark_domain) == 2:
        domain = Interval(*(_exact(value) for value in mark_domain))
    else:
        raise ValueError("explicit closed rational mark domain required")
    arithmetic.value(domain)  # check input size without widening the declared domain
    floor = _exact(pivot_floor)
    if floor < 0:
        raise ValueError("pivot_floor must be nonnegative")
    a = _matrix(conditioning_covariance, n, n, arithmetic, symmetric=True)
    c = _matrix(target_conditioning_covariance, m, n, arithmetic)
    b = _matrix(target_covariance, m, m, arithmetic, symmetric=True)
    c0 = _vector(conditioning_intercept, n, arithmetic)
    c1 = _vector(conditioning_slope, n, arithmetic)
    mx0 = _vector(prior_conditioning_intercept, n, arithmetic, default_zero=True)
    mx1 = _vector(prior_conditioning_slope, n, arithmetic, default_zero=True)
    my0 = _vector(prior_target_intercept, m, arithmetic, default_zero=True)
    my1 = _vector(prior_target_slope, m, arithmetic, default_zero=True)
    factor = _factor(a, arithmetic, floor)
    gain = tuple(_solve(factor, row, arithmetic) for row in c)
    delta0 = tuple(arithmetic.sub(x, y) for x, y in zip(c0, mx0))
    delta1 = tuple(arithmetic.sub(x, y) for x, y in zip(c1, mx1))
    intercept = tuple(arithmetic.add(my0[i], arithmetic.dot(gain[i], delta0)) for i in range(m))
    slope = tuple(arithmetic.add(my1[i], arithmetic.dot(gain[i], delta1)) for i in range(m))
    covariance = [[Interval.exact(0) for _ in range(m)] for _ in range(m)]
    for i in range(m):
        for j in range(i + 1):
            left = arithmetic.sub(b[i][j], arithmetic.dot(gain[i], c[j]))
            right = arithmetic.sub(b[j][i], arithmetic.dot(gain[j], c[i]))
            entry = left.intersect(right)
            if entry is None:
                raise ConditioningInconclusive("symmetric Schur enclosures have empty intersection")
            if i == j and entry.hi < 0:
                raise ConditioningInconclusive("negative conditional variance excludes joint PSD members")
            covariance[i][j] = covariance[j][i] = entry
    return ConditionedGaussian(intercept, slope, tuple(tuple(row) for row in covariance), gain,
                               factor, conditioning_order, target_order, covariance_law,
                               domain, arithmetic.operations)
