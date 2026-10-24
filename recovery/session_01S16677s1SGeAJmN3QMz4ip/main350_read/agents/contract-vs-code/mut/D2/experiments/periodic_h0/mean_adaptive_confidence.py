"""Exact outward KL confidence bounds, conditional on the stated IID model.

MEAN_ADAPTIVE_CONFIDENCE.md proves the finite-sample statement. Arithmetic
does not authenticate a source, a fixed plan, or a supplied grid certificate.
The preexisting count_confidence Hoeffding interface is unchanged.
"""
from fractions import Fraction as Q
from functools import lru_cache

import finite_certificate as fc
import finite_count_loss as cl
from gaussian_tail import exp_neg_bounds


def _positive_integer(value):
    fc.require(type(value) is int and value > 0, 'Positive strict integer required')
    return value


def _precision(bits):
    fc.require(type(bits) is int and 1 <= bits <= 256, 'Integer precision from 1 through 256 required')
    return bits


def _positive_rational(value):
    fc.require(type(value) is Q and value > 0, 'Positive exact Fraction required')
    return value


def _unit(value):
    fc.require(type(value) is Q and 0 <= value <= 1, 'Exact Fraction in [0,1] required')
    return value


@lru_cache(maxsize=256)
def _reduced_log(value, tolerance):
    """For 1<=value<=2, sum positive atanh terms with a geometric remainder."""
    z = (value-1)/(value+1)
    square = z*z
    power, total, odd = z, Q(0), 1
    while True:
        total += 2*power/odd
        power *= square
        odd += 2
        remainder = 2*power/(odd*(1-square))
        if remainder <= tolerance:
            return total, total+remainder


def log_bounds(value, bits=80):
    """Enclose log(value) by exact Fractions with width at most 2**(-bits).

    Reciprocal symmetry and power-of-two reduction put the positive series
    parameter in [0,1/3]. Its geometric remainder guarantees termination.
    """
    _positive_rational(value)
    _precision(bits)
    if value < 1:
        lo, hi = log_bounds(1/value, bits)
        return -hi, -lo
    if value == 1:
        return Q(0), Q(0)
    exponent = value.numerator.bit_length()-value.denominator.bit_length()
    reduced = value/(1 << exponent)
    if reduced < 1:
        exponent -= 1
        reduced *= 2
    tolerance = Q(1, (1 << bits)*(exponent+1))
    lo, hi = _reduced_log(reduced, tolerance)
    if exponent:
        two_lo, two_hi = _reduced_log(Q(2), tolerance)
        lo += exponent*two_lo
        hi += exponent*two_hi
    return lo, hi


def kl_bounds(sample_mean, proposed_mean, bits=80):
    """Enclose binary kl(s||q), s in [0,1], q strictly between zero and one.

    Endpoint s terms use 0*log(0)=0. Infinite divergences at q=0 or q=1
    are excluded here; confidence_bounds handles its endpoints explicitly.
    """
    s, q = _unit(sample_mean), _unit(proposed_mean)
    _precision(bits)
    fc.require(0 < q < 1, 'Proposed mean must lie strictly between zero and one')
    if s == q:
        return Q(0), Q(0)
    lo = hi = Q(0)
    for weight, ratio in ((s, s/q), (1-s, (1-s)/(1-q))):
        if weight:
            a, b = log_bounds(ratio, bits)
            lo += weight*a
            hi += weight*b
    return max(Q(0), lo), hi


def confidence_bounds(sample_mean, sample_count, beta, bits=48):
    """Outward bounds for {q: n*kl(sample_mean||q)<=beta}.

    Each root gets at most ``bits`` bisections. An ambiguous log comparison
    returns the current conservative bracket endpoint, never a guessed side.
    Thus bits is a work/precision control, not an unconditional root-error
    guarantee. Endpoint cases reuse the certified exponential enclosure.
    """
    s = _unit(sample_mean)
    n = _positive_integer(sample_count)
    _positive_rational(beta)
    _precision(bits)
    if s in (0, 1):
        x = beta/n
        if x > 4096:
            return Q(0), Q(1)
        exponential_lower = exp_neg_bounds(x)[0]
        # 1-exp(-x)<=x also holds below the exponential evaluator's ulp.
        zero_upper = min(1-exponential_lower, x)
        return (Q(0), zero_upper) if s == 0 else (1-zero_upper, Q(1))
    log_bits = min(256, bits+16+n.bit_length())
    roots = []
    for lower_root in (True, False):
        left, right = (Q(0), s) if lower_root else (s, Q(1))
        for _ in range(bits):
            mid = (left+right)/2
            a, b = kl_bounds(s, mid, log_bits)
            if n*a >= beta:
                if lower_root:
                    left = mid
                else:
                    right = mid
            elif n*b <= beta:
                if lower_root:
                    right = mid
                else:
                    left = mid
            else:
                break
        roots.append(left if lower_root else right)
    return tuple(roots)


def risk_upper(bins, beta):
    """Bound min(1,2*m*exp(-beta)); m and beta belong to the fixed plan."""
    m = _positive_integer(bins)
    _positive_rational(beta)
    return min(Q(1), 2*m*exp_neg_bounds(min(beta, Q(4096)))[1])


def mean_interval(cutoff, failure, rows, beta, sample_count, bits=48):
    """Conditional finite-Gaussian mean interval from all retained bin rows.

    Each row must actually satisfy L<=X^-<=X^+<=U for the fixed latent
    observables. None contributes [0,C] to the unchanged planned denominator.
    This checks numeric domains, not independence or scientific provenance.
    """
    fc.require(type(cutoff) is int and 1 <= cutoff <= 64, 'Integer cutoff in [1,64] required')
    cap = cl.bar_cap(cutoff)
    n = _positive_integer(sample_count)
    _positive_rational(beta)
    _precision(bits)
    _unit(failure)
    fc.require(type(rows) is list and len(rows) == n, 'Every planned draw must have a row')
    low = high = 0
    for row in rows:
        if row is None:
            lo, hi = 0, cap
        else:
            fc.require(type(row) in (list, tuple) and len(row) == 2, 'A lower/upper count pair is required')
            lo, hi = row
            fc.require(type(lo) is int and type(hi) is int and 0 <= lo <= hi <= cap,
                       'Ordered strict integer count bounds within the cap required')
        low += lo
        high += hi
    lower = confidence_bounds(Q(low, n*cap), n, beta, bits)[0]*cap
    upper = confidence_bounds(Q(high, n*cap), n, beta, bits)[1]*cap
    return cl.expectation_interval(cutoff, failure, lower, upper)
