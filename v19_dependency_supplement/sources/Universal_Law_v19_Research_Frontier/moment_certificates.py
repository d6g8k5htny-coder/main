"""Exact generic Gaussian determinant moments; no actual-field alignment claim.

The generator expands a sparse polynomial and enumerates Gaussian Wick pairings.
The separate checker uses binomial coefficients and univariate moment recurrence.
Only the fixed reviewed law/grammar is supported; submitted code is never accepted.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json

SCHEMA = 'ul-v19-gaussian-moment-1'
SCOPE = 'generic_independent_gaussian_determinant_variance_family'


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _multiply(left, right):
    result = {}
    for a, ca in left.items():
        for b, cb in right.items():
            powers = tuple(x + y for x, y in zip(a, b))
            result[powers] = result.get(powers, 0) + ca * cb
    return {k: v for k, v in result.items() if v}


@lru_cache(maxsize=128)
def _wick(labels):
    """Polynomial in s by recursive covariance pairing, not double factorials."""
    if not labels:
        return ((0, 1),)
    total = {}
    first = labels[0]
    for index in range(1, len(labels)):
        second = labels[index]
        # Diagonal covariance (1,1,s); cross-covariances are exactly zero.
        if first != second:
            continue
        shift = int(first == 2)
        remainder = labels[1:index] + labels[index + 1:]
        for power, coefficient in _wick(remainder):
            total[power + shift] = total.get(power + shift, 0) + coefficient
    return tuple(sorted(total.items()))


def make_certificate(case='variance_family'):
    """Produce the sole bounded grammar: X,Y~N(0,1), Z~N(0,s), 0≤s≤1."""
    if not isinstance(case, str) or case != 'variance_family':
        raise ValueError('unsupported case; only variance_family is reviewed')
    determinant = {(1, 1, 0): 1, (0, 0, 2): -1}
    expanded = {(0, 0, 0): 1}
    moments = {}
    for degree in range(1, 5):
        expanded = _multiply(expanded, determinant)
        if degree not in (2, 4):
            continue
        coefficients = [0] * (degree + 1)
        for powers, coefficient in expanded.items():
            labels = tuple(i for i, count in enumerate(powers) for _ in range(count))
            for power, weight in _wick(labels):
                coefficients[power] += coefficient * weight
        moments[str(degree)] = coefficients
    certificate = {
        'schema': SCHEMA, 'case': case, 'scope': SCOPE,
        'law': {'family': 'independent_centered_gaussian', 'variables': ['X', 'Y', 'Z'],
                'covariance': [[1, 0, 0], [0, 1, 0], [0, 0, 's']],
                'parameter_semantics': 's is variance'},
        'domain': {'parameter': 's', 'lower': [0, 1], 'upper': [1, 1]},
        'polynomial': {'variables': ['X', 'Y', 'Z'], 'terms': [
            {'coefficient': 1, 'powers': [1, 1, 0]},
            {'coefficient': -1, 'powers': [0, 0, 2]}]},
        'moments': moments, 'bounds': {degree: sum(coefficients) for degree, coefficients in moments.items()},
        'derivation': {'algorithm': 'sparse_polynomial_wick_pairings',
                       'pairing_counts': {str(k): dict(_wick((0,) * k)).get(0, 0) for k in (2, 4, 6, 8)},
                       'bound_rule': 'nonnegative_coefficients_on_unit_interval'},
        'actual_field_applicable': False, 'full_theorem_certificates': [],
    }
    certificate['certificate_sha256'] = _digest(certificate)
    return certificate


def verify_certificate(certificate):
    """Convenience export; all acceptance logic lives in the independent checker."""
    from moment_checker import verify_certificate as independent_verify
    return independent_verify(certificate)


if __name__ == '__main__':
    print(json.dumps(make_certificate(), indent=2))
