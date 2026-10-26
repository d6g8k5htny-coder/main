"""Independent closed-grammar exact checker for the Gaussian pilot family.

No generator routines are imported. Binomial expansion plus independence and the
univariate Gaussian recurrence E G^(2k)=(2k-1) variance E G^(2k-2) rederive each
coefficient. This is an exact symbolic check under the stated law, not a Lean
formalization of Gaussian probability or a proof of RN field applicability.
"""
from __future__ import annotations

import hashlib
import json
from math import comb

SCOPE = 'generic_independent_gaussian_determinant_variance_family'


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _normal_integer_moment(degree):
    if degree % 2:
        return 0
    result = 1
    for factor in range(1, degree, 2):
        result *= factor
    return result


def verify_certificate(certificate):
    """Return a fail-closed report; even correctly rehashed altered claims fail."""
    report = {'valid': False, 'state': 'invalid', 'errors': [], 'scope': SCOPE,
              'algorithm': 'binomial_independent_univariate_moments',
              'actual_field_applicable': False, 'full_theorem_certificates': []}
    try:
        if not isinstance(certificate, dict):
            raise ValueError('certificate must be a JSON object')
        encoded = _canonical(certificate)
        if len(encoded) > 32768:
            raise ValueError('certificate exceeds bounded grammar size')
        # Roundtrip eliminates custom containers and leaves exact JSON types.
        value = json.loads(encoded)
        claimed = value.pop('certificate_sha256', None)
        if claimed != hashlib.sha256(_canonical(value)).hexdigest():
            raise ValueError('certificate content digest mismatch')
        moments = {}
        for degree in (2, 4):
            coefficients = [0] * (degree + 1)
            # (XY-Z²)^degree = Σ C(degree,k)(XY)^(degree-k)(-Z²)^k.
            # Independence factors X, Y, Z moments; s appears with power k.
            for k in range(degree + 1):
                coefficients[k] = ((-1) ** k * comb(degree, k)
                    * _normal_integer_moment(degree - k) ** 2
                    * _normal_integer_moment(2 * k))
            if any(x < 0 for x in coefficients):
                raise ValueError('unit-interval bound rule not applicable')
            moments[str(degree)] = coefficients
        expected = {
            'schema': 'ul-v19-gaussian-moment-1', 'case': 'variance_family', 'scope': SCOPE,
            'law': {'family': 'independent_centered_gaussian', 'variables': ['X', 'Y', 'Z'],
                    'covariance': [[1, 0, 0], [0, 1, 0], [0, 0, 's']],
                    'parameter_semantics': 's is variance'},
            'domain': {'parameter': 's', 'lower': [0, 1], 'upper': [1, 1]},
            'polynomial': {'variables': ['X', 'Y', 'Z'], 'terms': [
                {'coefficient': 1, 'powers': [1, 1, 0]},
                {'coefficient': -1, 'powers': [0, 0, 2]}]},
            'moments': moments, 'bounds': {k: sum(v) for k, v in moments.items()},
            'derivation': {'algorithm': 'sparse_polynomial_wick_pairings',
                'pairing_counts': {str(k): _normal_integer_moment(k) for k in (2, 4, 6, 8)},
                'bound_rule': 'nonnegative_coefficients_on_unit_interval'},
            'actual_field_applicable': False, 'full_theorem_certificates': [],
        }
        # Compare canonical bytes: bool is never accepted in place of integer 1.
        if _canonical(value) != _canonical(expected):
            different = sorted(k for k in set(value) | set(expected)
                if _canonical(value.get(k)) != _canonical(expected.get(k)))
            raise ValueError('closed law/domain/claim grammar mismatch: ' + ', '.join(different))
        report.update(valid=True, state='generic_law_exact_checked', moments=moments,
                      bounds=expected['bounds'], certificate_sha256=claimed)
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        report['errors'].append(str(exc))
    return report
