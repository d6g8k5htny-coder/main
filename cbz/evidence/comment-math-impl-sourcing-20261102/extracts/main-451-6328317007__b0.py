"""Exact finite arithmetic for PROOF.md; no quadrature or transcendental calls.

Run: python -I -B -S truncation_certificate.py
The analytic majorization is proved in PROOF.md, not by this program.
"""
from fractions import Fraction as F
from math import factorial
import hashlib
import json
from pathlib import Path

U = F(13, 10)
E_LOWER = F(1359, 500)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def taylor_exp(x, order):
    return sum((x**j / factorial(j) for j in range(order + 1)), F())


def tail_polynomial_constant(x):
    return F(8, 3)*x**3 + F(11, 3)*x**2 + F(22, 9)*x + F(22, 27)


def verify_claim(n, target_power, *, e_lower=E_LOWER,
                 tail_floor=180, max_derivative=4):
    """Verify a rational threshold for the fixed U and the stated real domain."""
    require(type(n) is int and n >= 1, 'invalid term count')
    require(type(target_power) is int and target_power > 0, 'invalid threshold')
    require(type(max_derivative) is int and 0 <= max_derivative <= 4,
            'unsupported derivative scope')
    require(isinstance(e_lower, F) and 1 < e_lower < taylor_exp(F(1), 6),
            'uncertified lower bound for e')
    require(type(tail_floor) is int and tail_floor >= 1 and
            tail_floor < taylor_exp(4*U, 30), 'uncertified tail substitution floor')
    pi_lower_witness = 4*sum((F((-1)**j, 2*j+1) for j in range(8)), F())
    require(pi_lower_witness > 3, 'pi lower witness failed')
    # Analytic witnesses for pi < 4 and e < 3 are supplied in PROOF.md.
    require(9*U < 12 and 5*U < 7 and U > 1, 'compact-interval exponents')
    q = 1/e_lower
    m = n + 1
    ratio = F(m+1, m)**4*q**(3*(2*m+1))
    require(0 < ratio < 1, 'kernel geometric ratio failed')
    series = U**5*(32*3**12 + 12*3**7)*m**4*q**(3*m*m)/(1-ratio)
    x = tail_floor
    tail_ratio = 4*q**(9*x)
    require(0 < tail_ratio < 1, 'integration-tail geometric ratio failed')
    tail = tail_polynomial_constant(x)*q**(3*x)/(1-tail_ratio)
    require(series + tail < F(1, 10**target_power), 'claimed threshold not established')
    return {'series': series, 'tail': tail,
            'series_ratio': ratio, 'tail_ratio': tail_ratio}


def main():
    rows = [verify_claim(12, 207), verify_claim(14, 227)]
    require(rows[0]['series'] < F(2, 10**208), '12-term component bound')
    require(rows[1]['series'] < F(3, 10**281), '14-term component bound')
    require(rows[0]['tail'] < F(6, 10**228), 'integration component bound')
    controls = (
        ('12-term exaggerated threshold', lambda: verify_claim(12, 209)),
        ('14-term exaggerated threshold', lambda: verify_claim(14, 228)),
        ('too few terms', lambda: verify_claim(11, 207)),
        ('false e lower bound', lambda: verify_claim(12, 207, e_lower=F(3))),
        ('false substitution floor', lambda: verify_claim(12, 207, tail_floor=182)),
        ('out-of-scope derivative', lambda: verify_claim(12, 207, max_derivative=5)),
    )
    refused = []
    for label, action in controls:
        try:
            action()
        except ValueError as exc:
            refused.append({'case': label, 'reason': str(exc)})
        else:
            raise ValueError('negative control accepted: ' + label)
    report = {
        'arithmetic': 'fractions.Fraction; exact integer comparisons',
        'cases': [{'N': 12, 'absolute_truncation_error_lt': '1e-207'},
                  {'N': 14, 'absolute_truncation_error_lt': '1e-227'}],
        'scope': {'U': '13/10', 'z': 'real', 't': 'real, t<=0', 'k': 'integer 0..4'},
        'covers_quadrature_or_roundoff': False,
        'covers_coefficient_ratios_or_root_location': False,
        'covers_adaptive_kernel_stopping_rule': False,
        'scientific_acceptance_changed': False,
        'negative_controls_rejected': len(refused),
        'negative_controls': refused,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
