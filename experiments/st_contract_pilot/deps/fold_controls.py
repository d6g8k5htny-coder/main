"""Exact toy pushforward controls; no random-field admissibility or Lean claim.

If dI=C*r**alpha dr and ell=k*r**p, the density power is (alpha+1)/p-1.
The cubic fold alone supplies p, not alpha. These controls expose that gap.
"""
from fractions import Fraction as Q
import json


def _rational(value, name):
    if type(value) not in (int, Q):
        raise ValueError(name + ' must be an exact integer or Fraction')
    return Q(value)


def cumulative_exponent(radial_power, gap_order):
    alpha = _rational(radial_power, 'radial power')
    p = _rational(gap_order, 'gap order')
    if alpha <= -1 or p <= 0:
        raise ValueError('Integrable radial power > -1 and positive gap order required')
    return (alpha + 1) / p


def density_exponent(radial_power, gap_order):
    return cumulative_exponent(radial_power, gap_order) - 1


def fold_gap(half_separation):
    """Gap of x**3/3-h**2*x at critical points +/-h: 4*h**3/3.

    The actual separation is 2*h, hence gap=separation**3/6.
    Negative transverse quadratics produce a maximum and a merging-saddle
    candidate in any dimension; global elder status is a separate obligation.
    """
    h = _rational(half_separation, 'half separation')
    if h <= 0:
        raise ValueError('Positive half separation required')
    return Q(4, 3) * h**3


def _integer_root_exact(value, order):
    low, high = 0, value + 1
    while low + 1 < high:
        mid = (low + high) // 2
        if mid**order <= value:
            low = mid
        else:
            high = mid
    if low**order != value:
        raise ValueError('Endpoint root is not rational; exact control refused')
    return low


def _root_exact(value, order):
    return Q(_integer_root_exact(value.numerator, order),
             _integer_root_exact(value.denominator, order))


def lifetime_bin_mass(a, b, *, gap_coefficient=Q(1), radial_power=1,
                      gap_order=3, weight=Q(1)):
    """Exact integral of C*r**alpha dr over a<=k*r**p<b.

    This deliberately supports integer alpha,p and rational roots only.
    It refuses irrational endpoints instead of calling a float a certificate.
    """
    a, b = _rational(a, 'lower endpoint'), _rational(b, 'upper endpoint')
    k, c = _rational(gap_coefficient, 'gap coefficient'), _rational(weight, 'weight')
    if not 0 <= a < b or k <= 0 or c < 0:
        raise ValueError('Ordered nonnegative endpoints, positive k and nonnegative C required')
    if type(radial_power) is not int or radial_power < 0:
        raise ValueError('Nonnegative integer radial power required for exact mass')
    if type(gap_order) is not int or gap_order <= 0:
        raise ValueError('Positive integer gap order required for exact mass')
    lower, upper = _root_exact(a / k, gap_order), _root_exact(b / k, gap_order)
    return c * (upper**(radial_power + 1) - lower**(radial_power + 1)) / (radial_power + 1)


def report():
    cases = [{'name': 'cubic_' + str(a), 'gap_order': 3, 'radial_power': a,
              'density_exponent': str(density_exponent(a, 3)),
              'cumulative_exponent': str(cumulative_exponent(a, 3))}
             for a in (0, 1, 2, 5)]
    return {'scope': 'EXACT_TOY_PUSHFORWARD_ONLY', 'cases': cases,
            'fold_gap_at_half_separation_1_2': str(fold_gap(Q(1, 2))),
            'cubic_radial_r_dr_mass_1_8': str(lifetime_bin_mass(Q(1), Q(8))),
            'same_fold_different_sampling_exponents': True,
            'random_field_counterexample': False,
            'scientific_acceptance': False, 'lean_execution': False}


if __name__ == '__main__':
    print(json.dumps(report(), sort_keys=True, indent=2))
