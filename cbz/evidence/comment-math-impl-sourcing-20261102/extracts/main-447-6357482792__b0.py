#!/usr/bin/env python3
"""Exact-rational support for one fixed-rung determinant-payoff transfer.
Imported spectral/stability statements are premises, not proved by this file.
No project imports, numerical packages, network, or filesystem access.
"""
from collections import Counter
from fractions import Fraction as Q
from itertools import product
from math import prod
import json
import sys

COUNTS = Counter()


def check(condition, group):
    COUNTS[group] += 1
    if not condition:
        raise ValueError(group)


def gaussian_moment(n):
    if n % 2:
        return Q(0)
    return Q(prod(range(1, n, 2)))


def poly_mul(a, b):
    out = {}
    for ka, va in a.items():
        for kb, vb in b.items():
            key = tuple(x + y for x, y in zip(ka, kb))
            out[key] = out.get(key, Q(0)) + va * vb
    return {key: value for key, value in out.items() if value}


def norm_eighth(mean, loading):
    """Exact E||mean+B*xi||^8, xi independent standard normals (test models)."""
    dim = len(loading[0])
    zero = (0,) * dim
    quadratic = {}
    for m, row in zip(mean, loading):
        linear = {zero: Q(m)}
        for j, value in enumerate(row):
            key = tuple(int(k == j) for k in range(dim))
            linear[key] = Q(value)
        for key, value in poly_mul(linear, linear).items():
            quadratic[key] = quadratic.get(key, Q(0)) + value
    fourth = {zero: Q(1)}
    for _ in range(4):
        fourth = poly_mul(fourth, quadratic)
    return sum((value * prod(gaussian_moment(n) for n in key)
                for key, value in fourth.items()), Q(0))


def payoff(y, r):
    q_m, q_s, a_m, a_s, b_m, b_s = map(Q, y)
    return -(a_m*q_m-r*b_m*b_m)*(a_s*q_s-r*b_s*b_s)


def norm_squared(y):
    return sum((Q(x)**2 for x in y), Q(0))


def transfer_budget(trace_ceiling, pin_floor, pin_norm_squared, eta, shift_squared, dimension):
    if not (trace_ceiling > 0 and pin_floor > 0 and pin_norm_squared >= 0
            and 0 <= eta < 1 and shift_squared >= 0
            and type(dimension) is int and dimension > 0):
        raise ValueError('invalid_import_domain')
    mean_squared = trace_ceiling*pin_norm_squared/pin_floor
    # |p| <= ||y||^4/16; (u+v)^8 <= 128*(u^8+v^8).
    second_moment = Q(128, 256)*(mean_squared**4 + 105*trace_ceiling**4)
    log_divergence = (Q(dimension, 2)*eta**2/(1-eta**2)
                      + shift_squared/(1-eta))
    if log_divergence >= 1:
        raise ValueError('rational_shortcut_domain')
    chi_square = log_divergence/(1-log_divergence)
    return mean_squared, second_moment, log_divergence, chi_square


def main():
    rung = Q(1, 20)
    pin_norm = Q(6, 5)**2 + Q(1, 6)**2
    check(pin_norm == Q(1321, 900), 'pin_vector_identity')
    check(Q(128, 256) == Q(1, 2), 'moment_factor_identity')
    check(gaussian_moment(8) == 105, 'gaussian_eighth_identity')
    for n in range(2, 18, 2):
        check(gaussian_moment(n) == (n-1)*gaussian_moment(n-2), 'gaussian_recurrence')

    # Finite pointwise controls corroborate, but do not prove, the analytic envelope.
    for r, y in product((rung, Q(1, 3), Q(1, 2)), product((-1, 0, 1), repeat=6)):
        value = abs(payoff(y, r))
        norm2 = norm_squared(y)
        check(value <= norm2**2/16, 'payoff_grid')
    extremal = (1, 1, 1, 1, 0, 0)
    check(abs(payoff(extremal, rung)) == norm_squared(extremal)**2/16,
          'payoff_constant_attained')
    check(abs(payoff(extremal, rung)) > norm_squared(extremal)**2/32,
          'negative_smaller_payoff_constant')
    outside_domain = (0, 0, 0, 0, 1, 1)
    check(abs(payoff(outside_domain, Q(1))) > norm_squared(outside_domain)**2/16,
          'negative_missing_r_domain')

    # Scalar PSD block witnesses for the mean-energy estimate, not its matrix proof.
    for g, a, residual, v in product((Q(1, 7), Q(1), Q(3)),
                                    (Q(-2), Q(0), Q(3)),
                                    (Q(0), Q(1, 5), Q(4)),
                                    (Q(-3), Q(0), Q(6, 5))):
        c = a*a/g + residual
        mu = a*v/g
        check(c*v*v/g-mu*mu == residual*v*v/g >= 0, 'scalar_mean_energy')
    check((Q(2)*Q(3))**2 > Q(2)**2, 'negative_omitted_pin_energy')

    # Exact noncentral moments of correlated six-coordinate test models.
    models = [
        ([Q(0)]*6, [[1,0],[0,1],[1,1],[1,-1],[Q(1,2),0],[0,Q(1,3)]]),
        ([Q(i-2,3) for i in range(6)],
         [[Q(i+1,7),Q((-1)**i,5),Q(i%3,4)] for i in range(6)]),
        ([1,0,0,0,0,0], [[0,0] for _ in range(6)]),
        ([0]*6, [[1,0]]+[[0,0] for _ in range(5)]),
        ([Q(3,2)]*6, [[Q(i-j,4) for j in range(3)] for i in range(6)]),
    ]
    for mean, loading in models:
        moment = norm_eighth(mean, loading)
        trace = sum((Q(x)**2 for row in loading for x in row), Q(0))
        bound = 128*(norm_squared(mean)**4 + 105*trace**4)
        check(moment <= bound, 'correlated_noncentral_moment')
        centered = norm_eighth([Q(0)]*6, loading)
        check(centered <= 105*trace**4, 'centered_trace_moment')
    check(norm_eighth([0]*6, [[1]]+[[0] for _ in range(5)]) == 105,
          'rank_one_moment_attained')
    check(norm_eighth([0]*6, [[1]]+[[0] for _ in range(5)]) > 15,
          'negative_weakened_eighth_constant')
    check(norm_eighth([1,0,0,0,0,0], [[0] for _ in range(6)]) > 0,
          'negative_missing_mean_term')

    L, g = Q(2*10**9), Q(1,10**20)
    eta, b2 = Q(3,10**113), Q(4,10**263)
    mu2, m2, rr, divergence = transfer_budget(L,g,pin_norm,eta,b2,6)
    # Bind computed outputs through independently arranged exact identities.
    check(mu2*g == L*pin_norm, 'mean_energy_binding')
    check(2*m2 == (mu2*mu2)**2+gaussian_moment(8)*(L*L)**2, 'second_moment_binding')
    check(rr*(1-eta*eta)*(1-eta) == 3*eta*eta*(1-eta)+b2*(1-eta*eta),
          'log_divergence_binding')
    check(divergence*(1-rr) == rr, 'divergence_binding')
    m2_ceiling, divergence_ceiling = Q(5*10**117), Q(3,10**225)
    error_ceiling = Q(4,10**54)
    check(mu2 < 3*10**29, 'mean_energy_ceiling')
    check(m2 < m2_ceiling, 'second_moment_ceiling')
    check(0 <= rr < 1, 'strict_rational_domain')
    check(divergence < divergence_ceiling, 'chi_square_ceiling')
    check(m2*divergence < m2_ceiling*divergence_ceiling < error_ceiling**2,
          'outward_error_square')
    check(m2_ceiling*divergence_ceiling > (error_ceiling/2)**2,
          'negative_halved_error_ceiling')
    check(transfer_budget(L,g,pin_norm,Q(0),Q(0),6)[3] == 0,
          'identical_law_control')
    check(transfer_budget(L,g,pin_norm,Q(0),Q(1,10),6)[3] == Q(1,9),
          'mean_shift_only_control')
    refused = []
    for label, args, expected in (
        ('zero_pin_floor', (L,Q(0),pin_norm,eta,b2,6), 'invalid_import_domain'),
        ('eta_one', (L,g,pin_norm,Q(1),b2,6), 'invalid_import_domain'),
        ('negative_shift', (L,g,pin_norm,eta,Q(-1),6), 'invalid_import_domain'),
        ('shortcut_large', (L,g,pin_norm,Q(0),Q(1),6), 'rational_shortcut_domain'),
        ('bool_dimension', (L,g,pin_norm,eta,b2,True), 'invalid_import_domain'),
    ):
        try:
            transfer_budget(*args)
        except ValueError as exc:
            check(str(exc) == expected, 'named_domain_refusals')
            refused.append(label)
        else:
            raise ValueError('missing_domain_refusal:'+label)
    rationals = {k:str(v) for k,v in (
        ('pin_norm_squared',pin_norm),('mean_squared_upper',mu2),
        ('second_moment_upper',m2),('log_one_plus_divergence_upper',rr),
        ('chi_square_upper',divergence),('error_squared_upper',m2*divergence))}
    result = {'status':'AUTHOR_SUPPORT_IMPORTED_PREMISES_NOT_REPROVED',
              'python':sys.version.split()[0], 'checks':sum(COUNTS.values()),
              'counts':dict(sorted(COUNTS.items())), 'rational_bounds':rationals,
              'simple_bounds':{'E_Q_f_squared':'<5e117','chi_square_P_relative_Q':'<3e-225',
                               'absolute_expectation_difference':'<4e-54'},
              'source':'3279db8bd8cf93905376eb2ff782e8afc3fdb544',
              'program_blob':'34a1a3cd17f7a78975e3ed824520ef8b84ff8742',
              'rung':'1/20','square_cutoff':116,'named_refusals':refused,
              'limits':['same literal frame and same measurable payoff/event',
                        'centered field laws, conditioned at the same exact pin vector',
                        'imports reviewed spectral and fixed-rung stability bounds',
                        'does not establish any finite-law expectation lower bound',
                        'does not identify archived rounded coordinates or binary rung',
                        'not uniform as separation tends to zero; no scientific acceptance']}
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__ == '__main__':
    main()
