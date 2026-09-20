"""Exact parent-Gaussian amplitude identities and conditional radius arithmetic.

Source-bound author-side companion to LPW R05, not a replay of its full
interval engine or an independent mathematical review. No tails, conditioning,
field-law identification, canonical status or headline mutation is established.
"""
from copy import deepcopy
from fractions import Fraction as F
import hashlib
import json
from math import comb

from research.interval import Interval, pi

SCHEMA = 'LPW_AMPLITUDE_CERTIFICATE_V1'
SOURCE = {'id': '1ngaO6hzeIXdCYWMwKl8JYPPZTeryGFTx', 'bytes': 9628,
          'sha256': '840c75a7825c67b8d99a536394beb18976475fe9bd0e474ed5f80c375004ec81'}
COMPANION = {'id': '1cdE15_2dd0VLHJnDGcFQ5RDQJhTkXfr-', 'bytes': 6843,
             'sha256': 'b37150f0eb79ff5d11b9e8b60f80afd94c677f8150aa701573ce24b337f24069'}
ASSUMPTIONS = {
    'pair_law': 'xi and eta are independent N(0,1) parent variables',
    'rho': 'sqrt(xi^2+eta^2)', 'l1': 'abs(xi)+abs(eta)',
    'moments_are_conditional': False,
    'field_representation': 'R05 full-lattice real Fourier representation; no half-lattice rescaling',
    'field_norm': 'max-convention C^p',
    'infinite_sum_justification': 'R05 Gaussian-decay/Tonelli/Minkowski argument imported, not replayed',
}
SCOPE = {'authority': 'NONE', 'independence_credit': 0,
         'scientific_status_changed': False, 'original_prize_closed': False,
         'field_law_identified': False, 'conditional_norm_bound_verified': False,
         'tails_profile_modulus_verified': False, 'headline_mutated': False,
         'review_gate_discharged': False}
IMPORTED = {'endpoint_eigenfloor': '31/250',
            'K_ceiling_is_a_valid_conditional_bound': 'IMPORTED_NOT_VERIFIED',
            'full_lattice_tails_and_profile_modulus': 'IMPORTED_NOT_VERIFIED',
            'conditional_density_and_regression_norms': 'IMPORTED_NOT_VERIFIED',
            'LPW_topology_Taylor_weight_proof': 'IMPORTED_NOT_VERIFIED',
            'review': 'R05 targeted review and existing statuses remain unchanged'}


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('ascii')


def load_bytes(data):
    if type(data) is not bytes or len(data) > 131072:
        raise ValueError('certificate must be bytes within the 131072-byte limit')

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result

    def reject(value):
        raise ValueError('noninteger JSON numeric literal is forbidden')

    def integer(value):
        if len(value) > 10:
            raise ValueError('JSON integer too large')
        return int(value)

    return json.loads(data, object_pairs_hook=pairs, parse_float=reject,
                      parse_constant=reject, parse_int=integer)


def _require(condition, reason):
    if condition is not True:
        raise ValueError(reason)


def _same(actual, expected, reason):
    _require(canonical_bytes(actual) == canonical_bytes(expected), reason)


def absolute_normal_moment(order):
    """(c, parity) denotes E|Z|^n = c*(sqrt(2/pi))^parity.

    The base Gaussian integral and integration by parts give m0=1,
    m1=sqrt(2/pi), and mn=(n-1)m(n-2). This routine replays that algebra.
    """
    if type(order) is not int or not 0 <= order <= 4:
        raise ValueError('moment order must be an integer in 0..4')
    coefficient = F(1)
    for factor in range(order-1, 0, -2):
        coefficient *= factor
    return coefficient, order % 2


def symbolic(rational=0, inverse_pi=0):
    return {'rational': str(F(rational)), 'inverse_pi': str(F(inverse_pi))}


def fourth_moment_terms():
    """Binomial terms for independent |xi| and |eta|, in Q + Q/pi."""
    terms = []
    for k in range(5):
        x, px = absolute_normal_moment(k)
        y, py = absolute_normal_moment(4-k)
        coefficient = comb(4, k)*x*y
        _require(px == py, 'fourth-moment parity mismatch')
        terms.append(symbolic(0, 2*coefficient) if px else symbolic(coefficient, 0))
    return terms


def _sum_symbols(terms):
    return symbolic(sum(F(t['rational']) for t in terms),
                    sum(F(t['inverse_pi']) for t in terms))


def _poly_add(a, b, scale=1):
    out = dict(a)
    for powers, value in b.items():
        out[powers] = out.get(powers, F(0))+scale*value
    return {powers: value for powers, value in out.items() if value}


def _poly_mul(a, b):
    out = {}
    for aa, x in a.items():
        for bb, y in b.items():
            powers = tuple(i+j for i, j in zip(aa, bb))
            out[powers] = out.get(powers, F(0))+x*y
    return {powers: value for powers, value in out.items() if value}


def _pack_poly(p):
    return [{'powers': list(powers), 'coefficient': str(value)}
            for powers, value in sorted(p.items())]


def pointwise_witnesses():
    # a=|xi|, b=|eta| are nonnegative. Squaring preserves order here.
    a, b = {(1, 0): F(1)}, {(0, 1): F(1)}
    rho2 = _poly_add(_poly_mul(a, a), _poly_mul(b, b))
    l1 = _poly_add(a, b)
    l1sq = _poly_mul(l1, l1)
    lower = _poly_add(l1sq, rho2, -1)
    upper = _poly_add({p: 2*x for p, x in rho2.items()}, l1sq, -1)
    square = _poly_mul(_poly_add(a, b, -1), _poly_add(a, b, -1))
    _require(lower == {(1, 1): F(2)}, 'lower norm monomial identity failed')
    _require(upper == square, 'upper norm square identity failed')
    # Polynomial Cauchy-Schwarz identity in variables (x,y,c,s).
    x, y, c, s = ({tuple(int(i == j) for i in range(4)): F(1)} for j in range(4))
    norm = _poly_add(_poly_mul(x, x), _poly_mul(y, y))
    phase = _poly_add(_poly_mul(c, c), _poly_mul(s, s))
    dot = _poly_add(_poly_mul(x, c), _poly_mul(y, s))
    cross = _poly_add(_poly_mul(x, s), _poly_mul(y, c), -1)
    gap = _poly_add(_poly_mul(norm, phase), _poly_mul(dot, dot), -1)
    _require(gap == _poly_mul(cross, cross), 'phase square identity failed')
    return {'nonnegative_variables': ['a=abs(xi)', 'b=abs(eta)'],
            'l1_squared_minus_rho_squared': _pack_poly(lower),
            'twice_rho_squared_minus_l1_squared': _pack_poly(upper),
            'upper_square_linear_coefficients': ['1', '-1'],
            'phase_variable_order': ['x', 'y', 'c', 's'],
            'phase_gap_square': _pack_poly(gap),
            'phase_assumption': 'c^2+s^2=1'}


def _enclose(value):
    return [str(value.lo), str(value.hi)]


def derive():
    """Reconstruct all finite algebra, keeping analytic imports explicit."""
    terms = fourth_moment_terms()
    l1 = _sum_symbols(terms)
    second, _ = absolute_normal_moment(2)
    fourth, _ = absolute_normal_moment(4)
    rho4 = fourth+2*second*second+fourth
    p = pi(12)
    _require(p.lo > 0 and p.hi < 4, 'positive pi<4 enclosure required')
    old_budget = Interval.exact(12)+16/p
    old_actual = Interval.exact(F(l1['rational']))+F(l1['inverse_pi'])/p
    comparisons = {
        'historical_fourth_budget_minus_rayleigh_fourth': old_budget-rho4,
        'l1_fourth_minus_historical_fourth_budget': 16/p,
        'l1_fourth_minus_rayleigh_fourth': old_actual-rho4,
        'four_times_rayleigh_fourth_minus_l1_fourth': 4*rho4-old_actual,
        'first_budget_squared_minus_rayleigh_mean_squared': 8/p-p/2,
    }
    for name, bound in comparisons.items():
        _require(bound.lo > 0, 'comparison was not strictly proved: '+name)
    delta, k, radius_domain = F(1, 1024), F(9432), F(1, 100000)
    r0 = F(1)/(256*k)
    hard = 16*delta+8*k*r0
    _require(r0 == F(1, 2414592) and r0 < radius_domain, 'radius arithmetic failed')
    _require(hard == F(3, 64) and hard < F(1, 16), 'geometry budget arithmetic failed')
    return {'absolute_normal_moments': [
                {'coefficient': str(absolute_normal_moment(n)[0]),
                 'sqrt_2_over_pi_power': absolute_normal_moment(n)[1]} for n in range(5)],
            'l1_fourth_binomial_terms': terms, 'l1_fourth': l1,
            'rayleigh_fourth': str(rho4), 'historical_fourth_budget': symbolic(12, 16),
            'rayleigh_mean_squared': {'pi_coefficient': '1/2'},
            'first_budget_squared': symbolic(0, 8),
            'pointwise': pointwise_witnesses(), 'pi_enclosure': _enclose(p),
            'strict_positive_comparison_enclosures': {key: _enclose(x) for key, x in comparisons.items()},
            'radius': {'delta': str(delta), 'imported_K_ceiling': str(k),
                       'covariance_domain_R': str(radius_domain), 'r0': str(r0),
                       'geometry_budget': str(hard), 'geometry_limit': '1/16',
                       'geometry_slack': str(F(1, 16)-hard)}}


def produce():
    return deepcopy({'schema': SCHEMA, 'sources': [SOURCE, COMPANION],
            'assumptions': ASSUMPTIONS, 'imported_premises': IMPORTED,
            'scope': SCOPE, 'proof': derive()})


def verify_bytes(data, source_payloads):
    """Require exact source custody and every fixed proof field; execute no data."""
    try:
        certificate = load_bytes(data)
        _require(type(certificate) is dict and set(certificate) ==
                 {'schema', 'sources', 'assumptions', 'imported_premises', 'scope', 'proof'},
                 'unknown or missing certificate fields')
        _require(certificate['schema'] == SCHEMA, 'unsupported schema/version')
        _same(certificate['sources'], [SOURCE, COMPANION], 'source identity metadata mismatch')
        _require(set(source_payloads) == {SOURCE['id'], COMPANION['id']}, 'required source payload missing/unknown')
        for source in (SOURCE, COMPANION):
            payload = source_payloads[source['id']]
            _require(type(payload) is bytes and len(payload) == source['bytes'] and
                     hashlib.sha256(payload).hexdigest() == source['sha256'], 'source byte identity mismatch')
        _same(certificate['assumptions'], ASSUMPTIONS, 'parent-law assumptions mismatch')
        _same(certificate['imported_premises'], IMPORTED, 'analytic imports or review boundary mismatch')
        _same(certificate['scope'], SCOPE, 'no-authority scope mismatch')
        expected = derive()
        proof = certificate['proof']
        _require(type(proof) is dict and set(proof) == set(expected), 'unknown or missing proof fields')
        for name, value in expected.items():
            _same(proof[name], value, 'exact derivation mismatch: '+name)
        return {**SCOPE, 'certificate_valid': True, 'source_bytes_match': True,
                'outcome': 'VALID_AT_PARENT_AMPLITUDE_AND_CONDITIONAL_RADIUS_SCOPE',
                'l1_fourth': expected['l1_fourth'], 'rayleigh_fourth': expected['rayleigh_fourth'],
                'certificate_sha256': hashlib.sha256(canonical_bytes(certificate)).hexdigest()}
    except (ValueError, TypeError, KeyError, ArithmeticError, RecursionError) as error:
        return {**SCOPE, 'certificate_valid': False, 'outcome': 'REJECTED', 'reason': str(error)}
