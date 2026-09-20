"""Exact semantic challenges and corruption controls for LPW parent amplitudes.

No finite test here verifies infinite tails, conditional bounds, endpoint
topology/Taylor premises or review status. The historical repair is preserved.
"""
from copy import deepcopy
from fractions import Fraction as F
from math import factorial
import subprocess
import sys

import pytest

from research.interval import Interval
from research.lpw import amplitude as a
from tools.lpw_amplitude_check import CANDIDATE, ROOT, source_payloads


@pytest.fixture
def certificate():
    return a.load_bytes(CANDIDATE.read_bytes())


def check(certificate, payloads=None):
    return a.verify_bytes(a.canonical_bytes(certificate), source_payloads() if payloads is None else payloads)


def test_candidate_replays_source_bound_exact_identities(certificate):
    result = check(certificate)
    assert result['certificate_valid'] and result['source_bytes_match']
    assert result['l1_fourth'] == {'rational': '12', 'inverse_pi': '32'}
    assert result['rayleigh_fourth'] == '8'
    assert all(result[k] == value for k, value in a.SCOPE.items())
    assert a.canonical_bytes(certificate) == a.canonical_bytes(a.produce())


def test_independent_polar_derivation_matches_cartesian_binomial(certificate):
    # For the parent Gaussian density, U=rho^2/2 is Exp(1), so E rho^4=4*2!.
    # Uniform-angle averages: E|cos theta sin theta|=1/pi and
    # E cos^2(theta)sin^2(theta)=1/8. Thus E(cos_abs+sin_abs)^4=3/2+4/pi.
    radial_fourth = 4*factorial(2)
    angular_rational, angular_inverse_pi = F(1)+4*F(1, 8), F(4)
    assert radial_fourth == F(certificate['proof']['rayleigh_fourth'])
    assert radial_fourth*angular_rational == F(certificate['proof']['l1_fourth']['rational'])
    assert radial_fourth*angular_inverse_pi == F(certificate['proof']['l1_fourth']['inverse_pi'])
    assert certificate['proof']['l1_fourth_binomial_terms'] == [
        a.symbolic(3), a.symbolic(0, 16), a.symbolic(6), a.symbolic(0, 16), a.symbolic(3)]


def test_pointwise_squared_norms_for_signed_exact_inputs():
    for x in (F(-3), F(-1, 7), F(0), F(2, 5), F(7)):
        for y in (F(-5), F(-1, 3), F(0), F(4, 7), F(2)):
            rho2 = x*x+y*y
            l1sq = (abs(x)+abs(y))**2
            assert rho2 <= l1sq <= 2*rho2
            assert l1sq-rho2 == 2*abs(x*y)
            assert 2*rho2-l1sq == (abs(x)-abs(y))**2
    # Tightness witnesses for both squared constants, not a numerical search.
    assert (abs(F(1))+abs(F(0)))**2 == 1
    assert (abs(F(1))+abs(F(1)))**2 == 2*(1+1)


def test_phase_bound_independently_on_exact_unit_circle():
    # Rational Pythagorean parametrization avoids trigonometric floating point.
    for t in (F(-2), F(0), F(1, 3), F(1), F(3)):
        c, s = (1-t*t)/(1+t*t), 2*t/(1+t*t)
        assert c*c+s*s == 1
        for x, y in ((F(-2), F(3)), (F(1, 2), F(-1, 7)), (F(0), F(0))):
            assert x*x+y*y-(x*c+y*s)**2 == (x*s-y*c)**2 >= 0


def test_historical_budget_works_for_rayleigh_not_l1(certificate):
    p = Interval(*(F(x) for x in certificate['proof']['pi_enclosure']))
    assert 3 < p.lo <= p.hi < 4
    historical = Interval.exact(12)+16/p
    true_l1 = Interval.exact(12)+32/p
    assert 8 < historical.lo < historical.hi < true_l1.lo
    # First moments are nonnegative, so comparing their squares is valid.
    assert (8/p-p/2).lo > 0
    for bounds in certificate['proof']['strict_positive_comparison_enclosures'].values():
        assert 0 < F(bounds[0]) <= F(bounds[1])


def test_radius_arithmetic_is_conditional_on_imported_ceiling(certificate):
    r = certificate['proof']['radius']
    delta, k = F(r['delta']), F(r['imported_K_ceiling'])
    r0 = F(r['r0'])
    assert k == 9432 and r0 == F(1, 256*9432) == F(1, 2414592)
    assert r0 < F(1, 100000)
    assert 16*delta+8*k*r0 == F(3, 64) < F(1, 16)
    assert F(r['geometry_slack']) == F(1, 64)
    assert certificate['imported_premises']['K_ceiling_is_a_valid_conditional_bound'] == 'IMPORTED_NOT_VERIFIED'
    assert certificate['imported_premises']['endpoint_eigenfloor'] == '31/250'


@pytest.mark.parametrize('mutation', (
    lambda c: c['proof']['l1_fourth'].__setitem__('inverse_pi', '16'),
    lambda c: c['proof']['l1_fourth_binomial_terms'][1].__setitem__('inverse_pi', '8'),
    lambda c: c['proof'].__setitem__('rayleigh_fourth', '4'),
    lambda c: c['proof']['absolute_normal_moments'][3].__setitem__('coefficient', '1'),
    lambda c: c['proof']['pointwise']['l1_squared_minus_rho_squared'][0].__setitem__('coefficient', '-2'),
    lambda c: c['proof']['pointwise'].__setitem__('upper_square_linear_coefficients', ['1', '1']),
    lambda c: c['proof']['pointwise']['phase_gap_square'][0].__setitem__('coefficient', '2'),
    lambda c: c['proof'].__setitem__('pi_enclosure', ['3', '3']),
    lambda c: c['proof']['radius'].__setitem__('r0', '1/2414591'),
    lambda c: c['proof']['radius'].__setitem__('imported_K_ceiling', '9431'),
    lambda c: c['proof']['radius'].__setitem__('delta', '1/512'),
    lambda c: c['proof']['radius'].__setitem__('geometry_budget', '1/64'),
    lambda c: c['assumptions'].__setitem__('moments_are_conditional', True),
    lambda c: c['assumptions'].__setitem__('pair_law', 'arbitrary correlated Gaussians'),
    lambda c: c['assumptions'].__setitem__('field_representation', 'half-lattice with unchanged weights'),
    lambda c: c['imported_premises'].__setitem__('full_lattice_tails_and_profile_modulus', 'VERIFIED'),
    lambda c: c['scope'].__setitem__('review_gate_discharged', True),
    lambda c: c['scope'].__setitem__('independence_credit', False),
    lambda c: c['sources'][0].__setitem__('sha256', '0'*64),
    lambda c: c.__setitem__('schema', 'LPW_AMPLITUDE_CERTIFICATE_V2'),
    lambda c: c.__setitem__('execute', 'data is not executed'),
))
def test_adversarial_certificate_changes_refused(certificate, mutation):
    mutation(certificate)
    result = check(certificate)
    assert not result['certificate_valid'] and result['outcome'] == 'REJECTED'


def test_mutated_derivation_fails_against_fixed_candidate(certificate, monkeypatch):
    original = a.absolute_normal_moment
    monkeypatch.setattr(a, 'absolute_normal_moment', lambda n: (F(1), 1) if n == 3 else original(n))
    result = check(certificate)
    assert not result['certificate_valid'] and 'derivation mismatch' in result['reason']


def test_source_bytes_are_required_not_just_claimed_hashes(certificate):
    payloads = source_payloads()
    payloads[a.SOURCE['id']] += b'\n'
    assert not check(certificate, payloads)['certificate_valid']
    assert not check(certificate, {})['certificate_valid']


def test_producer_returned_data_cannot_mutate_checker_constants():
    generated = a.produce()
    generated['scope']['review_gate_discharged'] = True
    generated['sources'][0]['sha256'] = '0'*64
    assert a.SCOPE['review_gate_discharged'] is False
    assert a.SOURCE['sha256'] != '0'*64
    assert not check(generated)['certificate_valid']


@pytest.mark.parametrize('data', (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1.0}',
                                 b'[]', b'{} trailing', b'x'*131073, b'\xff'))
def test_malformed_data_fails_closed(data):
    assert not a.verify_bytes(data, source_payloads())['certificate_valid']


def test_normal_optimized_cli_and_actual_override_negative_control(certificate, tmp_path):
    script = str(ROOT/'tools/lpw_amplitude_check.py')
    outputs = []
    bad = deepcopy(certificate)
    bad['proof']['l1_fourth']['inverse_pi'] = '16'
    path = tmp_path/'bad.json'
    path.write_bytes(a.canonical_bytes(bad))
    for flags in ([], ['-O']):
        good = subprocess.run([sys.executable, *flags, script], text=True, capture_output=True)
        assert good.returncode == 0, good.stdout+good.stderr
        outputs.append(good.stdout)
        mutation = subprocess.run([sys.executable, *flags, script, '--certificate', str(path)], text=True, capture_output=True)
        assert mutation.returncode != 0 and 'derivation mismatch' in mutation.stdout
    assert outputs[0] == outputs[1]
