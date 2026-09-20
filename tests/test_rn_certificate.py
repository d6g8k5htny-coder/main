"""Independent semantic challenges and hostile data-only certificate controls.

These tests do not establish an RN field law, scientific promotion or an
independent review. Exact member checks challenge code; interval inclusion and
Bernstein partition-of-unity provide the uniform mathematical argument.
"""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as F
import json
import subprocess
import sys

import pytest

from research.interval import Interval as I
from research.rn.certificate import (FLAGS, MAX_BYTES, ResourceLimit, canonical_bytes,
                                     load_bytes, produce, verify_bytes)
from research.rn.gaussian_families import FamilyMomentEngine
from test_gaussian_moments import independent_expansion
from tools.rn_certificate import CANDIDATES, PILOTS, ROOT, SOURCE, context_data, pilot


@pytest.fixture
def certificate():
    return load_bytes((CANDIDATES/'affine4_certificate_v1.json').read_bytes())


def check(certificate, **kwargs):
    return verify_bytes(canonical_bytes(certificate), **kwargs)


@pytest.mark.parametrize('name', PILOTS)
def test_committed_certificate_replay_and_binding(name):
    law, degree = pilot(name)
    data = (CANDIDATES/f'{name}_certificate_v1.json').read_bytes()
    result = verify_bytes(data, source_bytes=SOURCE.read_bytes(), expected_context=context_data(law))
    assert result['certificate_valid'] and result['requested_checks_passed']
    assert result['source_bytes_match'] is result['expected_context_match'] is True
    assert {k: result[k] for k in FLAGS} == FLAGS
    assert result['certified_lower_bound_on_slack'] == '0'
    assert canonical_bytes(load_bytes(data)) == canonical_bytes(produce(law, degree=degree))


@pytest.mark.parametrize('degree,cap', ((2, 4), (4, 132)))
def test_variance_family_independent_closed_form_and_attained_endpoint(degree, cap):
    law, _ = pilot('variance'+str(degree))
    cert = produce(law, degree=degree)
    assert F(cert['proof']['upper']) == cap
    # Direct independent-normal expansion, separate from either Stein replay.
    for root in (F(0), F(1, 2), F(1)):
        variance = root**2
        direct = independent_expansion((0, 0, 0), ((1, 0, 0), (0, 1, 0), (0, 0, root)), degree)
        closed = 1+3*variance**2 if degree == 2 else 9+18*variance**2+105*variance**4
        assert direct == closed <= cap
    assert direct == cap  # variance one attains the cap for this family only.


@pytest.mark.parametrize('degree', (2, 4))
def test_fixed_correlated_affine_law_against_independent_expansion(degree):
    law, _ = pilot('affine4')
    lower = ((F(1), 0, 0), (F(1, 3), F(2, 3), 0), (F(-1, 2), F(1, 4), F(3, 4)))
    cov = tuple(tuple(sum(lower[i][k]*lower[j][k] for k in range(3)) for j in range(3)) for i in range(3))
    law = replace(law, intercept=(F(2), F(-1), F(1, 3)), slope=(F(-1), F(1, 2), F(2, 5)), covariance=cov)
    cert = produce(law, degree=degree)
    coefficients = cert['proof']['coefficients']
    assert all(lo == hi for lo, hi in coefficients)
    for t in (F(-1), F(-1, 3), F(0), F(2, 3), F(1)):
        m = tuple(a+b*t for a, b in zip(law.intercept, law.slope))
        direct = independent_expansion(m, lower, degree)
        assert direct == sum(F(c[0])*t**j for j, c in enumerate(coefficients))
        assert direct <= F(cert['proof']['upper'])


def test_uncertain_affine_family_contains_exact_independent_members(certificate):
    lower = ((F(1), 0, 0), (F(1, 8), F(1), 0), (0, 0, F(1, 2)))
    for a in (F(-1), F(-3, 4)):
        for b in (F(3, 4), F(1)):
            for t in (F(-1), F(0), F(1, 3), F(1)):
                direct = independent_expansion((a+b*t, -1-t, 0), lower, 4)
                assert direct <= F(certificate['proof']['upper'])


def test_interior_maximum_singular_and_zero_width_cases():
    law, _ = pilot('affine4')
    law = replace(law, intercept=(-1, -1, 0), slope=(1, -1, 0), covariance=((0, 0, 0),)*3)
    cert = produce(law, degree=4)
    assert cert['proof']['upper'] == '1'  # (1-t^2)^4; endpoint-only bound would be zero.
    law = replace(law, context=replace(law.context, domain=(F(0), F(0))))
    cert = produce(law, degree=4, depth=6)
    assert len(cert['proof']['leaves']) == 1
    assert check(cert)['certificate_valid']


def test_replay_does_not_call_producer_moment_methods(certificate, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('checker called the producer recurrence')
    monkeypatch.setattr(FamilyMomentEngine, 'determinant', forbidden)
    monkeypatch.setattr(FamilyMomentEngine, 'moment', forbidden)
    assert check(certificate)['certificate_valid']


@pytest.mark.parametrize('mutation', (
    lambda c: c['claim']['context'].__setitem__('dimension', 3),
    lambda c: c['claim']['context'].__setitem__('order', ['xx', 'xy', 'yy']),
    lambda c: c['claim']['context'].__setitem__('normalization', ''),
    lambda c: c['claim']['context'].__setitem__('source_sha256', '0'*64),
    lambda c: c['claim']['context'].__setitem__('source_id', 'stale'),
    lambda c: c['claim']['context'].__setitem__('evidence_tier', 'FIELD_CERTIFIED'),
    lambda c: c['claim'].__setitem__('mean_law', 'marginal-law'),
    lambda c: c['claim'].__setitem__('covariance_law', 'other-conditional-law'),
    lambda c: c['claim']['intercept'][0].__setitem__(0, '-2'),
    lambda c: c['claim']['covariance'][0][1].__setitem__(0, '0'),
    lambda c: c['proof']['coefficients'][0].__setitem__(1, '999999'),
    lambda c: c['proof']['feasibility'].__setitem__('quantifier', 'PSD_MEMBERS_ONLY'),
    lambda c: c['proof']['feasibility'].__setitem__('psd_witness', None),
    lambda c: c['proof']['leaves'].pop(),
    lambda c: c['proof']['leaves'].pop(0),
    lambda c: c['proof']['leaves'].reverse(),
    lambda c: c['proof']['leaves'].append(deepcopy(c['proof']['leaves'][-1])),
    lambda c: c['proof']['leaves'][1].__setitem__('domain', ['-1', '0']),
    lambda c: c['proof']['leaves'][0]['bernstein'][0].__setitem__(1, '999999'),
    lambda c: c['proof'].__setitem__('upper', '-1'),
    lambda c: c['scope'].__setitem__('field_certified', True),
    lambda c: c['scope'].__setitem__('independence_credit', False),
    lambda c: c.__setitem__('schema', 'RN_GAUSSIAN_MOMENT_CERTIFICATE_V2'),
    lambda c: c.__setitem__('execute', 'arbitrary code is data and is rejected'),
))
def test_adversarial_certificate_corruptions_reject(certificate, mutation):
    mutation(certificate)
    result = check(certificate)
    assert not result['certificate_valid'] and result['outcome'] == 'REJECTED'


def test_slack_is_sufficient_bound_not_claimed_actual_optimum(certificate):
    replay = F(certificate['proof']['upper'])
    certificate['proof']['upper'] = str(replay+1)
    result = check(certificate)
    assert result['certificate_valid'] and result['certified_lower_bound_on_slack'] == '1'
    certificate['proof']['upper'] = str(replay-F(1, 10))
    result = check(certificate)
    assert not result['certificate_valid']
    assert result['certified_lower_bound_on_slack'] == '-1/10'
    assert 'no actual counterexample' in result['reason']


def test_failed_sufficient_psd_test_is_not_a_counterexample():
    law, _ = pilot('variance2')
    s = tuple(tuple(I(1) if i == j else I(-1, 1) for j in range(3)) for i in range(3))
    cert = produce(replace(law, covariance=s), degree=2)
    f = cert['proof']['feasibility']
    assert f['quantifier'] == 'PSD_MEMBERS_ONLY' and f['nonempty_established']
    assert f['criterion'] == 'SUFFICIENT_TEST_INCONCLUSIVE'
    assert check(cert)['certificate_valid']
    f['quantifier'] = 'ALL_BOX_MEMBERS'
    assert not check(cert)['certificate_valid']
    s = ((I(-1, 1), 1, 0), (1, I(-1, 1), 0), (0, 0, 1))
    cert = produce(replace(law, covariance=s), degree=2)
    assert cert['proof']['feasibility']['nonempty_established'] is False
    assert check(cert)['certificate_valid']  # conditional quantifier, not emptiness.


def test_source_and_context_checks_are_separate_from_algebra(certificate):
    expected = deepcopy(certificate['claim']['context'])
    expected['normalization'] = 'different declared units'
    result = check(certificate, source_bytes=b'wrong bytes', expected_context=expected)
    assert result['certificate_valid'] and result['outcome'] == 'VALID'
    assert result['source_bytes_match'] is result['expected_context_match'] is False
    assert not result['requested_checks_passed']
    result = check(certificate)
    assert result['source_bytes_match'] is result['expected_context_match'] is None


def test_consistent_declaration_change_needs_external_expected_context(certificate):
    original = check(certificate)
    expected = deepcopy(certificate['claim']['context'])
    certificate['claim']['context']['conditioned_law'] = 'another-declared-law'
    certificate['claim']['mean_law'] = certificate['claim']['covariance_law'] = 'another-declared-law'
    result = check(certificate, expected_context=expected)
    assert result['certificate_valid'] and result['expected_context_match'] is False
    assert not result['requested_checks_passed']
    assert result['claim_sha256'] != original['claim_sha256']


def test_symmetric_covariance_change_cannot_reuse_old_coefficients(certificate):
    certificate['claim']['covariance'][0][1][0] = '0'
    certificate['claim']['covariance'][1][0][0] = '0'
    result = check(certificate)
    assert not result['certificate_valid'] and 'coefficient derivation mismatch' in result['reason']


@pytest.mark.parametrize('value', ('-0', '2/2', '1/0', '01', '1.5', 1, True))
def test_noncanonical_or_inexact_rationals_reject(certificate, value):
    certificate['claim']['intercept'][0][0] = value
    assert check(certificate)['outcome'] == 'REJECTED'


def test_producer_does_not_hide_float_or_bool_domains():
    law, _ = pilot('variance2')
    for value in (0.5, True):
        bad = replace(law, context=replace(law.context, domain=(value, F(1))))
        with pytest.raises(TypeError):
            produce(bad, degree=2)


@pytest.mark.parametrize('data', (b'{"schema":1,"schema":2}', b'{"x":NaN}',
                                 b'{"x":Infinity}', b'{"x":1.0}', b'{"x":1e9}',
                                 b'{} trailing', b'\xff'))
def test_hostile_json_rejected(data):
    result = verify_bytes(data)
    assert not result['certificate_valid'] and result['outcome'] == 'REJECTED'


def test_resource_limits_are_inconclusive_never_pass(certificate):
    data = canonical_bytes(certificate)
    cases = (verify_bytes(data, max_operations=0), verify_bytes(b' '*MAX_BYTES+b' '),
             verify_bytes(b'['*17+b'0'+b']'*17))
    certificate['claim']['intercept'][0][0] = '-'+('9'*97)
    cases += (check(certificate),)
    for result in cases:
        assert result['outcome'] == 'INCONCLUSIVE'
        assert result['certificate_valid'] is False
    with pytest.raises(ResourceLimit):
        produce(pilot('variance2')[0], degree=2, max_operations=0)


def test_cli_explicit_candidate_inventory_and_exit_codes(tmp_path):
    command = [sys.executable, str(ROOT/'tools/rn_certificate.py')]
    checked = subprocess.run(command+['check-candidates'], text=True, capture_output=True)
    assert checked.returncode == 0, checked.stdout+checked.stderr
    assert '3 exact replays' in checked.stdout
    path = CANDIDATES/'variance2_certificate_v1.json'
    limited = subprocess.run(command+['verify', str(path), '--max-operations', '0'], text=True, capture_output=True)
    assert limited.returncode == 2 and json.loads(limited.stdout)['outcome'] == 'INCONCLUSIVE'
    source = tmp_path/'wrong-source'
    source.write_bytes(b'wrong')
    mismatch = subprocess.run(command+['verify', str(path), '--source', str(source)], text=True, capture_output=True)
    assert mismatch.returncode == 1 and json.loads(mismatch.stdout)['certificate_valid']
