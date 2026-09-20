"""Actual local-cover replay and hostile supplied-report controls.

The real reconstruction runs once for structural assertions. Mutation unit
cases use that cached expected report; CLI mutations independently reconstruct
the mathematics in normal and optimized subprocesses.
"""
from copy import deepcopy
from fractions import Fraction as F
import json
import subprocess
import sys
from zipfile import ZipFile

import pytest

from research.cover import Box, check_exact_partition
from research.rn.certificate import canonical_bytes
from tools import rn_side24_spatial_check as C


@pytest.fixture(scope='module')
def candidate():
    return C.point_check.load_report(C.ROOT/C.CANDIDATE)


@pytest.fixture(scope='module')
def reconstructed(candidate):
    # This invokes live source custody, all five cell attempts, four sets of
    # moment replays, exact partition accounting and the complete upper budget.
    return C.check_report(candidate)


def box(data):
    return Box(*(F(data[k]) for k in ('u0', 'u1', 'v0', 'v1')))


def test_committed_report_reconstructs_exactly(candidate, reconstructed):
    assert canonical_bytes(candidate) == canonical_bytes(reconstructed)
    assert reconstructed['scope'] == C.SCOPE
    assert reconstructed['scope']['local_rectangle_covered'] is True
    assert reconstructed['scope']['near_annulus_covered'] is False
    assert reconstructed['scope']['h3_floor_status'] == 'EXPLICIT_IMPORTED_HYPOTHESIS'
    assert reconstructed['scope']['authority'] == 'NONE'
    assert type(reconstructed['scope']['independence_credit']) is int
    assert reconstructed['scope']['independence_credit'] == 0


def test_retained_failed_parent_four_leaves_and_exact_area(reconstructed):
    result = reconstructed
    attempts = result['attempts']
    assert len(attempts) == 5 and attempts[0]['status'] == 'INCONCLUSIVE'
    assert 'pivot' in attempts[0]['reason'] and 'proof' not in attempts[0]
    assert box(attempts[0]['parameter_box']) == C.RECTANGLE
    leaves = [box(a['parameter_box']) for a in attempts[1:]]
    assert all(a['status'] == 'BOUNDED' for a in attempts[1:])
    check_exact_partition(C.RECTANGLE, leaves)
    assert len(set(leaves)) == 4
    assert sum((leaf.param_area() for leaf in leaves), F(0)) == F(result['domain_area']) == F(1, 1000000)
    assert all(leaf.param_area() == F(1, 4000000) for leaf in leaves)
    receipt = result['cover']
    assert receipt['cells_by_disposition'] == {'ACCEPTED': 4, 'PENDING': 0, 'REFINED': 1, 'REJECTED': 0}
    assert receipt['pending_count'] == 0 and receipt['pending_cells'] == []
    assert receipt['provisional_leaf_counts'] == {'leaves': 4, 'omitted': 0, 'summed': 4}
    assert len(receipt['enclosure_failures']) == 1
    assert receipt['enclosure_failures'][0]['disposition'] == 'REFINED'
    total = receipt['total']
    assert total['certified'] is total['covers_region'] is total['enclosure_is_of_the_region'] is True
    assert F(total['area_accounted_lo']) <= F(1, 1000000) <= F(total['area_accounted_hi'])
    assert total['area_unresolved_boundary_bound'] == '0'
    exact_upper_sum = sum((leaf.param_area()*F(a['proof']['integrand_upper'])
                           for leaf, a in zip(leaves, attempts[1:])), F(0))
    assert 0 < exact_upper_sum <= F(total['enclosure_hi']) <= C.LOCAL_BUDGET
    assert F(total['enclosure_lo']) == 0
    assert result['composition']['integral_range'] == [total['enclosure_lo'], total['enclosure_hi']]
    assert result['composition']['simple_rational_upper'] == str(C.LOCAL_BUDGET)


def test_each_leaf_has_source_bound_full_mark_witnesses(reconstructed):
    for attempt in reconstructed['attempts'][1:]:
        proof = attempt['proof']
        assert proof['box'] == attempt['cartesian_box'] == attempt['parameter_box']
        assert proof['integrand_range'] == ['0', proof['integrand_upper']]
        assert proof['quantifier'] == 'FOR_EVERY_Y_IN_CLOSED_RECTANGLE_AND_EVERY_MARK_IN_FULL_WINDOW'
        assert proof['source_binding']['source_sha256'] == C.point_check.SOURCES[C.point_check.SOURCE][1]
        assert proof['imported_h3_floor']['status'] == 'IMPORTED_SOURCE_PREMISE_NOT_REPROVED'
        assert proof['h3_reproved'] is proof['scientific_status_changed'] is False
        for name, degree in (('M', 4), ('S', 4), ('y', 2)):
            certificate = proof['moment_certificates'][name]
            assert certificate['claim']['degree'] == degree
            assert certificate['claim']['context']['domain'] == ['-1/96000', '1/96000']
            assert certificate['claim']['context']['order'] == ['xx', 'yy', 'xy']
            assert certificate['claim']['context']['conditioned_law'] == proof['conditioning_law']
            assert proof['moment_replays'][name]['certificate_valid'] is True
            assert proof['moment_replays'][name]['requested_checks_passed'] is True
            assert all(F(p[0]) > 0 for p in proof['marginal_pivots'][name])
    target = reconstructed['unresolved_near_target']
    assert target['target'] == '44201/20000000'
    assert 'not proved by this local cover' in target['status']


@pytest.mark.parametrize('field,value', (
    ('domain', Box.of(0, 1, 0, 1).as_json()), ('round_bits', True),
    ('schema', 'ANNULUS_COVER'),
    ('scope', {**C.SCOPE, 'near_annulus_covered': True}),
    ('scope', {**C.SCOPE, 'h3_floor_reproved_here': True}),
))
def test_false_scope_or_geometry_stops_before_replay(candidate, monkeypatch, field, value):
    changed = deepcopy(candidate)
    changed[field] = value
    monkeypatch.setattr(C, 'build_report', lambda *_: pytest.fail('invalid scope entered replay'))
    with pytest.raises(ValueError, match='scope, rectangle'):
        C.check_report(changed)


@pytest.mark.parametrize('mutation', (
    'leaf_escape', 'duplicate_leaf', 'missing_leaf', 'missing_parent_failure',
    'wrong_area', 'zero_total', 'wrong_source', 'wrong_moment', 'short_mark',
    'accepted_zero', 'false_pending_count', 'altered_l2_error',
))
def test_geometry_accounting_source_and_witness_mutations_reject(reconstructed, monkeypatch, mutation):
    changed = deepcopy(reconstructed)
    proof = changed['attempts'][1]['proof']
    if mutation == 'leaf_escape':
        changed['attempts'][1]['parameter_box']['u0'] = '0'
    elif mutation == 'duplicate_leaf':
        changed['attempts'][2] = deepcopy(changed['attempts'][1])
    elif mutation == 'missing_leaf':
        changed['attempts'].pop()
    elif mutation == 'missing_parent_failure':
        changed['attempts'].pop(0)
        changed['cover']['enclosure_failures'] = []
    elif mutation == 'wrong_area':
        changed['domain_area'] = '1/2000000'
    elif mutation == 'zero_total':
        changed['composition']['integral_range'] = ['0', '0']
        changed['cover']['total']['enclosure_hi'] = '0'
    elif mutation == 'wrong_source':
        changed['sources'][C.point_check.SOURCE]['sha256'] = '0'*64
    elif mutation == 'wrong_moment':
        proof['moment_certificates']['M']['proof']['coefficients'][0][1] = '0'
    elif mutation == 'short_mark':
        proof['moment_certificates']['y']['claim']['context']['domain'][1] = '0'
    elif mutation == 'accepted_zero':
        proof['integrand_upper'] = '0'
        proof['integrand_range'] = ['0', '0']
    elif mutation == 'false_pending_count':
        changed['cover']['pending_count'] = 1
    else:
        proof['l2_increment_upper'] = ['0']*12
    # Use only the already live-reconstructed expected report for these cheap
    # mutation checks; actual --check subprocess tests below do not patch it.
    monkeypatch.setattr(C, 'build_report', lambda *_: reconstructed)
    with pytest.raises(ValueError, match='reconstructed spatial cover'):
        C.check_report(changed)


@pytest.mark.parametrize('optimization', ([], ['-O']))
def test_cli_reconstructs_supplied_numerical_mutant(candidate, tmp_path, optimization):
    changed = deepcopy(candidate)
    changed['composition']['integral_range'] = ['0', '0']
    changed['cover']['total']['enclosure_hi'] = '0'
    changed['cover']['total']['enclosure_width'] = '0'
    path = tmp_path/'false-zero-integral.json'
    path.write_text(json.dumps(changed))
    completed = subprocess.run([sys.executable, *optimization,
        str(C.ROOT/'tools/rn_side24_spatial_check.py'), '--check', str(path)],
        capture_output=True, text=True, timeout=120)
    assert completed.returncode == 1
    assert 'reconstructed spatial cover' in completed.stdout


def test_target_member_custody_checks_actual_bytes(tmp_path):
    archive = tmp_path/C.point_check.ARCHIVE
    archive.parent.mkdir(parents=True)
    with ZipFile(archive, 'w') as z:
        z.writestr(C.TARGET_MEMBER, b'wrong target source')
    with pytest.raises(ValueError, match='source member identity'):
        C.checked_target(tmp_path)


def test_output_never_overwrites_existing_file(tmp_path, monkeypatch):
    path = tmp_path/'preserved.json'
    path.write_bytes(b'existing candidate')
    monkeypatch.setattr(C, 'build_report', lambda: {})
    assert C.main(['--output', str(path)]) == 1
    assert path.read_bytes() == b'existing candidate'
