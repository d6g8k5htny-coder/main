"""SI01/SI02 regression controls. Synthetic statuses never mutate source data."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / 'tools/claims_check.py'
D1 = 'D1-v2.2(1)'
FLOOR = 'H3-RUNG-FLOOR'
CONSUMERS = ('RN3-FAR', 'RN5-NEAR-POINT-CERTS')
CERTIFYING_GRADES = ('CERTIFIED_RUNG', 'AUTHOR_SIDE_CERTIFIED', 'FROZEN_CERTIFICATE')
FLOOR_PATH = ('engine/rn_engine/frozen/K3_SIDE24_LB/UPPER2D/'
              'H3_closure/H3_RUNG_FLOOR.md')
FLOOR_SHA256 = '6347275d86c56842b719b36180e535bc1793d6995ddb68464db2960820440dfa'


def graph():
    return json.loads((ROOT / 'claims/graph.json').read_text(encoding='utf-8'))


def check(g, tmp_path, optimized, checker=CHECKER):
    path = tmp_path / 'candidate.json'
    path.write_text(json.dumps(g), encoding='utf-8')
    args = [sys.executable] + (['-O'] if optimized else [])
    return subprocess.run(args + [str(checker), '--graph', str(path)],
                          capture_output=True, text=True)


def rung_graph(grade='CERTIFIED_RUNG'):
    g = graph()
    g['claims'][D1]['grade'] = grade
    g['claims'][D1]['depends_on'] = ['H5-RIM']
    # This premise has no receipt-only evidence: positive controls isolate the
    # rung rule rather than failing an unrelated receipt-promotion firewall.
    return g


@pytest.mark.parametrize('optimized', [False, True])
@pytest.mark.parametrize('frozen,note', [
    ('NOT_CLOSED', 'OPEN'), ('CLOSED', 'OPEN'), ('OPEN', 'CLOSED'),
    ('NAMED_HYPOTHESIS', 'CLOSED'), ('CLOSED', 'RESTATED'),
    ('UNKNOWN', 'CLOSED'), ('CLOSED', 'UNKNOWN'),
    (None, 'CLOSED'), ('CLOSED', None), (False, 'CLOSED'),
    ('CLOSED', []), ('CLOSED', {'status': 'CLOSED'}),
    ([], 'CLOSED'), ({'status': 'CLOSED'}, 'CLOSED'),
    ('', 'CLOSED'), ('CLOSED', ''),
])
@pytest.mark.parametrize('grade', CERTIFYING_GRADES)
def test_certified_rung_refuses_each_undischarged_status(grade, tmp_path, optimized, frozen, note):
    g = rung_graph(grade)
    p = g['premises']['H5-RIM']
    for key, value in [('status_frozen_v2_2', frozen), ('status_register_note', note)]:
        if value is None:
            p.pop(key, None)
        else:
            p[key] = value
    r = check(g, tmp_path, optimized)
    assert r.returncode == 1, r.stdout + r.stderr
    assert 'FW-RUNG-OPEN-PREMISE:' in r.stdout
    assert D1 in r.stdout and 'H5-RIM' in r.stdout
    assert 'Traceback' not in r.stderr


@pytest.mark.parametrize('optimized', [False, True])
@pytest.mark.parametrize('grade', CERTIFYING_GRADES)
def test_certified_rung_accepts_discharged_synthetic_premise(grade, tmp_path, optimized):
    g = rung_graph(grade)
    for col in ('status_frozen_v2_2', 'status_register_note'):
        g['premises']['H5-RIM'][col] = 'CLOSED'
    r = check(g, tmp_path, optimized)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.parametrize('optimized', [False, True])
@pytest.mark.parametrize('grade', CERTIFYING_GRADES)
def test_certified_rung_refuses_transitive_open_premise(grade, tmp_path, optimized):
    g = rung_graph(grade)
    g['claims']['SYNTHETIC-INTERMEDIATE'] = {
        'track': 'UPPER2D', 'grade': 'CONDITIONAL', 'depends_on': ['H5-RIM']}
    g['claims'][D1]['depends_on'] = ['SYNTHETIC-INTERMEDIATE']
    r = check(g, tmp_path, optimized)
    assert r.returncode == 1, r.stdout + r.stderr
    assert 'FW-RUNG-OPEN-PREMISE:' in r.stdout and 'H5-RIM' in r.stdout


@pytest.mark.parametrize('optimized', [False, True])
def test_restoring_d1_historical_grade_is_rejected(tmp_path, optimized):
    g = graph()
    g['claims'][D1]['grade'] = 'CERTIFIED_RUNG'
    r = check(g, tmp_path, optimized)
    assert r.returncode == 1, r.stdout + r.stderr
    assert 'FW-RUNG-OPEN-PREMISE:' in r.stdout and 'D3-LEMMA-RN-UNIF' in r.stdout


def test_d1_operational_mapping_preserves_historical_grade_and_holds():
    node = graph()['claims'][D1]
    assert node['grade'] == 'CONDITIONAL'
    assert node['source_grade_verbatim'] == 'CERTIFIED_RUNG'
    assert node['audit_disposition'] == 'HOLD_WITH_DOMAIN'
    assert 'Q-RN5-MOMENT-003' in node['note']
    assert 'AMEND' in node['note']


def test_rn_consumers_use_the_exact_distinct_single_rung_floor():
    g = graph()
    assert FLOOR in g['claims']
    node = g['claims'][FLOOR]
    assert node['grade'] == 'AUTHOR_SIDE_CERTIFIED'
    assert node['source_grade_verbatim'] == 'CERTIFIED_RUNG'
    assert node['independence_credit'] == 0
    assert node['source_bindings'][0]['path'] == FLOOR_PATH
    assert node['source_bindings'][0]['expected_sha256'] == FLOOR_SHA256
    for consumer in CONSUMERS:
        assert FLOOR in g['claims'][consumer]['depends_on']
        assert 'H3-BAND-FLOOR' not in g['claims'][consumer]['depends_on']
    assert 'H3-BAND-FLOOR' in g['claims']


def floor_fixture(tmp_path):
    """Run the actual CLI with disposable repository bytes, never frozen inputs."""
    import shutil
    root = tmp_path / 'repository'
    checker = root / 'tools/claims_check.py'
    checker.parent.mkdir(parents=True)
    shutil.copyfile(CHECKER, checker)
    source = root / FLOOR_PATH
    source.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / FLOOR_PATH, source)
    g = graph()
    g['claims'][FLOOR] = {
        'track': 'UPPER2D', 'grade': 'AUTHOR_SIDE_CERTIFIED', 'depends_on': [],
        'source_bindings': [{
            'repo': 'd6g8k5htny-coder/main', 'path': FLOOR_PATH,
            'extraction_rule': 'whole_file', 'expected_bytes': 7003,
            'expected_sha256': FLOOR_SHA256,
        }],
    }
    for consumer in CONSUMERS:
        g['claims'][consumer]['depends_on'] = [FLOOR]
    return g, checker, source


@pytest.mark.parametrize('optimized', [False, True])
def test_matching_source_binding_passes_without_state_writes(tmp_path, optimized):
    g, checker, source = floor_fixture(tmp_path)
    before = source.read_bytes()
    r = check(g, tmp_path, optimized, checker)
    assert r.returncode == 0, r.stdout + r.stderr
    assert source.read_bytes() == before


@pytest.mark.parametrize('optimized', [False, True])
@pytest.mark.parametrize('mutation', ['alter', 'length', 'absent', 'symlink', 'parent_symlink'])
def test_source_binding_refuses_stale_rn_consumers(tmp_path, optimized, mutation):
    g, checker, source = floor_fixture(tmp_path)
    if mutation == 'alter':
        payload = source.read_bytes()
        source.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
    elif mutation == 'length':
        source.write_bytes(source.read_bytes() + b'\n')
    elif mutation == 'absent':
        source.unlink()
    elif mutation == 'symlink':
        target = tmp_path / 'outside.md'
        source.rename(target)
        source.symlink_to(target)
    elif mutation == 'parent_symlink':
        parent = source.parent
        target = tmp_path / 'outside-directory'
        parent.rename(target)
        parent.symlink_to(target, target_is_directory=True)
    r = check(g, tmp_path, optimized, checker)
    assert r.returncode == 1, r.stdout + r.stderr
    assert 'FW-RN-FLOOR-SOURCE:' in r.stdout
    assert all(name in r.stdout for name in (FLOOR, *CONSUMERS))
    if mutation == 'alter':
        assert 'source SHA-256 mismatch' in r.stdout
        assert 'source byte length mismatch' not in r.stdout
    elif mutation == 'length':
        assert 'source byte length mismatch' in r.stdout
    assert 'Traceback' not in r.stderr


@pytest.mark.parametrize('optimized', [False, True])
@pytest.mark.parametrize('mutation', [
    'wrong_edge', 'lost_node', 'lost_binding', 'unknown_repo', 'absolute_path',
    'parent_path', 'noncanonical_path', 'invalid_digest', 'invalid_bytes',
    'boolean_bytes', 'unsupported_extraction', 'coherent_repin',
])
def test_floor_contract_cannot_silently_lose_its_input(tmp_path, optimized, mutation):
    g, checker, source = floor_fixture(tmp_path)
    binding = g['claims'][FLOOR]['source_bindings'][0]
    if mutation == 'wrong_edge':
        g['claims'][CONSUMERS[0]]['depends_on'] = ['H3-BAND-FLOOR']
    elif mutation == 'lost_node':
        del g['claims'][FLOOR]
    elif mutation == 'lost_binding':
        del g['claims'][FLOOR]['source_bindings']
    elif mutation == 'unknown_repo':
        binding['repo'] = 'someone/else'
    elif mutation == 'absolute_path':
        binding['path'] = str(source)
    elif mutation == 'parent_path':
        binding['path'] = '../outside.md'
    elif mutation == 'noncanonical_path':
        binding['path'] = FLOOR_PATH.replace('/H3_closure/', '/./H3_closure/')
    elif mutation == 'invalid_digest':
        binding['expected_sha256'] = 'not-a-digest'
    elif mutation == 'invalid_bytes':
        binding['expected_bytes'] = 0
    elif mutation == 'boolean_bytes':
        binding['expected_bytes'] = True
    elif mutation == 'unsupported_extraction':
        binding['extraction_rule'] = 'frozen_body'
    elif mutation == 'coherent_repin':
        import hashlib
        source.write_bytes(source.read_bytes() + b'\n')
        binding['expected_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
        binding['expected_bytes'] = source.stat().st_size
    r = check(g, tmp_path, optimized, checker)
    assert r.returncode == 1, r.stdout + r.stderr
    assert ('FW-RN-FLOOR-SOURCE:' in r.stdout or
            'FW-RN-FLOOR-DEPENDENCY:' in r.stdout)
    assert 'Traceback' not in r.stderr


def test_floor_contract_matches_actual_frozen_runner_pin_and_parser():
    import ast
    import hashlib
    runner = ROOT / ('engine/rn_engine/frozen/K3_SIDE24_LB/UPPER2D/'
                     'D3_percolation/d3_rn_unif.py')
    payload = runner.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == (
        '85d7725fab42eeb0e823226f44d17f142a5c57e5d89084b2b6edffe4a8f0c930')
    tree = ast.parse(payload)
    pins = next(ast.literal_eval(n.value) for n in tree.body
                if isinstance(n, ast.Assign) and
                any(isinstance(t, ast.Name) and t.id == '_PINS' for t in n.targets))
    consumed = '../H3_closure/H3_RUNG_FLOOR.md'
    assert pins[consumed] == FLOOR_SHA256
    text = payload.decode('utf-8')
    assert "_floor_txt = open('../H3_closure/H3_RUNG_FLOOR.md').read()" in text
    assert "Z_LO = mpf(_m.group(1))" in text
    assert "ck(Z_LO == mpf('7.7592917375327855e-3')" in text
    source = (ROOT / FLOOR_PATH).read_bytes()
    assert len(source) == 7003
    assert hashlib.sha256(source).hexdigest() == FLOOR_SHA256


@pytest.mark.parametrize('optimized', [False, True])
def test_unrelated_document_change_does_not_invalidate_floor(tmp_path, optimized):
    g, checker, source = floor_fixture(tmp_path)
    doc = checker.parent.parent / 'README.md'
    doc.write_text('Unrelated navigation update.\n', encoding='utf-8')
    r = check(g, tmp_path, optimized, checker)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.parametrize('optimized', [False, True])
@pytest.mark.parametrize('grade', CERTIFYING_GRADES)
def test_certified_rung_refuses_empty_premise_record(grade, tmp_path, optimized):
    g = rung_graph(grade)
    g['premises']['H5-RIM'] = {}
    r = check(g, tmp_path, optimized)
    assert r.returncode == 1, r.stdout + r.stderr
    assert 'FW-RUNG-OPEN-PREMISE:' in r.stdout and 'H5-RIM' in r.stdout


@pytest.mark.parametrize('optimized', [False, True])
@pytest.mark.parametrize('grade', CERTIFYING_GRADES)
def test_d1_certifying_alias_refuses_actual_open_rn(tmp_path, optimized, grade):
    g = graph()
    g['claims'][D1]['grade'] = grade
    r = check(g, tmp_path, optimized)
    assert r.returncode == 1, r.stdout + r.stderr
    assert 'FW-RUNG-OPEN-PREMISE:' in r.stdout
    assert D1 in r.stdout and 'D3-LEMMA-RN-UNIF' in r.stdout
    assert f'is graded {grade}' in r.stdout


@pytest.mark.parametrize('optimized', [False, True])
@pytest.mark.parametrize('grade', ['MISSING', None, 'FUTURE_CERTIFIED', True, [], {}])
def test_d1_unknown_or_malformed_grade_is_not_a_bypass(tmp_path, optimized, grade):
    g = graph()
    if grade == 'MISSING':
        g['claims'][D1].pop('grade')
    else:
        g['claims'][D1]['grade'] = grade
    r = check(g, tmp_path, optimized)
    assert r.returncode == 1, r.stdout + r.stderr
    assert 'FW-RUNG-OPEN-PREMISE:' in r.stdout and D1 in r.stdout
    assert 'unknown or malformed operational grade' in r.stdout
    assert 'Traceback' not in r.stderr


@pytest.mark.parametrize('optimized', [False, True])
def test_source_labels_and_existing_author_side_scope_remain_compatible(tmp_path, optimized):
    g = graph()
    assert g['claims'][D1]['grade'] == 'CONDITIONAL'
    assert g['claims'][D1]['source_grade_verbatim'] == 'CERTIFIED_RUNG'
    assert g['claims']['RN3-FAR']['grade'] == 'AUTHOR_SIDE_CERTIFIED'
    assert g['claims'][FLOOR]['grade'] == 'AUTHOR_SIDE_CERTIFIED'
    assert g['claims']['P14-A..E']['grade'] == 'AUTHOR_SIDE_COMPLETE'
    r = check(g, tmp_path, optimized)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.parametrize('optimized', [False, True])
@pytest.mark.parametrize('grade', ['CONDITIONAL', 'AUTHOR_SIDE_PROOF_PRESENT', 'AMEND'])
@pytest.mark.parametrize('source_grade', CERTIFYING_GRADES + ('FUTURE_SOURCE_LABEL',))
def test_known_noncertifying_d1_grade_is_not_source_label_promotion(
        tmp_path, optimized, grade, source_grade):
    g = graph()
    g['claims'][D1]['grade'] = grade
    g['claims'][D1]['source_grade_verbatim'] = source_grade
    r = check(g, tmp_path, optimized)
    assert r.returncode == 0, r.stdout + r.stderr
