"""Adversarial completion and source-inventory controls."""
import copy
import json
from pathlib import Path
import pytest
from tools import closure_pipeline as C


def minimal_plan():
    plan = {'sources': {'x': {'sha256': 'a', 'bytes': 1}},
            'tasks': [{'task_id': 'CHECK-001', 'ci_command': 'python check.py'}],
            'items': [{'item_key': 'claim/X', 'source_id': 'X', 'source_path': 'x',
                       'source_locator': '$', 'verification_tasks': ['CHECK-001'],
                       'record': {'status': 'OPEN', 'next_action': 'supply actual law'}}]}
    return bind_plan(plan)


def bind_plan(plan):
    import hashlib
    plan.pop('payload_sha256', None)
    plan['payload_sha256'] = hashlib.sha256(C.runner.canonical_bytes(plan)).hexdigest()
    return plan


def passed_report():
    snapshot = {'inputs': {'x': {'sha256': 'a', 'bytes': 1}}}
    return {'status': 'PASS', 'before': snapshot, 'after': copy.deepcopy(snapshot),
            'pytest': {'executed': 1}, 'steps': [{'ci_command': 'python check.py',
            'passed': True, 'returncode': 0, 'log_identity': {'sha256': 'log'}}]}


def test_completion_keeps_original_obligation_and_status():
    result = C.resolve(minimal_plan(), passed_report())
    assert result['status'] == 'PASS'
    assert result['machine_tasks_complete'] == 1
    assert result['scientific_items_promoted'] == 0
    candidate = result['resolution_candidates'][0]
    assert candidate['original_record'] == {'status': 'OPEN', 'next_action': 'supply actual law'}
    assert candidate['scientific_disposition'] == 'SOURCE_RECORD_UNCHANGED'


@pytest.mark.parametrize('mutation', ['missing', 'exit', 'failed', 'log', 'binding', 'drift', 'empty_tests', 'global_fail'])
def test_failed_or_unbound_run_never_closes(mutation):
    report = passed_report()
    if mutation == 'missing': report['steps'] = []
    elif mutation == 'exit': report['steps'][0]['returncode'] = 1
    elif mutation == 'failed': report['steps'][0]['passed'] = False
    elif mutation == 'log': report['steps'][0].pop('log_identity')
    elif mutation == 'binding': report['before']['inputs']['x']['sha256'] = report['after']['inputs']['x']['sha256'] = 'other'
    elif mutation == 'drift': report['after']['inputs']['x']['sha256'] = 'other'
    elif mutation == 'empty_tests': report['pytest']['executed'] = 0
    elif mutation == 'global_fail': report['status'] = 'FAIL'
    result = C.resolve(minimal_plan(), report)
    assert result['status'] == 'FAIL'
    assert result['scientific_items_promoted'] == 0


def test_duplicate_or_unplanned_execution_refused():
    for command in ['python check.py', 'python extra.py']:
        report = passed_report()
        report['steps'].append({**report['steps'][0], 'ci_command': command})
        with pytest.raises(ValueError): C.resolve(minimal_plan(), report)


def test_unmapped_item_preserves_terminal_status_without_reopening():
    plan = minimal_plan()
    plan['items'][0]['verification_tasks'] = []
    plan['items'][0]['record']['status'] = 'TERMINAL; package independence OPEN'
    result = C.resolve(bind_plan(plan), passed_report())
    item = result['resolution_candidates'][0]
    assert item['verification_status'] == 'UNMAPPED_EXACT_OBLIGATION'
    assert item['original_record']['status'] == 'TERMINAL; package independence OPEN'


def test_actual_inventory_preserves_rows_layers_and_superseded_input():
    plan = C.build_plan()
    assert len(plan['tasks']) == len(C.runner.REQUIRED_COMMANDS)
    for path in C.REGISTERS:
        source = json.loads((C.ROOT / path).read_text())
        assert len([item for item in plan['items'] if item['source_path'] == path]) == len(source['rows'])
    a5 = next(item for item in plan['items'] if item['item_key'] == 'lane/A5')
    assert 'CR-RNU-DS3-SCALAR-SUPERSEDED' in a5['record']['inputs']
    assert all('CR-RNU-DS3-SCALAR-SUPERSEDED' not in task['ci_command'] for task in plan['tasks'])
    terminal = next(item for item in plan['items'] if item['source_id'] == 'EC-005')
    assert terminal['record']['Queue state'].startswith('TERMINAL')
    node = next(item for item in plan['items'] if item['item_key'] == 'graph/premises/B4.loc-damline')
    assert node['record']['status_frozen_v2_2'] == 'OPEN'
    assert node['record']['status_register_note'] == 'CLOSED'


def test_source_symlink_rejected(tmp_path):
    outside = tmp_path / 'outside'; outside.write_text('x')
    root = tmp_path / 'root'; root.mkdir()
    (root / 'link').symlink_to(outside)
    with pytest.raises(ValueError): C.read_source(root, 'link')


def test_duplicate_json_rejected():
    with pytest.raises(ValueError): C.strict_json('{"status":"OPEN","status":"CLOSED"}')


def test_planning_source_drift_rejected(monkeypatch):
    original = C.read_source
    seen = {}
    def drift(root, path):
        data, identity = original(root, path)
        seen[path] = seen.get(path, 0) + 1
        if seen[path] > 1: identity['sha256'] = 'changed'
        return data, identity
    monkeypatch.setattr(C, 'read_source', drift)
    with pytest.raises(ValueError, match='changed'): C.build_plan()


def test_exclusive_external_output(tmp_path):
    root = tmp_path / 'repo'; root.mkdir()
    with pytest.raises(ValueError): C.runner.prepare_output(root, root / 'receipt')
    output = tmp_path / 'receipt'; output.mkdir()
    with pytest.raises(FileExistsError): C.runner.prepare_output(root, output)


def test_new_lane_and_plan_hash_tamper_refused():
    report = passed_report()
    for snapshot in ('before', 'after'):
        report[snapshot]['inputs']['engine/lanes/NEW.json'] = {'sha256': 'new'}
    assert C.resolve(minimal_plan(), report)['status'] == 'FAIL'
    plan = minimal_plan(); plan['items'][0]['record']['status'] = 'CLOSED'
    with pytest.raises(ValueError, match='hash'): C.resolve(plan, passed_report())


@pytest.mark.parametrize('flag', ['timed_out', 'log_limit_exceeded', 'budget_exhausted', 'error'])
def test_inconsistent_success_flags_never_complete(flag):
    report = passed_report(); report['steps'][0][flag] = True
    result = C.resolve(minimal_plan(), report)
    assert result['status'] == 'FAIL'
    assert result['machine_tasks_complete'] == 0


def test_whole_run_failure_never_emits_completion():
    report = passed_report(); report['status'] = 'FAIL'
    result = C.resolve(minimal_plan(), report)
    assert result['machine_tasks_complete'] == 0


def test_exact_remaining_conditions_and_live_rows_preserved():
    plan = C.build_plan()
    observed = [item for item in plan['items'] if item['item_key'].startswith('observed/')]
    assert len(observed) == 174
    assert len({item['item_key'] for item in observed}) == 174
    a5 = next(item for item in plan['items'] if item['item_key'] == 'lane/A5')
    conditions = C.source_conditions(a5['record'])
    assert conditions['source_fields_verbatim']['next_exact_action'] == a5['record']['next_exact_action']
    assert 'pending' in conditions['source_fields_verbatim']['next_exact_action']
    assert plan['reconciliation_issues'][0]['id'] == 'P01-ROUTING-VERSION-CONFLICT'


def test_cycle_and_bad_edge_type_rejected(monkeypatch):
    original = C.read_source
    for mutation in ('cycle', 'type'):
        def altered(root, path):
            data, identity = original(root, path)
            if path == 'claims/graph.json':
                graph = json.loads(data)
                graph['premises']['OBL-H5-JETMOD']['depends_on'] = ['OBL-H5-JETMOD'] if mutation == 'cycle' else 'OBL-H5-JETMOD'
                data = json.dumps(graph).encode()
            return data, identity
        monkeypatch.setattr(C, 'read_source', altered)
        with pytest.raises(ValueError): C.build_plan()


@pytest.mark.parametrize('tamper', ['omit_log', 'wrong_step_digest', 'changed_file', 'plan', 'report', 'none'])
def test_execute_rejects_tampered_artifacts(tmp_path, monkeypatch, tamper):
    plan = minimal_plan()
    plan['tasks'][0]['ci_command'] = 'python -m pytest -q'
    bind_plan(plan)
    monkeypatch.setattr(C, 'build_plan', lambda root: plan)
    def fake_run(root, output, timeout, budget):
        output.mkdir()
        log = output / '01.log'; log.write_text('original')
        junit = output / 'pytest.xml'; junit.write_text('fixture')
        report = passed_report()
        report['steps'][0]['ci_command'] = 'python -m pytest -q'
        report['steps'][0].update(log='01.log', log_identity=C.runner.artifact_identity(log))
        report['pytest'].update(file='pytest.xml', identity=C.runner.artifact_identity(junit))
        report['artifacts'] = {'01.log': C.runner.artifact_identity(log), 'pytest.xml': C.runner.artifact_identity(junit)}
        if tamper == 'omit_log':
            report['artifacts'].pop('01.log'); log.write_text('tampered')
        elif tamper == 'wrong_step_digest': report['steps'][0]['log_identity']['sha256'] = 'wrong'
        elif tamper == 'changed_file': log.write_text('tampered')
        C.runner.atomic_json(output / 'report.json', report)
        if tamper == 'plan': (output.parent / 'plan.json').write_text('{}')
        elif tamper == 'report': (output / 'report.json').write_text('{}')
        return report
    monkeypatch.setattr(C.runner, 'run_checks', fake_run)
    repo = tmp_path / 'repo'; repo.mkdir()
    if tamper == 'none':
        assert C.execute(repo, tmp_path / 'result')['status'] == 'PASS'
    else:
        with pytest.raises(ValueError): C.execute(repo, tmp_path / 'result')
        assert not (tmp_path / 'result/closure-report.json').exists()


def test_oq013_receipt_custody_is_separate_from_current_lean_replay():
    plan = C.build_plan()
    oq13 = [item for item in plan['items'] if item['source_id'] == 'OQ-013']
    assert len(oq13) == 2
    for item in oq13:
        assert len(item['recorded_external_evidence']) == 1
        evidence = item['recorded_external_evidence'][0]
        assert evidence['replayed_here'] is False
        assert evidence['scientific_promotions'] == 0
        assert evidence['independence_credit'] == 0
        assert item['scientific_disposition'] == 'SOURCE_RECORD_UNCHANGED'
    assert all(path in plan['sources'] for path in C.BUNDLE_FILES)
