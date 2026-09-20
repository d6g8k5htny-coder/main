#!/usr/bin/env python3
"""Inventory dated obligations and execute every supported machine check once.

Completion applies to explicit verification tasks, never to the source theorem.
This derived work plan does not replace Drive registers or interpret status prose.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import run_checks as runner
from tools.research_frontier import graph_view
from tools.lean_receipt import BUNDLE_FILES, verify as verify_lean_receipt

LIVE_OBSERVATIONS = 'docs/context/CLOSURE_REGISTER_OBSERVATIONS_20260920_v1.json'
CRITERIA = 'docs/context/CLOSURE_AUTOMATION_CRITERIA_20260920_v1.md'
REGISTERS = {
    'registers/json/easy_closure_queue.json': 'Candidate ID',
    'registers/json/review_queue.json': 'Review key',
    'registers/json/open_questions.json': 'OQ ID',
}
# These are relevance links, NOT logical implications or theorem-discharge rules.
LINKS = {
    'python tools/rn_certificate.py check-candidates': ['D3-LEMMA-RN-UNIF', 'RN5-NEAR-POINT-CERTS', 'A5'],
    'python tools/rn_side24_check.py': ['D3-LEMMA-RN-UNIF', 'RN5-NEAR-POINT-CERTS', 'A5'],
    'python tools/rn_side24_spatial_check.py': ['D3-LEMMA-RN-UNIF', 'A5'],
    'python tools/parallel_math_check.py': ['OBL-H5-JETMOD', 'A1', 'D3-LEMMA-RN-UNIF', 'A5'],
    'python tools/twelve_project_check.py': ['OBL-H5-JETMOD', 'A1', 'A5', 'LPW-CONSTANT-DELIVERED'],
    'python tools/rn_side24_density_check.py': ['D3-LEMMA-RN-UNIF', 'RN5-NEAR-POINT-CERTS', 'A5'],
    'python tools/rn_moment_report.py --check research/rn/candidates/affine_moments_20260920.json': ['D3-LEMMA-RN-UNIF', 'A5'],
    'python tools/hermite_envelope_report.py --check research/bands/candidates/hermite_gaussian_20260919.json': ['OBL-H5-JETMOD', 'A1'],
    'python tools/lpw_amplitude_check.py': ['LPW-CONSTANT-DELIVERED', 'LPW_CONSTANT as delivered', 'R05 Rayleigh quantitative repair'],
    'python tools/closure_pipeline.py check-plan': ['OQ-013'],
    'python tools/claims_check.py': ['OBL-D1-PROMOTE'],
    'python tools/lanes_check.py': ['A1', 'A5'],
}
LIMITS = [
    'Dated repository snapshots only; no claim of a complete or current Drive inventory.',
    'All source rows are retained. No substring-based OPEN/CLOSED classification or precedence between layers.',
    'Machine-task completion is not theorem closure, register promotion, applicability or independent review.',
    'Input identifiers, including superseded carriers, are inventory only and never authorization to execute or certify them.',
    'Only the reviewed explicit CI command allowlist executes; missing adapters remain unmapped.',
    'Actual RN field law, interval conditioning, spatial cover and remote-budget reassembly are not supplied by synthetic moment certificates.',
    'Local receipt times are unsigned observations; read-only permissions do not provide immutable storage against the owner.',
]


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'duplicate key: {key}')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f'nonfinite number: {value}')
    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)


def source_paths(root):
    return sorted(['claims/graph.json', 'docs/OPEN_PROBLEMS.md', runner.WORKFLOW,
                   'tools/closure_pipeline.py', 'tools/run_checks.py', 'tools/research_frontier.py',
                   'tools/lean_receipt.py', *BUNDLE_FILES, LIVE_OBSERVATIONS, CRITERIA, *REGISTERS,
                   *(p.relative_to(root).as_posix() for p in (root / 'engine/lanes').glob('*.json'))])


def read_source(root, relative):
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'source escape: {relative}')
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink():
            raise ValueError(f'source symlink: {relative}')
        if ancestor == root:
            break
    identity = runner.artifact_identity(path)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != identity['sha256']:
        raise ValueError(f'source changed: {relative}')
    return data, identity


def build_plan(root=ROOT):
    root = root.resolve()
    paths = source_paths(root)
    sources, contents = {}, {}
    for relative in paths:
        data, sources[relative] = read_source(root, relative)
        contents[relative] = data
    steps = runner.parse_workflow(contents[runner.WORKFLOW].decode())
    graph = strict_json(contents['claims/graph.json'])
    _, dependency_edges, unresolved_paths = graph_view(graph)
    lean_evidence = verify_lean_receipt(root)
    items = []
    for kind in ('premises', 'claims'):
        nodes = graph[kind]
        if not isinstance(nodes, dict) or not nodes:
            raise ValueError(f'empty/malformed graph {kind}')
        for key, record in sorted(nodes.items()):
            if not isinstance(record, dict):
                raise ValueError(f'invalid node {key}')
            items.append({'item_key': f'graph/{kind}/{key}', 'source_id': key,
                          'source_path': 'claims/graph.json', 'source_locator': f'{kind}.{key}',
                          'record': record})
    graph_ids = {item['source_id'] for item in items}
    for item in items:
        record = item['record']
        for dep in (*record.get('depends_on', []), *record.get('sub_obligations', [])):
            if dep not in graph_ids:
                raise ValueError(f'unknown graph dependency: {dep}')
    lanes = [path for path in paths if path.startswith('engine/lanes/')]
    if not lanes:
        raise ValueError('no lane records')
    for path in lanes:
        record = strict_json(contents[path])
        key = record['key']
        if not isinstance(key, str) or not key:
            raise ValueError('invalid lane key')
        items.append({'item_key': f'lane/{key}', 'source_id': key, 'source_path': path,
                      'source_locator': '$', 'record': record})
    for path, id_column in REGISTERS.items():
        register = strict_json(contents[path])
        header, rows = register['header'], register['rows']
        if (not isinstance(header, list) or not all(isinstance(x, str) for x in header)
                or len(set(header)) != len(header) or id_column not in header or not rows):
            raise ValueError(f'malformed register: {path}')
        seen = set()
        for row_number, row in enumerate(rows, start=2):
            if not isinstance(row, list) or len(row) != len(header):
                raise ValueError(f'malformed row {row_number}: {path}')
            record = dict(zip(header, row))
            key = record[id_column]
            if not isinstance(key, str) or not key or key in seen:
                raise ValueError(f'missing/duplicate register id {key}: {path}')
            seen.add(key)
            items.append({'item_key': f'{Path(path).stem}/{key}', 'source_id': key,
                          'source_path': path, 'source_locator': f'row {row_number}', 'record': record})
    live = strict_json(contents[LIVE_OBSERVATIONS])
    if live['schema'] != 'read-only-live-register-observations-v1' or not live['sheets']:
        raise ValueError('invalid bounded live observations')
    for sheet, table in live['sheets'].items():
        header, *rows = table['values']
        if not header or len(set(header)) != len(header):
            raise ValueError(f'invalid live observation header: {sheet}')
        for row_number, row in enumerate(rows, start=2):
            if not row or not any(str(cell).strip() for cell in row):
                continue
            if len(row) > len(header):
                raise ValueError(f'oversized observed row: {sheet}/{row_number}')
            # Preserve source/row identity even when record IDs are duplicated.
            record = dict(zip(header, row + [''] * (len(header) - len(row))))
            items.append({'item_key': f'observed/{sheet}/{row_number}', 'source_id': str(row[0]),
                          'source_path': LIVE_OBSERVATIONS,
                          'source_locator': f'{sheet}!row {row_number}',
                          'observation_time': live['observed_at'], 'record': record})
    if len({item['item_key'] for item in items}) != len(items):
        raise ValueError('duplicate inventory item key')
    known = {item['source_id'] for item in items}
    tasks = []
    for index, step in enumerate(steps, start=1):
        refs = LINKS.get(step['ci_command'], [])
        if any(ref not in known for ref in refs):
            raise ValueError(f'unknown relevance link: {refs}')
        tasks.append({'task_id': f'CHECK-{index:03d}', **step,
                      'relevant_source_ids': refs, 'completion_scope': 'EXPLICIT_MACHINE_CHECK_ONLY'})
    for item in items:
        item['verification_tasks'] = [task['task_id'] for task in tasks
                                      if item['source_id'] in task['relevant_source_ids']]
        item['eligibility'] = 'SCOPED_MACHINE_CHECKS_AVAILABLE' if item['verification_tasks'] else 'NO_EXACT_DISCHARGE_ADAPTER'
        item['scientific_disposition'] = 'SOURCE_RECORD_UNCHANGED'
        item['recorded_external_evidence'] = [lean_evidence] if item['source_id'] == 'OQ-013' else []
    if source_paths(root) != paths or any(read_source(root, p)[1] != sources[p] for p in paths):
        raise ValueError('input membership or bytes changed while planning')
    payload = {'schema': 'q0.closure-plan/v1', 'sources': sources, 'items': items,
               'dependency_edges': dependency_edges, 'unresolved_dependency_paths': unresolved_paths,
               'reconciliation_issues': [{
                   'id': 'P01-ROUTING-VERSION-CONFLICT',
                   'observations': 'Open Questions OQ-009 v1.10; Easy Closure Queue v1.9; Help Board v1.6 routes',
                   'source': LIVE_OBSERVATIONS, 'criteria': CRITERIA,
                   'disposition': 'RECONCILE_EXACT_ROUTE_WITH_GOVERNING_SOURCE; no automatic precedence'},
                   {'id': 'A2-DATED-ROUTING', 'observations': 'Later v2.3 source says FROZEN midflight; older repository snapshot says resuming',
                    'source': CRITERIA, 'disposition': 'NO_AUTOMATIC_RESUMPTION'}],
               'tasks': tasks, 'limits': LIMITS, 'authority': 'NONE', 'independence_credit': 0}
    return {**payload, 'payload_sha256': hashlib.sha256(runner.canonical_bytes(payload)).hexdigest()}


def source_conditions(record):
    """Retain exact source actions as prose; do not manufacture formal predicates."""
    fields = ('next_exact_action', 'falsifier', 'not_a_substitute', 'does_not_establish',
              'Remaining decisive work', 'Next action', 'Next decisive action',
              'Scope / dependencies', 'Independent review needed', 'Independence status',
              'Human approval needed', 'Owner / next actor', 'Owner or capacity needed',
              'Next action / trigger', 'Next exact action', 'Blocker', 'note',
              'depends_on', 'sub_obligations', 'requires_independent_verdict',
              'independent_review_state', 'evidence_note')
    exact = {key: record[key] for key in fields if key in record and record[key] not in ('', None, [])}
    return {'source_fields_verbatim': exact,
            'machine_interpretation': 'NOT_FORMALIZED; source prose is preserved, not declared satisfied',
            'adapter_gap': None if exact else 'No structured closure criterion mapped; inspect original_record and exact governing source.'}


def resolve(plan, verification):
    """Translate actual runner outcomes without upgrading source records."""
    payload = {key: value for key, value in plan.items() if key != 'payload_sha256'}
    if plan.get('payload_sha256') != hashlib.sha256(runner.canonical_bytes(payload)).hexdigest():
        raise ValueError('plan payload hash mismatch')
    outcomes = verification.get('steps', [])
    indexed = {}
    for outcome in outcomes:
        command = outcome['ci_command']
        if command in indexed:
            raise ValueError('duplicate executed command')
        indexed[command] = outcome
    expected = {task['ci_command'] for task in plan['tasks']}
    if set(indexed) - expected:
        raise ValueError('unplanned command in verification report')
    stable = verification.get('before') is not None and verification.get('before') == verification.get('after')
    bound = stable and all(verification['before']['inputs'].get(path, {}).get('sha256') == identity['sha256']
                           for path, identity in plan['sources'].items())
    if bound:
        planned_lanes = {p for p in plan['sources'] if p.startswith('engine/lanes/')}
        actual_lanes = {p for p in verification['before']['inputs']
                        if p.startswith('engine/lanes/') and p.endswith('.json') and p.count('/') == 2}
        bound = planned_lanes == actual_lanes
    valid_run = (verification.get('status') == 'PASS' and not verification.get('failures')
                 and verification.get('pytest', {}).get('executed', 0) > 0
                 and not any(verification.get('pytest', {}).get(k, 0) for k in ('errors', 'failures')))
    completed = []
    for task in plan['tasks']:
        outcome = indexed.get(task['ci_command'])
        if not outcome:
            status = 'NOT_RUN'
        elif not bound:
            status = 'INVALID_INPUT_BINDING'
        elif (valid_run and outcome.get('passed') is True and outcome.get('returncode') == 0
              and outcome.get('log_identity') and not any(outcome.get(k) for k in
                  ('timed_out', 'log_limit_exceeded', 'budget_exhausted', 'error'))):
            status = 'MACHINE_VERIFICATION_COMPLETE'
        else:
            status = 'FAILED_OR_INCOMPLETE_RUN'
        completed.append({**task, 'status': status, 'execution_evidence': outcome})
    complete_ids = {task['task_id'] for task in completed if task['status'] == 'MACHINE_VERIFICATION_COMPLETE'}
    candidates = []
    for item in plan['items']:
        relevant = item['verification_tasks']
        candidates.append({'item_key': item['item_key'], 'source_id': item['source_id'],
                           'source_path': item['source_path'], 'source_locator': item['source_locator'],
                           'source_identity': plan['sources'][item['source_path']],
                           'verification_tasks': relevant,
                           'verification_status': ('SCOPED_CHECKS_COMPLETE' if all(x in complete_ids for x in relevant)
                                                   else 'SCOPED_CHECKS_INCOMPLETE') if relevant else 'UNMAPPED_EXACT_OBLIGATION',
                           'recorded_external_evidence': item.get('recorded_external_evidence', []),
                           'original_record': item['record'], 'scientific_disposition': 'SOURCE_RECORD_UNCHANGED',
                           'remaining_decisive_work': source_conditions(item['record'])})
    passed = (bool(completed) and len(complete_ids) == len(completed) and bound
              and verification.get('status') == 'PASS' and verification.get('pytest', {}).get('executed', 0) > 0)
    return {'status': 'PASS' if passed else 'FAIL', 'input_binding_valid': bound,
            'tasks': completed, 'resolution_candidates': candidates,
            'machine_tasks_complete': len(complete_ids), 'scientific_items_promoted': 0,
            'limits': LIMITS, 'authority': 'NONE', 'independence_credit': 0}


def execute(root, output, timeout=1800, budget=1800):
    plan = build_plan(root)
    output = runner.prepare_output(root.resolve(), output)
    runner.atomic_json(output / 'plan.json', plan)
    verification = runner.run_checks(root, output / 'verification', timeout, budget)
    # Re-read artifacts and source membership before deriving completion evidence.
    if strict_json((output / 'plan.json').read_bytes()) != plan:
        raise ValueError('written plan changed during verification')
    if strict_json((output / 'verification/report.json').read_bytes()) != verification:
        raise ValueError('written verification report differs from executed result')
    for name, identity in verification.get('artifacts', {}).items():
        if Path(name).name != name or runner.artifact_identity(output / 'verification' / name) != identity:
            raise ValueError(f'verification artifact changed: {name}')
    runner.validate_artifact_bindings(verification)
    if build_plan(root) != plan:
        raise ValueError('plan membership or content changed during verification')
    result = resolve(plan, verification)
    payload = {'schema': 'q0.closure-run/v1', 'observed_at_local_utc': runner.utc_now(),
               'plan': runner.artifact_identity(output / 'plan.json'),
               'verification_report': runner.artifact_identity(output / 'verification/report.json'), **result}
    payload['payload_sha256'] = hashlib.sha256(runner.canonical_bytes(payload)).hexdigest()
    runner.atomic_json(output / 'closure-report.json', payload)
    for path in (output / 'plan.json', output / 'closure-report.json'):
        path.chmod(0o444)
    output.chmod(0o555)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('check-plan')
    sub.add_parser('plan')
    run = sub.add_parser('run')
    run.add_argument('--output-dir', type=Path, required=True)
    run.add_argument('--timeout-seconds', type=float, default=1800)
    run.add_argument('--budget-seconds', type=float, default=1800)
    args = parser.parse_args(argv)
    try:
        if args.command == 'run':
            result = execute(ROOT, args.output_dir, args.timeout_seconds, args.budget_seconds)
            print(f"{result['status']} closure_pipeline: {result['machine_tasks_complete']} machine tasks; 0 scientific promotions; {args.output_dir / 'closure-report.json'}")
            return 0 if result['status'] == 'PASS' else 1
        plan = build_plan()
        if args.command == 'plan':
            print(json.dumps(plan, indent=2, ensure_ascii=True))
        else:
            print(f"PASS closure plan: {len(plan['items'])} dated source records; {len(plan['tasks'])} explicit checks; no source-status decisions")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'FAIL closure_pipeline: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
