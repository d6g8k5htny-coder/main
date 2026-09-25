"""Read-only, loss-only graph transition inspector. Does not award proof grades.
Supports the existing claims/premises graph shape; does not rewrite its statuses.
Use --fail-on-impact as an integration check only after binding exact input files.
"""
from __future__ import annotations
import argparse, hashlib, json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

class AuditError(ValueError):
    """Malformed or ambiguous audit input."""

def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise AuditError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result

def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique_object)

def graph_nodes(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if not isinstance(graph, dict):
        raise AuditError('Graph must be an object')
    result = {}
    for section in ('claims', 'premises'):
        part = graph.get(section)
        if not isinstance(part, dict):
            raise AuditError(f'Missing/invalid section: {section}')
        for key, node in part.items():
            if not isinstance(key, str) or not key or not isinstance(node, dict):
                raise AuditError('Invalid node')
            if key in result:
                raise AuditError(f'Node occurs in both sections: {key}')
            for field in ('depends_on', 'sub_obligations'):
                deps = node.get(field, [])
                if not isinstance(deps, list) or any(not isinstance(d, str) or not d for d in deps):
                    raise AuditError(f'Invalid {key}.{field}')
                if len(deps) != len(set(deps)):
                    raise AuditError(f'Duplicate edges in {key}.{field}')
            result[key] = node
    return result

def dependencies(node):
    return set(node.get('depends_on', []) + node.get('sub_obligations', []))

def structural_problems(nodes):
    errors = []
    for name, node in nodes.items():
        for dep in sorted(dependencies(node) - nodes.keys()):
            errors.append(f'MISSING_DEPENDENCY:{name}->{dep}')
    # Kahn's algorithm is iterative and flags cycles without recursive depth limits.
    indegree = {name: 0 for name in nodes}
    children = defaultdict(set)
    for name, node in nodes.items():
        for dep in dependencies(node) & nodes.keys():
            children[dep].add(name); indegree[name] += 1
    q = deque(n for n, degree in indegree.items() if degree == 0)
    visited = 0
    while q:
        name = q.popleft(); visited += 1
        for child in children[name]:
            indegree[child] -= 1
            if indegree[child] == 0:
                q.append(child)
    if visited != len(nodes):
        errors.append('DEPENDENCY_CYCLE')
    return errors

def transition_report(before, after):
    old, new = graph_nodes(before), graph_nodes(after)
    changed = {k for k in old.keys() | new.keys() if old.get(k) != new.get(k)}
    reverse = defaultdict(set)
    for snapshot in (old, new):
        for name, node in snapshot.items():
            for dep in dependencies(node):
                reverse[dep].add(name)
    affected = set(changed); q = deque(sorted(changed))
    while q:
        for dependent in reverse[q.popleft()]:
            if dependent not in affected:
                affected.add(dependent); q.append(dependent)
    errors = ['BEFORE:'+x for x in structural_problems(old)]
    errors += ['AFTER:'+x for x in structural_problems(new)]
    return {'changed_nodes': sorted(changed), 'revalidation_required': sorted(affected),
            'removed_nodes': sorted(old.keys() - new.keys()), 'structural_problems': errors,
            'hold_required': bool(affected or errors), 'scientific_acceptance': False,
            'gate_deployed': False,
            'scope': 'Recorded nodes/edges only. No semantic proof validation, release or live status mutation.'}

def check_ledger(ledger):
    if not isinstance(ledger, dict):
        raise AuditError('Ledger must be an object')
    items = ledger.get('items')
    if not isinstance(items, list) or len(items) != ledger.get('observation_count'):
        raise AuditError('Ledger count mismatch')
    if any(not isinstance(r, dict) for r in items):
        raise AuditError('Ledger rows must be objects')
    ids = [r.get('drive_id') for r in items]
    if any(not isinstance(k, str) or not k for k in ids) or len(ids) != len(set(ids)):
        raise AuditError('Missing or duplicate Drive ID')
    if ledger.get('intake_frozen') is not True:
        raise AuditError('This audit has no verified release; intake must remain frozen')
    for r in items:
        if r.get('scientific_authority') is not False:
            raise AuditError('Shelf records cannot confer scientific authority')
        if type(r.get('review_complete')) is not bool:
            raise AuditError('review_complete must be Boolean')
        if r['review_complete']:
            if (r.get('classification') != 'DUPLICATE_WITH_VERIFIED_SUCCESSOR'
                    or r.get('evidence_tier') != 'BYTE_IDENTITY_VERIFIED'
                    or not r.get('successor_id') or r.get('successor_id') in ids
                    or not r.get('sha256') or r.get('sha256') != r.get('successor_hash')
                    or len(r['sha256']) != 64 or any(c not in '0123456789abcdef' for c in r['sha256'])):
                raise AuditError('Unsupported completed duplicate review')
    expected = hashlib.sha256(('\n'.join(sorted(ids))+'\n').encode()).hexdigest()
    if ledger.get('observed_id_set_sha256') != expected:
        raise AuditError('Observed ID-set fingerprint mismatch')
    return {'items_checked': len(items), 'intake_frozen': True,
            'scientific_acceptance': False, 'live_permissions_enforced': False}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before'); parser.add_argument('--after'); parser.add_argument('--ledger')
    parser.add_argument('--fail-on-impact', action='store_true')
    args = parser.parse_args()
    try:
        if args.ledger:
            if args.before or args.after:
                raise AuditError('Choose ledger or graph mode, not both')
            result = check_ledger(load_json(args.ledger))
        else:
            if not args.before or not args.after:
                raise AuditError('Both --before and --after are required')
            result = transition_report(load_json(args.before), load_json(args.after))
        print(json.dumps(result, sort_keys=True, indent=2))
        return 2 if args.fail_on_impact and result.get('hold_required') else 0
    except (AuditError, OSError, ValueError) as exc:
        print(json.dumps({'error': str(exc), 'hold_required': True, 'scientific_acceptance': False}))
        return 2

if __name__ == '__main__':
    raise SystemExit(main())
