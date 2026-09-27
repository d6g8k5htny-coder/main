#!/usr/bin/env python3
"""Loss-only transition pilot. Structural validation, never scientific acceptance.

Python 3.11+, standard library only. No writes, network, credentials or execution
of referenced artifacts. Enrolment of the research graph is a separate review.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


class Invalid(ValueError):
    """A malformed, unsafe, stale, or unsupported transition."""


def need(condition: bool, message: str) -> None:
    if not condition:
        raise Invalid(message)


def keys(obj: Any, expected: set[str], where: str) -> None:
    need(isinstance(obj, dict), f'{where}: expected object')
    need(set(obj) == expected, f'{where}: unexpected/missing fields')


def text(value: Any, where: str) -> None:
    need(isinstance(value, str) and bool(value.strip()), f'{where}: empty/non-string')


def digest(value: Any, where: str) -> None:
    need(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None,
         f'{where}: expected lowercase SHA-256')


def unique_ids(value: Any, where: str) -> None:
    need(isinstance(value, list), f'{where}: expected list')
    for item in value:
        text(item, where)
    need(len(value) == len(set(value)), f'{where}: duplicate IDs')


def canonical_sha(obj: Any) -> str:
    """Version-1 canonical JSON: sorted keys, compact UTF-8, no NaN/Infinity."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _pairs(pairs: list[tuple[str, Any]]) -> dict:
    out: dict[str, Any] = {}
    for key, value in pairs:
        need(key not in out, f'duplicate JSON key: {key}')
        out[key] = value
    return out


def load(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise Invalid(f'non-finite JSON token: {value}')
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=_pairs,
                      parse_constant=reject_constant)


ADM = {'ELIGIBLE': 0, 'SUSPENDED': 1, 'INELIGIBLE': 2}
SCHEDULE = {'ACTIVE', 'BLOCKED', 'PAUSED', 'CLOSED'}
KINDS = {'CLAIM', 'PREMISE', 'PROOF', 'STRATEGY', 'CARRIER'}
EVIDENCE = {'UNASSESSED', 'OPEN', 'CONDITIONAL', 'ESTABLISHED', 'REFUTED', 'INVALID'}


def validate_state(state: Any) -> None:
    keys(state, {'schema_version', 'snapshot_id', 'coverage', 'nodes', 'grounds',
                 'routes', 'dashboard_population'}, 'snapshot')
    need(type(state['schema_version']) is int and state['schema_version'] == 1,
         'unsupported snapshot version')
    text(state['snapshot_id'], 'snapshot_id')
    c = state['coverage']
    keys(c, {'scope_id', 'status', 'review_sha256'}, 'coverage')
    text(c['scope_id'], 'coverage.scope_id')
    need(c['status'] in {'COMPLETE', 'PARTIAL', 'UNASSESSED'}, 'coverage.status')
    if c['review_sha256'] is not None:
        digest(c['review_sha256'], 'coverage.review_sha256')
    need(c['status'] != 'COMPLETE' or c['review_sha256'] is not None,
         'COMPLETE coverage needs a bound review artifact (not verified here)')
    for group in ('nodes', 'grounds', 'routes'):
        need(isinstance(state[group], dict), f'{group}: expected object')
    nodes = state['nodes']
    need(bool(nodes), 'empty scope is not a successful audit')
    for nid, node in nodes.items():
        text(nid, 'node ID')
        need('/' not in nid, 'node ID may not contain /')
        keys(node, {'kind', 'contract_sha256', 'scope_sha256', 'evidential_status',
                    'admissibility', 'scheduling'}, f'node {nid}')
        need(node['kind'] in KINDS, f'{nid}: kind')
        need(node['evidential_status'] in EVIDENCE, f'{nid}: evidential_status')
        need(node['admissibility'] in ADM, f'{nid}: admissibility')
        need(node['scheduling'] in SCHEDULE, f'{nid}: scheduling')
        digest(node['contract_sha256'], f'{nid}: contract')
        digest(node['scope_sha256'], f'{nid}: scope')
        need(node['evidential_status'] not in {'REFUTED', 'INVALID'} or
             node['admissibility'] != 'ELIGIBLE', f'{nid}: refuted/invalid yet eligible')
    for group in ('grounds', 'routes'):
        for oid, obj in state[group].items():
            text(oid, f'{group} ID')
            need('/' not in oid, f'{group} ID may not contain /')
            expected = {'conclusion', 'artifact_sha256', 'scope_sha256', 'admissibility'}
            if group == 'routes':
                expected |= {'requires'}
            keys(obj, expected, f'{group}/{oid}')
            need(obj['conclusion'] in nodes, f'{oid}: unknown conclusion')
            need(obj['admissibility'] in ADM, f'{oid}: admissibility')
            digest(obj['artifact_sha256'], f'{oid}: artifact')
            digest(obj['scope_sha256'], f'{oid}: scope')
            need(obj['scope_sha256'] == nodes[obj['conclusion']]['scope_sha256'],
                 f'{oid}: conclusion scope mismatch')
            if group == 'routes':
                unique_ids(obj['requires'], f'{oid}: requires')
                need(bool(obj['requires']), 'unconditional warrant belongs in grounds')
                need(all(x in nodes for x in obj['requires']), f'{oid}: dangling premise')
    unique_ids(state['dashboard_population'], 'dashboard_population')
    need(set(state['dashboard_population']) <= set(nodes), 'unknown dashboard member')


def support(state: dict) -> tuple[set[str], dict[str, list[str]]]:
    """Least fixed point: AND within each route; OR between routes/grounds.

    Labels alone provide no support. In particular, an empty dependency list
    in an imported graph is not an unconditional proof. Unseeded cycles add none.
    These are available recorded support paths, not a theorem prover's verdict.
    """
    nodes = state['nodes']
    eligible = {n for n, obj in nodes.items() if obj['admissibility'] == 'ELIGIBLE'}
    available = {g['conclusion'] for g in state['grounds'].values()
                 if g['admissibility'] == 'ELIGIBLE' and g['conclusion'] in eligible}
    while True:
        expanded = available | {r['conclusion'] for r in state['routes'].values()
                                if r['admissibility'] == 'ELIGIBLE'
                                and r['conclusion'] in eligible
                                and set(r['requires']) <= available}
        if expanded == available:
            break
        available = expanded
    witnesses: dict[str, list[str]] = {n: [] for n in sorted(available)}
    for group in ('grounds', 'routes'):
        for oid, obj in state[group].items():
            n = obj['conclusion']
            if obj['admissibility'] == 'ELIGIBLE' and n in available and (
                    group == 'grounds' or set(obj['requires']) <= available):
                witnesses[n].append(f'{group}/{oid}')
    return available, witnesses


def validate_transition(before: Any, after: Any, ticket: Any) -> dict:
    validate_state(before)
    validate_state(after)
    keys(ticket, {'schema_version', 'transition_id', 'operation', 'author_session',
                  'before_sha256', 'after_sha256', 'reason', 'changed_objects',
                  'expected_lost_support', 'expected_supported_nodes', 'publication'}, 'ticket')
    need(type(ticket['schema_version']) is int and ticket['schema_version'] == 1,
         'unsupported ticket version')
    for field in ('transition_id', 'author_session'):
        text(ticket[field], field)
    need(ticket['operation'] in {'WITHDRAW_SUPPORT', 'CHANGE_SCHEDULE'},
         'v1 does not adjudicate falsity, restore support, replace contracts or add evidence')
    for name, obj in (('before', before), ('after', after)):
        digest(ticket[f'{name}_sha256'], f'{name} digest')
        need(ticket[f'{name}_sha256'] == canonical_sha(obj), f'stale {name} binding')
    need(before['snapshot_id'] != after['snapshot_id'], 'successor needs a new snapshot ID')
    for field in ('schema_version', 'coverage', 'dashboard_population'):
        need(before[field] == after[field], f'{field} cannot change in a loss-only transaction')
    reason = ticket['reason']
    keys(reason, {'kind', 'description', 'evidence_sha256'}, 'reason')
    text(reason['description'], 'reason.description')
    if ticket['operation'] == 'CHANGE_SCHEDULE':
        need(reason['kind'] == 'RESOURCE_ALLOCATION', 'schedule reason must be resource allocation')
        if reason['evidence_sha256'] is not None:
            digest(reason['evidence_sha256'], 'schedule evidence')
    else:
        need(reason['kind'] in {'PROOF_DEFECT', 'TYPING_UNCERTAINTY',
                               'STRATEGY_LIMIT', 'COUNTEREXAMPLE_CANDIDATE'},
             'time, cost, or boredom cannot withdraw scientific support')
        digest(reason['evidence_sha256'], 'withdrawal evidence')
    changed: set[str] = set()
    for group in ('nodes', 'grounds', 'routes'):
        need(set(before[group]) == set(after[group]), f'{group}: no deletion, addition or renaming')
        for oid, old in before[group].items():
            new = after[group][oid]
            allowed = {'scheduling'} if ticket['operation'] == 'CHANGE_SCHEDULE' and group == 'nodes' else set()
            if ticket['operation'] == 'WITHDRAW_SUPPORT':
                allowed = {'admissibility'}
            for field in old:
                if field not in allowed:
                    need(old[field] == new[field], f'{group}/{oid}: forbidden change to {field}')
            if 'admissibility' in allowed:
                need(ADM[new['admissibility']] >= ADM[old['admissibility']],
                     f'{group}/{oid}: restoration needs a separate reviewed transition')
            if new != old:
                changed.add(f'{group}/{oid}')
    need(bool(changed), 'empty change is not a transition')
    unique_ids(ticket['changed_objects'], 'changed_objects')
    need(changed == set(ticket['changed_objects']), 'changed object coverage mismatch')
    prior, _ = support(before)
    current, witnesses = support(after)
    need(current <= prior, 'premise evaporation: withdrawal increased support')
    if ticket['operation'] == 'CHANGE_SCHEDULE':
        need(current == prior, 'scheduling changed support')
    for field, actual in (('expected_lost_support', prior - current),
                          ('expected_supported_nodes', current)):
        unique_ids(ticket[field], field)
        need(set(ticket[field]) == actual, f'{field}: dependent rewrite or survivor mismatch')
    pub = ticket['publication']
    keys(pub, {'phase', 'drive', 'github'}, 'publication')
    need(pub['phase'] in {'PREPARED', 'RECORDED'}, 'v1 cannot confer ACCEPTED status')
    for surface in ('drive', 'github'):
        binding = pub[surface]
        if binding is not None:
            keys(binding, {'transaction_id', 'snapshot_sha256', 'reference'}, surface)
            need(binding['transaction_id'] == ticket['transition_id'], f'{surface}: mixed transaction')
            need(binding['snapshot_sha256'] == ticket['after_sha256'], f'{surface}: mixed snapshot')
            text(binding['reference'], f'{surface}: reference')
        if pub['phase'] == 'RECORDED':
            need(binding is not None, 'interrupted cross-system publication: recorded side missing')
    if pub['phase'] == 'RECORDED':
        need(after['coverage']['status'] == 'COMPLETE', 'incomplete dependency coverage: HOLD')
    return {
        'result': 'STRUCTURALLY_VALID', 'scientific_acceptance': False,
        'authority': 'NONE', 'transition_id': ticket['transition_id'],
        'coverage': after['coverage'], 'publication_phase': pub['phase'],
        'lost_support': sorted(prior - current), 'available_support': sorted(current),
        'surviving_witnesses': witnesses,
        'disabled_routes': {rid: r['requires'] for rid, r in after['routes'].items()
                            if r['admissibility'] != 'ELIGIBLE'
                            or r['conclusion'] not in current
                            or not set(r['requires']) <= current},
        'dashboard_population': after['dashboard_population'],
        'limitations': ['Input provenance and review authenticity require external verification.',
                       'Coverage completeness and mathematical correctness are not proved here.',
                       'No live research record is changed; no final refutation or restoration is authorized.']}


def inventory(graph: Any) -> dict:
    """Read-only enrolment inventory; never infer proof routes from depends_on."""
    need(isinstance(graph, dict), 'legacy graph must be an object')
    for group in ('claims', 'premises'):
        need(isinstance(graph.get(group), dict), f'legacy graph missing {group}')
    need(bool(graph['claims']) or bool(graph['premises']), 'empty legacy scope')
    names = set(graph['claims']) | set(graph['premises'])
    need(not (set(graph['claims']) & set(graph['premises'])), 'ambiguous legacy object IDs')
    rows = []
    for group in ('claims', 'premises'):
        for oid, obj in sorted(graph[group].items()):
            need(isinstance(obj, dict), f'{oid}: invalid legacy record')
            deps = obj.get('depends_on', [])
            unique_ids(deps, f'{oid}: legacy dependencies')
            rows.append({'id': oid, 'source_section': group,
                         'source_record_sha256': canonical_sha(obj),
                         'declared_dependencies': deps,
                         'unresolved_references': [x for x in deps if x not in names],
                         'enrolment': 'NOT_MIGRATED', 'support_verdict': 'NOT_ASSESSED'})
    return {'mode': 'READ_ONLY_SHADOW', 'dependency_coverage': 'UNASSESSED',
            'source_graph_sha256': canonical_sha(graph), 'object_count': len(rows),
            'objects': rows, 'scientific_status_changes': [],
            'warning': 'Declared edges are not a complete proof-route map. No empty list is promoted to a warrant.'}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    check = sub.add_parser('check')
    for name in ('before', 'after', 'ticket'):
        check.add_argument(f'--{name}', type=Path, required=True)
    inv = sub.add_parser('inventory')
    inv.add_argument('--legacy-graph', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'check':
            result = validate_transition(load(args.before), load(args.after), load(args.ticket))
        else:
            result = inventory(load(args.legacy_graph))
        print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False))
        return 0
    except (Invalid, ValueError, OSError, TypeError, KeyError, RecursionError) as exc:
        print(f'WITHDRAWAL HOLD: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
