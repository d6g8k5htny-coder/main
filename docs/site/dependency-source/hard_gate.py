"""Fail-closed downstream promotion gate (main #90 / #86).

Scientific effect: NONE. This module never flips lemma_closed, prizes, or
premise registers. Green CI / hashes / same-author checks are not discharge.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
GRAPH_PATH = ROOT / 'GRAPH.json'
SELECTOR_REGION_PATH = ROOT / 'SELECTOR_REGION.json'
D0_PATCH_PATH = ROOT / 'patches' / 'd0_packet_allowlist.patch'

TERMINAL = frozenset({
    'PROVED_REVIEWED',
    'SUPERSEDED_NONBLOCKING',
    'REFUTED',
    'BLOCKED_ABSENT',
})

# Classifications that may appear on live nodes but are never terminal discharge.
NONTERMINAL = frozenset({
    'OPEN_HISTORICAL',
    'OPEN_ACTIVE',
    'AUTHOR_SIDE_CANDIDATE',
    'AUTHOR_SIDE_REDUCTION',
    'COVERED_BY_CANDIDATE',
    'ENGINEERING_CONTROL',
    'HOLD',
    'REVALIDATION_REQUIRED',
    'FALSE',
    'EXPLORATORY',
})

CONTROLLING_ELIGIBLE = frozenset({'PROVED_REVIEWED'})
# #90 disposition vocabulary is broader than premise satisfaction. A still-required
# REFUTED/SUPERSEDED/BLOCKED premise cannot be used as a positive theorem premise.
# Supersession satisfies only after the dependency edge is reviewed and removed/replaced.
REQUIRED_SATISFIED = frozenset({'PROVED_REVIEWED'})

NON_DISCHARGE_DEFAULT = (
    'GREEN_CI',
    'HASH_MATCH',
    'ARCHITECTURAL_ADMISSION',
    'NUMERICAL_EXPERIMENT',
    'SAME_AUTHOR_REVIEW',
    'NAVIGATION_SUCCESS',
    'AUTHOR_SELF_CHECK',
)


def _json_identity(value: Any) -> str:
    """Canonical JSON distinguishes Boolean/numeric edits and rejects non-finite data."""
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ValueError('graph contains non-JSON or non-finite data') from exc


def load_json_strict(text: str) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key: ' + key)
            result[key] = value
        return result
    def constant(value):
        raise ValueError('non-finite JSON constant: ' + value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def load_graph(path: Path | None = None) -> dict[str, Any]:
    data = load_json_strict((path or GRAPH_PATH).read_text())
    validate_graph_fail_closed(data)
    return data


def require_node(graph: dict[str, Any], node_id: str) -> dict[str, Any]:
    nodes = graph['nodes']
    if node_id not in nodes:
        raise KeyError('unknown node: ' + node_id)
    return nodes[node_id]


def required_dependencies(graph: dict[str, Any], node_id: str) -> list[str]:
    require_node(graph, node_id)
    return [e['to'] for e in graph['edges']
            if e['from'] == node_id and e.get('required', True)]


def all_dependencies(graph: dict[str, Any], node_id: str) -> list[str]:
    require_node(graph, node_id)
    return [e['to'] for e in graph['edges'] if e['from'] == node_id]


def transitive_required(graph: dict[str, Any], node_id: str) -> list[str]:
    """Depth-first required transitive closure, cycle-safe, deterministic order."""
    seen: set[str] = set()
    order: list[str] = []

    def walk(nid: str) -> None:
        for dep in required_dependencies(graph, nid):
            if dep in seen:
                continue
            seen.add(dep)
            order.append(dep)
            walk(dep)

    walk(node_id)
    return order


def dependents(graph: dict[str, Any], node_id: str, *, required_only: bool = False) -> list[str]:
    require_node(graph, node_id)
    out = []
    for e in graph['edges']:
        if e['to'] != node_id:
            continue
        if required_only and not e.get('required', True):
            continue
        out.append(e['from'])
    return sorted(set(out))


def transitive_dependents(graph: dict[str, Any], node_id: str) -> list[str]:
    seen: set[str] = set()
    order: list[str] = []

    def walk(nid: str) -> None:
        for child in dependents(graph, nid):
            if child in seen:
                continue
            seen.add(child)
            order.append(child)
            walk(child)

    walk(node_id)
    return order


def is_terminal(classification: str) -> bool:
    return classification in TERMINAL


def blocked_absent_hold(graph: dict[str, Any], node_id: str) -> list[str]:
    """Required transitive BLOCKED_ABSENT dependencies force HOLD."""
    return [dep for dep in transitive_required(graph, node_id)
            if require_node(graph, dep)['classification'] == 'BLOCKED_ABSENT']


def refuted_required_hold(graph: dict[str, Any], node_id: str) -> list[str]:
    """A still-required REFUTED premise blocks its dependent (#90 clarification)."""
    return [dep for dep in transitive_required(graph, node_id)
            if require_node(graph, dep)['classification'] == 'REFUTED']


def promotion_allowed(graph: dict[str, Any], node_id: str) -> dict[str, Any]:
    """Return whether node_id may become CONTROLLING under #90 rules."""
    validate_graph_fail_closed(graph)
    node = require_node(graph, node_id)
    required = transitive_required(graph, node_id)
    missing_terminal = []
    blocked = []
    refuted = []
    for dep in required:
        cls = require_node(graph, dep)['classification']
        if cls not in REQUIRED_SATISFIED and cls not in ('BLOCKED_ABSENT', 'REFUTED'):
            missing_terminal.append({'id': dep, 'classification': cls})
        if cls == 'BLOCKED_ABSENT':
            blocked.append(dep)
        if cls == 'REFUTED':
            refuted.append(dep)

    reasons: list[str] = []
    node_classification = node.get('classification')
    if node_classification not in CONTROLLING_ELIGIBLE:
        reasons.append(
            'node classification is not eligible for positive CONTROLLING status: '
            + str(node_classification)
        )
    if node_classification == 'REVALIDATION_REQUIRED':
        reasons.append('node requires revalidation after a dependency change')
    if blocked:
        reasons.append('required BLOCKED_ABSENT dependency forces HOLD')
    if refuted:
        reasons.append('required REFUTED dependency forces HOLD')
    if missing_terminal:
        reasons.append('required transitive dependency is not satisfied')
    if node_classification == 'FALSE' and node_id == 'hist.lemma_closed':
        reasons.append('lemma_closed register must remain FALSE; gate never promotes it')

    allowed = not reasons
    return {
        'node': node_id,
        'allowed': allowed,
        'would_be_controlling': allowed,
        'required_dependencies': required,
        'missing_terminal': missing_terminal,
        'blocked_absent': blocked,
        'refuted_required': refuted,
        'reasons': reasons,
        'meaning': 'integrity decision only; not theorem acceptance',
    }


def refuse_non_discharge_promotion(
    graph: dict[str, Any],
    node_id: str,
    evidence_tokens: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Fail closed when promotion is justified only by non-discharge tokens."""
    tokens = list(evidence_tokens)
    non_discharge = set(graph.get('non_discharge_tokens', NON_DISCHARGE_DEFAULT))
    unknown = [t for t in tokens if t not in non_discharge and t not in (
        'PROVED_REVIEWED', 'SUPERSEDED_NONBLOCKING', 'REFUTED', 'LINE_BY_LINE_ANALYTIC_REVIEW',
        'EXACT_COUNTEREXAMPLE', 'SUPERSESSION_CROSSWALK',
    )]
    only_non_discharge = bool(tokens) and all(t in non_discharge for t in tokens)
    base = promotion_allowed(graph, node_id)
    refused = only_non_discharge or (not base['allowed'])
    reasons = list(base['reasons'])
    if only_non_discharge:
        reasons.append('evidence consists only of non-discharge tokens')
    if unknown:
        reasons.append('unknown evidence tokens: ' + ','.join(unknown))
        refused = True
    return {
        'node': node_id,
        'refused': refused,
        'allowed': (not refused) and base['allowed'],
        'evidence_tokens': tokens,
        'reasons': reasons,
        'base': base,
    }


def apply_promotion(graph: dict[str, Any], node_id: str) -> dict[str, Any]:
    """Mutate a copy: set controlling only when legal; otherwise HOLD / refuse."""
    clone = json.loads(json.dumps(graph))
    decision = promotion_allowed(clone, node_id)
    node = clone['nodes'][node_id]
    if decision['blocked_absent'] or decision.get('refuted_required'):
        node['controlling'] = False
        node['classification'] = 'HOLD'
        decision = {**decision, 'applied': 'HOLD', 'ok': False}
    elif not decision['allowed']:
        node['controlling'] = False
        decision = {**decision, 'applied': 'REFUSED', 'ok': False}
    else:
        node['controlling'] = True
        decision = {**decision, 'applied': 'CONTROLLING', 'ok': True}
    return {'graph': clone, 'decision': decision}


def reverse_impact(
    graph: dict[str, Any],
    changed_node: str,
    *,
    old_fingerprint: str | None = None,
    new_fingerprint: str | None = None,
    old_classification: str | None = None,
    new_classification: str | None = None,
) -> dict[str, Any]:
    """Mark transitive dependents REVALIDATION_REQUIRED after a dependency change."""
    validate_graph_fail_closed(graph)
    require_node(graph, changed_node)
    clone = json.loads(json.dumps(graph))
    changed = False
    if old_fingerprint is not None and new_fingerprint is not None and old_fingerprint != new_fingerprint:
        changed = True
        clone['nodes'][changed_node]['fingerprint'] = new_fingerprint
    if old_classification is not None and new_classification is not None and old_classification != new_classification:
        changed = True
        clone['nodes'][changed_node]['classification'] = new_classification

    impacted: list[str] = []
    if changed:
        for dep in [changed_node, *transitive_dependents(clone, changed_node)]:
            node = clone['nodes'][dep]
            if node.get('controlling') or node.get('classification') in (
                'AUTHOR_SIDE_CANDIDATE', 'AUTHOR_SIDE_REDUCTION', 'COVERED_BY_CANDIDATE',
                'PROVED_REVIEWED', 'SUPERSEDED_NONBLOCKING',
            ):
                node['classification'] = 'REVALIDATION_REQUIRED'
                node['controlling'] = False
                impacted.append(dep)

    return {
        'changed_node': changed_node,
        'dependency_changed': changed,
        'impacted': impacted,
        'graph': clone,
        'meaning': 'reverse-impact revalidation marks; not theorem discharge',
    }



def _validate_graph_shape(graph: dict[str, Any]) -> None:
    if not isinstance(graph, dict) or type(graph.get('schema_version')) is not int or graph['schema_version'] != 1:
        raise ValueError('unsupported graph schema_version')
    nodes = graph.get('nodes')
    edges = graph.get('edges')
    if not isinstance(nodes, dict) or not nodes or not isinstance(edges, list):
        raise ValueError('malformed graph')
    _json_identity(graph)
    for nid, node in nodes.items():
        if not isinstance(nid, str) or not nid.strip() or not isinstance(node, dict):
            raise ValueError('malformed node record')
        cls = node.get('classification')
        if not isinstance(cls, str) or cls not in TERMINAL | NONTERMINAL:
            raise ValueError('unknown node classification')
        if type(node.get('controlling')) is not bool:
            raise ValueError('node controlling must be exact boolean')
    seen_edges: set[tuple[str, str, str]] = set()
    for edge in edges:
        if not isinstance(edge, dict) or not {'from', 'to', 'required', 'relation'} <= set(edge):
            raise ValueError('malformed edge record')
        if not isinstance(edge['from'], str) or not isinstance(edge['to'], str):
            raise ValueError('edge endpoints must be strings')
        if type(edge['required']) is not bool:
            raise ValueError('edge required must be exact boolean')
        if not isinstance(edge['relation'], str) or not edge['relation'].strip():
            raise ValueError('edge relation must be nonempty string')
        if edge['from'] not in nodes or edge['to'] not in nodes:
            raise ValueError('edge references missing node')
        key = (edge['from'], edge['to'], edge['relation'])
        if key in seen_edges:
            raise ValueError('duplicate or contradictory edge record')
        seen_edges.add(key)


def _required_cycle(graph: dict[str, Any]) -> list[str] | None:
    _validate_graph_shape(graph)
    visiting: set[str] = set()
    done: set[str] = set()
    stack: list[str] = []
    def visit(nid: str) -> list[str] | None:
        if nid in visiting:
            i = stack.index(nid)
            return stack[i:] + [nid]
        if nid in done:
            return None
        visiting.add(nid); stack.append(nid)
        for dep in required_dependencies(graph, nid):
            cycle = visit(dep)
            if cycle:
                return cycle
        stack.pop(); visiting.remove(nid); done.add(nid)
        return None
    for nid in sorted(graph['nodes']):
        cycle = visit(nid)
        if cycle:
            return cycle
    return None


def validate_graph_fail_closed(graph: dict[str, Any]) -> None:
    _validate_graph_shape(graph)
    cycle = _required_cycle(graph)
    if cycle:
        raise ValueError('required dependency cycle: ' + ' -> '.join(cycle))


def _outgoing_signature(graph: dict[str, Any], nid: str) -> list[str]:
    return sorted(_json_identity(e) for e in graph['edges'] if e['from'] == nid)


def reverse_impact_between(
    old_graph: dict[str, Any], new_graph: dict[str, Any], *,
    old_sources: dict[str, Any] | None = None,
    new_sources: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Loss-only proposal; source bytes are monitored only through supplied snapshots."""
    validate_graph_fail_closed(old_graph)
    validate_graph_fail_closed(new_graph)
    if (old_sources is None) != (new_sources is None):
        raise ValueError('both source snapshots are required together')
    old_nodes, new_nodes = old_graph['nodes'], new_graph['nodes']
    if old_sources is not None:
        for graph, sources in ((old_graph, old_sources), (new_graph, new_sources)):
            if not isinstance(sources, dict) or set(sources) != set(graph['nodes']):
                raise ValueError('source snapshot must cover exactly all graph nodes')
            _json_identity(sources)
    all_ids = set(old_nodes) | set(new_nodes)
    changed: set[str] = set()
    context_changed = _json_identity({k: v for k, v in old_graph.items() if k not in ('nodes', 'edges')}) != _json_identity({k: v for k, v in new_graph.items() if k not in ('nodes', 'edges')})
    for nid in all_ids:
        if nid not in old_nodes or nid not in new_nodes:
            changed.add(nid)
            continue
        if _json_identity(old_nodes[nid]) != _json_identity(new_nodes[nid]):
            changed.add(nid)
        if _outgoing_signature(old_graph, nid) != _outgoing_signature(new_graph, nid):
            changed.add(nid)
        if old_sources is not None and _json_identity(old_sources[nid]) != _json_identity(new_sources[nid]):
            changed.add(nid)
    if context_changed:
        changed.update(all_ids)
    union_edges = {(e['from'], e['to']) for g in (old_graph, new_graph) for e in g['edges']}
    reverse: dict[str, set[str]] = {}
    for child, dep in union_edges:
        reverse.setdefault(dep, set()).add(child)
    impacted: set[str] = {nid for nid in changed if nid in new_nodes}
    queue = list(changed)
    seen = set(queue)
    while queue:
        dep = queue.pop()
        for child in reverse.get(dep, ()):
            if child not in seen:
                seen.add(child)
                queue.append(child)
            if child in new_nodes:
                impacted.add(child)
    clone = json.loads(json.dumps(new_graph))
    for nid in sorted(impacted):
        node = clone['nodes'][nid]
        if node.get('classification') in (
            'AUTHOR_SIDE_CANDIDATE', 'AUTHOR_SIDE_REDUCTION', 'COVERED_BY_CANDIDATE',
            'PROVED_REVIEWED', 'SUPERSEDED_NONBLOCKING', 'ENGINEERING_CONTROL',
        ):
            node['classification'] = 'REVALIDATION_REQUIRED'
        node['controlling'] = False
    return {
        'changed_nodes': sorted(changed), 'impacted': sorted(impacted), 'graph': clone,
        'traversal_edges': [list(e) for e in sorted(union_edges)],
        'source_snapshots_supplied': old_sources is not None,
        'context_changed': context_changed,
        'promotion_permission': False,
        'meaning': 'complete-record/edge/source union reverse-impact hold; never promotion permission',
    }


def closure_report(graph: dict[str, Any]) -> dict[str, Any]:
    """Machine-readable D0–D7 closure report for the campaign queue."""
    validate_graph_fail_closed(graph)
    by_layer: dict[str, list[dict[str, Any]]] = {}
    controlling_illegal: list[dict[str, Any]] = []
    for nid, node in sorted(graph['nodes'].items()):
        layer = node.get('layer', '?')
        by_layer.setdefault(layer, []).append({
            'id': nid,
            'classification': node['classification'],
            'controlling': bool(node.get('controlling')),
            'kind': node.get('kind'),
        })
        if node.get('controlling'):
            decision = promotion_allowed(graph, nid)
            if not decision['allowed']:
                controlling_illegal.append(decision)

    layers = sorted(by_layer)
    open_active = [nid for nid, n in sorted(graph['nodes'].items())
                   if n['classification'] in ('OPEN_ACTIVE', 'OPEN_HISTORICAL',
                                              'AUTHOR_SIDE_CANDIDATE', 'AUTHOR_SIDE_REDUCTION',
                                              'HOLD', 'REVALIDATION_REQUIRED')]
    blocked = [nid for nid, n in sorted(graph['nodes'].items())
               if n['classification'] == 'BLOCKED_ABSENT']
    return {
        'object': graph.get('object'),
        'layers': {layer: by_layer[layer] for layer in layers},
        'open_or_author_side': open_active,
        'hold_proposals': [
            {'node': nid, 'unsatisfied_required': [
                dep for dep in transitive_required(graph, nid)
                if graph['nodes'][dep]['classification'] not in REQUIRED_SATISFIED
            ]}
            for nid in sorted(graph['nodes'])
            if any(graph['nodes'][dep]['classification'] not in REQUIRED_SATISFIED
                   for dep in transitive_required(graph, nid))
        ],
        'blocked_absent': blocked,
        'illegal_controlling': controlling_illegal,
        'lemma_closed': False,
        'scientific_effect': 'NONE',
        'gate_ok': len(controlling_illegal) == 0,
        'meaning': 'dependency closure inventory; publication is not acceptance',
    }


def d4_region_complement(graph: dict[str, Any]) -> dict[str, Any]:
    """Explicit D4/D5 region map with coverage attributed to its exact source."""
    fixed_remote = []
    other_scoped = []
    proved_reviewed = []
    open_regions = []
    for nid, node in sorted(graph['nodes'].items()):
        if node.get('kind') != 'region':
            continue
        entry = {
            'id': nid,
            'classification': node['classification'],
            'fingerprint': node.get('fingerprint'),
            'notes': node.get('notes'),
            'coverage_source': node.get('coverage_source'),
        }
        if node['classification'] == 'COVERED_BY_CANDIDATE':
            if node.get('coverage_source') == 'math.rn-fixed-remote-window':
                fixed_remote.append(entry)
            else:
                other_scoped.append(entry)
        elif node['classification'] == 'PROVED_REVIEWED':
            proved_reviewed.append(entry)
        else:
            open_regions.append(entry)
    return {
        'covered_by_fixed_remote_candidate': fixed_remote,
        'covered_by_other_scoped_candidates': other_scoped,
        'proved_reviewed_regions': proved_reviewed,
        'open_complement': open_regions,
        'no_event_to_expectation_reversal': True,
        'legacy_24jet_discharged': False,
        'meaning': 'region inventory for #76/#86 D4-D5; each covered region retains its source scope',
    }


def load_selector_region(path: Path | None = None) -> dict[str, Any]:
    data = json.loads((path or SELECTOR_REGION_PATH).read_text())
    if data.get('schema_version') != 1:
        raise ValueError('unsupported selector-region schema_version')
    if 'selectors' not in data or 'regions' not in data:
        raise ValueError('selector-region requires selectors and regions')
    return data


def selector_region_report(data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Summarize which historical selectors remain open on which RN regions."""
    table = data or load_selector_region()
    open_cells = []
    covered_or_bypassed = []
    for selector, row in sorted(table['selectors'].items()):
        for region in table['regions']:
            status = row[region]
            cell = {'selector': selector, 'region': region, 'status': status}
            if status in (
                'OPEN_ACTIVE', 'OPEN_HISTORICAL', 'NOT_DISCHARGED',
                'PARTIAL_COVER_ONLY', 'PARTIAL_PR7_REDUCTION', 'REOPENED',
                'PARTIAL_COVER_DECLARED_REGION',
            ):
                open_cells.append(cell)
            else:
                covered_or_bypassed.append(cell)
    return {
        'object': table.get('object'),
        'open_or_partial_cells': open_cells,
        'covered_bypassed_or_na_cells': covered_or_bypassed,
        'covered_region_ids': table.get('covered_region_ids'),
        'open_region_ids': table.get('open_region_ids'),
        'no_event_to_expectation_reversal': table.get('no_event_to_expectation_reversal'),
        'legacy_24jet_discharged': table.get('legacy_24jet_discharged'),
        'meaning': table.get('meaning'),
    }


def d0_ci_unblock_report() -> dict[str, Any]:
    """Record the exact main PR87 packet-allowlist diagnosis and portable patch identity."""
    patch = D0_PATCH_PATH.read_bytes()
    digest = hashlib.sha256(patch).hexdigest()
    text = patch.decode()
    required_markers = (
        'DOWNSTREAM_CROSSWALK_20260925.md',
        'TRANSCRIPTION_NAMES',
        'None promotes D3 lemma_closed.',
        'd53b286029d034576245406152c188fe2470ccb089e51f73e9087e34bf5a0e10',
    )
    doc = (ROOT / 'D0_CI_UNBLOCK.md').read_text()
    missing_doc = [m for m in ('unexpected files', '403', 'problems=0') if m not in doc]
    missing_patch = [m for m in required_markers if m not in text]
    return {
        'ci_error': "PROBLEM packet: unexpected files ['DOWNSTREAM_CROSSWALK_20260925.md']",
        'main_pr': 87,
        'patch_path': 'patches/d0_packet_allowlist.patch',
        'patch_bytes': len(patch),
        'patch_sha256': digest,
        'cursor_main_push': 'DENIED_403',
        'local_math_status_check_after_patch': 'problems=0',
        'missing_doc_markers': missing_doc,
        'missing_patch_markers': missing_patch,
        'ready': not missing_doc and not missing_patch,
        'lemma_closed': False,
        'meaning': 'engineering unblock recipe; not scientific promotion',
    }


def results_payload(graph: dict[str, Any] | None = None) -> dict[str, Any]:
    g = graph or load_graph()
    # Spot-check promotions that must fail closed on the live author-side graph.
    illegal_attempts = {
        'promote_fixed_remote': refuse_non_discharge_promotion(
            g, 'math.rn-fixed-remote-window',
            ['GREEN_CI', 'HASH_MATCH', 'SAME_AUTHOR_REVIEW']),
        'promote_lifetime_remainder': promotion_allowed(g, 'math.lifetime-remainder'),
        'promote_side24': promotion_allowed(g, 'math.side24-coefficient'),
        'promote_historical_env_rescov': promotion_allowed(g, 'hist.ENV-RESCOV'),
        'promote_pr87_before_allowlist': promotion_allowed(g, 'eng.main-pr87-crosswalk'),
    }
    impact = reverse_impact(
        g, 'math.uniform-matrix-cap-lifetime',
        old_fingerprint='main-63-author-side',
        new_fingerprint='main-63-amended-demo',
    )
    d0 = d0_ci_unblock_report()
    selectors = selector_region_report()
    return {
        'object': g.get('object'),
        'schema_version': g.get('schema_version'),
        'gate_ok': closure_report(g)['gate_ok'],
        'lemma_closed': False,
        'scientific_effect': 'NONE',
        'illegal_promotion_refused': all(
            (v.get('refused') if 'refused' in v else not v.get('allowed'))
            for v in illegal_attempts.values()
        ),
        'illegal_attempts': {
            k: {key: val for key, val in v.items() if key != 'base'}
            for k, v in illegal_attempts.items()
        },
        'reverse_impact_demo': {
            'changed_node': impact['changed_node'],
            'impacted': impact['impacted'],
            'dependency_changed': impact['dependency_changed'],
        },
        'd4_region_complement': d4_region_complement(g),
        'selector_region': {
            'open_or_partial_cell_count': len(selectors['open_or_partial_cells']),
            'covered_bypassed_or_na_cell_count': len(selectors['covered_bypassed_or_na_cells']),
            'legacy_24jet_discharged': selectors['legacy_24jet_discharged'],
            'no_event_to_expectation_reversal': selectors['no_event_to_expectation_reversal'],
        },
        'd0_ci_unblock': {
            'ready': d0['ready'],
            'cursor_main_push': d0['cursor_main_push'],
            'patch_sha256': d0['patch_sha256'],
            'patch_bytes': d0['patch_bytes'],
            'ci_error': d0['ci_error'],
        },
        'closure': {
            'blocked_absent': closure_report(g)['blocked_absent'],
            'open_or_author_side_count': len(closure_report(g)['open_or_author_side']),
            'illegal_controlling_count': len(closure_report(g)['illegal_controlling']),
        },
        'meaning': (
            'same-author integrity controls for main #90/#86; '
            'not analytic review or theorem acceptance'
        ),
    }


def main() -> None:
    payload = results_payload()
    text = json.dumps(payload, indent=2, sort_keys=True) + '\n'
    print(text, end='')
    # When executed as the package entry point, compare to pinned RESULTS.json.
    expected = (ROOT / 'RESULTS.json').read_text()
    if text != expected:
        raise SystemExit('RESULTS.json byte mismatch; regenerate deliberately')


if __name__ == '__main__':
    main()
