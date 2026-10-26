"""Source-bound investigation suggestions; no predicted mathematical premises."""
from __future__ import annotations

from frontier_ir import digest, validate_ir


def bounded_integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(name + ' must be an integer in the permitted bounds')
    return value


def rank_connections(ir, limit=10, seed=0):
    """Rank explicit open interfaces using a disclosed hand-built heuristic.

    Scores prioritize documented gaps, unvalidated imports and downstream reach.
    They are neither likelihoods nor proofs of logically necessary premises.
    Seed only provides a deterministic tie-break; no graph is changed.
    """
    validate_ir(ir)
    bounded_integer(limit, 'limit', 0, 100)
    bounded_integer(seed, 'seed', -(2**31), 2**31 - 1)
    nodes = {n['id']: n for n in ir['nodes']}
    imports = {i['id']: i for i in ir['imports']}
    children = {n: set() for n in nodes}
    for edge in ir['hyperedges']:
        for premise in edge['premises']:
            children[premise].add(edge['conclusion'])
    candidates = []

    def add(kind, source_id, target_id, required, explanation, priority, source_reference, source_scope=None):
        target = nodes[target_id]
        reached, pending = set(), list(children[target_id])
        while pending:
            child = pending.pop()
            if child not in reached:
                reached.add(child)
                pending.extend(children[child])
        item = {'kind': kind, 'source_id': source_id, 'target_id': target_id,
                'required_interface': required, 'explanation': explanation,
                'source_reference': source_reference, 'source_scope': source_scope,
                'target_scope': target['scope'], 'input_fingerprint': target['input_fingerprint'],
                'ir_sha256': ir['ir_sha256'], 'status': 'UNVERIFIED_SUGGESTION',
                'certified': False, 'installable': False,
                'ranking_basis': 'hand-built heuristic: documented gap priority plus recorded downstream reach; not logical necessity',
                'score': priority + min(len(reached), 20) * 0.1,
                'score_components': {'documented_priority': priority, 'downstream_records': len(reached)}}
        item['id'] = 'SUG:' + digest([kind, source_id, target_id, required, ir['ir_sha256']])[:24]
        candidates.append(item)

    for edge in ir['hyperedges']:
        target = nodes[edge['conclusion']]
        for import_id in edge['imports']:
            descriptor = imports[import_id]
            if descriptor['import_status'] != 'independently_certified':
                add('unvalidated_theorem_interface', import_id, target['id'],
                    descriptor['statement_or_required_evidence'],
                    'The recorded external import is not independently certified for this exact target scope; inspect source and applicability hypotheses.',
                    90 if target['id'] == 'D1-KR-BOREL' else 70,
                    descriptor.get('source_reference') or target['source'], descriptor['scope_description'])
        for condition in edge['side_conditions']:
            if condition.get('validation') != 'independently_discharged':
                add('open_scope_interface', condition['id'], target['id'], condition['text'],
                    'The source records this condition as open. The heuristic recommends checking its exact scope without adding a graph premise.',
                    40, target['source'], target['scope'])
    # This explicit source scope states the missing direction. It is not inferred
    # by semantic negation, a contradiction detector, or a learned edge.
    lower = nodes.get('D5d-M')
    if lower and 'no matching upper bound for the second factorial moment' in lower['scope']:
        add('missing_collision_interface', lower['id'], lower['id'],
            'A matching upper second factorial moment requires a separate conditioned multipoint collision analysis, including witness/witness and witness/pin contact regimes.',
            'D5d-M explicitly supplies a lower bound and excludes a matching upper bound. The missing upper estimate is an investigation target; fixed off-contact or first-moment bounds cannot discharge it automatically.',
            100, lower['source'], lower['scope'])
    neighbors = {node_id: set(children[node_id]) for node_id in nodes}
    existing = set()
    for edge in ir['hyperedges']:
        for premise in edge['premises']:
            neighbors[edge['conclusion']].add(premise)
            existing.add((premise, edge['conclusion']))
    node_ids = sorted(nodes)
    for index, source_id in enumerate(node_ids):
        for target_id in node_ids[index + 1:]:
            if (source_id, target_id) in existing or (target_id, source_id) in existing:
                continue
            source, target = nodes[source_id], nodes[target_id]
            common = sorted(neighbors[source_id] & neighbors[target_id])
            if not common or (source['topic'] != target['topic'] and source['source'] != target['source']):
                continue
            add('structural_connection', source_id, target_id,
                'Review whether the two scoped interfaces can share a source audit or auxiliary construction.',
                'These records share recorded hypergraph neighbors ' + ', '.join(common) +
                '. Graph proximity is a heuristic investigation cue; no implication or necessary premise is inferred and scope compatibility remains unverified.',
                45 + min(len(common), 5), source['source'], source['scope'])
    candidates.sort(key=lambda item: (-item['score'], digest([seed, item['id']])))
    return candidates[:limit]
