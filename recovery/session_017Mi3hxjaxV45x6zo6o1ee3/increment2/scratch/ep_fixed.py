#!/usr/bin/env python3
"""Read-only dependency tree and evidence profile for the downstream gate graph (standard library only).

Renders, for a node of frontiers/downstream_gate_20260925/GRAPH.json (or every math.* root), the tree of
its outgoing edges with each node's recorded classification, the providers its record names, and the
required dependencies that do not yet satisfy the hard gate. The graph is loaded and validated with the
gate's own strict loader; the dependency rule is the gate's (an edge is required unless it says
otherwise; only PROVED_REVIEWED satisfies a required premise).

This is the derived view proposed in main#275 (decision record v1.1, rows PROPOSED) and main#276
(governance/OP-CLOSURE-EVIDENCE-20261006.md, unmerged at publication): it writes
nothing, promotes nothing and is not a status record. Axes the graph does not record (blind
reconstruction, adversarial attack, formal evidence, numerical reproduction, novelty, human reading)
are listed as outside the graph rather than reported as passed or failed. Scientific effect: NONE.

Usage: python3 -B -S tools/evidence_profile.py [--graph PATH] [--node ID] [--json]
"""
import argparse
import importlib.util
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
GATE_DIR = ROOT / 'frontiers' / 'downstream_gate_20260925'
AXES_OUTSIDE_GRAPH = (
    'blind_reconstruction', 'adversarial_attack', 'formal_evidence',
    'numerical_reproduction', 'novelty', 'human_reading',
)
FAMILIES = (('openai', 'OpenAI'), ('chatgpt', 'OpenAI'), ('codex', 'OpenAI'), ('anthropic', 'Anthropic'),
            ('claude', 'Anthropic'), ('xai', 'xAI'), ('grok', 'xAI'), ('google', 'Google'), ('gemini', 'Google'))


def load_gate():
    spec = importlib.util.spec_from_file_location('downstream_hard_gate', GATE_DIR / 'hard_gate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def family(text):
    """Provider family named by a free-form record string, or None when it names none or several.
    Matching is on whole alphanumeric tokens, so 'xAI/Grok via Cursor' names xAI but 'notopenai' names nothing."""
    tokens = set(re.findall(r'[a-z0-9]+', text.lower()))
    found = {name for token, name in FAMILIES if token in tokens}
    return found.pop() if len(found) == 1 else None


def reviewer_strings(node):
    out = []
    for item in node.get('review_basis', []) or []:
        if isinstance(item, dict) and isinstance(item.get('provider'), str):
            out.append(item['provider'])
    providers = node.get('review_providers')
    if isinstance(providers, list):
        out.extend(p for p in providers if isinstance(p, str))
    if isinstance(node.get('review_provider'), str):
        out.append(node['review_provider'])
    return out


def provider_distinct(node):
    """'yes' if some recorded reviewer resolves to a family other than the recorded author's; 'no' only if
    every recorded reviewer resolves to the author's family; otherwise 'not recorded' (missing author or
    reviewers, or a reviewer string naming no single family). Never organizational independence."""
    author = node.get('author_provider')
    author_family = family(author) if isinstance(author, str) else None
    reviewer_families = [family(s) for s in reviewer_strings(node)]
    if author_family is None or not reviewer_families:
        return 'not recorded'
    if any(f is not None and f != author_family for f in reviewer_families):
        return 'yes'
    if all(f == author_family for f in reviewer_families):
        return 'no'
    return 'not recorded'


def profile(gate, graph, node_id):
    node = gate.require_node(graph, node_id)
    unsatisfied = [dep for dep in gate.transitive_required(graph, node_id)
                   if graph['nodes'][dep]['classification'] not in gate.REQUIRED_SATISFIED]
    return {
        'node': node_id,
        'classification': node['classification'],
        'review_disposition': node.get('review_disposition', 'not recorded'),
        'author_provider': node.get('author_provider', 'not recorded'),
        'reviewer_providers': reviewer_strings(node) or ['not recorded'],
        'provider_distinct': provider_distinct(node),
        'unsatisfied_required': [{'node': dep, 'classification': graph['nodes'][dep]['classification']}
                                 for dep in unsatisfied],
        'axes_outside_graph': list(AXES_OUTSIDE_GRAPH),
    }


def roots(graph):
    targets = {e['to'] for e in graph['edges']}
    return sorted(n for n in graph['nodes'] if n.startswith('math.') and n not in targets)


def tree_lines(gate, graph, node_id):
    lines = []
    seen = set()

    def label(nid):
        p = profile(gate, graph, nid)
        return '%s  %s  [author %s; reviewers %s; provider-distinct %s]' % (
            nid, p['classification'], p['author_provider'], ', '.join(p['reviewer_providers']),
            p['provider_distinct'])

    def walk(nid, prefix):
        edges = sorted((e for e in graph['edges'] if e['from'] == nid), key=lambda e: e['to'])
        for i, e in enumerate(edges):
            last = i == len(edges) - 1
            kind = 'requires' if e.get('required', True) else 'context (%s)' % e.get('relation', '?')
            child = e['to']
            again = child in seen
            lines.append('%s%s %s %s%s' % (prefix, '└──' if last else '├──', kind, label(child),
                                          '  (expanded above)' if again else ''))
            if not again:
                seen.add(child)
                walk(child, prefix + ('    ' if last else '│   '))

    seen.add(node_id)
    lines.append(label(node_id))
    walk(node_id, '')
    unsatisfied = profile(gate, graph, node_id)['unsatisfied_required']
    if unsatisfied:
        lines.append('Unsatisfied required dependencies (%d): %s' % (len(unsatisfied), ', '.join(
            '%s (%s)' % (u['node'], u['classification']) for u in unsatisfied)))
    else:
        lines.append('Unsatisfied required dependencies: none')
    return lines


def report(gate, graph, node_ids):
    return {
        'object': graph.get('object'),
        'profiles': [profile(gate, graph, nid) for nid in node_ids],
        'axes_outside_graph': list(AXES_OUTSIDE_GRAPH),
        'meaning': 'derived read-only view of recorded graph fields; not a status record or acceptance',
        'status_authority': False,
        'lemma_closed': False,
        'scientific_effect': 'NONE',
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--graph', default=str(GATE_DIR / 'GRAPH.json'))
    ap.add_argument('--node', action='append', help='node id (repeatable); default: every math.* root')
    ap.add_argument('--json', action='store_true', help='print the profiles as JSON instead of trees')
    args = ap.parse_args(argv)
    gate = load_gate()
    graph = gate.load_graph(pathlib.Path(args.graph))
    node_ids = args.node or roots(graph)
    for nid in node_ids:
        gate.require_node(graph, nid)
    if args.json:
        print(json.dumps(report(gate, graph, node_ids), indent=2, sort_keys=True))
        return 0
    for index, nid in enumerate(node_ids):
        if index:
            print()
        print('\n'.join(tree_lines(gate, graph, nid)))
    print()
    print('Not in the graph (consult review records): ' + ', '.join(AXES_OUTSIDE_GRAPH))
    print('Derived view only: no status authority; scientific effect NONE.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
