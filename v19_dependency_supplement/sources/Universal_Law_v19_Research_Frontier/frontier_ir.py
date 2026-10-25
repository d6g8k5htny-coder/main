"""Hash-bound semantic AND hypergraph; no independent mathematical authority."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / 'engine') not in sys.path:
    sys.path.insert(0, str(ROOT / 'engine'))
from model import validate as validate_dataset, fingerprints

SCHEMA = 'UL-frontier-ir-1'
EFFECTS = ('artifact.reference', 'failure.report', 'guidance.propose',
           'plan.propose', 'source.inspect')
SHA256 = re.compile(r'^[0-9a-f]{64}$')
MAX_BYTES = 8 * 1024 * 1024


def bounded_json(value, name='value', max_bytes=MAX_BYTES):
    """Bound JSON before hashing, rejecting cycles, NaN and Python coercions."""
    active = set()
    def visit(item, depth):
        if depth > 24:
            raise ValueError(name + ' nesting bound exceeded')
        if item is None or type(item) is bool:
            return
        if type(item) is int:
            if abs(item) > 2**53:
                raise ValueError(name + ' integer bound exceeded')
            return
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError(name + ' nonfinite number')
            return
        if type(item) is str:
            if len(item) > 100000:
                raise ValueError(name + ' string bound exceeded')
            return
        if type(item) not in (dict, list) or len(item) > 4096:
            raise ValueError(name + ' must contain bounded JSON types')
        if id(item) in active:
            raise ValueError(name + ' contains a cycle')
        active.add(id(item))
        if type(item) is dict:
            for key, child in item.items():
                if type(key) is not str or len(key) > 256:
                    raise ValueError(name + ' invalid object key')
                visit(child, depth + 1)
        else:
            for child in item:
                visit(child, depth + 1)
        active.remove(id(item))
    visit(value, 0)
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False,
                     allow_nan=False, separators=(',', ':')).encode('utf8')
    if len(raw) > max_bytes:
        raise ValueError(name + ' byte bound exceeded')
    return copy.deepcopy(value)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(',', ':')).encode('utf8')).hexdigest()


def require_hash(value, name):
    if type(value) is not str or SHA256.fullmatch(value) is None:
        raise ValueError('invalid ' + name + ' SHA256 hash')
    return value


def _semantic_types(data):
    def text(value):
        if type(value) is not str or not value.strip() or len(value)>50000:
            raise ValueError('invalid semantic text type or bound')
    def identifiers(value):
        if type(value) is not list or len(value)>256:
            raise ValueError('invalid semantic identifier list type')
        for item in value:
            text(item)
            if len(item)>128:
                raise ValueError('semantic identifier bound exceeded')
    if type(data) is not dict:
        raise ValueError('dataset must be a typed object')
    for group in ('records','obligations','imports','corrections','local_evidence','audit_history'):
        if type(data.get(group)) is not list or any(type(item) is not dict for item in data[group]):
            raise ValueError('invalid semantic collection type: '+group)
    for record in data['records']:
        for key in ('id','title','topic','statement','scope','source'):
            text(record.get(key))
        for key in ('source_authenticated','full_theorem_checker_passed','independently_reproduced_full_theorem'):
            if type(record.get(key)) is not bool:
                raise ValueError('invalid semantic boolean type: '+key)
        if record.get('source_content_sha256') is not None:
            require_hash(record['source_content_sha256'],'source content')
    for obligation in data['obligations']:
        for key in ('id','conclusion'):
            text(obligation.get(key))
        for key in ('premises','external_theorem_imports','source_corrections','local_evidence_refs'):
            identifiers(obligation.get(key))
        conditions = obligation.get('side_conditions')
        if type(conditions) is not list or not 1<=len(conditions)<=128:
            raise ValueError('invalid semantic side condition collection type')
        for condition in conditions:
            if type(condition) is not dict:
                raise ValueError('invalid semantic side condition type')
            for key in ('id','origin','text','validation'):
                text(condition.get(key))
    for imported in data['imports']:
        for key in ('id','title','scope_description','statement_or_required_evidence'):
            text(imported.get(key))
        identifiers(imported.get('applies_to'))
        if imported.get('source_content_sha256') is not None:
            require_hash(imported['source_content_sha256'],'import source')
    for correction in data['corrections']:
        for key in ('id','title','role','source_url'):
            text(correction.get(key))
        for key in ('authenticated','validated'):
            if type(correction.get(key)) is not bool:
                raise ValueError('invalid correction semantic boolean type')
        if correction.get('source_content_sha256') is not None:
            require_hash(correction['source_content_sha256'],'correction source')


def compile_ir(dataset, branch_sha256=None):
    """Compile only model-validated input; returned JSON owns its source copy."""
    data = bounded_json(dataset, 'dataset')
    _semantic_types(data)
    try:
        validate_dataset(data)
    except (KeyError, TypeError, AttributeError, IndexError, RecursionError) as exc:
        raise ValueError('invalid source dataset') from exc
    dataset_hash = digest(data)
    branch_hash = dataset_hash if branch_sha256 is None else require_hash(branch_sha256, 'branch')
    fp = fingerprints(data)
    nodes = []
    for record in sorted(data['records'], key=lambda r: r['id']):
        node = {'id': record['id'], 'kind': 'statement', 'title': record['title'],
                'topic': record['topic'], 'acceptance': record['project_acceptance'],
                'statement': record['statement'], 'scope': record['scope'],
                'source': record['source'], 'input_fingerprint': fp[record['id']],
                'statement_sha256': digest(record['statement']),
                'scope_sha256': digest(record['scope']), 'certified': False}
        node['node_sha256'] = digest(node)
        nodes.append(node)
    edges = []
    for obligation in sorted(data['obligations'], key=lambda o: o['conclusion']):
        edge = {'id': obligation['id'], 'kind': 'AND',
                'conclusion': obligation['conclusion'],
                'premises': copy.deepcopy(obligation['premises']),
                'side_conditions': copy.deepcopy(obligation['side_conditions']),
                'imports': copy.deepcopy(obligation['external_theorem_imports']),
                'corrections': copy.deepcopy(obligation['source_corrections']),
                'local_evidence': copy.deepcopy(obligation['local_evidence_refs']),
                'joint_premises_not_sufficient': True,
                'unlisted_side_conditions_possible': True,
                'effects': list(EFFECTS)}
        edge['edge_sha256'] = digest(edge)
        edges.append(edge)
    result = {'schema_version': SCHEMA, 'dataset_sha256': dataset_hash,
              'branch_sha256': branch_hash, 'source_dataset': data,
              'nodes': nodes, 'hyperedges': edges,
              'imports': sorted(copy.deepcopy(data['imports']), key=lambda i: i['id']),
              'corrections': sorted(copy.deepcopy(data['corrections']), key=lambda c: c['id']),
              'effects': list(EFFECTS)}
    result['ir_sha256'] = digest(result)
    return result


def validate_ir(ir):
    """Recompile instead of trusting self-supplied hashes or missing effects.

    This verifies the projection against its validated source, not source custody
    or theorem truth. Callers must bind branch_sha256 to their live coordinator.
    """
    value = bounded_json(ir, 'IR')
    if type(value) is not dict or value.get('schema_version') != SCHEMA:
        raise ValueError('invalid IR schema')
    for name in ('dataset_sha256', 'branch_sha256', 'ir_sha256'):
        require_hash(value.get(name), name)
    if 'source_dataset' not in value:
        raise ValueError('IR source dataset missing')
    expected = compile_ir(value['source_dataset'], value['branch_sha256'])
    if value != expected:
        raise ValueError('IR projection, effects or hash mismatch with validated source')
    return {'status': 'OPERATIONAL_PROJECTION_NOT_MATH_CERTIFIED',
            'ir_sha256': value['ir_sha256'], 'nodes': len(value['nodes']),
            'hyperedges': len(value['hyperedges']), 'full_theorem_certificates': 0}
