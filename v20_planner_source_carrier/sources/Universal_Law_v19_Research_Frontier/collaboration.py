"""Annotation-only dotted multi-value registers with observed-remove deletion.

Operations are untrusted observations, not authenticated remote identities. Their
union cannot write mathematical records, acceptance, jobs or certificates.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'engine'))
from conductor import Journal

SCHEMA = 'UL-annotation-crdt-1'
MAX_OPERATIONS = 10000
MAX_STATE_BYTES = 2 * 1024 * 1024
MAX_TEXT_BYTES = 8192
TAGS = frozenset({'note', 'question', 'todo', 'source', 'review'})
IDENTITY = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,119}')


def _id(value, field):
    if not isinstance(value, str) or not IDENTITY.fullmatch(value):
        raise ValueError('invalid ' + field)
    return value


def _json_size(value):
    try:
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode('utf8')
    except (ValueError, TypeError, RecursionError, UnicodeError) as exc:
        raise ValueError('annotation state must be bounded pure JSON') from exc
    if len(raw) > MAX_STATE_BYTES:
        raise ValueError('annotation state exceeds byte budget')


def _dot(value):
    if (not isinstance(value, list) or len(value) != 2
            or type(value[1]) is not int or not 1 <= value[1] <= 10**9):
        raise ValueError('invalid operation dot/counter')
    return (_id(value[0], 'actor'), value[1])


class AnnotationCRDT:
    """Union-of-operations MV-register. Concurrent sets survive observed deletes.

    Actor/counter collisions with different payloads fail closed. Replica actors
    must be unique by convention; this module supplies no authentication service.
    """
    def __init__(self, actor, record_ids=None):
        self.actor = _id(actor, 'actor')
        if record_ids is not None and not isinstance(record_ids, (list, tuple, set, frozenset)):
            raise ValueError('record IDs must be a bounded collection')
        if record_ids is not None and len(record_ids) > 10000:
            raise ValueError('too many record IDs')
        self.record_ids = None if record_ids is None else frozenset(_id(x, 'record ID') for x in record_ids)
        self._operations = {}

    def _validate(self, operations):
        if not isinstance(operations, list) or len(operations) > MAX_OPERATIONS:
            raise ValueError('operation budget exceeded')
        _json_size(operations)
        result, namespaces = {}, {}
        for operation in operations:
            if not isinstance(operation, dict):
                raise ValueError('annotation operation must be an object')
            action = operation.get('action')
            fields = {'actor', 'counter', 'action', 'annotation_id', 'observed'}
            if action == 'set':
                fields |= {'record_id', 'text', 'tag'}
            elif action != 'delete':
                raise ValueError('annotation-only operation required')
            if set(operation) != fields:
                raise ValueError('annotation-only fields required; scope write rejected')
            dot = _dot([operation['actor'], operation['counter']])
            annotation = _id(operation['annotation_id'], 'annotation ID')
            observed = operation['observed']
            if not isinstance(observed, list) or len(observed) > MAX_OPERATIONS:
                raise ValueError('invalid observed removal list')
            dots = [_dot(value) for value in observed]
            if len(dots) != len(set(dots)) or dots != sorted(dots):
                raise ValueError('observed dots must be unique and sorted')
            if any(d == dot or (d[0] == dot[0] and d[1] >= dot[1]) for d in dots):
                raise ValueError('operation cannot observe its own future counter')
            if action == 'set':
                record = _id(operation['record_id'], 'record ID')
                if self.record_ids is not None and record not in self.record_ids:
                    raise ValueError('unknown annotation record ID')
                if annotation in namespaces and namespaces[annotation] != record:
                    raise ValueError('annotation record namespace cannot be repointed')
                namespaces[annotation] = record
                text = operation['text']
                if (not isinstance(text, str) or not text.strip()
                        or len(text.encode('utf8')) > MAX_TEXT_BYTES):
                    raise ValueError('invalid annotation text or text byte budget')
                if not isinstance(operation['tag'], str) or operation['tag'] not in TAGS:
                    raise ValueError('unsupported annotation tag')
            if dot in result and result[dot] != operation:
                raise ValueError('duplicate actor counter has a different payload')
            result[dot] = deepcopy(operation)
        # Every removal must be an actually observed set in the same namespace.
        children = {dot: [] for dot in result}
        indegree = {dot: 0 for dot in result}
        for dot, operation in result.items():
            for seen in map(_dot, operation['observed']):
                prior = result.get(seen)
                if (prior is None or prior['action'] != 'set'
                        or prior['annotation_id'] != operation['annotation_id']):
                    raise ValueError('unobserved or cross-namespace removal dot')
                children[seen].append(dot); indegree[dot] += 1
        ready = deque(dot for dot, degree in indegree.items() if degree == 0)
        visited = 0
        while ready:
            visited += 1
            for child in children[ready.popleft()]:
                indegree[child] -= 1
                if indegree[child] == 0: ready.append(child)
        if visited != len(result):
            raise ValueError('cyclic fabricated observations')
        return result

    @staticmethod
    def _annotations(operations):
        removed = {tuple(dot) for op in operations.values() for dot in op['observed']}
        annotations = {}
        for dot, op in sorted(operations.items()):
            if op['action'] != 'set' or dot in removed:
                continue
            entry = annotations.setdefault(op['annotation_id'], {
                'annotation_id': op['annotation_id'], 'record_id': op['record_id'], 'versions': []})
            entry['versions'].append({'dot': list(dot), 'text': op['text'], 'tag': op['tag']})
        return [annotations[key] for key in sorted(annotations)]

    def _state(self, operations):
        return {'schema': SCHEMA, 'actor': self.actor,
                'operations': [deepcopy(v) for _, v in sorted(operations.items())],
                'annotations': self._annotations(operations),
                'actor_identity': 'unauthenticated', 'remote_multiuser': 'unconfigured'}

    def _accept(self, operations):
        # Derived concurrent-version text is part of the wire state budget too.
        _json_size(self._state(operations))
        self._operations = operations

    def _write(self, action, annotation_id, **values):
        _id(annotation_id, 'annotation ID')
        removed = {tuple(dot) for op in self._operations.values() for dot in op['observed']}
        observed = [list(dot) for dot, op in sorted(self._operations.items())
                    if op['action'] == 'set' and op['annotation_id'] == annotation_id and dot not in removed]
        counter = max((dot[1] for dot in self._operations if dot[0] == self.actor), default=0) + 1
        operation = {'actor': self.actor, 'counter': counter, 'action': action,
                     'annotation_id': annotation_id, 'observed': observed, **values}
        candidate = self._validate(list(self._operations.values()) + [operation])
        self._accept(candidate)
        return deepcopy(operation)

    def set(self, annotation_id, record_id, text, tag='note'):
        return self._write('set', annotation_id, record_id=record_id, text=text, tag=tag)

    def delete(self, annotation_id):
        return self._write('delete', annotation_id)

    def merge(self, other):
        """Validate then atomically union operations; unknown authority fields fail."""
        state = other.snapshot() if isinstance(other, AnnotationCRDT) else other
        _json_size(state)
        allowed = {'schema', 'actor', 'operations', 'annotations', 'actor_identity', 'remote_multiuser'}
        if (not isinstance(state, dict) or not {'schema', 'operations'} <= set(state)
                or set(state) - allowed or state['schema'] != SCHEMA):
            raise ValueError('annotation-only state schema required; scope write rejected')
        if 'actor' in state: _id(state['actor'], 'actor')
        if state.get('actor_identity', 'unauthenticated') != 'unauthenticated':
            raise ValueError('remote actor authentication is unconfigured')
        if state.get('remote_multiuser', 'unconfigured') != 'unconfigured':
            raise ValueError('remote multiuser service is unconfigured')
        incoming = self._validate(state['operations'])
        if 'annotations' in state and state['annotations'] != self._annotations(incoming):
            raise ValueError('derived annotations do not match operation state')
        union = dict(self._operations)
        for dot, operation in incoming.items():
            if dot in union and union[dot] != operation:
                raise ValueError('duplicate actor counter has a different payload')
            union[dot] = operation
        candidate = self._validate(list(union.values()))
        self._accept(candidate)
        return self.snapshot()

    def snapshot(self):
        return self._state(self._operations)


class AnnotationStore:
    """Durable local annotation service. Journals only annotation operations."""
    def __init__(self, workdir, record_ids=None):
        self.record_ids = record_ids
        self.journal = Journal(workdir)
        self.snapshot()

    def _replay(self, actor='local-owner'):
        self.journal._load()
        replica = AnnotationCRDT(actor, self.record_ids)
        for event in self.journal.events:
            if event['type'] != 'ANNOTATIONS_MERGED' or set(event['payload']) != {'operations'}:
                raise ValueError('non-annotation event in annotation journal')
            operations = list(replica._operations.values()) + event['payload']['operations']
            replica._accept(replica._validate(operations))
        return replica

    def apply(self, change, actor='local-owner'):
        if not isinstance(change, dict): raise ValueError('annotation change must be an object')
        action = change.get('action')
        fields = {'action', 'annotation_id'}
        if action == 'set':
            fields |= {'record_id', 'text'}
            if 'tag' in change: fields.add('tag')
        elif action != 'delete': raise ValueError('annotation-only action required')
        if set(change) != fields: raise ValueError('annotation-only fields required')
        with self.journal.transaction():
            replica = self._replay(actor)
            if action == 'set':
                operation = replica.set(change['annotation_id'], change['record_id'],
                                        change['text'], change.get('tag', 'note'))
            else: operation = replica.delete(change['annotation_id'])
            self.journal.append('ANNOTATIONS_MERGED', {'operations': [operation]})
            return replica.snapshot()

    def merge(self, state):
        with self.journal.transaction():
            replica = self._replay()
            before = set(replica._operations)
            replica.merge(state)
            added = [v for dot, v in sorted(replica._operations.items()) if dot not in before]
            if added: self.journal.append('ANNOTATIONS_MERGED', {'operations': added})
            return replica.snapshot()

    def snapshot(self):
        with self.journal.transaction(): return self._replay().snapshot()
