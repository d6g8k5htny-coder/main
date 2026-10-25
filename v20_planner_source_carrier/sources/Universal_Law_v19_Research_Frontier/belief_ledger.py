"""Durable four-valued *reported evidence*, without theorem inference.

Identity is the exact JSON triple (proposition, scope, input fingerprint).
SQLite transactions make repeated evidence idempotent across processes/restarts.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from frontier_ir import bounded_json, digest, require_hash, validate_ir

SCHEMA = 'UL-frontier-beliefs-1'


def _text(value, name, maximum=100000):
    if type(value) is not str or not value.strip() or len(value) > maximum:
        raise ValueError('invalid ' + name)
    return value


def _identity(proposition, scope, input_fingerprint):
    return {'proposition': proposition, 'scope': scope,
            'input_fingerprint': input_fingerprint}


class BeliefLedger:
    """An append-only observation store; reports never grant proof authority."""

    def __init__(self, workdir):
        directory = Path(workdir)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / 'beliefs.sqlite3'
        with self._connection() as connection:
            connection.execute('CREATE TABLE IF NOT EXISTS reports '
                               '(claim_id TEXT PRIMARY KEY, body TEXT NOT NULL)')

    def _connection(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute('PRAGMA synchronous=FULL')
        return connection

    def assert_claim(self, proposition, scope, polarity, source, evidence=None,
                     input_fingerprint=None, record_id=None):
        """Store unvalidated support/refutation, deduplicated by full payload.

        A record_id is only a proposed binding until snapshot(ir) verifies exact
        statement, scope and input fingerprint. Omitted fingerprints are explicitly
        unversioned and cannot attach to a canonical record.
        """
        _text(proposition, 'proposition')
        if type(scope) not in (str, dict) or not scope:
            raise ValueError('scope must be a nonempty exact string or JSON object')
        if type(scope) is str:
            _text(scope, 'scope')
        if polarity not in ('support', 'refute') or type(polarity) is not str:
            raise ValueError('polarity must be support or refute')
        _text(source, 'source', 4096)
        if input_fingerprint is not None:
            require_hash(input_fingerprint, 'input fingerprint')
        if record_id is not None:
            _text(record_id, 'record ID', 256)
            if input_fingerprint is None:
                raise ValueError('record binding requires an input fingerprint')
        body = bounded_json(dict(_identity(proposition, scope, input_fingerprint),
                                 polarity=polarity, source=source,
                                 evidence=evidence, record_id=record_id),
                            'evidence report', max_bytes=262144)
        claim_id = digest(body)
        with self._connection() as connection:
            cursor = connection.execute('INSERT OR IGNORE INTO reports VALUES (?, ?)',
                (claim_id, json.dumps(body, ensure_ascii=False, sort_keys=True, allow_nan=False)))
            inserted = cursor.rowcount == 1
        return {'claim_id': claim_id, 'belief_id': digest(_identity(proposition, scope, input_fingerprint)),
                'inserted': inserted, 'status': 'UNVALIDATED_EVIDENCE_REPORT', 'certified': False}

    def snapshot(self, ir=None):
        """Group observations with no vote, explosion or inferred contradiction.

        Contradictions recommend investigating an explicit, version-matched
        record and its recorded descendants. They do not invalidate those records.
        """
        if ir is not None:
            validate_ir(ir)
        nodes = {} if ir is None else {n['id']: n for n in ir['nodes']}
        beliefs = {}

        def ensure(identity):
            key = digest(identity)
            return beliefs.setdefault(key, dict(identity, belief_id=key,
                state='neither', evidence=[], certified=False))

        for node in nodes.values():
            ensure(_identity(node['statement'], node['scope'], node['input_fingerprint']))
        with self._connection() as connection:
            rows = connection.execute('SELECT claim_id, body FROM reports ORDER BY claim_id').fetchall()
        for claim_id, raw in rows:
            body = bounded_json(json.loads(raw), 'stored report', max_bytes=262144)
            if digest(body) != claim_id:
                raise ValueError('stored evidence report integrity mismatch')
            belief = ensure(_identity(body['proposition'], body['scope'], body['input_fingerprint']))
            record = nodes.get(body['record_id'])
            bound = record is not None and all(body[field] == record[field]
                for field in ('scope', 'input_fingerprint')) and body['proposition'] == record['statement']
            belief['evidence'].append(dict(body, claim_id=claim_id,
                binding_status='exact_current_record' if bound else 'unbound_or_mismatched',
                bound_record_id=record['id'] if bound else None,
                status='UNVALIDATED_EVIDENCE_REPORT', certified=False))
        downstream = {key: set() for key in nodes}
        if ir is not None:
            for edge in ir['hyperedges']:
                for premise in edge['premises']:
                    downstream[premise].add(edge['conclusion'])
        investigations = []
        for belief in beliefs.values():
            polarities = {report['polarity'] for report in belief['evidence']}
            belief['state'] = 'both' if len(polarities) == 2 else next(iter(polarities), 'neither')
            if belief['state'] != 'both':
                continue
            bound = sorted({report['bound_record_id'] for report in belief['evidence']
                            if report['bound_record_id'] is not None})
            affected = set(bound)
            pending = list(bound)
            while pending:
                for target in downstream[pending.pop()]:
                    if target not in affected:
                        affected.add(target)
                        pending.append(target)
            investigations.append({'belief_id': belief['belief_id'],
                'reason': 'Opposing reports for the identical proposition, scope and input version.',
                'action': 'investigate_evidence', 'record_ids': bound,
                'affected_records': sorted(affected),
                'effect': 'recommendation_only_no_record_invalidation', 'certified': False})
        return {'schema_version': SCHEMA, 'status': 'REPORTED_EVIDENCE_NOT_THEOREM_TRUTH',
                'ir_sha256': ir['ir_sha256'] if ir is not None else None,
                'beliefs': [beliefs[key] for key in sorted(beliefs)],
                'investigations': sorted(investigations, key=lambda i: i['belief_id']),
                'full_theorem_certificates': 0}
