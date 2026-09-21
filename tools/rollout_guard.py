#!/usr/bin/env python3
"""Read-only, author-side rollout-plan controls. Not a policy or truth engine.

Explicit declarations and supplied observation references are checked, not
cryptographically authenticated. No network, writes, execution of referenced
artifacts, deployment, scientific decision, or live admission is performed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class Invalid(ValueError):
    """A declared plan does not satisfy this pilot's structural controls."""


def require(ok: bool, why: str) -> None:
    if not ok:
        raise Invalid(why)


def fields(obj: Any, names: set[str], where: str) -> None:
    require(isinstance(obj, dict) and set(obj) == names, f'{where}: missing/unknown fields')


def nonempty(value: Any, where: str) -> None:
    require(isinstance(value, str) and bool(value.strip()), f'{where}: empty/non-string')


def sha(value: Any, where: str) -> None:
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None,
            f'{where}: expected full SHA-256')


def ids(value: Any, where: str) -> None:
    require(isinstance(value, list) and bool(value), f'{where}: nonempty list required')
    for item in value:
        nonempty(item, where)
    require(len(value) == len(set(value)), f'{where}: duplicate IDs')


def timestamp(value: Any) -> datetime:
    nonempty(value, 'UTC timestamp')
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(parsed.tzinfo is not None, 'timezone required')
    return parsed.astimezone(timezone.utc)


# Version-scoped documentary vocabulary, NOT a converter of old scientific data.
BINDINGS = {
    'OP-PROT-013@2026-07-26': {'A': 'RAW_CONTENT', 'B': 'EMBEDDED_PAYLOAD', 'C': 'NATIVE_EXPORT'},
    'OP-PROT-014-v1.0': {'A': 'METADATA_ONLY', 'B': 'MARKER_BODY', 'C': 'NATIVE_EXPORT',
                        'D': 'RAW_CONTENT', 'E': 'EMBEDDED_PAYLOAD'},
}


def binding_kind(standard: str, label: str) -> str:
    require(standard in BINDINGS, 'unknown/unqualified binding standard')
    require(label in BINDINGS[standard], 'unknown binding class for this standard')
    return BINDINGS[standard][label]


def entry_coverage(capacity: int, read_end: int, key_rows: dict[str, int],
                   complete: bool) -> list[str]:
    """Check a supplied bounded navigation observation; never infer unseen rows."""
    require(type(capacity) is int and capacity > 0, 'invalid allocated row count')
    require(type(read_end) is int and 0 < read_end <= capacity, 'invalid read range')
    require(complete is True, 'partial/truncated discovery cannot establish coverage')
    require(isinstance(key_rows, dict) and bool(key_rows), 'empty required key inventory')
    for key, row in key_rows.items():
        nonempty(key, 'navigation key')
        require(type(row) is int and 1 <= row <= capacity, 'key outside allocation')
    return sorted(key for key, row in key_rows.items() if row > read_end)


def execution_snapshot(expected: dict, observed: dict, claim_time: str,
                       expires: str, now: str) -> None:
    """Validate a preflight observation. This is NOT compare-and-swap."""
    names = {'base_head', 'candidate_head', 'policy_sha256'}
    fields(expected, names, 'expected snapshot')
    fields(observed, names, 'observed snapshot')
    for obj in (expected, observed):
        sha(obj['policy_sha256'], 'policy')
        for key in ('base_head', 'candidate_head'):
            require(isinstance(obj[key], str) and re.fullmatch('[0-9a-f]{40}', obj[key]) is not None,
                    'invalid Git commit identity')
    require(expected == observed, 'stale source/base/candidate: reconcile this target')
    start, stop, clock = map(timestamp, (claim_time, expires, now))
    require(start <= clock < stop, 'future or expired claim; new claim required')


def effective_rule(current: str, authority_verified: bool, selected: str,
                   superseded: list[str]) -> None:
    """An uptake audit cannot undo an already-authorized clause replacement."""
    nonempty(current, 'current rule')
    nonempty(selected, 'selected rule')
    require(authority_verified is True, 'authority must be resolved outside this checker')
    require(isinstance(superseded, list), 'superseded clauses must be explicit')
    require(selected == current and selected not in superseded,
            'retired control cannot be revived through compatibility or stricter-wins')


STAGES = ['PROPOSED', 'SHADOW', 'PILOT', 'REVIEWED', 'ACTIVE_SCOPED', 'ACTIVE_GENERAL']


def assess(plan: Any) -> dict:
    fields(plan, {'schema_version', 'change_id', 'change_kind', 'from_stage', 'to_stage',
                  'policy_sha256', 'authority_ref', 'target_consumers', 'observations',
                  'coverage_complete', 'rollback_sha256', 'review_sha256', 'effects',
                  'scientific_status_unchanged', 'frozen_bytes_preserved',
                  'reactivate_superseded', 'blocked_consumers'}, 'plan')
    require(type(plan['schema_version']) is int and plan['schema_version'] == 1,
            'unsupported schema version')
    nonempty(plan['change_id'], 'change_id')
    sha(plan['policy_sha256'], 'policy')
    sha(plan['rollback_sha256'], 'rollback')
    nonempty(plan['authority_ref'], 'authority_ref')
    ids(plan['target_consumers'], 'target_consumers')
    cohort = set(plan['target_consumers'])
    require(plan['coverage_complete'] is True, 'declared cohort coverage incomplete')
    require(plan['scientific_status_unchanged'] is True, 'scientific decision is outside this pilot')
    require(plan['frozen_bytes_preserved'] is True, 'frozen evidence cannot be modified')
    require(plan['reactivate_superseded'] is False, 'compatibility cannot resurrect retired controls')
    require(isinstance(plan['blocked_consumers'], list), 'blocked consumers must be explicit')
    require(len(set(plan['blocked_consumers'])) == len(plan['blocked_consumers']), 'duplicate blocked IDs')
    require(set(plan['blocked_consumers']) <= cohort, 'block escapes the named cohort')
    ids(plan['effects'], 'effects')
    require(set(plan['effects']) <= {'navigation', 'documentation', 'scheduling', 'admission', 'publication'},
            'unknown effect or scientific/permission change')
    require(isinstance(plan['observations'], list), 'observations must be a list')
    passed: dict[str, set[str]] = {x: set() for x in cohort}
    seen = set()
    for observation in plan['observations']:
        fields(observation, {'consumer', 'mode', 'policy_sha256', 'artifact_sha256', 'result'}, 'observation')
        who, mode = observation['consumer'], observation['mode']
        require(who in cohort, 'observation outside target cohort')
        require(mode in {'SHADOW', 'PILOT', 'READBACK'}, 'unknown observation mode')
        require((who, mode) not in seen, 'duplicate observation')
        seen.add((who, mode))
        require(observation['policy_sha256'] == plan['policy_sha256'], 'observation has stale policy identity')
        sha(observation['artifact_sha256'], 'observation artifact')
        require(observation['result'] == 'PASS', 'failed/unknown consumer observation')
        passed[who].add(mode)
    old, new, kind = plan['from_stage'], plan['to_stage'], plan['change_kind']
    if kind == 'METADATA_REPAIR':
        require((old, new) == ('BASELINE', 'APPLIED_SCOPED'), 'metadata repair has a separate bounded path')
        require(set(plan['effects']) <= {'navigation', 'documentation'}, 'metadata repair masks changed behavior')
        require(not plan['blocked_consumers'], 'metadata repair may not introduce a block')
        required = {'READBACK'}
    elif kind == 'BEHAVIOR_CHANGE':
        require(old in STAGES and new in STAGES, 'unknown material rollout stage')
        require(STAGES.index(new) == STAGES.index(old) + 1, 'material change skipped a stage')
        require(not plan['blocked_consumers'], 'behavior rollout cannot impose containment blocks')
        target = STAGES.index(new)
        required = {'SHADOW'} if target >= 1 else set()
        if target >= 2:
            required.add('PILOT')
        if target >= 3:
            sha(plan['review_sha256'], 'material rollout review')
        if target >= 4:
            required.add('READBACK')
    else:
        # Containment is not smuggled into this author-side rollout validator.
        raise Invalid('unsupported kind; containment requires its own exact defect/authority record')
    require(all(required <= modes for modes in passed.values()), 'missing required per-consumer observations')
    return {'result': 'PLAN_CONSISTENT', 'authority': 'NONE', 'live_admission': False,
            'scientific_acceptance': False, 'consumer_count': len(cohort),
            'declared_stage': new, 'limits': [
                'Reference authenticity, authority, and scope completeness require external review.',
                'A preflight read is not an atomic commit guard or a deployment.',
                'This validates the supplied cohort only; it makes no global adoption claim.']}


def load(path: Path) -> Any:
    def pairs(items):
        out = {}
        for k, v in items:
            require(k not in out, 'duplicate JSON key')
            out[k] = v
        return out
    def constant(value):
        raise Invalid('nonfinite JSON is not supported')
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=pairs, parse_constant=constant)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(assess(load(args.plan)), indent=2, sort_keys=True))
        return 0
    except (Invalid, ValueError, TypeError, KeyError, OSError, RecursionError) as error:
        print(f'ROLLOUT PLAN HOLD: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
