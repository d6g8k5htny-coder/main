#!/usr/bin/env python3
"""Offline, advisory board-capacity check. No network, writes or authorization.

Input: {"observed_at": "RFC3339", "issue": <native issue or selected fields>}.
The caller supplies the active URL from current governing entry instructions.
Observation time is a caller assertion, not authenticated acquisition evidence.
Exit 0: usable snapshot below advisory thresholds. Exit 1: attention needed.
Exit 2: input/identity/freshness unknown; never an operational all-clear.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

MAX_INPUT_BYTES = 1048576
PART = r'([A-Za-z0-9_.-]+)'
WEB = re.compile(r'https://github\.com/' + PART + '/' + PART + r'/issues/([1-9][0-9]*)')
API = re.compile(r'https://api\.github\.com/repos/' + PART + '/' + PART + r'/issues/([1-9][0-9]*)')
REPO = re.compile(r'https://api\.github\.com/repos/' + PART + '/' + PART)
TIME = re.compile(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-](?:[01]\d|2[0-3]):[0-5]\d)', re.ASCII)


def _identity(value, pattern):
    if type(value) is not str:
        return None
    match = pattern.fullmatch(value)
    if match is None or any(part in ('.', '..') for part in match.groups()[:2]):
        return None
    owner, repo, *number = match.groups()
    return (owner.casefold(), repo.casefold(), *number)


def _time(value):
    if type(value) is not str or TIME.fullmatch(value) is None:
        raise ValueError('INVALID_TIME')
    return datetime.fromisoformat(value.replace('Z', '+00:00')).astimezone(timezone.utc)


def _empty(reason):
    return {'status': 'UNKNOWN', 'capacity': 'UNKNOWN', 'reasons': [reason],
            'attention_required': True, 'advisory_only': True,
            'posting_permission': 'NOT_ESTABLISHED', 'scientific_effect': 'NONE'}


def assess(snapshot, active_url, now, *, prepare_at=2000, cutover_at=2250,
           max_age_seconds=900):
    """Validate a point observation; never elect a board or authorize a write."""
    if (any(type(v) is not int or v <= 0 for v in
            (prepare_at, cutover_at, max_age_seconds)) or prepare_at >= cutover_at):
        return _empty('INVALID_POLICY')
    active = _identity(active_url, WEB)
    if active is None:
        return _empty('INVALID_ACTIVE_URL')
    if (type(snapshot) is not dict or 'observed_at' not in snapshot or
            type(snapshot.get('issue')) is not dict):
        return _empty('INVALID_SNAPSHOT')
    issue = snapshot['issue']
    if 'pull_request' in issue:
        return _empty('NOT_ISSUE')
    if (any(type(issue.get(k)) is not int or issue[k] <= 0 for k in ('id', 'number')) or
            type(issue.get('comments')) is not int or issue['comments'] < 0 or
            type(issue.get('locked')) is not bool or
            issue.get('state') not in ('open', 'closed') or
            any(type(issue.get(k)) is not str for k in ('url', 'html_url', 'repository_url'))):
        return _empty('INVALID_ISSUE')
    if (str(issue['number']) != active[2] or
            _identity(issue['html_url'], WEB) != active or
            _identity(issue['url'], API) != active or
            _identity(issue['repository_url'], REPO) != active[:2]):
        return _empty('ROUTE_MISMATCH')
    try:
        observed = _time(snapshot['observed_at'])
        current = _time(now)
    except (ValueError, OverflowError):
        return _empty('INVALID_TIME')
    age = (current - observed).total_seconds()
    if age < 0:
        return _empty('FUTURE_SNAPSHOT')
    if age > max_age_seconds:
        return _empty('STALE_SNAPSHOT')
    count = issue['comments']
    level = 'CUTOVER' if count >= cutover_at else ('PREPARE' if count >= prepare_at else 'NORMAL')
    reasons = []
    if issue['state'] != 'open':
        reasons.append('CLOSED')
    if issue['locked']:
        reasons.append('LOCKED')
    return {'status': 'UNAVAILABLE' if reasons else 'OK', 'capacity': level,
            'reasons': reasons, 'attention_required': bool(reasons) or level != 'NORMAL',
            'advisory_only': True, 'posting_permission': 'NOT_ESTABLISHED',
            'scientific_effect': 'NONE',
            'active_url': 'https://github.com/' + '/'.join(active[:2]) + '/issues/' + active[2],
            'comments': count, 'observed_at': observed.isoformat(), 'age_seconds': age,
            'thresholds': {'prepare_at': prepare_at, 'cutover_at': cutover_at},
            'max_age_seconds': max_age_seconds}


def exit_code(result):
    if result['status'] == 'UNKNOWN':
        return 2
    return 1 if result['attention_required'] else 0


def _pairs(rows):
    value = {}
    for key, item in rows:
        if key in value:
            raise ValueError('DUPLICATE_KEY')
        value[key] = item
    return value


def _constant(_):
    raise ValueError('NONFINITE_JSON')


def decode_snapshot(raw):
    if len(raw) > MAX_INPUT_BYTES:
        raise ValueError('INPUT_TOO_LARGE')
    return json.loads(raw.decode('utf-8'), object_pairs_hook=_pairs, parse_constant=_constant)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('--active-url', required=True)
    parser.add_argument('--now', default=None)
    parser.add_argument('--prepare-at', type=int, default=2000)
    parser.add_argument('--cutover-at', type=int, default=2250)
    parser.add_argument('--max-age-seconds', type=int, default=900)
    args = parser.parse_args()
    raw = None
    try:
        with args.snapshot.open('rb') as stream:
            raw = stream.read(MAX_INPUT_BYTES + 1)
    except OSError:
        result = _empty('INPUT_UNREADABLE')
    else:
        try:
            snapshot = decode_snapshot(raw)
        except (ValueError, RecursionError):
            result = _empty('INPUT_INVALID')
        else:
            result = assess(snapshot, args.active_url,
                            args.now if args.now is not None else datetime.now(timezone.utc).isoformat(),
                            prepare_at=args.prepare_at, cutover_at=args.cutover_at,
                            max_age_seconds=args.max_age_seconds)
        # Only a complete within-bound input has a claimed whole-input digest.
        if len(raw) <= MAX_INPUT_BYTES:
            result['input_sha256'] = hashlib.sha256(raw).hexdigest()
    print(json.dumps(result, sort_keys=True, separators=(',', ':')))
    return exit_code(result)


if __name__ == '__main__':
    raise SystemExit(main())
