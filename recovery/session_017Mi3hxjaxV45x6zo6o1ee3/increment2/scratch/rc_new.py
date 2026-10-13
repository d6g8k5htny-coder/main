#!/usr/bin/env python3
"""Validate retrofit evidence records against reviews/retrofit_20261006/CONTRACT.md v0.1 (standard library only).

A RECORDS.json file states, for exact source objects, which evidence axes have a record, using a closed vocabulary
that cannot carry a scientific status. This checker parses strictly (duplicate keys and non-finite numbers are
rejected), enforces the closed keys and enums of the contract, and with --repo verifies every subject and file
evidence blob against git. --aggregate summarizes several valid files without writing anything.

A record's state says only whether evidence is linked. A workflow_run item keeps its own attempt, tested commit,
purpose, native conclusion and expected conclusion, so a failed run stays recorded as failed and a negative control
that fails as expected is not mistaken for a failed check. Nothing here selects a run as a success.

Scientific effect: NONE. A valid record is not a review, an acceptance or a register transition.

Usage:
  python3 -B -S tools/retrofit_record_check.py [--repo PATH] RECORDS.json [...]
  python3 -B -S tools/retrofit_record_check.py --aggregate [--repo PATH] RECORDS.json [...]
Exit status 0 iff every file is valid.
"""
import argparse
import json
import pathlib
import re
import subprocess
import sys

SCHEMA = 'retrofit-records/v0.1'
BASELINE = {
    'Math-': '08f86862f859ac4804b4fd6c6477a7ed0421e37f',
    'main': 'cddbb7f6cf3f57f3b495277f7da148287e027b19',
}
AXES = ('source_review', 'provider_distinct_review', 'blind_reconstruction', 'adversarial_attack', 'formal_evidence',
        'numerical_reproduction', 'novelty', 'human_reading', 'custody', 'observable_statement')
STATES = ('recorded', 'not_recorded', 'unknown', 'not_applicable')
EVIDENCE_KINDS = ('github_comment', 'github_review', 'workflow_run', 'file')
EVIDENCE_KEYS = {
    'file': {'kind', 'repository', 'commit', 'path', 'blob'},
    'github_comment': {'kind', 'repository', 'ref'},
    'github_review': {'kind', 'repository', 'ref', 'commit'},
    'workflow_run': {'kind', 'repository', 'ref', 'attempt', 'job', 'tested_commit', 'purpose', 'conclusion',
                     'expected_conclusion'},
}
RUN_PURPOSES = ('check', 'negative_control')
RUN_CONCLUSIONS = ('success', 'failure', 'cancelled', 'skipped', 'timed_out', 'neutral', 'action_required', 'stale',
                   'startup_failure')
EXPECTED_CONCLUSIONS = ('success', 'failure')
REPOSITORIES = ('Math-', 'main', 'Universal-Law-Workspace', 'd6g8k5htny-coder', 'google-drive', 'governance-',
                'meta-framework', 'query-', 'sandbox', 'trial')
TOP_KEYS = {'schema', 'shard', 'baseline', 'author', 'scientific_effect', 'status_authority', 'records'}
RECORD_KEYS = {'id', 'subject', 'delta', 'axis', 'state', 'evidence', 'performer', 'exposure', 'independence_credit',
               'alias_of', 'notes'}
SUBJECT_KEYS = {'repository', 'commit', 'path', 'blob'}
PERSON_KEYS = {'provider', 'model_or_agent', 'session'}
HEX40 = re.compile(r'[0-9a-f]{40}')
SHARD = re.compile(r'[A-Z][A-Za-z0-9_]{0,39}')
RECORD_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}')
DIGITS = re.compile(r'[1-9][0-9]{0,19}')


def load_strict(text):
    """json.loads that rejects duplicate keys and NaN/Infinity (the R-2 rule)."""
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError('duplicate JSON key: ' + key)
            out[key] = value
        return out

    def constant(name):
        raise ValueError('non-finite JSON constant: ' + name)

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def _is_text(value):
    return isinstance(value, str) and value.strip() != ''


def _safe_path(value):
    return (_is_text(value) and not value.startswith('/') and '\\' not in value
            and all(part not in ('', '.', '..') for part in value.split('/')))


def _git(repo, *args):
    run = subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True)
    return run.stdout.strip() if run.returncode == 0 else None


def _git_blob(repo, commit, path):
    """The blob id at commit:path; None unless commit names a commit object and path a blob in its tree."""
    if _git(repo, 'cat-file', '-t', commit) != 'commit':
        return None
    if _git(repo, 'cat-file', '-t', commit + ':' + path) != 'blob':
        return None
    return _git(repo, 'rev-parse', '--verify', '--quiet', commit + ':' + path)


def _person(value, where, errors):
    if not isinstance(value, dict) or set(value) != PERSON_KEYS:
        errors.append(where + ': requires exactly provider, model_or_agent, session')
        return
    for key in sorted(PERSON_KEYS):
        if not _is_text(value[key]):
            errors.append(where + '.' + key + ': non-empty string required (use UNKNOWN when unavailable)')


def validate(data, *, shard_dir=None, repos=None):
    """Return the list of contract violations in one parsed RECORDS.json (empty when valid)."""
    errors = []
    if not isinstance(data, dict) or set(data) != TOP_KEYS:
        return ['top level requires exactly: ' + ', '.join(sorted(TOP_KEYS))]
    if data['schema'] != SCHEMA:
        errors.append('schema must be ' + SCHEMA)
    if not isinstance(data['shard'], str) or not SHARD.fullmatch(data['shard']):
        errors.append('shard must match ' + SHARD.pattern)
    elif shard_dir is not None and shard_dir != data['shard']:
        errors.append('shard %r does not match its directory %r' % (data['shard'], shard_dir))
    if data['baseline'] != BASELINE:
        errors.append('baseline must be exactly the pinned v0.1 cut %r' % BASELINE)
    _person(data['author'], 'author', errors)
    if data['scientific_effect'] != 'NONE':
        errors.append('scientific_effect must be NONE')
    if data['status_authority'] is not False:
        errors.append('status_authority must be the Boolean false')
    records = data['records']
    if not isinstance(records, list) or not records:
        return errors + ['records must be a non-empty list']
    by_id = {}
    for record in records:
        if isinstance(record, dict) and isinstance(record.get('id'), str) and record['id'] not in by_id:
            by_id[record['id']] = record
    seen = set()
    for index, record in enumerate(records):
        where = 'records[%d]' % index
        if not isinstance(record, dict) or set(record) != RECORD_KEYS:
            errors.append(where + ': requires exactly ' + ', '.join(sorted(RECORD_KEYS)))
            continue
        rid = record['id']
        if not isinstance(rid, str) or not RECORD_ID.fullmatch(rid):
            errors.append(where + '.id: must match ' + RECORD_ID.pattern)
        elif rid in seen:
            errors.append(where + '.id: duplicate ' + rid)
        else:
            seen.add(rid)
        subject = record['subject']
        if not isinstance(subject, dict) or set(subject) != SUBJECT_KEYS:
            errors.append(where + '.subject: requires exactly repository, commit, path, blob')
        else:
            repo_name = subject['repository']
            if not isinstance(repo_name, str) or repo_name not in REPOSITORIES:
                errors.append(where + '.subject.repository: unknown repository')
                repo_name = None
            if not isinstance(subject['commit'], str) or not HEX40.fullmatch(subject['commit']):
                errors.append(where + '.subject.commit: 40 lowercase hex digits required')
            if not _safe_path(subject['path']):
                errors.append(where + '.subject.path: relative repository path required')
            if not isinstance(subject['blob'], str) or not HEX40.fullmatch(subject['blob']):
                errors.append(where + '.subject.blob: 40 lowercase hex digits required')
            if type(record['delta']) is not bool:
                errors.append(where + '.delta: Boolean required')
            elif not record['delta'] and (repo_name is None or BASELINE.get(repo_name) != subject['commit']):
                errors.append(where + '.subject.commit: not the pinned baseline; mark a candidate with delta true')
            if repos and repo_name is not None and repo_name in repos and not errors_for(where, errors):
                got = _git_blob(repos[repo_name], subject['commit'], subject['path'])
                if got != subject['blob']:
                    errors.append(where + '.subject: blob does not match git (%s)' % (got or 'no blob at that path of a commit'))
        if record['axis'] not in AXES:
            errors.append(where + '.axis: one of ' + ', '.join(AXES))
        if record['state'] not in STATES:
            errors.append(where + '.state: one of ' + ', '.join(STATES) + ' (no status words)')
        evidence = record['evidence']
        if not isinstance(evidence, list):
            errors.append(where + '.evidence: list required')
            evidence = []
        if record['state'] == 'recorded' and not evidence:
            errors.append(where + ': state recorded requires evidence')
        if record['state'] in ('not_recorded', 'not_applicable') and evidence:
            errors.append(where + ': state %s takes no evidence' % record['state'])
        for k, item in enumerate(evidence):
            _evidence(item, '%s.evidence[%d]' % (where, k), errors, repos)
        _person(record['performer'], where + '.performer', errors)
        if not _is_text(record['exposure']):
            errors.append(where + '.exposure: non-empty string required (state what was read)')
        if type(record['independence_credit']) is not int or record['independence_credit'] != 0:
            errors.append(where + '.independence_credit: must be the integer 0')
        _alias(record, by_id, where, errors)
        if not isinstance(record['notes'], str):
            errors.append(where + '.notes: string required (may be empty)')
    return errors


def _canonical(evidence):
    return sorted(json.dumps(item, sort_keys=True) for item in evidence) if isinstance(evidence, list) else None


def _alias(record, by_id, where, errors):
    """alias_of names a canonical record (alias_of null) in this file with the same axis, state and evidence."""
    alias = record['alias_of']
    if alias is None:
        return
    target = by_id.get(alias) if isinstance(alias, str) else None
    if target is None or target is record:
        errors.append(where + '.alias_of: null or the id of another record in this file')
    elif target.get('alias_of') is not None:
        errors.append(where + '.alias_of: must name a canonical record (one whose alias_of is null); no chains')
    elif (target.get('axis'), target.get('state')) != (record['axis'], record['state']):
        errors.append(where + '.alias_of: an alias has the axis and state of its canonical record')
    elif _canonical(target.get('evidence')) != _canonical(record['evidence']):
        errors.append(where + '.alias_of: an alias restates exactly the evidence of its canonical record')


def errors_for(where, errors):
    return any(e.startswith(where + '.subject') or e.startswith(where + '.delta') for e in errors)


def _evidence(item, where, errors, repos):
    if not isinstance(item, dict) or 'kind' not in item:
        errors.append(where + ': object with kind required')
        return
    kind = item['kind']
    if kind not in EVIDENCE_KINDS:
        errors.append(where + '.kind: one of ' + ', '.join(EVIDENCE_KINDS))
        return
    keys = EVIDENCE_KEYS[kind]
    if set(item) != keys:
        errors.append(where + ': %s evidence requires exactly %s' % (kind, ', '.join(sorted(keys))))
        return
    known = isinstance(item['repository'], str) and item['repository'] in REPOSITORIES
    if not known:
        errors.append(where + '.repository: unknown repository')
    if kind == 'file':
        ok = (isinstance(item['commit'], str) and HEX40.fullmatch(item['commit'])
              and isinstance(item['blob'], str) and HEX40.fullmatch(item['blob']) and _safe_path(item['path']))
        if not ok:
            errors.append(where + ': file evidence needs 40-hex commit and blob and a relative path')
        elif known and repos and item['repository'] in repos:
            if _git_blob(repos[item['repository']], item['commit'], item['path']) != item['blob']:
                errors.append(where + ': file evidence blob does not match git')
        return
    if not isinstance(item['ref'], str) or not DIGITS.fullmatch(item['ref']):
        errors.append(where + '.ref: numeric GitHub id as a string required')
    if kind == 'github_review' and not (isinstance(item['commit'], str) and HEX40.fullmatch(item['commit'])):
        errors.append(where + '.commit: the review\'s native commit_id (40 hex) required')
    if kind == 'workflow_run':
        if type(item['attempt']) is not int or item['attempt'] < 1:
            errors.append(where + '.attempt: positive integer required')
        if item['job'] is not None and (not isinstance(item['job'], str) or not DIGITS.fullmatch(item['job'])):
            errors.append(where + '.job: null or a numeric job id as a string')
        if not isinstance(item['tested_commit'], str) or not HEX40.fullmatch(item['tested_commit']):
            errors.append(where + '.tested_commit: the run\'s native head_sha (40 hex) required')
        if item['purpose'] not in RUN_PURPOSES:
            errors.append(where + '.purpose: one of ' + ', '.join(RUN_PURPOSES))
        if item['conclusion'] not in RUN_CONCLUSIONS:
            errors.append(where + '.conclusion: the native GitHub conclusion, one of ' + ', '.join(RUN_CONCLUSIONS))
        if item['expected_conclusion'] not in EXPECTED_CONCLUSIONS:
            errors.append(where + '.expected_conclusion: one of ' + ', '.join(EXPECTED_CONCLUSIONS))


def aggregate(datasets):
    """Per subject object and axis, the states recorded across files; aliases counted separately."""
    out = {}
    counts = {'files': len(datasets), 'records': 0, 'aliases': 0, 'evidence_items_non_alias': 0}
    distinct = set()
    for data in datasets:
        for record in data['records']:
            counts['records'] += 1
            if record['alias_of'] is not None:
                counts['aliases'] += 1
            else:
                counts['evidence_items_non_alias'] += len(record['evidence'])
                distinct.update(_canonical(record['evidence']))
            s = record['subject']
            key = '%s@%s:%s#%s' % (s['repository'], s['commit'], s['path'], s['blob'])
            out.setdefault(key, {}).setdefault(record['axis'], []).append(
                {'shard': data['shard'], 'id': record['id'], 'state': record['state'],
                 'alias_of': record['alias_of']})
    counts['distinct_evidence_items'] = len(distinct)
    return {'subjects': out, 'counts': counts, 'status_authority': False, 'scientific_effect': 'NONE',
            'meaning': 'derived summary of retrofit records; not a status record or acceptance'}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('files', nargs='+')
    ap.add_argument('--repo', help='local clone used to verify Math- blobs (main is verified only with --main-repo)')
    ap.add_argument('--main-repo', help='local clone of main for main-repository blobs')
    ap.add_argument('--aggregate', action='store_true')
    args = ap.parse_args(argv)
    repos = {}
    if args.repo:
        repos['Math-'] = pathlib.Path(args.repo)
    if args.main_repo:
        repos['main'] = pathlib.Path(args.main_repo)
    report, datasets, ok = {'files': {}, 'input_errors': []}, [], True
    resolved, shards = {}, {}
    for name in args.files:
        path = pathlib.Path(name)
        real = path.resolve()
        if real in resolved:
            report['input_errors'].append('%s: same file as %s; each input is read once' % (name, resolved[real]))
            ok = False
            continue
        resolved[real] = name
        try:
            data = load_strict(path.read_text(encoding='utf-8'))
            shard_dir = path.parent.name if path.name == 'RECORDS.json' else None
            errors = validate(data, shard_dir=shard_dir, repos=repos)
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            data, errors = None, [str(exc)]
        if not errors and data['shard'] in shards:
            errors = ['shard %s already supplied by %s; one RECORDS.json per shard' % (data['shard'],
                                                                                    shards[data['shard']])]
        report['files'][name] = {'valid': not errors, 'errors': errors}
        ok &= not errors
        if not errors:
            shards[data['shard']] = name
            datasets.append(data)
    report['passed'] = ok
    report['scientific_effect'] = 'NONE'
    if args.aggregate and ok:
        report['aggregate'] = aggregate(datasets)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
