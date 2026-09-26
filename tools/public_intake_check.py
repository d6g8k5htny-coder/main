"""Read-only PR intake boundary. Run only this trusted default-branch code in CI.

No submitted file is checked out, imported, rendered, or executed. The limited
credential-pattern scan is not a guarantee that a submission contains no secret.
A pass means packaging checks passed; scientific effect remains NONE.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import struct
import urllib.request
import zlib

MAX_FILES = 500  # Below the API's 3000-file ceiling; pagination is mandatory.
MAX_PACKAGE_FILES = 50
MAX_FILE_BYTES = 262144
MAX_PACKAGE_BYTES = 2097152
MAX_RESPONSE_BYTES = 8388608
SOURCE_REPOS = {'main', 'Math-', 'query-', 'Universal-Law-Workspace'}
TEXT_SUFFIXES = {'.md', '.txt', '.json', '.csv'}
PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'
MAX_PNG_SIDE = 16384
MAX_PNG_PIXELS = 1 << 24
HEX40 = re.compile(r'[0-9a-f]{40}\Z')
HEX64 = re.compile(r'[0-9a-f]{64}\Z')
REPOSITORY = re.compile(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z')
SAFE_PATH = re.compile(r'[A-Za-z0-9_][A-Za-z0-9_. /-]*\Z')
BRANCH = re.compile(r'[A-Za-z0-9_][A-Za-z0-9_./-]*\Z')  # a name this route can carry; '..' refused below
MAX_BRANCH_PAGES = 3
CREDENTIALS = [
    re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
    re.compile(rb'\bgh[pousr]_[A-Za-z0-9]{36,}\b'),
    re.compile(rb'\bgithub_pat_[A-Za-z0-9_]{60,}\b'),
    re.compile(rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    re.compile(rb'\bxox[baprs]-[A-Za-z0-9-]{10,}\b'),
    re.compile(rb'\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}\b'),
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def strict_json(raw):
    def bad_constant(_):
        raise ValueError('nonfinite JSON value')
    return json.loads(raw, object_pairs_hook=unique, parse_constant=bad_constant)


def safe_path(path):
    require(isinstance(path, str) and SAFE_PATH.fullmatch(path), 'unsafe path')
    parts = path.split('/')
    require(all(p not in ('', '.', '..') and not p.startswith('.') for p in parts), 'unsafe path')
    require(str(PurePosixPath(path)) == path, 'noncanonical path')
    return parts


def integer(value, low, high, message):
    require(type(value) is int and low <= value <= high, message)
    return value


def sha40(value, message):
    require(isinstance(value, str) and HEX40.fullmatch(value) and value != '0'*40, message)
    return value


def sha256(value, message):
    require(isinstance(value, str) and HEX64.fullmatch(value), message)
    return value


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('API redirects refused')


class API:
    def __init__(self, token):
        require(bool(token), 'read-only GitHub token missing')
        self.token = token
        self.opener = urllib.request.build_opener(NoRedirect())

    def get(self, route):
        require(route.startswith('/repos/') and '\n' not in route and '\r' not in route, 'invalid API route')
        request = urllib.request.Request('https://api.github.com' + route, headers={
            'Authorization': 'Bearer ' + self.token, 'Accept': 'application/vnd.github+json',
            'X-GitHub-Api-Version': '2022-11-28', 'User-Agent': 'public-intake-read-only'})
        with self.opener.open(request, timeout=30) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        require(len(raw) <= MAX_RESPONSE_BYTES, 'API response exceeds safe limit')
        return strict_json(raw)


def unpack_blob(data, max_size, expected_sha=None):
    require(isinstance(data, dict) and data.get('encoding') == 'base64', 'blob encoding unavailable')
    size = integer(data.get('size'), 0, max_size, 'blob exceeds size limit')
    require(isinstance(data.get('content'), str), 'blob content missing')
    raw = base64.b64decode(''.join(data['content'].split()), validate=True)
    require(len(raw) == size, 'blob size mismatch')
    identity = hashlib.sha1(b'blob ' + str(size).encode('ascii') + b'\0' + raw).hexdigest()
    require(data.get('sha') == identity and (expected_sha is None or identity == expected_sha), 'Git blob identity mismatch')
    return raw


def png_structure(raw):
    """Walk the chunk list without decoding pixels: IHDR first, ASCII chunk types,
    a CRC per chunk, IDAT present, an empty IEND last with nothing after it, and
    bounded dimensions. Bytes inside a chunk's data field are not inspected."""
    require(raw.startswith(PNG_SIGNATURE), 'invalid PNG signature')
    offset, kinds = 8, []
    while offset < len(raw) and kinds[-1:] != [b'IEND']:
        require(len(raw) - offset >= 12, 'invalid PNG chunk structure or trailing data')
        length, kind = struct.unpack('>I4s', raw[offset:offset + 8])
        require(kind.isalpha() and length <= len(raw) - offset - 12, 'invalid PNG chunk structure or trailing data')
        data = raw[offset + 8:offset + 8 + length]
        (crc,) = struct.unpack('>I', raw[offset + 8 + length:offset + 12 + length])
        require(zlib.crc32(kind + data) == crc, 'invalid PNG chunk CRC')
        if not kinds:
            require(kind == b'IHDR' and length == 13, 'invalid PNG chunk structure or trailing data')
            width, height = struct.unpack('>II', data[:8])
            require(1 <= width <= MAX_PNG_SIDE and 1 <= height <= MAX_PNG_SIDE and width * height <= MAX_PNG_PIXELS,
                    'PNG dimensions exceed limit')
        require(kind != b'IEND' or length == 0, 'invalid PNG chunk structure or trailing data')
        kinds.append(kind)
        offset += 12 + length
    require(offset == len(raw) and kinds[-1:] == [b'IEND'] and b'IDAT' in kinds, 'invalid PNG chunk structure or trailing data')


def tree(api, repo, ref):
    data = api.get(f'/repos/{repo}/git/trees/{ref}?recursive=1')
    require(isinstance(data, dict) and data.get('truncated') is False and isinstance(data.get('tree'), list),
            'tree unavailable or truncated')
    rows = {}
    for row in data['tree']:
        require(isinstance(row, dict) and isinstance(row.get('path'), str) and row['path'] not in rows, 'invalid tree entries')
        rows[row['path']] = row
    return rows


def ancestor_of_branch(api, repo, branch, commit):
    if not (isinstance(branch, str) and BRANCH.fullmatch(branch) and '..' not in branch):
        return False  # a name this route cannot carry safely never vouches for a source
    data = api.get(f'/repos/{repo}/compare/{branch}...{commit}?per_page=1')
    if not (isinstance(data, dict) and isinstance(data.get('merge_base_commit'), dict)):
        return False  # any other shape is not evidence of reachability
    return (type(data.get('ahead_by')) is int and data['ahead_by'] == 0 and data.get('status') in ('behind', 'identical')
            and data['merge_base_commit'].get('sha') == commit)


def reachable(api, repo, commit, default_branch):
    # /git/commits, /git/trees and /git/blobs serve any object in the repository's
    # store, including refs/pull/N/head of an outsider's PR and fork-network objects.
    # Only a branch of the pillar, which outsiders cannot create, vouches for a source.
    if ancestor_of_branch(api, repo, default_branch, commit):
        return True
    for page in range(1, MAX_BRANCH_PAGES + 1):
        batch = api.get(f'/repos/{repo}/branches?per_page=100&page={page}')
        require(isinstance(batch, list), 'branch listing unavailable')
        for row in batch:
            name = row.get('name') if isinstance(row, dict) else None
            if name != default_branch and ancestor_of_branch(api, repo, name, commit):
                return True
        if len(batch) < 100:
            break
    return False


def snapshot(pr, repo):
    require(isinstance(pr, dict) and pr.get('state') == 'open', 'PR is not open')
    base, head = pr.get('base'), pr.get('head')
    require(isinstance(base, dict) and isinstance(base.get('repo'), dict) and base['repo'].get('full_name') == repo,
            'base repository mismatch')
    require(base.get('ref') == 'main', 'this intake lane targets main only')
    require(isinstance(head, dict) and isinstance(head.get('repo'), dict), 'head repository missing')
    head_repo = head['repo'].get('full_name')
    require(isinstance(head_repo, str) and REPOSITORY.fullmatch(head_repo), 'invalid head repository')
    labels = pr.get('labels')
    require(isinstance(labels, list) and all(isinstance(x, dict) and isinstance(x.get('name'), str) for x in labels), 'labels unavailable')
    return (sha40(head.get('sha'), 'invalid head SHA'), sha40(base.get('sha'), 'invalid base SHA'),
            head_repo, tuple(sorted(x['name'] for x in labels)), integer(pr.get('changed_files'), 1, MAX_FILES, 'unsupported changed file count'))


def changes(api, repo, number, count):
    rows = []
    for page in range(1, (count + 99)//100 + 1):
        batch = api.get(f'/repos/{repo}/pulls/{number}/files?per_page=100&page={page}')
        require(isinstance(batch, list), 'file listing unavailable')
        rows.extend(batch)
    require(len(rows) == count, 'incomplete file listing')
    seen = set()
    for row in rows:
        require(isinstance(row, dict), 'invalid changed file')
        path = row.get('filename')
        require(isinstance(path, str) and bool(path) and not any(ord(c) < 32 for c in path)
                and '\\' not in path and all(p not in ('', '.', '..') for p in path.split('/')),
                'invalid changed path')
        require(path not in seen, 'duplicate changed path')
        seen.add(path)
        if 'previous_filename' in row:
            previous = row['previous_filename']
            require(isinstance(previous, str) and bool(previous), 'invalid previous path')
    return rows


def verify_package(api, repo, head, base, rows):
    require(len(rows) <= MAX_PACKAGE_FILES, 'too many package files')
    roots = set()
    for row in rows:
        # Package IDs are reported in the result. Inspect paths as well as blob
        # bytes, and never include the rejected value in the error message.
        for candidate_path in (row['filename'], row.get('previous_filename', '')):
            require(not any(pattern.search(candidate_path.encode('utf-8')) for pattern in CREDENTIALS),
                    'possible credential in path; values are not logged')
        parts = safe_path(row['filename'])
        require(len(parts) >= 3 and parts[0] == 'incoming', 'incoming-only edits required')
        require(re.fullmatch(r'[a-z0-9][a-z0-9-]{0,63}', parts[1]), 'invalid package id')
        require(row.get('status') == 'added' and 'previous_filename' not in row, 'add-only intake; use a new package id for a successor')
        require(not any(p.lower() == 'claims' or p.upper().split('.')[0] in {'STATUS', 'PROOF_INDEX', 'LANDING_CLAIMS'} for p in parts[2:]), 'protected scientific status path')
        roots.add('/'.join(parts[:2]) + '/')
    require(len(roots) == 1, 'one package per PR')
    prefix = roots.pop()
    old_tree = tree(api, repo, base)
    require(not any(p == prefix[:-1] or p.startswith(prefix) for p in old_tree), 'use a new package id; landed intake is immutable')
    new_tree = tree(api, repo, head)
    package_entries = {p: row for p, row in new_tree.items() if p.startswith(prefix) and row.get('type') != 'tree'}
    require(set(package_entries) == {row['filename'] for row in rows}, 'package tree does not match changed files')
    payload = {}
    total = 0
    for row in rows:
        name = row['filename']
        entry = package_entries[name]
        require(entry.get('type') == 'blob' and entry.get('mode') == '100644', 'regular non-executable files only')
        identity = sha40(entry.get('sha'), 'invalid blob SHA')
        require(identity == row.get('sha'), 'changed file/tree identity mismatch')
        integer(entry.get('size'), 0, MAX_FILE_BYTES, 'file exceeds size limit')
        suffix = PurePosixPath(name).suffix.lower()
        require(suffix in TEXT_SUFFIXES | {'.png'}, 'unsupported file extension')
        raw = unpack_blob(api.get(f'/repos/{repo}/git/blobs/{identity}'), MAX_FILE_BYTES, identity)
        total += len(raw)
        require(total <= MAX_PACKAGE_BYTES, 'package exceeds total size limit')
        require(not any(pattern.search(raw) for pattern in CREDENTIALS), 'possible credential detected; values are not logged')
        if suffix in TEXT_SUFFIXES:
            text = raw.decode('utf-8')
            require('\0' not in text, 'NUL byte in text')
            if suffix == '.json':
                strict_json(text)
        else:
            png_structure(raw)
        payload[name[len(prefix):]] = raw
    require('RESULT.md' in payload and payload['RESULT.md'].strip() and 'IDENTITY.json' in payload, 'RESULT.md and IDENTITY.json required')
    identity = strict_json(payload['IDENTITY.json'])
    require(isinstance(identity, dict) and set(identity) == {'schema', 'scientific_effect', 'review_status', 'sources', 'artifacts'}, 'invalid identity schema')
    require(type(identity['schema']) is int and identity['schema'] == 1, 'invalid schema version')
    require(identity['scientific_effect'] == 'NONE' and identity['review_status'] == 'REVIEW_REQUIRED', 'intake is review-only; no status promotion')
    artifacts = identity['artifacts']
    require(isinstance(artifacts, list), 'artifact list required')
    seen = set()
    for row in artifacts:
        require(isinstance(row, dict) and set(row) == {'path', 'bytes', 'sha256'}, 'invalid artifact record')
        name = row['path']
        safe_path(name)
        require(name not in seen and name != 'IDENTITY.json' and name in payload, 'duplicate or unknown artifact')
        seen.add(name)
        size = integer(row['bytes'], 0, MAX_FILE_BYTES, 'invalid artifact size')
        digest = sha256(row['sha256'], 'invalid artifact digest')
        require(size == len(payload[name]) and digest == hashlib.sha256(payload[name]).hexdigest(), 'artifact identity mismatch')
    require(seen == set(payload) - {'IDENTITY.json'}, 'manifest must exactly cover every artifact except itself')
    sources = identity['sources']
    require(isinstance(sources, list) and 1 <= len(sources) <= 8, 'one to eight pinned sources required')
    seen_sources = set()
    for row in sources:
        require(isinstance(row, dict) and set(row) == {'repository', 'path', 'commit', 'sha256'}, 'invalid source record')
        source_repo = row['repository']
        require(isinstance(source_repo, str) and source_repo in {'d6g8k5htny-coder/' + r for r in SOURCE_REPOS}, 'source repository outside public pillars')
        path = row['path']
        safe_path(path)
        commit = sha40(row['commit'], 'source commit must be a full SHA')
        digest = sha256(row['sha256'], 'invalid source digest')
        key = (source_repo, commit, path)
        require(key not in seen_sources, 'duplicate source')
        seen_sources.add(key)
        metadata = api.get(f'/repos/{source_repo}')
        require(isinstance(metadata, dict) and metadata.get('private') is False and metadata.get('full_name') == source_repo, 'source is not verified public')
        commit_record = api.get(f'/repos/{source_repo}/git/commits/{commit}')
        require(isinstance(commit_record, dict) and commit_record.get('sha') == commit, 'source is not an exact commit')
        require(reachable(api, source_repo, commit, metadata.get('default_branch')),
                'source commit is not reachable from any branch of the public repository')
        # The contents endpoint follows in-repository symlinks. Resolve the
        # exact path from the commit tree instead, then read that exact blob.
        source_tree = tree(api, source_repo, commit)
        source_entry = source_tree.get(path)
        require(isinstance(source_entry, dict) and source_entry.get('type') == 'blob'
                and source_entry.get('mode') == '100644', 'source must be a regular non-executable public file')
        source_sha = sha40(source_entry.get('sha'), 'invalid source blob SHA')
        integer(source_entry.get('size'), 0, MAX_PACKAGE_BYTES, 'source exceeds size limit')
        raw = unpack_blob(api.get(f'/repos/{source_repo}/git/blobs/{source_sha}'), MAX_PACKAGE_BYTES, source_sha)
        require(hashlib.sha256(raw).hexdigest() == digest, 'source digest mismatch')
    return {'package': prefix[:-1], 'artifact_files': len(payload)-1, 'verified_sources': len(sources), 'bytes': total}


def check(event, api, repository):
    require(repository == 'd6g8k5htny-coder/main', 'wrong repository')
    require(isinstance(event, dict) and isinstance(event.get('repository'), dict) and event['repository'].get('full_name') == repository,
            'event repository mismatch')
    number = integer(event.get('number'), 1, 10000000, 'invalid PR number')
    pr = api.get(f'/repos/{repository}/pulls/{number}')
    require(pr.get('number') == number, 'PR number mismatch')
    original = snapshot(pr, repository)
    head, base, head_repo, labels, count = original
    event_pr = event.get('pull_request')
    require(isinstance(event_pr, dict) and isinstance(event_pr.get('head'), dict) and event_pr['head'].get('sha') == head,
            'head changed since event; rerun latest event')
    rows = changes(api, repository, number, count)
    intake = head_repo != repository or 'results-for-review' in labels or any(
        row['filename'].split('/')[0] == 'incoming' or row.get('previous_filename', '').split('/')[0] == 'incoming' for row in rows)
    result = {'head': head, 'base': base, 'scientific_effect': 'NONE', 'lane': 'incoming' if intake else 'maintainer-engineering'}
    if intake:
        result.update(verify_package(api, repository, head, base, rows))
    require(snapshot(api.get(f'/repos/{repository}/pulls/{number}'), repository) == original,
            'PR changed during verification; rerun latest event')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--event', type=Path, default=Path(os.environ.get('GITHUB_EVENT_PATH', '/nonexistent-event')))
    args = parser.parse_args(argv)
    try:
        require(os.environ.get('GITHUB_EVENT_NAME') == 'pull_request_target', 'CI requires trusted pull_request_target context')
        event = strict_json(args.event.read_bytes())
        result = check(event, API(os.environ.get('GH_TOKEN', '')), os.environ.get('GITHUB_REPOSITORY', ''))
        print(json.dumps(result, sort_keys=True))
        return 0
    except (ValueError, OSError, TypeError, KeyError, UnicodeError, RecursionError) as error:
        # Never print response bodies, submitted file contents, or credentials.
        print(json.dumps({'result': 'REJECTED', 'reason': str(error), 'scientific_effect': 'NONE'}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
