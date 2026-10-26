"""Pure identity checks for an explicitly frozen pull-request merge source.

Callers must acquire literal commit bytes and authenticate this module before
loading it. This function performs no Git, filesystem, network or child work.
"""
import hashlib
import re


class SourceJoinError(ValueError):
    """The supplied source and native identities cannot establish this join."""


def verify_merge_join(*, event_name: str, repository: str, event_ref: str,
                      event_sha: str, pr_number: int, event_head_sha: str,
                      event_base_sha: str, checked_commit: str,
                      actual_commit_raw: bytes, frozen_base_commit: str) -> dict:
    """Validate literal native identities; return the retained join record."""
    for value in (event_sha, event_head_sha, event_base_sha,
                  checked_commit, frozen_base_commit):
        if type(value) is not str or re.fullmatch(r'[0-9a-f]{40}', value) is None:
            raise SourceJoinError('Invalid source commit identity')
    if (type(event_name) is not str or event_name != 'pull_request'
            or type(repository) is not str or repository != 'd6g8k5htny-coder/main'
            or type(pr_number) is not int or not 1 <= pr_number < 10**12
            or type(event_ref) is not str
            or event_ref != 'refs/pull/%d/merge' % pr_number):
        raise SourceJoinError('Unexpected repository or pull-request event')
    if checked_commit != event_sha:
        raise SourceJoinError('Checked commit differs from tested event')
    if type(actual_commit_raw) is not bytes or not 0 < len(actual_commit_raw) <= 65536:
        raise SourceJoinError('Invalid or oversized literal commit bytes')
    framed = b'commit ' + str(len(actual_commit_raw)).encode('ascii') + b'\0' + actual_commit_raw
    if hashlib.sha1(framed).hexdigest() != checked_commit:
        raise SourceJoinError('Literal commit bytes differ from checked commit')
    header, separator, _message = actual_commit_raw.partition(b'\n\n')
    if not separator or not header or b'\0' in header:
        raise SourceJoinError('Invalid literal commit header')
    lines = header.split(b'\n')
    if not 5 <= len(lines) <= 128 or re.fullmatch(rb'tree [0-9a-f]{40}', lines[0]) is None:
        raise SourceJoinError('Invalid tree header or header line count')
    if any(re.fullmatch(rb'parent [0-9a-f]{40}', line) is None for line in lines[1:3]):
        raise SourceJoinError('Exactly two literal ordered parents required')
    previous = None
    counts = {'author': 0, 'committer': 0}
    for line in lines[3:]:
        if not line or re.search(rb'[\x00-\x1f\x7f]', line):
            raise SourceJoinError('Invalid control or empty commit header')
        if line.startswith(b' '):
            if previous is None or previous in (b'author', b'committer', b'encoding'):
                raise SourceJoinError('Orphan or invalid header continuation')
            continue
        if re.fullmatch(rb'[a-z][a-z0-9-]* [^\x00-\x1f\x7f]+', line) is None:
            raise SourceJoinError('Malformed remaining commit header')
        key = line.split(b' ', 1)[0]
        if key in (b'parent', b'tree'):
            raise SourceJoinError('Extra parent or tree header')
        if key.decode('ascii') in counts:
            counts[key.decode('ascii')] += 1
        previous = key
    if counts != {'author': 1, 'committer': 1}:
        raise SourceJoinError('Missing or duplicate author or committer')
    parents = [line[7:].decode('ascii') for line in lines[1:3]]
    # The event base is retained as observed metadata; only the explicitly
    # frozen base and event head identify the required actual parent order.
    if parents != [frozen_base_commit, event_head_sha]:
        raise SourceJoinError('Frozen-base/head/tested-merge join mismatch')
    return {
        'event_name': event_name, 'repository': repository, 'event_ref': event_ref,
        'event_sha': event_sha, 'pull_request_number': pr_number,
        'pull_request_head_sha': event_head_sha, 'pull_request_base_sha': event_base_sha,
        'checked_commit': checked_commit, 'actual_tree_sha': lines[0][5:].decode('ascii'),
        'actual_merge_parents': parents, 'candidate_base_commit': frozen_base_commit,
    }
