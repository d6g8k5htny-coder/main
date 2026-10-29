#!/usr/bin/env python3
"""Exact-ref CBZ archives and guarded atomic branch pruning (stdlib + Git).

Default commands emit JSON plans. Only --execute writes. Snapshot writes a NEW,
standalone bare repository; publish adds one archive tag; prune deletes eligible
heads in one atomic push with explicit expected-old leases. No source checkout,
index, worktree, trunk update, tag deletion, or hosting-setting mutation occurs.

Caller must supply the complete protected/active branch inventory at snapshot
creation. Hosting protections remain enforced. No credentials are printed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit

MANIFEST_PATH = '.cbz/refs-manifest.json'
OID = re.compile(r'^[0-9a-f]{40}(?:[0-9a-f]{24})?$')


class SafetyError(RuntimeError):
    """A refusal or failed verification; never successful partial cleanup."""


def _git(repo, *args, data=None, env=None, allowed=(0,)):
    command = ['git'] + (['-C', str(repo)] if repo is not None else []) + list(args)
    clean_env = os.environ.copy()
    # Never inherit an index, alternate object database, or Git directory override.
    for key in list(clean_env):
        if key.startswith('GIT_') and key not in ('GIT_SSH_COMMAND', 'GIT_SSH', 'GIT_ASKPASS', 'GIT_TERMINAL_PROMPT'):
            clean_env.pop(key)
    clean_env.update({'GIT_TERMINAL_PROMPT': '0', 'GIT_NO_REPLACE_OBJECTS': '1'})
    if env:
        clean_env.update(env)
    p = subprocess.run(command, input=data, capture_output=True, env=clean_env)
    if p.returncode not in allowed:
        # Git errors can contain credential-bearing URLs. Do not forward stderr.
        raise SafetyError(f'git {args[0]} failed (exit {p.returncode}); no success claimed')
    return p.stdout


def _text(repo, *args, **kwargs):
    return _git(repo, *args, **kwargs).decode('utf-8').strip()


def _canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + '\n').encode()


def _ref(ref):
    if not isinstance(ref, str) or not ref.startswith(('refs/heads/', 'refs/tags/')):
        raise SafetyError('unsupported ref namespace')
    # Git check-ref-format rules, without a subprocess for each of thousands of refs.
    if (any(ord(char) < 33 or ord(char) == 127 or char in '~^:?*[\\' for char in ref)
            or '..' in ref or '@{' in ref or ref.endswith('.')
            or any(not part or part.startswith('.') or part.endswith('.lock') for part in ref.split('/'))):
        raise SafetyError('invalid Git ref name')
    return ref


def _remote(remote):
    if not isinstance(remote, str) or not remote or remote.startswith('-') or '\n' in remote:
        raise SafetyError('invalid remote')
    # URL credentials must never enter a manifest or diagnostic. SSH user names are allowed.
    if '://' in remote:
        parsed = urlsplit(remote)
        if parsed.password or (parsed.username and parsed.scheme != 'ssh') or parsed.query or parsed.fragment:
            raise SafetyError('credential-bearing remote URLs are not accepted')
    return remote


def remote_state(remote):
    """Read exact advertised heads/tags, annotated-tag peeled IDs, and HEAD target."""
    raw = _text(None, 'ls-remote', '--symref', _remote(remote))
    refs, peeled, default = {}, {}, None
    for line in raw.splitlines():
        value, sep, name = line.partition('\t')
        if not sep:
            raise SafetyError('malformed remote advertisement')
        if value.startswith('ref: '):
            if name == 'HEAD':
                default = value[5:]
            continue
        if not OID.fullmatch(value):
            raise SafetyError('malformed remote object ID')
        if name.startswith(('refs/heads/', 'refs/tags/')):
            target = peeled if name.endswith('^{}') else refs
            key = name[:-3] if name.endswith('^{}') else name
            _ref(key)
            if key in target:
                raise SafetyError('duplicate advertised ref')
            target[key] = value
    if default is None or not default.startswith('refs/heads/') or default not in refs:
        raise SafetyError('remote default branch is missing or detached')
    return {'default_ref': default, 'refs': dict(sorted(refs.items())), 'peeled_tags': dict(sorted(peeled.items()))}


def _same(actual, expected, phase):
    if actual != expected:
        raise SafetyError(f'{phase}: remote refs/default/peeled tags changed or mismatch; refresh snapshot')


def _state(manifest):
    return {key: manifest[key] for key in ('default_ref', 'refs', 'peeled_tags')}


def _local_refs(repo):
    raw = _text(repo, 'for-each-ref', '--format=%(objectname) %(refname)', 'refs/heads', 'refs/tags')
    return {ref: oid for oid, ref in (line.split(' ', 1) for line in raw.splitlines())}


def _tree(repo, manifest, manifest_bytes, temp):
    default = manifest['refs'][manifest['default_ref']]
    # Reserved namespace is absent in the default source tree, so its exact entries survive.
    if _git(repo, 'ls-tree', default, '--', '.cbz'):
        raise SafetyError('source tree already uses reserved .cbz path')
    index = str(Path(temp) / 'archive-index')
    env = {'GIT_INDEX_FILE': index}
    _git(repo, 'read-tree', f'{default}^{{tree}}', env=env)
    blob = _text(repo, 'hash-object', '-w', '--stdin', data=manifest_bytes)
    _git(repo, 'update-index', '--add', '--cacheinfo', f'100644,{blob},{MANIFEST_PATH}', env=env)
    return _text(repo, 'write-tree', env=env)


def snapshot(remote, output, archive_tag, *, execute=False, protected=(), active=()):
    """Plan or create a new standalone archive, never modifying the source repo."""
    remote = _remote(remote)
    tag_ref = _ref('refs/tags/' + archive_tag)
    if not archive_tag.startswith('pre-cbz-'):
        raise SafetyError('archive tag must start pre-cbz-')
    state = remote_state(remote)
    if tag_ref in state['refs']:
        raise SafetyError('archive tag already exists remotely; choose a new name')
    policies = []
    for names in (protected, active):
        refs = sorted(set(names))
        for ref in refs:
            if not ref.startswith('refs/heads/') or ref not in state['refs']:
                raise SafetyError('policy must name an existing full branch ref')
            _ref(ref)
        policies.append(refs)
    manifest = {'schema': 'cbz-refs-v1', 'remote': remote, 'archive_tag': archive_tag,
                'created_utc': datetime.now(timezone.utc).isoformat(), **state,
                'protected_refs': policies[0], 'active_refs': policies[1]}
    result = {'mode': 'execute' if execute else 'dry-run', 'action': 'snapshot', 'manifest': manifest}
    if not execute:
        return result
    output = Path(output).absolute()
    if output.exists():
        raise SafetyError('snapshot output exists; refusing overwrite')
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='.cbz-snapshot-', dir=output.parent))
    try:
        repo = temp / 'repository.git'
        # Match the advertised object format; repositories with mixed IDs are invalid.
        lengths = {len(oid) for oid in state['refs'].values()}
        if len(lengths) != 1:
            raise SafetyError('mixed object ID formats')
        fmt = 'sha256' if lengths == {64} else 'sha1'
        _git(None, 'init', '--bare', f'--object-format={fmt}', str(repo))
        _git(repo, 'fetch', '--no-tags', '--no-write-fetch-head', remote,
             '+refs/heads/*:refs/heads/*', '+refs/tags/*:refs/tags/*')
        _same(remote_state(remote), state, 'snapshot post-fetch')
        if _local_refs(repo) != state['refs']:
            raise SafetyError('fetched original refs do not match advertisement')
        encoded = _canonical(manifest)
        tree = _tree(repo, manifest, encoded, temp)
        parents = sorted({oid for ref, oid in state['refs'].items() if ref.startswith('refs/heads/')})
        args = ['commit-tree', tree]
        for parent in parents:
            args.extend(['-p', parent])
        ident = {'GIT_AUTHOR_NAME': 'CBZ ref archive', 'GIT_AUTHOR_EMAIL': 'cbz@invalid',
                 'GIT_COMMITTER_NAME': 'CBZ ref archive', 'GIT_COMMITTER_EMAIL': 'cbz@invalid'}
        commit = _text(repo, *args, data=b'Pre-CBZ exact ref archive; all unique original branch tips are parents.\n', env=ident)
        _git(repo, 'update-ref', tag_ref, commit, '0' * len(commit))
        # Archive custody has one named archive tag, not hundreds of dangling heads.
        # Keep original tag refs too: annotated/tag-only objects need exact retention.
        transaction = ['start']
        transaction.extend(f'delete {ref} {oid}' for ref, oid in state['refs'].items()
                           if ref.startswith('refs/heads/'))
        transaction.extend(['prepare', 'commit'])
        _git(repo, 'update-ref', '--stdin', data=('\n'.join(transaction) + '\n').encode())
        _git(repo, 'update-ref', '--no-deref', 'HEAD', commit)
        (temp / 'manifest.json').write_bytes(encoded)
        receipt = {'schema': 'cbz-refs-receipt-v1', 'archive_commit': commit,
                   'archive_tree': tree, 'manifest_sha256': hashlib.sha256(encoded).hexdigest()}
        (temp / 'receipt.json').write_bytes(_canonical(receipt))
        (temp / 'archive-index').unlink(missing_ok=True)
        verify_snapshot(temp)
        _same(remote_state(remote), state, 'snapshot final readback')
        temp.rename(output)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return {**result, 'verified': True, 'archive_commit': commit, 'output': str(output)}


def verify_snapshot(output):
    """Verify exact manifest bytes, expected tree/parents, tag and all original objects."""
    output = Path(output)
    try:
        encoded = (output / 'manifest.json').read_bytes()
        manifest = json.loads(encoded)
        receipt = json.loads((output / 'receipt.json').read_bytes())
        if manifest['schema'] != 'cbz-refs-v1' or encoded != _canonical(manifest):
            raise SafetyError('manifest schema or canonical bytes mismatch')
        if hashlib.sha256(encoded).hexdigest() != receipt['manifest_sha256']:
            raise SafetyError('manifest hash mismatch')
        tag_ref = _ref('refs/tags/' + manifest['archive_tag'])
        if not manifest['archive_tag'].startswith('pre-cbz-') or tag_ref in manifest['refs']:
            raise SafetyError('invalid archive tag')
        for ref, oid in manifest['refs'].items():
            _ref(ref)
            if not OID.fullmatch(oid):
                raise SafetyError('invalid manifest object ID')
        repo = output / 'repository.git'
        if _text(repo, 'rev-parse', '--is-shallow-repository') != 'false':
            raise SafetyError('shallow archive is not acceptable')
        commit = _text(repo, 'rev-parse', '--verify', tag_ref)
        if commit != receipt['archive_commit'] or _text(repo, 'cat-file', '-t', commit) != 'commit':
            raise SafetyError('archive tag/commit mismatch')
        expected_refs = {ref: oid for ref, oid in manifest['refs'].items() if ref.startswith('refs/tags/')}
        expected_refs[tag_ref] = commit
        if _local_refs(repo) != expected_refs:
            raise SafetyError('archive original tag refs/archive tag membership mismatch or unexpected heads')
        if (_text(repo, 'symbolic-ref', '-q', 'HEAD', allowed=(0, 1))
                or _text(repo, 'rev-parse', '--verify', 'HEAD') != commit):
            raise SafetyError('archive HEAD must be detached at archive commit')
        if _git(repo, 'show', f'{commit}:{MANIFEST_PATH}') != encoded:
            raise SafetyError('embedded exact manifest mismatch')
        tips = sorted({oid for ref, oid in manifest['refs'].items() if ref.startswith('refs/heads/')})
        actual_parents = _text(repo, 'show', '-s', '--format=%P', commit).split()
        if actual_parents != tips:
            raise SafetyError('archive parent membership mismatch')
        with tempfile.TemporaryDirectory(prefix='cbz-verify-') as temp:
            expected_tree = _tree(repo, manifest, encoded, temp)
        actual_tree = _text(repo, 'show', '-s', '--format=%T', commit)
        if actual_tree != expected_tree or receipt['archive_tree'] != actual_tree:
            raise SafetyError('archive source tree/manifest membership mismatch')
        # Batch object checks and one reachability walk keep thousand-tip archives bounded.
        originals = sorted(set(manifest['refs'].values()))
        checked = _text(repo, 'cat-file', '--batch-check=%(objectname) %(objecttype)',
                        data=('\n'.join(originals) + '\n').encode()).splitlines()
        if len(checked) != len(originals):
            raise SafetyError('original object membership mismatch')
        for oid, row in zip(originals, checked):
            parts = row.split()
            if len(parts) != 2 or parts[0] != oid or parts[1] not in ('commit', 'tree', 'blob', 'tag'):
                raise SafetyError('missing original ref object')
        reachable = set(_text(repo, 'rev-list', commit).splitlines())
        if not set(tips).issubset(reachable):
            raise SafetyError('original branch tip is not reachable from archive')
        for ref, oid in manifest['peeled_tags'].items():
            original = manifest['refs'][ref]
            if _text(repo, 'rev-parse', f'{original}^{{}}') != oid:
                raise SafetyError('original tag peeled identity mismatch')
        _git(repo, 'fsck', '--full', '--no-reflogs')
        return {'verified': True, 'archive_commit': commit, 'manifest': manifest}
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SafetyError('invalid or incomplete snapshot metadata') from error


def _published_state(manifest, commit):
    expected = _state(manifest)
    return {**expected, 'refs': {**expected['refs'], 'refs/tags/' + manifest['archive_tag']: commit}}


def _remote_archive_readback(repo, remote, tag_ref, commit):
    # Independent fresh fetch of the remote tag, not trust in a local receipt/tag.
    with tempfile.TemporaryDirectory(prefix='cbz-readback-') as temp:
        objfmt = _text(repo, 'rev-parse', '--show-object-format')
        _git(None, 'init', '--bare', f'--object-format={objfmt}', temp)
        _git(temp, 'fetch', '--no-tags', '--no-write-fetch-head', remote, f'{tag_ref}:refs/tags/readback')
        if _text(temp, 'rev-parse', 'refs/tags/readback') != commit:
            raise SafetyError('remote archive tag readback mismatch')
        if _git(temp, 'cat-file', 'commit', commit) != _git(repo, 'cat-file', 'commit', commit):
            raise SafetyError('remote archive commit bytes mismatch')
        _git(temp, 'fsck', '--full', '--no-reflogs')


def publish(output, *, execute=False):
    verified = verify_snapshot(output)
    manifest, commit = verified['manifest'], verified['archive_commit']
    remote = _remote(manifest['remote'])
    expected = _published_state(manifest, commit)
    actual = remote_state(remote)
    already = actual == expected
    if not already:
        _same(actual, _state(manifest), 'publish preflight')
    result = {'action': 'publish', 'mode': 'execute' if execute else 'dry-run',
              'archive_tag': manifest['archive_tag'], 'archive_commit': commit,
              'already_published': already}
    if not execute:
        return result
    repo = Path(output) / 'repository.git'
    tag_ref = 'refs/tags/' + manifest['archive_tag']
    if not already:
        _git(repo, 'push', '--atomic', f'--force-with-lease={tag_ref}:', remote, f'{commit}:{tag_ref}')
    _same(remote_state(remote), expected, 'publish final readback')
    _remote_archive_readback(repo, remote, tag_ref, commit)
    return {**result, 'verified': True}


def prune(output, *, execute=False):
    verified = verify_snapshot(output)
    manifest, commit = verified['manifest'], verified['archive_commit']
    remote = _remote(manifest['remote'])
    expected = _published_state(manifest, commit)
    _same(remote_state(remote), expected, 'prune preflight (remote archive required)')
    excluded = {manifest['default_ref'], *manifest['protected_refs'], *manifest['active_refs']}
    delete_refs = sorted(ref for ref in manifest['refs'] if ref.startswith('refs/heads/') and ref not in excluded)
    result = {'action': 'prune', 'mode': 'execute' if execute else 'dry-run',
              'delete_refs': delete_refs, 'expected_old_refs': {ref: manifest['refs'][ref] for ref in delete_refs},
              'excluded_refs': sorted(excluded), 'archive_commit': commit}
    if not execute:
        return result
    repo = Path(output) / 'repository.git'
    _remote_archive_readback(repo, remote, 'refs/tags/' + manifest['archive_tag'], commit)
    # Fresh full check immediately before CAS deletion; changed unrelated refs reject too.
    _same(remote_state(remote), expected, 'prune immediate pre-push')
    if delete_refs:
        leases = [f'--force-with-lease={ref}:{manifest["refs"][ref]}' for ref in delete_refs]
        _git(repo, 'push', '--atomic', *leases, remote, *(':' + ref for ref in delete_refs))
    final = {**expected, 'refs': {ref: oid for ref, oid in expected['refs'].items() if ref not in delete_refs}}
    _same(remote_state(remote), final, 'prune final readback')
    _remote_archive_readback(repo, remote, 'refs/tags/' + manifest['archive_tag'], commit)
    return {**result, 'verified': True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    snap = sub.add_parser('snapshot', help='read-only plan, or create a new standalone bare archive')
    snap.add_argument('--remote', required=True)
    snap.add_argument('--output', required=True)
    snap.add_argument('--archive-tag', required=True)
    snap.add_argument('--protected-ref', action='append', default=[])
    snap.add_argument('--active-ref', action='append', default=[])
    snap.add_argument('--execute', action='store_true')
    for name in ('publish', 'prune', 'verify'):
        command = sub.add_parser(name)
        command.add_argument('--snapshot', required=True)
        if name != 'verify':
            command.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.command == 'snapshot':
            result = snapshot(args.remote, args.output, args.archive_tag, execute=args.execute,
                              protected=args.protected_ref, active=args.active_ref)
        elif args.command == 'verify':
            result = verify_snapshot(args.snapshot)
        else:
            result = globals()[args.command](args.snapshot, execute=args.execute)
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    except SafetyError as error:
        print(json.dumps({'ok': False, 'error': str(error), 'action': args.command,
                          'note': 'No partial-success claim; inspect fresh remote refs after any failed execute.'}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
