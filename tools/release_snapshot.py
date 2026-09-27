#!/usr/bin/env python3
"""Read-only, as-of source/PR custody. No merge or scientific-status authority."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
from datetime import datetime, timezone
import urllib.parse
import urllib.request
import zipfile

OWNER = 'd6g8k5htny-coder'
REPOS = ('main', 'Math-', 'trial', 'governance-', 'sandbox', 'query-',
         'meta-framework', 'google-drive', 'Universal-Law-Workspace', OWNER)


def checked_sha(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{40}', value):
        raise ValueError('expected immutable 40-character lowercase commit')
    return value


def repo_url(name: str) -> str:
    if name not in REPOS:
        raise ValueError('repository is not in the owner allowlist')
    return f'https://github.com/{OWNER}/{name}.git'


def checked_path(value: str) -> str:
    if (not isinstance(value, str) or not value or '\\' in value or '\x00' in value
            or value.startswith('/') or any(p in ('', '.', '..') for p in value.split('/'))):
        raise ValueError('unsafe repository path')
    if PurePosixPath(value).is_absolute():
        raise ValueError('absolute repository path')
    return value


def fresh_output(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError('refusing to overwrite or mix prior snapshot evidence')
    path.mkdir(parents=True, exist_ok=True)


def pages(get, path: str, key: str | None = None) -> list:
    rows = []
    for page in range(1, 101):
        sep = '&' if '?' in path else '?'
        data = get(f'{path}{sep}page={page}&per_page=100')
        if key is not None:
            if not isinstance(data, dict) or key not in data:
                raise ValueError('missing pagination collection: '+key)
            data = data[key]
        if not isinstance(data, list):
            raise ValueError('expected a list page')
        rows.extend(data)
        if len(data) < 100:
            return rows
    raise ValueError('pagination safety cap reached; result is incomplete')


def blob_digest(oid: str, data: bytes) -> str:
    checked_sha(oid)
    actual = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    if actual != oid:
        raise ValueError('Git blob identity mismatch')
    return hashlib.sha256(data).hexdigest()


def git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(['git', '-C', str(repo), *args],
                                   stderr=subprocess.PIPE, timeout=300)


def snapshot_local(repo: Path, sha: str, archive: Path) -> dict:
    checked_sha(sha)
    if archive.exists():
        raise FileExistsError(str(archive))
    if git(repo, 'rev-parse', sha+'^{commit}').decode().strip() != sha:
        raise ValueError('resolved commit mismatch')
    tree = git(repo, 'ls-tree', '-r', '-z', '--full-tree', sha)
    members, gitlinks = [], []
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for row in tree.split(b'\0'):
            if not row:
                continue
            meta, raw_path = row.split(b'\t', 1)
            mode, typ, oid = meta.decode().split()
            path = checked_path(raw_path.decode('utf-8'))
            if typ == 'commit' and mode == '160000':
                gitlinks.append({'path': path, 'commit': checked_sha(oid)})
                continue
            if typ != 'blob' or mode not in ('100644', '100755', '120000'):
                raise ValueError('unsupported Git tree entry')
            data = git(repo, 'cat-file', 'blob', oid)
            digest = blob_digest(oid, data)
            info = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = int(mode, 8) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, data)
            members.append({'path': path, 'git_mode': mode, 'git_blob': oid,
                            'bytes': len(data), 'sha256': digest})
    with archive.open('rb') as f:
        archive_digest = hashlib.file_digest(f, 'sha256').hexdigest()
    return {'commit': sha, 'archive': archive.name,
            'sha256': archive_digest,
            'bytes': archive.stat().st_size, 'members': members,
            'unmaterialized_gitlinks': gitlinks,
            'scope': 'all tracked blobs at one commit; no history, LFS hydration or recursive submodules'}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Refuse renamed-repository and external redirects rather than forward credentials.
        return None


class Client:
    def __init__(self):
        self.token = os.environ.get('GITHUB_TOKEN', '')
        self.opener = urllib.request.build_opener(NoRedirect())

    def get(self, path: str):
        parsed = urllib.parse.urlsplit(path)
        if (parsed.scheme or parsed.netloc or not path.startswith(f'repos/{OWNER}/')
                or path.split('/')[2] not in REPOS or '..' in path.split('/')):
            raise ValueError('API path outside explicit repository allowlist')
        headers = {'Accept': 'application/vnd.github+json', 'User-Agent': 'owner-release-snapshot',
                   'X-GitHub-Api-Version': '2022-11-28'}
        if self.token:
            headers['Authorization'] = 'Bearer '+self.token
        req = urllib.request.Request('https://api.github.com/'+path, headers=headers)
        with self.opener.open(req, timeout=90) as response:
            return json.load(response)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True)+'\n')


def collect(out: Path, include_pr_sources: bool = True) -> dict:
    fresh_output(out)
    client = Client()
    summary = {'schema_version': 1, 'observed_at': datetime.now(timezone.utc).isoformat(),
               'scientific_effect': 'NONE', 'merge_authority': False,
               'scope_repositories': list(REPOS), 'repositories': [], 'errors': []}
    for name in REPOS:
        row = {'repository': f'{OWNER}/{name}', 'pull_requests': [], 'snapshots': []}
        summary['repositories'].append(row)
        dest = out/name; dest.mkdir()
        path = f'repos/{OWNER}/{name}'
        def read(suffix, filename, key=None, paginated=False):
            try:
                value = pages(client.get, path+suffix, key) if paginated else client.get(path+suffix)
                write_json(dest/filename, value)
                return value
            except Exception as exc:
                summary['errors'].append({'repository': name, 'resource': suffix,
                                          'error': type(exc).__name__+': '+str(exc)})
                return None
        meta = read('', 'repository.json')
        if not meta or meta.get('private') is not False or meta.get('full_name') != f'{OWNER}/{name}':
            summary['errors'].append({'repository': name, 'error': 'public identity not established'})
            continue
        branch = meta['default_branch']
        commit = read('/commits/'+urllib.parse.quote(branch,safe=''), 'default-commit.json')
        prs = read('/pulls?state=open', 'open-prs.json', paginated=True)
        targets = {}
        if commit:
            sha = checked_sha(commit['sha']); row['default_commit'] = sha; targets[sha] = ['default']
        if prs is not None:
            row['open_pr_count'] = len(prs)
            for pr in prs:
                n = pr['number']; prefix = f'pr-{n}/'
                detail = read(f'/pulls/{n}', prefix+'detail.json')
                if detail is None:
                    continue
                sha = checked_sha(detail['head']['sha'])
                item = {k:detail.get(k) for k in ('number','title','html_url','draft','mergeable','mergeable_state','state')}
                item.update(head=sha, base=detail['base']['sha'], author=detail['user']['login'])
                row['pull_requests'].append(item)
                read(f'/pulls/{n}/files', prefix+'files.json', paginated=True)
                read(f'/pulls/{n}/reviews', prefix+'reviews.json', paginated=True)
                read(f'/issues/{n}/comments', prefix+'comments.json', paginated=True)
                read(f'/pulls/{n}/comments', prefix+'inline-comments.json', paginated=True)
                read(f'/commits/{sha}/check-runs', prefix+'checks.json', key='check_runs', paginated=True)
                read(f'/commits/{sha}/status', prefix+'combined-status.json')
                read(f'/actions/runs?head_sha={sha}', prefix+'runs.json', key='workflow_runs', paginated=True)
                if include_pr_sources:
                    if detail['head']['repo'] and detail['head']['repo']['full_name'] == f'{OWNER}/{name}':
                        targets.setdefault(sha,[]).append(f'PR{n}')
                    else:
                        item['source_snapshot'] = 'not captured: fork outside same-repository scope'
        read('/rulesets', 'rulesets.json', paginated=True)
        with tempfile.TemporaryDirectory(prefix='release-git-') as tmp:
            repo = Path(tmp)
            subprocess.run(['git','init','--bare','-q',str(repo)],check=True,timeout=30)
            git(repo,'remote','add','origin',repo_url(name))
            for sha, labels in targets.items():
                try:
                    git(repo, 'fetch','--quiet','--depth=1','origin',sha)
                    if git(repo,'rev-parse','FETCH_HEAD').decode().strip() != sha:
                        raise ValueError('fetch returned a different commit')
                    result=snapshot_local(repo,sha,dest/f'{sha}.zip')
                    result['labels']=labels
                    write_json(dest/f'{sha}.manifest.json', result)
                    row['snapshots'].append({k:v for k,v in result.items() if k!='members'})
                except Exception as exc:
                    summary['errors'].append({'repository':name,'commit':sha,
                                              'error':type(exc).__name__+': '+str(exc)})
        write_json(out/'SUMMARY.json',summary)
    summary['completed_at']=datetime.now(timezone.utc).isoformat()
    summary['complete']=not summary['errors']
    write_json(out/'SUMMARY.json',summary)
    lines=[]
    for p in sorted(out.rglob('*')):
        if p.is_file():
            with p.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
            lines.append(f'{digest}  {p.relative_to(out).as_posix()}')
    (out/'SHA256SUMS').write_text('\n'.join(lines)+'\n')
    return summary


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--defaults-only',action='store_true')
    args=p.parse_args()
    result=collect(args.output,not args.defaults_only)
    print(json.dumps({'complete':result['complete'],'repositories':len(result['repositories']),
                      'errors':result['errors']},indent=2))
    return 0 if result['complete'] else 1

if __name__=='__main__': raise SystemExit(main())
