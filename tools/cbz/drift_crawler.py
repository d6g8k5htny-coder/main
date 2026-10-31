#!/usr/bin/env python3
"""Dynamic federation drift crawler: CBZ-0 gate G7 (stdlib plus git). Scientific effect: NONE.

Discovers the repository topology at run time; no repository list is coded in.
Starting from a root repository (default: the Universal-Law-Workspace federation
root), it follows, at each repository's default-branch head:
  * submodule gitlinks and `.gitmodules` URLs (the root pins);
  * `lake-manifest.json` git packages whose URL names a repository of the same owner;
  * JSON pin records: any JSON object that names a repository of the owner and a
    40-hex commit (keys `repository`/`repo`/`full_name` with `commit`/`rev`/`sha`/
    `revision`), optionally with a `path` and a `blob`/`git_blob`.
Every repository reached that way is crawled in turn, until closure.

Checks (blocking unless stated):
  P1 every pin resolves to an existing commit of its repository;
  P2 every pinned commit lies on that repository's default branch (no zombie pins);
  P3 a pin record's recorded blob equals the blob at commit:path;
  P4 every Lean package of the federation pins one Mathlib revision and one toolchain;
  D1 a namespaced Lean declaration that appears in several files has one statement,
     unless one file's bytes occur in the other file's own Git history (a pinned
     older snapshot, reported as SNAPSHOT) or the registry
     tools/cbz/drift_allowlist.json lists that exact case with a reason;
  L  root-pin lag (commits behind the default head) is reported; with
     --require-current-root-pins a lagging root pin is blocking (baseline lock).
Signatures are lexical (lean_decls), not elaborated types; see TOOLING.md.
Exit status 0 only when nothing blocking is found.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lean_decls  # noqa: E402

SCHEMA = 1
HEX40 = re.compile(r'[0-9a-f]{40}')
REPO_KEYS = ('repository', 'repo', 'full_name', 'source_repository')
COMMIT_KEYS = ('commit', 'rev', 'sha', 'revision', 'commit_sha')
BLOB_KEYS = ('blob', 'git_blob', 'blob_sha')
MAX_JSON_BYTES = 8 * 1024 * 1024
# Layer B rule (GOVERNANCE.md): JSON under these directory names is test data, not a federation pin.
# Excluded records are counted in the report, never dropped silently.
TEST_DATA_DIRS = {'tests', 'test', 'fixtures', 'examples', 'testdata', '__tests__'}


def run_git(path, *args, check=True, data=None):
    result = subprocess.run(['git', '-C', str(path), *args], input=data, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    if check and result.returncode != 0:
        raise RuntimeError(f'git {" ".join(args[:3])} failed in {path}: {result.stderr.decode()[:300]}')
    return result


def normalize_url(url):
    url = url.strip().rstrip('/')
    if url.endswith('.git'):
        url = url[:-4]
    m = re.match(r'(?:git@github\.com:|https?://github\.com/)([^/]+)/([^/]+)$', url)
    if m:
        return f'github:{m.group(1).lower()}/{m.group(2).lower()}', m.group(1), m.group(2)
    return 'path:' + str(Path(url).resolve()), None, Path(url).name


class Federation:
    def __init__(self, cache, local, offline, owner=None, heads=None):
        self.cache = Path(cache)
        self.local = {normalize_url(k)[0]: Path(v) for k, v in local.items()}
        self.heads = {normalize_url(k)[0]: v for k, v in (heads or {}).items()}
        self.offline = offline
        self.owner = owner
        self.repos = {}      # key -> {'key','url','name','path','head','branch','error'}
        self.fetched = set()

    def resolve(self, url):
        key, owner, name = normalize_url(url)
        if key in self.repos:
            return self.repos[key]
        info = {'key': key, 'url': url, 'name': name, 'owner': owner, 'path': None, 'head': None,
                'branch': None, 'error': None}
        self.repos[key] = info
        path = self.local.get(key)
        if path is None and key.startswith('path:'):
            path = Path(key[len('path:'):])
        if path is None:
            path = self.cache / f'{owner}__{name}'
            if not (path / '.git').exists() and not self.offline:
                self.cache.mkdir(parents=True, exist_ok=True)
                clone = subprocess.run(['git', 'clone', '-q', '--filter=blob:none', '--no-checkout', '--single-branch',
                                        url, str(path)],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if clone.returncode != 0:
                    info['error'] = 'unreachable repository: ' + clone.stderr.decode()[:200].strip()
                    return info
        if not (path / '.git').exists() and not (path / 'HEAD').exists():
            info['error'] = 'repository not available locally'
            return info
        info['path'] = path
        if not self.offline and key not in self.fetched and not key.startswith('path:'):
            run_git(path, 'fetch', '-q', 'origin', check=False)
            self.fetched.add(key)
        head = None
        refs = ('refs/remotes/origin/HEAD', 'refs/remotes/origin/main', 'refs/remotes/origin/master', 'HEAD')
        if key in self.heads:
            refs = (self.heads[key],)
        for ref in refs:
            r = run_git(path, 'rev-parse', '--verify', '-q', ref + '^{commit}', check=False)
            if r.returncode == 0:
                head = r.stdout.decode().strip()
                info['branch'] = ref
                break
        if head is None:
            info['error'] = 'no default branch head'
            return info
        info['head'] = head
        return info

    def object_type(self, info, sha):
        """'commit', 'blob', 'tree' or None when the object is absent even after a fetch."""
        if info['path'] is None:
            return None
        r = run_git(info['path'], 'cat-file', '-t', sha, check=False)
        if r.returncode != 0 and not self.offline:
            run_git(info['path'], 'fetch', '-q', 'origin', sha, check=False)
            r = run_git(info['path'], 'cat-file', '-t', sha, check=False)
        return r.stdout.decode().strip() if r.returncode == 0 else None

    def archive_refs(self, info):
        """{refname: sha} of the remote's archive refs (`refs/heads/archive/*`, `refs/tags/archive/*`).

        Only refs the remote publishes count; a local-only tag is not an archive.
        Offline, the remote-tracking `refs/remotes/origin/archive/*` branches are used.
        """
        if 'archive_refs' in info:
            return info['archive_refs']
        refs = {}
        if not self.offline and not info['key'].startswith('path:'):
            r = run_git(info['path'], 'ls-remote', 'origin', 'refs/heads/archive/*', 'refs/tags/archive/*', check=False)
            for line in r.stdout.decode().splitlines():
                sha, _, name = line.partition('\t')
                if name.endswith('^{}'):
                    name = name[:-3]
                refs[name] = sha
            if refs:
                run_git(info['path'], 'fetch', '-q', 'origin', *sorted(set(refs.values())), check=False)
        else:
            r = run_git(info['path'], 'for-each-ref', '--format=%(objectname) %(refname)', 'refs/remotes/origin/archive/',
                        check=False)
            for line in r.stdout.decode().splitlines():
                sha, _, name = line.partition(' ')
                refs['refs/heads/' + name[len('refs/remotes/origin/'):]] = sha
        info['archive_refs'] = refs
        return refs

    def archived(self, info, commit):
        out = []
        for name, sha in sorted(self.archive_refs(info).items()):
            if run_git(info['path'], 'merge-base', '--is-ancestor', commit, sha, check=False).returncode == 0:
                out.append(name)
        return out

    def on_default_branch(self, info, commit):
        r = run_git(info['path'], 'merge-base', '--is-ancestor', commit, info['head'], check=False)
        if r.returncode == 1 and not self.offline and run_git(info['path'], 'rev-parse', '--is-shallow-repository',
                                                             check=False).stdout.strip() == b'true':
            run_git(info['path'], 'fetch', '-q', '--unshallow', '--filter=blob:none', 'origin', check=False)
            r = run_git(info['path'], 'merge-base', '--is-ancestor', commit, info['head'], check=False)
        return r.returncode == 0

    def behind(self, info, commit):
        r = run_git(info['path'], 'rev-list', '--count', f'{commit}..{info["head"]}', check=False)
        return int(r.stdout.decode().strip()) if r.returncode == 0 else None


def tree(info, commit):
    out = run_git(info['path'], 'ls-tree', '-r', '-z', '--full-tree', commit).stdout.split(b'\0')
    entries = {}
    for entry in out:
        if not entry:
            continue
        meta, _, name = entry.partition(b'\t')
        mode, kind, sha = meta.split(b' ')
        entries[name.decode('utf-8', 'surrogateescape')] = (mode.decode(), kind.decode(), sha.decode())
    return entries


def read_blobs(info, shas):
    if not shas:
        return {}
    ordered = sorted(set(shas))
    raw = run_git(info['path'], 'cat-file', '--batch', data=''.join(s + '\n' for s in ordered).encode()).stdout
    out, offset = {}, 0
    for sha in ordered:
        header_end = raw.index(b'\n', offset)
        parts = raw[offset:header_end].split(b' ')
        if len(parts) == 2 and parts[1] == b'missing':
            offset = header_end + 1
            continue
        size = int(parts[2])
        out[sha] = raw[header_end + 1:header_end + 1 + size]
        offset = header_end + 1 + size + 1
    return out


def gitmodules(text):
    sections, current = [], None
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r'\[submodule\s+"([^"]+)"\]', line)
        if m:
            current = {'name': m.group(1)}
            sections.append(current)
        elif current is not None and '=' in line:
            key, _, value = line.partition('=')
            current[key.strip()] = value.strip()
    return sections


def json_pins(value, path_hint=''):
    """Yield (repository, commit, path, blob) from every JSON object that carries a pin."""
    stack = [value]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            repo = next((node[k] for k in REPO_KEYS if isinstance(node.get(k), str)), None)
            commit = next((node[k] for k in COMMIT_KEYS if isinstance(node.get(k), str) and HEX40.fullmatch(node[k])), None)
            if repo and commit:
                blob = next((node[k] for k in BLOB_KEYS if isinstance(node.get(k), str) and HEX40.fullmatch(node[k])), None)
                path = node.get('path') if isinstance(node.get('path'), str) else None
                yield repo, commit, path, blob
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)


def repo_url_for(name, owner, known):
    """Map a pin's repository field to a URL within the owner's namespace, or None."""
    name = name.strip()
    if name.startswith(('http://', 'https://', 'git@')):
        key, o, _ = normalize_url(name)
        return name if o and owner and o.lower() == owner.lower() else None
    if '/' in name:
        o, _, n = name.partition('/')
        if owner and o.lower() == owner.lower() and n:
            return f'https://github.com/{o}/{n}'
        return None
    if owner and re.fullmatch(r'[A-Za-z0-9._-]+', name):
        candidate = f'https://github.com/{owner}/{name}'
        return candidate if normalize_url(candidate)[0] in known else None
    return None


def crawl(root_url, fed, require_current_root_pins=False, allowlist=()):
    report = {'schema': SCHEMA, 'scientific_effect': 'NONE', 'root': root_url, 'repositories': [], 'edges': [],
              'findings': [], 'declarations': {}}
    root = fed.resolve(root_url)
    if fed.owner is None:
        fed.owner = root.get('owner')
    queue, seen = [root['key']], set()
    edges = []
    while queue:
        key = queue.pop(0)
        if key in seen:
            continue
        seen.add(key)
        info = fed.repos[key]
        if info['error']:
            # Every repository but the root is reached through a pin, and check_pins reports it there with the
            # citing file; only the root needs its own finding.
            if key == root['key']:
                report['findings'].append({'check': 'P1', 'blocking': True, 'repository': info['url'],
                                           'error': info['error']})
            continue
        entries = tree(info, info['head'])
        modules = {}
        if '.gitmodules' in entries:
            text = read_blobs(info, [entries['.gitmodules'][2]])[entries['.gitmodules'][2]].decode('utf-8', 'replace')
            modules = {s.get('path'): s for s in gitmodules(text) if s.get('path')}
        for path, (mode, kind, sha) in sorted(entries.items()):
            if kind == 'commit':
                url = modules.get(path, {}).get('url')
                if not url:
                    report['findings'].append({'check': 'P1', 'blocking': True, 'repository': info['url'],
                                               'error': f'gitlink {path} has no .gitmodules URL'})
                    continue
                if url.startswith(('./', '../')) and info['key'].startswith('path:'):
                    url = str((Path(info['path']) / url).resolve())
                edges.append({'kind': 'submodule', 'from': info['url'], 'file': path, 'to': url, 'commit': sha,
                              'root': key == root['key']})
        lake_revs, toolchains = [], []
        json_paths = [p for p, (m, k, s) in entries.items() if k == 'blob' and p.endswith('.json')]
        blobs = read_blobs(info, [entries[p][2] for p in json_paths if p.endswith('lake-manifest.json')
                                  or p.endswith('.json')])
        for p in json_paths:
            raw = blobs.get(entries[p][2])
            if raw is None or len(raw) > MAX_JSON_BYTES:
                continue
            try:
                data = json.loads(raw.decode('utf-8'))
            except (UnicodeDecodeError, ValueError):
                continue
            if p.endswith('lake-manifest.json') and isinstance(data, dict):
                for pkg in data.get('packages', []):
                    if isinstance(pkg, dict) and pkg.get('type') == 'git':
                        if pkg.get('name') == 'mathlib':
                            lake_revs.append((p, pkg.get('rev')))
                        url = pkg.get('url', '')
                        if repo_url_for(url, fed.owner, fed.repos) and HEX40.fullmatch(pkg.get('rev') or ''):
                            edges.append({'kind': 'lake', 'from': info['url'], 'file': p, 'to': url,
                                          'commit': pkg['rev'], 'root': False})
                continue
            test_data = bool(TEST_DATA_DIRS & set(p.split('/')[:-1]))
            for repo, commit, pin_path, blob in json_pins(data):
                if test_data:
                    report.setdefault('excluded_test_records', 0)
                    report['excluded_test_records'] += 1
                    continue
                url = repo_url_for(repo, fed.owner, fed.repos)
                if url is None and fed.owner and '/' not in repo and not repo.startswith(('http', 'git@')):
                    url = f'https://github.com/{fed.owner}/{repo}'
                if url is None:
                    continue
                edges.append({'kind': 'record', 'from': info['url'], 'file': p, 'to': url, 'commit': commit,
                              'path': pin_path, 'blob': blob, 'root': False})
        for p, (m, k, s) in entries.items():
            if p.endswith('lean-toolchain') and k == 'blob':
                toolchains.append((p, read_blobs(info, [s]).get(s, b'').decode('utf-8', 'replace').strip()))
        report['repositories'].append({'url': info['url'], 'head': info['head'], 'branch': info['branch'],
                                       'mathlib': sorted({r for _, r in lake_revs if r}),
                                       'toolchains': sorted({t for _, t in toolchains})})
        info['lake_revs'], info['toolchains'] = lake_revs, toolchains
        for edge in edges:
            if edge['from'] == info['url']:
                target = fed.resolve(edge['to'])
                if target['key'] not in seen and target['key'] not in queue:
                    queue.append(target['key'])
    report['edges'] = edges
    report.setdefault('excluded_test_records', 0)
    check_pins(report, fed, edges, require_current_root_pins)
    check_skew(report, fed, seen)
    check_declarations(report, fed, seen, set(allowlist))
    merged = {}
    for f in report['findings']:
        key = json.dumps({k: v for k, v in f.items() if k not in ('from',)}, sort_keys=True, default=str)
        if key in merged:
            merged[key]['count'] += 1
            merged[key]['sources'].add(f.get('from'))
        else:
            merged[key] = {**f, 'count': 1, 'sources': {f.get('from')}}
    report['findings'] = [{**f, 'sources': sorted(x for x in f['sources'] if x)} for f in merged.values()]
    report['summary'] = {
        'repositories': len(report['repositories']),
        'excluded_test_records': report['excluded_test_records'],
        'edges': {k: sum(1 for e in edges if e['kind'] == k) for k in ('submodule', 'lake', 'record')},
        'blocking': sum(f['count'] for f in report['findings'] if f['blocking']),
        'informational': sum(f['count'] for f in report['findings'] if not f['blocking']),
    }
    report['verdict'] = 'PASS' if report['summary']['blocking'] == 0 else 'FAIL'
    return report


def check_pins(report, fed, edges, require_current_root_pins):
    by_commit = {}
    for edge in edges:
        target = fed.resolve(edge['to'])
        where = {'from': edge['from'], 'file': edge['file'], 'to': target['url'], 'commit': edge['commit'],
                 'kind': edge['kind']}
        if target['error']:
            report['findings'].append({'check': 'P1', 'blocking': True, **where, 'error': target['error']})
            continue
        resolved = by_commit.get((target['key'], edge['commit']))
        if resolved is None:
            kind = fed.object_type(target, edge['commit'])
            if kind == 'commit':
                if fed.on_default_branch(target, edge['commit']):
                    resolved = 'on-default'
                else:
                    tags = fed.archived(target, edge['commit'])
                    resolved = ('archived', tags) if tags else 'off-default'
            else:
                resolved = kind
            by_commit[(target['key'], edge['commit'])] = resolved
        if resolved is None:
            report['findings'].append({'check': 'P1', 'blocking': True, **where, 'error': 'pinned object does not exist'})
            continue
        if resolved in ('blob', 'tree'):
            continue  # a content pin: the object exists, which is what it asserts
        if resolved == 'off-default':
            report['findings'].append({'check': 'P2', 'blocking': True, **where,
                                       'error': 'pinned commit is neither on the default branch nor under an archive tag'})
            continue
        if isinstance(resolved, tuple):
            report['findings'].append({'check': 'P2', 'blocking': False, 'status': 'ARCHIVED', **where,
                                       'tags': resolved[1]})
        if edge.get('path') and edge.get('blob'):
            entries = tree_cache(fed, target, edge['commit'])
            actual = entries.get(edge['path'])
            if actual is None or actual[2] != edge['blob']:
                report['findings'].append({'check': 'P3', 'blocking': True, **where, 'path': edge['path'],
                                           'recorded_blob': edge['blob'], 'actual_blob': actual[2] if actual else None})
        if edge['kind'] == 'submodule' and edge['root']:
            lag = fed.behind(target, edge['commit'])
            report['findings'].append({'check': 'L', 'blocking': bool(require_current_root_pins and lag), **where,
                                       'behind_default_head': lag, 'default_head': target['head']})


_TREES = {}


def tree_cache(fed, info, commit):
    key = (info['key'], commit)
    if key not in _TREES:
        _TREES[key] = tree(info, commit)
    return _TREES[key]


def check_skew(report, fed, keys):
    mathlib, toolchains = {}, {}
    for key in keys:
        info = fed.repos[key]
        for path, rev in info.get('lake_revs', []):
            mathlib.setdefault(rev, []).append(info['url'] + ':' + path)
        for path, value in info.get('toolchains', []):
            toolchains.setdefault(value, []).append(info['url'] + ':' + path)
    if len(mathlib) > 1:
        report['findings'].append({'check': 'P4', 'blocking': True, 'error': 'Mathlib revision skew',
                                   'revisions': {k: sorted(v) for k, v in mathlib.items()}})
    if len(toolchains) > 1:
        report['findings'].append({'check': 'P4', 'blocking': True, 'error': 'Lean toolchain skew',
                                   'toolchains': {k: sorted(v) for k, v in toolchains.items()}})


def check_declarations(report, fed, keys, allowlist):
    occurrences = {}
    for key in sorted(keys):
        info = fed.repos[key]
        if info['error']:
            continue
        entries = tree_cache(fed, info, info['head'])
        lean = {p: v for p, v in entries.items() if p.endswith('.lean') and v[1] == 'blob'}
        blobs = read_blobs(info, [v[2] for v in lean.values()])
        for path, (_, _, sha) in sorted(lean.items()):
            raw = blobs.get(sha)
            if raw is None:
                continue
            try:
                decls, _ = lean_decls.declarations(raw.decode('utf-8'))
            except UnicodeDecodeError:
                continue
            for d in decls:
                if d['qualified'] and '.' in d['qualified'] and d['kind'] not in ('instance', 'example'):
                    occurrences.setdefault(d['qualified'], []).append(
                        {'repository': info['url'], 'key': key, 'path': path, 'blob': sha, 'kind': d['kind'],
                         'signature_sha256': d['signature_sha256'], 'line': d['line']})
    groups = {name: occ for name, occ in occurrences.items() if len({o['signature_sha256'] for o in occ}) > 1}
    report['declarations'] = {'names': len(occurrences),
                              'shared_names': sum(1 for o in occurrences.values() if len(o) > 1),
                              'differing_names': len(groups)}
    history = {}
    for name, occ in sorted(groups.items()):
        drift = []
        for i, a in enumerate(occ):
            for b in occ[i + 1:]:
                if a['signature_sha256'] == b['signature_sha256']:
                    continue
                if snapshot_of(fed, history, a, b) or snapshot_of(fed, history, b, a):
                    report['findings'].append({'check': 'D1', 'blocking': False, 'status': 'SNAPSHOT', 'name': name,
                                               'files': [f"{a['repository']}:{a['path']}", f"{b['repository']}:{b['path']}"]})
                    continue
                pair = (name, *sorted((f"{a['repository']}:{a['path']}:{a['signature_sha256']}",
                                       f"{b['repository']}:{b['path']}:{b['signature_sha256']}")))
                if pair in allowlist:
                    report['findings'].append({'check': 'D1', 'blocking': False, 'status': 'ALLOWED', 'name': name,
                                               'files': [f"{a['repository']}:{a['path']}", f"{b['repository']}:{b['path']}"]})
                    continue
                drift.append(pair)
        for pair in drift:
            report['findings'].append({'check': 'D1', 'blocking': True, 'status': 'DRIFT', 'name': name,
                                       'occurrences': list(pair[1:])})


def snapshot_of(fed, history, older, newer):
    """True when older's file bytes occur in the Git history of newer's path (an older pinned copy)."""
    key = (newer['key'], newer['path'])
    if key not in history:
        info = fed.repos[newer['key']]
        out = run_git(info['path'], 'log', '--format=%H', '--all', '--', newer['path'], check=False).stdout.decode().split()
        blobs = set()
        for commit in out:
            r = run_git(info['path'], 'rev-parse', '-q', '--verify', f'{commit}:{newer["path"]}', check=False)
            if r.returncode == 0:
                blobs.add(r.stdout.decode().strip())
        history[key] = blobs
    return older['blob'] in history[key]


def load_allowlist(path):
    if not path or not Path(path).exists():
        return []
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('schema') != SCHEMA or not isinstance(data.get('allowed'), list):
        raise ValueError('invalid drift allowlist')
    out = []
    for row in data['allowed']:
        if set(row) != {'name', 'occurrences', 'reason'} or len(row['occurrences']) != 2 or len(row['reason'].strip()) < 20:
            raise ValueError('invalid drift allowlist entry')
        out.append((row['name'], *sorted(row['occurrences'])))
    return out


def main(argv=None):
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--root', default='https://github.com/d6g8k5htny-coder/Universal-Law-Workspace')
    parser.add_argument('--local', action='append', default=[], help='URL=PATH: reuse an existing clone')
    parser.add_argument('--cache', default=str(Path.home() / '.cache/cbz-drift'))
    parser.add_argument('--offline', action='store_true', help='never clone or fetch')
    parser.add_argument('--owner', help='federation owner (default: the root URL owner)')
    parser.add_argument('--allowlist', default=str(here / 'drift_allowlist.json'))
    parser.add_argument('--require-current-root-pins', action='store_true')
    parser.add_argument('--head', action='append', default=[],
                        help='URL=REV: check this revision instead of the default-branch head (a PR tested merge)')
    parser.add_argument('--emit-repos', help='write NAME=PATH@COMMIT for every crawled repository, for G1 and G6')
    parser.add_argument('--json')
    args = parser.parse_args(argv)
    local = dict(item.split('=', 1) for item in args.local)
    heads = dict(item.split('=', 1) for item in args.head)
    try:
        fed = Federation(args.cache, local, args.offline, args.owner, heads)
        report = crawl(args.root, fed, args.require_current_root_pins, load_allowlist(args.allowlist))
    except (ValueError, RuntimeError, OSError) as exc:
        print('DRIFT_ERROR: ' + str(exc), file=sys.stderr)
        return 2
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + '\n', encoding='utf-8')
    if args.emit_repos:
        lines = []
        for info in fed.repos.values():
            if info['path'] is not None and info['head'] and not info['error']:
                lines.append(f"{info['name']}={info['path']}@{info['head']}")
        Path(args.emit_repos).write_text(''.join(sorted(set(line + '\n' for line in lines))), encoding='utf-8')
    for repo in report['repositories']:
        print(f"crawled {repo['url']} @ {repo['head']}")
    for f in report['findings']:
        if f['blocking']:
            print('BLOCKING ' + json.dumps(f, sort_keys=True, default=str)[:600])
    print('DRIFT ' + report['verdict'] + ' ' + json.dumps(report['summary'], sort_keys=True))
    return 0 if report['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
