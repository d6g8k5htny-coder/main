#!/usr/bin/env python3
"""Verify the comment-math census against local Git clones.

Read-only. Every carrier in CENSUS.json names a repository, a pinned commit,
a path and a blob. This script checks each one with `git rev-parse`, checks
that each custody class agrees with blob membership in the pinned main and
Math- trees, recomputes the summary and gap lists, and renders CENSUS.md.

    python3 -I verify_census.py --repo main=PATH --repo Math-=PATH [--repo NAME=PATH ...] [--check]

Without --check it writes CENSUS.md next to this file. With --check it
renders to memory and fails on any difference from the committed CENSUS.md.
Carriers in a repository with no --repo clone are reported as unchecked and
make the run fail unless --allow-missing-repos is given.

    python3 -I verify_census.py --live [--token-env GITHUB_TOKEN]

re-fetches every cited comment through the GitHub REST API and reports body
SHA-256 drift. It is report-only; comments may be edited after the cut.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CENSUS = os.path.join(HERE, 'CENSUS.json')
REPORT = os.path.join(HERE, 'CENSUS.md')

CUSTODY = ('COMMITTED_ON_MAIN', 'COMMITTED_ON_BRANCH_ONLY', 'DRIVE_OR_DROPBOX_ONLY_CITED', 'INLINE_ONLY', 'UNKNOWN')
IMPLEMENTATION = ('FOUND', 'CITED_NOT_FOUND', 'NONE_NEEDED', 'NONE_FOUND')
FAMILIES = ('Anthropic', 'OpenAI', 'Grok', 'Google', 'Owner', 'Unknown')
MAIN_TREES = ('main', 'Math-')
HEX40 = set('0123456789abcdef')


def is_hex40(s):
    return isinstance(s, str) and len(s) == 40 and set(s) <= HEX40


def git(path, *args, binary=False):
    r = subprocess.run(['git', '-C', path, *args], capture_output=True)
    if r.returncode != 0:
        return None
    return r.stdout if binary else r.stdout.decode('utf-8', 'replace').strip()


class Checker:
    def __init__(self, repos, allow_missing):
        self.repos = repos
        self.allow_missing = allow_missing
        self.errors = []
        self.unchecked = []
        self.blob_sets = {}

    def fail(self, where, msg):
        self.errors.append(f'{where}: {msg}')

    def tree_blobs(self, repo, commit):
        key = (repo, commit)
        if key not in self.blob_sets:
            out = git(self.repos[repo], 'ls-tree', '-r', '--full-tree', commit)
            if out is None:
                self.blob_sets[key] = None
            else:
                self.blob_sets[key] = {line.split()[2] for line in out.splitlines() if line.split()[1] == 'blob'}
        return self.blob_sets[key]

    def carrier(self, where, c):
        repo, commit, path, blob = c.get('repo'), c.get('commit'), c.get('path'), c.get('blob')
        if not (is_hex40(commit) and is_hex40(blob) and isinstance(path, str) and path):
            self.fail(where, f'malformed carrier {c!r}')
            return False
        if repo not in self.repos:
            self.unchecked.append(f'{where}: {repo}:{commit[:8]}:{path}')
            return False
        got = git(self.repos[repo], 'rev-parse', '--verify', '--quiet', f'{commit}:{path}')
        if got != blob:
            self.fail(where, f'{repo}:{commit[:8]}:{path} is {got or "absent"}, expected {blob[:8]}')
            return False
        if c.get('sha256'):
            data = git(self.repos[repo], 'cat-file', 'blob', blob, binary=True)
            if data is None or hashlib.sha256(data).hexdigest() != c['sha256']:
                self.fail(where, f'{repo}:{blob[:8]} SHA-256 differs from the recorded value')
                return False
        return True

    def record(self, where, r):
        repo, commit, path, needle = r.get('repo'), r.get('commit'), r.get('path'), r.get('needle')
        if not (is_hex40(commit) and path and needle):
            self.fail(where, f'malformed record mention {r!r}')
            return
        if repo not in self.repos:
            self.unchecked.append(f'{where}: record {repo}:{commit[:8]}:{path}')
            return
        data = git(self.repos[repo], 'show', f'{commit}:{path}')
        if data is None:
            self.fail(where, f'record {repo}:{commit[:8]}:{path} is absent')
        elif needle not in data:
            self.fail(where, f'record {repo}:{commit[:8]}:{path} does not contain {needle!r}')


def on_main(chk, cuts, carriers):
    found = False
    for c in carriers:
        for tree in MAIN_TREES:
            if tree in chk.repos and tree in cuts:
                blobs = chk.tree_blobs(tree, cuts[tree])
                if blobs is not None and c.get('blob') in blobs:
                    found = True
    return found


def verify(census, chk):
    cuts = census['source_cuts']
    for tree in MAIN_TREES:
        if tree not in chk.repos:
            continue
        if not is_hex40(cuts.get(tree, '')) or git(chk.repos[tree], 'cat-file', '-t', cuts[tree]) != 'commit':
            chk.fail('source_cuts', f'{tree} cut {cuts.get(tree)} is not a commit in the clone')
        elif chk.tree_blobs(tree, cuts[tree]) is None:
            chk.fail('source_cuts', f'cannot list the {tree} tree at {cuts[tree][:8]}')
    keys = set()
    for a in census['artifacts']:
        where = a.get('key', '?')
        if where in keys:
            chk.fail(where, 'duplicate key')
        keys.add(where)
        if not a.get('versions'):
            chk.fail(where, 'no comment version')
        for v in a.get('versions', []):
            if not (isinstance(v.get('body_sha256'), str) and len(v['body_sha256']) == 64):
                chk.fail(where, f'version {v.get("comment")} lacks a body SHA-256')
        if a.get('family') not in FAMILIES:
            chk.fail(where, f'unknown family {a.get("family")}')
        cus, imp = a['custody'], a['implementation']
        if cus['class'] not in CUSTODY:
            chk.fail(where, f'unknown custody class {cus["class"]}')
        if imp['class'] not in IMPLEMENTATION:
            chk.fail(where, f'unknown implementation class {imp["class"]}')
        ok_c = [c for c in cus.get('carriers', []) if chk.carrier(where + ' custody', c)]
        for c in imp.get('carriers', []):
            chk.carrier(where + ' implementation', c)
        for r in a.get('records', []):
            chk.record(where, r)
        main_ok = all(t in chk.repos for t in MAIN_TREES)
        if cus['class'] == 'COMMITTED_ON_MAIN':
            if not cus.get('carriers'):
                chk.fail(where, 'COMMITTED_ON_MAIN without a carrier')
            elif main_ok and not on_main(chk, cuts, ok_c):
                chk.fail(where, 'COMMITTED_ON_MAIN but no carrier blob is in the main or Math- tree at the cut')
        elif cus['class'] == 'COMMITTED_ON_BRANCH_ONLY':
            if not cus.get('carriers'):
                chk.fail(where, 'COMMITTED_ON_BRANCH_ONLY without a carrier')
            elif main_ok and on_main(chk, cuts, cus['carriers']):
                chk.fail(where, 'COMMITTED_ON_BRANCH_ONLY but a carrier blob is in the main or Math- tree at the cut')
        elif cus.get('carriers'):
            chk.fail(where, f'{cus["class"]} must not list byte carriers')
        if imp['class'] == 'FOUND' and not imp.get('carriers'):
            chk.fail(where, 'implementation FOUND without a carrier')
        if imp['class'] != 'FOUND' and imp.get('carriers'):
            chk.fail(where, f'implementation {imp["class"]} must not list carriers')
    probe_uncommitted(census, chk)


def probe_uncommitted(census, chk):
    """Re-run the inline probe: no row outside COMMITTED_ON_MAIN may have two of its probe lines in one file at a cut."""
    rows = [a for a in census['artifacts'] if a['custody']['class'] != 'COMMITTED_ON_MAIN']
    for a in rows:
        if not (a.get('probe') or {}).get('lines'):
            chk.fail(a['key'], f'{a["custody"]["class"]} row has no probe lines')
    rows = [a for a in rows if (a.get('probe') or {}).get('lines')]
    pats = sorted({p for a in rows for p in a['probe']['lines']})
    if not pats:
        return
    import tempfile
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.txt', encoding='utf-8') as f:
        f.write('\n'.join(pats) + '\n')
        name = f.name
    try:
        where = {}
        for tree in MAIN_TREES:
            if tree not in chk.repos:
                continue
            out = subprocess.run(['git', '-C', chk.repos[tree], 'grep', '-F', '-I', '-f', name, census['source_cuts'][tree]], capture_output=True).stdout.decode('utf-8', 'replace')
            for line in out.splitlines():
                _, path, text = line.split(':', 2)
                for p in pats:
                    if p in text:
                        where.setdefault(p, set()).add((tree, path))
    finally:
        os.unlink(name)
    for a in rows:
        per = {}
        for p in a['probe']['lines']:
            for loc in where.get(p, ()):
                per.setdefault(loc, set()).add(p)
        worst = max((len(v) for v in per.values()), default=0)
        if worst >= 2:
            chk.fail(a['key'], f'{a["custody"]["class"]} but {worst} probe lines appear in one committed file at the cut')
        if worst != a['probe'].get('max_lines_in_one_file', 0):
            chk.fail(a['key'], f'probe recorded {a["probe"].get("max_lines_in_one_file")} lines in one file, re-run finds {worst}')


def summarize(census):
    arts = census['artifacts']
    count = lambda f: {k: sum(1 for a in arts if f(a) == k) for k in sorted({f(a) for a in arts})}
    gaps = {
        'A_inline_only': [a['key'] for a in arts if a['custody']['class'] == 'INLINE_ONLY'],
        'B_branch_only': [a['key'] for a in arts if a['custody']['class'] == 'COMMITTED_ON_BRANCH_ONLY'],
        'C_drive_or_dropbox_only': [a['key'] for a in arts if a['custody']['class'] == 'DRIVE_OR_DROPBOX_ONLY_CITED'],
        'D_computation_without_implementation': [a['key'] for a in arts if a['implementation']['class'] in ('CITED_NOT_FOUND', 'NONE_FOUND')],
        'E_not_in_any_record': [a['key'] for a in arts if not a.get('records')],
        'U_custody_unknown': [a['key'] for a in arts if a['custody']['class'] == 'UNKNOWN'],
    }
    return {
        'artifacts': len(arts),
        'comment_versions': sum(len(a['versions']) for a in arts),
        'custody': count(lambda a: a['custody']['class']),
        'implementation': count(lambda a: a['implementation']['class']),
        'kind': count(lambda a: a['kind']),
        'family': count(lambda a: a['family']),
        'gaps': gaps,
    }


def link(comment, url):
    return f'[{comment}]({url})' if url else comment


def render(census, summ):
    out = []
    w = out.append
    w('# Comment-delivered mathematics: census')
    w('')
    w('Generated by `verify_census.py` from `CENSUS.json`. Do not edit by hand.')
    w('')
    w(census['banner'])
    w('')
    w('## Source cuts')
    w('')
    for k, v in census['source_cuts'].items():
        w(f'- {k}: `{v}`')
    w('')
    w('## Corpus and coverage')
    w('')
    for line in census['coverage']:
        w(f'- {line}')
    w('')
    w('## Totals')
    w('')
    w(f'- Artifacts: **{summ["artifacts"]}**, from {summ["comment_versions"]} comment versions.')
    for name in ('custody', 'implementation', 'kind', 'family'):
        w(f'- {name.capitalize()}: ' + ', '.join(f'{k} {v}' for k, v in summ[name].items()) + '.')
    w('')
    w('## Gap lists')
    w('')
    titles = {
        'A_inline_only': 'A. Inline only: no committed byte copy found anywhere',
        'B_branch_only': 'B. Committed only on an unmerged branch or PR head',
        'C_drive_or_dropbox_only': 'C. Only Drive or Dropbox locations cited',
        'D_computation_without_implementation': 'D. Computational content, but no implementation found',
        'E_not_in_any_record': 'E. Not mentioned by any retro, ledger, CBZ or inventory record',
        'U_custody_unknown': 'U. Custody not determined',
    }
    by_key = {a['key']: a for a in census['artifacts']}
    for g, keys in summ['gaps'].items():
        w(f'### {titles[g]} ({len(keys)})')
        w('')
        for k in keys:
            a = by_key[k]
            v = a['versions'][0]
            w(f'- `{k}`: {link(v["comment"], v.get("url"))}, {v["created"][:10]}, {a["family"]}. {a["statement"]}')
        if not keys:
            w('- none')
        w('')
    w('## All artifacts')
    w('')
    w('| Key | First comment | Family | Kind | Custody | Implementation | Records |')
    w('|---|---|---|---|---|---|---|')
    for a in census['artifacts']:
        v = a['versions'][0]
        w(f'| `{a["key"]}` | {link(v["comment"], v.get("url"))} | {a["family"]} | {a["kind"]} | {a["custody"]["class"]} | {a["implementation"]["class"]} | {len(a.get("records", []))} |')
    w('')
    w('## Limits')
    w('')
    for line in census['limits']:
        w(f'- {line}')
    w('')
    return '\n'.join(out)


def live(census, token_env):
    token = os.environ.get(token_env, '')
    drift = 0
    for a in census['artifacts']:
        for v in a['versions']:
            req = urllib.request.Request('https://api.github.com' + v['api_path'])
            req.add_header('Accept', 'application/vnd.github+json')
            if token:
                req.add_header('Authorization', f'Bearer {token}')
            try:
                body = json.load(urllib.request.urlopen(req, timeout=30)).get('body') or ''
                got = hashlib.sha256(body.encode('utf-8')).hexdigest()
                state = 'same' if got == v['body_sha256'] else 'CHANGED'
            except Exception as e:
                state = f'UNREADABLE ({type(e).__name__})'
            if state != 'same':
                drift += 1
                print(f'{v["comment"]}: {state}')
    print(f'live: {drift} comment versions differ from the recorded bodies')
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', action='append', default=[], help='NAME=PATH of a local clone')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--allow-missing-repos', action='store_true')
    ap.add_argument('--live', action='store_true')
    ap.add_argument('--token-env', default='GITHUB_TOKEN')
    ns = ap.parse_args()
    census = json.load(open(CENSUS, encoding='utf-8'))
    if ns.live:
        return live(census, ns.token_env)
    repos = dict(r.split('=', 1) for r in ns.repo)
    for tree in MAIN_TREES:
        if tree not in repos:
            print(f'error: --repo {tree}=PATH is required', file=sys.stderr)
            return 2
    chk = Checker(repos, ns.allow_missing_repos)
    verify(census, chk)
    summ = summarize(census)
    if census.get('summary') != summ:
        chk.fail('summary', 'recorded summary differs from the recomputed one')
    text = render(census, summ)
    for u in chk.unchecked:
        print(f'unchecked (no clone): {u}')
    if chk.unchecked and not ns.allow_missing_repos:
        chk.fail('repos', f'{len(chk.unchecked)} items need clones that were not supplied')
    if ns.check:
        try:
            committed = open(REPORT, encoding='utf-8').read()
        except OSError:
            committed = None
        if committed != text:
            chk.fail('CENSUS.md', 'differs from the rendering of CENSUS.json')
    else:
        open(REPORT, 'w', encoding='utf-8').write(text)
    for e in chk.errors:
        print(f'FAIL {e}')
    n_car = sum(len(a['custody'].get('carriers', [])) + len(a['implementation'].get('carriers', [])) for a in census['artifacts'])
    print(f'artifacts {summ["artifacts"]}, carriers {n_car}, record mentions {sum(len(a.get("records", [])) for a in census["artifacts"])}, unchecked {len(chk.unchecked)}')
    if chk.errors:
        return 1
    print('check: OK' if ns.check else 'written: CENSUS.md')
    return 0


if __name__ == '__main__':
    sys.exit(main())
