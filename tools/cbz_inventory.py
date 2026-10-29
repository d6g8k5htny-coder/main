#!/usr/bin/env python3
"""CBZ-0 Phase 1 federation inventory.

Reads local clones of the in-scope repositories and the GitHub REST API
(through the `gh` CLI), and writes a machine-readable inventory plus a
readable summary:

  * default-branch head, remote branches, tags, stashes, worktrees
  * open pull requests (draft state, head, last update), with a Layer A scan
    of the Lean files each PR changes
  * check-run conclusions on the default-branch head
  * Layer A scan of the default-branch tree (tools/cbz_layer_a_scan.py)
  * which canonical status surfaces exist
  * which CBZ-0 gates can be run in this environment (the tooling-gap list)

Nothing is left as an unknown: anything that cannot be determined is
recorded as {"unavailable": "<reason>"}.  The tool certifies nothing by
itself; it reports observations with the exact commits they were read at.

Usage:
  cbz_inventory.py --owner OWNER --repo NAME=CLONE_PATH [...] \
      --out-json PATH --out-md PATH [--no-pr-scan]
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('cbz_layer_a_scan', HERE / 'cbz_layer_a_scan.py')
scanner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scanner)

SURFACES = ('STATUS.md', 'PROOF_INDEX.md', 'GOVERNANCE.md', 'TOOLING.md', 'BASELINE.md',
            'DECISIONS.md', 'INVENTORY.md', 'AGENTS.md')


def run(cmd, cwd=None, check=True):
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}: {proc.stderr.strip()[:300]}")
    return proc


def gh_json(path):
    proc = run(['gh', 'api', '--paginate', '--slurp', path], check=False)
    if proc.returncode != 0:
        return {'unavailable': proc.stderr.strip()[:300] or f'gh exit {proc.returncode}'}
    pages = json.loads(proc.stdout)
    if pages and isinstance(pages[0], list):
        return [item for page in pages for item in page]
    return pages


def git(clone, *args):
    return run(['git', '-C', str(clone), *args]).stdout


def scan_tree(clone, commit):
    tmp = tempfile.mkdtemp(prefix='cbz-tree-')
    try:
        archive = subprocess.run(['git', '-C', str(clone), 'archive', commit], capture_output=True, check=True)
        subprocess.run(['tar', '-x', '-C', tmp], input=archive.stdout, check=True)
        result = scanner.scan_roots([tmp])
        for item in result['hits'] + result['errors']:
            item['path'] = os.path.relpath(item['path'], tmp)
        return result
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def scan_blobs(clone, commit, paths):
    hits, errors = [], []
    for path in paths:
        proc = subprocess.run(['git', '-C', str(clone), 'show', f'{commit}:{path}'], capture_output=True)
        if proc.returncode != 0:
            errors.append({'path': path, 'error': 'not present at head (deleted by the PR)'})
            continue
        try:
            text = proc.stdout.decode('utf-8')
        except UnicodeDecodeError as exc:
            errors.append({'path': path, 'error': f'UnicodeDecodeError: {exc}'})
            continue
        hits += [{'path': path, 'line': line, 'token': tok} for tok, line in scanner.scan_text(text)]
    return {'files_scanned': len(paths), 'hits': hits, 'errors': [e for e in errors if 'deleted' not in e['error']],
            'deleted': [e['path'] for e in errors if 'deleted' in e['error']]}


def inventory_repo(owner, name, clone, pr_scan=True):
    repo = {'name': name, 'clone': str(clone)}
    run(['git', '-C', str(clone), 'fetch', '-q', '--prune', 'origin',
         '+refs/heads/*:refs/remotes/origin/*', '+refs/tags/*:refs/tags/*'], check=False)
    head = git(clone, 'rev-parse', 'origin/main').strip()
    repo['default_branch'] = 'main'
    repo['default_head'] = head
    branches = [b.strip() for b in git(clone, 'for-each-ref', '--format=%(refname:short) %(committerdate:iso-strict)',
                                       'refs/remotes/origin').splitlines() if b.strip()]
    repo['remote_branches'] = [{'name': b.split(' ')[0], 'last_commit': b.split(' ')[1]}
                               for b in branches if not b.startswith('origin/HEAD') and b.split(' ')[0] != 'origin']
    merged = set(git(clone, 'branch', '-r', '--merged', 'origin/main').split())
    for b in repo['remote_branches']:
        b['merged_into_default'] = b['name'] in merged
    repo['tags'] = git(clone, 'tag', '--list').split()
    repo['stashes'] = len(git(clone, 'stash', 'list').splitlines())
    repo['worktrees'] = [w.split()[0] for w in git(clone, 'worktree', 'list').splitlines()]
    repo['working_tree_dirty_entries'] = len(git(clone, 'status', '--porcelain').splitlines())
    names = set(git(clone, 'ls-tree', '--name-only', head).split('\n'))
    repo['status_surfaces'] = {s: (s in names) for s in SURFACES}
    all_files = git(clone, 'ls-tree', '-r', '--name-only', head).splitlines()
    repo['file_count'] = len(all_files)
    repo['lean_file_count'] = sum(1 for f in all_files if f.endswith('.lean'))
    repo['workflow_count'] = sum(1 for f in all_files if f.startswith('.github/workflows/'))
    repo['layer_a_default_tree'] = scan_tree(clone, head)

    runs = gh_json(f'repos/{owner}/{name}/commits/{head}/check-runs?per_page=100')
    if isinstance(runs, dict) and 'unavailable' in runs:
        repo['default_head_checks'] = runs
    else:
        check_runs = [r for page in runs for r in page.get('check_runs', [])] if runs and 'check_runs' in runs[0] else []
        repo['default_head_checks'] = sorted(({'name': r['name'], 'status': r['status'], 'conclusion': r['conclusion']}
                                              for r in check_runs), key=lambda r: r['name'])

    prs = gh_json(f'repos/{owner}/{name}/pulls?state=open&per_page=100')
    if isinstance(prs, dict) and 'unavailable' in prs:
        repo['open_prs'] = prs
        return repo
    out = []
    for pr in sorted(prs, key=lambda p: p['number']):
        entry = {'number': pr['number'], 'title': pr['title'], 'draft': pr['draft'],
                 'head_ref': pr['head']['ref'], 'head_sha': pr['head']['sha'], 'base_ref': pr['base']['ref'],
                 'created_at': pr['created_at'], 'updated_at': pr['updated_at']}
        if pr_scan:
            fetch = run(['git', '-C', str(clone), 'fetch', '-q', 'origin', f"pull/{pr['number']}/head"], check=False)
            if fetch.returncode != 0:
                entry['layer_a_changed_lean'] = {'unavailable': fetch.stderr.strip()[:200]}
            else:
                base = run(['git', '-C', str(clone), 'merge-base', head, pr['head']['sha']], check=False).stdout.strip()
                changed = [p for p in git(clone, 'diff', '--name-only', base or head, pr['head']['sha']).splitlines()
                           if p.endswith('.lean')] if base else []
                entry['changed_lean_files'] = len(changed)
                entry['layer_a_changed_lean'] = scan_blobs(clone, pr['head']['sha'], changed)
        out.append(entry)
    repo['open_prs'] = out
    return repo


def gate_matrix():
    have = lambda exe: shutil.which(exe) is not None
    lean = have('lean') and have('lake')
    return [
        {'gate': 'G0', 'check': 'Layer A scanner mutation self-test', 'runnable_here': True,
         'tool': 'tools/cbz_scanner_mutation_check.py'},
        {'gate': 'G1', 'check': 'Layer A static audit', 'runnable_here': True, 'tool': 'tools/cbz_layer_a_scan.py'},
        {'gate': 'G2', 'check': '#print axioms on all declarations', 'runnable_here': lean,
         'gap': None if lean else 'no Lean toolchain (lean/lake) in this environment; hosted Math- workflows only'},
        {'gate': 'G3', 'check': 'Lean build', 'runnable_here': lean,
         'gap': None if lean else 'no Lean toolchain in this environment; hosted Math- workflows build per-package'},
        {'gate': 'G4', 'check': 'kernel re-check', 'runnable_here': lean,
         'gap': None if lean else 'no Lean toolchain; no federation-wide kernel re-check tool exists'},
        {'gate': 'G5', 'check': 'full regression suite', 'runnable_here': True,
         'tool': 'python -B -S -m unittest discover -s tests -p "test_*.py" (main); per-repo suites differ'},
        {'gate': 'G6', 'check': 'mutation threshold', 'runnable_here': False,
         'gap': 'no declared federation mutation threshold or repo-wide mutation runner'},
        {'gate': 'G7', 'check': 'cross-repo pin verification', 'runnable_here': False,
         'gap': 'no tool verifies that cross-repo commit pins resolve and agree'},
        {'gate': 'G8', 'check': 'status-surface truth check', 'runnable_here': False,
         'gap': 'no tool compares STATUS.md/PROOF_INDEX.md claims with observed state'},
        {'gate': 'G9', 'check': 'cold-start test', 'runnable_here': False,
         'gap': 'requires canonical STATUS.md handoff surface (Phase 4)'},
        {'gate': 'G10', 'check': 'git hygiene', 'runnable_here': True, 'tool': 'this inventory (branches/PRs/worktrees)'},
        {'gate': 'G11', 'check': 'tooling hygiene', 'runnable_here': True,
         'tool': 'GOVERNANCE.md tool registry (root, PR 685) and each tool own tests'},
    ]


def render_md(inv):
    lines = ['# CBZ-0 Phase 1 inventory', '',
             f"Generated {inv['generated_at']} by `tools/cbz_inventory.py`. Observations only; this file certifies nothing.",
             'Regenerate with the command in this directory'"'"'s README.md.', '',
             '## Repositories', '',
             '| Repo | Default head | Files | Lean | Workflows | Remote branches (unmerged) | Tags | Open PRs (draft) | Layer A hits (default tree) |',
             '|---|---|---:|---:|---:|---|---:|---|---|']
    for r in inv['repos']:
        if 'unavailable' in r:
            lines.append(f"| {r['name']} | unavailable: {r['unavailable']} | | | | | | | |")
            continue
        unmerged = sum(1 for b in r['remote_branches'] if not b['merged_into_default'])
        prs = r['open_prs']
        pr_txt = (f"{len(prs)} ({sum(1 for p in prs if p['draft'])})" if isinstance(prs, list)
                  else f"unavailable: {prs['unavailable'][:60]}")
        la = r['layer_a_default_tree']
        lines.append(f"| {r['name']} | `{r['default_head'][:8]}` | {r['file_count']} | {r['lean_file_count']} | "
                     f"{r['workflow_count']} | {len(r['remote_branches'])} ({unmerged}) | {len(r['tags'])} | {pr_txt} | "
                     f"{len(la['hits'])} hits, {len(la['errors'])} errors, {la['files_scanned']} Lean files |")
    lines += ['', '## Default-branch checks', '']
    for r in inv['repos']:
        if 'unavailable' in r:
            continue
        checks = r['default_head_checks']
        if isinstance(checks, dict):
            lines.append(f"- **{r['name']}**: unavailable: {checks['unavailable'][:120]}")
        else:
            bad = [c for c in checks if c['conclusion'] not in ('success', 'skipped', 'neutral')]
            lines.append(f"- **{r['name']}** `{r['default_head'][:8]}`: {len(checks)} check runs, "
                         f"{len(bad)} not success/skipped" + (': ' + ', '.join(f"{c['name']}={c['conclusion'] or c['status']}" for c in bad) if bad else ''))
    lines += ['', '## Open pull requests', '',
              '| Repo | PR | Draft | Updated | Changed Lean files | Layer A hits in changed Lean | Title |',
              '|---|---|---|---|---:|---|---|']
    for r in inv['repos']:
        if 'unavailable' in r or not isinstance(r['open_prs'], list):
            continue
        for p in r['open_prs']:
            la = p.get('layer_a_changed_lean', {})
            la_txt = (f"unavailable: {la['unavailable'][:40]}" if 'unavailable' in la
                      else (', '.join(f"{h['path']}:{h['line']} {h['token']}" for h in la.get('hits', [])) or '0'))
            lines.append(f"| {r['name']} | #{p['number']} | {'yes' if p['draft'] else 'no'} | {p['updated_at'][:10]} | "
                         f"{p.get('changed_lean_files', '-')} | {la_txt} | {p['title'][:70].replace('|', '/')} |")
    lines += ['', '## Status surfaces present on the default branch', '',
              '| Repo | ' + ' | '.join(SURFACES) + ' |', '|---|' + '---|' * len(SURFACES)]
    for r in inv['repos']:
        if 'unavailable' in r:
            continue
        lines.append(f"| {r['name']} | " + ' | '.join('yes' if r['status_surfaces'][s] else 'no' for s in SURFACES) + ' |')
    lines += ['', '## Local hygiene (this inventory host)', '']
    for r in inv['repos']:
        if 'unavailable' in r:
            continue
        lines.append(f"- **{r['name']}**: worktrees {len(r['worktrees'])}, stashes {r['stashes']}, "
                     f"dirty entries {r['working_tree_dirty_entries']}")
    lines += ['', '## Gate runnability and tooling gaps', '', '| Gate | Check | Runnable here | Tool or gap |', '|---|---|---|---|']
    for g in inv['gates']:
        lines.append(f"| {g['gate']} | {g['check']} | {'yes' if g['runnable_here'] else 'no'} | {g.get('tool') or g.get('gap')} |")
    lines += ['', '## Inaccessible repositories', '']
    lines += [f"- {u['name']}: {u['reason']}" for u in inv['inaccessible']] or ['- none']
    return '\n'.join(lines) + '\n'


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--owner', required=True)
    ap.add_argument('--repo', action='append', default=[])
    ap.add_argument('--inaccessible', action='append', default=[], help='NAME=REASON')
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--out-md', required=True)
    ap.add_argument('--no-pr-scan', action='store_true')
    a = ap.parse_args(argv[1:])
    inv = {'schema': 'cbz-inventory/v1', 'generated_at': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
           'owner': a.owner, 'repos': [], 'gates': gate_matrix(),
           'inaccessible': [{'name': x.split('=', 1)[0], 'reason': x.split('=', 1)[1]} for x in a.inaccessible]}
    for spec in a.repo:
        name, path = spec.split('=', 1)
        try:
            inv['repos'].append(inventory_repo(a.owner, name, Path(path), pr_scan=not a.no_pr_scan))
        except Exception as exc:  # recorded, never dropped
            inv['repos'].append({'name': name, 'unavailable': f'{type(exc).__name__}: {exc}'[:300]})
    Path(a.out_json).write_text(json.dumps(inv, indent=1, sort_keys=True) + '\n')
    Path(a.out_md).write_text(render_md(inv))
    print(json.dumps({'repos': len(inv['repos']),
                      'open_prs': sum(len(r['open_prs']) for r in inv['repos'] if isinstance(r.get('open_prs'), list)),
                      'layer_a_default_hits': sum(len(r['layer_a_default_tree']['hits']) for r in inv['repos']
                                                  if 'layer_a_default_tree' in r)}))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
