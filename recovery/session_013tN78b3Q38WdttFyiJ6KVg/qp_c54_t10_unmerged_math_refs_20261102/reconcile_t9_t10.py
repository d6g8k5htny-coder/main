#!/usr/bin/env python3
"""Reconcile tranche-9 and tranche-10 refs with the state after the 2 November cut: the PR-level dormant record (from
dormant_index.py), whether the bound commit is now in Math- main's history, and whether the ref still points at the bound commit.
Read-only. Usage: python3 -I -B reconcile_t9_t10.py MATH_CLONE NEW_MAIN T9_FACTS_DIR T10_FACTS_DIR DORMANT_INDEX
PULL_HEADS_NOW BRANCH_HEADS_NOW OUT_JSON   (the *_NOW files are `git ls-remote` output taken at reconciliation time)"""
import collections, glob, json, os, subprocess, sys
clone, new_main, t9_dir, t10_dir, dorm_p, pulls_p, branches_p, out = sys.argv[1:]
dorm = json.load(open(dorm_p))
pnow = {l.split()[1]: l.split()[0] for l in open(pulls_p)}
bnow = {l.split()[1]: l.split()[0] for l in open(branches_p)}
new_main = subprocess.run(['git', '-C', clone, 'rev-parse', new_main], capture_output=True, text=True, check=True).stdout.strip()


def in_main(c):
    return subprocess.run(['git', '-C', clone, 'merge-base', '--is-ancestor', c, new_main]).returncode == 0


refs = set()
for f in glob.glob(os.path.join(t9_dir, 'pbatch*.json')):
    for o in json.load(open(f))['objects']:
        refs.add(('T9', f'refs/pull/{o["pr"]}/head', o['pr'], o['commit']))
for f in glob.glob(os.path.join(t10_dir, 'qbatch*.json')):
    for o in json.load(open(f))['objects']:
        refs.add(('T10' if json.load(open(f))['batch'] <= 22 else 'T10-supplement', o['ref'], o['pr'], o['commit']))
rows = []
for tr, ref, pr, commit in sorted(refs):
    d = dorm.get(f'Math-#{pr}') if pr else None
    now = pnow if ref.startswith('refs/pull/') else bnow
    rows.append({'tranche': tr, 'ref': ref, 'pr': pr, 'commit': commit, 'q_math': d['ids'] if d else [],
                 'pr_level_record': bool(d and (d['ids'] or d['record_comments'])),
                 'record_comments': d['record_comments'] if d else [], 'ref_still_at_commit': now.get(ref) == commit,
                 'ref_deleted': ref not in now, 'commit_in_math_main_now': in_main(commit)})
with open(out, 'x') as f:
    json.dump({'schema': 'claude54.t9-t10-reconciliation.v1', 'math_main_now': new_main, 'rows': rows}, f, indent=1)
c = collections.Counter((r['tranche'], 'PR-level record' if r['pr_level_record'] else 'no PR-level record',
                         'merged' if r['commit_in_math_main_now'] else ('deleted' if r['ref_deleted'] else 'ref kept')) for r in rows)
for k, v in sorted(c.items()):
    print(k, v)
