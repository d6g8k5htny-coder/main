#!/usr/bin/env python3
"""Reconcile tranche-11c refs (reconcile_t11b.py; pre_rewrite computed from git: no merge base with main d2b0216e) with the state after the cut: the PR-level dormant record (Q-MAIN, from dormant_index.py), whether
the bound commit is now in main's history, whether the ref still points at it, and whether the ref is a PR head (retained by
GitHub after branch deletion) or a branch tip only (lost with the branch unless tagged). Read-only.
Usage: python3 -I -B reconcile_t11c.py MAIN_CLONE NEW_MAIN FACTS_DIR RECORDS DORMANT_INDEX PULL_HEADS_NOW BRANCH_HEADS_NOW OUT"""
import collections, glob, json, os, subprocess, sys
clone, new_main, facts_dir, recs_p, dorm_p, pulls_p, branches_p, out = sys.argv[1:]
dorm = json.load(open(dorm_p))
pnow = {l.split()[1]: l.split()[0] for l in open(pulls_p)}
bnow = {l.split()[1]: l.split()[0] for l in open(branches_p)}
new_main = subprocess.run(['git', '-C', clone, 'rev-parse', new_main], capture_output=True, text=True, check=True).stdout.strip()
refs = {}
for f in glob.glob(os.path.join(facts_dir, 'qbatch*.json')):
    for o in json.load(open(f))['objects']:
        mb = subprocess.run(['git', '-C', clone, 'merge-base', o['commit'], 'd2b0216e15d545c47f25b6a9265dd3298971e9fe'], capture_output=True)
        refs[o['ref']] = (o['pr'], o['commit'], mb.returncode != 0)
recs = json.load(open(recs_p))
per_ref = collections.Counter(r['source_binding']['ref'] for r in recs['records'])
rows = []
for ref, (pr, commit, orphan) in sorted(refs.items()):
    d = dorm.get(f'main#{pr}') if pr else None
    now = pnow if ref.startswith('refs/pull/') else bnow
    rows.append({'ref': ref, 'pr': pr, 'commit': commit, 'pre_rewrite': orphan, 'kind': 'PR head' if pr else 'branch only',
                 'q_main': d['ids'] if d else [], 'ref_still_at_commit': now.get(ref) == commit, 'ref_deleted': ref not in now,
                 'commit_in_main_now': subprocess.run(['git', '-C', clone, 'merge-base', '--is-ancestor', commit, new_main]).returncode == 0,
                 'records': per_ref.get(ref, 0)})
with open(out, 'x') as f:
    json.dump({'schema': 'claude54.t11c-reconciliation.v1', 'main_now': new_main, 'rows': rows}, f, indent=1)
c = collections.Counter((r['kind'], 'Q-MAIN' if r['q_main'] else 'no Q-MAIN', 'merged' if r['commit_in_main_now'] else
                         ('deleted' if r['ref_deleted'] else ('moved' if not r['ref_still_at_commit'] else 'kept'))) for r in rows)
for k, v in sorted(c.items()):
    print(k, v, '| records', sum(r['records'] for r in rows if (r['kind'], 'Q-MAIN' if r['q_main'] else 'no Q-MAIN', 'merged' if r['commit_in_main_now'] else ('deleted' if r['ref_deleted'] else ('moved' if not r['ref_still_at_commit'] else 'kept'))) == k))
print('records on pre-rewrite heads:', sum(r['records'] for r in rows if r['pre_rewrite']), '| on branch-only refs:', sum(r['records'] for r in rows if r['kind'] == 'branch only'))
