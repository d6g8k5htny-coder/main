#!/usr/bin/env python3
"""Check blob/bytes/sha256 of every source binding in the adopted quarantines rows. Usage: CATALOG MATH_REPO MAIN_REPO"""
import hashlib, json, subprocess, sys, collections
cat, math, main = sys.argv[1:4]
repos = {'d6g8k5htny-coder/Math-': math, 'd6g8k5htny-coder/main': main}
q = json.load(open(cat))['quarantines']
stats, bad, skipped, seen = collections.Counter(), [], collections.Counter(), set()
for r in q:
    bs = []
    if isinstance(r.get('source_bindings'), list): bs += r['source_bindings']
    if isinstance(r.get('source_binding'), dict): bs.append(r['source_binding'])
    if not bs: stats['rows_without_binding'] += 1
    for b in bs:
        if not all(k in b for k in ('repository', 'commit', 'path', 'git_blob', 'sha256', 'bytes')):
            skipped['incomplete_binding_keys'] += 1; continue
        repo = repos.get(b['repository'])
        if not repo: skipped[b['repository']] += 1; continue
        key = (b['repository'], b['commit'], b['path'], b['git_blob'], b['sha256'], b['bytes'])
        if key in seen: continue
        seen.add(key)
        p = subprocess.run(['git', '-C', repo, 'cat-file', 'blob', f"{b['commit']}:{b['path']}"], capture_output=True)
        if p.returncode:
            bad.append((r['id'], b['path'], 'absent or commit missing')); continue
        blob = subprocess.run(['git', '-C', repo, 'rev-parse', f"{b['commit']}:{b['path']}"], capture_output=True, text=True).stdout.strip()
        if (blob, len(p.stdout), hashlib.sha256(p.stdout).hexdigest()) != (b['git_blob'], b['bytes'], b['sha256']):
            bad.append((r['id'], b['path'], 'mismatch'))
        stats['checked'] += 1
print(dict(stats), dict(skipped)); print('problems', len(bad)); [print(' ', x) for x in bad[:20]]
