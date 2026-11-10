#!/usr/bin/env python3
"""T11c triage audit sample, read-only. Draws a seeded, stratified sample (by area: drive, incoming, each recovery session) of
the triage-ledger rows labelled WORKING or ENGINEERING, at RATE (default 5%, at least 3 per stratum when available), adds every
row flagged by mislabel_screen.py, and writes audit batches of at most 40 rows for independent re-judgment.
Usage: python3 -I -B audit_sample.py LEDGER MISLABEL_FLAGS OUT_DIR [RATE] [SEED]"""
import collections, json, os, random, sys
led_p, flag_p, out = sys.argv[1:4]
rate = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 20261109
led = json.load(open(led_p)); flags = {r['git_blob'] for r in json.load(open(flag_p))}
def stratum(p):
    parts = p.split('/')
    return parts[1] if parts[0] == 'recovery' and len(parts) > 2 else parts[0]
pool = collections.defaultdict(list)
for r in led:
    if r['label'] in ('WORKING', 'ENGINEERING'):
        pool[stratum(r['path'])].append(r)
rnd = random.Random(seed)
pick, why = [], {}
for k in sorted(pool):
    v = sorted(pool[k], key=lambda r: r['git_blob'])
    n = min(len(v), max(3, round(rate * len(v))))
    for r in rnd.sample(v, n):
        pick.append(r); why[r['git_blob']] = 'random'
for r in led:
    if r['git_blob'] in flags and r['git_blob'] not in why:
        pick.append(r); why[r['git_blob']] = 'flagged'
    elif r['git_blob'] in flags:
        why[r['git_blob']] = 'random+flagged'
os.makedirs(out, exist_ok=True)
rows = [{**{k: r[k] for k in ('git_blob', 'path', 'ref', 'commit', 'label', 'evidence', 'note')}, 'why': why[r['git_blob']]} for r in pick]
for i in range(0, len(rows), 40):
    json.dump({'batch': i // 40 + 1, 'rows': rows[i:i + 40]}, open(os.path.join(out, f'abatch{i // 40 + 1}.json'), 'x'), indent=1, ensure_ascii=False)
print(len(rows), 'rows;', collections.Counter(why.values()), ';', (len(rows) + 39) // 40, 'batches;',
      {k: len(v) for k, v in pool.items()})
