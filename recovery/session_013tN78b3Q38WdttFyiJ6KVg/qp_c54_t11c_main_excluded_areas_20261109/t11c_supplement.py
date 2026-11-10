#!/usr/bin/env python3
"""T11c supplement: rows the triage labelled WORKING or ENGINEERING that the independent audit (audit_result.json) or the second
screen (screen2.json) found to carry mathematics of their own. Writes the relabelled ledger (each changed row keeps its triage label
in 'triage_label' and gains 'relabelled_by') and supplementary deep-pass objects, one per (ref, directory) with ids
'<directory>#sup@<ref>', appended to FACTS_DIR as qbatch<N>.json from START_N on (t10_batches.py limits: 45 files / 12 objects).
Usage: python3 -I -B t11c_supplement.py LEDGER AUDIT_RESULT SCREEN2_JSON SCREEN2_DIR MERGE_REFS FACTS_DIR START_N OUT_LEDGER"""
import collections, glob, json, os, re, sys
led_p, aud_p, s2_p, s2_dir, merge_p, facts, start_n, out_led = sys.argv[1:]
start_n = int(start_n)
open_prs = {int(m) for m in re.findall(r'refs/pull/(\d+)/merge', open(merge_p).read())}
led = json.load(open(led_p)); by = {r['git_blob']: r for r in led}
hits = {}
for r in json.load(open(aud_p))['audit_rows']:
    if r['missed_mathematics'] and r['label'] in ('MATH', 'MATH_REVIEW'):
        hits[r['git_blob']] = (r['label'], 'audit', r['evidence'])
s2 = json.load(open(s2_p)); problems = []
for f in sorted(glob.glob(os.path.join(s2_dir, 'sbatch*.json'))):
    n = str(json.load(open(f))['batch']); rows = json.load(open(f))['rows']
    res = s2.get(n)
    if not res:
        problems.append(f'screen2 batch {n}: no result'); continue
    lab = collections.Counter(x['i'] for x in res['rows'])
    if any(lab[i] != 1 for i in range(len(rows))):
        problems.append(f'screen2 batch {n}: not every row answered exactly once')
    for x in res['rows']:
        if 0 <= x['i'] < len(rows) and x['carries_math'] and x['kind'] in ('MATH', 'MATH_REVIEW'):
            hits.setdefault(rows[x['i']]['git_blob'], (x['kind'], 'screen2', x['evidence']))
if problems:
    print('\n'.join(problems)); sys.exit(1)
groups = collections.defaultdict(list)
for b, (kind, src, ev) in hits.items():
    r = by[b]
    r.update(triage_label=r['label'], label=kind, relabelled_by=src, relabel_evidence=ev)
    d = r['path'].rsplit('/', 1)[0]
    r['deep_pass_key'] = f'sup:{d}'
    groups[(r['ref'], d)].append(r)
objs = []
for (ref, d), rs in sorted(groups.items()):
    m = re.match(r'refs/pull/(\d+)/head$', ref); pr = int(m.group(1)) if m else None
    if len({x['commit'] for x in rs}) != 1:
        raise SystemExit(f'{d}@{ref}: several commits')
    objs.append({'object': f'{d}#sup@{ref}', 'directory': d, 'ref': ref, 'pr': pr, 'pr_open': (pr in open_prs) if pr else None,
                 'orphan': None, 'repository': 'd6g8k5htny-coder/main', 'commit': rs[0]['commit'],
                 'triage_title': f'Files in {d} that the triage labelled WORKING/ENGINEERING and an independent re-check found to carry mathematics',
                 'triage_summary': '; '.join(f'{x["path"].rsplit("/", 1)[1]}: {x["relabel_evidence"]}' for x in rs)[:1500],
                 'triage_hint': '',
                 'files': [{'path': x['path'], 'git_blob': x['git_blob'], 'also_at': [], 'same_bytes_at': [], 'triage_label': x['label'],
                            **({'drive_manifest_row': x['drive_manifest_row']} if x.get('drive_manifest_row') else {}),
                            **({'comment_census_rows': x['comment_census_rows']} if x.get('comment_census_rows') else {})}
                           for x in sorted(rs, key=lambda x: x['path'])]})
n, cur, files = start_n, [], 0
written = []
def flush():
    global n, cur, files
    if cur:
        kind = 'math' if any(f['triage_label'] == 'MATH' for o in cur for f in o['files']) else 'eng'
        p = os.path.join(facts, f'qbatch{n}.json')
        json.dump({'batch': n, 'kind': kind, 'objects': cur}, open(p, 'x'), indent=1, ensure_ascii=False)
        written.append((n, kind, len(cur), files)); n += 1
    cur, files = [], 0
for o in objs:
    k = len(o['files'])
    if cur and (files + k > 45 or len(cur) >= 12):
        flush()
    cur.append(o); files += k
flush()
json.dump(led, open(out_led, 'x'), indent=1, ensure_ascii=False)
print(len(hits), 'relabelled rows', collections.Counter(v[1] for v in hits.values()), collections.Counter(v[0] for v in hits.values()))
print(len(objs), 'supplementary objects; batches', written)
