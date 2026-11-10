#!/usr/bin/env python3
"""T11c: from the triage batches (tb/tbatchN.json) and the triage verdicts (triage.json: {"N": {"files": [...], "packets": [...]}}),
check that every file of every batch is labelled exactly once, write the triage ledger for every residual blob, and build the
deep-pass objects: each triage packet (MATH / MATH_REVIEW files), split by representative ref; MATH or MATH_REVIEW files in no packet
form one object per (ref, directory). An object is 'math' if it holds a MATH file, else 'eng' (review-only; draft and screen). Read-only on git.
Usage: python3 -I -B t11c_objects.py TB_DIR TRIAGE_JSON MERGE_REFS_TXT OUT_DIR"""
import collections, glob, json, os, re, sys
tb_dir, tri_p, merge_p, out = sys.argv[1:]
open_prs = {int(m) for m in re.findall(r'refs/pull/(\d+)/merge', open(merge_p).read())}
tri = json.load(open(tri_p))
batches = {int(re.search(r'tbatch(\d+)\.json$', f).group(1)): json.load(open(f))['files'] for f in glob.glob(os.path.join(tb_dir, 'tbatch*.json'))}
problems, ledger = [], []
groups = collections.defaultdict(list)  # (ref, key) -> [(file entry, label)]
titles = {}
for n in sorted(batches):
    files = batches[n]
    r = tri.get(str(n))
    if not r:
        problems.append(f'batch {n}: no triage result'); continue
    lab = collections.defaultdict(list)
    for x in r['files']:
        lab[x['i']].append(x)
    for i in range(len(files)):
        if len(lab.get(i, [])) != 1:
            problems.append(f'batch {n}: file {i} labelled {len(lab.get(i, []))} times')
    extra = sorted(set(lab) - set(range(len(files))))
    if extra:
        problems.append(f'batch {n}: labels for nonexistent indices {extra[:5]}')
    in_packet = {}
    for k, p in enumerate(r['packets']):
        for i in p['files']:
            if 0 <= i < len(files) and lab.get(i) and lab[i][0]['label'] in ('MATH', 'MATH_REVIEW'):
                in_packet.setdefault(i, k)
    for i, f in enumerate(files):
        if len(lab.get(i, [])) != 1:
            continue
        x = lab[i][0]
        row = {'git_blob': f['git_blob'], 'path': f['path'], 'ref': f['ref'], 'commit': f['commit'], 'bytes': f['bytes'],
               'triage_batch': n, 'label': x['label'], 'copy_of': x['copy_of'], 'evidence': x['evidence'], 'note': x['note']}
        for k in ('comment_census_rows', 'impl_sourcing_rows', 'drive_manifest_row'):
            if f.get(k):
                row[k] = f[k]
        if x['label'] in ('MATH', 'MATH_REVIEW'):
            if i in in_packet:
                k = in_packet[i]; key = f'b{n}p{k}'
                titles[key] = (r['packets'][k]['title'], r['packets'][k]['summary'], r['packets'][k]['already_accounted_hint'])
            else:
                key = 'dir:' + f['path'].rsplit('/', 1)[0]
                titles.setdefault(key, (f'Unpacketed mathematics in {f["path"].rsplit("/", 1)[0]}', '', ''))
            groups[(f['ref'], key)].append((f, x['label']))
            row['deep_pass_key'] = key
        ledger.append(row)
if problems:
    print('\n'.join(problems[:50])); print(len(problems), 'problem(s); nothing written'); sys.exit(1)
objs = {'math': [], 'eng': []}  # 'eng' = review-only objects (draft + screen), the t10_batches.py key
seen_ids = collections.Counter()
for (ref, key), fl in sorted(groups.items()):
    paths = [f['path'] for f, _ in fl]
    common = os.path.commonpath([p.rsplit('/', 1)[0] for p in paths])
    seen_ids[(common, ref)] += 1
    oid = f'{common}@{ref}' if seen_ids[(common, ref)] == 1 else f'{common}#{seen_ids[(common, ref)]}@{ref}'
    m = re.match(r'refs/pull/(\d+)/head$', ref)
    pr = int(m.group(1)) if m else None
    title, summary, hint = titles[key]
    o = {'object': oid, 'directory': common, 'ref': ref, 'pr': pr, 'pr_open': (pr in open_prs) if pr else None,
         'orphan': None, 'repository': 'd6g8k5htny-coder/main', 'commit': fl[0][0]['commit'],
         'triage_title': title, 'triage_summary': summary, 'triage_hint': hint,
         'files': [{'path': f['path'], 'git_blob': f['git_blob'], 'also_at': f.get('also_at_paths', []), 'same_bytes_at': [],
                    'triage_label': lab, **({'drive_manifest_row': f['drive_manifest_row']} if f.get('drive_manifest_row') else {}),
                    **({'comment_census_rows': f['comment_census_rows']} if f.get('comment_census_rows') else {}),
                    **({'impl_sourcing_rows': f['impl_sourcing_rows']} if f.get('impl_sourcing_rows') else {})}
                   for f, lab in sorted(fl, key=lambda t: t[0]['path'])]}
    if len({f['commit'] for f, _ in fl}) != 1:
        raise SystemExit(f'{oid}: files from several commits')
    objs['math' if any(l == 'MATH' for _, l in fl) else 'eng'].append(o)
os.makedirs(out, exist_ok=True)
json.dump(ledger, open(os.path.join(out, 'triage_ledger.json'), 'x'), indent=1, ensure_ascii=False)
json.dump(objs, open(os.path.join(out, 'objects_t11c.json'), 'x'), indent=1, ensure_ascii=False)
c = collections.Counter(r['label'] for r in ledger)
print('ledger', len(ledger), dict(c))
for k, v in objs.items():
    print(k, len(v), 'objects', sum(len(o['files']) for o in v), 'files', sum(f['path'].endswith('.lean') for o in v for f in o['files']), 'lean')
