#!/usr/bin/env python3
"""Build T11c triage batches from the residual of norm_dedup.json. Representative occurrence per blob: an open/closed PR head
before a branch (lowest PR number for stability), then the shortest path. Clusters: recovery/<session>/<a>/<b>, drive/mirrors/<lane>,
drive/<x>, incoming/<x>, split one level further above 150 files; packed in path order into batches of at most 160 files and
1.8 MB. For drive/ files the row of the directory's _MANIFEST.jsonl at the representative head is attached. Blobs named by
Claude S's comment census (683fdfa2 CENSUS.json) or sourcing (6af4b9ad SOURCING.json) carry those citations.
Usage: python3 -I -B t11c_batches.py MAIN EXCL_CENSUS NORM_DEDUP CENSUS_JSON SOURCING_JSON OUT_DIR"""
import collections, json, os, re, subprocess, sys
main, ex_p, nd_p, cen_p, src_p, out = sys.argv[1:]
ex = json.load(open(ex_p))['blobs']; nd = json.load(open(nd_p))['rows']
res = sorted(b for b, r in nd.items() if r['kind'] == 'residual')
def git(*a):
    return subprocess.run(['git', '-C', main, *a], capture_output=True).stdout
sizes = {}
for l in subprocess.run(['git', '-C', main, 'cat-file', '--batch-check=%(objectname) %(objectsize)'], input='\n'.join(res).encode(),
                        capture_output=True).stdout.decode().splitlines():
    s, n = l.split(); sizes[s] = int(n)
def cites(path):
    t = open(path, errors='replace').read(); d = collections.defaultdict(list)
    for m in re.finditer(r'\b[0-9a-f]{40}\b', t):
        ctx = t[max(0, m.start() - 300):m.start()]
        key = re.findall(r'"(?:key|census_key|row_key|id|comment_id|artifact_id)"\s*:\s*"([^"]{1,120})"', ctx)
        d[m.group(0)].append(key[-1] if key else '?')
    return d
cen = cites(cen_p); src = cites(src_p)
heads = {}
# representative ref: refs list holds ref names; heads from the census rows are needed
census = json.load(open(os.path.join(os.path.dirname(ex_p), '..', 'prinv', 'main_refs', 'main_history_census.json')))
for r in census['prs']:
    heads[r['ref']] = (r['pr'], r['head'])
def rank(ref):
    pr, head = heads[ref]
    return (0, pr, ref) if pr is not None else (1, 0, ref)
rep = {}
for b in res:
    ref = min(ex[b]['refs'], key=rank)
    rep[b] = (ref, heads[ref][1], min(ex[b]['paths'], key=lambda p: (len(p), p)))
def cl(p, extra=0):
    parts = p.split('/')
    d = {'recovery': 4, 'drive': 3 if p.startswith('drive/mirrors/') else 2, 'incoming': 2}[parts[0]] + extra
    return '/'.join(parts[:min(d, len(parts) - 1)])
groups = collections.defaultdict(list)
for b in res:
    groups[cl(rep[b][2])].append(b)
final = []
for k in sorted(groups):
    v = groups[k]; extra = 0
    stack = [(k, v, 0)]
    while stack:
        name, fl, e = stack.pop()
        if len(fl) > 150 and e < 3:
            sub = collections.defaultdict(list)
            for b in fl:
                sub[cl(rep[b][2], e + 1)].append(b)
            if len(sub) > 1:
                stack += sorted(((n, x, e + 1) for n, x in sub.items()), reverse=True); continue
        final.append((name, sorted(fl, key=lambda b: rep[b][2])))
manifest_cache = {}
def manifest_row(head, path):
    d, fn = path.rsplit('/', 1)
    key = (head, d)
    if key not in manifest_cache:
        rows = []
        for l in git('show', f'{head}:{d}/_MANIFEST.jsonl').decode('utf-8', 'replace').splitlines():
            try: rows.append(json.loads(l))
            except Exception: pass
        manifest_cache[key] = rows
    for r in manifest_cache[key]:
        vals = [str(v) for v in r.values() if isinstance(v, str)]
        if any(v == fn or v.endswith('/' + fn) for v in vals):
            return {k: r[k] for k in r if k in ('drive_id', 'id', 'exact', 'drive_path', 'mime_type', 'mimeType', 'sha256', 'bytes', 'modified', 'modifiedTime', 'export')}
    return None
chunked = []
for name, fl in final:
    part, pb, k = [], 0, 0
    for b in fl:
        if part and (len(part) >= 160 or pb + sizes[b] > 1_800_000):
            k += 1; chunked.append((f'{name} [part {k}]', part)); part, pb = [], 0
        part.append(b); pb += sizes[b]
    chunked.append((f'{name} [part {k + 1}]' if k else name, part))
final = chunked
batches, cur, cur_b = [], [], 0
for name, fl in final:
    if cur and (len(cur) + len(fl) > 160 or cur_b + sum(sizes[b] for b in fl) > 1_800_000):
        batches.append(cur); cur, cur_b = [], 0
    for b in fl:
        ref, head, path = rep[b]
        it = {'git_blob': b, 'path': path, 'bytes': sizes[b], 'ref': ref, 'commit': head, 'cluster': name,
              'also_at_paths': [p for p in ex[b]['paths'] if p != path][:3], 'ref_count': len(ex[b]['refs'])}
        if b in cen: it['comment_census_rows'] = sorted(set(cen[b]))[:5]
        if b in src: it['impl_sourcing_rows'] = sorted(set(src[b]))[:5]
        if path.startswith('drive/'):
            it['drive_manifest_row'] = manifest_row(head, path)
        cur.append(it); cur_b += sizes[b]
if cur: batches.append(cur)
os.makedirs(out, exist_ok=True)
for i, bt in enumerate(batches, 1):
    json.dump({'batch': i, 'files': bt}, open(os.path.join(out, f'tbatch{i}.json'), 'x'), indent=1, ensure_ascii=False)
print(len(final), 'clusters;', len(batches), 'batches;', sum(len(b) for b in batches), 'files')
print('drive manifest rows found:', sum(1 for bt in batches for it in bt if it.get('drive_manifest_row')), 'of',
      sum(1 for bt in batches for it in bt if it['path'].startswith('drive/')))
print('max files', max(len(b) for b in batches), 'max bytes', max(sum(i['bytes'] for i in b) for b in batches))
