#!/usr/bin/env python3
"""Compare C220-02 served files (git blobs at f61419d8) with the same paths inside
the github-pages artifact carried at carrier/PAGES.zip. Reads in memory only.
usage: python3 -I cmp_deployed.py PAGES.zip EXECUTOR_RESULT.json OUT.json"""
import hashlib, io, json, sys, tarfile, zipfile
zp, er, out = sys.argv[1:4]
zb = open(zp, 'rb').read()
res = {'zip_sha256': hashlib.sha256(zb).hexdigest(), 'zip_bytes': len(zb)}
z = zipfile.ZipFile(io.BytesIO(zb))
names = z.namelist(); res['zip_members'] = names
assert names == ['artifact.tar'], names
tb = z.read('artifact.tar'); res['tar_sha256'] = hashlib.sha256(tb).hexdigest(); res['tar_bytes'] = len(tb)
t = tarfile.open(fileobj=io.BytesIO(tb), mode='r:')
members = {}
bad = []
for m in t.getmembers():
    n = m.name
    if n.startswith('/') or '..' in n.split('/') or not (m.isfile() or m.isdir()):
        bad.append(n)
    if m.isfile():
        members[n.lstrip('./')] = m
res['unsafe_or_nonregular'] = bad
res['regular_files'] = len(members)
res['regular_bytes'] = sum(m.size for m in members.values())
served = json.load(open(er))['served']['files']
rows = []
for f in served:
    rel = f['path'].replace('docs/site/', '', 1)
    cands = [k for k in members if k == rel or k.endswith('/' + rel) and k.count('/') == rel.count('/') + 1]
    row = {'path': f['path'], 'served_sha256': f['sha256'], 'artifact_candidates': cands}
    if cands:
        data = t.extractfile(members[cands[0]]).read()
        row['artifact_path'] = cands[0]
        row['artifact_sha256'] = hashlib.sha256(data).hexdigest()
        row['artifact_bytes'] = len(data)
        row['git_blob_of_artifact'] = hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()
        row['identical'] = row['artifact_sha256'] == f['sha256']
    rows.append(row)
res['files'] = rows
json.dump(res, open(out, 'w'), indent=1)
for r in rows:
    print(r['path'], r.get('artifact_path'), r.get('identical'), r.get('artifact_bytes'))
print('regular', res['regular_files'], res['regular_bytes'], 'bad', bad[:5])
