"""Data-only check of the V3.3 / V3.4 archives: member manifest and internal SHA256 ledgers (no member is executed)."""
import hashlib, json, re, sys, zipfile
DRIVE = re.compile(r'(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{25,44}(?![A-Za-z0-9_-])')
_isid = lambda t: bool(re.search('[A-Z]', t) and re.search('[a-z]', t) and re.search('[0-9]', t))
red = lambda s: DRIVE.sub(lambda m: '<drive-id>' if _isid(m.group(0)) else m.group(0), s)
out = {}
for tag, path in (('V33', sys.argv[1]), ('V34', sys.argv[2])):
    z = zipfile.ZipFile(path); members = {}
    for i in z.infolist():
        if i.is_dir(): continue
        b = z.read(i); members[i.filename] = (len(b), hashlib.sha256(b).hexdigest())
    root = sorted({n.split('/')[0] for n in members})
    res = {'archive_sha256': hashlib.sha256(open(path, 'rb').read()).hexdigest(), 'members': len(members), 'roots': root, 'ledgers': {}}
    for n in members:
        if not re.search(r'(SHA256SUMS|_SHA256\.txt$|06_SHA256SUMS\.txt$)', n): continue
        txt = z.read(n).decode('utf-8', 'replace'); base = n.rsplit('/', 1)[0]; rows = ok = bad = absent = 0; bad_list = []; absent_list = []
        for line in txt.splitlines():
            m = re.match(r'^([0-9a-f]{64})\s+\*?(.+?)\s*$', line)
            if not m: continue
            rows += 1; h, rel = m.groups()
            cands = [k for k in members if k.endswith('/' + rel.lstrip('./')) or k == rel]
            pref = [k for k in cands if k.startswith(base + '/')] or cands
            if not pref: absent += 1; absent_list.append(red(rel)); continue
            if any(members[k][1] == h for k in pref): ok += 1
            else: bad += 1; bad_list.append(red(rel))
        res['ledgers'][red(n)] = {'rows': rows, 'match': ok, 'mismatch': bad, 'not_in_archive': absent, 'mismatch_paths': bad_list[:20], 'absent_paths': absent_list[:40]}
    res['manifest'] = [{'path': red(k), 'bytes': v[0], 'sha256': v[1], 'path_redacted': red(k) != k} for k, v in sorted(members.items())]
    out[tag] = res
json.dump(out, open(sys.argv[3], 'w'), indent=1)
for tag, r in out.items():
    print(tag, r['archive_sha256'][:16], 'members', r['members'], 'roots', r['roots'])
    for n, l in r['ledgers'].items(): print('   ', n, {k: v for k, v in l.items() if k not in ('mismatch_paths', 'absent_paths')}, l['mismatch_paths'][:3], l['absent_paths'][:3])
