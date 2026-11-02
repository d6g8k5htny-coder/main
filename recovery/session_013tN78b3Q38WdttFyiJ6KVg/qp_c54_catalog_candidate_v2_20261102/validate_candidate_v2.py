#!/usr/bin/env python3
"""Mechanical checks of a v2 candidate catalog built by build_candidate_v2.py. One line per check; exit 1 on any failure.

Usage: python3 -I -B validate_candidate_v2.py STAGING_CATALOG CANDIDATE RECOVERY_DIR CLONES_JSON

CLONES_JSON maps a repository name (Math-, main, meta-framework, ...) to a local clone path. Engineering accounting only: these
checks certify no mathematics, and passing them adopts nothing.
"""
import gzip, hashlib, io, json, os, re, subprocess, sys, tarfile, tempfile, zipfile

REQUIRED = ['id', 'title', 'status', 'proposal_status', 'location', 'layer_a_status', 'exact_boundary', 'blocking_reason',
            'attempted', 'resumption_cost', 'isolation', 'owner', 'maintenance']
NEW_KEYS = ['proposed_quarantines', 'proposed_quarantines_federation', 'proposed_pinning_amendments', 'proposed_byte_pins',
            'proposed_closed_candidates', 'held_by_other_lane', 'a3_advice', 'accounting_dispositions', 'proposal_provenance']
HERE = os.path.dirname(os.path.abspath(__file__))


def repo_name(full):
    return full.split('/', 1)[1] if '/' in full else full


def main(staging_path, cand_path, rec_dir, clones_path):
    clones = json.load(open(clones_path))
    cache = {}

    def blob(repo, commit, path):
        key = (repo, commit, path)
        if key not in cache:
            clone = clones.get(repo)
            r = subprocess.run(['git', '-C', clone, 'cat-file', 'blob', f'{commit}:{path}'], capture_output=True) if clone else None
            if r is None or r.returncode:
                cache[key] = None
            else:
                rev = subprocess.run(['git', '-C', clone, 'rev-parse', f'{commit}:{path}'], capture_output=True, text=True).stdout.strip()
                cache[key] = (rev, r.stdout)
        return cache[key]

    def read_member(name, data, member):
        if name.lower().endswith('.zip'):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                return z.read(member)
        if re.search(r'\.(tgz|tar\.gz|tar)$', name, re.I):
            with tarfile.open(fileobj=io.BytesIO(data)) as t:
                return t.extractfile(t.getmember(member)).read()
        if name.lower().endswith('.gz'):
            return gzip.decompress(data)
        return None

    def member(repo, commit, archive, name):
        # Archive members are named OUTER or OUTER!INNER (one level of nesting), as tranche 7 names them.
        got = blob(repo, commit, archive)
        if got is None:
            return None
        try:
            outer, _, inner = name.partition('!')
            b = read_member(archive, got[1], outer)
            return read_member(outer, b, inner) if inner else b
        except Exception:
            return None

    results = []
    check = lambda name, p: results.append((name, p))
    s_raw, c_raw = open(staging_path, 'rb').read(), open(cand_path, 'rb').read()
    staging, cand = json.loads(s_raw), json.loads(c_raw)

    # 1. Additive only.
    p = []
    if not (s_raw.endswith(b'\n}\n') and c_raw.startswith(s_raw[:-3] + b',\n')):
        p.append('staging bytes are not a prefix of the candidate')
    if list(cand)[:len(staging)] != list(staging) or any(cand[k] != staging[k] for k in staging):
        p.append('a staging key or value changed')
    if [k for k in cand if k not in staging] != NEW_KEYS:
        p.append(f'unexpected added keys {[k for k in cand if k not in staging]}')
    if json.dumps(cand, indent=2, ensure_ascii=False).encode() + b'\n' != c_raw:
        p.append('candidate is not in the staging serialization')
    check('additive-only: staging bytes and values unchanged; exactly the nine new keys appended', p)

    # 2. The candidate equals a fresh build from the retained tranche files (so every row equals its verified source).
    p = []
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, 'c.json')
        r = subprocess.run([sys.executable, '-I', '-B', os.path.join(HERE, 'build_candidate_v2.py'), staging_path, rec_dir, out],
                           capture_output=True, text=True)
        if r.returncode or open(out, 'rb').read() != c_raw:
            p.append('candidate differs from a fresh build of the retained tranche files')
    prov = cand['proposal_provenance']
    for s in prov['sources']:
        if hashlib.sha256(open(os.path.join(rec_dir, s['path']), 'rb').read()).hexdigest() != s['sha256']:
            p.append(f'source digest mismatch {s["path"]}')
    c = prov['counts']
    if any(c[k] != len(cand[k]) for k in c):
        p.append('counts disagree with the rows')
    builder = open(os.path.join(HERE, 'build_candidate_v2.py')).read()
    keys = re.findall(r"^    '(\w+)': '", builder.split('SOURCES = {', 1)[1].split('}', 1)[0], re.M)
    if prov.get('missing_sources') or [x['key'] for x in prov['sources']] != keys:
        p.append(f'sources {[x["key"] for x in prov["sources"]]} differ from the builder\'s required {keys}, or some are missing')
    check(f'rows equal a fresh build of all {len(prov["sources"])} required sources; digests and counts ({json.dumps(c)})', p)

    quarantines = cand['proposed_quarantines'] + cand['proposed_quarantines_federation']
    amend = cand['proposed_pinning_amendments']

    # 3. Required fields and status vocabulary.
    p = []
    for r in quarantines + amend:
        for k in REQUIRED:
            v = r.get(k)
            if v is None or (isinstance(v, str) and not v.strip()) or (isinstance(v, list) and not v):
                p.append(f'{r.get("id")}: empty or missing {k}')
        if (r.get('status'), r.get('proposal_status'), r.get('owner'), r.get('maintenance')) != ('QUARANTINED', 'PROPOSED', 'none', 'zero'):
            p.append(f'{r.get("id")}: status/proposal_status/owner/maintenance')
        if r['floating_point'].split()[0].rstrip(':;,.—-') not in ('USED', 'NONE', 'NO_SCRIPTS'):
            p.append(f'{r["id"]}: floating_point unclassified')
    check(f'{len(quarantines) + len(amend)} proposals carry the 13 required keys, QUARANTINED/PROPOSED/none/zero, classified floating point', p)

    # 4. Unique ids.
    p = []
    ids = [r['id'] for r in staging['quarantines'] + quarantines + amend + cand['proposed_byte_pins']] + \
          [r['id'] for r in cand['proposed_closed_candidates'] if 'id' in r]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        p.append(f'duplicate ids {dup}')
    check(f'ids unique across {len(ids)} rows', p)

    # 5. Bindings of every proposal (including its identical_copies), closed candidate, held group and accounting row. The four
    # tranche-4 mechanically resolved objects carry blob ids without a binding; their git_blob values are checked at main d2b0216e.
    p, n = [], 0
    def bind_rows():
        for r in quarantines + amend + cand['proposed_closed_candidates'] + cand['held_by_other_lane'] + cand['accounting_dispositions']:
            for b in [r.get('source_binding')] + [x.get('source_binding') for x in r.get('identical_copies') or []]:
                if not b:
                    continue
                entries = list(b.get('files') or []) + list(b.get('lean_files') or []) + list(b.get('context_files') or [])
                if 'proof' in b:
                    entries.append({'path': b['proof'], 'git_blob': b['proof_git_blob'], 'bytes': b['proof_bytes'], 'sha256': b['proof_sha256']})
                for f in entries:
                    yield r, b, f
    for r, b, f in bind_rows():
        n += 1
        got = blob(repo_name(b['repository']), b['commit'], f['path'])
        if got is None:
            p.append(f'{r.get("id", r.get("object"))}: {f["path"]} absent')
        elif (got[0], len(got[1]), hashlib.sha256(got[1]).hexdigest()) != (f['git_blob'], f['bytes'], f['sha256']):
            p.append(f'{r.get("id", r.get("object"))}: {f["path"]} blob/bytes/sha256 mismatch')
    for r in cand['accounting_dispositions']:
        if r.get('tranche') == 't4' and 'resolution' in r:
            for f in r['files']:
                n += 1
                got = blob('main', 'd2b0216e15d545c47f25b6a9265dd3298971e9fe', f['path'])
                if got is None or got[0] != f['git_blob']:
                    p.append(f'{r["object"]}: {f["path"]} blob mismatch')
    check(f'{n} file bindings match repository bytes (main, Math- and federation clones)', p)

    # 6. Statements verbatim.
    p, n = [], 0
    for r in quarantines + amend + cand['proposed_closed_candidates'] + cand['held_by_other_lane']:
        b = r.get('source_binding') or {}
        for st in r.get('source_statements') or []:
            n += 1
            if b.get('members'):
                mb = member(repo_name(b['repository']), b['commit'], b['files'][0]['path'], st['file'])
                got = (None, mb) if mb is not None else None
            else:
                got = blob(repo_name(b['repository']), b['commit'], st.get('file') or b.get('proof'))
            lines = got[1].decode('utf-8').split('\n') if got else []
            a, e = st['line_start'], st['line_end']
            if not (1 <= a <= e <= len(lines)) or '\n'.join(lines[a - 1:e]) != st['text'] or not st['text'].strip():
                p.append(f'{r.get("id", r.get("location"))}: {st.get("file")}:{a}-{e} not verbatim')
    check(f'{n} quoted statements verbatim at their lines', p)

    # 7. Locations exist.
    p = []
    for r in quarantines + amend:
        for loc in r['location']:
            repo_commit, _, path = loc.partition(':')
            full, _, commit = repo_commit.partition('@')
            path, _, inner = path.partition('#')
            if inner and member(repo_name(full), commit, path, inner) is None:
                p.append(f'{r["id"]}: {loc} member not found')
                continue
            if blob(repo_name(full), commit, path) is None and \
                    subprocess.run(['git', '-C', clones.get(repo_name(full), '/nonexistent'), 'cat-file', '-e', f'{commit}:{path}'],
                                   capture_output=True).returncode:
                p.append(f'{r["id"]}: {loc} not found')
    check('every proposal location exists at its commit', p)

    # 8. Byte pins: bytes match, and the target is an adopted record whose text names the path.
    p = []
    adopted = {r['id']: json.dumps(r, ensure_ascii=False) for r in staging['quarantines']}
    adopted_title = {r['id']: r.get('title', '') for r in staging['quarantines']}
    for pin in cand['proposed_byte_pins']:
        got = blob(repo_name(pin['repository']), pin['commit'], pin['path'])
        if got is None or (got[0], len(got[1]), hashlib.sha256(got[1]).hexdigest()) != (pin['git_blob'], pin['bytes'], pin['sha256']):
            p.append(f'{pin["id"]}: bytes mismatch')
        text = adopted.get(pin['record'], '')
        title = adopted_title.get(pin['record'], '')
        directory, _, name = pin['path'].rpartition('/')
        # The target must name the full path, or name the file by its bare name in its title while naming its directory as a path.
        if not text or not (pin['path'] in text or (re.search(rf'(?<![\w./-]){re.escape(name)}(?![\w-])', title) and directory + '/' in text)):
            p.append(f'{pin["id"]}: target {pin["record"]} missing or does not name the file')
    for a in amend:
        targets = [a['target_record']] if 'target_record' in a else a.get('target_records', [])
        if not targets or any(t not in adopted for t in targets):
            p.append(f'{a["id"]}: pinning target(s) {targets} not adopted')
    check(f'{len(cand["proposed_byte_pins"])} byte pins and {len(amend)} pinning amendments target adopted records', p)

    # 9. P15 closed candidates: the declaration is a theorem at its line in the bound source.
    p = []
    for r in cand['proposed_closed_candidates']:
        if r.get('candidate_kind') != 'HOSTED_GATE_DECLARATION':
            continue
        got = blob('main', r['source_commit'], r['path'])
        if got is None or hashlib.sha256(got[1]).hexdigest() != r['source_sha256']:
            p.append(f'{r["id"]}: source digest mismatch')
            continue
        lines = got[1].decode().split('\n')
        stack = []
        for ln in lines[:r['line'] - 1]:
            m = re.match(r'\s*namespace\s+([\w.]+)', ln)
            if m:
                stack.append(m.group(1))
            m = re.match(r'\s*end\s+([\w.]+)\s*$', ln)
            if m and stack and stack[-1] == m.group(1):
                stack.pop()
        m = re.match(r'\s*theorem\s+([\w.]+)\b', lines[r['line'] - 1])
        if not m or '.'.join(stack + [m.group(1)]) != r['declaration']:
            p.append(f'{r["id"]}: line {r["line"]} is not theorem {r["declaration"]}')
    check('P15 closed candidates: each fully qualified declaration is the theorem at its line in the bound source', p)

    # 10. No proposal at or under an adopted location. Adopted locations are read in every form staging uses: repo@sha:path
    # strings, GitHub blob/tree URLs, dict locations (path or paths) and the two 'main PR #N ... dir/{A,B}.md' prose rows; the
    # PROOF_ADVANCES_* prose rows name no repository path. A federation row may sit inside an adopted region only when its
    # claim_overlaps names that record with relation INSIDE_ADOPTED_ENGINEERING_REGION. Claim-level overlaps at different
    # locations are not detected here; they are recorded as claim_overlaps or amendments (E-R-* errata).
    p = []
    adopted_locs, unparsed = {}, []
    def add_adopted(repo, path, rid):
        adopted_locs.setdefault((repo, path.rstrip('/')), rid)
    for r in staging['quarantines']:
        for loc in (r['location'] if isinstance(r['location'], list) else [r['location']]):
            if isinstance(loc, dict):
                repo = loc.get('repository', '').split('/')[-1]
                for x in ([loc['path']] if 'path' in loc else []) + list(loc.get('paths', [])):
                    add_adopted(repo, x, r['id'])
                continue
            m = re.match(r'd6g8k5htny-coder/([\w.-]+)@[0-9a-f]{40}:([^#]+)', loc)
            if m:
                add_adopted(m.group(1), m.group(2), r['id'])
                continue
            m = re.match(r'https://github.com/d6g8k5htny-coder/([\w.-]+)/(?:blob|tree)/[0-9a-f]{7,40}/([^#?]+)', loc)
            if m:
                add_adopted(m.group(1), m.group(2), r['id'])
                continue
            m = re.match(r'main PR #\d+, head [0-9a-f]{40} .*?; ([\w./-]+)/\{([\w,]+)\}\.md', loc)
            if m:
                for stem in m.group(2).split(','):
                    add_adopted('main', f'{m.group(1)}/{stem}.md', r['id'])
                continue
            unparsed.append(r['id'])
    if sorted(set(unparsed)) != ['Q-10103', 'Q-10104', 'Q-10105', 'Q-10106', 'Q-10107']:
        p.append(f'unexpected unparsed adopted locations {sorted(set(unparsed))}')
    for key in ('proposed_quarantines', 'proposed_quarantines_federation'):
        for r in cand[key]:
            allowed = {o['record'] for o in r.get('claim_overlaps') or [] if o.get('relation') == 'INSIDE_ADOPTED_ENGINEERING_REGION'}
            for loc in r['location']:
                full, _, rest = loc.partition('@')
                path = rest.partition(':')[2].partition('#')[0]
                parts = path.split('/')
                hit = [(x, adopted_locs[(full.split('/')[-1], x)]) for x in ('/'.join(parts[:i]) for i in range(1, len(parts) + 1))
                       if (full.split('/')[-1], x) in adopted_locs]
                hit = [h for h in hit if h[1] not in allowed]
                if hit:
                    p.append(f'{r["id"]}: {path} at or under adopted {hit[0][1]} ({hit[0][0]})')
    check(f'no proposal sits at or under any of {len(adopted_locs)} adopted locations (all location forms; declared engineering regions excepted)', p)

    # 11. Reconciliations: every shared file has the same blob in the federation row and in a primary main row's binding.
    p, n = [], 0
    rows = {r['id']: r for r in quarantines + amend}
    rows.update({f'{r["object"]}#{r["item_id"]}': r for r in cand['accounting_dispositions'] if r.get('tranche') == 't4' and 'item_id' in r})
    for r in cand['proposed_quarantines_federation']:
        rec = r.get('reconciliation')
        if not rec:
            continue
        fed_blobs = {f['path']: f['git_blob'] for f in r['source_binding']['files']}
        for f in rec['shared_files']:
            n += 1
            keys = [m.get('record') or m.get('item') for m in f['main']]
            main_blobs = {g['path']: g['git_blob'] for k in keys if k in rows for g in rows[k]['source_binding']['files']}
            if fed_blobs.get(f['path']) != f['git_blob'] or not any(main_blobs.get(m['path']) == f['git_blob'] for m in f['main']):
                p.append(f'{r["id"]}: {f["path"]} not byte-identical to its primary main binding')
            if not all(any(x['record'] == r['id'] for x in rows[m].get('federation_replicas', [])) for m in rec['primary_main_records']):
                p.append(f'{r["id"]}: a primary main record lacks the federation_replicas back-link')
        shared_p, only_p = [f['path'] for f in rec['shared_files']], [f['path'] for f in rec['federation_only_files']]
        if set(shared_p) & set(only_p) or len(shared_p) + len(only_p) != len(fed_blobs) or set(shared_p) | set(only_p) != set(fed_blobs):
            p.append(f'{r["id"]}: shared and federation-only files do not partition the row\'s binding')
        for f in rec['federation_only_files']:
            if fed_blobs.get(f['path']) != f['git_blob']:
                p.append(f'{r["id"]}: federation-only {f["path"]} blob differs from the row\'s binding')
    check(f'{n} reconciled federation files are byte-identical to their primary main bindings', p)

    # 12. Archive members (tranche 7): each member a row binds has the stated bytes, SHA-256 and Git blob inside the bound archive.
    p, n = [], 0
    for r in quarantines + amend + cand['accounting_dispositions']:
        b = r.get('source_binding') or {}
        for m in b.get('members') or []:
            n += 1
            mb = member(repo_name(b['repository']), b['commit'], b['files'][0]['path'], m['member'])
            if mb is None or (len(mb), hashlib.sha256(mb).hexdigest(), hashlib.sha1(b'blob %d\0' % len(mb) + mb).hexdigest()) != \
                    (m['bytes'], m['sha256'], m['git_blob']):
                p.append(f'{r.get("id", r.get("object"))}: member {m["member"]} absent or bytes differ')
    check(f'{n} archive-member bindings match the bytes inside their bound archives', p)

    # 13. Errata: every item touched exactly the rows its target names and set the stated value; a3 notes are under audited ids.
    p = []
    errata = json.load(open(os.path.join(rec_dir, 'qp_c54_errata_20261102/errata.json')))['items']
    all_rows = quarantines + amend + cand['held_by_other_lane'] + cand['accounting_dispositions']
    tagged = {}
    for r in all_rows:
        for e in r.get('errata_applied') or []:
            tagged.setdefault(e, []).append(r)
    for it in errata:
        t = it['target']
        if 'a3' in t:
            if t['a3'] not in cand['a3_advice']['final'] or not any(x['errata'] == it['id'] and x['note'] == it['value']
                                                                     for x in cand['a3_advice']['notes'].get(t['a3'], [])):
                p.append(f'{it["id"]}: a3 note missing or wrong')
            continue
        rows_ = tagged.get(it['id'], [])
        want = len(t['records']) if 'records' in t else 1
        if len(rows_) != want:
            p.append(f'{it["id"]}: tags {len(rows_)} rows, expected {want}')
        if 'records' in t or 'record' in t:
            ids_ = set(t.get('records') or [t['record']])
            if {r.get('id') for r in rows_} != ids_:
                p.append(f'{it["id"]}: tagged rows {[r.get("id") for r in rows_]} differ from its target')
        for r in rows_:
            v = r
            for part in it['field'].split('.'):
                v = v.get(part) if isinstance(v, dict) else None
            if v != it['value']:
                p.append(f'{it["id"]}: {r.get("id", r.get("object"))}.{it["field"]} is not the erratum value')
    check(f'{len(errata)} errata items each set their value on exactly their target rows', p)

    failed = 0
    for name, problems in results:
        print(('PASS ' if not problems else 'FAIL ') + name)
        for x in problems[:15]:
            print('   ', x)
        failed += bool(problems)
    print(f'{len(results) - failed}/{len(results)} checks passed')
    return 1 if failed else 0


if __name__ == '__main__':
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    raise SystemExit(main(*sys.argv[1:]))
