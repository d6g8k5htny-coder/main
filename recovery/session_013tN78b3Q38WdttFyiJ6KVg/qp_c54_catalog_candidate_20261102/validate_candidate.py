#!/usr/bin/env python3
"""Mechanical checks of a candidate catalog built by build_candidate.py. Prints one line per check; exit 1 on any failure.

Usage: python3 -I -B validate_candidate.py STAGING_CATALOG CANDIDATE_CATALOG T1_RECORDS T2_RECORDS MATH_REPO

These checks are engineering accounting only. They do not certify any mathematics, and passing them adopts nothing.
"""
import hashlib
import json
import subprocess
import sys

REQUIRED = ['id', 'title', 'status', 'proposal_status', 'location', 'layer_a_status', 'exact_boundary', 'blocking_reason',
            'attempted', 'resumption_cost', 'isolation', 'owner', 'maintenance']
MATH = 'd6g8k5htny-coder/Math-'


def git_bytes(repo, commit, path):
    r = subprocess.run(['git', '-C', repo, 'cat-file', 'blob', f'{commit}:{path}'], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def git_rev(repo, commit, path):
    r = subprocess.run(['git', '-C', repo, 'rev-parse', '--verify', '-q', f'{commit}:{path}'], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def main(staging_path, candidate_path, t1_path, t2_path, math_repo):
    results = []

    def check(name, problems):
        results.append((name, problems))

    s_raw, c_raw = open(staging_path, 'rb').read(), open(candidate_path, 'rb').read()
    staging, cand = json.loads(s_raw), json.loads(c_raw)
    t1, t2 = json.load(open(t1_path)), json.load(open(t2_path))
    proposed, amendments, nonobl = cand['proposed_quarantines'], cand['proposed_pinning_amendments'], cand['proposed_non_obligations']

    # 1. Additive only: the staging bytes up to the closing brace are a prefix, and every staging value is unchanged.
    p = []
    if not (s_raw.endswith(b'\n}\n') and c_raw.startswith(s_raw[:-3] + b',\n')):
        p.append('staging bytes are not a prefix of the candidate')
    if list(cand)[:len(staging)] != list(staging) or any(cand[k] != staging[k] for k in staging):
        p.append('a staging key or value changed')
    extra = [k for k in cand if k not in staging]
    if extra != ['proposed_quarantines', 'proposed_pinning_amendments', 'proposed_non_obligations', 'proposal_provenance']:
        p.append(f'unexpected added keys {extra}')
    if json.dumps(cand, indent=2, ensure_ascii=False).encode() + b'\n' != c_raw:
        p.append('candidate is not in the staging serialization')
    check('additive-only: staging catalog bytes and values unchanged; four keys appended', p)

    # 2. Proposals are exactly the verified tranche files.
    p = []
    want_new = list(t1['records']) + [r for r in t2['records'] if r['proposal_kind'] == 'NEW_RECORD']
    if proposed != want_new:
        p.append('proposed_quarantines differ from the tranche files')
    want_am = [r for r in t2['records'] if r['proposal_kind'] == 'PINNING_AMENDMENT']
    if [{k: v for k, v in a.items() if k != 'target_record'} for a in amendments] != want_am:
        p.append('proposed_pinning_amendments differ from the tranche file')
    if nonobl != t2['not_obligations']:
        p.append('proposed_non_obligations differ from the tranche file')
    prov = cand['proposal_provenance']
    for src, path in zip(prov['sources'], (t1_path, t2_path)):
        if src['sha256'] != hashlib.sha256(open(path, 'rb').read()).hexdigest():
            p.append(f'source digest mismatch for tranche {src["tranche"]}')
    if prov['staging_catalog_sha256'] != hashlib.sha256(s_raw).hexdigest():
        p.append('staging digest mismatch')
    c = prov['counts']
    if (c['proposed_quarantines'], c['proposed_pinning_amendments'], c['proposed_non_obligations']) != (len(proposed), len(amendments), len(nonobl)):
        p.append('counts disagree with the rows')
    check(f'proposals equal the verified tranche files ({len(proposed)} records, {len(amendments)} amendments, {len(nonobl)} not-obligations); digests and counts', p)

    # 3. Required fields: every key that all adopted rows carry, non-empty.
    p = []
    common = set.intersection(*(set(r) for r in staging['quarantines']))
    if set(REQUIRED) != common:
        p.append(f'required-key list differs from the keys all adopted rows carry: {sorted(common ^ set(REQUIRED))}')
    for r in proposed + amendments:
        for k in REQUIRED:
            v = r.get(k)
            if v is None or (isinstance(v, str) and not v.strip()) or (isinstance(v, list) and not v):
                p.append(f'{r.get("id")}: empty or missing {k}')
    check(f'required fields present and non-empty ({len(REQUIRED)} keys carried by all {len(staging["quarantines"])} adopted rows)', p)

    # 4. Status vocabulary.
    p = []
    for r in proposed + amendments:
        if (r['status'], r['proposal_status'], r['owner'], r['maintenance']) != ('QUARANTINED', 'PROPOSED', 'none', 'zero'):
            p.append(f'{r["id"]}: status/proposal_status/owner/maintenance not QUARANTINED/PROPOSED/none/zero')
        if r['floating_point'].split()[0].rstrip(':;,.') not in ('USED', 'NONE', 'NO_SCRIPTS'):
            p.append(f'{r["id"]}: floating_point unclassified')
    check('status QUARANTINED, proposal_status PROPOSED, owner none, maintenance zero, floating_point classified', p)

    # 5. Unique ids, in their own range.
    p = []
    ids = [r['id'] for r in staging['quarantines'] + proposed + amendments]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        p.append(f'duplicate ids {dup}')
    p += [f'{r["id"]}: not in the QP-C54- range' for r in proposed + amendments if not r['id'].startswith('QP-C54-')]
    check(f'ids unique across {len(ids)} rows; proposals in the QP-C54- range', p)

    # 6. Bindings: blob, bytes and sha256 of every bound file, read from Math- at the bound commit.
    p, n = [], 0
    for r in proposed + amendments + nonobl:
        b = r['source_binding']
        if b['repository'] != MATH:
            p.append(f'{r.get("id", r.get("object"))}: binding repository {b["repository"]}')
            continue
        entries = list(b['files'])
        if 'proof' in b:
            entries.append({'path': b['proof'], 'git_blob': b['proof_git_blob'], 'bytes': b['proof_bytes'], 'sha256': b['proof_sha256']})
        for f in entries:
            n += 1
            data = git_bytes(math_repo, b['commit'], f['path'])
            if data is None:
                p.append(f'{r.get("id", r.get("object"))}: {f["path"]} absent at {b["commit"][:8]}')
            elif (git_rev(math_repo, b['commit'], f['path']), len(data), hashlib.sha256(data).hexdigest()) != (f['git_blob'], f['bytes'], f['sha256']):
                p.append(f'{r.get("id", r.get("object"))}: {f["path"]} blob/bytes/sha256 mismatch')
    check(f'source bindings match Math- bytes ({n} file bindings)', p)

    # 7. Statements: verbatim at their lines.
    p, n = [], 0
    for r in proposed + amendments:
        b = r['source_binding']
        for st in r['source_statements']:
            n += 1
            path = st.get('file') or b.get('proof')
            data = git_bytes(math_repo, b['commit'], path)
            lines = data.decode('utf-8').split('\n') if data is not None else []
            a, e = st['line_start'], st['line_end']
            if not (1 <= a <= e <= len(lines)) or '\n'.join(lines[a - 1:e]) != st['text'] or not st['text'].strip():
                p.append(f'{r["id"]}: {path}:{a}-{e} not verbatim')
    check(f'quoted statements verbatim at their lines ({n} statements)', p)

    # 8. Locations exist at their commits.
    p = []
    for r in proposed + amendments:
        for loc in r['location']:
            repo_commit, _, path = loc.partition(':')
            repo, _, commit = repo_commit.partition('@')
            if repo != MATH or git_rev(math_repo, commit, path) is None:
                p.append(f'{r["id"]}: location {loc} not found')
    check('every proposal location exists in Math- at its commit', p)

    # 9. Pinning amendments name an adopted record that mentions the object but pins no bytes for it.
    p = []
    by_id = {r['id']: r for r in staging['quarantines']}
    for a in amendments:
        t = by_id.get(a['target_record'])
        obj = a['location'][0].partition(':')[2]
        if t is None:
            p.append(f'{a["id"]}: target {a["target_record"]} not adopted')
        elif obj.rsplit('/', 1)[-1] not in json.dumps(t, ensure_ascii=False):
            p.append(f'{a["id"]}: target {a["target_record"]} does not mention {obj}')
        elif a['target_record'] not in a.get('mentioned_by', []):
            p.append(f'{a["id"]}: target not among mentioned_by')
    check(f'pinning amendments target adopted records that name the object ({len(amendments)})', p)

    # 10. No double disposition: no new proposal sits at or under an adopted Math- location.
    p = []
    adopted = set()
    for r in staging['quarantines']:
        for loc in (r['location'] if isinstance(r['location'], list) else [r['location']]):
            if isinstance(loc, str) and loc.startswith(MATH + '@'):
                adopted.add(loc.partition(':')[2].split('#', 1)[0].rstrip('/'))
    for r in proposed:
        path = r['location'][0].partition(':')[2]
        parts = path.split('/')
        hit = [x for x in ('/'.join(parts[:i]) for i in range(1, len(parts) + 1)) if x in adopted]
        if hit:
            p.append(f'{r["id"]}: {path} already under adopted location {hit[0]}')
    check('no new proposal sits at or under an adopted Math- location', p)

    failed = 0
    for name, problems in results:
        print(('PASS ' if not problems else 'FAIL ') + name)
        for x in problems[:20]:
            print('   ', x)
        if len(problems) > 20:
            print(f'    ... {len(problems) - 20} more')
        failed += bool(problems)
    print(f'{len(results) - failed}/{len(results)} checks passed')
    return 1 if failed else 0


if __name__ == '__main__':
    if len(sys.argv) != 6:
        raise SystemExit(__doc__)
    raise SystemExit(main(*sys.argv[1:]))
