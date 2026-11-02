#!/usr/bin/env python3
"""Directory-level census of a repository at REF against a candidate catalog (all adopted, proposed, amendment, federation,
accounting and pin rows, and CLOSED declarations). Uses ls-tree -z, so non-ASCII paths are kept.
Usage: python3 -I -B math_dirs.py CLONE REPO_FULL REF CANDIDATE CLOSED_JSON OUT [MATH_CLONE]"""
import json, re, subprocess, sys, collections
clone, repo_full, ref, cand_p, closed_p, out = sys.argv[1:7]
def git(*a):
    return subprocess.run(['git', '-C', clone, *a], capture_output=True, check=True).stdout
ref = git('rev-parse', ref).decode().strip()
rows = []
for rec in git('ls-tree', '-r', '-l', '-z', ref).split(b'\0'):
    if not rec:
        continue
    meta, path = rec.split(b'\t', 1)
    mode, typ, blob, size = meta.decode().split()
    rows.append((path.decode('utf-8', 'replace'), blob, int(size) if size.isdigit() else 0))
cand = json.load(open(cand_p))
short = repo_full.split('/', 1)[1]
locs, bound = set(), set()
def add_loc(l):
    if isinstance(l, str):
        m = re.match(r'(?:d6g8k5htny-coder/)?([\w.-]+)@[0-9a-f]{7,40}:([^#]+)', l)
        if m and m.group(1) == short:
            locs.add(m.group(2).rstrip('/'))
        m = re.match(r'https://github.com/d6g8k5htny-coder/([\w.-]+)/(?:blob|tree)/[0-9a-f]{7,40}/([^#?]+)', l)
        if m and m.group(1) == short:
            locs.add(m.group(2).rstrip('/'))
    elif isinstance(l, dict):
        if l.get('repository', '').endswith(short):
            for p in ([l['path']] if 'path' in l else []) + list(l.get('paths', [])):
                locs.add(p.rstrip('/'))
def add_binding(b):
    if isinstance(b, dict) and b.get('repository', '').endswith('/' + short):
        for f in (b.get('files') or []) + (b.get('lean_files') or []) + (b.get('context_files') or []):
            bound.add(f['path'])
        if b.get('proof'):
            bound.add(b['proof'])
for key in ('quarantines', 'proposed_quarantines', 'proposed_quarantines_federation', 'proposed_pinning_amendments',
            'proposed_closed_candidates', 'held_by_other_lane', 'accounting_dispositions'):
    for r in cand.get(key, []):
        for l in (r.get('location') if isinstance(r.get('location'), list) else [r.get('location')]):
            add_loc(l)
        add_binding(r.get('source_binding'))
        for f in r.get('files') or []:
            if isinstance(f, dict) and 'path' in f and r.get('tranche') == 't4' and short == 'main':
                bound.add(f['path'])
        if r.get('lake_root') and short == 'Math-':
            locs.add(r['lake_root'].rstrip('/'))
for p in cand.get('proposed_byte_pins', []):
    if p['repository'].endswith('/' + short):
        bound.add(p['path'])
# Adopted REGISTER anchors: specifications/lean_statements_20261010/REGISTER.json#claims/<id>. A landing claim covers its layer0
# source file; an existing_lean_package claim 'lean:<repo>-<top>-<name>' covers that package directory.
register_cache = {}
def register(commit):
    if commit not in register_cache:
        try:
            register_cache[commit] = json.loads(subprocess.run(['git', '-C', math_clone, 'show',
                f'{commit}:specifications/lean_statements_20261010/REGISTER.json'], capture_output=True, check=True).stdout)
        except Exception:
            register_cache[commit] = None
    return register_cache[commit]
math_clone = clone if short == 'Math-' else sys.argv[7] if len(sys.argv) > 7 else None
anchored = []
for r in cand.get('quarantines', []):
    for l in (r['location'] if isinstance(r['location'], list) else [r['location']]):
        m = isinstance(l, str) and re.match(r'(?:d6g8k5htny-coder/)?Math-@([0-9a-f]{40}):specifications/lean_statements_20261010/REGISTER\.json#claims/(.+)', l)
        if not m or not math_clone:
            continue
        reg = register(m.group(1))
        if not reg:
            continue
        claim = next((c for c in reg['claims'] if c['claim_id'] == m.group(2)), None)
        if not claim:
            continue
        if claim['kind'] == 'existing_lean_package':
            mm = re.match(r'lean:(math|main)-(frontier|formal)-(.+)', claim['claim_id'])
            if mm and {'math': 'Math-', 'main': 'main'}[mm.group(1)] == short:
                target = ('frontiers/' if mm.group(2) == 'frontier' else 'formal/') + mm.group(3)
                locs.add(target); anchored.append((r['id'], target))
        else:
            sid = (claim.get('layer0') or {}).get('source_id')
            src = next((x for x in reg['sources'] if x['id'] == sid), None)
            if src and src['repository'].endswith('/' + short):
                locs.add(src['path']); anchored.append((r['id'], src['path']))
print('REGISTER anchors resolved to paths in', short, ':', len(anchored))
closed = {e['path'] for e in json.load(open(closed_p))['entries'] if e['repository'].rstrip('/').split('/')[-1] == short}
proof_packets = {l.rsplit('/', 1)[0] for l in locs if l.endswith('/PROOF.md')}
# A disposed FILE location (NOTE.md, SCOPE.md, DESIGN.md, README.md, ...) makes its directory a packet whose companions are
# covered at claim level (their bytes may still need pinning); reported separately from location coverage.
file_packets = {l.rsplit('/', 1)[0] for l in locs if '/' in l and re.search(r'\.[A-Za-z0-9]+$', l)}
closed_dirs = {c.rsplit('/', 1)[0] for c in closed}
mathy = re.compile(r'\.(md|tex|py|json|lean|txt|sage|ipynb|m|jl|csv)$')
dirs = collections.defaultdict(list)
for path, blob, size in rows:
    if not mathy.search(path):
        continue
    dirs[path.rsplit('/', 1)[0] if '/' in path else '(root)'].append({'path': path, 'git_blob': blob, 'bytes': size,
                                                                       'bound': path in bound, 'closed': path in closed})
def covered(d):
    parts = d.split('/')
    for i in range(len(parts), 0, -1):
        x = '/'.join(parts[:i])
        if x in locs:
            return x
        if x in proof_packets:
            return x + '/PROOF.md (packet)'
    for i in range(len(parts), 0, -1):
        x = '/'.join(parts[:i])
        if x in file_packets:
            return x + ' (file-located packet)'
    return None
EXCL = ('cbz/', 'tests/', 'tools/', '.github/', '.claude/', 'docs/site/', 'docs/retro/', 'docs/superpowers/', 'governance/',
        'ctx/', 'history/', 'interface/', 'records/', 'architecture/', 'data/', 'formal/tests/')
texts = {}
for key in ('quarantines', 'proposed_quarantines', 'proposed_quarantines_federation', 'proposed_pinning_amendments'):
    for r in cand.get(key, []):
        texts[r['id']] = json.dumps(r, ensure_ascii=False)
out_rows = []
for d in sorted(dirs):
    fs = dirs[d]
    c = covered(d)
    unbound = [f for f in fs if not (f['bound'] or f['closed'] or f['path'] in locs)]
    excluded = (d + '/').startswith(EXCL) or d == '(root)'
    mentioned = sorted(rid for rid, t in texts.items() if d + '/' in t or t.count(d) and any(f['path'] in t for f in fs)) if not c else []
    out_rows.append({'dir': d, 'n_files': len(fs), 'covered_by_location': c, 'n_unbound': len(unbound), 'excluded_engineering': excluded,
                     'mentioned_by': mentioned, 'has_closed': d in closed_dirs, 'files': fs})
unc = [r for r in out_rows if not r['covered_by_location'] and r['n_unbound'] and not r['excluded_engineering']]
print('excluded engineering dirs with unbound files:', sum(1 for r in out_rows if not r['covered_by_location'] and r['n_unbound'] and r['excluded_engineering']))
print('of the remaining, mentioned by some record text:', sum(1 for r in unc if r['mentioned_by']))
json.dump({'repo': repo_full, 'ref': ref, 'dirs': out_rows}, open(out, 'x'), indent=1)
print(repo_full, ref[:8], 'dirs', len(out_rows), 'uncovered dirs with unbound files', len(unc), 'files', sum(r['n_unbound'] for r in unc))
top = collections.Counter(r['dir'].split('/')[0] for r in unc)
print(top.most_common(30))
