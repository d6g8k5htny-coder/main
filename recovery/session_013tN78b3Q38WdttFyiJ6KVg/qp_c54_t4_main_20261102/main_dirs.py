#!/usr/bin/env python3
"""Directory-level census of main at REF against the candidate catalog. Usage: CANDIDATE MAIN_CLONE MATH_CLONE COMMENT_DISP OUT"""
import json, re, subprocess, sys, collections
REF = 'd2b0216e15d545c47f25b6a9265dd3298971e9fe'
cand_p, main_c, math_c, disp_p, out = sys.argv[1:6]
def git(repo, *a):
    r = subprocess.run(['git', '-C', repo, *a], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ''
cand = json.load(open(cand_p))
rows = []
for line in git(main_c, 'ls-tree', '-r', '-l', REF).splitlines():
    meta, path = line.split('\t', 1)
    mode, typ, blob, size = meta.split()
    rows.append((path, blob, int(size) if size.isdigit() else 0))
EXCL = ('cbz/', 'tests/', 'tools/', '.github/', 'docs/site/', 'docs/retro/', 'governance/', 'ctx/', 'history/', 'interface/',
        'records/', 'architecture/', 'data/')
mathy = re.compile(r'\.(md|tex|py|json|lean|txt)$')
# Disposed main locations and disposed blobs (both repositories) at each location's commit.
locs, loc_commits = set(), []
REPOS = {'d6g8k5htny-coder/main': ('main', main_c), 'd6g8k5htny-coder/Math-': ('Math-', math_c)}
def add_loc(repo, commit, path):
    if repo == 'd6g8k5htny-coder/main':
        locs.add(path.rstrip('/'))
    loc_commits.append((repo, commit, path.rstrip('/')))
for r in cand['quarantines'] + cand['proposed_quarantines'] + cand['proposed_pinning_amendments']:
    for l in (r['location'] if isinstance(r['location'], list) else [r['location']]):
        if not isinstance(l, str):
            continue
        m = re.match(r'(d6g8k5htny-coder/(?:main|Math-))@([0-9a-f]{40}):([^#]+)', l)
        if m:
            add_loc(m.group(1), m.group(2), m.group(3))
        m = re.match(r'https://github.com/(d6g8k5htny-coder/(?:main|Math-))/blob/([0-9a-f]{40})/([^#]+)', l)
        if m:
            add_loc(m.group(1), m.group(2), m.group(3))
for r in cand['main_census_independent_validation']:
    add_loc('d6g8k5htny-coder/main', r['commit'], r['path'])
disposed_blobs = {}
for repo, commit, path in sorted(set(loc_commits)):
    name, clone = REPOS[repo]
    for line in git(clone, 'ls-tree', '-r', commit, '--', path).splitlines():
        meta, p = line.split('\t', 1)
        disposed_blobs.setdefault(meta.split()[2], f'{name}@{commit[:8]}:{p}')
disp = json.load(open(disp_p))
census_by_dir = collections.defaultdict(set)
for a in disp['rows']:
    for c in a['carriers']:
        if c['repo'] == 'main' and '/' in c['path']:
            census_by_dir[c['path'].rsplit('/', 1)[0]].add(a['key'])
dirs = collections.defaultdict(list)
for path, blob, size in rows:
    if path.startswith(EXCL) or '/' not in path or not mathy.search(path):
        continue
    dirs[path.rsplit('/', 1)[0]].append({'path': path, 'git_blob': blob, 'bytes': size, 'copy_of': disposed_blobs.get(blob),
                                         'exact_location': path in locs})
proof_packets = {l.rsplit('/', 1)[0] for l in locs if l.endswith('/PROOF.md')}
def covered(d):
    # EXACT or ANCESTOR: a disposed location is this directory or one of its ancestors.
    parts = d.split('/')
    for i in range(len(parts), 0, -1):
        if '/'.join(parts[:i]) in locs:
            return '/'.join(parts[:i])
    # PACKET: this directory, or an ancestor, directly holds a disposed PROOF.md.
    for i in range(len(parts), 0, -1):
        if '/'.join(parts[:i]) in proof_packets:
            return '/'.join(parts[:i]) + '/PROOF.md (packet)'
    return None
out_rows = []
for d in sorted(dirs):
    c = covered(d)
    fs = dirs[d]
    rec = {'dir': d, 'files': fs, 'n_files': len(fs), 'covered_by_location': c,
           'all_copies_of_disposed': all(f['copy_of'] for f in fs),
           'all_exact_or_copies': all(f['copy_of'] or f['exact_location'] for f in fs),
           'n_exact_location': sum(f['exact_location'] for f in fs), 'census_artifacts': sorted(census_by_dir.get(d, []))}
    out_rows.append(rec)
unc = [r for r in out_rows if not r['covered_by_location']]
summary = {'ref': REF, 'dirs': len(out_rows), 'covered_by_location': len(out_rows) - len(unc), 'uncovered': len(unc),
           'uncovered_all_copies': sum(r['all_copies_of_disposed'] for r in unc),
           'uncovered_all_exact_or_copies': sum(r['all_exact_or_copies'] for r in unc),
           'uncovered_with_census_artifacts': sum(bool(r['census_artifacts']) for r in unc)}
json.dump({'summary': summary, 'dirs': out_rows}, open(out, 'x'), indent=1)
print(json.dumps(summary))
