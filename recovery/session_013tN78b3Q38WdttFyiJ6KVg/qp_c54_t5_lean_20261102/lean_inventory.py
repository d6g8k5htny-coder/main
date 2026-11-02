#!/usr/bin/env python3
"""Static inventory of the Math- Lean files that have no CLOSED declaration and no quarantine.

Usage: python3 -I -B lean_inventory.py FED_TIP_JSON MATH_CLONE MAIN_CLONE OUT

Reads only Git objects (git show / ls-tree / grep at fixed commits); runs no Lean and no repository code. For each file it records
the declarations (keyword, name, line), risk tokens (sorry, admit, axiom declarations, native_decide, implemented_by, extern,
unsafe, opaque, set_option ... maxHeartbeats 0), imports, the nearest Lake package root, and every file under cbz/evidence on main
or anywhere in Math- whose text names the path (kernel evidence leads)."""
import json, re, subprocess, sys, collections
fed_p, math_c, main_c, out = sys.argv[1:5]
MATH_REF = '3b11bff6479f3f40ac838fe1d8b87c7ed6814606'
MAIN_REF = 'd2b0216e15d545c47f25b6a9265dd3298971e9fe'
NOT_OBL = ('specifications/lean_statements_20261010/', 'formal/ResearchFormalCoreR1.lean',
           'frontiers/determinant_moment_20261025/', 'frontiers/source_smooth_realization_20261011/',
           'specifications/lean_statement_proofs_20261011/')
def git(repo, *a):
    r = subprocess.run(['git', '-C', repo, *a], capture_output=True)
    return r.stdout.decode('utf-8', 'replace') if r.returncode == 0 else ''
fed = json.load(open(fed_p))
paths = sorted(r['path'] for r in fed['rows'] if r['repository'] == 'Math-' and r['kind'] == 'lean'
               and r['coverage'] not in ('CLOSED_SOURCE', 'QUARANTINED', 'COPY_OF_COVERED') and not r['path'].startswith(NOT_OBL))
tree = set(git(math_c, 'ls-tree', '-r', '--name-only', MATH_REF).split('\n'))
DECL = re.compile(r'^\s*(?:@\[[^\]]*\]\s*)?(?:(?:private|protected|noncomputable|partial|unsafe)\s+)*'
                  r'(theorem|lemma|def|abbrev|structure|class|instance|inductive|axiom|opaque|example)\b\s*([^\s:({\[]*)')
RISK = {'sorry': r'\bsorry\b', 'admit': r'\badmit\b', 'axiom_decl': r'^\s*axiom\b', 'native_decide': r'\bnative_decide\b',
        'implemented_by': r'implemented_by', 'extern': r'@\[extern', 'unsafe': r'\bunsafe\b', 'opaque': r'^\s*opaque\b',
        'maxHeartbeats_0': r'maxHeartbeats\s+0'}
def strip_comments(src):
    src = re.sub(r'/-.*?-/', lambda m: '\n' * m.group(0).count('\n'), src, flags=re.S)
    return '\n'.join(re.sub(r'--.*$', '', l) for l in src.split('\n'))
rows = []
for p in paths:
    src = git(math_c, 'show', f'{MATH_REF}:{p}')
    code = strip_comments(src)
    decls = []
    for i, l in enumerate(code.split('\n'), 1):
        m = DECL.match(l)
        if m:
            decls.append({'keyword': m.group(1), 'name': m.group(2), 'line': i})
    risk = {k: sum(1 for l in code.split('\n') if re.search(v, l)) for k, v in RISK.items()}
    imports = re.findall(r'^\s*import\s+(\S+)', code, flags=re.M)
    parts = p.split('/')
    lake = None
    for i in range(len(parts) - 1, 0, -1):
        d = '/'.join(parts[:i])
        if any(f'{d}/{n}' in tree for n in ('lakefile.lean', 'lakefile.toml')):
            lake = d
            break
    leads = sorted(set(l.split(':', 2)[1] for l in git(main_c, 'grep', '-l', '-F', p, MAIN_REF, '--', 'cbz/evidence').split('\n') if l)) + \
            sorted(set('Math-:' + l.split(':', 1)[1] for l in git(math_c, 'grep', '-l', '-F', p, MATH_REF).split('\n') if l))
    blob = git(math_c, 'rev-parse', f'{MATH_REF}:{p}').strip()
    rows.append({'path': p, 'git_blob': blob, 'bytes': len(src.encode()), 'lake_root': lake, 'imports': imports,
                 'declarations': decls, 'theorems': sum(d['keyword'] in ('theorem', 'lemma') for d in decls), 'risk': risk,
                 'evidence_leads': [x for x in leads if x != 'Math-:' + p][:20]})
groups = collections.defaultdict(list)
for r in rows:
    groups[r['lake_root'] or '/'.join(r['path'].split('/')[:2])].append(r['path'])
summary = {'math_ref': MATH_REF, 'files': len(rows), 'groups': {k: len(v) for k, v in sorted(groups.items())},
           'theorems': sum(r['theorems'] for r in rows),
           'files_with_sorry_or_admit': sum(bool(r['risk']['sorry'] or r['risk']['admit']) for r in rows),
           'files_with_axiom_decl': sum(bool(r['risk']['axiom_decl']) for r in rows),
           'files_with_native_decide': sum(bool(r['risk']['native_decide']) for r in rows),
           'files_with_evidence_leads': sum(bool(r['evidence_leads']) for r in rows)}
json.dump({'summary': summary, 'files': rows}, open(out, 'x'), indent=1)
print(json.dumps(summary, indent=1))
