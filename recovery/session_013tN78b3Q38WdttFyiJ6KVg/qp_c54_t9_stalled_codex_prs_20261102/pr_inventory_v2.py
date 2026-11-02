#!/usr/bin/env python3
"""Read-only inventory of unmerged mathematical content in open Math- PRs (heads fetched as refs/remotes/pr/N).
For each PR: the math-type files it adds or changes relative to its merge base with Math- main, whether each file's exact bytes
(Git blob) already exist anywhere in main's tree, and the directories of those that do not. Usage: MATH_CLONE MAIN_REF OUT PR..."""
import json, re, subprocess, sys, collections
clone, main_ref, out, prs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
def git(*a):
    return subprocess.run(['git', '-C', clone, *a], capture_output=True, check=True).stdout
main_ref = git('rev-parse', main_ref).decode().strip()
main_blobs = set()
for rec in git('ls-tree', '-r', '-z', main_ref).split(b'\0'):
    if rec:
        main_blobs.add(rec.split(b'\t', 1)[0].split()[2].decode())
mathy = re.compile(r'\.(md|tex|py|json|lean|txt|sage|ipynb|m|jl)$')
rows = []
for n in prs:
    head = git('rev-parse', f'refs/remotes/pr/{n}').decode().strip()
    base = git('merge-base', head, main_ref).decode().strip()
    subj = git('log', '-1', '--format=%s', head).decode().strip()
    changed = []
    for rec in git('diff', '--no-renames', '--no-abbrev', '-z', '--raw', base, head).split(b':')[1:]:
        meta, _, path = rec.partition(b'\0')
        parts = meta.split()
        status, newblob = parts[4].decode().rstrip('\0'), parts[3].decode()
        path = path.rstrip(b'\0').decode('utf-8', 'replace')
        if status.startswith('D') or not mathy.search(path):
            continue
        changed.append({'path': path, 'git_blob': newblob, 'on_main_tip': newblob in main_blobs})
    off = [c for c in changed if not c['on_main_tip']]
    dirs = collections.Counter(c['path'].rsplit('/', 1)[0] if '/' in c['path'] else '(root)' for c in off)
    lean = sum(1 for c in off if c['path'].endswith('.lean'))
    rows.append({'pr': int(n), 'head': head, 'merge_base': base, 'subject': subj, 'math_type_files_changed': len(changed),
                 'already_on_main_tip': len(changed) - len(off), 'not_on_main_tip': len(off), 'lean_not_on_main': lean,
                 'top_dirs_not_on_main': dirs.most_common(8), 'files_not_on_main': off})
json.dump({'schema': 'claude54.math-open-pr-inventory.v1', 'main_ref': main_ref, 'prs': rows}, open(out, 'w'), indent=1)
for r in rows:
    print(f"#{r['pr']} {r['head'][:8]} math-files {r['math_type_files_changed']} on-main {r['already_on_main_tip']} off-main {r['not_on_main_tip']} (lean {r['lean_not_on_main']}) | {r['top_dirs_not_on_main'][:3]}")
