#!/usr/bin/env python3
"""Resolution step for tranche 9: mark object files whose bytes are at the Math- main tip. pr_inventory.py compared abbreviated
blob ids from `git diff --raw` with main's full ids, so no file was ever found on main and the objects the workflows classified
include such files. A file whose full blob (at its object's commit) equals a file of main's tree at MAIN_REF is a byte copy; its
disposition is that main file's.

Usage: python3 -I -B resolve_main_copies.py MATH_CLONE MAIN_REF FACTS_DIR OUT_DIR

Reads pbatch<N>.json from FACTS_DIR (as the workflows saw them) and writes the same objects to OUT_DIR, with those files moved
from "files" to "copies" (each with its full blob and copy_of: every main path holding the bytes, as Math-@MAIN_REF:<path>).
Nothing else changes."""
import glob, json, os, subprocess, sys
clone, main_ref, facts_dir, out_dir = sys.argv[1:5]


def git(*a):
    return subprocess.run(['git', '-C', clone, *a], capture_output=True, check=True).stdout


main_ref = git('rev-parse', main_ref).decode().strip()
at_main = {}
for rec in git('ls-tree', '-r', '-z', main_ref).split(b'\0'):
    if rec:
        meta, p = rec.split(b'\t', 1)
        at_main.setdefault(meta.split()[2].decode(), []).append(p.decode('utf-8', 'replace'))
os.makedirs(out_dir, exist_ok=True)
moved = total = 0
for path in sorted(glob.glob(os.path.join(facts_dir, 'pbatch*.json'))):
    d = json.load(open(path))
    for o in d['objects']:
        keep, copies = [], []
        for f in o['files']:
            total += 1
            full = git('rev-parse', f'{o["commit"]}:{f["path"]}').decode().strip()
            if not full.startswith(f['git_blob']):
                raise SystemExit(f'{o["object"]}: {f["path"]} blob {full} does not extend {f["git_blob"]}')
            if full in at_main:
                copies.append(dict(f, git_blob_full=full,
                                   copy_of=[f'd6g8k5htny-coder/Math-@{main_ref}:{p}' for p in sorted(at_main[full])]))
            else:
                keep.append(f)
        o['files'], o['copies'] = keep, copies
        moved += len(copies)
    with open(os.path.join(out_dir, os.path.basename(path)), 'x') as f:
        json.dump(d, f, indent=1)
print(f'{moved} of {total} object files resolved as byte copies of Math- main files at {main_ref[:8]}')
