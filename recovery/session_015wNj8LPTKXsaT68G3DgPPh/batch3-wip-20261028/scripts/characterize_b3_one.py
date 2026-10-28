#!/usr/bin/env python3
"""Run one family of characterize_b3.py on its own (same children, same records).

Usage: characterize_b3_one.py FAMILY WORKTREE OUTDIR

It calls characterize_b3's own torus() or theorem_a(), so every child's command,
environment and record are exactly those of the two-family driver; only the
scheduling differs, which changes timing alone. Finished children recorded in
OUTDIR/<family>.jsonl are skipped on a restart, and OUTDIR/<family>.json is
written when the family completes.
"""
import json
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import characterize_b3 as c  # noqa: E402


def main():
    family, worktree, outdir = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
    outdir.mkdir(parents=True, exist_ok=True)
    c.OUT = outdir
    path = outdir / (family + '.jsonl')
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            c.DONE[(r['family'], r['mode'], r['label'])] = r
    print('resumed', len(c.DONE), 'recorded children', flush=True)
    records = {c.TORUS[0]: c.torus, c.THEOREM_A[0]: c.theorem_a}[family](worktree)
    (outdir / (family + '.json')).write_text(json.dumps(records, indent=1, sort_keys=True) + '\n')
    for r in records:
        print(family, r['mode'], r['label'], r['returncode'], r['started'], r['seconds'], len(r['stdout']),
              len(r['stderr']), (r['stdout'].strip() or r['stderr'].strip())[:200].replace('\n', ' | '), flush=True)


if __name__ == '__main__':
    main()
