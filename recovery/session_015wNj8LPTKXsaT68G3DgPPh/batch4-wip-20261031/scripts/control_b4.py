#!/usr/bin/env python3
"""Control: the sampled --check of each batch-4 checker in optimized mode with --procs 2.

Usage: control_b4.py WORKTREE OUTDIR

Both wrappers pass --procs max(2, cpu_count); the characterization runs used the
host's count (4). This control runs the same optimized --check child with --procs
2 and records it, label 'procs-2', in OUTDIR/<family>.jsonl in the same format as
characterize_b4.py, so its stdout and stderr can be compared byte for byte with
the run-A and run-B optimized baselines. (The record's 'procs' field is the
module constant; the child's actual arguments are in 'args'.)
"""
import json
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import characterize_b4 as c  # noqa: E402


def main():
    worktree, outdir = sys.argv[1], pathlib.Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    c.OUT = outdir
    for family, directory, script, _mutants in c.FAMILIES:
        path = outdir / (family + '.jsonl')
        if path.exists():
            for line in path.read_text().splitlines():
                r = json.loads(line)
                c.DONE[(r['family'], r['mode'], r['label'])] = r
        r = c.run(worktree, family, directory, script, 'optimized', c.OPTIMIZED[1], 'procs-2',
                  ('--check', '--procs', '2'))
        print(family, r['returncode'], r['started'], r['seconds'], r['stdout_sha256'][:16], r['stderr_sha256'][:16],
              (r['stdout'].strip() or r['stderr'].strip())[:120].replace('\n', ' | '), flush=True)


if __name__ == '__main__':
    main()
