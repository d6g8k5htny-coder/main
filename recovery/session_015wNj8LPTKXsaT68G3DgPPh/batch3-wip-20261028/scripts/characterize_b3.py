#!/usr/bin/env python3
"""Measure the unchanged torus and theorem-A planar checkers with their wrappers' exact arguments.

Usage: characterize_b3.py WORKTREE OUTDIR

  c8-torus-vanishing-remainder (one mode per matrix job):
      theorem_v.py --check, then theorem_v.py --mutant M for all five mutants at once.
  c8-theorem-a-constants-planar:
      normal theorem_a.py --check-full --procs 2, optimized theorem_a.py --check --procs 2,
      then normal-mode theorem_a.py --check --procs 2 --mutant M for its four mutants.

Every child is a separate process started from its packet directory with
PYTHONDONTWRITEBYTECODE=1. Exit status, stdout and stderr are kept whole. The two
families run side by side; scheduling and the generous timeout here only affect
timing, never a child's arguments. Each finished child is appended at once to
OUTDIR/<family>.jsonl, and a restarted run skips the children already recorded
there, so a container restart loses at most the children still running.
"""
import concurrent.futures
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import threading
import time

TIMEOUT = 10800
NORMAL, OPTIMIZED = ('normal', ('-B', '-S')), ('optimized', ('-B', '-O', '-S'))
TORUS = ('c8-torus-vanishing-remainder', 'frontiers/c8_torus_vanishing_remainder_20261001', 'theorem_v.py',
         ('torus-const-only', 'exp-radius', 'no-lin', 'no-om-term', 'coupling-half'))
THEOREM_A = ('c8-theorem-a-constants-planar', 'frontiers/c8_theorem_a_constants_planar_20260930', 'theorem_a.py',
             ('no-delta', 'cap-half', 'no-zfloor-correction', 'no-om-term'))


DONE = {}
LOCK = threading.Lock()
OUT = None


def run(worktree, family, directory, script, mode, flags, label, args):
    key = (family, mode, label)
    if key in DONE:
        return DONE[key]
    root = pathlib.Path(worktree, directory).resolve()
    command = [sys.executable, *flags, str(root / script), *args]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    start = time.time()
    child = subprocess.run(command, cwd=root, capture_output=True, timeout=TIMEOUT, env=env)
    record = {'family': family, 'mode': mode, 'label': label, 'args': list(args), 'returncode': child.returncode,
              'stdout': child.stdout.decode('utf-8', 'replace'), 'stderr': child.stderr.decode('utf-8', 'replace'),
              'stdout_sha256': hashlib.sha256(child.stdout).hexdigest(),
              'stderr_sha256': hashlib.sha256(child.stderr).hexdigest(),
              'started': time.strftime('%H:%M:%S', time.gmtime(start)), 'seconds': round(time.time() - start, 1)}
    with LOCK:
        with open(OUT / (family + '.jsonl'), 'a') as fh:
            fh.write(json.dumps(record, sort_keys=True) + '\n')
        DONE[key] = record
    return record


def torus(worktree):
    family, directory, script, mutants = TORUS
    records = []
    for mode, flags in (NORMAL, OPTIMIZED):
        records.append(run(worktree, family, directory, script, mode, flags, None, ('--check',)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(mutants)) as pool:
            records += list(pool.map(lambda m: run(worktree, family, directory, script, mode, flags, m,
                                                   ('--mutant', m)), mutants))
    return records


def theorem_a(worktree):
    family, directory, script, mutants = THEOREM_A
    records = [run(worktree, family, directory, script, 'normal', NORMAL[1], None, ('--check-full', '--procs', '2')),
               run(worktree, family, directory, script, 'optimized', OPTIMIZED[1], None, ('--check', '--procs', '2'))]
    for m in mutants:
        records.append(run(worktree, family, directory, script, 'normal', NORMAL[1], m,
                           ('--check', '--procs', '2', '--mutant', m)))
    return records


def main():
    global OUT
    worktree, outdir = sys.argv[1], pathlib.Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    OUT = outdir
    for family in (TORUS[0], THEOREM_A[0]):
        path = outdir / (family + '.jsonl')
        if path.exists():
            for line in path.read_text().splitlines():
                r = json.loads(line)
                DONE[(r['family'], r['mode'], r['label'])] = r
    print('resumed', len(DONE), 'recorded children', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs = {TORUS[0]: pool.submit(torus, worktree), THEOREM_A[0]: pool.submit(theorem_a, worktree)}
        results = {family: job.result() for family, job in jobs.items()}
    for family, records in results.items():
        (outdir / (family + '.json')).write_text(json.dumps(records, indent=1, sort_keys=True) + '\n')
        for r in records:
            print(family, r['mode'], r['label'], r['returncode'], r['started'], r['seconds'], len(r['stdout']),
                  len(r['stderr']), (r['stdout'].strip() or r['stderr'].strip())[:200].replace('\n', ' | '), flush=True)


if __name__ == '__main__':
    main()
