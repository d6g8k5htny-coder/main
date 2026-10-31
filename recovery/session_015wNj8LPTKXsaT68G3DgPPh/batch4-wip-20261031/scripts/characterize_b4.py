#!/usr/bin/env python3
"""Measure the unchanged lifetime-difference and d = 3 Theorem A checkers with their wrappers' exact arguments.

Usage: characterize_b4.py WORKTREE OUTDIR

Both wrappers run one child at a time, with PROCS = str(max(2, os.cpu_count() or 2)):
  normal   <script> --check-full --procs PROCS
  optimized <script> --check --procs PROCS
  normal   <script> --check --procs PROCS --mutant M      for each mutant, in the wrapper's order

  c8-lifetime-difference-constant: theorem_b.py, mutants no-om-term, no-bad-set, no-det, no-torus, cap-half
  c8-theorem-a-constants-d3:       theorem_a3.py, mutants no-delta, cap-half, no-zfloor-correction, no-om-term, no-lambda2

Every child is a separate process started from its packet directory with
PYTHONDONTWRITEBYTECODE=1. Exit status, stdout and stderr are kept whole. The two
families run side by side; that and the generous timeout here affect timing only,
never a child's arguments. Each finished child is appended at once to
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

TIMEOUT = 4 * 3600
PROCS = str(max(2, os.cpu_count() or 2))
NORMAL, OPTIMIZED = ('normal', ('-B', '-S')), ('optimized', ('-B', '-O', '-S'))
FAMILIES = (
    ('c8-lifetime-difference-constant', 'frontiers/c8_lifetime_difference_constant_20261001', 'theorem_b.py',
     ('no-om-term', 'no-bad-set', 'no-det', 'no-torus', 'cap-half')),
    ('c8-theorem-a-constants-d3', 'frontiers/c8_theorem_a_constants_d3_20261001', 'theorem_a3.py',
     ('no-delta', 'cap-half', 'no-zfloor-correction', 'no-om-term', 'no-lambda2')),
)

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
              'procs': PROCS, 'cpu_count': os.cpu_count(),
              'started': time.strftime('%H:%M:%S', time.gmtime(start)), 'seconds': round(time.time() - start, 1)}
    with LOCK:
        with open(OUT / (family + '.jsonl'), 'a') as fh:
            fh.write(json.dumps(record, sort_keys=True) + '\n')
        DONE[key] = record
    return record


def sequential(worktree, spec):
    family, directory, script, mutants = spec
    records = [run(worktree, family, directory, script, 'normal', NORMAL[1], None, ('--check-full', '--procs', PROCS)),
               run(worktree, family, directory, script, 'optimized', OPTIMIZED[1], None, ('--check', '--procs', PROCS))]
    for m in mutants:
        records.append(run(worktree, family, directory, script, 'normal', NORMAL[1], m,
                           ('--check', '--procs', PROCS, '--mutant', m)))
    return records


def main():
    global OUT
    worktree, outdir = sys.argv[1], pathlib.Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    OUT = outdir
    for spec in FAMILIES:
        path = outdir / (spec[0] + '.jsonl')
        if path.exists():
            for line in path.read_text().splitlines():
                r = json.loads(line)
                DONE[(r['family'], r['mode'], r['label'])] = r
    print('resumed', len(DONE), 'recorded children; procs', PROCS, flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(FAMILIES)) as pool:
        jobs = {spec[0]: pool.submit(sequential, worktree, spec) for spec in FAMILIES}
        results = {family: job.result() for family, job in jobs.items()}
    for family, records in results.items():
        (outdir / (family + '.json')).write_text(json.dumps(records, indent=1, sort_keys=True) + '\n')
        for r in records:
            print(family, r['mode'], r['label'], r['returncode'], r['started'], r['seconds'], len(r['stdout']),
                  len(r['stderr']), (r['stdout'].strip() or r['stderr'].strip())[:200].replace('\n', ' | '), flush=True)


if __name__ == '__main__':
    main()
