#!/usr/bin/env python3
"""Corpus search with provenance.
  cs.py -e REGEX [-e REGEX ...] [-i] [-F] [--max-files N] [--lines K] [--src git|custody|dropbox|pdf]
      ripgrep over every extracted text; prints, per matching file: provenance rows
      (source, id = git blob sha1 or custody sha256 or dropbox id, sha256, kind, original names)
      and up to K matching lines with line numbers.
  cs.py --prov TEXTFILE           provenance rows for one corpus text file
  cs.py --show TEXTFILE [--range A:B]   print lines A..B of a corpus text file
Dating a git blob: git -C <repo> log --all --format='%h %cI %an | %s' --find-object=<blob sha1>
Repos: main=/home/user/main ; others under scratchpad/repos/<name> (Math-full, trial, governance-, meta-framework, sandbox, query-, google-drive, Universal-Law-Workspace).
"""
import argparse, subprocess, sys
from collections import defaultdict
from pathlib import Path
C = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('-e', action='append', default=[]); ap.add_argument('-i', action='store_true'); ap.add_argument('-F', action='store_true')
ap.add_argument('--max-files', type=int, default=40); ap.add_argument('--lines', type=int, default=4)
ap.add_argument('--src', default=''); ap.add_argument('--prov'); ap.add_argument('--show'); ap.add_argument('--range', default='1:80')
a = ap.parse_args()
prov = defaultdict(list)
for l in open(C / 'index.tsv', encoding='utf-8', errors='replace'):
    r = l.rstrip('\n').split('\t')
    if len(r) >= 7 and r[5]: prov[r[5]].append(r)
def show_prov(tf):
    for r in prov.get(tf, [])[:6]:
        print(f'   [{r[0]}] id={r[1]} sha256={r[2][:16] or "-"} kind={r[3]} bytes={r[4]} names={r[6][:300]}')
    if len(prov.get(tf, [])) > 6: print(f'   (+{len(prov[tf])-6} more provenance rows)')
if a.prov: show_prov(a.prov); sys.exit()
if a.show:
    lo, hi = (int(x) for x in a.range.split(':'))
    for i, line in enumerate(open(C / a.show, encoding='utf-8', errors='replace'), 1):
        if lo <= i <= hi: print(f'{i}: {line.rstrip()[:400]}')
        if i > hi: break
    show_prov(a.show); sys.exit()
dirs = [d for d in ('git', 'custody', 'dropbox', 'pdf') if not a.src or d == a.src]
cmd = ['rg', '-l', '--no-messages'] + (['-i'] if a.i else []) + (['-F'] if a.F else [])
for e in a.e: cmd += ['-e', e]
cmd += [str(C / d) for d in dirs if (C / d).exists()]
files = subprocess.run(cmd, capture_output=True, text=True).stdout.split()
print(f'{len(files)} matching files' + (f' (showing {a.max_files})' if len(files) > a.max_files else ''))
for f in files[:a.max_files]:
    tf = str(Path(f).relative_to(C))
    print(f'== {tf}'); show_prov(tf)
    g = ['rg', '-n', '-m', str(a.lines), '--no-messages', '-M', '300'] + (['-i'] if a.i else []) + (['-F'] if a.F else [])
    for e in a.e: g += ['-e', e]
    out = subprocess.run(g + [f], capture_output=True, text=True).stdout
    for line in out.splitlines()[:a.lines]: print('   | ' + line[:300])
