"""Manual-review controls for main#403, gate-test variant (reviewer-private).

Same positive checks and mutants as controls403.py; each mutant must also make
`-m unittest tests.test_formal_gate` fail in both -B -S and -B -O -S modes.

usage: python3 -I controls403.py WORKTREE BASE_REF HEAD_REF PYTHON
Positive: source-only gate passes at the head; --refresh-hashes on a copy is a no-op (0 changed, bytes equal);
the two derived manifest hashes equal the SHA-256 of the head files; every other manifest field equals base.
Negative: each mutant edits a copy-in-place and must make the source-only gate exit non-zero; the worktree
is restored byte for byte after each mutant and verified at the end.
"""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

WT, BASE, HEAD, PY = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
MAN = WT / 'formal/manifest.json'
README, LANE = 'formal/README.md', 'formal/REVIEW_LANE.md'
out = []


def sha(b):
    return hashlib.sha256(b).hexdigest()


def git(*a):
    return subprocess.run(('git', '-C', str(WT)) + a, capture_output=True, check=True).stdout


def gate(*extra):
    p = subprocess.run([PY, '-B', '-S', 'tools/formal_gate_check.py', *extra], cwd=WT, capture_output=True, text=True, timeout=300)
    return p.returncode, (p.stdout + p.stderr).strip().splitlines()[-1:] if (p.stdout + p.stderr).strip() else ['']


def snapshot():
    return {p: (WT / p).read_bytes() for p in ('formal/manifest.json', README, LANE)}


orig = snapshot()
head_m = json.loads(orig['formal/manifest.json'])
base_m = json.loads(git('show', f'{BASE}:formal/manifest.json'))
# derived hashes
for p in (README, LANE):
    out.append(f"derived {p}: manifest {head_m['files'][p][:16]} file {sha(orig[p])[:16]} -> {'MATCH' if head_m['files'][p] == sha(orig[p]) else 'MISMATCH'}")
    out.append(f"  base manifest {base_m['files'][p][:16]} base file {sha(git('show', f'{BASE}:{p}'))[:16]}")
# every other field equal to base
diff_keys = sorted(k for k in set(head_m) | set(base_m) if head_m.get(k) != base_m.get(k))
file_diff = sorted(k for k in set(head_m['files']) | set(base_m['files']) if head_m['files'].get(k) != base_m['files'].get(k))
out.append(f"top-level fields differing from base: {diff_keys}")
out.append(f"files entries differing from base: {file_diff}")
# positive gate
rc, last = gate()
out.append(f"P1 source-only gate at head: rc={rc} {last[0][:110]}")
# refresh no-op on the live tree (restored afterwards regardless)
rc, last = gate('--refresh-hashes')
after = (WT / 'formal/manifest.json').read_bytes()
out.append(f"P2 --refresh-hashes: rc={rc} {last[0][:60]}; manifest bytes {'UNCHANGED' if after == orig['formal/manifest.json'] else 'CHANGED'}")
(WT / 'formal/manifest.json').write_bytes(orig['formal/manifest.json'])


def write_manifest(m):
    MAN.write_text(json.dumps(m, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def flip(h):
    return h[:-1] + ('0' if h[-1] != '0' else '1')


mutants = []
for p in (README, LANE):
    mutants.append((f'wrong hash for {p}', lambda p=p: write_manifest({**head_m, 'files': {**head_m['files'], p: flip(head_m['files'][p])}})))
    mutants.append((f"base (stale) hash for {p}", lambda p=p: write_manifest({**head_m, 'files': {**head_m['files'], p: base_m['files'][p]}})))
    mutants.append((f'one byte appended to {p}, hash not updated', lambda p=p: (WT / p).write_bytes(orig[p] + b' ')))
mutants.append(('README entry removed from files', lambda: write_manifest({**head_m, 'files': {k: v for k, v in head_m['files'].items() if k != README}})))
mutants.append(('malformed manifest (truncated JSON)', lambda: MAN.write_bytes(orig['formal/manifest.json'][:-40])))
mutants.append(('duplicate key in manifest', lambda: MAN.write_bytes(orig['formal/manifest.json'].replace(b'{', b'{"schema_version": 1,', 1))))
import re
import time


def tests(mode):
    t = time.time()
    p = subprocess.run([PY, *mode, '-m', 'unittest', 'tests.test_formal_gate'], cwd=WT, capture_output=True, text=True, timeout=600)
    lines = (p.stdout + p.stderr).strip().splitlines()
    ran = next((l for l in lines if l.startswith('Ran ')), '?')
    res = lines[-1] if lines else ''
    return p.returncode, '%s; %s (%.1fs)' % (ran, res, time.time() - t)


MODES = (('-B', '-S'), ('-B', '-O', '-S'))
for mode in MODES:
    rc, s = tests(mode)
    out.append(f"T0 head {' '.join(mode)}: rc={rc} {s}")
for name, apply in mutants:
    apply()
    for mode in MODES:
        rc, s = tests(mode)
        out.append(f"TM {name} [{' '.join(mode)}]: rc={rc} {'CAUGHT' if rc != 0 else 'SURVIVED'} | {s}")
    for p, b in orig.items():
        (WT / p).write_bytes(b)
final = snapshot()
out.append('restored: ' + ('OK' if final == orig else 'MISMATCH') + '; git status: ' + (git('status', '--porcelain').decode().strip() or 'clean'))
print('\n'.join(out))
