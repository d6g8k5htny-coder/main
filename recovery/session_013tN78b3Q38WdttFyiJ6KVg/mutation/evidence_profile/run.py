import os, shutil, subprocess, sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent
SRC = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(HERE)); from mutants import M
TARGET = 'tools/evidence_profile.py'
orig = (SRC / TARGET).read_text()
work = HERE / 'work'
only = set(sys.argv[2:])
def tree(dst):
    if dst.exists(): shutil.rmtree(dst)
    for rel in ('tools/evidence_profile.py', 'tests/test_evidence_profile.py'):
        (dst / rel).parent.mkdir(parents=True, exist_ok=True); shutil.copy2(SRC / rel, dst / rel)
    shutil.copytree(SRC / 'frontiers/downstream_gate_20260925', dst / 'frontiers/downstream_gate_20260925')
def test(dst, opt):
    cmd = [sys.executable, '-B'] + (['-O'] if opt else []) + ['-S', '-m', 'unittest', 'tests.test_evidence_profile']
    r = subprocess.run(cmd, cwd=dst, capture_output=True, text=True, timeout=600)
    last = [l for l in r.stderr.splitlines() if l.startswith(('OK', 'FAILED', 'Ran '))]
    return r.returncode, ' '.join(last)
rows = []
for mid, desc, old, new in [("BASE", "unmodified", None, None)] + M:
    if only and mid not in only and mid != 'BASE': continue
    dst = work / mid; tree(dst)
    if old is not None:
        n = orig.count(old)
        if n != 1:
            rows.append((mid, desc, 'BAD-PATTERN count=%d' % n, '')); print(rows[-1], flush=True); continue
        (dst / TARGET).write_text(orig.replace(old, new))
    a = test(dst, False); b = test(dst, True)
    verdict = ('baseline ' if mid == 'BASE' else '') + ('KILLED' if a[0] and b[0] else 'SURVIVED' if not a[0] and not b[0] else 'MIXED')
    rows.append((mid, desc, verdict, '%s | -O: %s' % (a[1], b[1]))); print(rows[-1], flush=True)
    shutil.rmtree(dst)
with open(HERE / 'results.tsv', 'w') as f:
    for r in rows: f.write('\t'.join(r) + '\n')
