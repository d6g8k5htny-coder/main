import os, subprocess, sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent
WT = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(HERE)); from mutants import M
TARGET = WT / 'tools/retrofit_record_check.py'
orig = TARGET.read_text()
ids = (HERE / 'ids.txt').read_text().split()
env = dict(os.environ, PATH=str(HERE / 'shim') + os.pathsep + os.environ['PATH'])
only = set(sys.argv[2:])
def test(opt):
    cmd = [sys.executable, '-B'] + (['-O'] if opt else []) + ['-S', '-m', 'unittest', *ids]
    r = subprocess.run(cmd, cwd=WT, capture_output=True, text=True, timeout=900, env=env)
    last = [l for l in r.stderr.splitlines() if l.startswith(('OK', 'FAILED', 'Ran '))]
    return r.returncode, ' '.join(last)
rows = []
try:
    for mid, desc, old, new in [("BASE", "unmodified", None, None)] + M:
        if only and mid not in only and mid != 'BASE': continue
        TARGET.write_text(orig)
        if old is not None:
            n = orig.count(old)
            if n != 1:
                rows.append((mid, desc, 'BAD-PATTERN count=%d' % n, '')); print(rows[-1], flush=True); continue
            TARGET.write_text(orig.replace(old, new))
        a = test(False); b = test(True)
        verdict = ('baseline ' if mid == 'BASE' else '') + ('KILLED' if a[0] and b[0] else 'SURVIVED' if not a[0] and not b[0] else 'MIXED')
        rows.append((mid, desc, verdict, '%s | -O: %s' % (a[1], b[1]))); print(rows[-1], flush=True)
finally:
    TARGET.write_text(orig)
out = HERE / ('results-%s.tsv' % ('all' if not only else 'subset'))
with open(out, 'w') as f:
    for r in rows: f.write('\t'.join(r) + '\n')
