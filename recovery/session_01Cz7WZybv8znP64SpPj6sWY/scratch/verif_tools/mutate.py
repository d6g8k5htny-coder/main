import subprocess, sys, pathlib, re
side = pathlib.Path(sys.argv[1])
target = side / sys.argv[2]
old, new = sys.argv[3], sys.argv[4]
orig = target.read_text(encoding='utf-8')
assert orig.count(old) >= 1, 'pattern not found: ' + old
target.write_text(orig.replace(old, new, 1), encoding='utf-8')
try:
    flags = ['-B', '-S'] + (['-O'] if len(sys.argv) > 5 and sys.argv[5] == 'O' else [])
    run = subprocess.run([sys.executable, *flags, str(side / 'test_torus_filtration.py')], capture_output=True, text=True, timeout=900)
    tail = run.stderr.strip().splitlines()
    fails = [l for l in tail if l.startswith('FAIL:') or l.startswith('ERROR:')]
    names = sorted(set(re.sub(r" \(.*", "", l) for l in fails)); print("exit", run.returncode, "|", tail[-1] if tail else "", "|", names)
finally:
    target.write_text(orig, encoding='utf-8')
