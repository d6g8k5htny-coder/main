"""Run the #493 head workflow's protocol step, extracted as the test extracts it, against the AUTHENTIC packet (stdlib)."""
import subprocess, sys, os, hashlib
W = sys.argv[1]
text = open(os.path.join(W, '.github/workflows/elder-lower-review.yml')).read()
name = '      - name: Verify sources, replay both modes, reject mutants\n'
assert text.count(name) == 1
tail = text.split(name, 1)[1].split('      - ', 1)[0]
lines = tail.split('        run: |\n', 1)[1].splitlines(keepends=True)
shell = ''.join(line[10:] if line.startswith('          ') else line for line in lines)
assert shell.startswith('set -euo pipefail\n') and shell.endswith('git diff --exit-code\n')
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
p = subprocess.run(['bash', '-c', shell], cwd=W, env=env, capture_output=True, timeout=900)
print('rc', p.returncode); print('stdout', p.stdout.decode()[-600:]); print('stderr bytes', len(p.stderr), p.stderr.decode()[-400:])
root = os.path.join(W, 'reviews/d1_elder_lower_claude_20260928')
for flags in (['-B', '-S'], ['-B', '-O', '-S']):
    out = subprocess.run([sys.executable, *flags, os.path.join(root, 'elder_lower_exact_check.py')], capture_output=True, timeout=300).stdout
    print('baseline', ' '.join(flags), len(out), hashlib.sha256(out).hexdigest())
