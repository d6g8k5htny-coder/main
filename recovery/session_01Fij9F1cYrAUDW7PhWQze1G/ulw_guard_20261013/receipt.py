"""Run the original and candidate 'run' blocks of ULW ci.yml / pins.yml under the shells GitHub Actions uses.
Original step (no `shell:`): bash -e {0}.  Candidate (`shell: bash`): bash --noprofile --norc -eo pipefail {0}."""
import json, os, subprocess, sys, tempfile, textwrap, hashlib, platform
from pathlib import Path

ORIG = {
    'ci':   "python3 -m unittest discover -s tests -p 'test_*.py' -v\n",
    'pins': "python3 -m unittest discover -s tests -v\n",
}
CAND = {
    'ci': textwrap.dedent("""\
        set -euo pipefail
        python3 -m unittest discover -s tests -p 'test_*.py' -v 2>&1 | tee "$RUNNER_TEMP/discovery.log"
        grep -Eq '^Ran [1-9][0-9]* tests? in ' "$RUNNER_TEMP/discovery.log" || { echo "::error::unittest discovery collected 0 tests or returned no count"; exit 1; }
        """),
    'pins': textwrap.dedent("""\
        set -euo pipefail
        python3 -m unittest discover -s tests -v 2>&1 | tee "$RUNNER_TEMP/discovery.log"
        grep -Eq '^Ran [1-9][0-9]* tests? in ' "$RUNNER_TEMP/discovery.log" || { echo "::error::unittest discovery collected 0 tests or returned no count"; exit 1; }
        """),
}
FIXTURES = {
    'empty':        {'README.txt': 'no tests here\n'},
    'one_passing':  {'test_ok.py': 'import unittest\nclass T(unittest.TestCase):\n    def test_ok(self): self.assertTrue(True)\n'},
    'one_failing':  {'test_fail.py': 'import unittest\nclass T(unittest.TestCase):\n    def test_fail(self): self.assertEqual(1, 2)\n'},
    'import_error': {'test_bad.py': 'import module_that_does_not_exist_ulw\n'},
}

def run(script, workdir, shell_args):
    with tempfile.TemporaryDirectory() as rt, tempfile.NamedTemporaryFile('w', suffix='.sh', delete=False) as f:
        f.write(script); path = f.name
    try:
        rt = tempfile.mkdtemp()
        env = dict(os.environ, RUNNER_TEMP=rt)
        p = subprocess.run(['bash', *shell_args, path], cwd=workdir, env=env, capture_output=True, text=True)
        out = p.stdout + p.stderr
        ran = [l for l in out.splitlines() if l.startswith('Ran ')]
        return {'exit': p.returncode, 'ran': ran[-1] if ran else None,
                'error_line': any('::error::unittest discovery collected 0 tests' in l for l in out.splitlines()),
                'log_exists': (Path(rt) / 'discovery.log').exists()}
    finally:
        os.unlink(path)

def main():
    real = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    results = {'python': platform.python_version(), 'bash': subprocess.run(['bash', '--version'], capture_output=True, text=True).stdout.splitlines()[0], 'cases': []}
    targets = []
    for name, files in FIXTURES.items():
        d = Path(tempfile.mkdtemp()); (d / 'tests').mkdir()
        for fn, body in files.items(): (d / 'tests' / fn).write_text(body)
        targets.append((name, d))
    if real: targets.append(('ulw_real_tests_a152b9d1', real))
    for name, d in targets:
        for wf in ('ci', 'pins'):
            o = run(ORIG[wf], d, ['-e'])
            c = run(CAND[wf], d, ['--noprofile', '--norc', '-eo', 'pipefail'])
            results['cases'].append({'fixture': name, 'workflow': wf, 'original': o, 'candidate': c})
    print(json.dumps(results, indent=1))

if __name__ == '__main__':
    main()
