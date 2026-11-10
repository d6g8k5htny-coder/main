"""Exercise the actual workflow shell guards against synthetic child outcomes."""
import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / '.github/workflows/cbz-auditor-origin.yml'


def workflow_block():
    source = WORKFLOW.read_text()
    start = '      - name: Run both actual modes and retain each mutation observation\n'
    end = '      - name: Preserve complete observations including failures\n'
    if source.count(start) != 1 or source.count(end) != 1:
        raise ValueError('ambiguous workflow step')
    step = source.split(start, 1)[1].split(end, 1)[0]
    if step.count('        run: |\n') != 1:
        raise ValueError('missing workflow shell block')
    return textwrap.dedent(step.split('        run: |\n', 1)[1])


PYTHON_STUB = r'''#!/usr/bin/env python3
import os, pathlib, sys
if '--version' in sys.argv:
    print('Python synthetic guard fixture'); raise SystemExit(0)
origin = 'tests.test_cbz_auditor_origin_mutations' in sys.argv
key = 'CBZ_AUDITOR_ORIGIN_MUTATION_EVIDENCE_DIR' if origin else 'CBZ_AUDITOR_EVIDENCE_DIR'
count = 10 if origin else 204
if os.environ.get('CBZ_GUARD_CASE') == 'missing' and not origin:
    count -= 1
out = pathlib.Path(os.environ[key])
if os.environ.get('CBZ_GUARD_CASE') == 'origin_exit' and origin:
    (out.parent / 'origin.exit').mkdir()
if os.environ.get('CBZ_GUARD_CASE') == 'census_exit' and not origin:
    (out.parent / 'census.exit').mkdir()
for i in range(count):
    (out / ('fixture-%03d.json' % i)).write_text('{}')
print('Ran %d tests in 0.001s' % (22 if origin else 85))
if os.environ.get('CBZ_GUARD_CASE') == 'skip' and origin:
    print("test_fixture ... skipped 'synthetic'")
print('OK')
raise SystemExit(1 if os.environ.get('CBZ_GUARD_CASE') == 'failure' and origin else 0)
'''
GIT_STUB = '''#!/bin/sh
case "$1" in
  rev-parse) printf 'synthetic-source\\n' ;;
  status) if [ "$CBZ_GUARD_CASE" = dirty ]; then printf ' M source\\n'; fi ;;
  *) exit 2 ;;
esac
'''
HASH_STUB = '''#!/bin/sh
if [ "$1" = -c ] && [ "$CBZ_GUARD_CASE" = hash ]; then exit 1; fi
exec /usr/bin/sha256sum "$@"
'''


def run_block(block, variant):
    with tempfile.TemporaryDirectory(prefix='cbz-origin-workflow-guard-') as tmp:
        root = Path(tmp)
        binary = root / 'bin'
        binary.mkdir()
        for name, data in [('python', PYTHON_STUB), ('git', GIT_STUB), ('sha256sum', HASH_STUB)]:
            p = binary / name
            p.write_text(data)
            p.chmod(0o755)
        script = root / 'workflow.sh'
        script.write_text(block)
        env = dict(os.environ, PATH=str(binary) + os.pathsep + os.defpath,
                   RUNNER_TEMP=str(root), CBZ_GUARD_CASE=variant)
        return subprocess.run(['bash', str(script)], cwd=ROOT, env=env,
                              text=True, capture_output=True, timeout=30)


class AuditorOriginWorkflowGuards(unittest.TestCase):
    def test_exit_status_write_failures_are_not_green(self):
        block = workflow_block()
        for variant in ('origin_exit', 'census_exit'):
            with self.subTest(variant=variant):
                result = run_block(block, variant)
                self.assertNotEqual(result.returncode, 0, result.stderr)

    def test_actual_shell_guards_and_removals(self):
        block = workflow_block()
        baseline = run_block(block, 'good')
        self.assertEqual(baseline.returncode, 0, baseline.stderr)
        variants = (
            ('dirty', 'if [ -s "$evidence/worktree-before.txt" ]; then', 'if false; then'),
            ('failure', 'else status=$?; failed=1; fi', 'else status=$?; :; fi'),
            ('skip', "if grep -Fq ' ... skipped '", "if false && grep -Fq ' ... skipped '"),
            ('missing', '-eq 204 ] || failed=1', '-eq 203 ] || failed=1'),
            ('origin_exit', '> "$evidence/$mode/origin.exit" || failed=1',
             '> "$evidence/$mode/origin.exit" || :'),
            ('census_exit', '> "$evidence/$mode/census.exit" || failed=1',
             '> "$evidence/$mode/census.exit" || :'),
            ('hash', '"$evidence/source-sha256.txt" > "$evidence/source-readback.log" 2>&1 || failed=1',
             '"$evidence/source-sha256.txt" > "$evidence/source-readback.log" 2>&1 || :'),
        )
        for variant, old, new in variants:
            with self.subTest(variant=variant):
                self.assertEqual(block.count(old), 2 if variant == 'failure' else 1)
                rejected = run_block(block, variant)
                self.assertNotEqual(rejected.returncode, 0, variant)
                removed = run_block(block.replace(old, new, 1), variant)
                self.assertEqual(removed.returncode, 0, (variant, removed.stderr))


if __name__ == '__main__':
    unittest.main()
