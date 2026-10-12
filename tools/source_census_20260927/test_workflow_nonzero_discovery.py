"""Source-bound subprocess controls for source-census zero-test discovery.

The fixtures are disposable and entirely synthetic. The current workflow is
read, not executed against source-census project inputs; only the extracted
Bash recipe runs against generated tiny unittest fixtures. No theorem or
scientific-status claim follows from these checks.
"""

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / '.github/workflows/source-census.yml'
OLD_GIT_BLOB = '4c43edea2e0c277779305622853b691d77646d5d'
NEW_GIT_BLOB = '0d01c1e23ebd0dd5c03423a999a99b68b6b8798d'
RUN_MARKER = '        run: |\n'
OLD_BLOCK = (
    "python -B -S -m unittest discover -s tools/source_census_20260927 -p 'test_*.py' -v\n"
    "python -B -O -S -m unittest discover -s tools/source_census_20260927 -p 'test_*.py' -v\n"
)


def git_blob(text):
    data = text.encode('utf-8')
    return hashlib.sha1(b'blob ' + str(len(data)).encode('ascii') + b'\0' + data).hexdigest()


def source_blocks():
    """Authenticate the complete old and new YAML before running exact Bash.

    Do not invent a simplified old block: the original source is recovered by
    reversing the candidate's only changed run block, then Git-blob checked.
    """
    candidate = WORKFLOW.read_text(encoding='utf-8')
    if git_blob(candidate) != NEW_GIT_BLOB or candidate.count(RUN_MARKER) != 1:
        raise AssertionError('source-census candidate bytes changed')
    prefix, indented = candidate.split(RUN_MARKER)
    lines = indented.splitlines(keepends=True)
    if not lines or any(not line.startswith('          ') for line in lines):
        raise AssertionError('unexpected source-census run block')
    current = ''.join(line[10:] for line in lines)
    if (current.count(' -m unittest discover ') != 2 or
            current.count('grep -Eq ') != 2 or
            not current.startswith('set -euo pipefail\n')):
        raise AssertionError('candidate no longer has exact two-mode discovery guard')
    original = prefix + RUN_MARKER + ''.join('          ' + line for line in OLD_BLOCK.splitlines(keepends=True))
    if git_blob(original) != OLD_GIT_BLOB:
        raise AssertionError('reconstructed original workflow identity changed')
    return OLD_BLOCK, current


def fixture_source(kind):
    if kind == 'empty':
        return None
    if kind == 'pass':
        expression = 'self.assertEqual(1, 1)'
        guard = 'True'
    elif kind == 'fail':
        expression = 'self.assertEqual(1, 2)'
        guard = 'True'
    elif kind == 'normal_only':
        expression = 'self.assertEqual(1, 1)'
        guard = '__debug__'
    elif kind == 'optimized_only':
        expression = 'self.assertEqual(1, 1)'
        guard = 'not __debug__'
    elif kind == 'optimized_fail':
        expression = 'self.assertTrue(__debug__)'
        guard = 'True'
    else:
        raise ValueError('unknown fixture: ' + kind)
    return ('import unittest\n'
            'if ' + guard + ':\n'
            '    class Synthetic(unittest.TestCase):\n'
            '        def test_control(self):\n'
            '            ' + expression + '\n')


def run_recipe(script, fixture):
    """Run exact workflow Bash with real Python and isolated invented inputs."""
    with tempfile.TemporaryDirectory(prefix='census-discovery-contract-') as td:
        root = Path(td)
        test_dir = root / 'tools/source_census_20260927'
        test_dir.mkdir(parents=True)
        payload = fixture_source(fixture)
        if payload is not None:
            (test_dir / 'test_synthetic.py').write_text(payload, encoding='utf-8')
        run_temp = root / 'runner-temp'
        run_temp.mkdir()
        bindir = root / 'bin'
        bindir.mkdir()
        wrapper = bindir / 'python'
        wrapper.write_text('#!/bin/sh\nexec "$FIXTURE_PYTHON" "$@"\n', encoding='utf-8')
        wrapper.chmod(0o755)
        env = {key: value for key, value in os.environ.items()
               if not key.startswith('GIT_')}
        env.update(PATH=str(bindir) + os.pathsep + env.get('PATH', ''),
                   FIXTURE_PYTHON=sys.executable,
                   RUNNER_TEMP=str(run_temp),
                   PYTHONDONTWRITEBYTECODE='1')
        proc = subprocess.run(['bash', '--noprofile', '--norc', '-e', '-o', 'pipefail', '-c', script],
                              cwd=root, env=env, capture_output=True, timeout=20)
        return (proc.returncode,
                proc.stdout + proc.stderr,
                ((run_temp / 'source-census-normal.log').read_bytes()
                 if (run_temp / 'source-census-normal.log').exists() else b''),
                ((run_temp / 'source-census-optimized.log').read_bytes()
                 if (run_temp / 'source-census-optimized.log').exists() else b''))


class SourceCensusNonzeroDiscoveryTests(unittest.TestCase):
    def test_original_and_candidate_git_blob_authentication(self):
        original, current = source_blocks()
        self.assertEqual(original.count(' -m unittest discover '), 2)
        self.assertEqual(current.count(' -m unittest discover '), 2)
        self.assertIn('source-census-normal.log', current)
        self.assertIn('source-census-optimized.log', current)
        self.assertNotIn('tee ', original)

    def test_positive_and_failing_real_unittest_controls(self):
        # These cases work with any supported Python version; 3.11-specific
        # empty-discovery evidence is the independently gated method below.
        original, candidate = source_blocks()
        for kind, expected in (('pass', 0), ('fail', 1), ('optimized_fail', 1)):
            for source_name, script in (('original', original), ('candidate', candidate)):
                with self.subTest(fixture=kind, source=source_name):
                    code, stream, normal, optimized = run_recipe(script, kind)
                    self.assertEqual(code == 0, expected == 0,
                                     (source_name, kind, code, stream[-400:]))
                    self.assertIn(b'Ran 1 test in ', stream)
                    if expected == 0:
                        self.assertIn(b'OK', stream)
                        if source_name == 'candidate':
                            self.assertIn(b'Ran 1 test in ', normal)
                            self.assertIn(b'Ran 1 test in ', optimized)
                    elif kind == 'fail':
                        self.assertIn(b'FAILED (failures=1)', stream)
                    else:
                        self.assertIn(b'FAILED (failures=1)', stream)

    @unittest.skipUnless(sys.version_info[:2] == (3, 11),
                         'empty-suite contract requires actual hosted CPython3.11')
    def test_actual_python311_zero_in_each_mode_old_vs_candidate(self):
        original, candidate = source_blocks()
        # No mocking of unittest or its count. With CPython3.11 an empty suite
        # prints Ran 0 tests / OK and exits 0; only the candidate rejects it.
        for kind in ('empty', 'normal_only', 'optimized_only'):
            for source_name, script in (('original', original), ('candidate', candidate)):
                with self.subTest(fixture=kind, source=source_name):
                    code, stream, normal, optimized = run_recipe(script, kind)
                    self.assertEqual(code == 0, source_name == 'original',
                                     (source_name, kind, code, stream[-450:]))
                    self.assertIn(b'Ran 0 tests in ', stream)
                    if source_name == 'candidate':
                        self.assertIn(b'::error::source-census ', stream)
                        self.assertEqual(bool(normal), True)
                        if kind == 'normal_only':
                            self.assertIn(b'Ran 1 test in ', normal)
                            self.assertIn(b'Ran 0 tests in ', optimized)
                        else:
                            # Original normal-mode zero stops the repaired shell.
                            self.assertIn(b'Ran 0 tests in ', normal)
                            self.assertFalse(optimized)
                    else:
                        self.assertFalse(normal)
                        self.assertFalse(optimized)
