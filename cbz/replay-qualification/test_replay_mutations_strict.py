#!/usr/bin/env python3
"""Adverse controls for replay_mutations_strict.py and the historical harness.

Set CBZ_REPLAY_DIR to the untouched historical replay-tool directory. Run with:
    CBZ_REPLAY_DIR=... python3 -B -S test_replay_mutations_strict.py
"""
import copy
import hashlib
import importlib.util
import importlib._bootstrap_external
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest

HERE = Path(__file__).resolve().parent
HARNESS = HERE / 'replay_mutations_strict.py'
HISTORICAL = Path(os.environ['CBZ_REPLAY_DIR']).resolve()

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_candidate(text=None):
    source = HARNESS.read_text() if text is None else text
    module = types.ModuleType('candidate_harness')
    module.__file__ = str(HARNESS)
    exec(compile(source, str(HARNESS), 'exec'), module.__dict__)
    return module

candidate = load_candidate()
TARGET = 'test_replay_tool.TranscriptTests.test_wrong_owned_name'
REASON = 'AssertionError: "owned names differ" does not match "module assignment differs from complete compiled census"'
NAMES = [TARGET] + ['test_replay_tool.Other.test_' + str(i) for i in range(30)]

def good():
    return {'count': 31, 'names': NAMES.copy(), 'errors': [], 'skipped': [],
            'expected_failures': [], 'unexpected_successes': [], 'failures': []}

def run_script(script, arguments, cwd):
    environment = {k: v for k, v in os.environ.items() if not k.startswith('PYTHON')}
    return subprocess.run([sys.executable, '-I', '-B', '-S', str(script), *arguments],
                          cwd=cwd, env=environment, capture_output=True)

class ClassifierTests(unittest.TestCase):
    def test_positive_and_adverse_results(self):
        baseline = good()
        self.assertEqual(candidate.grade(baseline), NAMES)
        mutant = good()
        mutant['failures'] = [[TARGET, REASON]]
        candidate.grade(mutant, NAMES, (TARGET, REASON))
        cases = []
        bad = good(); bad['count'] = 0; cases.append((bad, None, None))
        bad = good(); bad['names'][-1] = bad['names'][0]; cases.append((bad, None, None))
        bad = good(); bad['names'][-1] = 'foreign'; cases.append((bad, NAMES, None))
        for field in ('errors', 'skipped', 'expected_failures', 'unexpected_successes'):
            bad = good(); bad[field] = ['foreign']; cases.append((bad, None, None))
        bad = good(); bad['failures'] = [['foreign', 'AssertionError: unrelated']]; cases.append((bad, None, None))
        for failures in (
            [[TARGET, 'AssertionError: unrelated']],
            [['foreign', REASON]],
            [[TARGET, REASON], ['foreign', 'AssertionError: unrelated']],
            [],
        ):
            bad = good(); bad['failures'] = failures
            cases.append((bad, NAMES, (TARGET, REASON)))
        bad = copy.deepcopy(mutant); bad['errors'] = ['foreign']; cases.append((bad, NAMES, (TARGET, REASON)))
        for observation, names, expected in cases:
            with self.subTest(observation=observation, expected=expected):
                with self.assertRaises(ValueError):
                    candidate.grade(observation, names, expected)

    def test_classifier_mutation_controls(self):
        """Each planted removal must be caught by its adverse example."""
        template = HARNESS.read_text()
        checks = []
        bad = good(); bad['count'] = 0
        checks.append(("if observation['count'] != COUNT:", bad, None, None))
        bad = good(); bad['names'][-1] = bad['names'][0]
        checks.append(("if len(observation['names']) != COUNT or len(set(observation['names'])) != COUNT:", bad, None, None))
        bad = good(); bad['names'][-1] = 'foreign'
        checks.append(("if names is not None and set(observation['names']) != set(names):", bad, NAMES, None))
        for field in ('errors', 'skipped', 'expected_failures', 'unexpected_successes'):
            bad = good(); bad[field] = ['foreign']
            checks.append(("if observation['" + field + "']:", bad, None, None))
        bad = good(); bad['failures'] = [['foreign', 'AssertionError: unrelated']]
        checks.append(("if observation['failures']:", bad, None, None))
        bad = good(); bad['failures'] = [[TARGET, 'AssertionError: unrelated']]
        checks.append(("if observation['failures'] != [list(expected)]:", bad, NAMES, (TARGET, REASON)))
        for condition, observation, names, expected in checks:
            with self.subTest(condition=condition):
                self.assertEqual(template.count(condition), 1)
                altered = template.replace(condition, 'if False:', 1)
                mutant = load_candidate(altered)
                # The control catches the removed guard by showing that the
                # corrupted classifier accepts this known-bad observation.
                mutant.grade(observation, names, expected)
                with self.assertRaises(ValueError):
                    candidate.grade(observation, names, expected)

class SourceExecutionTests(unittest.TestCase):
    def copy_historical(self, root):
        for name in ('replay_tool.py', 'test_replay_tool.py', 'run_mutations.py'):
            shutil.copyfile(HISTORICAL / name, root / name)

    def test_source_only_worker_rejects_poisoned_cache_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.copy_historical(root)
            markers = []
            for filename in ('replay_tool.py', 'test_replay_tool.py'):
                source = root / filename
                marker = root / ('cache-executed-' + filename)
                markers.append(marker)
                st = source.stat()
                poison = source.read_text() + '\nopen(' + repr(str(marker)) + ', "w").write("poisoned")\n'
                code = compile(poison, str(source), 'exec')
                for opt in ('', '1'):
                    cache = Path(importlib.util.cache_from_source(str(source), optimization=opt))
                    cache.parent.mkdir(exist_ok=True)
                    cache.write_bytes(importlib._bootstrap_external._code_to_timestamp_pyc(code, int(st.st_mtime), st.st_size))
            for mode in ('normal', 'optimized'):
                output = root / (mode + '.json')
                command = [sys.executable, '-I', '-B', '-S']
                if mode == 'optimized':
                    command.append('-O')
                command += [str(HARNESS), '--worker', str(root / 'replay_tool.py'),
                            str(root / 'test_replay_tool.py'), digest(root / 'replay_tool.py'),
                            digest(root / 'test_replay_tool.py'), str(output)]
                result = subprocess.run(command, cwd=root, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))
                candidate.grade(json.loads(output.read_text()))
                self.assertFalse(any(marker.exists() for marker in markers), 'poisoned project bytecode ran')
            # Prove that the seeded caches are actually usable by the original
            # importlib loader, rather than merely invalid or ignored.
            importer = (
                "import importlib.util,os,sys; "
                "os.environ['REPLAY_TOOL_MODULE']=sys.argv[2]; "
                "s=importlib.util.spec_from_file_location('probe',sys.argv[1]); "
                "m=importlib.util.module_from_spec(s); s.loader.exec_module(m)"
            )
            env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHON')}
            for filename in ('replay_tool.py', 'test_replay_tool.py'):
                result = subprocess.run([sys.executable, '-I', '-B', '-S', '-c', importer,
                                         str(root / filename), str(root / 'replay_tool.py')],
                                        cwd=root, env=env, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))
            self.assertTrue(all(marker.exists() for marker in markers), 'cache poisoning fixture was ineffective')

    def test_historical_harness_accepts_wrong_named_assertions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.copy_historical(root)
            testfile = root / 'test_replay_tool.py'
            text = testfile.read_text()
            for mutant, _, _, name, _ in candidate.MUTATIONS:
                anchor = '    def ' + name + '(self):\n'
                self.assertEqual(text.count(anchor), 1)
                injected = anchor + "        if Path(os.environ.get('REPLAY_TOOL_MODULE', '')).stem == " + repr(mutant) + ":\n            self.fail('unrelated injected assertion')\n"
                text = text.replace(anchor, injected, 1)
            testfile.write_text(text)
            output = root / 'legacy-output'
            result = run_script(root / 'run_mutations.py', ['--output', str(output)], root)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))
            report = json.loads((output / 'REPORT.json').read_text())
            self.assertEqual(report['status'], 'PASS_ALL_DECLARED_MUTANTS_KILLED')
            for mutant, _, _, _, _ in candidate.MUTATIONS:
                for mode in ('normal', 'optimized'):
                    stderr = (output / (mutant + '-' + mode + '.stderr')).read_text()
                    self.assertIn('AssertionError: unrelated injected assertion', stderr)

    def test_historical_harness_accepts_foreign_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.copy_historical(root)
            testfile = root / 'test_replay_tool.py'
            text = testfile.read_text()
            anchor = 'class TranscriptTests(unittest.TestCase):\n'
            self.assertEqual(text.count(anchor), 1)
            injected = anchor + """    def test_foreign_error(self):
        if Path(os.environ.get('REPLAY_TOOL_MODULE', '')).stem != 'replay_tool':
            raise RuntimeError('foreign error')
"""
            testfile.write_text(text.replace(anchor, injected, 1))
            output = root / 'legacy-output'
            result = run_script(root / 'run_mutations.py', ['--output', str(output)], root)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))
            self.assertEqual(json.loads((output / 'REPORT.json').read_text())['status'], 'PASS_ALL_DECLARED_MUTANTS_KILLED')
            for mutant, _, _, _, _ in candidate.MUTATIONS:
                for mode in ('normal', 'optimized'):
                    stderr = (output / (mutant + '-' + mode + '.stderr')).read_text()
                    self.assertIn('RuntimeError: foreign error', stderr)
                    self.assertIn('errors=1', stderr)

if __name__ == '__main__':
    unittest.main()
