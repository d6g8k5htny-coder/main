#!/usr/bin/env python3
"""Independent, source-only replay mutation harness. See accompanying tests."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest

MUTATIONS = (
    ('owned-names', '    if seen != expected_names:  # MUTATION_ANCHOR_OWNED_NAMES\n',
     '    if False:  # removed owned-name guard\n', 'test_wrong_owned_name',
     'AssertionError: "owned names differ" does not match "module assignment differs from complete compiled census"'),
    ('module-assignment', '    if any(set(observed[m]) != set(declarations[m]) for m in declarations):  # MUTATION_ANCHOR_MODULE_ASSIGNMENT\n',
     '    if False:  # removed exact module-membership guard\n', 'test_wrong_module_assignment',
     'AssertionError: ValueError not raised'),
    ('axioms', '            if not set(axioms) <= ALLOWED_AXIOMS:  # MUTATION_ANCHOR_AXIOMS\n',
     '            if False:  # removed fixed axiom guard\n', 'test_forbidden_axiom',
     'AssertionError: ValueError not raised'),
)
COUNT = 31
SOURCE_PIN = 'b97bee53a590651b83d68056c090ba70e8a055157e65d2bd3fcc5cd0219840b3'
TEST_PIN = 'a07eaab8b6fd40a911d313197fae688927ebb3e1d74644628e4888a7e6505fc0'
LOADER = """spec = importlib.util.spec_from_file_location('tested_replay_tool', MODULE)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)
"""
SOURCE_LOADER = """tool = __import__('types').ModuleType('tested_replay_tool')
tool.__file__ = str(MODULE)
exec(compile(_cbz_verified_target_bytes, str(MODULE), 'exec'), tool.__dict__)
"""

def sha(data):
    return hashlib.sha256(data).hexdigest()

def grade(observation, names=None, expected=None):
    """Reject everything except exactly the expected structured test outcome."""
    if observation['count'] != COUNT:
        raise ValueError('wrong count')
    if len(observation['names']) != COUNT or len(set(observation['names'])) != COUNT:
        raise ValueError('test names not unique or complete')
    if names is not None and set(observation['names']) != set(names):
        raise ValueError('test identities changed')
    if observation['errors']:
        raise ValueError('unexpected errors')
    if observation['skipped']:
        raise ValueError('unexpected skips')
    if observation['expected_failures']:
        raise ValueError('unexpected expected failures')
    if observation['unexpected_successes']:
        raise ValueError('unexpected successes')
    if expected is None:
        if observation['failures']:
            raise ValueError('baseline failures')
    else:
        if observation['failures'] != [list(expected)]:
            raise ValueError('wrong mutant failure or diagnostic')
    return observation['names']

class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.names = []
    def startTest(self, test):
        self.names.append(test.id())
        super().startTest(test)

def worker(target, test_path, expected_target, expected_test, result_path):
    target, test_path, result_path = map(Path, (target, test_path, result_path))
    target_bytes, test_bytes = target.read_bytes(), test_path.read_bytes()
    if sha(target_bytes) != expected_target or sha(test_bytes) != expected_test:
        raise ValueError('source hash mismatch')
    test_text = test_bytes.decode('utf-8')
    if test_text.count(LOADER) != 1:
        raise ValueError('test loader missing or ambiguous')
    adapted = test_text.replace(LOADER, SOURCE_LOADER, 1)
    os.environ['REPLAY_TOOL_MODULE'] = str(target)
    module = types.ModuleType('test_replay_tool')
    module.__file__ = str(test_path)
    module._cbz_verified_target_bytes = target_bytes
    sys.modules[module.__name__] = module
    exec(compile(adapted, str(test_path), 'exec'), module.__dict__)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    result = unittest.TextTestRunner(verbosity=2, resultclass=RecordingResult).run(suite)
    observation = {
        'count': result.testsRun, 'names': result.names,
        'adapted_test_sha256': sha(adapted.encode('utf-8')), 'target_sha256': sha(target_bytes),
        'errors': [t.id() for t, _ in result.errors],
        'skipped': [t.id() for t, _ in result.skipped],
        'expected_failures': [t.id() for t, _ in result.expectedFailures],
        'unexpected_successes': [t.id() for t in result.unexpectedSuccesses],
        'failures': [[t.id(), trace.rstrip().splitlines()[-1]] for t, trace in result.failures],
    }
    if sha(target.read_bytes()) != expected_target or sha(test_path.read_bytes()) != expected_test:
        raise ValueError('source changed during run')
    result_path.write_text(json.dumps(observation, indent=2) + '\n')

def main():
    if len(sys.argv) == 7 and sys.argv[1] == '--worker':
        worker(*sys.argv[2:])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--test-sha256', required=True)
    args = parser.parse_args()
    root, out = args.directory.resolve(), args.output.resolve()
    source, test = root / 'replay_tool.py', root / 'test_replay_tool.py'
    source_bytes, test_bytes = source.read_bytes(), test.read_bytes()
    if args.source_sha256 != SOURCE_PIN or args.test_sha256 != TEST_PIN:
        raise ValueError('only the pinned historical source and test are in scope')
    if sha(source_bytes) != args.source_sha256 or sha(test_bytes) != args.test_sha256:
        raise ValueError('expected input hashes do not match')
    original = source_bytes.decode('utf-8')
    for name, old, _, _, _ in MUTATIONS:
        if original.count(old) != 1:
            raise ValueError('missing or ambiguous anchor: ' + name)
    out.mkdir(parents=True, exist_ok=False)
    report = {'status': 'RUNNING', 'source_sha256': sha(source_bytes),
              'test_sha256': sha(test_bytes), 'python': sys.version,
              'harness_sha256': sha(Path(__file__).read_bytes()), 'runs': {}}
    def run(label, mode, target, target_hash, expected, names):
        result_path = out / (label + '-' + mode + '.json')
        command = [sys.executable, '-I', '-B', '-S']
        if mode == 'optimized':
            command.append('-O')
        command += [str(Path(__file__).resolve()), '--worker', str(target), str(test),
                    target_hash, args.test_sha256, str(result_path)]
        environment = {k: v for k, v in os.environ.items()
                       if not k.startswith('PYTHON') and k != 'REPLAY_TOOL_MODULE'}
        try:
            run_result = subprocess.run(command, cwd=root, env=environment, capture_output=True, timeout=120)
        except subprocess.TimeoutExpired as error:
            (out / (label + '-' + mode + '.stdout')).write_bytes(error.stdout or b'')
            (out / (label + '-' + mode + '.stderr')).write_bytes(error.stderr or b'')
            raise RuntimeError('worker timed out: ' + label + '/' + mode) from error
        (out / (label + '-' + mode + '.stdout')).write_bytes(run_result.stdout)
        (out / (label + '-' + mode + '.stderr')).write_bytes(run_result.stderr)
        if run_result.returncode != 0:
            raise ValueError('worker failed: ' + label + '/' + mode)
        observation = json.loads(result_path.read_text())
        ids = grade(observation, names, expected)
        report['runs'][label + '-' + mode] = {
            'target_sha256': target_hash, 'result_sha256': sha(result_path.read_bytes()),
            'stdout_sha256': sha(run_result.stdout), 'stderr_sha256': sha(run_result.stderr)}
        return ids
    try:
        baselines = {mode: run('baseline', mode, source, args.source_sha256, None, None)
                     for mode in ('normal', 'optimized')}
        if set(baselines['normal']) != set(baselines['optimized']):
            raise ValueError('baseline identities vary by mode')
        with tempfile.TemporaryDirectory(prefix='cbz-source-mutations-') as tmp:
            for name, old, new, test_name, reason in MUTATIONS:
                mutant = Path(tmp) / (name + '.py')
                mutant.write_bytes(original.replace(old, new, 1).encode('utf-8'))
                mutant_hash = sha(mutant.read_bytes())
                expected = ('test_replay_tool.TranscriptTests.' + test_name, reason)
                for mode in ('normal', 'optimized'):
                    run(name, mode, mutant, mutant_hash, expected, baselines[mode])
        if source.read_bytes() != source_bytes or test.read_bytes() != test_bytes:
            raise ValueError('historical source changed')
        report['status'] = 'PASS_THREE_DECLARED_MUTATIONS'
    except BaseException as exc:
        report['status'] = 'FAIL'
        report['failure'] = repr(exc)
        raise
    finally:
        (out / 'REPORT.json').write_text(json.dumps(report, indent=2) + '\n')
    print(report['status'])

if __name__ == '__main__':
    main()
