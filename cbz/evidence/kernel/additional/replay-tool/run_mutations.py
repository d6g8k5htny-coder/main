#!/usr/bin/env python3
"""Run three unique-anchor validator-source mutations, with reason-bound kills.

The owned-name guard has a second module-assignment rejection behind it. Its
mutant is killed by the test requiring the specific owned-name diagnostic;
the remaining guard still rejects the malformed census. No syntax/import error
or process-launch failure counts as a mutation kill.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

MUTATIONS = (
    ('owned-names', '    if seen != expected_names:  # MUTATION_ANCHOR_OWNED_NAMES\n',
     '    if False:  # removed owned-name guard\n', 'test_wrong_owned_name'),
    ('module-assignment', '    if any(set(observed[m]) != set(declarations[m]) for m in declarations):  # MUTATION_ANCHOR_MODULE_ASSIGNMENT\n',
     '    if False:  # removed exact module-membership guard\n', 'test_wrong_module_assignment'),
    ('axioms', '            if not set(axioms) <= ALLOWED_AXIOMS:  # MUTATION_ANCHOR_AXIOMS\n',
     '            if False:  # removed fixed axiom guard\n', 'test_forbidden_axiom'),
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='new directory; never overwrites an old run')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = (root / 'replay_tool.py').read_bytes()
    text = source.decode()
    for name, old, new, test in MUTATIONS:
        if text.count(old) != 1:
            raise ValueError('mutation anchor is absent or ambiguous: ' + name)
    report = {'schema': 'cbz-replay-source-mutations-v1', 'source_sha256': sha(source),
              'test_sha256': sha((root / 'test_replay_tool.py').read_bytes()),
              'python': sys.version, 'python_executable': sys.executable,
              'threshold': '100% of the three declared validator-source mutants in each Python mode',
              'baseline': {}, 'mutants': {}, 'status': 'RUNNING'}

    def run(label, module, mode):
        environment = dict(os.environ, REPLAY_TOOL_MODULE=str(module), PYTHONDONTWRITEBYTECODE='1')
        command = [sys.executable, '-B', '-S'] + (['-O'] if mode == 'optimized' else []) + ['-m', 'unittest', '-v', 'test_replay_tool.py']
        result = subprocess.run(command, cwd=root, env=environment, capture_output=True)
        (output / (label + '-' + mode + '.stdout')).write_bytes(result.stdout)
        (output / (label + '-' + mode + '.stderr')).write_bytes(result.stderr)
        return result

    try:
        for mode in ('normal', 'optimized'):
            baseline = run('baseline', root / 'replay_tool.py', mode)
            if baseline.returncode != 0 or b'\nOK\n' not in baseline.stderr:
                raise RuntimeError('unmutated baseline suite did not pass: ' + mode)
            report['baseline'][mode] = {'exit_code': 0, 'stderr_sha256': sha(baseline.stderr)}
        with tempfile.TemporaryDirectory(prefix='cbz-replay-mutants-') as scratch:
            for name, old, new, test in MUTATIONS:
                mutant = Path(scratch) / (name + '.py')
                mutant.write_text(text.replace(old, new, 1))
                per_mode = {}
                for mode in ('normal', 'optimized'):
                    result = run(name, mutant, mode)
                    stderr = result.stderr.decode('utf-8', 'replace')
                    diagnostic = 'FAIL: ' + test + ' (test_replay_tool.TranscriptTests.' + test + ')'
                    if result.returncode != 1 or diagnostic not in stderr or 'AssertionError:' not in stderr or any(token in stderr for token in ('ImportError:', 'SyntaxError:', 'Failed to import test module')):
                        raise RuntimeError('mutant not killed by its intended assertion: ' + name + '/' + mode)
                    per_mode[mode] = {'exit_code': result.returncode, 'expected_failed_test': test,
                                      'reason_bound_kill': True, 'stderr_sha256': sha(result.stderr)}
                report['mutants'][name] = {'source_sha256': sha(mutant.read_bytes()), 'modes': per_mode}
        if (root / 'replay_tool.py').read_bytes() != source:
            raise RuntimeError('source changed during mutation run')
        report['status'] = 'PASS_ALL_DECLARED_MUTANTS_KILLED'
    except BaseException as error:
        report['status'] = 'FAIL'
        report['failure'] = repr(error)
        raise
    finally:
        (output / 'REPORT.json').write_text(json.dumps(report, indent=2) + '\n')
    print(report['status'])


if __name__ == '__main__':
    main()
