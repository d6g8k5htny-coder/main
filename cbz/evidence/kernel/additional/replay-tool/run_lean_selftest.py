#!/usr/bin/env python3
"""Finite actual-Lean controls and generated-auditor source mutations.

Every violating fixture/program is generated in a disposable temporary directory.
No mathematical source in a repository is modified. Raw execution streams and
program hashes are retained. Syntax/import/launch failures never count as kills.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import replay_tool as tool

BEGIN = '  IO.println s!"KERNEL_CLOSURE_REPLAY_BEGIN SEEDS {seeds.size} CLOSURE {selected.size}"\n'
COLLECT = '  let (_, state) := (seeds.forM (_inv.collect env)).run {}\n'
LOOKUP = '    if ci.isUnsafe || ci.isPartial then\n'
PRESENCE = ('  for name in seeds do\n'
            '    if (checked.find? name).isNone then\n'
            '      throwError "kernel replay omitted candidate constant: {name}"\n')
REPLAY = '  let checked ← fresh.toKernelEnv.replay selected\n'
FRESH = '  let fresh ← mkEmptyEnvironment 0\n'
ALLOW = '    unless allowed.contains ax do\n'
PASS = 'KERNEL_CLOSURE_STANDARD_ONLY_PASS '


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('missing or ambiguous mutation anchor: ' + repr(old))
    return text.replace(old, new, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lean', type=Path, required=True)
    parser.add_argument('--collector', type=Path, required=True)
    parser.add_argument('--collector-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    lean = args.lean.resolve()
    report = {'schema': 'cbz-replay-actual-lean-selftest-v1', 'status': 'RUNNING',
              'tool_sha256': tool.digest(Path(tool.__file__).read_bytes()),
              'selftest_sha256': tool.digest(Path(__file__).read_bytes()),
              'python_executable': sys.executable, 'python_version': sys.version,
              'python_optimized': not __debug__,
              'lean_path': str(lean), 'lean_sha256': tool.digest(lean.read_bytes()),
              'scope': 'Finite disposable fixtures testing the generated audit program; no global proof, source-tree or execution-authentication claim.',
              'threshold': '100% of six registered generated-Lean auditor source mutations',
              'executions': [], 'controls': {}, 'mutants': {}}

    def run(label, command, cwd, environment, program=None):
        result = subprocess.run([str(a) for a in command], cwd=cwd, env=environment,
                                capture_output=True, timeout=120)
        (output / (label + '.stdout')).write_bytes(result.stdout)
        (output / (label + '.stderr')).write_bytes(result.stderr)
        record = {'name': label, 'command': [str(a) for a in command], 'exit_code': result.returncode,
                  'stdout_sha256': tool.digest(result.stdout), 'stderr_sha256': tool.digest(result.stderr)}
        if program is not None:
            record['program_sha256'] = tool.digest(program.read_bytes())
        report['executions'].append(record)
        return result.returncode, result.stdout.decode('utf-8'), result.stderr.decode('utf-8')

    def reject(result, required):
        code, stdout, stderr = result
        text = stdout + stderr
        if code != 1 or stderr or required not in stdout or PASS in stdout:
            raise RuntimeError('control was not rejected for its registered reason: ' + required)
        if any(token in text for token in ('unexpected token', 'unknown identifier', 'unknown module prefix', 'failed to open file')):
            raise RuntimeError('syntax/import failure cannot count as intended rejection')

    def accepted(result):
        code, stdout, stderr = result
        if code != 0 or stderr or stdout.count(PASS) != 1:
            raise RuntimeError('mutant did not reach its intended false-acceptance marker')

    try:
        environment = dict(os.environ, LEAN_NUM_THREADS='2')
        code, version, stderr = run('version', [lean, '--version'], output, environment)
        if code or stderr or 'Lean (version 4.34.1,' not in version or 'commit 5045d0056413266e57c625dcd7c365b10e377c52' not in version:
            raise RuntimeError('exact pinned Lean 4.34.1 runtime required')
        report['lean_version'] = version.strip()
        with tempfile.TemporaryDirectory(prefix='cbz-actual-lean-replay-') as temporary:
            scratch = Path(temporary)
            safe, forbidden = scratch / 'safe', scratch / 'forbidden'
            safe.mkdir()
            forbidden.mkdir()
            source = 'namespace ReplayFixture\ndef value : Nat := 0\ntheorem witness : True := True.intro\nend ReplayFixture\n'
            for label, directory, fixture_source in (
                    ('safe', safe, source),
                    ('forbidden', forbidden, source + '\naxiom ReplayFixture.forbidden : False\n')):
                fixture = directory / 'Fixture.lean'
                fixture.write_text(fixture_source)
                env = dict(environment, LEAN_PATH=str(directory))
                code, stdout, stderr = run('compile-' + label, [lean, '--root=' + str(directory), '-j2', '-M4096', '-o', directory / 'Fixture.olean', fixture], directory, env, fixture)
                if code or stderr:
                    raise RuntimeError('fixture compilation failed: ' + label)
            program, metadata = tool.generate_program(args.collector, args.collector_sha256, ['Fixture'], ['Fixture'])
            report['generator_metadata'] = metadata
            environment['LEAN_PATH'] = str(safe)

            def execute(label, source_text, directory=safe):
                path = scratch / (label + '.lean')
                path.write_text(source_text)
                env = dict(environment, LEAN_PATH=str(directory))
                return run(label, [lean, '--root=' + str(scratch), '-j2', '-M4096', path], scratch, env, path)

            census_program = '''import Fixture
import Lean
open Lean
#eval show CoreM Unit from do
  let env := (← getEnv).setExporting false
  let some idx := env.getModuleIdx? `Fixture | throwError "fixture module absent"
  for name in env.header.moduleData[idx]!.constNames do
    IO.println s!"FIXTURE_CENSUS {name}"
'''
            code, stdout, stderr = execute('independent-module-table-census', census_program)
            names = [line.removeprefix('FIXTURE_CENSUS ') for line in stdout.splitlines() if line.startswith('FIXTURE_CENSUS ')]
            if code or stderr or not names or len(names) != len(set(names)) or stdout != ''.join('FIXTURE_CENSUS ' + name + '\n' for name in names):
                raise RuntimeError('independent module-table census failed')
            census = {'module_counts': {'Fixture': len(names)}, 'module_declarations': {'Fixture': names}}
            positive = execute('positive', program)
            code, stdout, stderr = positive
            report['positive'] = tool.validate_transcript(stdout, census, tool.digest(program.encode()), exit_code=code, stderr=stderr)

            illtyped = replace_once(program, COLLECT, '  let (_, state) := ((seeds.push `Nat.zero).forM (_inv.collect env)).run {}\n')
            illtyped = replace_once(illtyped, BEGIN, '''  let target : Name := `ReplayFixture.witness
  let some (.thmInfo original) := selected[target]?
    | throwError "fixture theorem missing"
  selected := selected.insert target (.thmInfo { original with value := mkConst `Nat.zero })
''' + BEGIN)
            reject(execute('control-illtyped-value', illtyped), "(kernel) declaration type mismatch, 'ReplayFixture.witness'")
            report['controls']['illtyped-value'] = 'REJECTED_KERNEL_TYPE_MISMATCH'

            omitted = replace_once(program, BEGIN, '  selected := selected.erase `ReplayFixture.witness\n' + BEGIN)
            reject(execute('control-omitted-owned', omitted), 'kernel replay omitted candidate constant: ReplayFixture.witness')
            report['controls']['omitted-owned'] = 'REJECTED_OWNED_PRESENCE'

            empty = replace_once(program, BEGIN, '  selected := {}\n' + BEGIN)
            reject(execute('control-empty-selection', empty), 'kernel replay omitted candidate constant:')
            report['controls']['empty-selection'] = 'REJECTED_NO_IMPORTED_ENVIRONMENT_REUSE'

            reject(execute('control-forbidden-axiom', program, forbidden), 'forbidden proof-closure axiom: ReplayFixture.forbidden')
            report['controls']['forbidden-axiom'] = 'REJECTED_FIXED_AXIOM_ALLOWLIST'

            safety_programs = {}
            for safety in ('unsafe', 'partial'):
                injected = replace_once(program, LOOKUP, '''    let ci := if name == `ReplayFixture.value then
      match ci with
      | .defnInfo value => .defnInfo { value with safety := .SAFETY }
      | _ => ci
    else ci
'''.replace('SAFETY', safety) + LOOKUP)
                reject(execute('control-' + safety, injected), 'proof closure contains unsafe/partial constant: ReplayFixture.value')
                report['controls'][safety] = 'REJECTED_EXPLICIT_SAFETY_GUARD'
                safety_programs[safety] = injected

            mutants = {
                'reuse-imported-environment': (replace_once(empty, FRESH, '  let fresh := env\n'), 'FALSE_ACCEPTED_EMPTY_SELECTION', None),
                'remove-owned-presence': (replace_once(omitted, PRESENCE, ''), 'FALSE_ACCEPTED_OMITTED_OWNED_THEOREM', None),
                'bypass-kernel-replay': (replace_once(illtyped, REPLAY, '  let _ := fresh\n  let checked := env.toKernelEnv\n'), 'FALSE_ACCEPTED_ILLTYPED_THEOREM_VALUE', None),
                'remove-axiom-guard': (replace_once(program, ALLOW, '    unless true do\n'), 'FALSE_ACCEPTED_FORBIDDEN_AXIOM_AT_LEAN_STAGE', forbidden),
                'remove-unsafe-guard': (replace_once(safety_programs['unsafe'], LOOKUP, '    if false then\n'), 'UNSAFE_DIAGNOSTIC_LOST_TO_REPLAY_SKIP', None),
                'remove-partial-guard': (replace_once(safety_programs['partial'], LOOKUP, '    if false then\n'), 'PARTIAL_DIAGNOSTIC_LOST_TO_REPLAY_SKIP', None),
            }
            for name, (mutant, reason, directory) in mutants.items():
                result = execute('mutant-' + name, mutant, directory or safe)
                if name in ('remove-unsafe-guard', 'remove-partial-guard'):
                    reject(result, 'kernel replay omitted candidate constant: ReplayFixture.value')
                    if 'proof closure contains unsafe/partial constant:' in result[1]:
                        raise RuntimeError('safety guard mutation did not change the intended behavior')
                else:
                    accepted(result)
                report['mutants'][name] = {'reason_bound_kill': True, 'reason': reason, 'program_sha256': tool.digest(mutant.encode())}
            if len(report['mutants']) != 6:
                raise RuntimeError('not all six declared source mutations were exercised')
        if tool.digest(Path(tool.__file__).read_bytes()) != report['tool_sha256'] or tool.digest(Path(__file__).read_bytes()) != report['selftest_sha256']:
            raise RuntimeError('auditor or self-test source changed during execution')
        if tool.digest(lean.read_bytes()) != report['lean_sha256'] or tool.digest(args.collector.read_bytes()) != args.collector_sha256:
            raise RuntimeError('runtime or collector source changed during execution')
        report['status'] = 'PASS_ALL_DECLARED_LEAN_CONTROLS_AND_SOURCE_MUTATIONS'
    except BaseException as error:
        report['status'] = 'FAIL'
        report['failure'] = repr(error)
        raise
    finally:
        (output / 'REPORT.json').write_text(json.dumps(report, indent=2) + '\n')
    print(report['status'])


if __name__ == '__main__':
    main()
