#!/usr/bin/env python3
"""Isolated mutation checks. Errors/skips/empty tests are never mutant kills."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent


def classify(value, code):
    if (code not in (0, 1) or not value.get('tests_run') or value.get('errors')
            or value.get('skipped') or value.get('expected_failures')
            or value.get('unexpected_successes')):
        return 'ERROR'
    if code == 1 and value.get('assertion_failures', 0) > 0 and not value.get('successful'):
        return 'KILLED'
    if code == 0 and value.get('successful') and value.get('assertion_failures') == 0:
        return 'SURVIVED'
    return 'ERROR'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def definitions():
    wrapper = 'test_kernel_replay.WrapperTests'
    real = 'test_real_lean.RealLeanTests.'
    items = [
        ('empty-coverage', 'kernel_replay.py', 'if not selections:', 'if False:', wrapper),
        ('allow-artifact-symlink', 'kernel_replay.py',
         "    if not p.is_absolute() or p.is_symlink() or canonical != p:",
         '    if False:', 'test_real_lean.RealLeanTests.test_real_private_body_cannot_hide_behind_olean_symlink'),
        ('ignore-source-binding', 'kernel_replay.py', "        check_binding(entry['source'])", '        pass', wrapper),
        ('ignore-compiled-parts-binding', 'kernel_replay.py', "if parts_of(olean) != entry['parts']:", 'if False:', wrapper),
        ('ignore-tool-binding', 'kernel_replay.py', "if m['tool'] != tool_bindings(m['sysroot']):", 'if False:', wrapper),
        ('ignore-nonzero-exit', 'kernel_replay.py', "result['returncode'] == 0 and result['stdout'].splitlines().count(marker) == 1", "result['stdout'].splitlines().count(marker) == 1", wrapper),
        ('ignore-completion-marker', 'kernel_replay.py', "result['returncode'] == 0 and result['stdout'].splitlines().count(marker) == 1", "result['returncode'] == 0", wrapper),
        ('ignore-timeout', 'kernel_replay.py', 'p.wait(timeout=timeout_seconds)', 'p.wait(timeout=None)', wrapper),
        ('ignore-memory-limit', 'kernel_replay.py', 'preexec_fn=limits(memory_mib)', 'preexec_fn=None', wrapper),
        ('truncate-module-coverage', 'kernel_replay.py', "for entry in m['modules']:\n            validate_manifest(m)", "for entry in m['modules'][:1]:\n            validate_manifest(m)", wrapper),
        ('concurrent-module-processes', 'kernel_replay.py',
         "        for entry in m['modules']:\n            validate_manifest(m)\n            report['results'].append(run_one(m, entry, memory_mib, timeout_seconds))\n            if report['results'][-1].get('cleanup', {}).get('reaped') is False:\n                break  # Do not start another process while an earlier PID remains unreaped.",
         "        from concurrent.futures import ThreadPoolExecutor\n        with ThreadPoolExecutor(max_workers=2) as pool:\n            report['results'] = list(pool.map(lambda entry: run_one(m, entry, memory_mib, timeout_seconds), m['modules']))",
         wrapper + '.test_one_child_at_a_time_and_exact_coverage'),
        ('skip-kernel-replay', 'ExactReplay.lean', '  replayFromImports target', '  pure ()', real + 'test_public_self_reference_rejected'),
        ('prefix-expansion', 'ExactReplay.lean', '  replayFromImports target',
         '  let sp ← searchPathRef.get\n  for path in (← SearchPath.findAllWithExt sp "olean") do\n    if let some name ← searchModuleNameOfFileName path sp then\n      if target.isPrefixOf name then replayFromImports name',
         real + 'test_exact_module_no_prefix_glob'),
        ('untrusted-checker-search-path', 'kernel_replay.py',
         "env['LEAN_PATH'] = os.pathsep.join([str(Path(m['sysroot']) / 'lib/lean'), *m['lean_path']])",
         "env['LEAN_PATH'] = os.pathsep.join(m['lean_path'])", real + 'test_user_lean_path_cannot_shadow_checker'),
    ]
    return items


def resource_definitions():
    wrapper = 'test_kernel_replay.WrapperTests.'
    return [
        ('resource-ignore-memory-upper-bound', 'kernel_replay.py',
         'not 64 <= memory_mib <= memory_mib_max', 'memory_mib < 64',
         wrapper + 'test_resource_memory_upper_bounds_before_spawn'),
        ('resource-ignore-inherited-memory-hard-limit', 'kernel_replay.py',
         'if hard_memory != resource.RLIM_INFINITY:', 'if False:',
         wrapper + 'test_resource_memory_hard_limit_before_spawn'),
        ('resource-ignore-timeout-upper-bound', 'kernel_replay.py',
         'not 0 < timeout_seconds <= MAX_TIMEOUT_SECONDS', 'not 0 < timeout_seconds',
         wrapper + 'test_resource_timeout_upper_bounds_before_spawn'),
        ('resource-accept-boolean-timeout', 'kernel_replay.py',
         'type(timeout_seconds) not in (int, float)', 'type(timeout_seconds) not in (int, float, bool)',
         wrapper + 'test_resource_boolean_arguments_rejected'),
        ('resource-drop-process-exception-normalization', 'kernel_replay.py',
         'PROCESS_ERRORS = (OSError, subprocess.SubprocessError, OverflowError, ValueError)',
         'PROCESS_ERRORS = (OSError,)',
         wrapper + 'test_resource_spawn_exceptions_are_structured_errors'),
        ('resource-unbounded-cleanup-wait', 'kernel_replay.py',
         'p.wait(timeout=CLEANUP_TIMEOUT_SECONDS)', 'p.wait()',
         wrapper + 'test_resource_unreapable_child_cleanup_is_bounded'),
        ('resource-skip-abnormal-cleanup', 'kernel_replay.py',
         "if p is not None and result['status'] != 'FINISHED':", 'if False:',
         wrapper + 'test_resource_unreapable_child_cleanup_is_bounded'),
        ('resource-overwrite-error-with-success', 'kernel_replay.py',
         "    if result['status'] == 'FINISHED':", "    if result['status'] != 'TIMEOUT':",
         wrapper + 'test_resource_wait_errors_cannot_become_success'),
        ('resource-continue-after-unreaped-child', 'kernel_replay.py',
         "if report['results'][-1].get('cleanup', {}).get('reaped') is False:", 'if False:',
         wrapper + 'test_resource_unreaped_child_stops_further_modules'),
    ]


def derived_driver_mutations(lean):
    upstream = (Path(lean).resolve().parent.parent / 'src/lean/LeanChecker.lean').read_text()
    start = upstream.index('unsafe def replayFromImports ')
    end = upstream.index('unsafe def replayFromFresh ', start)
    routine = upstream[start:end]
    # These intentionally unsound adapters exist only inside isolated mutants.
    # The copied fragment retains the upstream Apache-2.0 provenance in the
    # driver's file header; production calls the installed implementation.
    exported = routine.replace('replayFromImports', 'cbzExportedOnly', 1)
    a, b = exported.index('  let mut fnames'), exported.index('  let parts')
    exported = exported[:a] + '  let fnames := #[mFile]\n' + exported[b:]
    preload = routine.replace('replayFromImports', 'cbzPreloaded', 1)
    preload = preload.replace('importModulesCore mod.imports', 'importModulesCore #[{ module }]')
    preload = preload.replace('finalizeImport s mod.imports', 'finalizeImport s #[{ module }]')
    original = (ROOT / 'ExactReplay.lean').read_text()
    return [
        ('exported-part-only', 'ExactReplay.lean', original,
         original.replace('open Lean\n', 'open Lean\n\n' + exported).replace('  replayFromImports target', '  cbzExportedOnly target'),
         'test_real_lean.RealLeanTests.test_private_self_reference_rejected'),
        ('preload-target-self-reference', 'ExactReplay.lean', original,
         original.replace('open Lean\n', 'open Lean\n\n' + preload).replace('  replayFromImports target', '  cbzPreloaded target'),
         'test_real_lean.RealLeanTests.test_public_self_reference_rejected'),
    ]


def run_mutations(output, only=None, group='all'):
    lean = os.environ.get('CBZ_TEST_LEAN', '/home/agent/.elan/toolchains/leanprover--lean4---v4.34.1/bin/lean')
    plan = definitions() + resource_definitions()
    if group == 'resource':
        plan = resource_definitions()
    elif group == 'wrapper':
        plan = [x for x in plan if x[4].startswith('test_kernel_replay.')]
    else:
        plan += derived_driver_mutations(lean)
    if only:
        if only not in [x[0] for x in plan]:
            raise ValueError('unknown mutation: ' + only)
        plan = [x for x in plan if x[0] == only]
    # Keep old-generation mutation receipts immutable when selecting this repair.
    evidence = ROOT / ('evidence/mutations' if group == 'all' else 'evidence/resource-mutations')
    evidence.mkdir(parents=True, exist_ok=True)
    results = []
    for name, filename, old, new, test in plan:
        with tempfile.TemporaryDirectory(dir=ROOT / 'evidence', prefix='mutant-') as d:
            work = Path(d) / 'kernel_replay'
            shutil.copytree(ROOT, work, ignore=shutil.ignore_patterns('evidence', '__pycache__', '*.pyc'))
            (work / 'evidence').mkdir()
            file = work / filename
            original = file.read_text()
            if original.count(old) != 1:
                raise ValueError(f'mutation {name} did not identify exactly one source occurrence')
            changed = original.replace(old, new, 1)
            file.write_text(changed)
            result_path = work / 'evidence/tests.json'
            command = [sys.executable, str(work / 'run_tests.py'), '--output', str(result_path), '--test', test]
            try:
                p = subprocess.run(command, capture_output=True, text=True, timeout=90)
                value = json.loads(result_path.read_text()) if result_path.exists() else {}
                record = {'name': name, 'file': filename, 'test': test,
                          'outcome': classify(value, p.returncode), 'returncode': p.returncode,
                          'original_sha256': digest(original.encode()),
                          'mutated_sha256': digest(changed.encode()), 'tests': value,
                          'stdout': p.stdout, 'stderr': p.stderr}
            except subprocess.TimeoutExpired as e:
                record = {'name': name, 'outcome': 'ERROR', 'error': 'mutation test timed out'}
            (evidence / (name + '.json')).write_text(json.dumps(record, indent=2) + '\n')
            results.append(record)
            print(name + ': ' + record['outcome'], flush=True)
    result = {'group': group, 'threshold': 1.0, 'planned': len(plan), 'executed': len(results),
              'killed': sum(x['outcome'] == 'KILLED' for x in results),
              'ok': bool(results) and len(results) == len(plan) and all(x['outcome'] == 'KILLED' for x in results),
              'results': results}
    Path(output).write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--output', required=True)
    parser.add_argument('--only')
    parser.add_argument('--group', choices=('all', 'wrapper', 'resource'), default='all')
    args = parser.parse_args()
    try:
        return 0 if run_mutations(args.output, args.only, args.group)['ok'] else 1
    except (ValueError, OSError) as e:
        print('mutation harness ERROR: ' + str(e), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
