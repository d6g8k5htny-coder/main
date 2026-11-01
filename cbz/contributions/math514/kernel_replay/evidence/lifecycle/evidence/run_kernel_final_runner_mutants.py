#!/usr/bin/env python3
"""Targeted actual runner/classifier source mutants; no Lean or original-mutant replay."""
from pathlib import Path
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile

BASE = Path(__file__).resolve().parents[1]
WORK = BASE / 'work/kernel_replay'
OUT = BASE / 'evidence/kernel-final-runner-mutants'
OUT.mkdir(exist_ok=True)
GUARD = OUT / 'guard_run_tests.py'
GUARD.write_bytes((WORK / 'run_tests.py').read_bytes())
spec = importlib.util.spec_from_file_location('trusted_classifier', WORK / 'mutations.py')
trusted = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trusted)
sha = lambda data: hashlib.sha256(data).hexdigest()

def replace_once(old, new):
    def transform(source):
        assert source.count(old) == 1, old
        return source.replace(old, new, 1)
    return transform

def legacy(source):
    a, b = source.index('def classify('), source.index('\n\ndef digest')
    return source[:a] + '''def classify(value, code, expected_selection=None, expected_ids=None):
    if code == 1 and value.get('assertion_failures', 0) > 0 and not value.get('errors'):
        return 'KILLED'
    return 'SURVIVED' if code == 0 else 'ERROR'
''' + source[b:]

def remove_completion(source):
    anchor = 'result.body_finished_ids.append(method_id(test))'
    assert source.count(anchor) == 2
    return source.replace(anchor, 'pass  # mutant drops body completion')

def erase_subtest_origin(source):
    source = replace_once("phase = self.phase.get(id(test))", "phase = 'body'")(source)
    return replace_once("if phase == 'body' and id(test) not in self.body_active:", "if False:")(source)

PREFIX = 'test_mutation_harness.MutationHarnessTests.'
plan = [
    ('restore-summary-only-grading', 'mutations.py', legacy, 'test_setup_assertion_is_not_a_kill'),
    ('trust-overridable-display-id', 'run_tests.py', replace_once("return cls.__module__ + '.' + cls.__qualname__ + '.' + name", 'return test.id()'), 'test_empty_duplicate_synthetic_and_spoofed_population_is_error'),
    ('ignore-body-population-receipt', 'mutations.py', replace_once('keys[1:5]', 'keys[1:3]'), 'test_forged_or_missing_accounting_is_error'),
    ('accept-boolean-counts', 'mutations.py', replace_once('type(value.get(key)) is not int', 'not isinstance(value.get(key), int)'), 'test_forged_or_missing_accounting_is_error'),
    ('ignore-invalid-population', 'mutations.py', replace_once("if value.get('invalid_population') != []:", 'if False:'), 'test_empty_duplicate_synthetic_and_spoofed_population_is_error'),
    ('ignore-selector-binding', 'mutations.py', replace_once('or any(not any(selector', 'or False and any(not any(selector'), 'test_forged_or_missing_accounting_is_error'),
    ('drop-body-completion', 'run_tests.py', remove_completion, 'test_body_assertion_kills_after_all_selected_bodies_finish'),
    ('erase-subtest-lifecycle-origin', 'run_tests.py', erase_subtest_origin, 'test_lifecycle_subtests_and_subtest_errors_are_errors'),
]
plan = [item for item in plan if item[1] == 'run_tests.py']
def drop_dispatch_authorization(source):
    anchor = 'if id(test) not in result.dispatch_active or result.lifecycle_depth.get(id(test), 0):'
    assert source.count(anchor) == 2
    return source.replace(anchor, 'if False:')
plan.append(('drop-selected-method-dispatch-authorization', 'run_tests.py', drop_dispatch_authorization,
             'test_selected_method_invoked_from_lifecycle_is_error'))
results = []
for name, filename, transform, test in plan:
    with tempfile.TemporaryDirectory(prefix='kernel-harness-mutant-') as tmp:
        root = Path(tmp) / 'kernel_replay'
        shutil.copytree(WORK, root, ignore=shutil.ignore_patterns('evidence', '__pycache__', '*.pyc'))
        target = root / filename
        original = target.read_text()
        changed = transform(original)
        compile(changed, filename, 'exec')
        target.write_text(changed)
        output = OUT / (name + '.json')
        driver = Path(tmp) / 'monitor.py'
        driver.write_text('import importlib.util,sys\nfrom pathlib import Path\n'
                          's=importlib.util.spec_from_file_location("guard",sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\n'
                          'm.ROOT=Path(sys.argv[2]);sys.argv=["guard","--output",sys.argv[3],"--test",sys.argv[4]]\n'
                          'raise SystemExit(m.main())\n')
        command = [sys.executable, '-B', '-S', str(driver), str(GUARD), str(root), str(output), PREFIX + test]
        p = subprocess.run(command, capture_output=True, text=True, timeout=30)
        value = json.loads(output.read_text()) if output.exists() else {}
        record = {'name': name, 'file': filename, 'test': PREFIX + test,
                  'outcome': trusted.classify(value, p.returncode, [PREFIX + test], [PREFIX + test]),
                  'returncode': p.returncode, 'original_sha256': sha(original.encode()),
                  'mutated_sha256': sha(changed.encode()), 'guard_sha256': sha(GUARD.read_bytes()),
                  'tests': value, 'stderr': p.stderr}
        output.write_text(json.dumps(record, indent=2) + '\n')
        results.append({k: v for k, v in record.items() if k not in ('tests', 'stderr')})
        print(name + ': ' + record['outcome'], flush=True)
report = {'schema_version': 1, 'scope': '4 final runner-source mutants after dispatch guard change; unchanged classifier controls reused from prior generation',
          'planned': len(plan), 'killed': sum(r['outcome'] == 'KILLED' for r in results),
          'ok': all(r['outcome'] == 'KILLED' for r in results), 'results': results}
(OUT / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
raise SystemExit(0 if report['ok'] else 1)
