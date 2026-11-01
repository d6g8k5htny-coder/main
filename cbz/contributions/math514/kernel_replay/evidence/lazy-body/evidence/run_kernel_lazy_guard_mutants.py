#!/usr/bin/env python3
"""Only new lazy-body safeguards; old lifecycle and production cases are reused."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

BASE = Path(__file__).resolve().parents[1]
WORK = BASE / 'work/kernel_replay'
OUT = BASE / 'evidence/kernel-lazy-guard-mutants'
OUT.mkdir(exist_ok=True)
GUARD = OUT / 'guard_run_tests.py'
GUARD.write_bytes((WORK / 'run_tests.py').read_bytes())
spec = importlib.util.spec_from_file_location('trusted_classifier', WORK / 'mutations.py')
trusted = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trusted)
sha = lambda data: hashlib.sha256(data).hexdigest()
def replacement(source, old, new):
    if source.count(old) != 1:
        raise ValueError('nonunique source mutant anchor: ' + old)
    return source.replace(old, new, 1)
plan = [
    ('drop-pristine-lazy-shape-rejection', "identity is not None and not supported_method_shape(getattr(test, test._testMethodName))", 'False', 'test_declared_lazy_method_shapes_refuse_mixed_population_before_execution'),
    ('drop-lazy-return-rejection', 'if is_lazy_return(value):', 'if False:', 'test_sync_lazy_returns_refuse_without_advancing_suspended_work'),
    ('ignore-decorator-lazy-shape-links', "if not hasattr(current, '__wrapped__'):", 'if True:', 'test_wrapped_lazy_method_shapes_refuse_before_execution'),
]
results = []
for optimize in (False, True):
    mode = 'optimized' if optimize else 'normal'
    for name, old, new, test in plan:
        with tempfile.TemporaryDirectory(prefix='kernel-lazy-guard-mutant-') as tmp:
            root = Path(tmp) / 'kernel_replay'
            shutil.copytree(WORK, root, ignore=shutil.ignore_patterns('evidence', '__pycache__', '*.pyc'))
            target = root / 'run_tests.py'
            original = target.read_text()
            changed = replacement(original, old, new)
            compile(changed, str(target), 'exec')
            target.write_text(changed)
            output = OUT / (mode + '-' + name + '.json')
            driver = Path(tmp) / 'monitor.py'
            driver.write_text('import importlib.util,sys\nfrom pathlib import Path\n'
                              's=importlib.util.spec_from_file_location("guard",sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\n'
                              'm.ROOT=Path(sys.argv[2]);sys.argv=["guard","--output",sys.argv[3],"--test",sys.argv[4]]\n'
                              'raise SystemExit(m.main())\n')
            selection = 'test_mutation_harness.LazyBodyHarnessTests.' + test
            command = [sys.executable, '-B', '-S', *(['-O'] if optimize else []), str(driver), str(GUARD), str(root), str(output), selection]
            env = dict(os.environ)
            env['PYTHONOPTIMIZE'] = '1' if optimize else '0'
            p = subprocess.run(command, capture_output=True, text=True, timeout=30, env=env)
            value = json.loads(output.read_text()) if output.exists() else {}
            record = {'name': name, 'mode': mode, 'test': selection,
                      'outcome': trusted.classify(value, p.returncode, [selection], [selection]),
                      'returncode': p.returncode, 'original_sha256': sha(original.encode()),
                      'mutated_sha256': sha(changed.encode()), 'guard_sha256': sha(GUARD.read_bytes()),
                      'tests': value, 'stderr': p.stderr}
            output.write_text(json.dumps(record, indent=2) + '\n')
            results.append({k: v for k, v in record.items() if k not in ('tests', 'stderr')})
            print(mode + ' ' + name + ': ' + record['outcome'], flush=True)
report = {'schema_version': 1, 'scope': '3 new lazy-body safeguard source mutants in each mode; no original lifecycle or production mutant replay',
          'distinct_mutations': len(plan), 'executions': len(results),
          'ok': all(r['outcome'] == 'KILLED' for r in results), 'results': results}
(OUT / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
raise SystemExit(0 if report['ok'] else 1)
