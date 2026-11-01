"""Reproducible affected-scope controller; reuses unchanged mutation accounting.

It selects whole original test methods without rewriting their bodies. No Lean
compiler or project command is invoked. Unchanged Python-only methods are kept
in the complete source and are excluded only from this temporary selected copy.
"""
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

SOURCE = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1]).resolve()
OUT.mkdir(parents=True, exist_ok=True)
OLD = SOURCE / 'evidence/pre-canonical698-source'


def sha(data):
    return hashlib.sha256(data).hexdigest()


spec = importlib.util.spec_from_file_location('scoped_mutation_accounting', SOURCE / 'mutation_test.py')
accounting = importlib.util.module_from_spec(spec)
spec.loader.exec_module(accounting)
test_path = SOURCE / 'tests/test_auditor.py'
text = test_path.read_text()
lines = text.splitlines(keepends=True)
parsed = ast.parse(text)
selection, removed = [], []
for cls in parsed.body:
    if not isinstance(cls, ast.ClassDef):
        continue
    for method in cls.body:
        if not isinstance(method, ast.FunctionDef) or not method.name.startswith('test_'):
            continue
        calls = {n.attr for n in ast.walk(method) if isinstance(n, ast.Attribute)}
        literals = {n.value for n in ast.walk(method) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        affected = bool(calls & {'lean_codes', 'check_lean', 'cleanlean', 'qlean'}) or any('.lean' in value or value in {'lean', 'project_lean'} for value in literals)
        body = ''.join(lines[method.lineno - 1:method.end_lineno])
        row = {'id': 'test_auditor.' + cls.name + '.' + method.name,
               'line': method.lineno, 'end_line': method.end_lineno,
               'source_sha256': sha(body.encode())}
        if affected:
            selection.append(row)
        else:
            removed.append((method.lineno, method.end_lineno))
for first, last in sorted(removed, reverse=True):
    del lines[first - 1:last]
selected_text = ''.join(lines)
selected_ast = ast.parse(selected_text)
selected_methods = {cls.name + '.' + method.name: ast.dump(method, include_attributes=False)
                    for cls in selected_ast.body if isinstance(cls, ast.ClassDef)
                    for method in cls.body if isinstance(method, ast.FunctionDef) and method.name.startswith('test_')}
for cls in parsed.body:
    if isinstance(cls, ast.ClassDef):
        for method in cls.body:
            key = cls.name + '.' + method.name if isinstance(method, ast.FunctionDef) else ''
            if key in selected_methods:
                assert selected_methods[key] == ast.dump(method, include_attributes=False)

selected_mutants = [row for row in accounting.MUTATIONS
                    if row[1] == 'canonical_lean.py'
                    or row[1] == 'source_checks.py' and (
                        row[2] in text or row[0] in {
                            'empty-lean-findings', 'disable-placeholder-category', 'disable-axiom-category',
                            'disable-native-category', 'disable-unchecked-declaration-refusal',
                            'skip-malformed-character-refusal', 'miss-escaped-placeholder-helpers',
                            'drop-escaped-constant-normalization', 'miss-multiline-commented-native-config',
                            'disable-macro-gap-category', 'drop-lean-dependencies',
                            'truncate-primed-lean-imports', 'silence-unsupported-lean-import-tail',
                            'drop-literal-import-operands', 'discard-canonical-lexical-errors',
                            'ignore-canonical-metadata-shape'})]
source_paths = ['auditor.py', 'source_checks.py', 'canonical_lean.py', 'mutation_test.py', 'tests/test_auditor.py', 'tests/test_mutation_runner.py']
starting = {p: sha((SOURCE / p).read_bytes()) for p in source_paths}
report = {'started_utc': datetime.now(timezone.utc).isoformat(), 'selected_methods': selection,
          'selected_source_sha256': sha(selected_text.encode()), 'full_test_sha256': sha(text.encode()),
          'selected_mutants': [row[0] for row in selected_mutants], 'modes': {},
          'claim': 'affected Lean integration checks and actual source mutants; no fresh full-suite, Lean compiler, actual repository G1 or CBZ claim'}
(OUT / 'selected-test_auditor.py.txt').write_text(selected_text)
for mode in ['normal', 'optimized']:
    previous = os.environ.get('PYTHONOPTIMIZE')
    if mode == 'optimized':
        os.environ['PYTHONOPTIMIZE'] = '1'
    else:
        os.environ.pop('PYTHONOPTIMIZE', None)
    with tempfile.TemporaryDirectory(prefix='cbz-lexer698-scope-') as tmp:
        root = Path(tmp)
        (root / 'tests').mkdir()
        for name in ['auditor.py', 'source_checks.py', 'canonical_lean.py']:
            shutil.copy2(SOURCE / name, root / name)
        (root / 'tests/test_auditor.py').write_text(selected_text)
        accounting.HERE = root
        logdir = OUT / mode
        baseline = accounting.run_one('baseline-identity', 'source_checks.py', lambda source: source, logdir)
        results = []
        if baseline['outcome'] == 'SURVIVED' and baseline['tests_run'] == len(selection):
            for name, filename, before, after in selected_mutants:
                def transform(source, before=before, after=after):
                    if source.count(before) != 1:
                        raise ValueError('mutation anchor is not unique/present')
                    return source.replace(before, after, 1)
                results.append(accounting.run_one(name, filename, transform, logdir))
        report['modes'][mode] = {'baseline': baseline, 'mutants': results,
                                 'killed': sum(row['outcome'] == 'KILLED' for row in results)}
    if previous is None:
        os.environ.pop('PYTHONOPTIMIZE', None)
    else:
        os.environ['PYTHONOPTIMIZE'] = previous
ending = {p: sha((SOURCE / p).read_bytes()) for p in source_paths}
report.update(ended_utc=datetime.now(timezone.utc).isoformat(), source_sha256=ending,
              source_unchanged_during_run=starting == ending)
report['observed_status'] = 'PASS' if starting == ending and all(
    mode['baseline']['outcome'] == 'SURVIVED' and mode['baseline']['tests_run'] == len(selection)
    and len(mode['mutants']) == len(selected_mutants) and mode['killed'] == len(selected_mutants)
    for mode in report['modes'].values()) else 'FAIL'
(OUT / 'receipt.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
print(json.dumps({'status': report['observed_status'], 'selected_tests': len(selection),
                  'selected_mutants': len(selected_mutants),
                  'modes': {mode: {'baseline': r['baseline']['outcome'], 'tests': r['baseline'].get('tests_run'),
                                   'killed': r['killed'], 'other': [x['name'] + ':' + x['outcome'] for x in r['mutants'] if x['outcome'] != 'KILLED']}
                            for mode, r in report['modes'].items()}}, sort_keys=True))
raise SystemExit(0 if report['observed_status'] == 'PASS' else 1)
