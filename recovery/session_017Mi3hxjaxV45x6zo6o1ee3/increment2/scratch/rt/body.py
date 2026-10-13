import hashlib, json, os, pathlib, subprocess, sys
root = pathlib.Path('frontiers/downstream_gate_20260925').resolve()
pins = json.loads((root / 'SOURCE_FILES.json').read_text())['files']
def verify():
    for name, identity in pins.items():
        part = pathlib.PurePosixPath(name)
        path = root / name
        if part.is_absolute() or '..' in part.parts or not path.resolve().is_relative_to(root):
            raise ValueError('safe repository-relative pinned source required')
        if any((root / pathlib.Path(*part.parts[:i])).is_symlink() for i in range(1, len(part.parts)+1)):
            raise ValueError('symlink pinned source rejected')
        raw = path.read_bytes()
        if len(raw) != identity['bytes'] or hashlib.sha256(raw).hexdigest() != identity['sha256']:
            raise ValueError('source identity mismatch: ' + name)
if sys.version_info[:2] != (3, 11):
    raise RuntimeError('actual Python 3.11 required')
verify()
out = pathlib.Path(os.environ['RUNNER_TEMP']) / 'downstream-gate-results'
subprocess.run([sys.executable, '-B', '-S', str(root / 'run_validation.py'), '--output', str(out)], check=True, timeout=480)
subprocess.run([
    sys.executable, '-B', '-S', str(root / 'git_transition_audit.py'),
    '--repo', str(pathlib.Path.cwd()), '--base', os.environ['BASE_SHA'],
    '--head', os.environ['GITHUB_SHA'], '--output', str(out / 'TRANSITION_AUDIT.json'),
], check=True, timeout=120)
regressions = [
    ('coefficients/side24_v1', 30),
    ('frontiers/three_fronts_20260924', 57),
    ('frontiers/price_budget_20260924', 26),
    ('frontiers/full_price_20260924', 36),
    ('frontiers/remote_window_20260924', 28),
]
for mode, flags in [('normal', []), ('optimized', ['-O'])]:
    for index, (folder, count) in enumerate(regressions):
        run = subprocess.run(
            [sys.executable, '-B', *flags, '-S', '-m', 'unittest', 'discover', '-s', folder, '-p', 'test_*.py', '-v'],
            capture_output=True, text=True, timeout=90,
        )
        (out / f'regression_{mode}_{index}.stdout').write_text(run.stdout)
        (out / f'regression_{mode}_{index}.stderr').write_text(run.stderr)
        if run.returncode or f'Ran {count} tests' not in run.stderr or 'skipped=' in run.stderr:
            raise RuntimeError('regression failed: ' + folder)
verify()
report = json.loads((out / 'REPORT.json').read_text())
if not report['passed'] or report['distinct_tests'] != 71 or report['distinct_semantic_mutations'] != 24:
    raise RuntimeError('unexpected new-test scope')
(out / 'GITHUB_CONTEXT.json').write_text(json.dumps({
    'run': os.environ['GITHUB_RUN_ID'], 'tested_commit': os.environ['GITHUB_SHA'],
    'base_commit': os.environ['BASE_SHA'], 'event': os.environ['GITHUB_EVENT_NAME'],
    'python': sys.version, 'gate_distinct_tests': 71,
    'regression_distinct_tests': 177, 'total_distinct_tests': 248,
    'semantic_mutations': 24, 'source_files_verified': len(pins),
    'lemma_closed': False, 'scientific_effect': 'NONE',
    'meaning': 'same-author software verification; no independent theorem acceptance',
}, indent=2) + '\n')
print('DOWNSTREAM_GATE_PASSED')