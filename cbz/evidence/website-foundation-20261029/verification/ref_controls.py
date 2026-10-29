"""Bounded fixture scale and safety-guard mutation controls; no real remote."""
import argparse
import hashlib
import shutil
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

TOOL_NAME = 'cbz_refs.py'
TEST_NAME = 'test_cbz_refs.py'
CLASS_NAME = 'RefSafetyTests'
SCALE = True


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mutations(root, output):
    source = (root / 'tools' / TOOL_NAME).read_text()
    lifecycle = load(root / 'tools/cbz_mutation.py', 'cbz_lifecycle_' + TOOL_NAME)
    controls = [
        ('original_tag_ref_guard', 'if _local_refs(repo) != expected_refs:', 'if False:',
         'test_missing_original_annotated_tag_ref_is_rejected_before_gc'),
        ('full_state_guard', 'if actual != expected:', 'if False:',
         'test_unexpected_new_ref_refuses_all_deletions'),
        ('expected_old_leases', 'leases = [f\'--force-with-lease={ref}:{manifest["refs"][ref]}\' for ref in delete_refs]',
         'leases = []', 'test_lease_rejects_branch_changed_after_final_preflight_atomically'),
        ('default_exclusion', "excluded = {manifest['default_ref'], *manifest['protected_refs'], *manifest['active_refs']}",
         "excluded = {*manifest['protected_refs'], *manifest['active_refs']}",
         'test_publish_then_atomic_prune_preserves_default_policy_and_tag'),
        ('atomic_push', "'push', '--atomic', *leases", "'push', *leases",
         'test_server_rejection_is_atomic_and_never_silent_success'),
        ('archive_readback', 'def _remote_archive_readback(repo, remote, tag_ref, commit):',
         'def _remote_archive_readback(repo, remote, tag_ref, commit):\n    return', 'test_actual_tag_fetch_rejects_forged_advertised_readback'),
        ('manifest_hash_guard', "if hashlib.sha256(encoded).hexdigest() != receipt['manifest_sha256']:",
         'if False:', 'test_receipt_hash_tampering_is_detected'),
    ]
    results = []
    for name, old, new, test_name in controls:
        if source.count(old) != 1:
            raise RuntimeError(f'mutation location missing or ambiguous: {name}')
        with tempfile.TemporaryDirectory(prefix='cbz-mutation-') as tmp:
            mutant = Path(tmp) / 'mutant.py'
            mutant.write_text(source.replace(old, new))
            tests = load(root / 'tests' / TEST_NAME, 'cbz_tests_' + name)
            tests.TOOL = mutant
            suite = unittest.TestSuite([tests.RefSafetyTests(test_name)])
            with (output / ('mutation-' + name + '.log')).open('w') as stream:
                result = lifecycle.run_suite(suite, stream, [tests.__name__ + '.' + CLASS_NAME + '.' + test_name])
            results.append({'mutation': name, 'test': test_name,
                            'detected': result['body_kill'],
                            'failures': result['failures'], 'errors': result['errors'], 'lifecycle': result})
    (output / 'mutations.json').write_text(json.dumps(results, indent=2) + '\n')
    return results


def scale(root, output):
    tool = load(root / 'tools' / TOOL_NAME, 'cbz_scale_tool')
    with tempfile.TemporaryDirectory(prefix='cbz-scale-') as tmp:
        tmp = Path(tmp)
        remote = tmp / 'fixture.git'
        subprocess.run(['git', 'init', '-q', '--bare', str(remote)], check=True)
        # 2,215 genuinely unique branch-tip commits; one payload blob + distinct messages.
        lines = ['blob', 'mark :1', 'data 26', 'recoverable scale payload\n']
        for i in range(2215):
            name = 'main' if i == 0 else f'scale-{i:04d}'
            message = f'unique original tip {i}\n'
            lines.extend([f'commit refs/heads/{name}', 'committer Fixture <fixture@invalid> 1793293200 +0000',
                          f'data {len(message)}', message.rstrip('\n')])
            if i:
                lines.append('from refs/heads/main')
            lines.extend(['M 100644 :1 valuable', ''])
        subprocess.run(['git', '-C', str(remote), 'fast-import', '--quiet'],
                       input=('\n'.join(lines) + '\n').encode(), capture_output=True, check=True)
        subprocess.run(['git', '-C', str(remote), 'symbolic-ref', 'HEAD', 'refs/heads/main'], check=True)
        start = time.monotonic()
        out = tmp / 'snapshot'
        result = tool.snapshot(str(remote), out, 'pre-cbz-scale-2215', execute=True)
        snapshot_seconds = time.monotonic() - start
        repo = out / 'repository.git'
        parents = subprocess.run(['git', '-C', str(repo), 'show', '-s', '--format=%P', result['archive_commit']],
                                 capture_output=True, text=True, check=True).stdout.split()
        if len(parents) != 2215 or len(set(parents)) != 2215:
            raise AssertionError('unique archive parents missing')
        size = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
        # Exercise large expected-old CAS argument set and atomic deletion, on this local fixture only.
        start = time.monotonic()
        tool.publish(out, execute=True)
        publish_seconds = time.monotonic() - start
        start = time.monotonic()
        pruned = tool.prune(out, execute=True)
        prune_seconds = time.monotonic() - start
        state = tool.remote_state(str(remote))
        if [ref for ref in state['refs'] if ref.startswith('refs/heads/')] != ['refs/heads/main']:
            raise AssertionError('scale cleanup did not preserve exactly default branch')
        tip = parents[-1]
        recovered = subprocess.run(['git', '-C', str(repo), 'show', f'{tip}:valuable'],
                                   capture_output=True, text=True, check=True).stdout
        if recovered != 'recoverable scale payload\n':
            raise AssertionError('lost original content')
        report = {'unique_original_branch_tips': 2215, 'archive_parents': len(parents),
                  'deleted_branches': len(pruned['delete_refs']), 'snapshot_bytes': size,
                  'snapshot_seconds': round(snapshot_seconds, 3),
                  'publish_seconds': round(publish_seconds, 3), 'prune_seconds': round(prune_seconds, 3),
                  'recovery_verified': True, 'fixture_removed_on_exit': True}
        (output / 'scale.json').write_text(json.dumps(report, indent=2) + '\n')
        return report


def positive(root, output, test_name, class_name):
    tests = load(root / 'tests' / test_name, 'unmutated_positive')
    tests.TOOL = root / 'tools' / ('cbz_refs.py' if class_name == 'RefSafetyTests' else 'cbz_baseline.py')
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(getattr(tests, class_name))
    lifecycle = load(root / 'tools/cbz_mutation.py', 'cbz_positive_lifecycle_' + TOOL_NAME)
    with (output / 'positive.log').open('w') as stream:
        result = lifecycle.run_suite(suite, stream)
    report = {'tests_run': result['tests_run'], 'failures': result['failures'],
              'errors': result['errors'], 'skipped': result['skips'],
              'passed': result['baseline_success'], 'lifecycle': result}
    (output / 'positive.json').write_text(json.dumps(report, indent=2) + '\n')
    if not report['passed']:
        raise RuntimeError('unmutated positive harness failed; mutations cannot be credited')
    return report


def cli_controls(root, output):
    script = str(Path(__file__).resolve())
    missing_root = output / 'missing-source-root'
    existing = output / 'cli-existing-output'
    existing.mkdir()
    marker = existing / 'preserve'
    marker.write_text('keep exact bytes\n')
    inside = root / ('.cbz-control-must-not-create-' + output.name)
    if inside.exists() or inside.is_symlink():
        raise RuntimeError('finite inside-root control target unexpectedly exists')
    cases = [
        ('missing_output', ['--root', str(root)]),
        ('missing_root', ['--root', str(missing_root), '--output', str(output / 'must-not-create')]),
        ('existing_output', ['--root', str(root), '--output', str(existing)]),
        ('inside_root_output', ['--root', str(root), '--output', str(inside)]),
    ]
    rows = []
    for name, args in cases:
        result = subprocess.run([sys.executable, '-B', '-S', script, *args],
                                capture_output=True, timeout=15)
        rows.append({'control': name, 'exit_code': result.returncode, 'rejected': result.returncode == 2})
    if (marker.read_text() != 'keep exact bytes\n' or inside.exists()
            or (output / 'must-not-create').exists() or not all(r['rejected'] for r in rows)):
        raise RuntimeError('finite CLI/output preservation controls failed')
    shutil.rmtree(existing)
    (output / 'cli-controls.json').write_text(json.dumps(rows, indent=2) + '\n')
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='actual source checkout containing tools/ and tests/')
    parser.add_argument('--output', required=True, help='NEW output directory outside the source checkout')
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    original_output = Path(args.output)
    output = original_output.resolve()
    try:
        if (not root.is_dir() or not (root / 'tools' / TOOL_NAME).is_file()
                or not (root / 'tests' / TEST_NAME).is_file()):
            raise ValueError('source root/tool/test unavailable')
        if output.is_relative_to(root) or original_output.is_symlink() or output.exists():
            raise ValueError('output must be NEW and outside the source root')
        output.mkdir(parents=True, exist_ok=False)
    except (OSError, ValueError) as error:
        print(json.dumps({'status': 'REFUSED', 'error': str(error)}), file=sys.stderr)
        return 2
    sources = [root / 'tools' / TOOL_NAME, root / 'tests' / TEST_NAME, root / 'tools/cbz_mutation.py']
    before = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    report = {'schema': 'cbz.portable-controls.v1', 'status': 'FAIL', 'source_sha256': before,
              'control_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'finite engineering controls only; no global G0..G11, Layer A or CBZ-0 certification'}
    try:
        report['cli_controls'] = cli_controls(root, output)
        report['positive'] = positive(root, output, TEST_NAME, CLASS_NAME)
        report['mutations'] = mutations(root, output)
        if not all(row['detected'] and row['failures'] == 1 and row['errors'] == 0 for row in report['mutations']):
            raise RuntimeError('mutation survived or harness error occurred; no mutation credit')
        if SCALE:
            report['scale'] = scale(root, output)
        after = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
        if after != before:
            raise RuntimeError('source/test changed during controls; evidence is stale')
        report['status'] = 'PASS'
    except Exception as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
    (output / 'report.json').write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
