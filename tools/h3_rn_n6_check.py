#!/usr/bin/env python3
"""Replay the immutable H3/RN N6 successor; no mathematical status promotion.

The five reviewed jobs run in normal and optimized child interpreters. All
34 generated outputs must reproduce the archived bytes. Original author
tests remain in the archive and are not collected by master pytest/JUnit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import time
import zipfile

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import twelve_project_check as runtime

ROOT = runtime.ROOT
ARCHIVE = 'research/campaigns/h3_rn_n6_20260920_v1.zip'
ARCHIVE_BYTES = 571735
ARCHIVE_SHA256 = '73b9e63800f77c677c504f3097fe0bb2459ff6c94cdee5772b56062320aefc21'
MANIFEST_SHA256 = '81770e26ec654d3b462f06cebaa20d0dd282cfd4df10ac593c7887ea0d292b40'
PLAN_SHA256 = 'f0fae03e8bb0af0f389868db5cf332522c5cce7589f5f8a85d07aaeebb073db9'
RUNNER_SHA256 = '96893b45126ecfdc82904db0a7cce1f0b485dd23bc61e69b61f360992f8383f3'
MODES = ('normal', 'optimized')
MAX_OUTPUT_BYTES = 8_000_000
need = runtime.require


def reviewed_jobs():
    jobs = []
    for name, project, checker, report, verify, output in (
        ('h3_baseline', 'h3_floor', 'check_explicit_floor.py', 'result.json', '--verify', '--output'),
        ('h3_floor', 'h3_floor', 'check_midpoint_floor.py', 'result_midpoint.json', '--verify', '--output'),
        ('h3_ceiling', 'h3_ceiling', 'check_uniform_ceiling.py', 'certificate.json', '--certificate', '--output'),
        ('pin_energy', 'pin_energy', 'check_energy.py', 'result-v2.json', '--verify', '--write'),
    ):
        jobs.append(dict(name=name, project=project, checker=checker, report=report,
                         verify_flag=verify, output_flag=output,
                         outputs=[dict(generated='report.json', original=project + '/' + report)]))
    names = ['RUN.json'] + [case + '_' + site + '.json'
                           for case in ('original_pilot_admitted', 'pilot_admitted', 'x_inner_admitted', 'y_inner_admitted')
                           for site in ('M', 'S', 'y')]
    jobs.append(dict(name='rn_n6', project='rn_n6', outputs=[
        dict(generated=name, original='rn_n6/results_final/' + name) for name in names]))
    return jobs


def inspect_archive(raw, exclusions):
    need(runtime.identity(raw) == {'bytes': ARCHIVE_BYTES, 'sha256': ARCHIVE_SHA256},
         'canonical successor archive identity mismatch')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist()
        need(len(infos) == 71 and len({i.filename for i in infos}) == 71, 'archive member set mismatch')
        need(sum(i.file_size for i in infos) <= runtime.MAX_EXPANDED, 'expanded archive limit')
        for item in infos:
            runtime.safe_name(item.filename)
            need(stat.S_IFMT(item.external_attr >> 16) in (0, stat.S_IFREG), 'nonregular archive member')
            need(0 <= item.file_size <= runtime.MAX_MEMBER and not item.flag_bits & 1, 'oversized or encrypted member')
            need(item.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED), 'unsupported compression')
        manifest_raw = archive.read('MANIFEST.json')
        need(runtime.identity(manifest_raw)['sha256'] == MANIFEST_SHA256, 'manifest identity mismatch')
        records = runtime.strict_json(manifest_raw)['files']
        need(type(records) is list and len(records) == 70, 'incomplete manifest')
        manifest = {row['path']: {key: row[key] for key in ('bytes', 'sha256')} for row in records}
        need(len(manifest) == 70 and set(manifest) | {'MANIFEST.json'} == {i.filename for i in infos},
             'manifest does not cover exact archive allowlist')
        runtime.check_exclusions(manifest, exclusions)  # Before source-body extraction.
        members = {item.filename: archive.read(item.filename) for item in infos}
    for name, expected in manifest.items():
        need(runtime.identity(members[name]) == expected, 'bundle input identity mismatch: ' + name)
    need(runtime.identity(members['verification_plan.json'])['sha256'] == PLAN_SHA256, 'plan identity mismatch')
    need(runtime.identity(members['verify_closure.py'])['sha256'] == RUNNER_SHA256, 'runner identity mismatch')
    plan = runtime.strict_json(members['verification_plan.json'])
    need(plan['schema'] == 'existing-math-replay-plan-v1' and plan['jobs'] == reviewed_jobs(), 'unreviewed replay jobs')
    need(len(plan['bundle_inputs']) == 62 and len({r['path'] for r in plan['bundle_inputs']}) == 62,
         'incomplete bundle input identities')
    for row in plan['bundle_inputs']:
        need(manifest[row['path']] == {key: row[key] for key in ('bytes', 'sha256')}, 'plan/manifest disagreement')
    deps = {row['path']: {key: row[key] for key in ('bytes', 'sha256')} for row in plan['repository_inputs']}
    need(len(plan['repository_inputs']) == len(deps) == 25, 'incomplete repository dependencies')
    runtime.check_exclusions(deps, exclusions)
    return members, deps


def runtime_receipt(raw):
    need(len(raw) <= runtime.MAX_JSON, 'runtime receipt size limit')

    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'duplicate runtime receipt key')
            result[key] = value
        return result

    def nonfinite(_):
        raise ValueError('nonfinite runtime receipt number')

    # Only this execution receipt permits floats, solely for observed timings.
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)


def verify_outputs(output, members):
    receipt_raw = runtime.read_bounded(output / 'VERIFICATION.json', runtime.MAX_JSON)
    receipt = runtime_receipt(receipt_raw)
    need(receipt['schema'] == 'existing-math-replay-receipt-v1', 'unknown runtime receipt')
    need(receipt['verification_plan_sha256'] == PLAN_SHA256 and receipt['runner_sha256'] == RUNNER_SHA256,
         'runtime receipt does not bind reviewed runner and plan')
    need(type(receipt['bundle_inputs_unchanged']) is int and receipt['bundle_inputs_unchanged'] == 62
         and type(receipt['repository_inputs_unchanged']) is int and receipt['repository_inputs_unchanged'] == 25,
         'runtime input coverage mismatch')
    need(receipt['all_outputs_byte_identical'] is True and receipt['scientific_status_changed'] is False
         and type(receipt['organizational_independence_credit']) is int
         and receipt['organizational_independence_credit'] == 0, 'runtime receipt scope mismatch')
    jobs = reviewed_jobs()
    need([(item['job'], item['mode']) for item in receipt['jobs']] ==
         [(job['name'], mode) for job in jobs for mode in MODES], 'missing/duplicate/unexpected job executions')
    expected_files = {'VERIFICATION.json'}
    expected_identities = {'VERIFICATION.json': runtime.identity(receipt_raw)}
    matches = []
    for job in jobs:
        for mode in MODES:
            prefix = job['name'] + '-' + mode
            expected_files |= {prefix + '/stdout.txt', prefix + '/stderr.txt'}
            row = next(item for item in receipt['jobs'] if (item['job'], item['mode']) == (job['name'], mode))
            need(type(row['exit_code']) is int and row['exit_code'] == 0, 'child execution failed')
            elapsed = row['elapsed_seconds']
            need(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0, 'invalid observed child time')
            expected = []
            for pair in job['outputs']:
                name = prefix + ('/results/' if job['name'] == 'rn_n6' else '/') + pair['generated']
                expected_files.add(name)
                actual = runtime.read_bounded(output / name, runtime.MAX_MEMBER)
                runtime.compare_report(actual, members[pair['original']])
                recorded = dict(name=pair['generated'], **runtime.identity(actual))
                expected_identities[name] = runtime.identity(actual)
                expected.append(recorded)
                matches.append(dict(path=name, original=pair['original'], **runtime.identity(actual)))
            # Typed canonical JSON equality prevents bool/int aliases in counts.
            need(json.dumps(row['byte_identical_outputs'], sort_keys=True) == json.dumps(expected, sort_keys=True),
                 'runtime output identities differ from actual files')
    actual_files = set()
    artifacts = {}
    total = 0
    for path in output.rglob('*'):
        need(not path.is_symlink(), 'output symlink rejected')
        if path.is_file():
            name = str(path.relative_to(output))
            need(name in expected_files and len(actual_files) < 55, 'unexpected output artifact')
            raw = runtime.read_bounded(path, runtime.MAX_MEMBER)
            total += len(raw)
            need(total <= MAX_OUTPUT_BYTES, 'total output size limit')
            actual_files.add(name)
            artifacts[name] = runtime.identity(raw)
            if name in expected_identities:
                need(artifacts[name] == expected_identities[name], 'verified artifact changed during readback')
    need(len(matches) == 34 and actual_files == expected_files and len(actual_files) == 55, 'incomplete output artifact set')
    return matches, artifacts


def run(*, repo=ROOT, archive=None, output_dir=None, budget_seconds=180):
    need(sys.version_info[:2] == (3, 11) and not sys.flags.optimize and not os.environ.get('PYTHONOPTIMIZE'),
         'normal Python 3.11 parent required')
    need(type(budget_seconds) in (int, float) and 0 < budget_seconds <= 300, 'invalid whole-run budget')
    repo = Path(repo).resolve()
    archive = Path(archive) if archive is not None else repo / ARCHIVE
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='h3-rn-n6-', dir=Path(tempfile.gettempdir()).resolve()) as directory:
        work = Path(directory)
        need(not work.is_relative_to(repo) and not work.is_relative_to(ROOT), 'temporary extraction inside repository')
        output = runtime.fresh_output(output_dir if output_dir is not None else work / 'output', repo)
        report = dict(schema='h3-rn-n6-portable-replay-v1', passed=False,
                      started_utc=datetime.now(timezone.utc).isoformat(), budget_seconds=budget_seconds,
                      scientific_status_changed=False, original_prize_closed=False,
                      organizational_independence_credit=0, source_exposed=True,
                      historical_source_code_executed=False, snapshot_normalization=[],
                      original_author_tests='Historical archive receipts; not executed by this wrapper.',
                      scope='Fixed-axis continuum H3 and four local fixed-radius RN N6 squares; no full annulus, all-angle or q0 closure.')
        try:
            source_paths = [Path(__file__).resolve(), Path(runtime.__file__).resolve()]
            source_before = {str(path): runtime.identity(runtime.read_bounded(path, runtime.MAX_MEMBER)) for path in source_paths}
            exclusions = runtime.strict_json(runtime.read_bounded(repo / 'quarantine/EXCLUSIONS.json', runtime.MAX_JSON))['exclusions']
            raw = runtime.read_bounded(archive, ARCHIVE_BYTES)
            members, expected = inspect_archive(raw, exclusions)
            before = runtime.repository_inputs(repo, expected)
            report.update(archive=runtime.identity(raw), repository_inputs=before, execution_sources=source_before)
            bundle = work / 'bundle'
            for name, data in members.items():
                target = bundle / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as handle:
                    handle.write(data)
            argv = [sys.executable, '-B', str(bundle / 'verify_closure.py'), '--repo', str(repo), '--output', str(output / 'replay')]
            remaining = budget_seconds - (time.monotonic() - started)
            need(remaining > 0, 'whole-run budget exhausted before execution')
            report['execution'] = runtime.run_bounded(argv, bundle, output, 'runner', remaining)
            matches, artifacts = verify_outputs(output / 'replay', members)
            need({str(path.relative_to(bundle)) for path in bundle.rglob('*') if path.is_file()} == set(members),
                 'extracted input file set changed')
            for name, data in members.items():
                need(runtime.read_bounded(bundle / name, runtime.MAX_MEMBER) == data, 'extracted input changed: ' + name)
            need(runtime.repository_inputs(repo, before) == before and runtime.read_bounded(archive, ARCHIVE_BYTES) == raw,
                 'source inputs changed during replay')
            need({str(path): runtime.identity(runtime.read_bounded(path, runtime.MAX_MEMBER)) for path in source_paths} == source_before,
                 'wrapper/runtime source changed')
            for name, wanted in report['execution']['logs'].items():
                need(runtime.identity(runtime.read_bounded(output / ('runner.' + name + '.txt'), runtime.MAX_LOG)) == wanted,
                     'runner execution log changed')
            for name, wanted in artifacts.items():
                need(runtime.identity(runtime.read_bounded(output / 'replay' / name, runtime.MAX_MEMBER)) == wanted,
                     'replay artifact changed after validation')
            need({path.name for path in output.iterdir()} == {'replay', 'runner.stdout.txt', 'runner.stderr.txt'},
                 'unexpected wrapper output artifact')
            need(time.monotonic() - started <= budget_seconds, 'whole-run budget exhausted')
            report.update(passed=True, actual_executions=10, byte_identical_outputs=matches,
                          replay_artifacts=artifacts, bundle_inputs_unchanged=71, repository_inputs_unchanged=25)
        except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, subprocess.SubprocessError) as error:
            report['error'] = type(error).__name__ + ': ' + str(error)
        report.update(ended_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=round(time.monotonic() - started, 6))
        runtime.publish_report(output, report)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=ROOT)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--budget-seconds', type=float, default=180)
    args = parser.parse_args(argv)
    try:
        report = run(repo=args.repo, archive=args.archive, output_dir=args.output_dir, budget_seconds=args.budget_seconds)
        need(report['passed'], report.get('error', 'incomplete replay'))
        print('PASS: H3/RN N6; 10 executions; 34 byte-identical outputs; independence=0; no scientific promotion')
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print('REJECTED: ' + str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
