#!/usr/bin/env python3
"""Replay the immutable one-box RN Bernstein and sharp-variance successor.

Exact source admission and normal/-O report equality are execution evidence,
not a scientific status promotion or independent review. Only the existing
fixed-radius, fixed-axis wedge and its full mark window are covered.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
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
ARCHIVE = 'research/campaigns/rn_bernstein_sharp_variance_20260921_v1.zip'
ARCHIVE_BYTES = 157355
ARCHIVE_SHA256 = '868682c92d41015fe9a41714aeef243d9a8ff4713bba29d613ca13529e797ce9'
MANIFESTS = {
    'MANIFEST.json': ('571fcd218db033f1571a3402d13d34d3e9bfa0e2d76799be985bcd9c88aa01dd',
                      'RN_TWO_SUCCESSORS_FROZEN_ALLOWLIST_V1', 47),
    'bernstein/MANIFEST.json': ('42eba66b67eb5c883ece5ab702a36006bff20768c928388a224d9d2621ff4720',
                                'RN_BERNSTEIN_FROZEN_ALLOWLIST_V1', 19),
    'sharp_variance/MANIFEST.json': ('cf78dd42b9a68807ea51e23ed03838b30959cc83be9c8e432068d9763563ac53',
                                     'sharp-variance-frozen-allowlist-v1', 11),
}
MANIFEST_BYTES = {'MANIFEST.json': 7197, 'bernstein/MANIFEST.json': 2960, 'sharp_variance/MANIFEST.json': 1766}
MODES = ('normal', 'optimized')
REPORTS = {
    'spatial/result.json': 'bernstein/result.json',
    'sharp.json': 'sharp_variance/result-v2.json',
    'result.json': 'combined-result.json',
}
ABSENT_PACKAGE_INITIALIZERS = ('research/__init__.py', 'research/rn/__init__.py',
                               'tools/__init__.py', 'engine/__init__.py')
ELIGIBILITY_METADATA = ('quarantine/EXCLUSIONS.json', 'drive/inventory.jsonl', 'drive/source_map/Payloads.csv')
MAX_OUTPUT_BYTES = 8_000_000
need = runtime.require


def identity_map(value):
    need(type(value) is dict and value, 'identity map required')
    for name, row in value.items():
        if name != 'quarantine/EXCLUSIONS.json':
            runtime.safe_name(name)
        need(type(row) is dict and set(row) == {'bytes', 'sha256'}, 'invalid identity fields')
        need(type(row['bytes']) is int and 0 <= row['bytes'] <= runtime.MAX_REPOSITORY_FILE,
             'invalid identity byte count')
        need(type(row['sha256']) is str and re.fullmatch('[0-9a-f]{64}', row['sha256']),
             'invalid identity digest')
    return value


def inspect_archive(raw, exclusions):
    """Admit this exact authored packet; exclude sources before materialization."""
    need(runtime.identity(raw) == {'bytes': ARCHIVE_BYTES, 'sha256': ARCHIVE_SHA256},
         'canonical successor archive identity mismatch')
    runtime.check_exclusions({ARCHIVE: runtime.identity(raw)}, exclusions)
    runtime.check_exclusions({name: {'bytes': MANIFEST_BYTES[name], 'sha256': contract[0]}
                              for name, contract in MANIFESTS.items()}, exclusions)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist()
        names = {item.filename for item in infos}
        need(len(infos) == len(names) == 48, 'archive member set mismatch')
        need(sum(item.file_size for item in infos) <= runtime.MAX_EXPANDED, 'expanded archive limit')
        for item in infos:
            runtime.safe_name(item.filename)
            need(stat.S_IFMT(item.external_attr >> 16) in (0, stat.S_IFREG), 'nonregular archive member')
            need(0 <= item.file_size <= runtime.MAX_MEMBER and not item.flag_bits & 1,
                 'oversized or encrypted member')
            need(item.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED), 'unsupported compression')
        manifests = {}
        for name, (digest, schema, count) in MANIFESTS.items():
            raw_manifest = archive.read(name)
            need(runtime.identity(raw_manifest) == {'bytes': MANIFEST_BYTES[name], 'sha256': digest},
                 'manifest identity mismatch')
            parsed = runtime.strict_json(raw_manifest)
            need(parsed['schema'] == schema, 'manifest schema mismatch')
            rows = identity_map(parsed['files'])
            prefix = name.removesuffix('MANIFEST.json')
            scoped = {prefix + relative: expected for relative, expected in rows.items()}
            need(len(scoped) == count and set(scoped) | {name} ==
                 {member for member in names if member.startswith(prefix)}, 'manifest allowlist mismatch')
            runtime.check_exclusions(scoped, exclusions)
            manifests[name] = scoped
        members = {name: archive.read(name) for name in sorted(names)}
    for scoped in manifests.values():
        for name, expected in scoped.items():
            need(runtime.identity(members[name]) == expected, 'bundle identity mismatch: ' + name)
    bernstein = runtime.strict_json(members['bernstein/DEPENDENCIES.json'])
    sharp = runtime.strict_json(members['sharp_variance/DEPENDENCIES.json'])
    need(bernstein['schema'] == 'RN_BERNSTEIN_INPUTS_V1' and
         sharp['schema'] == 'sharp-variance-repository-dependencies-v1', 'dependency schema mismatch')
    deps = dict(identity_map(bernstein['files']))
    need(len(deps) == 30 and all(name.endswith('.py') for name in deps), 'Python dependency set mismatch')
    rows = sharp['files']
    need(type(rows) is list and len(rows) == 32 and len({row['path'] for row in rows}) == 32,
         'sharp dependency set mismatch')
    for row in rows:
        need(type(row) is dict and set(row) == {'path', 'bytes', 'sha256'}, 'invalid dependency fields')
        name, expected = row['path'], {key: row[key] for key in ('bytes', 'sha256')}
        identity_map({name: expected})
        need(name not in deps or deps[name] == expected, 'conflicting shared dependency')
        deps[name] = expected
    need(len(deps) == 36, 'repository dependency union mismatch')
    runtime.check_exclusions(deps, exclusions)
    return members, deps


def admitted_inputs(repo, expected):
    """Check code, then live metadata eligibility, then original source bytes.

    Loading the pinned gate by its selected checkout path keeps --repo portable.
    authenticated_inputs does not execute the archived Taylor module or RN5
    historical programs. Numerical children later execute the authored Taylor.
    """
    # This frozen replay already pins these metadata bytes. Refuse any new
    # exclusion/hold before the older gate could read an upstream source.
    runtime.repository_inputs(repo, {name: expected[name] for name in ELIGIBILITY_METADATA})
    runtime.repository_inputs(repo, {name: row for name, row in expected.items() if name.endswith('.py')})
    for name in ABSENT_PACKAGE_INITIALIZERS:
        need(not (repo / name).exists() and not (repo / name).is_symlink(), 'unexpected package initializer: ' + name)
    from tools import h3_rn_n6_check
    for module in (runtime, h3_rn_n6_check):
        path = Path(module.__file__).resolve()
        name = 'tools/' + path.name
        need(runtime.identity(runtime.read_bounded(path, runtime.MAX_MEMBER)) == expected[name],
             'admission runtime differs from pinned dependency')
    spec = importlib.util.spec_from_file_location('rn_successor_admission', repo / 'research/rn/n6_inputs.py')
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    _, source_identities = gate.authenticated_inputs(repo)
    return runtime.repository_inputs(repo, expected), source_identities


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()


def check_timings(raw, mode):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'duplicate timing key')
            result[key] = value
        return result

    def reject(_):
        raise ValueError('nonfinite timing value')

    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=reject)
    need(type(value) is dict and set(value) == {'timings', 'python', 'optimize', 'scope'}, 'timing fields mismatch')
    need(value['python'] == sys.version and type(value['optimize']) is int and
         value['optimize'] == MODES.index(mode), 'timing interpreter mismatch')
    need(value['scope'] == 'Local observed wall time only; includes warm-cache effects, not an independent compute-cost experiment.',
         'timing scope mismatch')
    need(type(value['timings']) is list and len(value['timings']) == 2, 'timing attempt count mismatch')
    for row, method in zip(value['timings'], ('original', 'bernstein')):
        need(type(row) is dict and set(row) == {'pieces', 'method', 'elapsed_seconds'} and
             type(row['pieces']) is int and row['pieces'] == 1 and row['method'] == method, 'timing attempt mismatch')
        elapsed = row['elapsed_seconds']
        need(type(elapsed) in (int, float) and math.isfinite(elapsed) and 0 <= elapsed <= 300, 'invalid observed elapsed time')


def verify_outputs(output, members, mode):
    expected = {name: members[source] for name, source in REPORTS.items()}
    attempts = runtime.strict_json(members['bernstein/result.json'])['attempts']
    need([(item['method'], item['pieces']) for item in attempts] == [('original', 1), ('bernstein', 1)],
         'unreviewed frozen attempts')
    for entry in attempts:
        expected['spatial/attempt_1_' + entry['method'] + '.json'] = canonical(entry)
    files = set(expected) | {'spatial/timings.json', 'replay-0.stdout', 'replay-0.stderr',
                             'replay-1.stdout', 'replay-1.stderr'}
    artifacts, total = {}, 0
    for path in output.rglob('*'):
        need(not path.is_symlink(), 'output symlink rejected')
        name = path.relative_to(output).as_posix()
        if path.is_dir():
            need(name == 'spatial', 'unexpected output directory')
            continue
        need(name in files, 'unexpected output artifact')
        raw = runtime.read_bounded(path, runtime.MAX_MEMBER)
        total += len(raw)
        need(total <= MAX_OUTPUT_BYTES, 'output size limit')
        artifacts[name] = runtime.identity(raw)
        if name in expected:
            runtime.compare_report(raw, expected[name])
        elif name == 'spatial/timings.json':
            check_timings(raw, mode)
    need(set(artifacts) == files and len(artifacts) == 10, 'incomplete output set')
    return artifacts


def run(*, repo=None, archive=None, output_dir=None, budget_seconds=180):
    need(sys.version_info[:2] == (3, 11) and not sys.flags.optimize and not os.environ.get('PYTHONOPTIMIZE'),
         'normal Python 3.11 parent required')
    need(type(budget_seconds) in (int, float) and math.isfinite(budget_seconds) and 0 < budget_seconds <= 300,
         'invalid whole-run budget')
    repo = Path(ROOT if repo is None else repo).resolve()
    archive = Path(archive) if archive is not None else repo / ARCHIVE
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='rn-bernstein-sharp-', dir=Path(tempfile.gettempdir()).resolve()) as directory:
        work = Path(directory)
        need(not work.is_relative_to(repo) and not work.is_relative_to(ROOT), 'extraction inside repository')
        output = runtime.fresh_output(output_dir if output_dir is not None else work / 'output', repo)
        report = dict(schema='rn-bernstein-sharp-portable-replay-v1', passed=False,
                      started_utc=datetime.now(timezone.utc).isoformat(), budget_seconds=budget_seconds,
                      scientific_status_changed=False, original_prize_closed=False,
                      organizational_independence_credit=0, source_exposed=True,
                      historical_source_code_executed=False, snapshot_normalization=[],
                      original_author_tests='Historical archive receipts; not executed by this wrapper.',
                      scope='Same fixed-r 1/20, b=6/5, x-axis wedge and full height window; no full annulus, all radii, orientations, event or q0 closure.',
                      limitations='Unsigned local receipt; no trusted time, environment lock, sandbox, or change-and-revert detection.')
        try:
            source_paths = (Path(__file__).resolve(), Path(runtime.__file__).resolve())
            sources = {str(path): runtime.identity(runtime.read_bounded(path, runtime.MAX_MEMBER)) for path in source_paths}
            exclusions = runtime.strict_json(runtime.read_bounded(repo / 'quarantine/EXCLUSIONS.json', runtime.MAX_JSON))['exclusions']
            raw = runtime.read_bounded(archive, ARCHIVE_BYTES)
            members, expected = inspect_archive(raw, exclusions)
            before, source_before = admitted_inputs(repo, expected)
            report.update(archive=runtime.identity(raw), repository_inputs=before,
                          source_admission=source_before, execution_sources=sources)
            bundle = work / 'bundle'
            for name, data in members.items():
                target = bundle / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as stream:
                    stream.write(data)
            executions, artifacts = {}, {}
            for mode in MODES:
                argv = [sys.executable, '-B'] + (['-O'] if mode == 'optimized' else [])
                argv += [str(bundle / 'close_local_wedge.py'), '--repo', str(repo), '--output', str(output / mode)]
                remaining = budget_seconds - (time.monotonic() - started)
                need(remaining > 0, 'whole-run budget exhausted before execution')
                executions[mode] = runtime.run_bounded(argv, bundle, output, mode, remaining)
                artifacts[mode] = verify_outputs(output / mode, members, mode)
            need({p.relative_to(bundle).as_posix() for p in bundle.rglob('*') if p.is_file()} == set(members),
                 'extracted input set changed')
            for name, data in members.items():
                need(runtime.read_bounded(bundle / name, runtime.MAX_MEMBER) == data, 'extracted input changed: ' + name)
            after, source_after = admitted_inputs(repo, expected)
            need((before, source_before) == (after, source_after) and runtime.read_bounded(archive, ARCHIVE_BYTES) == raw,
                 'source inputs changed during replay')
            need({str(path): runtime.identity(runtime.read_bounded(path, runtime.MAX_MEMBER)) for path in source_paths} == sources,
                 'wrapper/runtime changed during replay')
            for mode in MODES:
                need(verify_outputs(output / mode, members, mode) == artifacts[mode], 'output changed after validation')
                for channel, wanted in executions[mode]['logs'].items():
                    need(runtime.identity(runtime.read_bounded(output / (mode + '.' + channel + '.txt'), runtime.MAX_LOG)) == wanted,
                         'execution log changed')
            need({p.name for p in output.iterdir()} == set(MODES) |
                 {mode + '.' + channel + '.txt' for mode in MODES for channel in ('stdout', 'stderr')},
                 'unexpected wrapper output')
            need(time.monotonic() - started <= budget_seconds, 'whole-run budget exhausted')
            report.update(passed=True, executions=executions, replay_artifacts=artifacts,
                          composition_executions=2, numerical_child_executions=4,
                          byte_identical_frozen_reports=6, canonical_attempt_matches=4,
                          bundle_inputs_unchanged=48, repository_inputs_unchanged=36,
                          timings='NON-CERTIFYING observed wall times; retained, not compared to historical timings.',
                          result=runtime.strict_json(members['combined-result.json']))
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
        print('PASS: RN Bernstein/sharp; normal/-O; 6 byte-identical reports; same local wedge; independence=0; no promotion')
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print('REJECTED: ' + str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
