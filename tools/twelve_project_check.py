#!/usr/bin/env python3
"""Replay one immutable author candidate, not a mathematical status promotion.

The original ZIP is the only intake: no author scratch directory or source
reader is executed. Nine original CLIs accept --repo; the other three need
only bundled sources. Reports compare byte-for-byte, with no normalization.

--author-tests additionally runs the original tests in a separate process.
Those tests have host-bound defaults, so this optional mode refuses other
roots. Their subTests never enter the master pytest/JUnit collection. Portable
CI runs candidate replay plus this wrapper's own negative controls instead.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import selectors
import signal
import stat
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = 'research/campaigns/q0_twelve_20260920_v1.zip'
ARCHIVE_SHA256 = '7a36b7dbaa45b80df9e75a2c063c141440a329c1fb66b85ba828a7fe6517dc8f'
ARCHIVE_BYTES = 1109839
MANIFEST_SHA256 = 'a17f9fc0767af86b5e9a1a6a4aaa4c5e68bc77cbfb1d1ff6ec99a2bbdc8401ba'
PLAN_SHA256 = 'aba7fede7860e7fb0c0480a78b445c446faec1624aab0d6d32a7533e5d41b87e'
INCIDENT_SHA256 = '579ef6cd88d543ec14db45558e5e7c1899078095e0df295c7d46fd1810273831'
INCIDENT_ID = '1hnd7jbFwcCQIuOcbrzilMQoymsOzFQZb'
MAX_JSON = 1_000_000
MAX_MEMBER = 2_000_000
MAX_EXPANDED = 5_000_000
MAX_LOG = 2_000_000
MAX_REPOSITORY_FILE = 32_000_000
# The original aggregate plan omitted these runtime reads. Their identities
# are already pinned in the ceiling report and Lambda dependency metadata.
# Bind both without rewriting the original plan, candidates or archive.
SUPPLEMENTAL_DEPENDENCIES = {
    'drive/mirrors/2026-09-17_RN5_MOMENT_REPAIR_AND_REVIEW_ERRATUM/RN5_NEAR_MOMENT_REPAIR.md': {
        'bytes': 13725, 'sha256': 'ac89f60b8206bfe011e6c2bc653e7acb39bc83a55c2a2c8fd1e17fd70c6c0383'},
    'docs/OPEN_PROBLEMS.md': {
        'bytes': 28417, 'sha256': '8f404f87e44d2861770eb86deed79710be3bc072aebeaff74d232304670dfb4d'},
}
# script, certificate option, needs repo, shared sources, original test, count
PROJECTS = {
    'h3_uniform': ('check_h3_uniform.py', '--verify', True, False, 'test_h3_uniform.py', 20),
    'h3_ceiling': ('check_ceiling.py', '--certificate', True, False, 'test_ceiling.py', 15),
    'lpw_amplitude': ('lpw_amplitude.py', '--verify', True, False, 'test_lpw_amplitude.py', 30),
    'jet_definition': ('checker.py', '--verify', True, False, 'test_checker.py', 16),
    'lpw_headline': ('lpw_headline.py', '--verify', True, False, 'test_lpw_headline.py', 31),
    'tb_multiplicity': ('multiplicity.py', '--candidate', False, True, 'test_multiplicity.py', 29),
    'tb_contact': ('contact_check.py', '--certificate', False, False, 'test_contact.py', 35),
    'tb_tails': ('tail_bounds.py', '--verify', True, True, 'test_tail_bounds.py', 25),
    'wp_event': ('wp_event.py', '--verify', True, False, 'test_wp_event.py', 36),
    'lambda_uniform': ('lambda_repair.py', '--verify', True, False, 'test_lambda_repair.py', 33),
    'gaussian_tube': ('gaussian_tube.py', '--verify', True, False, 'test_gaussian_tube.py', 22),
    'bonferroni_eta': ('bonferroni_eta.py', '--certificate', False, False, 'test_bonferroni_eta.py', 28),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def read_bounded(path, limit):
    path = Path(path).absolute()
    require(path == path.resolve() and not path.is_symlink(), 'symlink or noncanonical input path')
    require(stat.S_ISREG(path.stat().st_mode), 'regular input file required')
    with path.open('rb') as handle:
        raw = handle.read(limit + 1)
    require(len(raw) <= limit, 'input size limit exceeded: ' + str(path))
    return raw


def strict_json(raw):
    require(len(raw) <= MAX_JSON, 'JSON size limit exceeded')

    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result

    def inexact(_):
        raise ValueError('noninteger JSON number')

    return json.loads(raw, object_pairs_hook=pairs, parse_float=inexact, parse_constant=inexact)


def safe_name(name):
    require(type(name) is str and name and '\\' not in name, 'invalid member path')
    path = PurePosixPath(name)
    require(not path.is_absolute() and str(path) == name
            and all(part not in ('', '.', '..') for part in path.parts), 'unsafe member path')
    require(not any(token in name.lower() for token in
                    ('99_do_not_open', 'quarantine/', 'legacy/', 'verify_lambda_grid_v2.py')),
            'prohibited source member')
    return name


def reviewed_argv(project):
    script, certificate, repo, sources, _, _ = PROJECTS[project]
    args = ['{bundle}/' + project + '/' + script]
    if repo:
        args += ['--repo', '{repo}']
    if sources:
        args += ['--sources', '{bundle}/sources']
    return args + [certificate, '{bundle}/' + project + '/result.json', '--output', '{output}/result.json']


def check_exclusions(manifest, exclusions):
    require(type(exclusions) is list and all(type(row) is dict for row in exclusions), 'invalid exclusions')
    hashes = {row.get('payload_sha256') for row in exclusions} | {INCIDENT_SHA256}
    members = {row.get('member_path') for row in exclusions if row.get('kind') == 'archive_member'}
    for name, expected in manifest.items():
        require(expected['sha256'] not in hashes, 'excluded payload in bundle: ' + name)
        require(not any(member and (name == member or name.endswith('/' + member))
                        for member in members), 'excluded member in bundle: ' + name)


def check_source_metadata(raw, exclusions):
    record = strict_json(raw)
    if not isinstance(record, dict) or 'inventory' not in record:
        return
    forbidden = {row.get('carrier_id') for row in exclusions if row.get('kind') == 'drive_object'} | {INCIDENT_ID}
    require(record.get('drive_id') not in forbidden, 'excluded source identity')
    for row in (record['inventory'], record['coverage']):
        require(row.get('context') == 'RESEARCH_SOURCE_CHECK_STATUS', 'ineligible source context')
        require(row.get('id') not in forbidden, 'excluded source identity')
    name = record['inventory'].get('path', '')
    require(name.startswith('01_ACTIVE_RESEARCH_PACKAGES/'), 'source outside active package')
    require(not any(token in name.lower() for token in ('quarantine', 'legacy', '99_do_not_open')),
            'prohibited source path')


def inspect_archive(raw, exclusions):
    require(identity(raw) == {'bytes': ARCHIVE_BYTES, 'sha256': ARCHIVE_SHA256}, 'canonical archive identity mismatch')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist()
        require(len(infos) == 345 and len({info.filename for info in infos}) == 345, 'archive member set mismatch')
        require(sum(info.file_size for info in infos) <= MAX_EXPANDED, 'expanded archive size limit')
        for info in infos:
            safe_name(info.filename)
            require(stat.S_IFMT(info.external_attr >> 16) in (0, stat.S_IFREG), 'nonregular archive member')
            require(0 <= info.file_size <= MAX_MEMBER and not info.flag_bits & 1, 'member size or encryption rejected')
            require(info.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED), 'unsupported compression')
        manifest_raw = archive.read('MANIFEST.json')
        require(identity(manifest_raw)['sha256'] == MANIFEST_SHA256, 'manifest identity mismatch')
        manifest = strict_json(manifest_raw)['files']
        require(type(manifest) is dict and set(manifest) | {'MANIFEST.json'} == {i.filename for i in infos}
                and len(manifest) == 344, 'manifest coverage mismatch')
        check_exclusions(manifest, exclusions)  # Before materializing any source body.
        for name in manifest:
            if name.endswith(('IDENTITY.json', '.identity.json')):
                data = archive.read(name)
                require(identity(data) == manifest[name], 'metadata identity mismatch')
                check_source_metadata(data, exclusions)
        members = {info.filename: archive.read(info.filename) for info in infos}
    for name, expected in manifest.items():
        require(identity(members[name]) == expected, 'bundle input identity mismatch: ' + name)
    require(identity(members['verification_plan.json'])['sha256'] == PLAN_SHA256, 'verification plan identity mismatch')
    plan = strict_json(members['verification_plan.json'])
    require(plan['schema'] == 'q0-twelve-project-verification-plan-v1', 'unknown plan schema')
    require(len(plan['files']) == 305 and len(plan['repository_dependencies']) == 24, 'incomplete input coverage')
    require(plan['files'] == {name: manifest[name] for name in plan['files']}, 'plan inputs differ from manifest')
    projects = plan['projects']
    require([p['key'] for p in projects] == list(PROJECTS), 'reviewed project set/order mismatch')
    for project in projects:
        name = project['key']
        require(project['argv'] == reviewed_argv(name) and project['working_directory'] == name
                and project['expected_report'] == name + '/result.json'
                and project['result_file'] == 'result.json' and project['timeout_seconds'] == 300,
                'unreviewed command plan')
    return members, plan


def repository_inputs(repo, expected):
    result = {}
    for name, wanted in expected.items():
        # EXCLUSIONS.json is policy metadata, never a quarantined proof body.
        if name != 'quarantine/EXCLUSIONS.json':
            safe_name(name)
        file = repo / name
        require(file.resolve().is_relative_to(repo), 'repository input escapes root')
        result[name] = identity(read_bounded(file, MAX_REPOSITORY_FILE))
        require(result[name] == wanted, 'repository dependency mismatch: ' + name)
    return result


def fresh_output(path, repo):
    path = Path(path).absolute()
    require(path == path.resolve(), 'symlink or noncanonical output path')
    require(not path.is_relative_to(repo) and not path.is_relative_to(ROOT), 'output must be outside repository')
    require(path.parent.is_dir(), 'output parent missing')
    path.mkdir()  # Exclusive; an existing directory or symlink is never reused.
    return path


def run_bounded(argv, cwd, output, stem, timeout_seconds):
    require(type(timeout_seconds) in (int, float) and 0 < timeout_seconds <= 300, 'invalid timeout')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
    started = time.monotonic()
    logs = {name: output / (stem + '.' + name + '.txt') for name in ('stdout', 'stderr')}
    with logs['stdout'].open('xb') as stdout, logs['stderr'].open('xb') as stderr:
        process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, start_new_session=True)
        finished = False
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ, stdout)
                selector.register(process.stderr, selectors.EVENT_READ, stderr)
                size = 0
                while selector.get_map():
                    require(time.monotonic() - started < timeout_seconds, 'subprocess timeout')
                    for key, _ in selector.select(min(0.05, timeout_seconds)):
                        data = os.read(key.fileobj.fileno(), 65536)
                        if not data:
                            selector.unregister(key.fileobj)
                            continue
                        remaining = MAX_LOG - size
                        key.data.write(data[:remaining])
                        size += len(data)
                        require(size <= MAX_LOG, 'subprocess log size limit')
                process.wait(timeout=max(0.001, timeout_seconds - (time.monotonic() - started)))
                finished = True
        finally:
            if not finished or process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            process.stdout.close()
            process.stderr.close()
    record = {'argv': argv, 'returncode': process.returncode,
              'elapsed_seconds': round(time.monotonic() - started, 6),
              'logs': {name: identity(read_bounded(path, MAX_LOG)) for name, path in logs.items()}}
    require(process.returncode == 0, 'subprocess failed: ' +
            (logs['stdout'].read_text(errors='replace') + logs['stderr'].read_text(errors='replace'))[-2000:])
    return record


def compare_report(actual, expected):
    # Byte equality retains boolean/int distinctions and every scope/assumption.
    require(actual == expected, 'fresh mathematical report differs from frozen candidate')


def publish_report(output, report):
    temporary = output / 'report.json.partial'
    with temporary.open('xb') as handle:
        handle.write((json.dumps(report, sort_keys=True, indent=2) + '\n').encode())
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, output / 'report.json')  # Atomic publication, no overwrite.
    temporary.unlink()


def run_project(project, *, repo=ROOT, archive=None, output_dir=None, timeout_seconds=60, author_tests=False):
    require(project in PROJECTS, 'unknown project')
    require(type(timeout_seconds) in (int, float) and 0 < timeout_seconds <= 300, 'invalid timeout')
    require(sys.version_info[:2] == (3, 11) and not sys.flags.optimize
            and not os.environ.get('PYTHONOPTIMIZE'), 'normal Python 3.11 required')
    repo = Path(repo).resolve()
    archive = Path(archive) if archive is not None else repo / ARCHIVE
    # Resolve the OS temp root first (macOS /var may itself be a symlink).
    with tempfile.TemporaryDirectory(prefix='q0-twelve-', dir=Path(tempfile.gettempdir()).resolve()) as temporary:
        work = Path(temporary)
        require(not work.is_relative_to(repo), 'temporary extraction must be outside repository')
        output = fresh_output(output_dir if output_dir is not None else work / 'output', repo)
        report = {'schema': 'q0-twelve-project-portable-replay-v1', 'project': project,
                  'started_utc': datetime.now(timezone.utc).isoformat(), 'passed': False,
                  'scientific_status_changed': False, 'original_prize_closed': False,
                  'organizational_independence_credit': 0, 'source_exposed': True,
                  'source_bodies_executed': False, 'snapshot_normalization': [],
                  'python': sys.version, 'author_tests': None}
        deadline = time.monotonic() + timeout_seconds
        try:
            exclusions = strict_json(read_bounded(repo / 'quarantine/EXCLUSIONS.json', MAX_JSON))['exclusions']
            raw = read_bounded(archive, ARCHIVE_BYTES)
            members, plan = inspect_archive(raw, exclusions)
            expected = plan['repository_dependencies'] | SUPPLEMENTAL_DEPENDENCIES
            check_exclusions(expected, exclusions)
            before = repository_inputs(repo, expected)
            report['archive'] = identity(raw)
            report['repository_dependencies'] = before
            report['original_dependency_count'] = 24
            report['supplemental_dependency_count'] = len(SUPPLEMENTAL_DEPENDENCIES)
            detail = next(item for item in plan['projects'] if item['key'] == project)
            report['technical_scope'] = detail['technical_scope']
            report['remaining_obligations'] = detail['remaining_obligations']
            bundle = work / 'bundle'
            for name, data in members.items():
                path = bundle / name
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as handle:
                    handle.write(data)
            argv = [sys.executable, '-B'] + [part.replace('{bundle}', str(bundle)).replace('{repo}', str(repo))
                    .replace('{output}', str(output)) for part in reviewed_argv(project)]
            report['execution'] = run_bounded(argv, bundle / project, output, 'candidate', deadline - time.monotonic())
            actual = read_bounded(output / 'result.json', MAX_MEMBER)
            compare_report(actual, members[project + '/result.json'])
            report['result'] = identity(actual)
            if author_tests:
                require(repo == Path(plan['repository']), 'original author tests are host-bound; portable replay does not run them')
                test, count = PROJECTS[project][4:]
                pytest_mode = project in ('lpw_amplitude', 'lpw_headline')
                args = [sys.executable, '-B']
                if pytest_mode:
                    args += ['-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                             '--junitxml=' + str(output / 'author-tests.xml')]
                args += [str(bundle / project / test)]
                tested = run_bounded(args, bundle / project, output, 'author-tests', deadline - time.monotonic())
                if pytest_mode:
                    tree = ET.fromstring(read_bounded(output / 'author-tests.xml', MAX_JSON))
                    suites = list(tree.iter('testsuite'))
                    require(sum(int(s.get('tests', '-1')) for s in suites) == count
                            and len(list(tree.iter('testcase'))) == count
                            and not list(tree.iter('failure')) and not list(tree.iter('error'))
                            and not list(tree.iter('skipped')), 'author test counts/outcomes differ')
                else:
                    text = read_bounded(output / 'author-tests.stderr.txt', MAX_LOG).decode()
                    require(re.search(r'Ran ' + str(count) + r' tests in ', text) is not None
                            and re.search(r'^OK\s*$', text, re.MULTILINE) is not None,
                            'author test count/success summary missing')
                report['author_tests'] = {**tested, 'tests': count, 'scope': 'original host-bound test methods; includes subTests'}
            require({str(path.relative_to(bundle)) for path in bundle.rglob('*') if path.is_file()} == set(members),
                    'extracted input file set changed')
            for name, data in members.items():
                require(read_bounded(bundle / name, MAX_MEMBER) == data, 'extracted input changed: ' + name)
            require(repository_inputs(repo, before) == before and read_bounded(archive, ARCHIVE_BYTES) == raw,
                    'inputs changed during replay')
            compare_report(read_bounded(output / 'result.json', MAX_MEMBER), members[project + '/result.json'])
            expected_outputs = {'candidate.stdout.txt', 'candidate.stderr.txt', 'result.json'}
            executions = [('candidate', report['execution'])]
            if author_tests:
                expected_outputs |= {'author-tests.stdout.txt', 'author-tests.stderr.txt'}
                executions.append(('author-tests', report['author_tests']))
                if project in ('lpw_amplitude', 'lpw_headline'):
                    expected_outputs.add('author-tests.xml')
            require({path.name for path in output.iterdir()} == expected_outputs, 'unexpected output artifact set')
            for stem, execution in executions:
                for name, wanted in execution['logs'].items():
                    require(identity(read_bounded(output / (stem + '.' + name + '.txt'), MAX_LOG)) == wanted,
                            'execution log changed after replay')
            report['inputs_unchanged'] = True
            report['passed'] = True
        except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, subprocess.SubprocessError, ET.ParseError) as error:
            report['error'] = type(error).__name__ + ': ' + str(error)
        report['ended_utc'] = datetime.now(timezone.utc).isoformat()
        report['artifacts'] = {path.name: identity(read_bounded(path, MAX_LOG))
                               for path in sorted(output.iterdir()) if path.is_file()}
        publish_report(output, report)
        return report


def run_campaign(*, repo=ROOT, archive=None, output_dir=None, timeout_seconds=60,
                 budget_seconds=300, author_tests=False):
    require(type(budget_seconds) in (int, float) and 0 < budget_seconds <= 3600, 'invalid campaign budget')
    repo = Path(repo).resolve()
    with tempfile.TemporaryDirectory(prefix='q0-twelve-campaign-', dir=Path(tempfile.gettempdir()).resolve()) as temporary:
        output = fresh_output(output_dir if output_dir is not None else Path(temporary) / 'output', repo)
        started = time.monotonic()
        report = {'schema': 'q0-twelve-project-portable-campaign-v1', 'passed': False,
                  'started_utc': datetime.now(timezone.utc).isoformat(), 'results': [],
                  'budget_seconds': budget_seconds, 'scientific_status_changed': False,
                  'original_prize_closed': False, 'organizational_independence_credit': 0,
                  'source_exposed': True, 'source_bodies_executed': False}
        try:
            for project in PROJECTS:
                remaining = budget_seconds - (time.monotonic() - started)
                require(remaining > 0, 'campaign budget exhausted before all twelve projects')
                result = run_project(project, repo=repo, archive=archive, output_dir=output / project,
                                     timeout_seconds=min(timeout_seconds, remaining), author_tests=author_tests)
                report['results'].append(result)
            require(len(report['results']) == 12 and all(result['passed'] for result in report['results']),
                    'one or more of the twelve projects failed')
            require(time.monotonic() - started <= budget_seconds, 'campaign budget exhausted')
            require(repository_inputs(repo, report['results'][0]['repository_dependencies']) ==
                    report['results'][0]['repository_dependencies'], 'campaign repository inputs changed')
            require(identity(read_bounded(Path(archive) if archive else repo / ARCHIVE, ARCHIVE_BYTES)) ==
                    report['results'][0]['archive'], 'campaign archive changed')
            for result in report['results']:
                child = output / result['project']
                require(read_bounded(child / 'report.json', MAX_JSON) ==
                        (json.dumps(result, sort_keys=True, indent=2) + '\n').encode(), 'project receipt changed')
                require({path.name for path in child.iterdir()} == set(result['artifacts']) | {'report.json'},
                        'project artifact set changed')
                for name, wanted in result['artifacts'].items():
                    require(identity(read_bounded(child / name, MAX_LOG)) == wanted, 'project artifact changed')
            report['passed'] = True
        except (OSError, ValueError, KeyError, TypeError) as error:
            report['error'] = type(error).__name__ + ': ' + str(error)
        report['ended_utc'] = datetime.now(timezone.utc).isoformat()
        report['elapsed_seconds'] = round(time.monotonic() - started, 6)
        report['project_receipts'] = {path.parent.name: identity(read_bounded(path, MAX_JSON))
                                     for path in sorted(output.glob('*/report.json'))}
        publish_report(output, report)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', choices=PROJECTS, nargs='?')
    parser.add_argument('--repo', type=Path, default=ROOT)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--timeout-seconds', type=float, default=60)
    parser.add_argument('--budget-seconds', type=float, default=300)
    parser.add_argument('--author-tests', action='store_true')
    args = parser.parse_args(argv)
    try:
        kwargs = dict(repo=args.repo, archive=args.archive, output_dir=args.output_dir,
                      timeout_seconds=args.timeout_seconds, author_tests=args.author_tests)
        report = (run_project(args.project, **kwargs) if args.project is not None else
                  run_campaign(**kwargs, budget_seconds=args.budget_seconds))
        if not report['passed']:
            raise ValueError(report['error'])
        print('PASS: ' + (args.project + '; report_sha256=' + report['result']['sha256'] if args.project else '12/12 projects') +
              '; exact original candidates; independence=0; no scientific promotion')
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print('REJECTED: ' + str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
