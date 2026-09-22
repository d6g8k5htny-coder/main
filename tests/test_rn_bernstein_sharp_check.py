"""Admission and replay-binding controls; archived author tests stay frozen."""
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest

from tools import rn_bernstein_sharp_check as checker


@pytest.fixture(scope='module')
def original():
    raw = (checker.ROOT / checker.ARCHIVE).read_bytes()
    exclusions = checker.runtime.strict_json((checker.ROOT / 'quarantine/EXCLUSIONS.json').read_bytes())['exclusions']
    members, deps = checker.inspect_archive(raw, exclusions)
    return raw, exclusions, members, deps


@pytest.fixture
def relocated(original, tmp_path):
    repo = tmp_path / 'minimal-repository'
    for name in original[3]:
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checker.ROOT / name, target)
    return repo


def emit(output, members, mode):
    """Synthetic exact-output fixture, not a fresh mathematical execution."""
    output.mkdir()
    (output / 'spatial').mkdir()
    for generated, archived in checker.REPORTS.items():
        (output / generated).write_bytes(members[archived])
    for entry in checker.runtime.strict_json(members['bernstein/result.json'])['attempts']:
        (output / ('spatial/attempt_1_' + entry['method'] + '.json')).write_bytes(checker.canonical(entry))
    timings = dict(timings=[dict(pieces=1, method=method, elapsed_seconds=0.01) for method in ('original', 'bernstein')],
                   python=sys.version, optimize=checker.MODES.index(mode),
                   scope='Local observed wall time only; includes warm-cache effects, not an independent compute-cost experiment.')
    (output / 'spatial/timings.json').write_bytes(checker.canonical(timings))
    for name in ('replay-0.stdout', 'replay-0.stderr', 'replay-1.stdout', 'replay-1.stderr'):
        (output / name).write_bytes(b'')


@pytest.fixture
def emitted(original, tmp_path):
    output = tmp_path / 'emitted'
    emit(output, original[2], 'normal')
    return output


@pytest.fixture
def synthetic_execution(original, monkeypatch):
    # Exercise downstream controls under pytest -O too. The real parent runtime
    # gate has a separate normal/-O CLI control, without this test double.
    monkeypatch.setattr(checker, 'sys', SimpleNamespace(version_info=(3, 11), flags=SimpleNamespace(optimize=0),
                                                     executable=sys.executable, version=sys.version))
    calls = []

    def execute(argv, cwd, output, stem, timeout):
        assert 0 < timeout <= 300
        assert argv == [sys.executable, '-B'] + (['-O'] if stem == 'optimized' else []) + [
            str(cwd / 'close_local_wedge.py'), '--repo', argv[-3], '--output', str(output / stem)]
        emit(output / stem, original[2], stem)
        for channel in ('stdout', 'stderr'):
            (output / (stem + '.' + channel + '.txt')).write_bytes(b'')
        calls.append((argv, cwd, output, stem))
        return dict(argv=argv, returncode=0, elapsed_seconds=0.01,
                    logs={name: checker.runtime.identity(b'') for name in ('stdout', 'stderr')})

    monkeypatch.setattr(checker.runtime, 'run_bounded', execute)
    return calls, execute


def test_exact_intake_and_relocated_admission(original, relocated):
    assert len(original[2]) == 48 and len(original[3]) == 36
    assert len([p for p in relocated.rglob('*') if p.is_file()]) == 36
    checked, sources = checker.admitted_inputs(relocated, original[3])
    assert checked == original[3]
    assert sources['research/campaigns/h3_rn_n6_20260920_v1.zip']['sha256'] == '73b9e63800f77c677c504f3097fe0bb2459ff6c94cdee5772b56062320aefc21'


def test_exact_output_fixture_retains_refusal_and_success(original, emitted):
    assert len(checker.verify_outputs(emitted, original[2], 'normal')) == 10
    attempts = checker.runtime.strict_json(original[2]['bernstein/result.json'])['attempts']
    assert attempts[0]['outcome'] == 'RETAINED_REFUSAL'
    assert attempts[1]['outcome'] == 'CERTIFIED_SCOPED_BOUND'
    result = checker.runtime.strict_json(original[2]['combined-result.json'])
    assert result['spatial_auxiliary_rectangles'] == 1
    assert result['mahalanobis_lower_certified'] == '103'
    assert result['scope']['full_annulus_closed'] is False
    assert result['imported_h3_and_pin_energy_reproved'] is False


@pytest.mark.parametrize('mutation', ['digest', 'duplicate', 'escape', 'symlink', 'extra', 'missing'])
def test_archive_corruption_rejected(original, monkeypatch, mutation):
    if mutation == 'digest':
        raw = original[0][:-1] + bytes([original[0][-1] ^ 1])
    else:
        output = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(original[0])) as source, zipfile.ZipFile(output, 'w') as target:
            for index, item in enumerate(source.infolist()):
                data = source.read(item.filename)
                if index == 0:
                    if mutation == 'escape':
                        item.filename = '../escape'
                    elif mutation == 'symlink':
                        item.external_attr = (stat.S_IFLNK | 0o777) << 16
                    elif mutation == 'missing':
                        continue
                target.writestr(item, data)
            if mutation == 'extra':
                target.writestr('unreviewed.py', b'')
            if mutation == 'duplicate':
                with pytest.warns(UserWarning, match='Duplicate'):
                    target.writestr('CLAIMS.json', b'{}')
        raw = output.getvalue()
        monkeypatch.setattr(checker, 'ARCHIVE_BYTES', len(raw))
        monkeypatch.setattr(checker, 'ARCHIVE_SHA256', checker.runtime.identity(raw)['sha256'])
    with pytest.raises(ValueError):
        checker.inspect_archive(raw, original[1])


def test_duplicate_manifest_keys_rejected_beyond_outer_digest(original, monkeypatch):
    output = io.BytesIO()
    manifest = original[2]['MANIFEST.json'].replace(b'{', b'{"schema":"duplicate",', 1)
    with zipfile.ZipFile(output, 'w') as target:
        for name, raw in original[2].items():
            target.writestr(name, manifest if name == 'MANIFEST.json' else raw)
    raw = output.getvalue()
    monkeypatch.setattr(checker, 'ARCHIVE_BYTES', len(raw))
    monkeypatch.setattr(checker, 'ARCHIVE_SHA256', checker.runtime.identity(raw)['sha256'])
    manifests = deepcopy(checker.MANIFESTS)
    manifests['MANIFEST.json'] = (checker.runtime.identity(manifest)['sha256'], *manifests['MANIFEST.json'][1:])
    monkeypatch.setattr(checker, 'MANIFESTS', manifests)
    monkeypatch.setattr(checker, 'MANIFEST_BYTES', dict(checker.MANIFEST_BYTES, **{'MANIFEST.json': len(manifest)}))
    with pytest.raises(ValueError, match='duplicate'):
        checker.inspect_archive(raw, original[1])


@pytest.mark.parametrize('whole_archive', [False, True])
def test_exclusion_precedes_packet_body_reads(original, monkeypatch, whole_archive):
    digest = checker.ARCHIVE_SHA256 if whole_archive else checker.runtime.identity(original[2]['bernstein/successor_wedge_v2.py'])['sha256']
    excluded = original[1] + [{'payload_sha256': digest}]
    reads, actual = [], zipfile.ZipFile.read

    def read(self, name, *args, **kwargs):
        reads.append(name)
        return actual(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, 'read', read)
    with pytest.raises(ValueError, match='excluded payload'):
        checker.inspect_archive(original[0], excluded)
    assert reads == ([] if whole_archive else ['MANIFEST.json'])


@pytest.mark.parametrize('name', ['MANIFEST.json', 'bernstein/MANIFEST.json', 'sharp_variance/MANIFEST.json'])
def test_manifest_exclusion_refuses_before_member_reads(original, monkeypatch, name):
    excluded = original[1] + [{'payload_sha256': checker.MANIFESTS[name][0]}]

    def forbidden(*args, **kwargs):
        pytest.fail('excluded manifest must be rejected before any ZIP member read')

    monkeypatch.setattr(zipfile.ZipFile, 'read', forbidden)
    with pytest.raises(ValueError, match='excluded payload'):
        checker.inspect_archive(original[0], excluded)


@pytest.mark.parametrize('mutation', ['excluded_id', 'upstream_excluded', 'unknown_exclusion', 'malformed_policy',
                                    'inventory_held', 'coverage_held', 'code_changed', 'initializer'])
def test_live_source_gate_precedes_historical_body_reads(original, relocated, monkeypatch, mutation):
    if mutation in ('excluded_id', 'upstream_excluded', 'unknown_exclusion', 'malformed_policy'):
        path = relocated / 'quarantine/EXCLUSIONS.json'
        value = json.loads(path.read_bytes())
        ids = {'excluded_id': '1LtnvNd0vAW-y3pzbyHLjgtF7Uw5sTph5',
               'upstream_excluded': '1g5UP_KYlvPmx7LFwjGk6qK5QZLnL1hLj', 'unknown_exclusion': 'newly-held-source'}
        if mutation == 'malformed_policy':
            path.write_bytes(b'{"exclusions":[],"exclusions":[]}')
        else:
            value['exclusions'].append({'kind': 'drive_object', 'carrier_id': ids[mutation]})
            path.write_bytes(checker.canonical(value))
    elif mutation == 'inventory_held':
        path = relocated / 'drive/inventory.jsonl'
        rows = [json.loads(line) for line in path.read_bytes().splitlines()]
        next(row for row in rows if row.get('id') == '1LtnvNd0vAW-y3pzbyHLjgtF7Uw5sTph5')['context'] = 'HELD'
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    elif mutation == 'coverage_held':
        path = relocated / 'drive/source_map/Payloads.csv'
        path.write_text(path.read_text().replace('RESEARCH_SOURCE_CHECK_STATUS', 'HELD'))
    elif mutation == 'code_changed':
        with (relocated / 'research/rn/n6_inputs.py').open('ab') as stream:
            stream.write(b'\nraise RuntimeError("must not execute")\n')
    else:
        (relocated / 'research/__init__.py').write_text('raise RuntimeError("must not execute")\n')
    body_reads, actual = [], checker.runtime.read_bounded

    def read(path, limit):
        if Path(path).name in {'RN5_NEAR_MOMENT_REPAIR.md', 'RN5_REPAIR_AND_ERRATUM_BUNDLE.zip', 'h3_rn_n6_20260920_v1.zip'}:
            body_reads.append(str(path))
        return actual(path, limit)

    monkeypatch.setattr(checker.runtime, 'read_bounded', read)
    with pytest.raises(ValueError):
        checker.admitted_inputs(relocated, original[3])
    assert body_reads == []


@pytest.mark.parametrize('name', ['spatial/result.json', 'sharp.json', 'result.json', 'spatial/attempt_1_original.json', 'spatial/attempt_1_bernstein.json'])
def test_each_proof_and_refusal_output_is_load_bearing(original, emitted, name):
    with (emitted / name).open('ab') as stream:
        stream.write(b' ')
    with pytest.raises(ValueError, match='differs'):
        checker.verify_outputs(emitted, original[2], 'normal')


@pytest.mark.parametrize('mutation', ['extra', 'missing', 'symlink', 'empty_directory', 'promote_bool'])
def test_output_set_and_scope_mutations_rejected(original, emitted, mutation):
    if mutation == 'extra':
        (emitted / 'other.json').write_text('{}')
    elif mutation == 'empty_directory':
        (emitted / 'unreported').mkdir()
    elif mutation == 'missing':
        (emitted / 'replay-0.stderr').unlink()
    elif mutation == 'symlink':
        (emitted / 'result.json').unlink()
        (emitted / 'result.json').symlink_to(emitted / 'sharp.json')
    else:
        path = emitted / 'result.json'
        value = json.loads(path.read_bytes())
        value['scope']['scientific_status_changed'] = 0
        path.write_bytes(checker.canonical(value))
    with pytest.raises(ValueError):
        checker.verify_outputs(emitted, original[2], 'normal')


@pytest.mark.parametrize('mutation', ['bool_count', 'bool_elapsed', 'nan', 'infinity', 'negative', 'wrong_mode', 'duplicate_key'])
def test_timing_receipt_typed_and_noncertifying(original, emitted, mutation):
    path = emitted / 'spatial/timings.json'
    value = json.loads(path.read_bytes())
    if mutation == 'bool_count':
        value['timings'][0]['pieces'] = True
    elif mutation == 'wrong_mode':
        value['optimize'] = 1
    elif mutation == 'duplicate_key':
        path.write_bytes(path.read_bytes().replace(b'{', b'{"python":"duplicate",', 1))
    else:
        value['timings'][0]['elapsed_seconds'] = {'bool_elapsed': False, 'nan': float('nan'), 'infinity': float('inf'), 'negative': -1}[mutation]
    if mutation != 'duplicate_key':
        path.write_bytes(checker.canonical(value))
    with pytest.raises(ValueError):
        checker.verify_outputs(emitted, original[2], 'normal')


def test_synthetic_full_run_binds_exact_argv_counts_and_relocated_inputs(original, relocated, synthetic_execution, tmp_path):
    archive = tmp_path / 'packet.zip'
    archive.write_bytes(original[0])
    report = checker.run(repo=relocated, archive=archive, output_dir=tmp_path / 'out')
    assert report['passed'], report
    assert [call[3] for call in synthetic_execution[0]] == ['normal', 'optimized']
    assert report['composition_executions'] == 2 and report['numerical_child_executions'] == 4
    assert report['byte_identical_frozen_reports'] == 6 and report['canonical_attempt_matches'] == 4
    assert report['repository_inputs_unchanged'] == 36 and report['bundle_inputs_unchanged'] == 48
    assert report['scientific_status_changed'] is False and report['organizational_independence_credit'] == 0
    assert json.loads((tmp_path / 'out/report.json').read_bytes()) == report


@pytest.mark.parametrize('mutation', ['source', 'archive', 'bundle', 'first_output', 'first_log', 'timeout'])
def test_after_execution_drift_and_failures_cannot_publish_pass(original, relocated, synthetic_execution, monkeypatch, tmp_path, mutation):
    archive = tmp_path / 'packet.zip'
    archive.write_bytes(original[0])
    calls, execute = synthetic_execution

    def altered(argv, cwd, output, stem, timeout):
        if mutation == 'timeout':
            raise subprocess.TimeoutExpired(argv, timeout)
        result = execute(argv, cwd, output, stem, timeout)
        if stem == 'optimized':
            targets = {'source': relocated / 'research/rn/side24.py', 'archive': archive,
                       'bundle': cwd / 'extra.py', 'first_output': output / 'normal/result.json',
                       'first_log': output / 'normal.stdout.txt'}
            with targets[mutation].open('ab') as stream:
                stream.write(b' ')
        return result

    monkeypatch.setattr(checker.runtime, 'run_bounded', altered)
    report = checker.run(repo=relocated, archive=archive, output_dir=tmp_path / 'out')
    assert report['passed'] is False and 'error' in report
    assert 'byte_identical_frozen_reports' not in report
    assert json.loads((tmp_path / 'out/report.json').read_bytes())['passed'] is False


@pytest.mark.parametrize('budget', [True, 0, -1, 301, float('nan'), float('inf')])
def test_invalid_budgets_refuse_before_output(synthetic_execution, tmp_path, budget):
    with pytest.raises(ValueError, match='budget'):
        checker.run(output_dir=tmp_path / 'out', budget_seconds=budget)
    assert not (tmp_path / 'out').exists() and synthetic_execution[0] == []


@pytest.mark.parametrize('location', ['selected_repo', 'wrapper_repo', 'existing', 'symlink'])
def test_output_protection(relocated, synthetic_execution, tmp_path, location):
    if location == 'selected_repo':
        output = relocated / 'forbidden'
    elif location == 'wrapper_repo':
        output = checker.ROOT / 'must-not-be-created-by-rn-test'
    elif location == 'existing':
        output = tmp_path / 'existing'
        output.mkdir()
        (output / 'sentinel').write_text('preserve')
    else:
        output = tmp_path / 'alias'
        output.symlink_to(relocated, target_is_directory=True)
    with pytest.raises((ValueError, FileExistsError)):
        checker.run(repo=relocated, output_dir=output)
    assert synthetic_execution[0] == []
    if location == 'existing':
        assert (output / 'sentinel').read_text() == 'preserve'


@pytest.mark.parametrize('optimized', [False, True])
def test_cli_runtime_and_archive_failure_are_real_nonzero(tmp_path, optimized):
    archive = tmp_path / 'wrong.zip'
    archive.write_bytes(b'not the reviewed archive')
    output = tmp_path / 'out'
    command = [sys.executable, '-B'] + (['-O'] if optimized else []) + [str(Path(checker.__file__).resolve()),
              '--archive', str(archive), '--output-dir', str(output)]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONOPTIMIZE', None)
    result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=10)
    assert result.returncode == 1
    assert ('normal Python 3.11 parent required' if optimized else 'archive identity mismatch') in result.stderr
    if optimized:
        assert not output.exists()
    else:
        assert json.loads((output / 'report.json').read_bytes())['passed'] is False
