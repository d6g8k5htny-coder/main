"""Successor intake, exact output accounting and minimal-root replay controls."""
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import zipfile

import pytest

from tools import h3_rn_n6_check as checker


ARCHIVE = Path(os.environ.get('H3_RN_N6_TEST_ARCHIVE', checker.ROOT / checker.ARCHIVE))


@pytest.fixture(scope='module')
def original():
    raw = ARCHIVE.read_bytes()
    exclusions = json.loads((checker.ROOT / 'quarantine/EXCLUSIONS.json').read_bytes())['exclusions']
    members, deps = checker.inspect_archive(raw, exclusions)
    return raw, exclusions, members, deps


@pytest.fixture
def emitted(original, tmp_path):
    """Frozen expected-output fixture; the relocation test performs real execution."""
    members = original[2]
    output = tmp_path / 'emitted'
    output.mkdir()
    receipt = json.loads(members['VERIFICATION.json'])
    (output / 'VERIFICATION.json').write_bytes(members['VERIFICATION.json'])
    for job in checker.reviewed_jobs():
        for mode in checker.MODES:
            dest = output / (job['name'] + '-' + mode)
            dest.mkdir()
            row = next(r for r in receipt['jobs'] if (r['job'], r['mode']) == (job['name'], mode))
            (dest / 'stdout.txt').write_text(row['stdout'] + '\n')
            (dest / 'stderr.txt').write_bytes(b'')
            for pair in job['outputs']:
                target = dest / ('results' if job['name'] == 'rn_n6' else '') / pair['generated']
                target.parent.mkdir(exist_ok=True)
                target.write_bytes(members[pair['original']])
    return output


def test_complete_frozen_output_fixture(original, emitted):
    matches, artifacts = checker.verify_outputs(emitted, original[2])
    assert len(matches) == 34
    assert len(artifacts) == 55
    assert sum(path['original'].startswith('rn_n6/') for path in matches) == 26


def test_real_ten_replays_in_minimal_relocated_root(original, tmp_path):
    repo = tmp_path / 'unrelated-host' / 'repository'
    for name in original[3]:
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checker.ROOT / name, target)
    archive = repo / checker.ARCHIVE
    archive.parent.mkdir(parents=True)
    archive.write_bytes(original[0])
    assert len([p for p in repo.rglob('*') if p.is_file()]) == 26
    report = checker.run(repo=repo, output_dir=tmp_path / 'receipt')
    assert report['passed'], report
    assert report['actual_executions'] == 10
    assert len(report['byte_identical_outputs']) == 34
    assert report['bundle_inputs_unchanged'] == 71
    assert report['repository_inputs_unchanged'] == 25
    assert report['snapshot_normalization'] == []
    assert report['organizational_independence_credit'] == 0
    assert json.loads((tmp_path / 'receipt/report.json').read_bytes()) == report


def test_tampered_archive_rejected(original):
    raw = original[0]
    with pytest.raises(ValueError, match='archive identity'):
        checker.inspect_archive(raw[:-1] + bytes([raw[-1] ^ 1]), original[1])


@pytest.mark.parametrize('mutation', ['escape', 'symlink', 'extra', 'omitted'])
def test_archive_structure_is_independently_guarded(original, monkeypatch, mutation):
    source = zipfile.ZipFile(io.BytesIO(original[0]))
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as target:
        for i, item in enumerate(source.infolist()):
            data = source.read(item.filename)
            if i == 0:
                if mutation == 'escape':
                    item.filename = '../outside'
                elif mutation == 'symlink':
                    item.external_attr = (stat.S_IFLNK | 0o777) << 16
                elif mutation == 'omitted':
                    continue
            target.writestr(item, data)
        if mutation == 'extra':
            target.writestr('unreviewed.py', b'raise SystemExit(0)')
    raw = output.getvalue()
    monkeypatch.setattr(checker, 'ARCHIVE_BYTES', len(raw))
    monkeypatch.setattr(checker, 'ARCHIVE_SHA256', checker.runtime.identity(raw)['sha256'])
    with pytest.raises(ValueError):
        checker.inspect_archive(raw, original[1])


def test_current_exclusions_block_before_body_materialization(original, monkeypatch):
    target = 'h3_floor/support/check_h3_uniform.py'
    exclusions = original[1] + [{'payload_sha256': checker.runtime.identity(original[2][target])['sha256']}]
    reads = []
    read = zipfile.ZipFile.read

    def spy(self, name, *args, **kwargs):
        reads.append(name)
        return read(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, 'read', spy)
    with pytest.raises(ValueError, match='excluded payload'):
        checker.inspect_archive(original[0], exclusions)
    assert reads == ['MANIFEST.json']


@pytest.mark.parametrize('name', ['h3_floor-normal/report.json', 'h3_ceiling-optimized/report.json',
                                  'rn_n6-normal/results/RUN.json',
                                  'rn_n6-optimized/results/pilot_admitted_M.json'])
def test_any_mathematical_output_tamper_rejects(original, emitted, name):
    path = emitted / name
    raw = path.read_bytes()
    path.write_bytes(raw.replace(b'false', b'true', 1) if b'false' in raw else raw + b' ')
    with pytest.raises(ValueError, match='mathematical report'):
        checker.verify_outputs(emitted, original[2])


def test_missing_one_of_thirty_four_outputs_rejects(original, emitted):
    (emitted / 'rn_n6-optimized/results/y_inner_admitted_y.json').unlink()
    with pytest.raises((OSError, ValueError)):
        checker.verify_outputs(emitted, original[2])


def test_unexpected_output_is_not_hidden_by_passing_known_outputs(original, emitted):
    (emitted / 'extra.json').write_bytes(b'{}')
    with pytest.raises(ValueError, match='unexpected output'):
        checker.verify_outputs(emitted, original[2])


@pytest.mark.parametrize('mutation', ['missing_job', 'duplicate_job', 'missing_comparison', 'failed_child',
                                      'bool_exit_code', 'promoted', 'independence', 'float_count', 'wrong_hash'])
def test_runtime_receipt_cannot_claim_incomplete_or_false_success(original, emitted, mutation):
    path = emitted / 'VERIFICATION.json'
    value = json.loads(path.read_bytes())
    if mutation == 'missing_job':
        value['jobs'].pop()
    elif mutation == 'duplicate_job':
        value['jobs'][-1] = deepcopy(value['jobs'][0])
    elif mutation == 'missing_comparison':
        value['jobs'][-1]['byte_identical_outputs'].pop()
    elif mutation == 'failed_child':
        value['jobs'][0]['exit_code'] = 3
    elif mutation == 'bool_exit_code':
        value['jobs'][0]['exit_code'] = False
    elif mutation == 'promoted':
        value['scientific_status_changed'] = True
    elif mutation == 'independence':
        value['organizational_independence_credit'] = 1
    elif mutation == 'float_count':
        value['repository_inputs_unchanged'] = 25.0
    else:
        value['jobs'][0]['byte_identical_outputs'][0]['sha256'] = '0' * 64
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        checker.verify_outputs(emitted, original[2])


@pytest.mark.parametrize('raw', [b'{"a":1,"a":2}', b'{"elapsed":NaN}', b'{"elapsed":Infinity}'])
def test_runtime_json_rejects_duplicates_and_nonfinite_numbers(raw):
    with pytest.raises(ValueError):
        checker.runtime_receipt(raw)


def test_budget_exhaustion_retains_failure_receipt(tmp_path):
    report = checker.run(archive=ARCHIVE, output_dir=tmp_path / 'expired', budget_seconds=0.000001)
    assert not report['passed']
    assert 'budget' in report['error']
    assert json.loads((tmp_path / 'expired/report.json').read_bytes())['passed'] is False


def cli(*args):
    return subprocess.run([sys.executable, '-B', str(Path(checker.__file__)), '--archive', str(ARCHIVE), *map(str, args)],
                          capture_output=True, text=True, timeout=10,
                          env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(checker.ROOT)))


def test_cli_refuses_existing_output_without_overwrite(tmp_path):
    output = tmp_path / 'existing'
    output.mkdir()
    marker = output / 'keep'
    marker.write_bytes(b'unchanged')
    result = cli('--output-dir', output)
    assert result.returncode != 0
    assert marker.read_bytes() == b'unchanged'


def test_cli_refuses_repository_output():
    target = checker.ROOT / 'must-not-exist-h3-rn-n6-output'
    result = cli('--output-dir', target)
    assert result.returncode != 0
    assert 'outside repository' in result.stderr
    assert not target.exists()


def test_cli_bad_archive_is_failure_with_retained_receipt(tmp_path):
    bad = tmp_path / 'bad.zip'
    bad.write_bytes(b'not a delivery archive')
    result = cli('--archive', bad, '--output-dir', tmp_path / 'failed')
    assert result.returncode != 0
    report = json.loads((tmp_path / 'failed/report.json').read_bytes())
    assert not report['passed']
    assert 'archive identity' in report['error']


def test_default_cli_uses_whole_reviewed_campaign(monkeypatch):
    calls = []
    monkeypatch.setattr(checker, 'run', lambda **kwargs: calls.append(kwargs) or {'passed': True})
    assert checker.main([]) == 0
    assert len(calls) == 1
    assert calls[0]['budget_seconds'] == 180
