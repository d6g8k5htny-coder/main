"""Portable replay and fail-closed intake controls; never collect frozen tests."""
from copy import deepcopy
from contextlib import nullcontext
import io
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import time
import zipfile

import pytest

from tools import twelve_project_check as checker


@pytest.fixture(scope='module')
def original():
    raw = (checker.ROOT / checker.ARCHIVE).read_bytes()
    exclusions = json.loads((checker.ROOT / 'quarantine/EXCLUSIONS.json').read_bytes())['exclusions']
    members, plan = checker.inspect_archive(raw, exclusions)
    return raw, exclusions, members, plan


@pytest.mark.parametrize('project', tuple(checker.PROJECTS))
def test_real_original_candidate_replay(project, tmp_path):
    result = checker.run_project(project, output_dir=tmp_path / project)
    assert result['passed'], result
    assert result['inputs_unchanged']
    assert result['snapshot_normalization'] == []
    assert result['organizational_independence_credit'] == 0
    assert result['source_bodies_executed'] is False
    assert result['remaining_obligations']
    assert result['author_tests'] is None
    assert json.loads((tmp_path / project / 'report.json').read_bytes()) == result


def test_all_twelve_relocate_to_an_unrelated_repository(original, tmp_path):
    _, _, _, plan = original
    repo = tmp_path / 'different-machine' / 'repository'
    for name in [checker.ARCHIVE, *plan['repository_dependencies'], *checker.SUPPLEMENTAL_DEPENDENCIES]:
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checker.ROOT / name, target)
    result = checker.run_campaign(repo=repo, output_dir=tmp_path / 'portable-receipt')
    assert result['passed'], result
    assert [item['project'] for item in result['results']] == list(checker.PROJECTS)
    assert len(result['project_receipts']) == 12
    assert all(item['snapshot_normalization'] == [] for item in result['results'])


@pytest.mark.parametrize('raw', [b'{"a":1,"a":2}', b'{"x":1.5}', b'{"x":NaN}', b'{"x":Infinity}'])
def test_ambiguous_json_refused(raw):
    with pytest.raises(ValueError):
        checker.strict_json(raw)


@pytest.mark.parametrize('name', ['../x', '/x', 'a/../x', 'a//x', 'a/./x', 'a\\x',
                                  'quarantine/proof.py', 'x/99_DO_NOT_OPEN/proof',
                                  'x/verify_lambda_grid_v2.py'])
def test_unsafe_member_names_refused(name):
    with pytest.raises(ValueError):
        checker.safe_name(name)


def test_tampered_archive_fails_and_retains_failure_receipt(original, tmp_path):
    raw = original[0]
    damaged = tmp_path / 'damaged.zip'
    damaged.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    result = checker.run_project('h3_uniform', archive=damaged, output_dir=tmp_path / 'failure')
    assert not result['passed']
    assert 'canonical archive identity mismatch' in result['error']
    assert json.loads((tmp_path / 'failure/report.json').read_bytes())['passed'] is False


@pytest.mark.parametrize('mutation', ['traversal', 'symlink', 'duplicate', 'extra', 'oversize'])
def test_archive_structural_guards_before_execution(original, monkeypatch, mutation):
    # Rebind only the outer test pin to exercise the structural layer beneath it.
    # The production CLI exposes no override for any pin.
    source = zipfile.ZipFile(io.BytesIO(original[0]))
    out = io.BytesIO()
    first = True
    with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for item in source.infolist():
            data = source.read(item.filename)
            if first:
                if mutation == 'traversal':
                    item.filename = '../outside'
                elif mutation == 'symlink':
                    item.external_attr = (stat.S_IFLNK | 0o777) << 16
                elif mutation == 'oversize':
                    data = b'x' * (checker.MAX_MEMBER + 1)
                first = False
            archive.writestr(item, data)
        if mutation in ('extra', 'duplicate'):
            with pytest.warns(UserWarning) if mutation == 'duplicate' else nullcontext():
                archive.writestr(source.namelist()[0] if mutation == 'duplicate' else 'extra.txt', b'extra')
    raw = out.getvalue()
    monkeypatch.setattr(checker, 'ARCHIVE_SHA256', checker.identity(raw)['sha256'])
    monkeypatch.setattr(checker, 'ARCHIVE_BYTES', len(raw))
    with pytest.raises(ValueError):
        checker.inspect_archive(raw, original[1])


def test_new_exclusion_blocks_before_source_body_read(original, monkeypatch):
    raw, exclusions, members, _ = original
    name = next(name for name in members if name.endswith('/source.raw'))
    denied = {'kind': 'archive_member', 'member_path': 'unrelated-name',
              'payload_sha256': checker.identity(members[name])['sha256']}
    reads = []
    read = zipfile.ZipFile.read

    def spy(self, member, *args, **kwargs):
        reads.append(member)
        return read(self, member, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, 'read', spy)
    with pytest.raises(ValueError, match='excluded payload'):
        checker.inspect_archive(raw, exclusions + [denied])
    assert reads == ['MANIFEST.json']


@pytest.mark.parametrize('where', ['inventory', 'coverage', 'path', 'id'])
def test_source_reader_v2_metadata_guards(original, where):
    _, exclusions, members, _ = original
    record = next(json.loads(raw) for name, raw in members.items()
                  if name.endswith('/IDENTITY.json') and 'inventory' in json.loads(raw))
    record = deepcopy(record)
    if where in ('inventory', 'coverage'):
        record[where]['context'] = 'QUARANTINE_OR_HISTORY'
    elif where == 'path':
        record['inventory']['path'] = '01_ACTIVE_RESEARCH_PACKAGES/RECOVERED_OUTSIDE_K3_QUARANTINE/helper'
    else:
        record['drive_id'] = checker.INCIDENT_ID
    with pytest.raises(ValueError):
        checker.check_source_metadata(json.dumps(record).encode(), exclusions)


def test_existing_output_and_symlink_preserved(tmp_path):
    existing = tmp_path / 'keep'
    existing.mkdir()
    marker = existing / 'marker'
    marker.write_bytes(b'unchanged')
    link = tmp_path / 'link'
    link.symlink_to(existing, target_is_directory=True)
    for target in (existing, link):
        with pytest.raises((ValueError, FileExistsError)):
            checker.run_project('h3_uniform', output_dir=target)
    assert marker.read_bytes() == b'unchanged'


def test_repository_output_rejected(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    with pytest.raises(ValueError, match='outside repository'):
        checker.fresh_output(repo / 'output', repo)
    assert not (repo / 'output').exists()


def test_relocated_repo_does_not_authorize_wrapper_checkout_outputs(tmp_path):
    with pytest.raises(ValueError, match='outside repository'):
        checker.fresh_output(checker.ROOT / 'must-not-create-twelve-tests', tmp_path)
    assert not (checker.ROOT / 'must-not-create-twelve-tests').exists()


def test_symlink_dependency_refused(tmp_path):
    (tmp_path / 'real').write_bytes(b'source')
    (tmp_path / 'alias').symlink_to(tmp_path / 'real')
    with pytest.raises(ValueError, match='symlink'):
        checker.repository_inputs(tmp_path, {'alias': checker.identity(b'source')})


@pytest.mark.parametrize('name', tuple(checker.SUPPLEMENTAL_DEPENDENCIES))
def test_supplemental_runtime_source_is_required(name, tmp_path):
    target = tmp_path / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b'changed source')
    with pytest.raises(ValueError, match='repository dependency mismatch'):
        checker.repository_inputs(tmp_path, {name: checker.SUPPLEMENTAL_DEPENDENCIES[name]})


@pytest.mark.parametrize('wrong', [b'{"bound":"0"}\n', b'{"passed":1}\n', b'{"passed":true}\n'])
def test_mathematical_report_requires_exact_bytes(wrong):
    with pytest.raises(ValueError, match='mathematical report'):
        checker.compare_report(wrong, b'{"bound":"1","passed":false}\n')


def test_fake_success_cannot_replace_mathematical_report(monkeypatch, tmp_path):
    def fake(argv, cwd, output, stem, timeout_seconds):
        (output / 'result.json').write_bytes(b'{"fake":true}\n')
        logs = {}
        for kind in ('stdout', 'stderr'):
            raw = b'PASS\n' if kind == 'stdout' else b''
            (output / (stem + '.' + kind + '.txt')).write_bytes(raw)
            logs[kind] = checker.identity(raw)
        return {'returncode': 0, 'logs': logs}

    monkeypatch.setattr(checker, 'run_bounded', fake)
    result = checker.run_project('h3_uniform', output_dir=tmp_path / 'receipt')
    assert not result['passed']
    assert 'mathematical report' in result['error']


def test_subprocess_failure_retains_meaningful_log(tmp_path):
    with pytest.raises(ValueError, match='meaningful failure'):
        checker.run_bounded([sys.executable, '-c', 'import sys; print("meaningful failure"); sys.exit(4)'],
                            tmp_path, tmp_path, 'failed', 2)
    assert b'meaningful failure' in (tmp_path / 'failed.stdout.txt').read_bytes()


def test_subprocess_log_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(checker, 'MAX_LOG', 100)
    with pytest.raises(ValueError, match='log size limit'):
        checker.run_bounded([sys.executable, '-c', 'print("x"*10000)'], tmp_path, tmp_path, 'flood', 2)
    assert sum(path.stat().st_size for path in tmp_path.glob('flood.*.txt')) <= 100


def test_timeout_kills_descendant_after_leader_exits(tmp_path):
    marker = tmp_path / 'escaped'
    code = ('import os,time,pathlib; child=os.fork(); '
            'os._exit(0) if child else None; time.sleep(.5); '
            'pathlib.Path(' + repr(str(marker)) + ').write_text("escaped")')
    with pytest.raises(ValueError, match='timeout'):
        checker.run_bounded([sys.executable, '-c', code], tmp_path, tmp_path, 'timeout', .15)
    time.sleep(.5)
    assert not marker.exists()


def test_campaign_budget_cannot_return_partial_pass(tmp_path):
    result = checker.run_campaign(output_dir=tmp_path / 'campaign', budget_seconds=0.000001)
    assert not result['passed']
    assert len(result['results']) < 12
    assert 'budget' in result['error']


def test_one_failed_child_cannot_make_campaign_pass(tmp_path, monkeypatch):
    def fake(project, **kwargs):
        return {'project': project, 'passed': project != 'h3_uniform'}

    monkeypatch.setattr(checker, 'run_project', fake)
    result = checker.run_campaign(output_dir=tmp_path / 'campaign')
    assert len(result['results']) == 12
    assert not result['passed']
    assert 'one or more' in result['error']


def test_default_cli_dispatches_all_twelve(monkeypatch):
    calls = []
    monkeypatch.setattr(checker, 'run_campaign', lambda **kwargs: calls.append(kwargs) or {'passed': True})
    assert checker.main([]) == 0
    assert len(calls) == 1


def test_original_tests_fail_closed_on_relocated_root(original, tmp_path):
    _, _, _, plan = original
    repo = tmp_path / 'relocated'
    for name in [checker.ARCHIVE, *plan['repository_dependencies'], *checker.SUPPLEMENTAL_DEPENDENCIES]:
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checker.ROOT / name, target)
    result = checker.run_project('h3_uniform', repo=repo, author_tests=True, output_dir=tmp_path / 'refused')
    assert not result['passed']
    assert 'host-bound' in result['error']
    assert result['author_tests'] is None
