"""Held content never becomes corroborating quotation text, even after relocation."""
import builtins
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def mq():
    path = Path(__file__).resolve().parents[1]/'tools'/'mirror_quotes_check.py'
    spec = importlib.util.spec_from_file_location('scope_quote_controls', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def put(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def corpus(mq, root):
    return mq.build_corpus(str(root), mq.SCAN_RELS, mq.REGISTER_JSON_REL,
                           mq.INVENTORY_REL, mq.GOVERNANCE_RELS)[0]


def deny_body_reads(monkeypatch, paths):
    blocked = {str(path.absolute()) for path in paths}
    original = builtins.open

    def guarded(path, *args, **kwargs):
        if str(Path(path).absolute()) in blocked or str(Path(path).resolve()) in blocked:
            raise AssertionError('forbidden body was opened')
        return original(path, *args, **kwargs)

    monkeypatch.setattr(builtins, 'open', guarded)


@pytest.mark.parametrize('lane', [
    '99_DO_NOT_OPEN', '90_QUARANTINE_AND_TRIAGE', '02_LEGACY_Q0_ARCHIVE',
    '03_PERSONAL_AND_EARLIER_RESEARCH', 'legacy', 'quarantine', 'vault',
])
def test_held_lanes_pruned_before_body_or_readme_open(tmp_path, monkeypatch, mq, lane):
    root = tmp_path/'repo'
    body = root/'drive'/'mirrors'/lane/'source.md'
    readme = body.with_name('README.md')
    put(body, 'A forbidden source fragment of sufficient length.')
    put(readme, '"A forbidden source fragment of sufficient length."')
    good = root/'drive'/'mirrors'/'ACTIVE'/'source.md'
    put(good, 'An eligible retained source fragment.')
    deny_body_reads(monkeypatch, (body, readme))
    assert corpus(mq, root) == ['An eligible retained source fragment.']
    assert mq.readmes(str(root), mq.SCAN_RELS) == []


@pytest.mark.parametrize('identity', ['id', 'hash', 'declared_path'])
def test_manifest_exclusion_survives_innocent_filename(tmp_path, monkeypatch, mq, identity):
    root = tmp_path/'repo'
    base = root/'drive'/'mirrors'/'ACTIVE'
    body = base/'innocent.md'
    put(body, 'This hidden text cannot rescue the active README.')
    row = {'id': 'held-id', 'dest': 'innocent.md', 'stored': True, 'sha256': 'a'*64}
    exclusion = {'kind': 'drive_object', 'carrier_id': 'other-id'}
    if identity == 'id':
        exclusion['carrier_id'] = 'held-id'
    elif identity == 'hash':
        exclusion.update(kind='archive_member', payload_sha256='a'*64)
    else:
        row['drive_path'] = '01_ACTIVE_RESEARCH_PACKAGES/99_DO_NOT_OPEN/object'
    put(base/'_MANIFEST.jsonl', json.dumps(row)+'\n')
    put(root/'quarantine'/'EXCLUSIONS.json', json.dumps({'exclusions': [exclusion]}))
    put(base/'README.md', '"This hidden text cannot rescue the active README."')
    deny_body_reads(monkeypatch, (body,))
    result = mq.scan(str(root), mq.SCAN_RELS, mq.REGISTER_JSON_REL,
                     mq.INVENTORY_REL, mq.GOVERNANCE_RELS)
    assert len(result[3]) == 1
    assert result[3][0][2] == 'This hidden text cannot rescue the active README.'


def test_ancestor_manifest_excludes_nested_destination(tmp_path, monkeypatch, mq):
    root = tmp_path/'repo'
    base = root/'drive'/'mirrors'/'ACTIVE'
    body = base/'nested'/'innocent.md'
    put(body, 'A held nested body cannot be read.')
    put(base/'_MANIFEST.jsonl', json.dumps({'id': 'held', 'dest': 'nested/innocent.md'})+'\n')
    put(root/'quarantine'/'EXCLUSIONS.json', json.dumps({'exclusions': [
        {'kind': 'drive_object', 'carrier_id': 'held'}]}))
    deny_body_reads(monkeypatch, (body,))
    assert all('A held nested body' not in text for text in corpus(mq, root))


@pytest.mark.parametrize('kind', ['file', 'directory', 'inventory', 'register', 'governance'])
def test_symlink_targets_are_never_opened(tmp_path, monkeypatch, mq, kind):
    root = tmp_path/'repo'
    target = tmp_path/'outside'/'target.md'
    put(target, 'Outside text must never corroborate the quotation.')
    base = root/'drive'/'mirrors'/'ACTIVE'
    base.mkdir(parents=True)
    if kind == 'file':
        (base/'source.md').symlink_to(target)
    elif kind == 'directory':
        (base/'link').symlink_to(target.parent, target_is_directory=True)
    elif kind == 'inventory':
        (root/'drive'/'inventory.jsonl').symlink_to(target)
    elif kind == 'register':
        (root/'registers'/'json').mkdir(parents=True)
        (root/'registers'/'json'/'tab.json').symlink_to(target)
    else:
        (root/'CLAUDE.md').symlink_to(target)
    deny_body_reads(monkeypatch, (target,))
    assert corpus(mq, root) == []


def test_scan_cannot_escape_root(tmp_path, monkeypatch, mq):
    root = tmp_path/'repo'
    root.mkdir()
    body = tmp_path/'outside'/'source.md'
    put(body, 'No outside-root quotation evidence.')
    deny_body_reads(monkeypatch, (body,))
    assert mq.eligible_files(str(root), ('../outside',)) == []


def test_exclusion_metadata_symlink_fails_closed(tmp_path, mq):
    root = tmp_path/'repo'
    target = tmp_path/'outside.json'
    put(target, '{"exclusions": []}')
    (root/'quarantine').mkdir(parents=True)
    (root/'quarantine'/'EXCLUSIONS.json').symlink_to(target)
    with pytest.raises(ValueError, match='exclusion metadata'):
        corpus(mq, root)


def test_manifest_symlink_cannot_disable_exclusion_matching(tmp_path, mq):
    root = tmp_path/'repo'
    base = root/'drive'/'mirrors'/'ACTIVE'
    put(base/'source.md', 'An innocent filename is insufficient authority.')
    target = tmp_path/'outside.jsonl'
    put(target, '{}\n')
    (base/'_MANIFEST.jsonl').symlink_to(target)
    with pytest.raises(ValueError, match='manifest metadata'):
        corpus(mq, root)


def test_weakening_path_guard_activates_forbidden_open_control(tmp_path, monkeypatch, mq):
    root = tmp_path/'repo'
    body = root/'drive'/'mirrors'/'99_DO_NOT_OPEN'/'source.md'
    put(body, 'The weakened guard must fail this planted control.')
    deny_body_reads(monkeypatch, (body,))
    monkeypatch.setattr(mq, '_safe_path', lambda *args, **kwargs: True)
    with pytest.raises(AssertionError, match='forbidden body'):
        corpus(mq, root)
