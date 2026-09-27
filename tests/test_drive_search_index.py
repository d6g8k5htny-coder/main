"""Index compatibility controls; no source content or scientific status is read."""
import copy
import random
import string

import pytest

from tools import drive_index, drive_search
from tools import drive_search_index as index


def scan(rows, query):
    """Pre-index behavior, retained as the independent compatibility oracle."""
    q = query.casefold()
    result = [dict(row, match_kind='METADATA') for row in rows
              if any(q in str(row.get(field, '')).casefold()
                     for field in ('title', 'path', 'path_snapshot', 'id', 'context'))]
    return sorted(result, key=lambda row: (
        row['title'].casefold() != q,
        row['source_role'] in ('READING_COPY', 'METADATA_ONLY_EXCLUDED'), row['record_key']))


def indexed(rows, query):
    return drive_search.keyword_matches(rows, query, use_index=True)


def row(key, title, **extra):
    return {'record_key': key, 'id': key, 'title': title, 'path': title,
            'context': 'RESEARCH', 'source_role': 'SOURCE_CHECK_STATUS', **extra}


@pytest.fixture
def fresh(monkeypatch):
    instance = index.MetadataIndex()
    monkeypatch.setattr(index._LOCAL, 'index', instance, raising=False)
    yield instance
    instance.close()


def require_fts(instance, rows):
    instance.candidates(rows, 'first')
    instance.candidates(rows, 'second')
    if instance._connection is None:
        pytest.skip('runtime has no FTS5 trigram tokenizer; fallback tested separately')


@pytest.mark.parametrize('query', [
    'alpha', 'ALPHA', 'a"b', '" OR alpha', 'NEAR(alpha)', "');DROP TABLE metadata;--",
    '%a_', 'x\\y', '   alpha', 'alpha   ', 'absent',
])
def test_literal_query_escaping_and_exact_order(fresh, query):
    rows = [row('z', 'Alpha'), row('a', 'alpha', source_role='READING_COPY'),
            row('b', 'alphabet'), row('c', query + 'suffix'), row('d', 'αlpha'),
            row('tie', 'not exact', path=query, version=1),
            row('tie', 'not exact', path=query, version=2)]
    require_fts(fresh, rows)
    assert indexed(rows, query) == scan(rows, query)


def test_default_queries_scan_without_constructing_an_index(fresh):
    rows = [row('a', 'alpha')]
    for _ in range(3):
        assert drive_search.keyword_matches(rows, 'alpha') == scan(rows, 'alpha')
    assert fresh._connection is None and fresh.fingerprint is None
    indexed(rows, 'alpha')
    if index.sqlite3 is not None and fresh._connection is not None:
        assert fresh.fingerprint


@pytest.mark.parametrize('query', ['ab', 'Σίσυφος', 'Straße', '\x00alpha', 'alpha\n'])
def test_short_unicode_and_control_queries_never_use_tokenizer(fresh, monkeypatch, query):
    rows = [row('a', 'prefix ' + query + ' suffix')]
    monkeypatch.setattr(fresh, '_build', lambda *_: pytest.fail('unsupported query reached SQLite'))
    assert indexed(rows, query) == scan(rows, query)


def test_nul_surrogate_and_casefolded_source_fields_cannot_hide_hits(fresh):
    rows = [row('a', 'prefix\0alpha'), row('b', 'prefix\ud800alpha'),
            row('c', 'Straße'), row('d', 'KELVIN'), row('e', 'İSTANBUL')]
    require_fts(fresh, rows)
    for query in ('alpha', 'strasse', 'kelvin', 'stanbul'):
        assert indexed(rows, query) == scan(rows, query)


@pytest.mark.parametrize('field', index.FIELDS)
def test_changed_search_fields_rebuild_and_old_hits_disappear(fresh, field):
    rows = [row('a', 'base')]
    rows[0][field] = 'old-marker'
    require_fts(fresh, rows)
    old_fingerprint = fresh.fingerprint
    rows[0][field] = 'new-marker'
    assert indexed(rows, 'new-marker') == scan(rows, 'new-marker')
    assert indexed(rows, 'old-marker') == []
    assert fresh.fingerprint != old_fingerprint


def test_insert_delete_reorder_and_duplicate_occurrences(fresh):
    rows = [row('same', 'match', version=1), row('same', 'match', version=2)]
    require_fts(fresh, rows)
    for mutate in (lambda: rows.reverse(), lambda: rows.append(row('new', 'match')),
                   lambda: rows.pop(0)):
        mutate()
        assert indexed(rows, 'match') == scan(rows, 'match')


def test_nonsearch_identity_and_scope_updates_use_current_rows(fresh):
    rows = [row('a', 'alpha', sha256='a' * 64, content_eligible=True, custody={'revision': 1}),
            row('b', 'alpha')]
    require_fts(fresh, rows)
    connection = fresh._connection
    rows[0].update(sha256='b' * 64, source_role='METADATA_ONLY_EXCLUDED', content_eligible=False)
    rows[0]['custody']['revision'] = 2
    assert indexed(rows, 'alpha') == scan(rows, 'alpha')
    assert fresh._connection is connection  # no stale cached result rows exist


def test_held_metadata_navigation_never_hydrates_content(fresh, monkeypatch):
    rows = [row('hold', 'alpha 99_DO_NOT_OPEN', source_role='METADATA_ONLY_EXCLUDED',
                content_eligible=False, custody={'path': '/must/not/read'}),
            row('archive', 'alpha', record_type='ARCHIVE_MEMBER', content_eligible=False)]
    monkeypatch.setattr(drive_search, 'Reader', lambda *_: pytest.fail('metadata query hydrated content'))
    require_fts(fresh, rows)
    assert indexed(rows, 'alpha') == scan(rows, 'alpha')
    assert not any(hit['content_eligible'] for hit in indexed(rows, 'alpha'))
    assert not drive_search.content_allowed(rows[0])


@pytest.mark.parametrize('corruption', ['delete_postings', 'drop_schema', 'closed_connection'])
def test_corrupt_or_unusable_private_cache_rebuilds_or_scans(fresh, corruption):
    rows = [row('a', 'alpha')]
    require_fts(fresh, rows)
    old = fresh._connection
    if corruption == 'closed_connection':
        old.close()
    else:
        old.execute('PRAGMA query_only = OFF')
        old.execute('DELETE FROM metadata' if corruption == 'delete_postings' else 'DROP TABLE metadata')
        old.commit()
    assert indexed(rows, 'alpha') == scan(rows, 'alpha')
    # The subsequent lookup also succeeds, and never reuses the damaged object.
    assert indexed(rows, 'alpha') == scan(rows, 'alpha')
    assert fresh._connection is not old


def test_cache_rejects_unexpected_writes_by_default(fresh):
    rows = [row('a', 'alpha')]
    require_fts(fresh, rows)
    with pytest.raises(index.sqlite3.OperationalError):
        fresh._connection.execute('DELETE FROM metadata')


def test_optional_sqlite_extension_missing_uses_original_scan(fresh, monkeypatch):
    monkeypatch.setattr(index, 'sqlite3', None)
    rows = [row('a', 'alpha')]
    assert indexed(rows, 'alpha') == scan(rows, 'alpha')
    assert indexed(rows, 'alpha') == scan(rows, 'alpha')


def test_optional_fts5_unavailable_uses_original_scan(fresh, monkeypatch):
    if index.sqlite3 is None:
        pytest.skip('missing extension covered by separate test')
    calls = []

    class Unavailable:
        def execute(self, _):
            calls.append('create')
            raise index.sqlite3.OperationalError('no such tokenizer: trigram')

        def close(self):
            pass

    monkeypatch.setattr(index.sqlite3, 'connect', lambda *_: Unavailable())
    rows = [row('a', 'alpha')]
    for _ in range(4):
        assert indexed(rows, 'alpha') == scan(rows, 'alpha')
    assert calls == ['create']


def test_iterables_are_not_consumed_twice(fresh):
    rows = [row('a', 'alpha'), row('b', 'beta')]
    require_fts(fresh, rows)
    assert indexed((value for value in rows), 'alpha') == scan(rows, 'alpha')


@pytest.mark.parametrize('query', ['', ' ', None, 3])
def test_invalid_query_behavior_preserved(fresh, query):
    with pytest.raises(ValueError, match='nonempty keyword'):
        indexed([], query)


def test_generated_ascii_substrings_preserve_parity(fresh):
    rng = random.Random(20260922)
    alphabet = string.ascii_letters + string.digits + ' _%"\\:;()-'
    rows = [row(str(n), ''.join(rng.choices(alphabet, k=45))) for n in range(200)]
    require_fts(fresh, rows)
    for value in rng.sample(rows, 25):
        query = value['title'][5:10]
        assert indexed(rows, query) == scan(rows, query)


def test_actual_reconciled_file_and_archive_metadata_parity(fresh):
    rows = drive_search.load_current(drive_index.load()) + drive_search.archive_records()
    assert len(rows) == 4934 + 11649
    before = copy.deepcopy(rows)
    require_fts(fresh, rows)
    for query in ('rnu_ds3.py', 'RN', 'Cholesky', 'q0', 'THEOREM B', 'D3-LEMMA-RN-UNIF',
                  'P15', 'reusable operations', 'H5', 'review', 'GP-DATA-214',
                  'matrix', 'Gaussian', '99_DO_NOT_OPEN', 'QUARANTINE', 'LEGACY',
                  'no-such-marker-510fea', '—', 'µ', 'ϑ'):
        assert indexed(rows, query) == scan(rows, query)
    assert rows == before
