#!/usr/bin/env python3
"""Pinned source acquisition and offline byte custody, never theorem validation.

The canonical v16 data remains immutable.  This separate ledger records the
bytes actually obtained over HTTPS, independently hashes them, and leaves failed
retrievals and theorem applicability explicit.  Verification uses no network.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent
LEDGER_NAME = 'source_provenance.json'
SCHEMA = 'UL-source-custody-v17.1'
HEX40 = re.compile(r'^[0-9a-f]{40}$')
HEX64 = re.compile(r'^[0-9a-f]{64}$')
MAX_SOURCE_BYTES = 4 * 1024 * 1024


class SourceCustodyError(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':')).encode('utf-8')


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def blob_sha1(value):
    return hashlib.sha1(b'blob ' + str(len(value)).encode('ascii') + b'\0' + value).hexdigest()


def source_id(url):
    return 'SRC17-' + sha256(url.encode('utf-8'))[:20]


def safe_local_path(root, name):
    if not isinstance(name, str):
        raise SourceCustodyError('source path must be text')
    rel = PurePosixPath(name)
    if rel.is_absolute() or '..' in rel.parts or not rel.parts or rel.parts[0] != 'upstream_sources':
        raise SourceCustodyError('unsafe source path: ' + name)
    root = Path(root).resolve()
    candidate = root.joinpath(*rel.parts)
    if candidate.is_symlink() or root not in candidate.resolve().parents:
        raise SourceCustodyError('source path escapes package: ' + name)
    return candidate


def _reference_entry(reference):
    url = reference['url']
    entry = {key: reference.get(key) for key in ('repo', 'commit', 'path', 'pin_kind')}
    entry.update({'source_id': source_id(url), 'url': url,
                  'roles': reference.get('roles', []),
                  'dataset_locations': reference.get('dataset_locations', [])})
    return entry


def _pinned_raw_url(entry):
    if (entry.get('pin_kind') != 'git_commit_sha40' or
            not isinstance(entry.get('commit'), str) or not HEX40.fullmatch(entry['commit'])):
        raise SourceCustodyError('source is not pinned to a full Git commit')
    repo = entry.get('repo', '')
    path = entry.get('path', '')
    if (not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repo) or
            not path or PurePosixPath(path).is_absolute() or '..' in PurePosixPath(path).parts):
        raise SourceCustodyError('invalid repository/path identity')
    quoted = urllib.parse.quote(path, safe='/')
    expected = f"https://github.com/{repo}/blob/{entry['commit']}/{quoted}"
    # The handoff's paths contain no encoded reserved characters; accept the
    # decoded path representation while requiring an exact repository and pin.
    if urllib.parse.unquote(entry['url']) != urllib.parse.unquote(expected):
        raise SourceCustodyError('reference URL disagrees with repository/commit/path')
    return f"https://raw.githubusercontent.com/{repo}/{entry['commit']}/{quoted}"


def acquire_sources(root=ROOT, timeout=15, concurrency=4):
    """Fetch indexed full-commit sources; unresolved references stay in ledger.

    Uses Python's inherited proxy and default TLS verification.  Files have
    content-addressed names and are never overwritten with different bytes.
    Acquisitions are bounded to two attempts, a timeout and four MiB per file.
    """
    root = Path(root)
    references = json.loads((root / 'SOURCE_REFERENCE_INDEX.json').read_text(encoding='utf-8'))['references']
    (root / 'upstream_sources').mkdir(exist_ok=True)

    def fetch(reference):
        entry = _reference_entry(reference)
        if entry['pin_kind'] != 'git_commit_sha40':
            entry.update(status='unresolved_unversioned_reference', local_path=None,
                         sha256=None, byte_count=None, git_blob_sha1=None,
                         unresolved_obligation='Supply an immutable source revision before source custody can be established.')
            return entry
        raw_url = _pinned_raw_url(entry)
        errors = []
        for attempt in range(1, 3):
            try:
                request = urllib.request.Request(raw_url, headers={'User-Agent': 'Universal-Law-v17-source-custody/1'})
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    if response.geturl() != raw_url:
                        raise SourceCustodyError('unexpected redirect away from exact pinned raw URL')
                    data = response.read(MAX_SOURCE_BYTES + 1)
                    if len(data) > MAX_SOURCE_BYTES:
                        raise SourceCustodyError('source exceeds acquisition byte limit')
                    if response.status != 200:
                        raise SourceCustodyError('non-success HTTP status')
                    response_metadata = {'http_status': response.status,
                                         'content_type': response.headers.get('Content-Type'),
                                         'etag': response.headers.get('ETag')}
                digest = sha256(data)
                local_path = 'upstream_sources/' + digest + '.raw'
                destination = safe_local_path(root, local_path)
                if destination.exists() and destination.read_bytes() != data:
                    raise SourceCustodyError('content-addressed source collision')
                if not destination.exists():
                    destination.write_bytes(data)
                # Re-read the saved original bytes, independently of the
                # response-buffer hash and of any source-published digest.
                with destination.open('rb') as saved:
                    saved_digest = hashlib.file_digest(saved, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else sha256(saved.read())
                if saved_digest != digest:
                    raise SourceCustodyError('saved original bytes do not match acquired bytes')
                entry.update(status='acquired_pinned_https', local_path=local_path,
                             sha256=digest, saved_bytes_sha256=saved_digest,
                             byte_count=len(data), git_blob_sha1=blob_sha1(data),
                             acquisition={'method': 'HTTPS_GET_PINNED_RAW_GITHUB_DEFAULT_TLS_INHERITED_PROXY',
                                          'requested_url': raw_url, 'resolved_url': raw_url,
                                          'acquired_utc': datetime.now(timezone.utc).isoformat(),
                                          'attempt': attempt, **response_metadata},
                             publisher_signature_verified=False,
                             theorem_statement_alignment='requires_separate_mathematical_review',
                             mathematical_certification='not_certified')
                return entry
            except Exception as exc:
                errors.append(type(exc).__name__ + ': ' + str(exc))
                if attempt == 1:
                    time.sleep(.1)
        entry.update(status='acquisition_failed', local_path=None, sha256=None,
                     byte_count=None, git_blob_sha1=None,
                     acquisition={'method': 'HTTPS_GET_PINNED_RAW_GITHUB_DEFAULT_TLS_INHERITED_PROXY',
                                  'requested_url': raw_url, 'attempts': 2, 'errors': errors},
                     unresolved_obligation='Retrieve the exact pinned original bytes; no source authenticity or applicability inferred.')
        return entry

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(concurrency, 8))) as pool:
        entries = list(pool.map(fetch, references))
    ledger = {'schema_version': SCHEMA,
              'claim': 'Pinned HTTPS acquisition and preserved-byte integrity; no publisher signature or mathematical certification.',
              'source_reference_index_sha256': sha256((root / 'SOURCE_REFERENCE_INDEX.json').read_bytes()),
              'entries': entries,
              'summary': {'indexed_references': len(entries),
                          'pinned_references': sum(e['pin_kind'] == 'git_commit_sha40' for e in entries),
                          'acquired': sum(e['status'] == 'acquired_pinned_https' for e in entries),
                          'unresolved': sum(e['status'] != 'acquired_pinned_https' for e in entries)}}
    temporary = root / (LEDGER_NAME + '.tmp')
    temporary.write_text(json.dumps(ledger, ensure_ascii=False, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, root / LEDGER_NAME)
    return verify_sources(root)


def _load_ledger(root, ledger=None):
    if ledger is None:
        return json.loads((Path(root) / LEDGER_NAME).read_text(encoding='utf-8'))
    return ledger


def _descriptor(entry):
    return {key: entry.get(key) for key in
            ('source_id', 'url', 'repo', 'commit', 'path', 'status',
             'version', 'local_path', 'sha256', 'byte_count', 'git_blob_sha1')}


def _external_entries(root):
    path = Path(root) / 'external_theorem_sources.json'
    if not path.exists():
        return []
    ledger = json.loads(path.read_text(encoding='utf-8'))
    if ledger.get('schema_version') != 'UL-external-theorem-source-v17.1':
        raise SourceCustodyError('unsupported external theorem source schema')
    entries = ledger.get('entries', [])
    if len(entries) != len({e.get('url') for e in entries}):
        raise SourceCustodyError('duplicate external theorem source')
    return entries


def verify_external_sources(root=ROOT, only_urls=None):
    """Check original versioned external PDFs and recorded text extracts offline."""
    checked = 0
    for entry in _external_entries(root):
        if only_urls is not None and entry['url'] not in only_urls:
            continue
        version = entry.get('version', '')
        if (not re.fullmatch(r'[0-9]{4}\.[0-9]{4,5}v[1-9][0-9]*', version) or
                entry.get('url') != 'https://arxiv.org/pdf/' + version):
            raise SourceCustodyError('external theorem source is not an exact arXiv version')
        if entry.get('status') != 'acquired_versioned_https':
            raise SourceCustodyError('unsupported external theorem source acquisition state')
        value = safe_local_path(root, entry.get('local_path')).read_bytes()
        if (not value.startswith(b'%PDF') or sha256(value) != entry.get('sha256') or
                len(value) != entry.get('byte_count')):
            raise SourceCustodyError('tampered original external theorem source')
        acquisition = entry.get('acquisition', {})
        if (acquisition.get('requested_url') != entry['url'] or
                acquisition.get('resolved_url') != entry['url']):
            raise SourceCustodyError('external theorem acquisition version mismatch')
        if entry.get('formal_certification') != 'not_certified':
            raise SourceCustodyError('external source acquisition falsely promoted to proof certification')
        extract = entry.get('text_extraction')
        if extract and extract.get('exit_code') == 0:
            text = safe_local_path(root, extract.get('path')).read_bytes()
            if sha256(text) != extract.get('sha256'):
                raise SourceCustodyError('external theorem text extraction hash mismatch')
        checked += 1
    return checked


def verify_sources(root=ROOT, ledger=None, require_all=False, only_urls=None):
    """Verify indexed identities and preserved bytes offline; never promote math.

    ``only_urls`` selects byte checks for a worker's dependency branch.  Ledger
    identity/coverage remains validated globally, while unrelated original-file
    reads are omitted.  An unresolved source is reported, not silently passed.
    """
    root = Path(root)
    ledger = _load_ledger(root, ledger)
    if ledger.get('schema_version') != SCHEMA:
        raise SourceCustodyError('unsupported source custody schema')
    index_path = root / 'SOURCE_REFERENCE_INDEX.json'
    if ledger.get('source_reference_index_sha256') != sha256(index_path.read_bytes()):
        raise SourceCustodyError('source reference index hash mismatch')
    refs = json.loads(index_path.read_text(encoding='utf-8'))['references']
    expected = {r['url']: _reference_entry(r) for r in refs}
    entries = ledger.get('entries', [])
    if len(entries) != len(expected) or {e.get('url') for e in entries} != set(expected):
        raise SourceCustodyError('source ledger coverage differs from preserved reference index')
    external_urls = {entry['url'] for entry in _external_entries(root)}
    selected = set(expected) | external_urls if only_urls is None else set(only_urls)
    if not selected <= set(expected) | external_urls:
        raise SourceCustodyError('selected source URL is not in the preserved source index')
    acquired = unresolved = checked_bytes = 0
    for entry in entries:
        exp = expected[entry['url']]
        for field in ('source_id', 'repo', 'commit', 'path', 'pin_kind'):
            if entry.get(field) != exp.get(field):
                raise SourceCustodyError('source identity mismatch: ' + str(field))
        if entry['url'] not in selected:
            continue
        if entry.get('status') == 'acquired_pinned_https':
            raw_url = _pinned_raw_url(entry)
            if (entry.get('acquisition', {}).get('requested_url') != raw_url or
                    entry.get('acquisition', {}).get('resolved_url') != raw_url):
                raise SourceCustodyError('acquisition URL does not match immutable source')
            expected_hash = entry.get('sha256')
            if not isinstance(expected_hash, str) or not HEX64.fullmatch(expected_hash):
                raise SourceCustodyError('invalid original source SHA-256')
            f = safe_local_path(root, entry.get('local_path'))
            if not f.is_file():
                raise SourceCustodyError('missing acquired original source: ' + str(entry.get('local_path')))
            value = f.read_bytes()
            if (sha256(value) != expected_hash or entry.get('saved_bytes_sha256') != expected_hash or
                    len(value) != entry.get('byte_count') or blob_sha1(value) != entry.get('git_blob_sha1')):
                raise SourceCustodyError('tampered original source bytes: ' + entry['url'])
            if entry.get('mathematical_certification') != 'not_certified':
                raise SourceCustodyError('source custody falsely promoted to mathematical certification')
            acquired += 1
            checked_bytes += len(value)
        elif entry.get('status') in {'acquisition_failed', 'unresolved_unversioned_reference'}:
            if entry.get('sha256') is not None or entry.get('local_path') is not None:
                raise SourceCustodyError('unresolved source claims acquired bytes')
            unresolved += 1
        else:
            raise SourceCustodyError('unknown source custody state')
    if require_all and unresolved:
        raise SourceCustodyError(str(unresolved) + ' selected sources remain unresolved')
    external_checked = verify_external_sources(root, selected)
    result = {'result': 'SOURCE_BYTE_INTEGRITY_CHECKED_NOT_THEOREM_VALIDATION',
            'indexed_references': len(expected), 'selected_references': len(selected),
            'acquired_sources_checked': acquired, 'unresolved_sources': unresolved,
            'external_versioned_sources_checked': external_checked,
            'original_bytes_checked': checked_bytes,
            'publisher_signature_verified': False, 'theorem_certificates_awarded': 0}
    if only_urls is None and (root / 'import_applicability.json').exists():
        result['applicability_ledger'] = verify_applicability(root)
    return result


def verify_applicability(root=ROOT):
    """Check source-text bindings and explicit open assumptions, not their truth."""
    root = Path(root)
    data_bytes = (root / 'dataset.json').read_bytes()
    data = json.loads(data_bytes)
    source_entries = [*_load_ledger(root)['entries'], *_external_entries(root)]
    by_url = {e['url']: e for e in source_entries}

    def check_anchor(anchor):
        if anchor.get('url') not in by_url:
            raise SourceCustodyError('comparison cites unknown source')
        entry = by_url[anchor['url']]
        if anchor.get('source_sha256') != entry.get('sha256'):
            raise SourceCustodyError('comparison source hash mismatch')
        path = anchor.get('text_path', entry['local_path'])
        value = safe_local_path(root, path).read_bytes()
        if anchor.get('text_sha256') and sha256(value) != anchor['text_sha256']:
            raise SourceCustodyError('comparison text hash mismatch')
        lines = value.decode('utf-8').splitlines()
        lo, hi = anchor.get('line_start'), anchor.get('line_end')
        if (not isinstance(lo, int) or not isinstance(hi, int) or
                lo < 1 or hi < lo or hi > len(lines)):
            raise SourceCustodyError('invalid comparison line anchor')
        excerpt = '\n'.join(lines[lo - 1:hi])
        if excerpt != anchor.get('excerpt') or sha256(excerpt.encode('utf-8')) != anchor.get('excerpt_sha256'):
            raise SourceCustodyError('comparison excerpt does not match original source')

    ledger = json.loads((root / 'import_applicability.json').read_text(encoding='utf-8'))
    if (ledger.get('schema_version') != 'UL-import-applicability-v17.1' or
            ledger.get('canonical_dataset_sha256') != sha256(data_bytes)):
        raise SourceCustodyError('applicability ledger is not bound to canonical dataset')
    imports = {i['id']: i for i in data['imports']}
    entries = ledger.get('entries', [])
    if len(entries) != len(imports) or {e.get('id') for e in entries} != set(imports):
        raise SourceCustodyError('applicability import inventory changed')
    requirements = 0
    for entry in entries:
        imp = imports[entry['id']]
        if (entry.get('import_status') != 'unvalidated' or
                entry.get('all_conditions_satisfied') is not False or
                entry.get('formal_certification') != 'not_certified'):
            raise SourceCustodyError('unearned imported theorem promotion')
        if (entry.get('canonical_required_evidence') != imp['statement_or_required_evidence'] or
                entry.get('canonical_scope_description') != imp['scope_description'] or
                entry.get('applies_to') != imp['applies_to']):
            raise SourceCustodyError('import applicability statement/scope mismatch')
        if not entry.get('requirements'):
            raise SourceCustodyError('missing explicit applicability assumptions')
        for requirement in entry['requirements']:
            if (not requirement.get('requirement') or
                    requirement.get('validation_state') != 'unresolved_independent_applicability_obligation'):
                raise SourceCustodyError('unbacked applicability requirement discharge')
            requirements += 1
        for ref in entry.get('source_references', []):
            if ref.get('url') not in by_url or ref.get('sha256') != by_url[ref['url']].get('sha256'):
                raise SourceCustodyError('applicability source reference mismatch')
        for anchor in entry.get('primary_theorem_anchors', []):
            check_anchor(anchor)

    comparisons = json.loads((root / 'source_statement_comparisons.json').read_text(encoding='utf-8'))
    if (comparisons.get('schema_version') != 'UL-source-statement-comparison-v17.1' or
            comparisons.get('canonical_dataset_sha256') != sha256(data_bytes)):
        raise SourceCustodyError('source comparison ledger is not bound to canonical dataset')
    records = {r['id']: r for r in data['records']}
    rows = comparisons.get('records', [])
    if len(rows) != len(records) or {r.get('record_id') for r in rows} != set(records):
        raise SourceCustodyError('source comparison record inventory mismatch')
    compared = 0
    for row in rows:
        rec = records[row['record_id']]
        if (row.get('canonical_statement_sha256') != sha256(rec['statement'].encode('utf-8')) or
                row.get('canonical_scope_sha256') != sha256(rec['scope'].encode('utf-8')) or
                row.get('canonical_source_url') != rec['source'] or
                row.get('mathematical_certification') != 'not_certified'):
            raise SourceCustodyError('source comparison statement mismatch or unearned promotion')
        if 'anchor' in row:
            check_anchor(row['anchor']); compared += 1
        elif row.get('comparison_status') != 'source_bytes_acquired_semantic_comparison_unresolved':
            raise SourceCustodyError('unreviewed source comparison lacks unresolved state')
    expected_corrections = {c['id']: c for c in data['corrections']}
    corrections = comparisons.get('mandatory_d1_corrections', [])
    if (len(corrections) != len(expected_corrections) or
            {c.get('id') for c in corrections} != set(expected_corrections)):
        raise SourceCustodyError('mandatory source correction comparison missing')
    for correction in corrections:
        original = expected_corrections[correction['id']]
        entry = by_url[original['source_url']]
        if (correction.get('canonical_role') != original['role'] or
                correction.get('url') != original['source_url'] or
                correction.get('sha256') != entry.get('sha256') or
                correction.get('mathematical_validation_state') != 'not_independently_certified'):
            raise SourceCustodyError('mandatory correction identity/role mismatch')
        check_anchor(correction['anchor'])
        published = correction.get('comparison_to_reconciliation_quoted_identity')
        if published and (published.get('matches') is not True or
                          published.get('expected_sha256') != entry['sha256'] or
                          published.get('expected_git_blob_sha1') != entry['git_blob_sha1'] or
                          published.get('expected_byte_count') != entry['byte_count']):
            raise SourceCustodyError('mandatory correction disagrees with reconciliation identity')
    return {'import_classes_unvalidated': len(entries),
            'explicit_unresolved_applicability_requirements': requirements,
            'mandatory_d1_source_comparisons': len(corrections),
            'targeted_statement_scope_comparisons': compared,
            'remaining_record_semantic_comparisons': len(rows) - compared,
            'formal_theorem_certificates': 0}


def source_fingerprints(root=ROOT):
    return {e['url']: sha256(canonical(_descriptor(e)))
            for e in [*_load_ledger(root)['entries'], *_external_entries(root)]}


def source_inputs(data, record_id, root=ROOT):
    """Source descriptors for one record, transitive premises and named imports.

    The descriptors exclude acquisition timestamps, so a repeat acquisition of
    identical original bytes does not invalidate an unrelated mathematical job.
    """
    records = {r['id']: r for r in data['records']}
    obs = {o['conclusion']: o for o in data['obligations']}
    corrections = {c['id']: c for c in data['corrections']}
    imports = {i['id']: i for i in data['imports']}
    if record_id not in records:
        raise SourceCustodyError('unknown record for source selection: ' + record_id)
    visited, strings, import_ids = set(), [], set()

    def texts(value):
        if isinstance(value, str):
            strings.append(value)
        elif isinstance(value, dict):
            for v in value.values():
                texts(v)
        elif isinstance(value, list):
            for v in value:
                texts(v)

    def visit(rid):
        if rid in visited:
            return
        visited.add(rid)
        if rid not in records or rid not in obs:
            raise SourceCustodyError('missing source prerequisite: ' + rid)
        texts(records[rid]); texts(obs[rid])
        for cid in obs[rid]['source_corrections']:
            texts(corrections[cid])
        for iid in obs[rid]['external_theorem_imports']:
            texts(imports[iid])
            import_ids.add(iid)
        for pid in obs[rid]['premises']:
            visit(pid)

    visit(record_id)
    corpus = '\n'.join(strings)
    selected = [_descriptor(e) for e in _load_ledger(root)['entries'] if e['url'] in corpus]
    if 'EXT-KR-AAL' in import_ids:
        selected.extend(_descriptor(e) for e in _external_entries(root)
                        if e['source_id'] == 'EXTSRC-AAL-2304.07424v3')
    return sorted(selected, key=lambda e: e['url'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['verify', 'acquire'], nargs='?', default='verify')
    parser.add_argument('--require-all', action='store_true')
    args = parser.parse_args()
    result = acquire_sources() if args.command == 'acquire' else verify_sources(require_all=args.require_all)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
