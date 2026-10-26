"""Inspectable need-based literature ranking and bounded authenticated byte custody.

Ranking is a documented heuristic, not a learned mathematical relevance model.
Acquisition never establishes theorem applicability.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'engine'))
from conductor import Journal

ALLOWED_HOSTS = frozenset({'arxiv.org', 'export.arxiv.org', 'raw.githubusercontent.com',
    'www.numdam.org', 'numdam.org', 'projecteuclid.org', 'www.math.kyoto-u.ac.jp',
    'www.maths.ed.ac.uk', 'perso.ens-lyon.fr', 'people.maths.ox.ac.uk',
    'www.jstage.jst.go.jp', 'jstage.jst.go.jp', 'drops.dagstuhl.de'})
MAX_FILE = 16 * 1024 * 1024
SHA = re.compile(r'^[0-9a-f]{64}$')


def validate_source_url(url):
    parsed = urlsplit(url)
    if (parsed.scheme != 'https' or parsed.hostname not in ALLOWED_HOSTS
            or parsed.username or parsed.password or parsed.port not in (None, 443)
            or parsed.fragment):
        raise ValueError('source URL must use HTTPS on an explicitly configured primary-source host')
    return url


class _UnreadRedirectBody:
    """Let urllib close a redirect without its default unbounded body drain."""
    def __init__(self, response):
        self.response = response

    def read(self, *args, **kwargs):
        return b''

    def __getattr__(self, name):
        return getattr(self.response, name)


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_source_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)

    def http_error_302(self, req, fp, code, msg, headers):
        # Default urllib calls fp.read() with no size on each redirect. Preserve
        # its destination/loop logic but never consume an unbudgeted body.
        try:
            return super().http_error_302(req, _UnreadRedirectBody(fp), code, msg, headers)
        finally:
            fp.close()

    http_error_301 = http_error_303 = http_error_307 = http_error_308 = http_error_302


class ResearchLibrary:
    def __init__(self, catalog, cache):
        self.catalog_path = Path(catalog)
        self.cache = Path(cache)
        self._verified_objects = {}
        self.journal = Journal(self.cache)
        data = json.loads(self.catalog_path.read_text(encoding='utf8'))
        rows = data.get('sources', data.get('items', []))
        if not isinstance(rows, list) or len(rows) > 2000:
            raise ValueError('literature catalog requires at most 2000 sources')
        self.items = {}
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get('id'), str):
                raise ValueError('literature source needs a stable ID')
            if row['id'] in self.items:
                raise ValueError('duplicate literature ID')
            self.items[row['id']] = row
        omissions = data.get('distribution_omissions', [])
        if not isinstance(omissions, list) or len(omissions) > len(self.items):
            raise ValueError('distribution omissions must be a bounded list of explicit source references')
        self.distribution_omissions = {}
        for omission in omissions:
            if not isinstance(omission, dict):
                raise ValueError('distribution omission must be an explicit source reference')
            source_id = omission.get('source_id')
            item = self.items.get(source_id) if isinstance(source_id, str) else None
            checksum, size = omission.get('sha256'), omission.get('bytes')
            reason, url = omission.get('reason'), omission.get('url')
            if (item is None or source_id in self.distribution_omissions
                    or not isinstance(checksum, str) or not SHA.fullmatch(checksum)
                    or checksum != item.get('source_sha256')
                    or type(size) is not int or not 0 < size <= MAX_FILE
                    or not isinstance(reason, str) or not reason.strip()
                    or not isinstance(url, str)
                    or url not in (item.get('url'), item.get('acquisition_url'))):
                raise ValueError('distribution omission must match a unique catalog source and exact SHA-256 pin')
            parsed = urlsplit(url)
            if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
                raise ValueError('distribution omission requires a valid HTTPS external reference')
            self.distribution_omissions[source_id] = dict(omission)
        self.coverage = data.get('coverage', {})
        self.limitations = data.get('limitations', [
            'Targeted multilingual search; coverage is not exhaustive across all languages.',
            'Ranking estimates reuse value; source acquisition does not validate applicability.'])

    def _captures(self):
        return [e['payload'] for e in self.journal.since(0) if e['type'] == 'LITERATURE_ACQUIRED']

    @staticmethod
    def _source_identity(item):
        fields = {key: item.get(key) for key in ('id', 'url', 'acquisition_url',
                  'source_sha256', 'version', 'acquired_version')}
        return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()

    def _current_capture(self, capture):
        item = self.items.get(capture['source_id'])
        if item is None:
            return False
        expected = item.get('source_sha256')
        if expected:
            # Identical pinned primary bytes can be reused across URL aliases.
            return expected == capture['sha256']
        return capture.get('source_identity_sha256') == self._source_identity(item)

    def _source_available(self, capture):
        checksum = capture.get('sha256', '')
        if not isinstance(checksum, str) or not SHA.fullmatch(checksum):
            return False
        path = self.cache / 'objects' / checksum
        try:
            if path.is_symlink() or not path.resolve().is_relative_to(self.cache.resolve()):
                return False
            stat = path.stat()
            signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            if not path.is_file() or stat.st_size != capture['bytes']:
                return False
            if self._verified_objects.get(checksum) != signature:
                if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
                    return False
                self._verified_objects[checksum] = signature
            return True
        except (OSError, ValueError):
            return False

    def rank(self, targets=None):
        targets = set(targets or [])
        retained = {x['source_id'] for x in self._captures()
                    if self._current_capture(x) and self._source_available(x)}
        result = []
        for item in self.items.values():
            score, reasons = 0, []
            overlap = targets.intersection(item.get('project_targets', []))
            if overlap:
                score += 100 * len(overlap)
                reasons.append('Addresses current project need: ' + ', '.join(sorted(overlap)))
            inspected = item.get('inspection_status', '')
            if inspected in {'primary_full_text_inspected', 'primary_text_inspected',
                             'pinned_formal_source_inspected', 'formal_source_inspected',
                             'theorem_statement_inspected'}:
                score += 30
                reasons.append('Primary statement or formal source inspected')
            else:
                reasons.append('Reading needed before exact theorem applicability can be assessed')
            if item['id'] in retained:
                score += 15
                reasons.append('Preserved source bytes available offline')
            if item.get('exact_location'):
                score += 8
                reasons.append('Exact theorem location recorded')
            if item.get('remaining_project_work'):
                score += 5
                reasons.append('Missing project alignment is explicit')
            result.append({**item, 'rank_score': score, 'ranking_reasons': reasons,
                'available_offline': item['id'] in retained,
                'applicability': 'unvalidated', 'formal_status': 'not_certified'})
        return sorted(result, key=lambda x: (-x['rank_score'], x['id']))

    def snapshot(self, targets=None):
        return {'items': self.rank(targets), 'coverage': self.coverage,
                'limitations': self.limitations, 'acquired_sources': len(self._captures()),
                'ranking_method': 'explicit need overlap + inspected statement + offline custody; v1'}

    def plan_prefetch(self, targets=None, *, max_items=3):
        if type(max_items) is not int or not 1 <= max_items <= 4:
            raise ValueError('prefetch plan count must be between 1 and 4')
        selected = []
        for item in self.rank(targets):
            if item['available_offline']:
                continue
            try:
                validate_source_url(item.get('acquisition_url') or item['url'])
            except (KeyError, ValueError, TypeError):
                continue
            selected.append(item['id'])
            if len(selected) == max_items:
                break
        return {'source_ids': selected, 'targets': sorted(set(targets or [])),
                'max_items': max_items, 'reason': 'Highest ranked missing primary-source bytes on configured hosts',
                'applicability': 'unvalidated'}

    def retain(self, source_id, raw, *, acquisition_method, url=None):
        if source_id not in self.items:
            raise ValueError('unknown literature source ID')
        if not isinstance(raw, bytes) or not raw or len(raw) > MAX_FILE:
            raise ValueError('source bytes are empty or exceed the 16 MiB per-source bound')
        item = self.items[source_id]
        checksum = hashlib.sha256(raw).hexdigest()
        expected = item.get('source_sha256')
        if expected and (not isinstance(expected, str) or not SHA.fullmatch(expected) or expected != checksum):
            raise ValueError('source bytes differ from the pinned catalog SHA-256')
        relative = 'objects/' + checksum
        descriptor = {'source_id': source_id, 'sha256': checksum, 'bytes': len(raw),
            'artifact_path': relative, 'url': url or item.get('url'),
            'acquisition_method': acquisition_method, 'catalog_hash_matched': bool(expected),
            'source_identity_sha256': self._source_identity(item),
            'applicability': 'unvalidated', 'formal_status': 'not_certified'}
        with self.journal.transaction():
            path = self.cache / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.is_symlink() or not path.resolve().is_relative_to(self.cache.resolve()):
                raise ValueError('unsafe literature object path')
            if path.exists():
                if path.read_bytes() != raw:
                    raise ValueError('stored literature object is corrupt')
            else:
                with path.open('xb') as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
            # Acquisition methods can differ for the same exact source bytes.
            eid = 'literature:' + hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()
            self.journal.append('LITERATURE_ACQUIRED', descriptor, event_id=eid)
        return descriptor

    def verify(self):
        captures = self._captures()
        verified, external = 0, []
        for item in captures:
            path = self.cache / item['artifact_path']
            if path.is_symlink() or not path.resolve().is_relative_to(self.cache.resolve()):
                raise ValueError('unsafe literature artifact path')
            try:
                raw = path.read_bytes()
            except FileNotFoundError:
                omission = self.distribution_omissions.get(item['source_id'])
                if (omission is None
                        or any(omission[k] != item[k] for k in ('source_id', 'sha256', 'bytes', 'url'))
                        or item['artifact_path'] != 'objects/' + omission['sha256']):
                    raise
                external.append({**omission, 'artifact_path': item['artifact_path'],
                    'state': 'external_reference', 'body_verified': False,
                    'applicability': 'unvalidated', 'formal_status': 'not_certified'})
                continue
            if len(raw) != item['bytes'] or hashlib.sha256(raw).hexdigest() != item['sha256']:
                raise ValueError('literature artifact hash mismatch')
            verified += 1
        current = sum(self._current_capture(c) for c in captures)
        return {'state': 'passed', 'sources_verified': verified,
                'current_captures': current, 'stale_captures': len(captures) - current,
                'external_references': external, 'integrity_errors': [],
                'custody_status': 'verified_bodies_with_external_references' if external else 'verified_bodies',
                'applicability': 'unvalidated', 'formal_status': 'not_certified'}

    def prefetch(self, source_ids, *, max_items=3, max_total_bytes=24 * 1024 * 1024):
        if (type(max_items) is not int or not 1 <= max_items <= 4
                or type(max_total_bytes) is not int or not 1 <= max_total_bytes <= 32 * 1024 * 1024):
            raise ValueError('prefetch budget must be 1–4 sources and at most 32 MiB')
        if not isinstance(source_ids, list) or len(source_ids) > max_items:
            raise ValueError('prefetch selection exceeds the source count budget')
        if len(source_ids) != len(set(source_ids)) or any(x not in self.items for x in source_ids):
            raise ValueError('prefetch requires distinct known literature IDs')
        # Validate the whole plan before any network access.
        urls = [validate_source_url(self.items[x].get('acquisition_url') or self.items[x]['url'])
                for x in source_ids]
        opener = urllib.request.build_opener(_SafeRedirect())
        results, consumed, uncertain_read = [], 0, False
        retained = {x['source_id']: x for x in self._captures()
                    if self._current_capture(x) and self._source_available(x)}
        for source_id, url in zip(source_ids, urls):
            if source_id in retained:
                self.verify()
                results.append({**retained[source_id], 'state': 'already_retained'})
                continue
            remaining = max_total_bytes - consumed
            if remaining <= 0:
                results.append({'source_id': source_id, 'state': 'blocked', 'reason': 'byte budget exhausted'})
                continue
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'UniversalLaw-research/18 (bounded primary-source acquisition)'})
                with opener.open(req, timeout=15) as response:
                    validate_source_url(response.url)
                    # The per-file overflow probe is itself a body byte. Never
                    # ask the transport for a byte outside the shared budget.
                    read_limit = min(MAX_FILE + 1, remaining)
                    # A chunked/timeout read can consume bytes and then raise
                    # without reporting all of them in its exception. Reserve
                    # the entire request first; refund only a successful return.
                    consumed += read_limit
                    try:
                        raw = response.read(read_limit)
                    except Exception:
                        uncertain_read = True
                        raise
                    consumed -= read_limit - len(raw)
                    if len(raw) > MAX_FILE:
                        raise ValueError('source response exceeds the per-source byte budget')
                    headers = response.headers
                    length = headers.get('Content-Length')
                    framed_length = (int(length) if not headers.get('Transfer-Encoding')
                                     and isinstance(length, str) and length.isascii()
                                     and length.isdecimal() else None)
                    if framed_length is not None and framed_length != len(raw):
                        raise ValueError('source response is incomplete or exceeds remaining byte budget')
                    if len(raw) == read_limit:
                        # At the total boundary an extra EOF probe is forbidden.
                        # urllib's HTTPResponse enforces Content-Length framing;
                        # an unframed/chunked prefix cannot establish completeness.
                        if framed_length != len(raw):
                            raise ValueError('byte budget exhausted before complete response could be established')
                descriptor = self.retain(source_id, raw, acquisition_method='bounded HTTPS acquisition', url=url)
                results.append({**descriptor, 'state': 'acquired'})
            except Exception as error:
                failure = {'source_id': source_id, 'url': url, 'state': 'blocked',
                           'reason': type(error).__name__ + ': ' + str(error)}
                self.journal.append('LITERATURE_ACQUISITION_FAILED', failure)
                results.append(failure)
        return {'results': results, 'network_bytes_read': consumed,
                'network_bytes_read_is_upper_bound': uncertain_read,
                'max_total_bytes': max_total_bytes, 'applicability': 'unvalidated'}
