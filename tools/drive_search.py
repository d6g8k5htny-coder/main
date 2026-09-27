"""Derived search view over existing custody records; never a governing catalog.

No search result authenticates a scientific claim. Absence is not deletion or
account completeness. Held/history records are metadata-only, never hydrated.
"""
from collections import Counter
from contextlib import ExitStack
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import os
import tempfile
import xml.etree.ElementTree as ET
import zipfile

from tools.drive_coverage import DEFAULT as BASELINE
from tools.drive_reconcile import DEFAULT as RECON
from tools.native_export_check import Reader

ROOT = Path(__file__).resolve().parents[1]
SEARCH = 'drive/search/2026-09-20'
EXCLUDED = ('PERSONAL', 'HELD_', 'DO_NOT_OPEN', 'QUARANTINE', 'LEGACY', 'HISTORY')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def content_allowed(row):
    labels = ' '.join(str(row.get(k, '')) for k in ('context', 'status', 'scope_holds', 'path'))
    return not any(x in labels.upper() for x in EXCLUDED)


def role(row):
    if not content_allowed(row):
        return 'METADATA_ONLY_EXCLUDED'
    if row.get('source_role'):
        return row['source_role']
    if 'ACCESSIBILITY' in row.get('context', '') or 'READING COP' in row.get('path', '').upper():
        return 'READING_COPY'
    return 'SOURCE_CHECK_STATUS'  # not a claim of originality or current scientific approval


def load_current(entries, root=None):
    root = Path(root) if root is not None else ROOT
    base = root / BASELINE
    scope = json.loads((base / 'scope.json').read_text())
    rows = {e['id']: dict(e) for e in entries}
    for e in scope['additional_sources']:
        if e['id'] not in rows:
            rows[e['id']] = dict(e)
    cover = {e['id']: e for e in map(json.loads, (base / 'coverage.jsonl').read_text().splitlines())}
    spec = json.loads((root / RECON / 'reconciliation.json').read_text())
    observations = {}
    for folder in json.loads((root / RECON / 'current-membership.json').read_text()):
        observations.update({x['id']: x for x in folder['items']})
    for e in spec['payloads']:
        cover[e['id']] = e
        if e['id'] not in rows:
            m = observations.get(e['id'], {})
            rows[e['id']] = {'id': e['id'], 'title': e.get('title', m.get('title', e['id'])),
                             'path': m.get('title', e.get('title', e['id'])),
                             'mimeType': m.get('mime_type', e['export_mime_type']),
                             'context': 'OPERATIONS', 'access_status': e['status']}
    for ident, e in rows.items():
        c = cover[ident]
        hashes = []
        if e.get('sha256'):
            hashes.append({'sha256': e['sha256'], 'kind': 'SOURCE_MAP_REPORTED_SHA256',
                           'snapshot': '2026-09-17', 'bytes': e.get('bytes')})
        e.update(record_key='file:' + ident, record_type='FILE', status=c['status'],
                 snapshot=spec['membership_cutoff'], custody=c,
                 link='https://drive.google.com/' + ('drive/folders/' if c['status']=='FOLDER_METADATA' else 'file/d/') + ident,
                 native_revision=c.get('revision_observation', c.get('native_revision_observation')),
                 revision_pinned=False if c['status']=='NATIVE_EXPORT' else None)
        if c.get('sha256'):
            kind = 'NATIVE_EXPORT_SHA256' if c['status']=='NATIVE_EXPORT' else 'RAW_FILE_SHA256'
            hashes.insert(0, {'sha256': c['sha256'], 'kind': kind, 'snapshot': c.get('observed_at', spec['membership_cutoff']), 'bytes': c['bytes']})
            e.update(sha256=c['sha256'], bytes=c['bytes'])
        e.setdefault('bytes', None)
        e['hashes'] = hashes
        e['source_role'] = role(e)
        e['content_eligible'] = content_allowed(e)
    if len(rows) != spec['scoped_objects']:
        raise ValueError('current scope does not reconcile')
    for e in json.loads((root / SEARCH / 'deliveries.json').read_text())['records']:
        if e['id'] in rows:
            raise ValueError('delivery collides with reconciled file identity')
        data = (root / e['path']).read_bytes()
        if len(data) != e['bytes'] or digest(data) != e['sha256'] or e.get('raw_readback_verified') is not True:
            raise ValueError('delivery custody mismatch')
        rows[e['id']] = {**e, 'record_key': 'file:'+e['id'], 'record_type': 'FILE',
                        'access_status':e['status'], 'link':'https://drive.google.com/file/d/'+e['id'],
                        'hashes':[{'sha256':e['sha256'],'kind':'RAW_FILE_SHA256','snapshot':e['snapshot'],'bytes':e['bytes']}],
                        'content_eligible':content_allowed(e), 'custody': e, 'revision_pinned':None}
    return list(rows.values())


def archive_records(root=None):
    root = Path(root) if root is not None else ROOT
    out = []
    with (root / 'drive/source_map/Archive_Members.csv').open() as f:
        for number, row in enumerate(csv.DictReader(f), 2):
            e = {'record_key':f'archive-row:{number}', 'record_type':'ARCHIVE_MEMBER',
                 'id':row['Carrier ID'], 'title':row['Member path'],
                 'path':row['Carrier title']+'!'+row['Member path'],
                 'member_path':row['Member path'], 'carrier_title':row['Carrier title'],
                 'link':row['Carrier link'], 'carrier_sha256':row['Carrier SHA-256'],
                 'sha256':row['Payload SHA-256'], 'bytes':int(row['Bytes']),
                 'context':row['Context'], 'scope_holds':row['Scope holds'],
                 'status':row['Read status'], 'snapshot':'2026-09-17 source-map archive table',
                 'occurrence_row':number, 'content_eligible':False,
                 'hashes':[{'sha256':row['Payload SHA-256'],'kind':'ARCHIVE_MEMBER_SHA256',
                            'snapshot':'2026-09-17 source-map archive table','bytes':int(row['Bytes'])}]}
            e['source_role'] = 'ARCHIVE_MEMBER_CHECK_STATUS' if content_allowed(e) else 'METADATA_ONLY_EXCLUDED'
            out.append(e)
    return out


def validate_prefix(prefix):
    if not isinstance(prefix, str) or not re.fullmatch('[0-9a-fA-F]{8,64}', prefix):
        raise ValueError('SHA-256 query requires 8..64 hexadecimal characters')
    return prefix.lower()


def sha_matches(records, prefix):
    prefix = validate_prefix(prefix)
    hits = []
    for e in records:
        matches = [h for h in e.get('hashes', []) if h['sha256'].lower().startswith(prefix)]
        if matches:
            hits.append({**e, 'match_kind':'HASH_IDENTITY', 'matched_hashes':matches})
    distinct = sorted({h['sha256'].lower() for e in hits for h in e['matched_hashes']})
    return {'query':prefix, 'distinct_digests':distinct, 'ambiguous':len(distinct)>1,
            'matches':hits, 'match_count':len(hits)}


def keyword_matches(records, query, *, use_index=False):
    if not isinstance(query, str) or not query.strip():
        raise ValueError('nonempty keyword query required')
    q = query.casefold()
    # Index construction pays off only for sustained in-process workloads.
    # Default/CLI callers keep the original scan. An opted-in index supplies
    # candidates only; matching/ranking and held metadata navigation stay here.
    if use_index and query.isascii():
        from tools.drive_search_index import candidates
        if not isinstance(records, (list, tuple)):
            records = list(records)
        selected = candidates(records, q)
        if selected is not None:
            records = selected
    out = [{**e, 'match_kind':'METADATA'} for e in records
           if any(q in str(e.get(k, '')).casefold() for k in ('title','path','path_snapshot','id','context'))]
    # A rank is navigation only. It cannot confer proof authority.
    return sorted(out, key=lambda e:(e['title'].casefold()!=q,
                  e['source_role'] in ('READING_COPY','METADATA_ONLY_EXCLUDED'), e['record_key']))


def extract_text(data, row):
    mime = row.get('export_mime_type', row.get('mimeType', ''))
    title = row.get('title', '').lower()
    if mime.endswith('wordprocessingml.document'):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            xml = z.read('word/document.xml')
        body = ET.fromstring(xml)
        ns = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
        return '\n'.join(''.join(x.text or '' for x in p.iter(ns+'t')) for p in body.iter(ns+'p')), 'DOCX_PARAGRAPH_TEXT'
    if mime.startswith('text/') or mime in ('application/json','application/x-python','application/javascript') or title.endswith(('.md','.txt','.py','.json','.tex','.csv','.yaml','.yml')):
        return data.decode('utf-8-sig'), 'UTF8_RAW_TEXT'
    return None, 'UNSUPPORTED_FORMAT'


def cache_path(root):
    """Disposable derived cache keyed by the committed extraction report."""
    report = (root / SEARCH / 'text-coverage.json').read_bytes()
    return Path(tempfile.gettempdir()) / ('codex-drive-search-'+digest(report)+'.jsonl.gz')


def build_text_cache(records, root=None, *, write_report=True):
    root = Path(root) if root is not None else ROOT
    texts = []; dispositions = []
    with ExitStack() as stack:
        reader = Reader(root, stack)
        for e in records:
            why = None
            if not e.get('content_eligible'): why = 'SCOPE_EXCLUDED'
            elif not e.get('sha256'): why = 'NO_STORED_PAYLOAD'
            elif e['bytes'] > 5_000_000: why = 'SIZE_LIMIT_5000000'
            if why is None:
                c = e['custody']; data = reader.read(c)
                try:
                    text, why = extract_text(data, {**e, **c})
                except UnicodeError:
                    text, why = None, 'NOT_UTF8'
                if text is not None:
                    texts.append({'record_key':e['record_key'], 'source_sha256':e['sha256'],
                                  'extraction':why, 'text_sha256':digest(text.encode()), 'text':text})
                    why = 'INDEXED_TEXT'
            dispositions.append({'record_key':e['record_key'], 'disposition':why})
    out = root / SEARCH
    raw = ''.join(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n' for x in texts).encode()
    compressed = gzip.compress(raw,mtime=0)
    report = {'schema_version':1,'records':len(records),'text_records':len(texts),
              'counts':dict(Counter(x['disposition'] for x in dispositions)), 'dispositions':dispositions,
              'text_uncompressed_sha256':digest(raw),
              'archive_member_text_indexed':False, 'all_native_features_preserved':False,
              'whole_account_complete':False, 'scientific_authority':'NONE'}
    if write_report:
        (out / 'text-coverage.json').write_text(json.dumps(report,indent=2)+'\n')
    elif report != json.loads((out / 'text-coverage.json').read_text()):
        raise ValueError('regenerated extraction differs from pinned report')
    target = cache_path(root)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as f:
        f.write(compressed)
        temporary = f.name
    os.replace(temporary, target)
    return {k:v for k,v in report.items() if k!='dispositions'}


def text_matches(records, query, root=None, *, hash_mention=False):
    root = Path(root) if root is not None else ROOT
    if not isinstance(query,str) or not query.strip():
        raise ValueError('nonempty content query required')
    q = query.casefold(); by_key = {e['record_key']:e for e in records}; out = []
    path = cache_path(root)
    if not path.exists():
        build_text_cache(records,root,write_report=False)
    cache = path.read_bytes()
    report = json.loads((root / SEARCH / 'text-coverage.json').read_text())
    raw = gzip.decompress(cache)
    if digest(raw) != report['text_uncompressed_sha256']:
        raise ValueError('text cache digest mismatch')
    for line in raw.split(b'\n'):
        if not line: continue
        t = json.loads(line); e = by_key.get(t['record_key'])
        if e is None or not e.get('content_eligible') or e['sha256'] != t['source_sha256'] or digest(t['text'].encode()) != t['text_sha256']:
            raise ValueError('text cache source/scope mismatch')
        pos = t['text'].casefold().find(q)
        if pos >= 0:
            out.append({**e, 'match_kind':'HASH_MENTION' if hash_mention else 'CONTENT_TEXT',
                        'extraction':t['extraction'], 'text_sha256':t['text_sha256'],
                        'snippet':t['text'][max(0,pos-80):pos+len(query)+160].replace('\n',' ')})
    return out


def public_record(e):
    return {k:v for k,v in e.items() if k!='custody'}
