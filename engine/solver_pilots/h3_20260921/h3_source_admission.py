"""Local, candidate-only H3 source admission; no live authority or status moves.

A pinned numbered provenance record links the authored source to its registered
parent delivery. Current repository exclusions remain a separate, fresh deny
input. Only the manifest metadata and one declared proof body are materialized;
no archived program is imported and unrelated leaves are never opened.
"""
from pathlib import Path, PurePosixPath
import csv
import hashlib
import io
import json
import stat
import zipfile
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ADMISSION = HERE / 'sources/H3_AUTHORED_NESTED_SOURCE_V1.json'
ADMISSION_SHA256 = '104cb5d72f06ac74979a976f43492226145182a141f382fc8eae7b7b59d84df0'
ARCHIVE = 'research/campaigns/h3_rn_n6_20260920_v1.zip'
ARCHIVE_ID = {'bytes': 571735, 'sha256': '73b9e63800f77c677c504f3097fe0bb2459ff6c94cdee5772b56062320aefc21'}
MANIFEST_ID = {'bytes': 12008, 'sha256': '81770e26ec654d3b462f06cebaa20d0dd282cfd4df10ac593c7887ea0d292b40'}
BODY_ALLOWLIST = {'h3_floor/PROOF.md': {'bytes': 10423, 'sha256': 'b48d8baa39b80be9a15af9833234d9339ea0afa4cdff2e1c60ae9ef79f5cb4fa'}}
ANCESTOR_ID = '1ni4q7t4a6rIpDkxapmc9yU4Xt1wv3983'
FORBIDDEN_IDS = {'1VTiBaRlBvHGptiXqoEli4E5630nO7mNb', '1hnd7jbFwcCQIuOcbrzilMQoymsOzFQZb'}
FORBIDDEN_SHA = {'579ef6cd88d543ec14db45558e5e7c1899078095e0df295c7d46fd1810273831'}
CONTEXT_KEYS = {'context_id', 'scientific_use', 'source_usable_for_candidate', 'basis', 'scope',
                'source_archive_sha256', 'current_use_requires_revalidation', 'organizational_independence_credit'}
MAX_METADATA = 1_000_000
MAX_SOURCE_MAP = 8_000_000


def need(condition, message):
    if not condition:
        raise ValueError(message)


def identity(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def read_bounded(path, limit):
    path = Path(path).absolute()
    need(path == path.resolve() and not path.is_symlink(), 'noncanonical or symlink source input')
    need(stat.S_ISREG(path.stat().st_mode), 'regular source input required')
    with path.open('rb') as handle:
        raw = handle.read(limit + 1)
    need(len(raw) <= limit, 'source input exceeds size limit')
    return raw


def strict_json(raw):
    need(len(raw) <= MAX_METADATA, 'source metadata size limit')
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'duplicate source metadata key')
            result[key] = value
        return result
    def inexact(_):
        raise ValueError('noninteger source metadata number')
    return json.loads(raw, object_pairs_hook=pairs, parse_float=inexact, parse_constant=inexact)


def validate_context(context):
    """The caller can restrict use; it cannot supply the provenance warrant."""
    need(type(context) is dict and set(context) == CONTEXT_KEYS, 'unsupported or missing context fields')
    need(all(type(context[key]) is str and context[key] for key in ('context_id','basis','scope')),
         'context identity, basis and scope must be nonempty strings')
    need(context.get('scientific_use') == 'REQUIRES_REVIEW', 'context does not permit candidate-only use')
    need(context.get('source_usable_for_candidate') is True, 'source use denied or unknown')
    need(context.get('source_archive_sha256') == ARCHIVE_ID['sha256'], 'context/source identity mismatch')
    need(type(context.get('organizational_independence_credit')) is int
         and context['organizational_independence_credit'] == 0, 'no independence credit may be imported')
    need(context.get('current_use_requires_revalidation') is True, 'current revalidation required')


def safe_name(name):
    need(type(name) is str and name and '\\' not in name, 'invalid archive name')
    path = PurePosixPath(name)
    need(not path.is_absolute() and str(path) == name and '..' not in path.parts, 'unsafe archive name')
    return name


def check_exclusions(record, rows):
    need(type(rows) is list, 'exclusions must be a list')
    upstream = record['upstream']
    carriers = {ANCESTOR_ID, upstream['carrier']['drive_id']}
    hashes = {record['ancestor']['sha256'], ARCHIVE_ID['sha256'], MANIFEST_ID['sha256'],
              record['ancestor_manifest']['sha256'], upstream['carrier']['sha256']}
    hashes |= {v['sha256'] for v in BODY_ALLOWLIST.values()}
    hashes |= {v['sha256'] for v in upstream['members'].values()}
    paths = {record['ancestor']['name'], ARCHIVE, 'MANIFEST.json', 'DELIVERY_MANIFEST.json'}
    paths |= set(BODY_ALLOWLIST) | set(upstream['members'])
    need(not hashes.intersection(FORBIDDEN_SHA) and not carriers.intersection(FORBIDDEN_IDS), 'permanent source exclusion')
    for row in rows:
        need(type(row) is dict and row.get('kind') in ('drive_object', 'archive_member'), 'unknown exclusion record')
        need(type(row.get('carrier_id')) is str and row['carrier_id'], 'missing exclusion carrier')
        need(type(row.get('member_path')) is str and row['member_path'], 'missing exclusion path')
        sha = row.get('payload_sha256')
        need(sha is None or (type(sha) is str and len(sha) == 64
                            and all(ch in '0123456789abcdef' for ch in sha)), 'invalid excluded payload digest')
        need(sha not in hashes, 'excluded source payload')
        if row['kind'] == 'drive_object':
            need(row['carrier_id'] not in carriers, 'excluded source ancestor')
        else:
            # An unrelated held leaf in the same container does not deny this leaf.
            name = row['member_path'].replace('!/', '/')
            matches = any(name == path or name.endswith('/' + path) for path in paths)
            relevant = row['carrier_id'] in carriers or row.get('carrier_sha256') in hashes
            need(not (matches and relevant), 'excluded source member')


def upstream_gate(record):
    """Authenticate two declared upstream identities from current metadata only."""
    raw_inventory = read_bounded(ROOT / 'drive/inventory.jsonl', MAX_SOURCE_MAP)
    raw_payloads = read_bounded(ROOT / 'drive/source_map/Payloads.csv', MAX_SOURCE_MAP)
    inventory = [strict_json(line) for line in raw_inventory.splitlines() if line.strip()]
    need(all(type(row) is dict for row in inventory), 'invalid inventory row')
    carrier = record['upstream']['carrier']
    matches = [row for row in inventory if row.get('id') == carrier['drive_id']]
    need(len(matches) == 1, 'missing or ambiguous upstream carrier mapping')
    source = matches[0]
    need(type(source.get('bytes')) is int and source['bytes'] == carrier['bytes']
         and source.get('sha256') == carrier['sha256'], 'upstream carrier identity mismatch')
    need(source.get('context') == 'RESEARCH_SOURCE_CHECK_STATUS'
         and source.get('access_status') == 'ARCHIVE_INDEXED', 'upstream carrier unavailable')
    name = source.get('path', '')
    need(type(name) is str and name.startswith('01_ACTIVE_RESEARCH_PACKAGES/')
         and not any(word in name.lower() for word in ('quarantine', 'legacy', '99_do_not_open')),
         'upstream carrier outside eligible active source path')
    def drive_link(link):
        need(type(link) is str, 'missing source link')
        url = urlsplit(link)
        need(url.scheme == 'https' and url.netloc == 'drive.google.com'
             and url.path == '/file/d/' + carrier['drive_id'] + '/view' and url.query in ('', 'usp=drivesdk') and not url.fragment,
             'upstream source link mismatch')
    drive_link(source.get('link'))
    try:
        rows = list(csv.reader(io.StringIO(raw_payloads.decode('utf-8')), strict=True))
    except csv.Error as error:
        raise ValueError('malformed source-map CSV') from error
    need(rows and len(rows[0]) == len(set(rows[0])), 'missing or duplicate source-map headers')
    required = {'Source name', 'SHA-256', 'Bytes', 'Conversion', 'Context', 'Scope holds', 'Original source link'}
    need(required <= set(rows[0]) and all(len(row) == len(rows[0]) for row in rows[1:]), 'malformed source-map shape')
    payloads = [dict(zip(rows[0], row)) for row in rows[1:]]
    for name, expected in record['upstream']['members'].items():
        matches = [row for row in payloads if row['SHA-256'] == expected['sha256'] or row['Source name'] == name]
        need(len(matches) == 1, 'missing or ambiguous upstream member mapping')
        row = matches[0]
        need(row['Source name'] == name and row['SHA-256'] == expected['sha256']
             and row['Bytes'] == str(expected['bytes']), 'upstream member identity mismatch')
        need(row['Conversion'] == 'TEXT_READING_COPY' and row['Context'] == 'RESEARCH_SOURCE_CHECK_STATUS'
             and strict_json(row['Scope holds']) == [], 'upstream member unavailable or held')
        drive_link(row['Original source link'])
    return {'drive/inventory.jsonl': identity(raw_inventory),
            'drive/source_map/Payloads.csv': identity(raw_payloads)}


def metadata_gate(context):
    """Read only local policy/provenance metadata, before opening any ZIP."""
    validate_context(context)
    raw = read_bounded(ADMISSION, MAX_METADATA)
    record = strict_json(raw)
    need(identity(raw)['sha256'] == ADMISSION_SHA256, 'missing, changed or unreviewed source admission record')
    need(record['schema'] == 'h3-authored-nested-source-v1'
         and record['candidate_use'] == 'REQUIRES_REVIEW'
         and record['eligibility'] == 'CANDIDATE_READING'
         and record['scope'] == context.get('scope'), 'ineligible or mismatched source scope')
    need(record['archive'] == dict(path=ARCHIVE, **ARCHIVE_ID)
         and record['manifest'] == dict(path='MANIFEST.json', **MANIFEST_ID)
         and record['body_allowlist'] == BODY_ALLOWLIST
         and record['ancestor']['drive_id'] == ANCESTOR_ID, 'source admission identity mismatch')
    custody_raw = read_bounded(HERE / record['current_custody']['path'], MAX_METADATA)
    need(identity(custody_raw) == {key: record['current_custody'][key] for key in ('bytes','sha256')},
         'current custody receipt identity mismatch')
    custody = strict_json(custody_raw)
    need(custody['schema'] == 'C11_SOURCE_CUSTODY_V1'
         and custody['source_kind'] == 'AUTHORED_ARCHIVE_MEMBER'
         and custody['ancestor'] == {key: record['ancestor'][key] for key in ('drive_id','bytes','sha256')} | {'exact_fresh_raw_readback': True}
         and custody['member'] == record['archive']
         and custody['receipt_utc'] == record['current_custody']['observed_receipt_utc']
         and record['implementation_claim'] in custody['current_claim_ids']
         and custody['scientific_acceptance'] is False
         and type(custody['organizational_independence_credit']) is int
         and custody['organizational_independence_credit'] == 0, 'custody/admission scope mismatch')
    exclusions_raw = read_bounded(ROOT / 'quarantine/EXCLUSIONS.json', MAX_METADATA)
    exclusions = strict_json(exclusions_raw)
    need(type(exclusions) is dict and 'exclusions' in exclusions, 'missing current exclusions')
    check_exclusions(record, exclusions['exclusions'])
    upstream = upstream_gate(record)
    return {'schema': 'h3-local-source-gate-v1',
            'admission_record': identity(raw), 'custody_receipt': identity(custody_raw),
            'exclusions': identity(exclusions_raw),
            'upstream_metadata': upstream,
            'ancestor_drive_id': ANCESTOR_ID, 'candidate_use': 'REQUIRES_REVIEW',
            'body_allowlist': BODY_ALLOWLIST,
            'authority': 'Pinned local provenance snapshot plus current repository exclusions; not live Drive permissions or canonical scientific admission'}


def source_check(path, context):
    gate = metadata_gate(context)  # Must precede even opaque archive reads.
    raw = read_bounded(path, ARCHIVE_ID['bytes'])
    need(identity(raw) == ARCHIVE_ID, 'original H3 archive identity mismatch')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist()
        names = {item.filename for item in infos}
        need(len(infos) == len(names) == 71, 'archive member set mismatch')
        need(sum(item.file_size for item in infos) <= 5_000_000, 'expanded archive size limit')
        for item in infos:
            safe_name(item.filename)
            need(stat.S_IFMT(item.external_attr >> 16) in (0, stat.S_IFREG), 'nonregular archive member')
            need(0 <= item.file_size <= 2_000_000 and not item.flag_bits & 1, 'oversized or encrypted archive member')
            need(item.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED), 'unsupported compression')
        manifest_raw = archive.read('MANIFEST.json')
        need(identity(manifest_raw) == MANIFEST_ID, 'source manifest mismatch')
        rows = strict_json(manifest_raw)['files']
        need(type(rows) is list and len(rows) == 70, 'source manifest cardinality mismatch')
        records = {row['path']: {'bytes': row['bytes'], 'sha256': row['sha256']} for row in rows}
        need(len(records) == 70 and set(records) | {'MANIFEST.json'} == names, 'duplicate or incomplete manifest mapping')
        for name, expected in BODY_ALLOWLIST.items():
            need(records.get(name) == expected, 'selected body manifest identity mismatch')
        # Recheck current policy after archive metadata and before analytic bodies.
        need(metadata_gate(context) == gate, 'source metadata changed before body read')
        bodies = {name: identity(archive.read(name)) for name in BODY_ALLOWLIST}
        need(bodies == BODY_ALLOWLIST, 'selected source body identity mismatch')
    need(metadata_gate(context) == gate, 'source metadata changed during body read')
    return {'archive_sha256': ARCHIVE_ID['sha256'], 'manifest_sha256': MANIFEST_ID['sha256'],
            'archive_entries': len(infos), 'manifest_entries_checked_as_metadata': len(rows),
            'analytic_bodies_read': bodies, 'analytic_proof_sha256': bodies['h3_floor/PROOF.md']['sha256'],
            'admission': gate, 'scope': 'Candidate-only source byte custody and local eligibility; not acceptance'}
