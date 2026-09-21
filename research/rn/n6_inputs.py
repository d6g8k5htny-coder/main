"""Authenticate the previously delivered N6 implementation for new RN work.

The authored Taylor module executes; historic RN5 source programs remain data.
Identity/eligibility is checked on each entry, including a warm module cache.
No source status, imported premise or scientific gate is promoted here.
"""
from pathlib import Path
import csv
import io
import types
from urllib.parse import urlparse

from tools import h3_rn_n6_check as archive_check
from tools import twelve_project_check as runtime

ROOT = Path(__file__).resolve().parents[2]
MEMBER = 'rn_n6/side24_taylor.py'
RN5_ID = '1LtnvNd0vAW-y3pzbyHLjgtF7Uw5sTph5'
RN5_SHA = 'ac89f60b8206bfe011e6c2bc653e7acb39bc83a55c2a2c8fd1e17fd70c6c0383'
_MODULE = None
METADATA_LIMIT = 8 * 1024 * 1024


def authenticated_inputs(root=None):
    """Return checked archive members/identities without executing a program."""
    root = ROOT if root is None else Path(root)
    if root.resolve() != ROOT.resolve():
        raise ValueError('N6 imports require this exact runtime checkout')
    exclusions_raw = runtime.read_bounded(root / 'quarantine/EXCLUSIONS.json', runtime.MAX_JSON)
    exclusions = runtime.strict_json(exclusions_raw)['exclusions']
    forbidden = {row.get('carrier_id') for row in exclusions if row.get('kind') == 'drive_object'}
    if RN5_ID in forbidden:
        raise ValueError('RN5 source is excluded')
    inventory_raw = runtime.read_bounded(root / 'drive/inventory.jsonl', METADATA_LIMIT)
    inventory = [runtime.strict_json(line) for line in inventory_raw.splitlines() if line.strip()]
    matches = [row for row in inventory if row.get('id') == RN5_ID]
    if len(matches) != 1:
        raise ValueError('RN5 source inventory identity is ambiguous')
    record = matches[0]
    path = record.get('path', '')
    if (record.get('context') != 'RESEARCH_SOURCE_CHECK_STATUS' or
            record.get('access_status') != 'TEXT_READING_COPY' or
            record.get('sha256') != RN5_SHA or record.get('bytes') != 13725 or
            not path.startswith('01_ACTIVE_RESEARCH_PACKAGES/') or
            any(token in path.lower() for token in ('quarantine', 'legacy', '99_do_not_open'))):
        raise ValueError('RN5 source is ineligible')
    coverage_raw = runtime.read_bounded(root / 'drive/source_map/Payloads.csv', METADATA_LIMIT)
    reader = csv.DictReader(io.StringIO(coverage_raw.decode('utf-8')), strict=True)
    required_headers = {'SHA-256', 'Bytes', 'Context', 'Scope holds', 'Original source link'}
    if (reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames) or
            not required_headers <= set(reader.fieldnames)):
        raise ValueError('invalid source coverage columns')
    coverage = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in coverage):
        raise ValueError('invalid source coverage row width')
    matching_coverage = [row for row in coverage if row.get('SHA-256') == RN5_SHA]
    if (len(matching_coverage) != 1 or matching_coverage[0].get('Context') != 'RESEARCH_SOURCE_CHECK_STATUS' or
            matching_coverage[0].get('Bytes') != '13725' or
            matching_coverage[0].get('Scope holds') != '[]'):
        raise ValueError('RN5 source coverage eligibility changed')
    link = urlparse(matching_coverage[0]['Original source link'])
    if (link.scheme != 'https' or link.netloc != 'drive.google.com' or
            link.path != '/file/d/' + RN5_ID + '/view'):
        raise ValueError('RN5 source coverage identity changed')
    raw = runtime.read_bounded(root / archive_check.ARCHIVE, archive_check.ARCHIVE_BYTES)
    members, dependencies = archive_check.inspect_archive(raw, exclusions)
    identities = runtime.repository_inputs(root, dependencies)
    identities[archive_check.ARCHIVE] = runtime.identity(raw)
    identities[archive_check.ARCHIVE + '::' + MEMBER] = runtime.identity(members[MEMBER])
    identities['quarantine/EXCLUSIONS.json'] = runtime.identity(exclusions_raw)
    identities['drive/inventory.jsonl'] = runtime.identity(inventory_raw)
    identities['drive/source_map/Payloads.csv'] = runtime.identity(coverage_raw)
    return members, identities


def checked_n6(root=None):
    """Return (module, identities), refusing stale inputs before cache reuse."""
    members, identities = authenticated_inputs(root)
    global _MODULE
    if _MODULE is None:
        module = types.ModuleType('authenticated_rn_n6')
        module.__file__ = archive_check.ARCHIVE + '::' + MEMBER
        exec(compile(members[MEMBER], module.__file__, 'exec'), module.__dict__)
        _MODULE = module
    return _MODULE, identities
