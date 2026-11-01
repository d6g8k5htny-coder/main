#!/usr/bin/env python3
"""Federation census of mathematical artifacts against a CBZ quarantine catalog.

Usage: python3 -I -B federation_census.py CONFIG_JSON CATALOG_JSON CLOSED_JSON QP_RECORDS_JSON OUTPUT_JSON

CONFIG_JSON maps a repository label to {"path": checkout, "ref": commit-ish,
"catalog_name": name used in catalog locations or null}. Candidate artifacts are
files matching the patterns below. Each is classified, first match wins:

  CLOSED_SOURCE    a .lean file that hosts at least one CLOSED declaration
  QUARANTINED      the path or one of its ancestor directories is a quarantine location
  PROPOSED_QP_C54  the path is inside a packet covered by a QP-C54 proposal
  COPY_OF_COVERED  its Git blob equals the blob of a file classified above (any repository)
  INVENTORY_ONLY   listed in proof_source_inventory, noncore_lean_inventory or
                   main_census_independent_validation, but given no disposition
  NONE             not mentioned by the catalog at all

Locations are compared by path for the same repository only; the catalog's own commits
are recorded but a path match at another commit is reported as path coverage, so the
output also lists, per covered file, whether its blob at REF equals the blob the catalog
pinned (when the catalog pins one).
"""
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict

PATTERNS = {
    'lean': re.compile(r'\.lean$'),
    'proof_note': re.compile(r'(^|/)[^/]*(PROOF|THEOREM|DERIVATION|ERRATUM|LEMMA)[^/]*\.md$|^(imports|specifications/[^/]+/sources)/.*\.md$', re.I),
    'certificate_program': re.compile(r'(^|/)(certificate|cert_|verify_|check)[^/]*\.py$', re.I),
    'certificate_data': re.compile(r'(^|/)(CERTIFICATE|RESULTS|ENCLOSURE)[^/]*\.json$'),
}


def git(path, *args):
    return subprocess.run(['git', '-C', path, *args], capture_output=True, text=True, check=True).stdout


def kind_of(path):
    for kind, pattern in PATTERNS.items():
        if pattern.search(path):
            return kind
    return None


def location_pairs(catalog):
    """(catalog repository name, path) for every quarantine location string."""
    pairs = []
    for record in catalog['quarantines']:
        locations = record['location'] if isinstance(record['location'], list) else [record['location']]
        for location in locations:
            if isinstance(location, dict):
                if {'repository', 'path'} <= set(location):
                    pairs.append((location['repository'], location['path'], record['id']))
                for p in location.get('paths', []) if isinstance(location.get('paths'), list) else []:
                    pairs.append((location.get('repository', ''), p, record['id']))
                continue
            repo_commit, _, path = location.partition(':')
            name = repo_commit.split('@', 1)[0]
            path = path.split('#', 1)[0].rstrip('/')
            if path:
                pairs.append((name, path, record['id']))
    return pairs


def short(name):
    return name.split('/')[-1] if name else name


def main(config_path, catalog_path, closed_path, qp_path, output):
    config = json.load(open(config_path))
    catalog = json.load(open(catalog_path))
    closed = json.load(open(closed_path))['entries']
    qp = json.load(open(qp_path))['records']
    quarantined = defaultdict(dict)
    for name, path, rid in location_pairs(catalog):
        quarantined[short(name)].setdefault(path, rid)
    closed_paths = defaultdict(set)
    for entry in closed:
        closed_paths[short(entry['repository'])].add(entry['path'])
    inventory = defaultdict(dict)
    for key in ('proof_source_inventory', 'noncore_lean_inventory', 'main_census_independent_validation'):
        for row in catalog.get(key, []):
            inventory[short(row.get('repository'))][row['path']] = (key, row.get('git_blob'))
    qp_packets = defaultdict(dict)
    for record in qp:
        location = record['location'][0]
        name, _, packet = location.partition(':')
        qp_packets[short(name.split('@')[0])][packet] = record['id']

    def covering(prefixes, path, packet):
        # Packet-level semantics: the path itself, an ancestor, or any location inside its packet.
        parts = path.split('/')
        for i in range(len(parts), 0, -1):
            candidate = '/'.join(parts[:i])
            if candidate in prefixes:
                return candidate, prefixes[candidate]
        if packet:
            for location, rid in prefixes.items():
                if location == packet or location.startswith(packet + '/'):
                    return location, rid
        return None

    def packet_of(path, proof_dirs):
        parts = path.split('/')[:-1]
        for i in range(len(parts), 0, -1):
            candidate = '/'.join(parts[:i])
            if candidate in proof_dirs:
                return candidate
        return '/'.join(parts) or None

    rows = []
    blob_cover = {}
    pending = []
    for label, spec in config.items():
        name = spec.get('catalog_name')
        files = [f for f in git(spec['path'], 'ls-tree', '-r', '--name-only', spec['ref']).split('\n') if f]
        blobs = {}
        if files:
            for line in git(spec['path'], 'ls-tree', '-r', spec['ref']).split('\n'):
                if line:
                    meta, path = line.split('\t', 1)
                    blobs[path] = meta.split()[2]
        proof_dirs = {f.rsplit('/', 1)[0] for f in files if f.endswith('/PROOF.md')}
        for path in files:
            kind = kind_of(path)
            if not kind:
                continue
            packet = packet_of(path, proof_dirs)
            row = {'repository': label, 'ref': spec['ref'], 'path': path, 'kind': kind, 'git_blob': blobs.get(path),
                   'packet': packet, 'role': 'review_evidence' if path.startswith('reviews/') else 'artifact'}
            if name and kind == 'lean' and path in closed_paths[name]:
                row['coverage'] = 'CLOSED_SOURCE'
            elif name and covering(quarantined[name], path, packet if kind != 'lean' else None):
                where, rid = covering(quarantined[name], path, packet if kind != 'lean' else None)
                row['coverage'], row['by'] = 'QUARANTINED', f'{rid} ({where})'
            elif name and covering(qp_packets[name], path, packet):
                where, rid = covering(qp_packets[name], path, packet)
                row['coverage'], row['by'] = 'PROPOSED_QP_C54', f'{rid} ({where})'
            else:
                row['coverage'] = None
            if name and path in inventory[name]:
                key, pinned = inventory[name][path]
                row['inventory'] = key
                if pinned:
                    row['inventory_blob_matches_ref'] = pinned == row['git_blob']
            if row['coverage']:
                blob_cover.setdefault(row['git_blob'], f"{label}:{path}")
            rows.append(row)
            if not row['coverage']:
                pending.append(row)
    for row in pending:
        source = blob_cover.get(row['git_blob'])
        if source:
            row['coverage'], row['by'] = 'COPY_OF_COVERED', source
        elif row.get('inventory'):
            row['coverage'] = 'INVENTORY_ONLY'
        else:
            row['coverage'] = 'NONE'
    summary = defaultdict(Counter)
    for row in rows:
        summary[f"{row['repository']}@{row['ref'][:8]}"][f"{row['kind']}:{row['coverage']}" + (':review' if row['role'] == 'review_evidence' else '')] += 1
    report = {'schema': 'claude54.federation-math-census.v3', 'patterns': {k: p.pattern for k, p in PATTERNS.items()},
              'config': config, 'summary': {k: dict(sorted(v.items())) for k, v in summary.items()}, 'rows': rows}
    with open(output, 'x') as stream:
        json.dump(report, stream, indent=1)
        stream.write('\n')
    for repo, counts in report['summary'].items():
        print(repo, json.dumps(counts))


if __name__ == '__main__':
    if len(sys.argv) != 6:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
