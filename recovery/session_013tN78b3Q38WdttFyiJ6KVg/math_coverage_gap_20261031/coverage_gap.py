#!/usr/bin/env python3
"""Math- PROOF.md packets versus a CBZ quarantine catalog: which have no disposition.

Usage: python3 -I -B coverage_gap.py MATH_REPO CATALOG_JSON CUT OUTPUT_JSON

MATH_REPO is a Git checkout that contains commit CUT. CATALOG_JSON is a
cbz/quarantine/catalog.json. Every */PROOF.md at CUT is classified:
QUARANTINED if its path or packet directory occurs anywhere in the catalog's
quarantines; INVENTORY_ONLY if it is only in proof_source_inventory; ABSENT
otherwise. INVENTORY_ONLY packets are then resolved by Git blob: a
byte-identical copy of a quarantined source is COVERED_BY_COPY, a
byte-identical copy of another listed packet is DUPLICATE_OF, and the rest
are NO_DISPOSITION. Name-only matches are not coverage.

Scope limit: only files named PROOF.md are censused.
"""
import hashlib
import json
import subprocess
import sys
from collections import Counter


def git(repo, *args):
    return subprocess.run(['git', '-C', repo, *args], capture_output=True, check=True).stdout


def main(repo, catalog_path, cut, output):
    catalog_bytes = open(catalog_path, 'rb').read()
    catalog = json.loads(catalog_bytes)
    quarantine_text = json.dumps(catalog['quarantines'])
    inventory = {row['path']: row for row in catalog['proof_source_inventory']
                 if row.get('repository', '').endswith('Math-')}
    proofs = sorted(p for p in git(repo, 'ls-tree', '-r', '--name-only', cut).decode().splitlines()
                    if p.endswith('/PROOF.md'))
    classes = {}
    for proof in proofs:
        packet = proof.rsplit('/', 1)[0]
        if proof in quarantine_text or packet + '/' in quarantine_text:
            classes[packet] = 'QUARANTINED'
        elif proof in inventory:
            classes[packet] = 'INVENTORY_ONLY'
        else:
            classes[packet] = 'ABSENT'
    blob = {packet: git(repo, 'rev-parse', f'{cut}:{packet}/PROOF.md').decode().strip() for packet in classes}
    # Quarantined sources may sit in other repositories (for example main): collect their blobs
    # from the catalog's own location strings when they name a PROOF.md at an exact commit.
    quarantined_blobs = {}
    for record in catalog['quarantines']:
        locations = record['location'] if isinstance(record['location'], list) else [record['location']]
        for location in locations:
            if isinstance(location, dict):  # structured test references; never a PROOF.md
                if not {'repository', 'commit', 'path'} <= set(location):
                    continue
                location = f"{location['repository']}@{location['commit']}:{location['path']}"
            repo_commit, _, path = location.partition(':')
            if not path.endswith('PROOF.md') or '@' not in repo_commit:
                continue
            name, commit = repo_commit.split('@', 1)
            source_repo = repo if name.endswith('Math-') else sys.argv[5] if len(sys.argv) > 5 else None
            if source_repo is None:
                continue
            try:
                quarantined_blobs[git(source_repo, 'rev-parse', f'{commit}:{path}').decode().strip()] = record['id']
            except subprocess.CalledProcessError:
                continue
    # Math- packets quarantined by directory also cover byte-identical copies elsewhere.
    for packet, cls in classes.items():
        if cls == 'QUARANTINED' and blob[packet] not in quarantined_blobs:
            ids = [r['id'] for r in catalog['quarantines'] if packet + '/' in json.dumps(r) or
                   packet + '/PROOF.md' in json.dumps(r)]
            quarantined_blobs[blob[packet]] = '/'.join(ids[:3]) + ' via ' + packet
    first_by_blob = {}
    for packet in sorted(classes, key=lambda p: (p.count('/'), p)):
        first_by_blob.setdefault(blob[packet], packet)
    rows = []
    for packet, cls in sorted(classes.items()):
        if cls != 'INVENTORY_ONLY':
            continue
        data = git(repo, 'show', f'{cut}:{packet}/PROOF.md')
        b = blob[packet]
        if b in quarantined_blobs:
            coverage = 'COVERED_BY_COPY ' + quarantined_blobs[b]
        elif first_by_blob[b] != packet:
            coverage = 'DUPLICATE_OF ' + first_by_blob[b]
        else:
            coverage = 'NO_DISPOSITION'
        rows.append({'packet': packet, 'proof': packet + '/PROOF.md', 'git_blob': b,
                     'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data),
                     'catalog_inventory_admission': inventory[packet + '/PROOF.md']['mathematical_admission'],
                     'coverage': coverage})
    counts = Counter(classes.values())
    kinds = Counter(r['coverage'].split()[0] for r in rows)
    report = {'schema': 'claude54.math-coverage-gap.v2', 'math_cut': cut,
              'catalog_sha256': hashlib.sha256(catalog_bytes).hexdigest(),
              'counts': {'proof_files': len(proofs), 'quarantined': counts['QUARANTINED'],
                         'inventory_only': counts['INVENTORY_ONLY'], 'absent': counts['ABSENT'],
                         'covered_by_copy': kinds['COVERED_BY_COPY'], 'duplicate_copies': kinds['DUPLICATE_OF'],
                         'no_disposition': kinds['NO_DISPOSITION']},
              'scope_limit': 'Only files named PROOF.md are censused.', 'rows': rows}
    with open(output, 'x') as stream:
        json.dump(report, stream, indent=1)
        stream.write('\n')
    print(json.dumps(report['counts']))


if __name__ == '__main__':
    if len(sys.argv) not in (5, 6):
        raise SystemExit(__doc__)
    main(*sys.argv[1:5])
