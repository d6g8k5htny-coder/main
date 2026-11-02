#!/usr/bin/env python3
"""Build a candidate cbz/quarantine/catalog.json that carries the QP-C54 proposals beside, not inside, the adopted rows.

Usage: python3 -I -B build_candidate.py STAGING_CATALOG T1_RECORDS T2_RECORDS OUTPUT

The staging catalog's keys and values are kept exactly (same order, same serialization: indent 2,
ensure_ascii False, trailing newline), so its bytes up to its closing brace are unchanged. Four keys are
appended at the end:

  proposed_quarantines          tranche-1 records and tranche-2 NEW_RECORD records, unchanged
  proposed_pinning_amendments   tranche-2 PINNING_AMENDMENT records, each with target_record added
  proposed_non_obligations      tranche-2 objects classified NOT_OBLIGATION, unchanged
  proposal_provenance           source files, digests, counts and the adoption boundary

Nothing is adopted: `quarantines`, `counts`, `status` and `adoption_scope` are untouched, so every consumer
of the adopted rows sees exactly what it saw before. Adoption is a separate composer decision.
"""
import hashlib
import json
import sys


def sha256(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def main(staging_path, t1_path, t2_path, output):
    staging = json.load(open(staging_path))
    t1, t2 = json.load(open(t1_path)), json.load(open(t2_path))
    new = list(t1['records']) + [r for r in t2['records'] if r['proposal_kind'] == 'NEW_RECORD']
    amendments = []
    for r in t2['records']:
        if r['proposal_kind'] != 'PINNING_AMENDMENT':
            continue
        target = r['title'].split()[0]
        amendments.append({'target_record': target, **r})
    for key in ('proposed_quarantines', 'proposed_pinning_amendments', 'proposed_non_obligations', 'proposal_provenance'):
        if key in staging:
            raise SystemExit(f'{key} already present in the staging catalog')
    candidate = dict(staging)
    candidate['proposed_quarantines'] = new
    candidate['proposed_pinning_amendments'] = amendments
    candidate['proposed_non_obligations'] = t2['not_obligations']
    candidate['proposal_provenance'] = {
        'status': 'PROPOSED_NOT_ADOPTED',
        'adoption_boundary': ('These rows are proposals. They are not part of quarantines, counts or adoption_scope, and they '
                              'change no existing record. Adopting a proposed_quarantines row means moving it into quarantines '
                              'with proposal_status ADOPTED; adopting a pinning amendment means adding its source_binding to '
                              'the target_record. Either is a composer decision. Scientific effect NONE; no CLOSED promotion.'),
        'author': 'session_013tN78b3Q38WdttFyiJ6KVg ("Claude 54"), Anthropic Claude',
        'independence': ('All drafters and verifiers were Anthropic Claude (the author\'s family). Different-family checks: '
                         'tranche 1 by GROK-H04 (main#738/6565322273), tranche 2 offered to GROK-H02 (main#738/6566781875); '
                         'neither result is recorded in this file.'),
        'sources': [
            {'tranche': 1, 'path': 'recovery/session_013tN78b3Q38WdttFyiJ6KVg/qp_c54_quarantine_proposals_20261101/records_verified.json',
             'sha256': sha256(t1_path), 'schema': t1['schema'], 'math_commit': t1['math_cut']},
            {'tranche': 2, 'path': 'recovery/session_013tN78b3Q38WdttFyiJ6KVg/qp_c54_t2_20261102/records_verified_t2.json',
             'sha256': sha256(t2_path), 'schema': t2['schema'], 'math_commit': t2['math_ref'], 'catalog_cut': t2['catalog_cut']},
        ],
        'staging_catalog_sha256': sha256(staging_path),
        'counts': {'proposed_quarantines': len(new), 'proposed_quarantines_tranche1': len(t1['records']),
                   'proposed_quarantines_tranche2': len(new) - len(t1['records']),
                   'proposed_pinning_amendments': len(amendments), 'proposed_non_obligations': len(t2['not_obligations'])},
    }
    with open(output, 'x', encoding='utf-8') as stream:
        stream.write(json.dumps(candidate, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(candidate['proposal_provenance']['counts']))


if __name__ == '__main__':
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
