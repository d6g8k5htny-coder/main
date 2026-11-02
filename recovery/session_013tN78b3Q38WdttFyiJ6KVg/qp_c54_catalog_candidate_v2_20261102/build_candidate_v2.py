#!/usr/bin/env python3
"""Build candidate v2 of cbz/quarantine/catalog.json: every QP-C54 tranche beside, not inside, the adopted rows.

Usage: python3 -I -B build_candidate_v2.py STAGING_CATALOG RECOVERY_DIR OUTPUT

RECOVERY_DIR is recovery/session_013tN78b3Q38WdttFyiJ6KVg. Every source in SOURCES is required; a missing one is an error. The
staging keys and values are kept exactly (indent 2, ensure_ascii False, trailing newline), so the staging bytes up to the closing
brace are unchanged. Appended keys:

  proposed_quarantines            NEW_RECORD rows of tranches 1-5, 7 and 8 for main and Math-, unchanged except for the overlays
  proposed_quarantines_federation tranche-6 rows (and any federation tranche-7 row): repositories outside the catalog's
                                  mathematical scope
  proposed_pinning_amendments     PINNING_AMENDMENT rows of any tranche, each with target_record (tranche 2) or target_records
  proposed_byte_pins              the mechanical pinning sweep (41 pins for 14 adopted records)
  proposed_closed_candidates      tranche-5 groups whose own receipts bind the current bytes, and the 28 P15 rows landed by #740
  held_by_other_lane              tranche-5 groups another lane holds (no proposal)
  a3_advice                       the final A3 verdicts for QP-C54-T2-008, -016 and -029, the boundary and the executed-replay
                                  note; the audits themselves are referenced by source path and SHA-256, not copied
  accounting_dispositions         every object or item classified as not needing a record (REVIEW_OF, COPY_OR_EXPOSITION_OF,
                                  NOT_OBLIGATION, tranche-4 mechanical resolutions and tranche-7 MECHANICAL_COPY archive members),
                                  with its cited reason, so every censused object has a stated outcome
  proposal_provenance             sources, digests, counts, the adoption boundary and the independence record

Overlays (the retained tranche files are never edited):
- Errata (qp_c54_errata_20261102/errata.json). Each item sets one field of one row, or of an explicit list of rows; every touched
  row records the item id in errata_applied. A target that matches no row, a section target that matches more than one row, and
  an a3 target absent from the audits are errors. Routing (new record or amendment) follows proposal_kind after the errata.
- Reconciliation (qp_c54_t4_main_20261102/reconciliation_t6.json). Where a federation row binds Git blobs that a main row also
  binds, main is primary: the federation row gets `reconciliation` and each primary main row gets `federation_replicas`.

Nothing is adopted: quarantines, counts, status and adoption_scope are untouched.
"""
import hashlib, json, os, sys

SOURCES = {
    't1': 'qp_c54_quarantine_proposals_20261101/records_verified.json',
    't2': 'qp_c54_t2_20261102/records_verified_t2.json',
    't3': 'qp_c54_t3_20261102/records_verified_t3.json',
    't4': 'qp_c54_t4_main_20261102/records_verified_t4.json',
    't5': 'qp_c54_t5_lean_20261102/records_verified_t5.json',
    't6': 'qp_c54_t6_federation_20261102/records_verified_t6.json',
    't7': 'qp_c54_t7_archives_20261102/records_verified_t7.json',
    't8': 'qp_c54_t8_math_residual_20261102/records_verified_t8.json',
    'pins': 'pinning_sweep_20261102/pins.json',
    'p15': 'p15_closed_candidates_20261102/p15_closed_candidates.json',
    'a3': 'qp_c54_t3_20261102/a3_audits.json',
    'errata': 'qp_c54_errata_20261102/errata.json',
    'recon': 'qp_c54_t4_main_20261102/reconciliation_t6.json',
    't4scope': 'qp_c54_t4_main_20261102/scope_addendum.json',
}
NEW_KEYS = ['proposed_quarantines', 'proposed_quarantines_federation', 'proposed_pinning_amendments', 'proposed_byte_pins',
            'proposed_closed_candidates', 'held_by_other_lane', 'a3_advice', 'accounting_dispositions', 'proposal_provenance']
IN_SCOPE = ('d6g8k5htny-coder/main', 'd6g8k5htny-coder/Math-')
INDEPENDENCE = (
    'All drafters and verifiers were Anthropic Claude. Different-family checks on main#738, by Grok Bot agents: candidate v1 by '
    'a15 and a16, PASS MATCH (6567327690); tranche 2 by a12 and a13 (6569098238, 6569136696), tranche 3 by a3 and a16 (6568629873, '
    '6568685335) and tranche 6 by a13 and a5 (6568869813, 6568931499), each PASS_WITH_NITS, MATCH, with the nits applied as errata '
    'or README fixes except T3 nit 4 (kept, as the reviewer judged acceptable) and the ** markup in T6\'s quoted reasons (not '
    'taken); the tranche-5 retip by a15 and a8, PASS (6568128371, 6568149184); the executed replay of QP-C54-T2-029 by a16 and a3 '
    '(6568215011, 6568411087). Same-family adversarial review of this candidate: workflow wf_1e2e73e5-c9f, four lenses each with an '
    'independent refuter; its reproduced findings are applied as E-R-* errata, tooling fixes and README corrections. Not yet checked '
    'by another family: tranche 1, tranche 5\'s classification, tranches 4, 7 and 8, the byte-pin sweep, the 28 P15 closed '
    'candidates, the A3 verdicts for QP-C54-T2-008 and -016 (the tranche-2 check left A3 undecided), the reconciliation, the scope '
    'addendum, the E-R-* errata and this candidate.')


def sha256(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def main(staging_path, rec_dir, output):
    staging = json.load(open(staging_path))
    paths = {k: os.path.join(rec_dir, v) for k, v in SOURCES.items()}
    missing = [SOURCES[k] for k, p in paths.items() if not os.path.exists(p)]
    if missing:
        raise SystemExit(f'missing sources: {missing}')
    src = {k: json.load(open(p)) for k, p in paths.items()}
    for key in NEW_KEYS:
        if key in staging:
            raise SystemExit(f'{key} already present in the staging catalog')
    a3_final = src['a3']['final']
    a3_notes = {}
    for item in src['errata']['items']:
        t = item['target']
        if 'a3' in t:
            if t['a3'] not in a3_final:
                raise SystemExit(f'errata {item["id"]}: a3 target {t["a3"]} is not an audited record')
            a3_notes.setdefault(t['a3'], []).append({'errata': item['id'], 'field': item['field'], 'note': item['value']})
            continue
        tranche = src[t['tranche']]
        if 'section' in t:
            rows = [r for r in tranche[t['section']] if r.get('object') == t['object'] or
                    (r.get('location') and r['location'][0].split(':', 1)[1] == t['object'])]
            if len(rows) != 1:
                raise SystemExit(f'errata {item["id"]}: section target matches {len(rows)} rows, not 1')
        else:
            ids = t.get('records') or [t['record']]
            rows = [r for r in tranche['records'] if r['id'] in ids]
            if len(rows) != len(ids):
                raise SystemExit(f'errata {item["id"]}: target not found')
        for r in rows:
            obj, _, leaf = item['field'].rpartition('.')
            holder = r
            for part in (obj.split('.') if obj else []):
                holder = holder[part]
            holder[leaf] = item['value']
            r.setdefault('errata_applied', []).append(item['id'])
    new, fed, amend, closed, held, acct = [], [], [], [], [], []

    def route(r):
        kind = r.get('proposal_kind', 'NEW_RECORD')
        if kind == 'PINNING_AMENDMENT':
            if not (r.get('target_records') or r.get('target_record')):
                raise SystemExit(f'{r["id"]}: amendment without a target')
            amend.append(r)
        elif kind == 'NEW_RECORD':
            (new if r['source_binding']['repository'] in IN_SCOPE else fed).append(r)
        else:
            raise SystemExit(f'{r["id"]}: unknown proposal_kind {kind}')

    for r in src['t1']['records']:
        route(r)
    for r in src['t2']['records']:
        route(dict(r, target_record=r['title'].split()[0]) if r['proposal_kind'] == 'PINNING_AMENDMENT' and 'target_records' not in r else r)
    acct += [dict(x, tranche='t2', classification='NOT_OBLIGATION') for x in src['t2']['not_obligations']]
    for r in src['t3']['records']:
        route(r)
    acct += [dict(x, tranche='t3', classification='REVIEW_OF') for x in src['t3']['reviews']]
    acct += [dict(x, tranche='t3', classification='NOT_OBLIGATION') for x in src['t3']['not_obligations']]
    for r in src['t4']['records']:
        route(r)
    acct += [dict(x, tranche='t4') for x in src['t4']['other_items']]
    acct += [dict(x, tranche='t4', classification=x['resolution']) for x in src['t4']['mechanically_resolved_objects']]
    acct += [dict(x, tranche='t4') for x in src['t4scope']['rows']]
    for r in src['t5']['records']:
        route(r)
    closed += [dict(x, tranche='t5', candidate_kind='OWN_RECEIPT_GROUP') for x in src['t5']['closed_candidates']]
    held += [dict(x, tranche='t5') for x in src['t5']['held_by_other_lane']]
    acct += [dict(x, tranche='t5', classification='NOT_OBLIGATION') for x in src['t5']['not_obligations']]
    fed += src['t6']['records']
    acct += [dict(x, tranche='t6') for x in src['t6']['other_items']]
    for r in src['t7']['records']:
        route(r)
    acct += [dict(x, tranche='t7') for x in src['t7']['other_items']]
    acct += [dict(x, tranche='t7') for x in src['t7']['mechanical_copies']]
    for r in src['t8']['records']:
        route(r)
    acct += [dict(x, tranche='t8') for x in src['t8']['other_items']]
    by_id = {r['id']: r for r in new + amend + fed}
    for x in src['recon']['reconciliations']:
        if x['t6_record'] not in by_id or any(m not in by_id for m in x['primary_main_records']):
            raise SystemExit(f'reconciliation {x["t6_record"]}: record not found')
        by_id[x['t6_record']]['reconciliation'] = {k: x[k] for k in ('primary_main_records', 'disposition', 'shared_files',
                                                                      'federation_only_files', 'federation_only_statements')}
        for m in x['primary_main_records']:
            by_id[m].setdefault('federation_replicas', []).append(
                {'record': x['t6_record'], 'repository': x['t6_repository'], 'commit': x['t6_commit'],
                 'shared_paths': sorted(f['path'] for f in x['shared_files'] if any(e.get('record') == m for e in f['main']))})
    closed += [dict(x, tranche='p15', candidate_kind='HOSTED_GATE_DECLARATION') for x in src['p15']['rows']]
    candidate = dict(staging)
    candidate['proposed_quarantines'] = new
    candidate['proposed_quarantines_federation'] = fed
    candidate['proposed_pinning_amendments'] = amend
    candidate['proposed_byte_pins'] = src['pins']['pins']
    candidate['proposed_closed_candidates'] = closed
    candidate['held_by_other_lane'] = held
    candidate['a3_advice'] = {'final': a3_final, 'boundary': src['a3']['boundary'], 'notes': a3_notes,
                              'audits_by_reference': {'source': SOURCES['a3'], 'source_sha256': sha256(paths['a3'])}}
    candidate['accounting_dispositions'] = acct
    candidate['proposal_provenance'] = {
        'status': 'PROPOSED_NOT_ADOPTED',
        'adoption_boundary': ('These rows are proposals. They are not part of quarantines, counts or adoption_scope, and they change '
                              'no existing record. Adopting a proposed_quarantines row means moving it into quarantines with '
                              'proposal_status ADOPTED; a pinning amendment or byte pin adds its binding to the target record; a '
                              'closed candidate needs the kernel-index lane\'s receipt before it can be CLOSED; federation rows '
                              'need the catalog\'s mathematical scope to be widened first. claim_overlaps fields name existing '
                              'records that already hold part of a row\'s claim; the composer narrows such a row on adoption. '
                              'Every choice is a composer decision. Scientific effect NONE.'),
        'author': 'session_013tN78b3Q38WdttFyiJ6KVg ("Claude 54"), Anthropic Claude',
        'independence': INDEPENDENCE,
        'sources': [{'key': k, 'path': SOURCES[k], 'sha256': sha256(paths[k])} for k in SOURCES],
        'missing_sources': [],
        'staging_catalog_sha256': sha256(staging_path),
        'counts': {'proposed_quarantines': len(new), 'proposed_quarantines_federation': len(fed),
                   'proposed_pinning_amendments': len(amend), 'proposed_byte_pins': len(src['pins']['pins']),
                   'proposed_closed_candidates': len(closed), 'held_by_other_lane': len(held),
                   'accounting_dispositions': len(acct)},
    }
    with open(output, 'x', encoding='utf-8') as stream:
        stream.write(json.dumps(candidate, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(candidate['proposal_provenance']['counts']))


if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
