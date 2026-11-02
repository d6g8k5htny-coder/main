#!/usr/bin/env python3
"""Build candidate v2 of cbz/quarantine/catalog.json: every QP-C54 tranche beside, not inside, the adopted rows.

Usage: python3 -I -B build_candidate_v2.py STAGING_CATALOG RECOVERY_DIR OUTPUT

RECOVERY_DIR is recovery/session_013tN78b3Q38WdttFyiJ6KVg. The staging keys and values are kept exactly (indent 2, ensure_ascii
False, trailing newline), so the staging bytes up to the closing brace are unchanged. Appended keys:

  proposed_quarantines            T1, T2 NEW_RECORD, T3, T4 NEW_RECORD and T5 records (main and Math-), unchanged
  proposed_quarantines_federation T6 records, for repositories outside the catalog's current scope
  proposed_pinning_amendments     T2 and T4 PINNING_AMENDMENT records, each with target_record(s)
  proposed_byte_pins              the mechanical pinning sweep (41 pins for 14 adopted records)
  proposed_closed_candidates      T5 groups whose own receipts bind the current bytes, and the 28 P15 rows landed by main#740
  held_by_other_lane              T5 groups another lane holds (no proposal)
  a3_advice                       read-only interval-arithmetic audits of QP-C54-T2-008, -016 and -029
  accounting_dispositions         every object or item classified as not needing a record (REVIEW_OF, COPY_OR_EXPOSITION_OF,
                                  NOT_OBLIGATION), with its cited reason, so every censused object has a stated outcome
  proposal_provenance             sources, digests, counts and the adoption boundary

Errata (qp_c54_errata_20261102/errata.json): corrections accepted from different-family checks. Each item sets one field of
one row (or of a listed set of rows); every touched row records the item id in errata_applied. A missing target is an error.

Reconciliation (qp_c54_t4_main_20261102/reconciliation_t6.json): where a federation row binds Git blobs that a main row also binds,
main is primary. The federation row gets `reconciliation` (primary main records, shared and federation-only files) and each primary
main row gets `federation_replicas`; no row is removed.

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
    'pins': 'pinning_sweep_20261102/pins.json',
    'p15': 'p15_closed_candidates_20261102/p15_closed_candidates.json',
    'a3': 'qp_c54_t3_20261102/a3_audits.json',
    'errata': 'qp_c54_errata_20261102/errata.json',
    'recon': 'qp_c54_t4_main_20261102/reconciliation_t6.json',
    't4scope': 'qp_c54_t4_main_20261102/scope_addendum.json',
}
NEW_KEYS = ['proposed_quarantines', 'proposed_quarantines_federation', 'proposed_pinning_amendments', 'proposed_byte_pins',
            'proposed_closed_candidates', 'held_by_other_lane', 'a3_advice', 'accounting_dispositions', 'proposal_provenance']


def sha256(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def main(staging_path, rec_dir, output):
    staging = json.load(open(staging_path))
    paths = {k: os.path.join(rec_dir, v) for k, v in SOURCES.items()}
    present = {k: os.path.exists(p) for k, p in paths.items()}
    src = {k: json.load(open(p)) for k, p in paths.items() if present[k]}
    for key in NEW_KEYS:
        if key in staging:
            raise SystemExit(f'{key} already present in the staging catalog')
    a3_notes = {}
    for item in (src['errata']['items'] if 'errata' in src else []):
        t = item['target']
        if 'a3' in t:
            a3_notes.setdefault(t['a3'], []).append({'errata': item['id'], 'note': item['value']})
            continue
        tranche = src[t['tranche']]
        if 'section' in t:
            rows = [r for r in tranche[t['section']] if r.get('object') == t['object'] or
                    (r.get('location') and r['location'][0].split(':', 1)[1] == t['object'])]
        else:
            ids = t.get('records') or [t['record']]
            rows = [r for r in tranche['records'] if r['id'] in ids]
            if len(rows) != len(ids):
                raise SystemExit(f'errata {item["id"]}: target not found')
        if not rows:
            raise SystemExit(f'errata {item["id"]}: target not found')
        for r in rows:
            obj, _, leaf = item['field'].rpartition('.')
            holder = r
            for part in (obj.split('.') if obj else []):
                holder = holder[part]
            holder[leaf] = item['value']
            r.setdefault('errata_applied', []).append(item['id'])
    new, fed, amend, closed, held, acct = [], [], [], [], [], []
    new += src['t1']['records']
    for r in src['t2']['records']:
        (amend if r['proposal_kind'] == 'PINNING_AMENDMENT' else new).append(
            dict(r, target_record=r['title'].split()[0]) if r['proposal_kind'] == 'PINNING_AMENDMENT' else r)
    acct += [dict(x, tranche='t2', classification='NOT_OBLIGATION') for x in src['t2']['not_obligations']]
    new += src['t3']['records']
    acct += [dict(x, tranche='t3', classification='REVIEW_OF') for x in src['t3']['reviews']]
    acct += [dict(x, tranche='t3', classification='NOT_OBLIGATION') for x in src['t3']['not_obligations']]
    if 't4' in src:
        for r in src['t4']['records']:
            (amend if r['proposal_kind'] == 'PINNING_AMENDMENT' else new).append(r)
        acct += [dict(x, tranche='t4') for x in src['t4']['other_items']]
        acct += [dict(x, tranche='t4', classification=x['resolution']) for x in src['t4']['mechanically_resolved_objects']]
    if 't4scope' in src:
        acct += [dict(x, tranche='t4') for x in src['t4scope']['rows']]
    new += src['t5']['records']
    closed += [dict(x, tranche='t5', candidate_kind='OWN_RECEIPT_GROUP') for x in src['t5']['closed_candidates']]
    held += [dict(x, tranche='t5') for x in src['t5']['held_by_other_lane']]
    acct += [dict(x, tranche='t5', classification='NOT_OBLIGATION') for x in src['t5']['not_obligations']]
    fed += src['t6']['records']
    if 'recon' in src:
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
    acct += [dict(x, tranche='t6') for x in src['t6']['other_items']]
    closed += [dict(x, tranche='p15', candidate_kind='HOSTED_GATE_DECLARATION') for x in src['p15']['rows']]
    candidate = dict(staging)
    candidate['proposed_quarantines'] = new
    candidate['proposed_quarantines_federation'] = fed
    candidate['proposed_pinning_amendments'] = amend
    candidate['proposed_byte_pins'] = src['pins']['pins']
    candidate['proposed_closed_candidates'] = closed
    candidate['held_by_other_lane'] = held
    candidate['a3_advice'] = {'final': src['a3']['final'], 'boundary': src['a3']['boundary'], 'notes': a3_notes,
                              'source': SOURCES['a3'], 'source_sha256': sha256(paths['a3'])}
    candidate['accounting_dispositions'] = acct
    candidate['proposal_provenance'] = {
        'status': 'PROPOSED_NOT_ADOPTED',
        'adoption_boundary': ('These rows are proposals. They are not part of quarantines, counts or adoption_scope, and they change '
                              'no existing record. Adopting a proposed_quarantines row means moving it into quarantines with '
                              'proposal_status ADOPTED; a pinning amendment or byte pin adds its binding to the target record; a '
                              'closed candidate needs the kernel-index lane\'s receipt before it can be CLOSED; federation rows '
                              'need the catalog\'s scope to be widened first. Every choice is a composer decision. Scientific '
                              'effect NONE.'),
        'author': 'session_013tN78b3Q38WdttFyiJ6KVg ("Claude 54"), Anthropic Claude',
        'independence': ('All drafters and verifiers were Anthropic Claude. Different-family checks on main#738, by Grok Bot '
                         'agents: candidate v1 by a15 and a16, PASS MATCH (6567327690); tranche 2 by a12 and a13 (6569098238, '
                         '6569136696), tranche 3 by a3 and a16 (6568629873, 6568685335) and tranche 6 by a13 and a5 (6568869813, '
                         '6568931499), each PASS_WITH_NITS, MATCH, with the nits applied as errata or README fixes; the tranche-5 '
                         'retip by a15 and a8, PASS (6568128371, 6568149184); the executed replay of QP-C54-T2-029 by a16 and a3 '
                         '(6568215011, 6568411087). Not yet checked by another family: tranche 1, tranche 5\'s classification, '
                         'tranche 4, the byte-pin sweep and this v2 candidate.'),
        'sources': [{'key': k, 'path': SOURCES[k], 'sha256': sha256(paths[k])} for k in SOURCES if present[k]],
        'missing_sources': [SOURCES[k] for k in SOURCES if not present[k]],
        'staging_catalog_sha256': sha256(staging_path),
        'counts': {'proposed_quarantines': len(new), 'proposed_quarantines_federation': len(fed),
                   'proposed_pinning_amendments': len(amend), 'proposed_byte_pins': len(src['pins']['pins']),
                   'proposed_closed_candidates': len(closed), 'held_by_other_lane': len(held),
                   'accounting_dispositions': len(acct)},
    }
    with open(output, 'x', encoding='utf-8') as stream:
        stream.write(json.dumps(candidate, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(candidate['proposal_provenance']['counts']), 'missing:', candidate['proposal_provenance']['missing_sources'])


if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
