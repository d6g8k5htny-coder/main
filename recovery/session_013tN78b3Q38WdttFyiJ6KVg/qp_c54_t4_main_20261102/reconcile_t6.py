#!/usr/bin/env python3
"""Mechanical cross-tranche reconciliation: find tranche-6 (federation) records whose bound files are byte-identical (same Git
blob) to files bound by tranche-4 (main) records, and state which record is primary for the shared bytes.

Usage: python3 -I -B reconcile_t6.py RECORDS_T4 RECORDS_T6 OUTPUT

Rule: main is inside the catalog's scope and the federation repositories are not, so for every shared blob main's copy is primary:
the main obligation record when one binds it (primary_main_records), otherwise the main accounting item that binds it (named per
file). A tranche-6 record keeps standing only for files (and statements) whose bytes main does not hold. Nothing is removed:
the output is an overlay that catalog candidate v2 attaches to both rows as `reconciliation`.
"""
import json, sys

t4_path, t6_path, out = sys.argv[1:4]
t4, t6 = json.load(open(t4_path)), json.load(open(t6_path))
main_blobs = {}
for r in t4['records']:
    for f in r['source_binding']['files']:
        main_blobs.setdefault(f['git_blob'], []).append({'record': r['id'], 'path': f['path']})
for r in t4.get('other_items', []):
    for f in r['source_binding']['files']:
        main_blobs.setdefault(f['git_blob'], []).append({'item': f'{r["object"]}#{r["item_id"]}', 'classification': r['classification'],
                                                         'path': f['path']})
rows = []
for r in t6['records']:
    b = r['source_binding']
    shared, only = [], []
    for f in b['files']:
        hit = main_blobs.get(f['git_blob'])
        (shared if hit else only).append({'path': f['path'], 'git_blob': f['git_blob'], **({'main': hit} if hit else {})})
    if not shared:
        continue
    only_paths = {f['path'] for f in only}
    stmts_only = [f'{s["file"]}:{s["line_start"]}-{s["line_end"]}' for s in r['source_statements'] if s['file'] in only_paths]
    primaries = sorted({m['record'] for f in shared for m in f['main'] if 'record' in m})
    rows.append({
        't6_record': r['id'], 't6_repository': b['repository'], 't6_commit': b['commit'],
        'primary_main_records': primaries, 'shared_files': shared, 'federation_only_files': only,
        'federation_only_statements': stmts_only,
        'disposition': ('T6 record SUPERSEDED_BY_MAIN for its shared bytes; it stays PROPOSED only for its federation-only files '
                        'and statements' if only else 'T6 record is a byte-identical replica of main: SUPERSEDED_BY_MAIN in full'),
    })
with open(out, 'x') as f:
    json.dump({'schema': 'claude54.cross-tranche-reconciliation.v1', 'sources': {'t4': t4_path.split('/')[-1], 't6': t6_path.split('/')[-1]},
               'rule': 'main is primary for every shared blob; federation rows keep standing only for bytes main does not hold',
               'reconciliations': rows}, f, indent=1, ensure_ascii=False)
    f.write('\n')
for x in rows:
    print(x['t6_record'], '->', x['primary_main_records'], f'shared {len(x["shared_files"])}', f'federation-only {len(x["federation_only_files"])}',
          'statements-only', x['federation_only_statements'])
