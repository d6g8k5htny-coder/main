#!/usr/bin/env python3
"""Merge tranche-10 workflow results and apply one mechanical normalization: on a non-OBLIGATION item, claim_held_by (reserved for
OBLIGATIONs) names the records holding the claims the item expounds, so its ids are moved to refers_to and it is emptied. Each
change is logged. Nothing else changes. Usage: python3 -I -B normalize_verdicts_t10.py OUT VERDICTS [VERDICTS...]"""
import json, sys
out, *paths = sys.argv[1:]
merged, log = {'batches': []}, []
for p in paths:
    for b in json.load(open(p))['batches']:
        for vv in b['verdicts'] or []:
            for it in vv['items']:
                if it['classification'] != 'OBLIGATION' and it['claim_held_by']:
                    log.append(f'{vv["object"]}#{it["item_id"]} ({it["classification"]}): claim_held_by {it["claim_held_by"]} moved to refers_to')
                    it['refers_to'] = it['refers_to'] + [h for h in it['claim_held_by'] if h not in it['refers_to']]
                    it['claim_held_by'] = []
        merged['batches'].append(b)
merged['batches'].sort(key=lambda b: b['batch'])
with open(out, 'x') as f:
    json.dump(merged, f, indent=1, ensure_ascii=False)
print('\n'.join(log))
print(len(log), 'item changes;', len(merged['batches']), 'batches')
