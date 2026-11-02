#!/usr/bin/env python3
"""Restrict verified tranche-9 items to the files that remain after resolve_main_copies.py. An item's explicit file names that are
no longer object files are dropped; a prefix spec ("dir/") is kept if it still matches an object file; an item left with no file
is dropped and logged with its classification. On a non-OBLIGATION item, claim_held_by (reserved for OBLIGATIONs) names the
records that hold the claims the item expounds, so its ids are moved to refers_to and it is emptied, also logged. Classifications,
reasons and problems_fixed are unchanged.
Usage: python3 -I -B restrict_verdicts_t9.py FACTS_DIR OUT VERDICTS [VERDICTS...]   (OUT holds every batch of every input)"""
import glob, json, os, sys
facts_dir, out, *paths = sys.argv[1:]
files = {}
for path in glob.glob(os.path.join(facts_dir, 'pbatch*.json')):
    for o in json.load(open(path))['objects']:
        files[o['object']] = {f['path'] for f in o['files']}
merged, log = {'batches': []}, []
for p in paths:
    for b in json.load(open(p))['batches']:
        for vv in b['verdicts'] or []:
            fs = files[vv['object']]
            items = []
            for it in vv['items']:
                keep = [s for s in it['files'] if (s in fs) or (s.endswith('/') and any(m.startswith(s) for m in fs))]
                dropped = len(it['files']) - len(keep)
                if not keep:
                    log.append(f'{vv["object"]}#{it["item_id"]} ({it["classification"]}): dropped (every file is a main copy)')
                    continue
                if dropped:
                    log.append(f'{vv["object"]}#{it["item_id"]} ({it["classification"]}): {dropped} file names resolved as main copies')
                it = dict(it, files=keep)
                if it['classification'] != 'OBLIGATION' and it['claim_held_by']:
                    log.append(f'{vv["object"]}#{it["item_id"]} ({it["classification"]}): claim_held_by {it["claim_held_by"]} moved to refers_to')
                    it['refers_to'] = it['refers_to'] + [h for h in it['claim_held_by'] if h not in it['refers_to']]
                    it['claim_held_by'] = []
                items.append(it)
            vv['items'] = items
        merged['batches'].append(b)
with open(out, 'x') as f:
    json.dump(merged, f, indent=1, ensure_ascii=False)
print('\n'.join(log))
print(len(log), 'item changes')
