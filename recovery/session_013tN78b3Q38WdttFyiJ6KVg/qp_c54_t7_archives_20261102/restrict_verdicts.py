#!/usr/bin/env python3
"""Restrict verified tranche-7 items to the members that remain after resolve_t6.py. An item's explicit member names that are no
longer members are dropped; a prefix spec ("dir/") is kept if it still matches a member; an item left with no member is dropped.
Classifications, reasons and problems_fixed are unchanged. Usage: python3 -I -B restrict_verdicts.py FACTS_DIR VERDICTS OUT"""
import glob, json, os, sys
facts_dir, verdicts_path, out = sys.argv[1:4]
members = {}
for path in glob.glob(os.path.join(facts_dir, 'abatch*.json')):
    for o in json.load(open(path))['objects']:
        members[o['object']] = {m['member'] for m in o['members']}
v = json.load(open(verdicts_path))
log = []
for b in v['batches']:
    for vv in b['verdicts'] or []:
        ms = members[vv['object']]
        items = []
        for it in vv['items']:
            files = [s for s in it['files'] if (s in ms) or (s.endswith('/') and any(m.startswith(s) for m in ms))]
            dropped = len(it['files']) - len(files)
            if not files:
                log.append(f'{vv["object"]}#{it["item_id"]}: dropped (every member resolved as a copy)')
                continue
            if dropped:
                log.append(f'{vv["object"]}#{it["item_id"]}: {dropped} member names resolved as copies')
            items.append(dict(it, files=files))
        vv['items'] = items
with open(out, 'x') as f:
    json.dump(v, f, indent=1, ensure_ascii=False)
print('\n'.join(log))
