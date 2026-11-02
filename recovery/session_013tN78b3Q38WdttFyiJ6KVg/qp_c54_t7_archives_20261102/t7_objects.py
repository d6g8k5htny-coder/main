#!/usr/bin/env python3
"""Turn archive_inventory.json into tranche-7 objects and batches. An object is one archive, or for large archives one archive
member directory. Each object lists its distinct non-copy members (first occurrence of each Git blob across all archives) to
classify, and, separately, its members that are copies (of a main or Math- tree file, or of a member listed in another object).
Usage: python3 -I -B t7_objects.py INVENTORY OUT_DIR"""
import json, os, sys
inv_p, out_dir = sys.argv[1:3]
inv = json.load(open(inv_p))
GROUPS = [  # (batch, archive path suffix, how to split)
    (1, 'Main188_landed_formal_36358139870.zip', None), (1, 'Math96_landed_formal_36359933550.zip', None),
    (1, 'side24-main-3b39/verified-execution.zip', None), (1, 'completion/evidence.zip', None),
    (2, 'Math96_landed_downstream_36359933550.zip', None),
    (3, 'raw-evidence.zip', None),
    (4, '38ac44c585feb2f9a480d8e54ad1a7fe56ad35aff19e2eab929861ea81974644/archive.zip', None),
    (4, '7c583e28878d2bf2c94bf37325820bd5eb1e7f3f557e8a78dd27881a27f6548c/archive.zip', None),
    (4, 'qualified-observation-5596.json.gz', None),
    (5, 'trial-portable-main-fixes.tgz', 'top'),
]
first = {}
batches = {}
for batch, suffix, split in GROUPS:
    a = next(x for x in inv['archives'] if x['path'].endswith(suffix))
    flat = []
    for m in a['members']:
        if 'nested' in m:
            flat.append(dict(m, member_spec=[m['member']], is_archive=True))
            flat += [dict(n, member_spec=[m['member'], n['member']]) for n in m['nested']]
        else:
            flat.append(dict(m, member_spec=[m['member']]))
    objs = {}
    for m in flat:
        name = '!'.join(m['member_spec'])
        key = a['path'] if split is None else a['path'] + '!' + (m['member_spec'][0].split('/')[0] if '/' in m['member_spec'][0] else '(root)')
        o = objs.setdefault(key, {'object': key, 'repository': a['repository'], 'commit': a['commit'], 'archive': a['path'],
                                  'archive_git_blob': a['git_blob'], 'archive_sha256': a['sha256'], 'members': [], 'copies': []})
        rec = {'member': name, 'member_spec': m['member_spec'], 'bytes': m['bytes'], 'sha256': m['sha256'], 'git_blob': m['git_blob']}
        if m.get('is_archive'):
            o['copies'].append(dict(rec, copy_of='nested archive; its members are listed individually'))
        elif m['copy_of']:
            o['copies'].append(dict(rec, copy_of=m['copy_of'], copy_disposed_location=m['copy_disposed_location']))
        elif m['git_blob'] in first:
            o['copies'].append(dict(rec, copy_of=f'archive member {first[m["git_blob"]]}'))
        else:
            first[m['git_blob']] = f'{a["path"]}!{name}'
            o['members'].append(rec)
    batches.setdefault(batch, []).extend(objs.values())
os.makedirs(out_dir, exist_ok=True)
for b, objs in sorted(batches.items()):
    with open(os.path.join(out_dir, f'abatch{b}.json'), 'x') as f:
        json.dump({'batch': b, 'objects': objs}, f, indent=1)
    print(b, len(objs), 'objects', sum(len(o['members']) for o in objs), 'to classify', sum(len(o['copies']) for o in objs), 'copies')
