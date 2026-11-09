#!/usr/bin/env python3
"""G12 field overlay for Claude 54's Lean rows (tranche-5 quarantine-pending-replay records and the CLOSED candidates of catalog
candidate v2), from Claude01N's G12 census of Math- at dfa9b30a (main cbz/evidence/g12-statement-fuzz-01n/census_math_dfa9b30a_
20261104/). Read-only.

The field rule is Claude01N's proposed B-007 text: `G12: PASS | CORROBORATED | not applicable (<construct>)`. A census PASS or
CORROBORATED is carried as is; an UNSUPPORTED target becomes `not applicable (<construct>)`, where the construct is the census
detail; nothing is upgraded. A row is BOUND only when every Lean file it names has, at the row's own commit, the same SHA-256 as
the census receipt records for that file at dfa9b30a; otherwise it is UNBOUND and gets no G12 value. This is not a CLOSED decision
and changes no status.

Declaration-level rows in main formal/ (the P15 CLOSED candidates) are bound by (source_sha256, declaration) to the census
receipt of main formal/ at their source commit (main_formal_ec6962ff_r4.json, which equals r3 per target).

Usage: python3 -I -B g12_overlay.py MAIN_CLONE MAIN_REF MATH_CLONE T5_RECORDS CANDIDATE_V2 OUT_JSON"""
import collections, hashlib, json, subprocess, sys

main_clone, main_ref, math_clone, t5_p, cand_p, out = sys.argv[1:]
CENSUS = 'cbz/evidence/g12-statement-fuzz-01n/census_math_dfa9b30a_20261104'


def git(repo, *a):
    return subprocess.run(['git', '-C', repo, *a], capture_output=True)


def receipt_for(directory):
    name = directory.replace('/', '_') + '.json'
    r = git(main_clone, 'show', f'{main_ref}:{CENSUS}/{name}')
    return (name, json.loads(r.stdout)) if r.returncode == 0 else (name, None)


def rows():
    t5 = json.load(open(t5_p))
    for r in (t5['records'] if isinstance(t5, dict) else t5):
        yield 'tranche-5 record', r['id'], r
    for i, r in enumerate(json.load(open(cand_p))['proposed_closed_candidates']):
        yield 'v2 CLOSED candidate', r.get('id') or f'closed-candidate-{i + 1:02d}', r


MAIN_FORMAL = {'ec6962ff5fafe04364e094a954896dc1e62941f5': 'cbz/evidence/g12-statement-fuzz-01n/main_formal_ec6962ff_r4.json'}
res, summary = [], collections.Counter()
for kind, rid, r in rows():
    if 'location' not in r and r.get('repository') == 'main' and r.get('path', '').startswith('formal/'):
        e = {'row': rid, 'kind': kind, 'title': r['title'][:160], 'declaration': r['declaration'], 'path': r['path'],
             'source_commit': r['source_commit']}
        rp = MAIN_FORMAL.get(r['source_commit'])
        if not rp:
            e['g12'] = 'NO_CENSUS (no main formal/ receipt at this commit)'
        else:
            rec = json.loads(git(main_clone, 'show', f'{main_ref}:{rp}').stdout)
            module = r['path'][len('formal/'):]
            t = rec['per_target'].get(r['declaration'])
            e['receipt'] = rp
            if rec['module_sha256'].get(module) != r['source_sha256']:
                e['g12'] = 'UNBOUND (module bytes differ from the receipt)'
            elif t is None or t['module'] != module:
                e['g12'] = 'UNBOUND (declaration not a target of the receipt)'
            else:
                st = t['status']
                e['g12'] = st if st in ('PASS', 'CORROBORATED') else f'not applicable ({t.get("detail")})'
                e['detail'] = t.get('detail')
                summary[e['g12'] if not e['g12'].startswith('not applicable') else 'not applicable'] += 1
        res.append(e)
        continue
    locs = r['location'] if isinstance(r['location'], list) else [r['location']]
    entry = {'row': rid, 'kind': kind, 'title': r['title'][:160], 'locations': locs, 'groups': []}
    for loc in locs:
        if not loc.startswith('d6g8k5htny-coder/Math-@'):
            entry['groups'].append({'location': loc, 'g12': 'NO_CENSUS (not a Math- location)'}); continue
        commit, directory = loc.split('@', 1)[1].split(':', 1)
        name, rec = receipt_for(directory.rstrip('/'))
        if rec is None:
            entry['groups'].append({'location': loc, 'g12': f'NO_CENSUS (no receipt {name})'}); continue
        mism = []
        for fname, sha in rec['file_sha256'].items():
            data = git(math_clone, 'show', f'{commit}:{directory.rstrip("/")}/{fname}').stdout
            if hashlib.sha256(data).hexdigest() != sha:
                mism.append(fname)
        g = {'location': loc, 'receipt': f'{CENSUS}/{name}', 'census_files': len(rec['file_sha256']),
             'files_with_different_bytes_at_row_commit': mism}
        if mism:
            g['g12'] = 'UNBOUND (bytes differ from the census cut)'
        else:
            per = {}
            for key, t in sorted(rec['per_target'].items()):
                st = t['status']
                per[key] = st if st in ('PASS', 'CORROBORATED') else (
                    f'not applicable ({t.get("detail", "construct not modelled")})' if st == 'UNSUPPORTED' else st)
            c = collections.Counter(v if not v.startswith('not applicable') else 'not applicable' for v in per.values())
            g.update(g12='BOUND', counts=dict(c), per_declaration=per)
            summary.update(c)
        entry['groups'].append(g)
    res.append(entry)
with open(out, 'x') as f:
    json.dump({'schema': 'claude54.g12-overlay.v1', 'census': f'main@{main_ref}:{CENSUS}', 'field_rule':
               'G12: PASS | CORROBORATED | not applicable (<construct>); UNSUPPORTED is never upgraded; not a CLOSED decision',
               'rows': res, 'bound_counts': dict(summary)}, f, indent=1, ensure_ascii=False)
for e in res:
    if 'groups' not in e:
        print(e['row'], '|', e['declaration'], ':', e['g12'])
        continue
    print(e['row'], '|', '; '.join(f"{g['location'].split(':', 1)[1] if ':' in g['location'] else g['location']}: {g['g12']}"
                                   + (f" {g['counts']}" if 'counts' in g else '') for g in e['groups']))
print('bound totals', dict(summary))
