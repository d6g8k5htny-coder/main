import json, glob, collections, re, sys, os
S = sys.argv[1]; R = f'{S}/recovery'
gaps = json.load(open(f'{R}/gaps.json')); gid = {g['id']: g for g in gaps}
OWN = {}  # dedicated group owner per gap (non live/artifact agents take precedence)
groups = json.load(open(f'{R}/groups.json'))
for g in groups:
    for x in re.findall(r'G\d+', g['gaps']):
        OWN.setdefault(x, []).append(g['key'])
res, frags, verd = {}, {}, {}
for f in sorted(glob.glob(f'{R}/results/*.json')):
    k = os.path.basename(f)[:-5]; d = json.load(open(f)); res[k] = d
    for fr in d.get('fragments', []): frags[(k, fr['recovery_id'])] = fr
for f in sorted(glob.glob(f'{R}/verify/*.json')):
    k = os.path.basename(f)[:-5]; d = json.load(open(f))
    for v in d.get('verdicts', []): verd[(k, v['recovery_id'])] = v
PRIMARY_LAST = ('a8-artifacts-drive',)  # lead/live-Drive agent: secondary unless sole assessor
rows = {}
for k, d in res.items():
    for a in d.get('gap_assessments', []):
        for g in re.findall(r'G\d+', str(a.get('gap', ''))):
            rows.setdefault(g, []).append((k, a))
OVERRIDES = {'G98': ('G','a3 verifier: H to G; accepted r->0 carriers already on GitHub; H3_BAND_FLOOR shares the tail-conversion defect'), 'G136': ('C','a5 verifier: adjudication is author-side critique; independent proofread pending'), 'G14': ('C','a1 verifier: J to C'), 'G46': ('D','a6 verifier: superseded post-snapshot by Math-#125/#127/#129 reviewed lower bound'), 'G60': ('C','a7 verifier: d=2 and d=3 parts are C, not G'), 'G47': ('C','a7 verifier: overall C; Table 4.1 sub-item I (referenced, unlocated)')}
matrix = []
for g in gaps:
    i = g['id']; ass = rows.get(i, [])
    prim = [x for x in ass if x[0] not in PRIMARY_LAST] or ass
    k, a = prim[0] if prim else (None, None)
    fids = (a or {}).get('fragment_ids', []) or []
    fr = []
    for fid in fids:
        key = (k, fid); f = frags.get(key); v = verd.get(key)
        if f: fr.append(dict(id=fid, conf=f.get('confidence'), conf_after=(v or {}).get('confidence_after_check'), verdict=(v or {}).get('verdict', 'UNVERIFIED' if f.get('confidence') in ('EXACT CARRIER','BYTE-IDENTICAL COPY','VERIFIED TRANSCRIPTION','STRONG HISTORICAL REFERENCE') else 'NOT_IN_VERIFY_SCOPE'), class_after=(re.match(r'\s*([A-J])', str((v or {}).get('classification_after_check') or '')) or [None,None])[1], class_note=str((v or {}).get('classification_after_check') or '')[:200], locator=f.get('locator'), sha256=f.get('sha256',''), relation=f.get('relation_to_current_gap','')[:300]))
    cls = (a or {}).get('classification', 'NOT_ASSESSED')
    va = [x['class_after'] for x in fr if x.get('class_after')]
    matrix.append(dict(gap=i, name=g['name'], cluster=g['cluster'], status_2026_09_28=g['current_status'], assessed_by=k, classification=cls,
        classification_after_verify=(OVERRIDES[i][0] if i in OVERRIDES else cls), override_note=(OVERRIDES[i][1] if i in OVERRIDES else ''), priority=(a or {}).get('priority'),
        other_assessments=[(x[0], x[1].get('classification')) for x in ass if x[0] != k],
        best_evidence=(a or {}).get('best_evidence', '')[:900], missing_piece=(a or {}).get('missing_piece', '')[:500],
        next_action=(a or {}).get('recommended_next_action', '')[:500], fragments=fr))
json.dump(matrix, open(f'{R}/out/matrix.json', 'w'), indent=1, ensure_ascii=False)
c = collections.Counter(m['classification_after_verify'] for m in matrix)
vc = collections.Counter(x['verdict'] for m in matrix for x in m['fragments'])
print('classes', dict(sorted(c.items()))); print('fragment verdicts in matrix', dict(vc))
print('verify files', sorted(os.path.basename(f) for f in glob.glob(f'{R}/verify/*.json')))
