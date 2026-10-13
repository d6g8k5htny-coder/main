"""Second, independent verifier of the update_register7 diff. Own parsers; shares nothing with update_register7.py beyond the two
packet checkers' loaders. Prints FAIL lines; writes verify7b_findings.json."""
import importlib.util, json, re, sys
from pathlib import Path
S = Path('<SCRATCH>')
REPO = S/'repos/math'; STMT = REPO/'specifications/lean_statements_20261010'; PROOF = REPO/'specifications/lean_statement_proofs_20261011'
HERE = S/'lvs/register_update7/verifier2_register'
PP = 'd6g8k5htny-coder/Math- specifications/lean_statement_proofs_20261011'; SP = 'specifications/lean_statements_20261010'
H = json.load(open(HERE/'REGISTER_HEAD.json')); W = json.load(open(HERE/'REGISTER_WORKTREE.json'))
hr = {c['claim_id']: c for c in H['claims']}; wr = {c['claim_id']: c for c in W['claims']}
TRIAGE = json.load(open(S/'lvs/register_update7/triage_by_row.json'))
F = []
def fail(row, msg): F.append(f'[{row}] {msg}'); print('FAIL', row, msg[:400])
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m
scheck = load('stmt_check', STMT/'check.py')
sources, _ = scheck.load_sources(STMT, None, W)

# ---- 1. independent inventory of the three new modules from the Lean text (column-0 theorem + preceding docstring)
NEW = {'SpecificationProofs/Side24Analysis.lean', 'SpecificationProofs/Side24Interfaces.lean', 'SpecificationProofs/RNCluster.lean'}
indep = {}
for m in sorted(NEW):
    text = (PROOF/m).read_text(encoding='utf-8')
    ns = re.search(r'^namespace (UniversalLaw\.Proofs\.\w+)$', text, re.M).group(1)
    # strip Aux block(s)
    stripped = re.sub(r'^namespace Aux\n.*?^end Aux\n', '', text, flags=re.S | re.M)
    for mt in re.finditer(r'/--(.*?)-/\s*\ntheorem (\w+)\b([^\n]*(?:\n    [^\n]*)*?)(?= := | :=\n|\n  [a-z])', stripped, re.S):
        doc, name, sig = mt.group(1), mt.group(2), mt.group(3)
        pm = re.search(r'Proves: `([\w.]+)`', doc); cm = re.findall(r'(?:claim|register row) `([^`]+)`', doc)
        dne = re.search(r'Does not establish: (.*)$', doc, re.S).group(1)
        dne = ' '.join(dne.split())
        typ = re.search(r'UniversalLaw\.Spec\.\w+\.\w+', sig).group(0)
        indep[f'{ns}.{name}'] = {'module': m, 'proves': pm.group(1), 'type_head': typ, 'claims': cm, 'dne': dne}
print('independent inventory:', len(indep))
for n, d in indep.items():
    if d['proves'] != d['type_head']: fail('INV', f'{n}: Proves {d["proves"]} but type head {d["type_head"]}')
if len(indep) != 13: fail('INV', f'expected 13 column-0 non-Aux theorems in the new modules, found {len(indep)}')

# ---- 2. added entries
added = {}
for cid in hr:
    hs, ws = hr[cid]['statements'], wr[cid]['statements']
    if ws[:len(hs)] != hs: fail(cid, 'existing statement entries modified/reordered')
    for s in ws[len(hs):]:
        added.setdefault(s['declaration'], []).append(cid)
        d = indep.get(s['declaration'])
        if d is None: fail(cid, f'added proved entry {s["declaration"]} not in the independent inventory'); continue
        if s['module'] != d['module']: fail(cid, f'{s["declaration"]} module {s["module"]} != {d["module"]}')
        if s['kind'] != 'theorem' or s['formal_progress'] != 'proved' or s['package'] != PP: fail(cid, f'{s["declaration"]} bad kind/progress/package')
        if cid not in d['claims']: fail(cid, f'{s["declaration"]} docstring names claims {d["claims"]}, not this row')
        if ' '.join(s['does_not_claim'].split()) != d['dne']: fail(cid, f'{s["declaration"]} does_not_claim differs from docstring Does-not-establish:\n  REG: {s["does_not_claim"][:200]}\n  DOC: {d["dne"][:200]}')
        loc = [x for x in ws if x['package'] == SP and x['declaration'] == d['proves']]
        if len(loc) != 1: fail(cid, f'{s["declaration"]}: Prop {d["proves"]} not a unique local statement on this row')
        elif loc[0]['anchor'] != s['anchor']: fail(cid, f'{s["declaration"]}: anchor differs from the local Prop entry')
        if loc and loc[0]['formal_progress'] != 'specified': fail(cid, f'local Prop {d["proves"]} formal_progress changed to {loc[0]["formal_progress"]}')
if set(added) != set(indep): fail('STMT', f'added set != independent inventory: {set(added) ^ set(indep)}')
for n, d in indep.items():
    for cid in d['claims']:
        if cid not in added.get(n, []): fail(cid, f'inventory theorem {n} names this row but no entry was added')
print('added entries:', sum(len(v) for v in added.values()))

# ---- 3. anchors: every statement of every row, verbatim in its layer-0 source (own check, via the checker's loader)
for cid, c in wr.items():
    sid = c['layer0']['source_id']
    for s in c['statements']:
        if s['package'] == SP: continue
        if sid is None: fail(cid, f'{s["declaration"]}: no layer0 source'); continue
        if len(s['anchor']) < 16 or s['anchor'] not in sources[sid]['text']: fail(cid, f'{s["declaration"]}: anchor not verbatim (>=16) in {sid}: {s["anchor"]!r}')
print('anchors checked')

# ---- 4. claim-level labels and allowed field changes
for cid in hr:
    if hr[cid]['formal_progress'] != wr[cid]['formal_progress']: fail(cid, 'claim-level formal_progress changed')
    for k in hr[cid]:
        if hr[cid][k] != wr[cid][k] and k not in ('statements', 'kernel_evidence', 'not_established', 'next_step'): fail(cid, f'field {k} changed')
    for s in wr[cid]['statements']:
        if s['package'] == SP and s['formal_progress'] != 'specified': fail(cid, f'local statement {s["declaration"]} label {s["formal_progress"]}')
        if s['package'] != SP and s['formal_progress'] != 'proved': fail(cid, f'external statement {s["declaration"]} label {s["formal_progress"]}')
        if s['formal_progress'] == 'kernel-checked': fail(cid, 'kernel-checked label on a statement')

# ---- 5. run ids: every 11-digit run id in every string must be accompanied by its tested merge; distinguish strings changed by update7
MERGE = {'38116277597': 'c4c66bb8cb26f0cd9bc03d52bb25f1b3925b272c', '38121713840': '5b198699308e0585904c21d12bf540cf2475d2f0',
         '38149888699': '4363b6928c62da242952631c6b0357d096c3b252', '38149524759': '3306233bbdeadb15d9b4b07b94cc52a80ed8caeb',
         '38150363063': '86a6e8c876f7904cadda2d006a56f8792dc5c1af', '38122386459': '866136cca66b46e6dbd3186b47c9340f7e93b4f6',
         '38124262242': '318ff194eecc42c39a6b27c1d7bf03f30d6a34d9'}
HEADS = {'38116277597': '00d2ac6e', '38121713840': '910809d6', '38149888699': '0dcb0016', '38149524759': '88c21058', '38150363063': '12e072d6', '38122386459': 'd9b8dd33', '38124262242': '79731b96'}
def strings(o, p=''):
    if isinstance(o, str): yield p, o
    elif isinstance(o, dict):
        for k, v in o.items(): yield from strings(v, f'{p}.{k}')
    elif isinstance(o, list):
        for i, v in enumerate(o): yield from strings(v, f'{p}[{i}]')
unbound = []
for cid, c in wr.items():
    head_strings = {s for _, s in strings(hr[cid])}
    for p, s in strings(c):
        changed = s not in head_strings
        for r in set(re.findall(r'(?<!\d)\d{11}(?!\d)', s)):
            if r not in MERGE:
                unbound.append(f'{cid}{p}: run {r} ({"CHANGED by update7" if changed else "pre-existing, untouched"})')
                if changed: fail(cid, f'unknown run id {r} in a changed string at {p}')
                continue
            if MERGE[r] not in s:
                unbound.append(f'{cid}{p}: run {r} without merge {MERGE[r][:8]} ({"CHANGED" if changed else "pre-existing"})')
                if changed: fail(cid, f'run {r} without its tested merge in a changed string at {p}: {s[:200]}')
            if HEADS[r] not in s and changed: fail(cid, f'run {r} without its PR head {HEADS[r]} in a changed string at {p}')
print('unbound/unknown run mentions:', len(unbound))

# ---- 6. T4 theorems: no clause attributes a hosted receipt / kernel-check / corroboration to them
T4 = sorted(n.split('.')[-1] for n in indep)
for cid, c in wr.items():
    head_strings = {s for _, s in strings(hr[cid])}
    for p, s in strings(c):
        if s in head_strings: continue
        for clause in re.split(r'(?<=[.;]) (?=[A-Z0-9`(])', s):
            names = [n for n in T4 if re.search(r'(?<![A-Za-z])' + n + r'(?![A-Za-z])', clause)]
            if not names: continue
            bad = re.search(r'kernel-checked by|receipted by|corroborated (by|on)|has a hosted receipt|hosted run \d', clause)
            if bad and not re.search(r'no hosted receipt|first receipt|await|not evidence about any commit', clause):
                fail(cid, f'T4 {names} in a clause with receipt language at {p}: {clause[:300]}')
print('T4 clause scan done')

# ---- 7. triage sentences: per-Prop verdict by my own parse
VERD = ['PROVABLE_NOW', 'PROVABLE_WITH_WORK', 'BLOCKED_MISSING_MATHLIB', 'BLOCKED_INTERFACE', 'PREDICATE_VOCABULARY']
PROVED = {d['proves'] for d in indep.values()} | {s['declaration'] for c in H['claims'] for s in c['statements'] if s['package'] == SP and any(
    x['package'] == PP for x in c['statements'])}  # (approximate; only used for PROVABLE_* consistency below via explicit list)
proved_props_round4 = {d['proves'].split('.', 3)[-1] for d in indep.values()}
def verdict_of(short, sent):
    """Find the verdict governing `short` in the sentence: the items are '; '-separated; grouped forms handled explicitly."""
    body = sent.split('no label moves): ', 1)[1].rstrip('.')
    items = body.split('; ')
    # grouped items with inner '; ' inside parentheses: rejoin by parenthesis depth
    merged, buf, depth = [], '', 0
    for it in items:
        buf = it if not buf else buf + '; ' + it
        depth += it.count('(') - it.count(')')
        if depth <= 0: merged.append(buf); buf = ''; depth = 0
    if buf: merged.append(buf)
    hits = []
    for it in merged:
        if not re.search(r'(?<![A-Za-z0-9_])' + re.escape(short) + r'(?![A-Za-z0-9_])', it): continue
        head = it.split(': ', 1)[0]
        if head.startswith('BLOCKED_INTERFACE ('): hits.append(('BLOCKED_INTERFACE', it)); continue
        if 'PREDICATE_VOCABULARY' in head and re.search(r'(?<![A-Za-z0-9_])' + re.escape(short) + r'(?![A-Za-z0-9_])', head): hits.append(('PREDICATE_VOCABULARY', it)); continue
        # 'A, B PROVABLE_NOW and C PROVABLE_WITH_WORK: proved in round 4' -> split head on ' and '
        for grp in head.split(' and '):
            if re.search(r'(?<![A-Za-z0-9_])' + re.escape(short) + r'(?![A-Za-z0-9_])', grp.split(' (')[0]):
                vs = [v for v in VERD if v in grp.split(' (')[0]]
                if vs: hits.append((vs[0], it))
    return hits
for cid, props in TRIAGE.items():
    ns = wr[cid]['next_step']
    sents = re.findall(r'Round-4 triage \(2026-10-11, same-account, credit 0, no label moves\): .*?(?=\. [A-Z][a-z]|\.$|\. Independent|\. The enclosure)', ns)
    cnt = ns.count('Round-4 triage (')
    if cnt != 1: fail(cid, f'{cnt} triage sentences')
    i = ns.find('Round-4 triage (')
    sent = ns[i:]
    # the triage sentence ends at the first '. ' followed by a capital letter that starts a new sentence in the C rows; others end at string end
    m = re.search(r'\.( (?:The enclosure|Independent reviewer)|$)', sent)
    sent = sent[:m.start() + 1] if m else sent
    seen_short = {}
    for p in props:
        short = p['prop'].split('.', 3)[-1]
        seen_short.setdefault(short, set()).add(p['verdict'])
    for short, vs in seen_short.items():
        if len(vs) > 1: fail(cid, f'triage table gives two verdicts for {short}: {vs}')
        v = next(iter(vs))
        hits = verdict_of(short, sent)
        if not hits: fail(cid, f'Prop {short} ({v}) not found in the triage sentence'); continue
        got = {h[0] for h in hits}
        if got != {v}: fail(cid, f'Prop {short}: table {v}, sentence {got}: {[h[1][:160] for h in hits]}')
        if v in ('PROVABLE_NOW', 'PROVABLE_WITH_WORK'):
            it = hits[0][1]
            if short in proved_props_round4 and 'proved in round 4' not in it: fail(cid, f'{short} proved in round 4 but sentence item lacks it: {it[:200]}')
            if short not in proved_props_round4 and 'proved in round 4' in it:
                hd = it.split(': ', 1)[0]
                if re.search(r'(?<![A-Za-z0-9_])' + re.escape(short) + r'(?![A-Za-z0-9_])', hd): fail(cid, f'{short} said proved in round 4 but has no theorem')
            if v == 'PROVABLE_WITH_WORK' and short not in proved_props_round4:
                eff = [p['effort'] for p in props if p['prop'].endswith('.' + short)][0]
                if f'{short} PROVABLE_WITH_WORK ({eff})' not in it: fail(cid, f'{short} effort {eff} not stated: {it[:200]}')
        if v == 'BLOCKED_MISSING_MATHLIB':
            eff = [p['effort'] for p in props if p['prop'].endswith('.' + short)][0]
            it = hits[0][1]
            if f'{short} BLOCKED_MISSING_MATHLIB ({eff}; Mathlib d13f23b7: ' not in it: fail(cid, f'{short} BLOCKED_MISSING_MATHLIB item malformed/effort: {it[:200]}')
            mls = [p['mathlib'] for p in props if p['prop'].endswith('.' + short)][0]
            lemmas = [re.split(r'\s+[(—-]|\s+and\s', e.strip(), 1)[0].strip().split('.')[-1] for e in mls[:2]]
            for lem in lemmas:
                if lem not in it: fail(cid, f'{short}: cited Mathlib lemma {lem} (from the table) not in the item: {it[:300]}')
    # verdicts named in the sentence that no Prop of the row has
    row_v = {p['verdict'] for p in props}
    for v in VERD:
        if v in sent and v not in row_v: fail(cid, f'sentence names verdict {v} absent from the row')
print('triage sentences parsed for', len(TRIAGE), 'rows')

# ---- 8. overstatement regex over changed strings only (hits listed for manual reading)
PAT = [r'coefficient is enclosed', r'(?<!Nothing here )(?<!no )(?<!any )(?<!the )encloses? the coefficient(?! itself)', r'interface is inhabited', r'(?<!no )inhabitant is constructed',
       r'\baccepted\b(?! by)', r'credit (?!0)\d', r'alignment (is|was|has been) (reviewed|verified|confirmed|approved)', r'independently reviewed', r'\bverified\b',
       r'review(ed)? and accept', r'formal_progress[^.]{0,30}kernel-checked', r'\bcertif', r'\bTheorem O (is|was) (proved|established|true)', r'enclosure is (proved|established)']
hits = []
for cid, c in wr.items():
    head_strings = {s for _, s in strings(hr[cid])}
    for p, s in strings(c):
        if s in head_strings: continue
        for pat in PAT:
            for mt in re.finditer(pat, s):
                hits.append({'row': cid, 'path': p, 'pat': pat, 'ctx': s[max(0, mt.start()-100):mt.end()+100]})
Path(HERE/'overstatement_hits_b.json').write_text(json.dumps(hits, indent=1, ensure_ascii=False))
print('overstatement regex hits in changed strings:', len(hits))
for h in hits: print('  HIT', h['row'], h['path'], '|', h['pat'], '|', h['ctx'].replace('\n', ' '))
Path(HERE/'verify7b_findings.json').write_text(json.dumps({'findings': F, 'unbound_run_mentions': unbound}, indent=1, ensure_ascii=False))
print('\nFINDINGS', len(F)); [print(' -', f[:500]) for f in F]
print('UNBOUND', len(unbound)); [print(' -', u) for u in unbound]
