"""Verifier (independent of update_register7.py): structural diff of REGISTER.json HEAD vs worktree, with the refutation checks
of the task. Writes findings as it goes; exit 1 on any FAIL."""
import importlib.util
import json
import re
import sys
from pathlib import Path

SCRATCH = Path('<SCRATCH>')
REPO = SCRATCH / 'repos/math'
STMT = REPO / 'specifications/lean_statements_20261010'
PROOF = REPO / 'specifications/lean_statement_proofs_20261011'
HERE = SCRATCH / 'lvs/register_update7/verifier_register'
TRIAGE = json.loads((SCRATCH / 'lvs/register_update7/triage_by_row.json').read_text(encoding='utf-8'))
R5 = json.loads((SCRATCH / 'lvs/register_update5/receipted_52.json').read_text())
T1 = set(R5['tier1_run_38116277597_head_00d2ac6e']); T2 = set(R5['tier2_run_38121713840_head_910809d6'])

STMT_PACKAGE = 'specifications/lean_statements_20261010'
PROOF_PACKAGE = 'd6g8k5htny-coder/Math- specifications/lean_statement_proofs_20261011'
# FACTS table (from the task), independent of the script's constants
RUN_MERGE = {
    '38116277597': 'c4c66bb8cb26f0cd9bc03d52bb25f1b3925b272c',
    '38121713840': '5b198699308e0585904c21d12bf540cf2475d2f0',
    '38149888699': '4363b6928c62da242952631c6b0357d096c3b252',
    '38149524759': '3306233bbdeadb15d9b4b07b94cc52a80ed8caeb',
    '38150363063': '86a6e8c876f7904cadda2d006a56f8792dc5c1af',
    # earlier runs used by the inherited base_wf text (not in the task's FACTS; checked against register_update5 base_wf)
    '38122386459': '866136cca66b46e6dbd3186b47c9340f7e93b4f6',
    '38124262242': '318ff194eecc42c39a6b27c1d7bf03f30d6a34d9',
}
RUN_HEAD = {'38116277597': '00d2ac6e', '38121713840': '910809d6', '38149888699': '0dcb0016', '38149524759': '88c21058', '38150363063': '12e072d6'}
SHAS = {'38149888699': 'cdeb5c73aa03124ed0db16fa14d25fb03a8fa842457fce1e0c6565dbc5ee7ca8',
        '38149524759': 'ddefbdcc63223d05fd8ed34addbeebc199e271525d87dead01109cddc529e836',
        '38150363063': 'bf0e0515e2035103b07d894e2a73eeb5b542885f6da16b98c228877a5e0c1690'}
NEW_MODULES = {'SpecificationProofs/Side24Analysis.lean', 'SpecificationProofs/Side24Interfaces.lean', 'SpecificationProofs/RNCluster.lean'}
T3_MODULE = 'SpecificationProofs/P15Interpolation.lean'

findings = []
def fail(row, text):
    findings.append(f'[{row}] {text}')
    print('FAIL', row, text[:300])
def ok(msg):
    print('ok  ', msg)

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

head = json.loads((HERE / 'REGISTER_HEAD.json').read_text(encoding='utf-8'))
work = json.loads((STMT / 'REGISTER.json').read_text(encoding='utf-8'))

# ---- 0. top level unchanged
for k in head:
    if k != 'claims' and head[k] != work[k]:
        fail('TOP', f'top-level key {k} changed')
if set(head) != set(work):
    fail('TOP', 'top-level key set changed')
hrows = {c['claim_id']: c for c in head['claims']}
wrows = {c['claim_id']: c for c in work['claims']}
if list(hrows) != list(wrows):
    fail('TOP', 'claim id list/order changed')
ok(f'{len(wrows)} claim rows, same ids and order')

# ---- 1. claim-level formal_progress and other fixed fields
changed_rows = {}
for cid in hrows:
    h, w = hrows[cid], wrows[cid]
    if h['formal_progress'] != w['formal_progress']:
        fail(cid, f'claim-level formal_progress changed {h["formal_progress"]} -> {w["formal_progress"]}')
    if w['formal_progress'] not in ('none', 'specified'):
        fail(cid, f'claim-level formal_progress {w["formal_progress"]} outside none/specified')
    for k in ('kind', 'title', 'layer0', 'disposition_transcribed'):
        if h[k] != w[k]:
            fail(cid, f'field {k} changed')
    diff = [k for k in h if h[k] != w[k]]
    if diff:
        changed_rows[cid] = diff
ok(f'claim-level formal_progress unchanged for all rows; rows changed: {len(changed_rows)}')
print(json.dumps(changed_rows, indent=1))

# ---- 2. statements: only additions; existing unchanged; added entries are proof-packet theorems of the inventory
pcheck = load_module('proof_check', PROOF / 'check.py')
scheck = load_module('stmt_check', STMT / 'check.py')
inv = pcheck.source_check(PROOF, REPO, allow_dirty=True)
inv_by_name = {t['name']: t for t in inv['theorems']}
assert inv['theorem_count'] == 73 and len(inv['covered_props']) == 72, 'inventory differs from FACTS'
T4 = {t['name'] for t in inv['theorems'] if t['module'] in NEW_MODULES}
T3 = {t['name'] for t in inv['theorems'] if t['module'] == T3_MODULE}
assert len(T4) == 13 and len(T3) == 8
# independent inventory from the Lean text: column-0 theorems whose type is a Spec Prop
indep = {}
for m in NEW_MODULES:
    text = (PROOF / m).read_text(encoding='utf-8')
    for mt in re.finditer(r'^theorem (\w+)[^\n]*?:\s*UniversalLaw\.Spec\.(\w+)\.(\w+)', text, re.M):
        indep[mt.group(1)] = f'UniversalLaw.Spec.{mt.group(2)}.{mt.group(3)}'
short_t4 = {n.split('.')[-1]: n for n in T4}
if set(indep) != set(short_t4):
    fail('INV', f'independent regex inventory of the new modules differs from check.py: {set(indep) ^ set(short_t4)}')
for s, full in short_t4.items():
    if inv_by_name[full]['proves'] != indep[s]:
        fail('INV', f'{full} proves {inv_by_name[full]["proves"]} but the Lean text says {indep[s]}')
ok(f'13 new inventory theorems agree with the column-0 regex inventory: {sorted(indep)}')

sources, _ = scheck.load_sources(STMT, None, work)
added = []
for cid in hrows:
    hs, ws = hrows[cid]['statements'], wrows[cid]['statements']
    if ws[:len(hs)] != hs:
        fail(cid, 'an existing statement entry was modified or reordered')
    for s in ws[len(hs):]:
        added.append((cid, s))
        if s['package'] != PROOF_PACKAGE or s['kind'] != 'theorem' or s['formal_progress'] != 'proved':
            fail(cid, f'added entry {s["declaration"]} is not a proof-packet proved theorem: {s}')
        t = inv_by_name.get(s['declaration'])
        if t is None:
            fail(cid, f'added proved entry {s["declaration"]} is NOT in the proof packet inventory')
            continue
        if t['module'] != s['module']:
            fail(cid, f'added entry {s["declaration"]} module {s["module"]} != inventory {t["module"]}')
        if cid not in t['claim_ids']:
            fail(cid, f'added entry {s["declaration"]}: inventory docstring does not name this row (names {t["claim_ids"]})')
        if s['declaration'] not in T4:
            fail(cid, f'added entry {s["declaration"]} is not a round-4 theorem')
        local = [x for x in ws if x['package'] == STMT_PACKAGE and x['declaration'] == t['proves']]
        if len(local) != 1:
            fail(cid, f'added entry {s["declaration"]}: Prop {t["proves"]} not a unique local statement of the row')
        elif local[0]['anchor'] != s['anchor']:
            fail(cid, f'added entry {s["declaration"]}: anchor differs from the local Prop anchor')
        sid = wrows[cid]['layer0']['source_id']
        if sid is None or s['anchor'] not in sources[sid]['text'] or len(s['anchor']) < 16:
            fail(cid, f'added entry {s["declaration"]}: anchor not a verbatim substring (>=16) of layer-0 source {sid}: {s["anchor"][:80]!r}')
        dne = t.get('does_not_establish', '')
        # does_not_claim must begin with the docstring's Does-not-establish first line
        if not s['does_not_claim'].startswith(dne.rstrip()):
            fail(cid, f'added entry {s["declaration"]}: does_not_claim does not start with the docstring line {dne!r}')
if len(added) != 13 or {s['declaration'] for _, s in added} != T4:
    fail('STMT', f'added entries {len(added)}; names {sorted({s["declaration"] for _, s in added}) }')
ok(f'{len(added)} statement entries added, all T4; anchors verbatim; existing entries untouched')
# every proof-packet entry (old too) is in the inventory
for cid, w in wrows.items():
    for s in w['statements']:
        if s['package'] == PROOF_PACKAGE and s['declaration'] not in inv_by_name:
            fail(cid, f'proof-packet entry {s["declaration"]} not in inventory')
        if s['package'] == PROOF_PACKAGE and inv_by_name.get(s['declaration'], {}).get('module') != s['module']:
            fail(cid, f'proof-packet entry {s["declaration"]} module mismatch')
# every inventory theorem listed on every row its docstring names, exactly once
for t in inv['theorems']:
    for cid in t['claim_ids']:
        n = sum(1 for s in wrows[cid]['statements'] if s['package'] == PROOF_PACKAGE and s['declaration'] == t['name'])
        if n != 1:
            fail(cid, f'inventory theorem {t["name"]} listed {n} times on row (docstring names it)')
ok('every inventory theorem listed exactly once on every row its docstring names; every proof-packet entry inventoried')

# ---- 3. all anchors verbatim (full re-check of every statement, via the loader)
for cid, w in wrows.items():
    sid = w['layer0']['source_id']
    for s in w['statements']:
        if s['package'] == STMT_PACKAGE:
            continue
        if sid is None or s['anchor'] not in sources[sid]['text']:
            fail(cid, f'anchor of {s["declaration"]} not verbatim in {sid}')
ok('all non-local anchors verbatim in their layer-0 source')

# ---- 4. run-id binding: every string with a run id has its merge sha; for the FACTS runs also the PR head
def strings(o, path=''):
    if isinstance(o, str):
        yield path, o
    elif isinstance(o, dict):
        for k, v in o.items():
            yield from strings(v, f'{path}.{k}')
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from strings(v, f'{path}[{i}]')
run_re = re.compile(r'\b3\d{10}\b')
unbound = []
for c in work['claims']:
    cid = c['claim_id']
    for path, s in strings(c):
        ids = set(run_re.findall(s))
        for r in ids:
            if r not in RUN_MERGE:
                if s in set(x for _, x in strings(hrows[cid])):
                    print('note', cid, f'pre-existing (identical in HEAD) string with run id {r} at {path}; not touched by update7')
                else:
                    fail(cid, f'unknown run id {r} at {path}: {s[:150]}')
                unbound.append(f'{cid}{path}: run {r} (unknown)')
                continue
            if RUN_MERGE[r] not in s:
                fail(cid, f'run {r} mentioned without its tested merge {RUN_MERGE[r]} at {path}: {s[:200]}')
                unbound.append(f'{cid}{path}: run {r}')
            if r in RUN_HEAD and RUN_HEAD[r] not in s:
                fail(cid, f'run {r} mentioned without its PR head {RUN_HEAD[r]} at {path}')
                unbound.append(f'{cid}{path}: run {r} (no head)')
            # the merge must be described as the tested/actual merge, not as the head
            for mt in re.finditer(re.escape(RUN_MERGE[r]), s):
                ctx = s[max(0, mt.start() - 40):mt.start()]
                if 'merge' not in ctx:
                    fail(cid, f'merge sha of run {r} not introduced as a merge at {path}: ...{ctx!r}')
ok('run-id binding (merge sha + PR head) holds in every string')

# artifact sha256 consistency where a run and 'sha256' appear
for c in work['claims']:
    for path, s in strings(c):
        for r, sha in SHAS.items():
            for mt in re.finditer(r'lean-specification-proofs-' + r + r'-1, sha256 ([0-9a-f]{64})', s):
                if mt.group(1) != sha:
                    fail(c['claim_id'], f'artifact sha for run {r} is {mt.group(1)} not {sha} at {path}')
ok('artifact sha256 values match FACTS where stated')

# ---- 5. T4 text: no hosted receipt claimed; "kernel-checked" never attributed to a T4 theorem
t4_short = sorted(n.split('.')[-1] for n in T4)
t4_mentions = []
for c in work['claims']:
    cid = c['claim_id']
    for path, s in strings(c):
        if not any(n in s for n in t4_short):
            continue
        # split into sentences/clauses and check clauses that mention T4 names
        for clause in re.split(r'(?<=[.;]) ', s):
            names_here = [n for n in t4_short if n in clause]
            if not names_here:
                continue
            if 'kernel-checked by' in clause or 'receipted by' in clause or 'corroborated by' in clause:
                # allowed only if the clause describes the T4 as NOT receipted
                if not re.search(r'no hosted receipt|first receipt|await|not evidence', clause):
                    fail(cid, f'T4 theorem(s) {names_here} appear in a clause claiming a receipt at {path}: {clause[:300]}')
            t4_mentions.append((cid, path, clause[:200]))
ok(f'T4 mentions scanned: {len(t4_mentions)} clauses; none claims a hosted receipt')

# "receipt" sentences that assert all theorems are receipted on rows with T4
for cid, c in wrows.items():
    names = [s['declaration'] for s in c['statements'] if s['package'] == PROOF_PACKAGE]
    has_t4 = any(n in T4 for n in names)
    for path, s in strings(c):
        if has_t4 and 'every theorem of this row has a hosted receipt' in s:
            fail(cid, f'row has T4 theorems but says every theorem has a hosted receipt at {path}')
        if 'The 60 theorems of PR head 12e072d6 (every theorem of the tiers above)' in s and has_t4:
            # the corroboration sentence must not come after a T4 tier listing; check order: T4 clause must follow it
            i = s.find('The 60 theorems of PR head 12e072d6'); j = s.find('were added after PR head 12e072d6')
            if j != -1 and j < i:
                fail(cid, 'T4 tier listed before the "every theorem of the tiers above" corroboration sentence')
ok('row-level receipt sentences consistent with T4 presence')

# ---- 6. overstatement scan
PATTERNS = [
    (r'\bcoefficient is enclosed\b', 'coefficient enclosed'),
    (r'\bencloses the coefficient\b(?! *\.)', 'encloses the coefficient (check negation)'),
    (r'\binterface is inhabited\b', 'interface inhabited'),
    (r'\binhabitant is constructed\b(?<!no inhabitant is constructed)', 'inhabitant constructed'),
    (r'\bclaim (is|was) accepted\b', 'claim accepted'),
    (r'\baccepted\b', 'accepted (inspect)'),
    (r'independence credit (1|of 1|awarded|credited)', 'independence credited'),
    (r'\bcredit [1-9]', 'nonzero credit'),
    (r'\balignment (is |was |has been )?(reviewed|verified|confirmed|approved)', 'alignment reviewed'),
    (r'\bindependently reviewed\b', 'independently reviewed'),
    (r'\bnonauthor review (is|was|has been) (done|performed|completed|claimed)', 'review claimed'),
    (r'\bkernel-checked\b(?! by the hosted run| appears only| by run|, worktree|; receipt| for each|\))', 'kernel-checked usage (inspect)'),
    (r'formal_progress.{0,20}kernel-checked', 'kernel-checked label'),
    (r'\bproved in round 4\b', 'proved in round 4 (inspect: must refer to packet theorems)'),
    (r'\bclosed\b', 'closed (inspect)'),
    (r'\bTheorem O (is|was) (proved|established)', 'Theorem O proved'),
    (r'\bc_\{d,24\} (is|was) (enclosed|determined|proved)', 'c24 determined'),
]
hits = {}
for cid, c in wrows.items():
    if cid not in changed_rows:
        continue
    for path, s in strings(c):
        if path in ('.formal_progress',) or hrows[cid].get(path.strip('.').split('.')[0]) == s:
            pass
        for pat, label in PATTERNS:
            for mt in re.finditer(pat, s):
                ctx = s[max(0, mt.start() - 120):mt.end() + 120]
                hits.setdefault(label, []).append((cid, path, ctx))
(HERE / 'overstatement_hits.json').write_text(json.dumps(hits, indent=1, ensure_ascii=False), encoding='utf-8')
print('overstatement pattern hits (to inspect manually):', {k: len(v) for k, v in hits.items()})

# ---- 7. triage sentences vs triage_by_row.json
VERDICTS = ('PROVABLE_NOW', 'PROVABLE_WITH_WORK', 'BLOCKED_MISSING_MATHLIB', 'BLOCKED_INTERFACE', 'PREDICATE_VOCABULARY')
PROVED_PROPS = {t['proves'] for t in inv['theorems']}
triage_rows = {}
for cid, props in TRIAGE.items():
    ns = wrows[cid]['next_step']
    m = re.search(r'Round-4 triage \(2026-10-11, same-account, credit 0, no label moves\): (.*?)(?=(?: [A-Z][^;]*?:)|$)', ns)
    sents = [x for x in re.split(r'(?<=\.) ', ns) if x.startswith('Round-4 triage (')]
    if len(sents) != 1:
        fail(cid, f'expected exactly one Round-4 triage sentence, found {len(sents)}')
        continue
    sent = sents[0]
    triage_rows[cid] = sent
    sent = sent.split('no label moves): ', 1)[1]   # strip the fixed prefix before item parsing
    by_prop = {}
    for p in props:
        by_prop.setdefault(p['prop'], set()).add(p['verdict'])
    for prop, vs in by_prop.items():
        if len(vs) != 1:
            fail(cid, f'triage_by_row gives several verdicts for {prop}: {vs}')
        v = next(iter(vs))
        short = prop.split('.', 3)[-1]
        if short not in sent:
            fail(cid, f'triage sentence omits Prop {short} ({v})')
            continue
        # locate the item (split by "; ") that mentions the short name and check its verdict
        items = [it for it in sent.split('; ') if re.search(r'(?<![A-Za-z0-9_])' + re.escape(short) + r'(?![A-Za-z0-9_])', it)]
        if not items:
            fail(cid, f'no triage item names {short}')
            continue
        # verdict words in those items
        vwords = {vv for it in items for vv in VERDICTS if vv in it}
        if v == 'PROVABLE_NOW' or v == 'PROVABLE_WITH_WORK':
            # may be grouped "X PROVABLE_NOW and Y PROVABLE_WITH_WORK: proved in round 4"
            found = False
            for it in items:
                # split the item by ' and ' and ':' to find the group with the short name
                head_part = it.split(':')[0]
                for grp in head_part.split(' and '):
                    if re.search(r'(?<![A-Za-z0-9_])' + re.escape(short) + r'(?![A-Za-z0-9_])', grp) and v in grp:
                        found = True
                if re.search(r'(?<![A-Za-z0-9_])' + re.escape(short) + r' ' + v, it):
                    found = True
            if not found:
                fail(cid, f'Prop {short} verdict {v} not stated as such; items: {items}')
            if 'proved in round 4' in ' '.join(items) and prop not in PROVED_PROPS:
                # ensure the short name is in the proved group specifically
                for it in items:
                    if 'proved in round 4' in it:
                        hp = it.split(':')[0]
                        if re.search(r'(?<![A-Za-z0-9_])' + re.escape(short) + r'(?![A-Za-z0-9_])', hp):
                            fail(cid, f'Prop {short} said proved in round 4 but no theorem proves it')
            if v == 'PROVABLE_NOW' and prop not in PROVED_PROPS:
                fail(cid, f'PROVABLE_NOW Prop {short} has no theorem; sentence says: {items}')
        elif v == 'BLOCKED_INTERFACE':
            if not any('BLOCKED_INTERFACE' in it for it in items):
                fail(cid, f'Prop {short} BLOCKED_INTERFACE not stated; items: {items}')
            other = vwords - {'BLOCKED_INTERFACE'}
            if other and not any(f'{short} BLOCKED_INTERFACE' in it or it.startswith('BLOCKED_INTERFACE') for it in items):
                fail(cid, f'Prop {short} appears with verdict words {other}')
        else:
            if not any(re.search(r'(?<![A-Za-z0-9_])' + re.escape(short) + r'[^;]*? ' + v, it) or (v == 'PREDICATE_VOCABULARY' and v in it) for it in items):
                fail(cid, f'Prop {short} verdict {v} not stated as such; items: {items}')
    # verdict words in the sentence that no Prop carries
    sent_verdicts = {vv for vv in VERDICTS if vv in sent}
    row_verdicts = {next(iter(vs)) for vs in by_prop.values()}
    if sent_verdicts - row_verdicts:
        fail(cid, f'triage sentence uses verdicts {sent_verdicts - row_verdicts} absent from triage_by_row')
    # effort stated must match
    for p in props:
        short = p['prop'].split('.', 3)[-1]
        if p['verdict'] in ('PROVABLE_WITH_WORK', 'BLOCKED_MISSING_MATHLIB') and p['prop'] not in PROVED_PROPS:
            mt = re.search(re.escape(short) + r' ' + p['verdict'] + r' \(([^;)]*)', sent)
            if mt and p['effort'] not in mt.group(1):
                fail(cid, f'Prop {short}: effort {p["effort"]!r} not in stated {mt.group(1)!r}')
            if not mt:
                fail(cid, f'Prop {short}: no "{short} {p["verdict"]} (" item')
(HERE / 'triage_sentences.json').write_text(json.dumps(triage_rows, indent=1, ensure_ascii=False), encoding='utf-8')
ok(f'triage sentences checked for {len(triage_rows)} rows')
# rows with a triage sentence but not in TRIAGE
for cid, c in wrows.items():
    if cid not in TRIAGE and 'Round-4 triage' in json.dumps(c, ensure_ascii=False):
        fail(cid, 'Round-4 triage text on a row absent from triage_by_row.json')

# ---- 8. kernel_evidence: regenerated only on rows with proof-packet statements; other evidence rows untouched
for cid in hrows:
    he = [e for e in hrows[cid]['kernel_evidence'] if e['package'] != PROOF_PACKAGE]
    we = [e for e in wrows[cid]['kernel_evidence'] if e['package'] != PROOF_PACKAGE]
    if he != we:
        fail(cid, 'a non-proof-packet kernel_evidence row changed')
    hp = [e for e in hrows[cid]['kernel_evidence'] if e['package'] == PROOF_PACKAGE]
    wp = [e for e in wrows[cid]['kernel_evidence'] if e['package'] == PROOF_PACKAGE]
    names = [s['declaration'] for s in wrows[cid]['statements'] if s['package'] == PROOF_PACKAGE]
    if names and len(wp) != 1:
        fail(cid, f'row has proof-packet theorems but {len(wp)} proof-packet evidence rows')
    if not names and wp:
        fail(cid, 'proof-packet evidence row without proof-packet theorems')
    if wp:
        note = wp[0]['note']
        r1 = [n for n in names if n in T1]; r2 = [n for n in names if n in T2]; r3 = [n for n in names if n in T3]; r4 = [n for n in names if n in T4]
        if f'{len(names)} theorem(s) of this row' not in note:
            fail(cid, f'evidence note count mismatch: expects {len(names)}')
        for n in names:
            if n.split('.', 2)[-1] not in note:
                fail(cid, f'evidence note omits theorem {n}')
        if r4 and f'{len(r4)} ({", ".join(n.split(".", 2)[-1] for n in r4)}) were added after PR head 12e072d6' not in note:
            fail(cid, f'T4 clause missing or miscounted in evidence note: expected {len(r4)} T4')
        if r1 and f'{len(r1)} of them (' not in note:
            fail(cid, 'T1 clause count mismatch')
        if not r4 and ('no hosted receipt' in note or 'were added after PR head 12e072d6' in note):
            fail(cid, 'T4 clause on a row without T4 theorems')
        if not (r1 or r2 or r3) and ('corroborated' in note or 'kernel-checked by' in note):
            fail(cid, f'row with only T4 theorems has receipt/corroboration language: {note[:400]}')
        if 'worktree' in note and r4 and 'unverified worktree' not in note:
            fail(cid, 'T4 note lacks unverified-worktree wording')
        wf = wp[0]['receipt_or_workflow']
        if r4 and 'no hosted receipt yet' not in wf:
            fail(cid, 'receipt_or_workflow lacks the T4 no-receipt clause')
        if not r4 and 'no hosted receipt yet' in wf:
            fail(cid, 'receipt_or_workflow has the T4 clause without T4 theorems')
ok('kernel_evidence regenerated consistently with the tiers')

# ---- 9. other fields: not_established/next_step changed only on expected rows
for cid, diff in changed_rows.items():
    extra = set(diff) - {'statements', 'kernel_evidence', 'not_established', 'next_step'}
    if extra:
        fail(cid, f'unexpected fields changed: {extra}')
    if 'next_step' in diff and cid not in TRIAGE:
        fail(cid, 'next_step changed on a row outside triage_by_row')
    if 'not_established' in diff and cid not in ('side24-coefficient', 'math.side24-coefficient', 'OA-C6-RARE-CLUSTER-20260929-v1'):
        fail(cid, 'not_established changed on an unexpected row')
    if cid in TRIAGE and cid not in ('side24-coefficient', 'math.side24-coefficient', 'OA-C6-RARE-CLUSTER-20260929-v1'):
        hn, wn = hrows[cid]['next_step'], wrows[cid]['next_step']
        if not wn.startswith(hn.rstrip() + ' Round-4 triage ('):
            fail(cid, 'next_step not equal to old text + one appended triage sentence')
for cid in TRIAGE:
    if cid not in changed_rows:
        fail(cid, 'triage row unchanged')
ok('changed fields limited to statements/kernel_evidence/not_established/next_step on the expected rows')

(HERE / 'findings.json').write_text(json.dumps({'findings': findings, 'unbound_run_mentions': unbound}, indent=1, ensure_ascii=False), encoding='utf-8')
print('\nFINDINGS:', len(findings))
for f in findings:
    print(' -', f[:400])
sys.exit(1 if findings else 0)
