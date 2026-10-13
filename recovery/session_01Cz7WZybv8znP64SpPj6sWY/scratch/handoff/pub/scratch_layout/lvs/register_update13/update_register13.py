"""Round-13 register update (round 6 of the proof packet): statement entries, a pending receipt tier and the regenerated
side24 texts for the three round-6 proof-packet theorems

    UniversalLaw.Proofs.Side24Images.imageBound               : UniversalLaw.Spec.Side24.ImageBound
    UniversalLaw.Proofs.Side24Images.entrywiseCovarianceBound : UniversalLaw.Spec.Side24.EntrywiseCovarianceBound
    UniversalLaw.Proofs.Side24Images.covarianceOrdering       : UniversalLaw.Spec.Side24.CovarianceOrdering

(module SpecificationProofs/Side24Images.lean, uncommitted in the worktree on top of Math- main 4afcca6a, the merge of PR
#439). Derived from update_register10.py (round 5: proved entries from docstrings, the pending tier PENDING5_PHRASE /
WF_PENDING5, the receipt clause 'await the next hosted ...', the triage item and the regenerated row texts). Its input is
the committed register of 4afcca6a (blob 05cc80c1..., the round-5 receipt binding of update_register11.py as landed), so
every edit is an exact-string replacement over that committed text with an asserted occurrence count; no row text is
re-generated wholesale.

Scientific effect: NONE. Edits, and nothing else (asserted):
  (a) three `proved` statement entries appended to the statements of row side24-coefficient (the only row registering any
      of the three Props: check.py's claim_ids), in the exact shape of the round-5 entries: package, module, declaration,
      kind theorem, formal_progress proved, the Prop's own registered anchor, and does_not_claim = the theorem docstring's
      'Does not establish:' field (lines joined by single spaces);
  (b) row side24-coefficient's proof-packet kernel_evidence: the theorem count/listing regenerated, one pending-tier
      sentence (PENDING6_PHRASE, round 5's PENDING5_PHRASE shape) for the three theorems, naming PR head c6ce0256 (run
      38195810463, actual tested merge 1320c5b2... with base be014794..., receipt theorem_count 75) as the last head whose
      receipt this register binds; WF_PENDING6 appended to its receipt_or_workflow; in the other nine proof-packet
      kernel_evidence notes ONLY the inventory parenthetical moves (75 -> 78 theorems over 13 -> 14 modules, 74 -> 77
      Props: the source-mode inventory of the tree this register describes);
  (c) not_established of side24-coefficient: the Prop listing and entry count, the pending tier, one description per
      theorem (what it proves for the formal objects, from its docstring), the 'None of them is ...' sentence (ImageBound
      and the covariance ordering leave it) and the 'still specified with no theorem' sentence;
  (d) next_step of side24-coefficient: the targets clause, the receipt clause (three theorems await their first receipt;
      the 'every theorem of this row' sentence narrowed to the 19 receipted ones), the three round-4 triage items (verdict,
      effort and finding kept verbatim, plus ': proved in round 6 (2026-10-12; see entries, no hosted receipt yet; ...)'
      naming what the module supplied), the README-reasons sentence, the reviewer count and the reviewer anchors;
  (e) not_established of math.side24-coefficient: the landing-row count/listing and the pending tier.
No claim-level label changes (claim formal_progress stays none/specified on all 171 rows); `proved` only on the three
theorem entries; `kernel-checked` only as a receipt word; every mention of a hosted run names the PR head AND the actual
tested merge (C422-E1), asserted per string leaf. A theorem is evidence about its Prop for the formal objects only, never
about an informal claim; alignment PENDING_INDEPENDENT_REVIEW; same-account author, organizational-independence credit 0;
dispositions are transcribed, never re-derived; no triage verdict is altered.

FACTS (coordinator-verified 2026-10-12): the committed register of 4afcca6a binds round 5 to lean-specification-proofs.yml
run 38195810463 (PR #439 head c6ce025652f76be6e580fa24947380253450cd3b, actual tested merge
1320c5b217e491d301ea140e5cc25b07ff5a21b7 with base be01479435f98f34b802258ee9ec3c3578f37169, receipt theorem_count 75,
artifact sha256 d8abd42be0facd4248647c3c7eb67a264bd19b9b47c6128733b5b5fc93b0a6ff). The three theorems: kernel-checked
locally at the pin with -DwarningAsError=true, `#print axioms` within {propext, Classical.choice, Quot.sound}, types
checked against the registered Props by `#check`, on an unverified worktree (2026-10-12): evidence about no commit.

usage: python3 -B -S update_register13.py   (R13_OUT=<file> writes the result there instead of the worktree register)
"""
import difflib
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

SCRATCH = Path(os.environ['LVS_SCRATCH'])  # scrubbed for publication: as-run value was the session scratch root
REPO = SCRATCH / 'repos/math_r5'
STMT_REL = 'specifications/lean_statements_20261010'
PROOF_REL = 'specifications/lean_statement_proofs_20261011'
STMT, PROOF = REPO / STMT_REL, REPO / PROOF_REL
REG_PATH = STMT / 'REGISTER.json'
OUT_DIR = SCRATCH / 'lvs/register_update13'
OUT_REG = Path(os.environ['R13_OUT']) if os.environ.get('R13_OUT') else REG_PATH
STMT_PACKAGE = STMT_REL
PROOF_PACKAGE = 'd6g8k5htny-coder/Math- specifications/lean_statement_proofs_20261011'
EXPECTED_HEAD = '4afcca6a79a8a0acef251ae73988a7914c8080e7'
EXPECTED_BLOB = '05cc80c1ed9040aa7282fb1543679743807d55ba'      # 4afcca6a:specifications/lean_statements_20261010/REGISTER.json
AXIOMS = '{propext, Classical.choice, Quot.sound}'
MIN_ANCHOR = 16

# ---- rounds 1-5 (as bound in the committed register; never re-derived here)
RUN1, C1, M1 = '38116277597', '00d2ac6e', 'c4c66bb8cb26f0cd9bc03d52bb25f1b3925b272c'
RUN2, C2, M2 = '38121713840', '910809d6', '5b198699308e0585904c21d12bf540cf2475d2f0'
RUN3B, C3B, M3B = '38122386459', 'd9b8dd33', '866136cca66b46e6dbd3186b47c9340f7e93b4f6'
RUN4, C4, M4 = '38124262242', '79731b96', '318ff194eecc42c39a6b27c1d7bf03f30d6a34d9'
RUN6, C6, M6 = '38149888699', '0dcb0016', '4363b6928c62da242952631c6b0357d096c3b252'
RUNA, C5A, M5A = '38149524759', '88c21058', '3306233bbdeadb15d9b4b07b94cc52a80ed8caeb'
RUN7, C7, M7 = '38150363063', '12e072d6', '86a6e8c876f7904cadda2d006a56f8792dc5c1af'
RUN8, C8, M8 = '38180705952', '48bfcbae', '9833d6d9f9facc335e1a56e243c3144adefe6225'
RUN9, C9, M9 = '38187538744', 'be014794', '79d3964cf70e2eadad5c9629eef8f4b47f218dd0'
RUN10, C10, M10 = '38195810463', 'c6ce0256', '1320c5b217e491d301ea140e5cc25b07ff5a21b7'
BASE4 = 'be01479435f98f34b802258ee9ec3c3578f37169'     # head of PR #434 = base of PR #439
assert BASE4.startswith(C9)
RUN_MERGE = ((RUN1, M1), (RUN2, M2), (RUN3B, M3B), (RUN4, M4), (RUN6, M6), (RUNA, M5A), (RUN7, M7), (RUN8, M8), (RUN9, M9),
             (RUN10, M10))
RUN_HEAD = {RUN1: C1, RUN2: C2, RUN3B: C3B, RUN4: C4, RUN6: C6, RUNA: C5A, RUN7: C7, RUN8: C8, RUN9: C9, RUN10: C10}
for r, m in RUN_MERGE:
    assert r.isdigit() and len(r) == 11 and re.fullmatch(r'[0-9a-f]{40}', m), (r, m)
assert len({r for r, _ in RUN_MERGE}) == len(RUN_MERGE)
WORKSPACE_LANDING_RUN = '38096827443'   # main's workspace-landing run, pre-existing in the register; not a pull_request run

NEW_MODULES = ('SpecificationProofs/Side24Images.lean',)   # root import order: last
PREV_MODULES = ('SpecificationProofs/Side24Covariance.lean', 'SpecificationProofs/Side24ConeMoment.lean')
S24 = 'UniversalLaw.Spec.Side24.'
NEW_PROPS = (S24 + 'ImageBound', S24 + 'EntrywiseCovarianceBound', S24 + 'CovarianceOrdering')
ROW, NODE = 'side24-coefficient', 'math.side24-coefficient'

# ---- the pending forms of round 6: round 5's PENDING5_PHRASE / WF_PENDING5, with c6ce0256's facts
PENDING6_PHRASE = (f'were added in round 6 on top of Math- main 4afcca6a, the merge of PR #439, after PR head {C10} (the last head whose '
                   f'receipt this register binds: run {RUN10}, actual tested merge {M10} with base {BASE4}, receipt theorem_count 75): '
                   f'locally `lake env lean -DwarningAsError=true` exit 0 and `#print axioms` within {AXIOMS} on an unverified worktree, '
                   'not evidence about any commit; the next hosted lean-specification-proofs.yml run is the first receipt for them')
WF_PENDING6 = (f'; the theorems added in round 6 after PR head {C10} (module Side24Images; run {RUN10} (PR head {C10}; actual tested merge '
               f'{M10} with base {BASE4}), receipt theorem_count 75, does not name them) have no hosted receipt yet: the next hosted '
               'lean-specification-proofs.yml run is their first receipt')
PROVED_LABEL = '`proved` is the label of a committed proof package; `kernel-checked` appears only in a run receipt.'

# ----------------------------------------------------------------------------- load: HEAD, committed == worktree == blob
def git(*args):
    return subprocess.run(['git', '-C', str(REPO), *args], check=True, capture_output=True, text=True, encoding='utf-8').stdout


head = git('rev-parse', 'HEAD').strip()
if head != EXPECTED_HEAD:
    sys.exit(f'REFUSED: worktree HEAD is {head}, expected {EXPECTED_HEAD}')
committed = git('show', f'{EXPECTED_HEAD}:{STMT_REL}/REGISTER.json')
raw = REG_PATH.read_bytes()
blob = hashlib.sha1(b'blob %d\0' % len(raw) + raw).hexdigest()
if raw.decode('utf-8') != committed or blob != EXPECTED_BLOB:
    sys.exit(f'REFUSED: the worktree REGISTER.json (blob {blob}) is not the 4afcca6a blob {EXPECTED_BLOB} (already transformed?)')
old = json.loads(committed)
reg = json.loads(committed)
rows = {c['claim_id']: c for c in reg['claims']}
old_rows = {c['claim_id']: c for c in old['claims']}
assert len(rows) == 171


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pcheck = load_module('proof_check13', PROOF / 'check.py')
scheck = load_module('stmt_check13', STMT / 'check.py')
inv = pcheck.source_check(PROOF, REPO, allow_dirty=True)
assert inv['formalization_status'] == 'proved' and inv['worktree'] != 'CLEAN'      # the module is uncommitted
assert inv['theorem_count'] == 78 and len(inv['covered_props']) == 77, (inv['theorem_count'], len(inv['covered_props']))
assert inv['duplicate_proofs'] == ['UniversalLaw.Spec.P15.TriangleNeedsThree'], inv['duplicate_proofs']
assert list(inv['modules'][-3:]) == list(PREV_MODULES + NEW_MODULES) and len(inv['modules']) == 14, inv['modules']
theorems = inv['theorems']
T6 = [t['name'] for t in theorems if t['module'] in NEW_MODULES]
assert T6 == ['UniversalLaw.Proofs.Side24Images.imageBound', 'UniversalLaw.Proofs.Side24Images.entrywiseCovarianceBound',
              'UniversalLaw.Proofs.Side24Images.covarianceOrdering'], T6
assert [t['proves'] for t in theorems if t['module'] in NEW_MODULES] == list(NEW_PROPS)
assert all(t['claim_ids'] == [ROW] for t in theorems if t['module'] in NEW_MODULES)
listed_before = {s['declaration'] for c in old['claims'] for s in c['statements'] if s['package'] == PROOF_PACKAGE}
assert listed_before == {t['name'] for t in theorems} - set(T6) and len(listed_before) == 75
assert all(p in inv['covered_props'] for p in NEW_PROPS)
LABELS = ('Proves:', 'Method:', 'Does not establish:')


def docstring_fields(module: str, line: int) -> dict:      # update_register10 verbatim
    text = (PROOF / module).read_text(encoding='utf-8')
    code, docs = pcheck.lex(text)
    lines = code.split('\n')
    doc_end = {end: raw for _, end, raw in docs}
    i = line - 1
    j = i if i in doc_end else i - 1
    while j >= 0 and j not in doc_end and not lines[j].strip():
        j -= 1
    raw = doc_end[j][3:-2]
    out = {}
    for label in LABELS:
        m = re.search(r'^[ \t]*' + re.escape(label) + r'(.*)$', raw, re.M)
        cont = []
        for ln in raw[m.end():].split('\n')[1:]:
            if not ln.strip() or any(ln.strip().startswith(l) for l in LABELS):
                break
            cont.append(ln.strip())
        out[label.rstrip(':')] = ' '.join([m.group(1).strip()] + cont).strip()
    return out


for t in theorems:
    if t['module'] in NEW_MODULES:
        t['doc_full'] = docstring_fields(t['module'], t['line'])
        assert t['doc_full']['Does not establish'] and t['doc_full']['Proves'].split('`')[1] == t['proves'], t['name']
        src = (PROOF / t['module']).read_text(encoding='utf-8')
        k = src.index('theorem ' + t['name'].rsplit('.', 1)[1] + ' :')
        d = src[src.rindex('/--', 0, k) + 3:src.index('-/', src.rindex('/--', 0, k))]
        assert ' '.join(d[d.index('Does not establish:') + len('Does not establish:'):].split()) == t['doc_full']['Does not establish'], t['name']
        assert t['doc_full']['Does not establish'].endswith('It promotes, closes, discharges or reclassifies nothing.'), t['name']

sources, _ = scheck.load_sources(STMT, None, reg)


def strings(o):
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from strings(v)


def leaves(o, path=''):
    if isinstance(o, str):
        yield path, o
    elif isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, f'{path}/{k}')
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, f'{path}[{i}]')


def count_all(o, needle):
    return sum(s.count(needle) for s in strings(o))


def census(r):
    return {'statements': sum(len(c['statements']) for c in r['claims']),
            'proved': sum(1 for c in r['claims'] for s in c['statements'] if s['formal_progress'] == 'proved'),
            'kernel_evidence': sum(len(c['kernel_evidence']) for c in r['claims']),
            'claims': len(r['claims']),
            'claim_fp': {k: sum(1 for c in r['claims'] if c['formal_progress'] == k) for k in ('none', 'specified')}}


before = census(old)
assert before == {'statements': 354, 'proved': 157, 'kernel_evidence': 28, 'claims': 171, 'claim_fp': {'none': 138, 'specified': 33}}, before
for needle in ('round 6', 'Side24Images', 'imageBound', 'entrywiseCovarianceBound', 'covarianceOrdering', '4afcca6a'):
    assert needle not in committed, needle
TRIAGE_MARK = 'Round-4 triage (2026-10-11'
triaged_before = {c['claim_id'] for c in old['claims'] if TRIAGE_MARK in c['next_step']}
assert len(triaged_before) == 30

# ----------------------------------------------------------------------------- shared listing helpers (update_register10)
short = lambda ns: ', '.join(n.split('.', 2)[-1] for n in ns)
prop_short = lambda n: n.split('.', 3)[-1]
listing = lambda props: ', '.join(prop_short(p) for p in props) if props else 'none'
PREDICATES = {'IsRealizedFamily', 'IsGood', 'IsKDecomposable', 'IsGeneratorCover', 'IsPaletteFeasible', 'IsMinimalForbidden',
              'IsTransversal', 'IsProperDownset', 'IsUnitTuple', 'NegDefCone'}


def row_props(row, names):
    local = [s['declaration'] for s in row['statements'] if s['package'] == STMT_PACKAGE and s['kind'] == 'def_prop']
    proved_props = {}
    for t in theorems:
        if t['name'] in names and row['claim_id'] in t['claim_ids']:
            proved_props.setdefault(t['proves'], []).append(t['name'])
    return local, proved_props, [p for p in local if p in proved_props], [p for p in local if p not in proved_props]


def split_targets(props):
    return [p for p in props if prop_short(p) not in PREDICATES], [p for p in props if prop_short(p) in PREDICATES]


def remaining_sentence(without):           # update_register10 verbatim
    targets, preds = split_targets(without)
    s = f'Theorem-shaped Props of this row still `specified` with no theorem: {listing(targets)}'
    if preds:
        s += (f'; the parametrised predicates {listing(preds)} are the vocabulary of the other Props, not closed statements '
              '(a theorem of their type would assert them for every argument, which is false), and are not targets')
    return s + '. '


def targets_clause(without, lane):         # update_register10 verbatim
    targets, preds = split_targets(without)
    if targets:
        return f'{lane}: {listing(targets)} remain the targets of this row' + (f' (the parametrised predicates {listing(preds)} are vocabulary, not targets)' if preds else '')
    return f'{lane}: every theorem-shaped Prop of this row has a theorem' + (f' (the parametrised predicates {listing(preds)} are vocabulary, not targets)' if preds else '')


ALL_NAMES = {t['name'] for t in theorems}
OLD_NAMES = ALL_NAMES - set(T6)
counts = {}


def replace_once(text, old_s, new_s, label):
    n = text.count(old_s)
    assert n == 1, (label, n, old_s[:140])
    counts[label] = counts.get(label, 0) + 1
    return text.replace(old_s, new_s)


# ----------------------------------------------------------------------------- A. the three statement entries
added = []
for t in theorems:
    if t['module'] not in NEW_MODULES:
        continue
    for cid in t['claim_ids']:
        row = rows[cid]
        assert not any(s['package'] == PROOF_PACKAGE and s['declaration'] == t['name'] for s in row['statements'])
        local = [s for s in row['statements'] if s['package'] == STMT_PACKAGE and s['declaration'] == t['proves']]
        assert len(local) == 1 and local[0]['kind'] == 'def_prop' and local[0]['formal_progress'] == 'specified', (cid, t['proves'])
        anchor = local[0]['anchor']
        sid = row['layer0']['source_id']
        assert sid and anchor in sources[sid]['text'] and len(anchor) >= MIN_ANCHOR, (cid, t['name'])   # anchor rule: refuse, never weaken
        row['statements'].append({'package': PROOF_PACKAGE, 'module': t['module'], 'declaration': t['name'], 'kind': 'theorem',
                                  'formal_progress': 'proved', 'anchor': anchor, 'does_not_claim': t['doc_full']['Does not establish']})
        added.append((cid, t['name']))
assert added == [(ROW, T6[0]), (ROW, T6[1]), (ROW, T6[2])], added
r5 = [s for s in old_rows[ROW]['statements'] if s['package'] == PROOF_PACKAGE and s['module'] in PREV_MODULES]
assert len(r5) == 2 and all(list(s) == list(r5[0]) for s in rows[ROW]['statements'][-3:])
for c in reg['claims']:
    for s in c['statements']:
        if s['package'] == PROOF_PACKAGE:
            assert s['declaration'] in ALL_NAMES, s['declaration']
assert {s['declaration'] for c in reg['claims'] for s in c['statements'] if s['package'] == PROOF_PACKAGE} == ALL_NAMES

# ----------------------------------------------------------------------------- B. kernel_evidence notes and the WF clause
OLD_MODULES_TXT = 'SpecificationProofs/{' + ','.join(Path(m).stem for m in inv['modules'] if m not in NEW_MODULES) + '}.lean'
NEW_MODULES_TXT = 'SpecificationProofs/{' + ','.join(Path(m).stem for m in inv['modules']) + '}.lean'
OLD_INV = f'(source mode SOURCE_CHECK_PASS: 75 inventory theorems over {OLD_MODULES_TXT}, 74 distinct registered Props, each theorem typed exactly'
NEW_INV = (f'(source mode SOURCE_CHECK_PASS: {inv["theorem_count"]} inventory theorems over {NEW_MODULES_TXT}, {len(inv["covered_props"])} distinct '
           'registered Props, each theorem typed exactly')
assert count_all(old, OLD_INV) == 10
row_mine_old = [s['declaration'] for s in old_rows[ROW]['statements'] if s['package'] == PROOF_PACKAGE]
row_mine = [s['declaration'] for s in rows[ROW]['statements'] if s['package'] == PROOF_PACKAGE]
assert len(row_mine_old) == 19 and row_mine == row_mine_old + T6
pp_rows = []
for cid, row in rows.items():
    pp = [e for e in row['kernel_evidence'] if e['package'] == PROOF_PACKAGE]
    mine = [s['declaration'] for s in row['statements'] if s['package'] == PROOF_PACKAGE]
    assert len(pp) == (1 if mine else 0), cid
    if not pp:
        continue
    pp_rows.append(cid)
    e = pp[0]
    e['note'] = replace_once(e['note'], OLD_INV, NEW_INV, 'B.inventory_parenthetical')
    if cid == ROW:
        e['note'] = replace_once(e['note'], f'19 theorems of this row ({short(row_mine_old)}) are inventoried',
                                 f'{len(row_mine)} theorems of this row ({short(row_mine)}) are inventoried', 'B.row_count_listing')
        e['note'] = replace_once(e['note'], '. ' + PROVED_LABEL, f'. {len(T6)} ({short(T6)}) {PENDING6_PHRASE}. ' + PROVED_LABEL, 'B.pending_tier')
        assert WF_PENDING6 not in e['receipt_or_workflow'] and e['receipt_or_workflow'].endswith(')')
        e['receipt_or_workflow'] = e['receipt_or_workflow'] + WF_PENDING6
        counts['B.wf_pending'] = 1
assert len(pp_rows) == 10 and counts['B.inventory_parenthetical'] == 10, counts

# ----------------------------------------------------------------------------- C. not_established of side24-coefficient
r = rows[ROW]
_, pp_old, with_old, without_old = row_props(old_rows[ROW], OLD_NAMES)
_, pp_new, with_new, without_new = row_props(r, ALL_NAMES)
assert len(with_old) == 19 and len(without_old) == 7 and len(with_new) == 22 and len(without_new) == 4
assert set(with_new) - set(with_old) == set(NEW_PROPS) and [p for p in without_old if p not in NEW_PROPS] == without_new
assert sorted(prop_short(p) for p in without_new) == ['CoefficientEnclosure', 'IsUnitTuple', 'NegDefCone', 'ReferenceComparison']
ne = r['not_established']
ne = replace_once(ne, f'specifications/lean_statement_proofs_20261011: {listing(with_old)} (19 entries below, label `proved`;',
                  f'specifications/lean_statement_proofs_20261011: {listing(with_new)} (22 entries below, label `proved`;', 'C.listing_and_count')
ne = replace_once(ne, 'the base of PR #439, and the merge has the same tree as its head). What those theorems prove',
                  f'the base of PR #439, and the merge has the same tree as its head; {len(T6)} of them ({short(T6)}) {PENDING6_PHRASE}). '
                  'What those theorems prove', 'C.pending_tier')
IMG_DESC = ('; and, for the formal K24, Kinf, C24 and Cref, display (2): |D^q K24(0) u - D^q Kinf(0) u| < imageConstE for d = 2 or 3, '
            'every q <= 6 and every unit tuple u (ImageBound: D^q K24(0) = theta(0)^(-1) sum_n D^q Kinf(24 n) for q <= 6, by the term-wise '
            'derivative on the unit ball through a local version of iteratedFDeriv_tsum on an open preconnected set, proved inside the module as the helper '
            'Aux.ib_iteratedFDeriv_tsum_of_isOpen (a helper, not an inventory theorem and not a Mathlib result) from Mathlib\'s '
            'hasFDerivAt_tsum_of_isPreconnected, then the packet\'s DerivativeBoundAtLeastOne, DerivativeBoundAtZero, LatticeImageSumBound and '
            'ExponentialBounds theorems, 1458 (76 * 24^6 + 15) e^(-288) < imageConstE; the hypothesis ThetaSummable d is not used; nothing '
            'for q > 6; differentiability of K24 is used inside the proof but stated by no inventory theorem)')
EC_DESC = ('; the entrywise bound |C24 a b - Cref a b| <= 2 imageConstE for d = 2 or 3 and every pair of joint coordinates a, b, at u = R 0 '
           'in an orthonormal frame (EntrywiseCovarianceBound: '
           'the common factor svecScale a * svecScale b * (-1)^|b| has absolute value at most 2 and the contractions run along at most '
           'six unit frame vectors, so ImageBound applies; the spectral-norm bound 20E is not specified)')
CO_DESC = ('; and display (3): C24 - (1 - epsilon) Cref and (1 + epsilon) Cref - C24 positive semidefinite in Mathlib\'s sense, '
           'IsHermitian included, at u = R 0 in an orthonormal frame for d = 2 or 3 (CovarianceOrdering: C24 is symmetric because each '
           'contraction of K24 at 0 of total order at most 6 is theta(0)^(-1) sum_n of contractions of the analytic Kinf at 24 n, invariant '
           'under the block swap by Mathlib\'s ContDiffAt.iteratedFDeriv_comp_perm, with the odd total orders vanishing by n -> -n together '
           'with the evenness of Kinf (D^q Kinf(-y) = (-1)^q D^q Kinf(y)); Cref is symmetric with '
           'Cref - I/3 positive semidefinite by ReferenceCovarianceLowerBound; Cauchy-Schwarz over at most 13 coordinates gives '
           '|x . (C24 - Cref) x| <= 26 imageConstE (x . x), and 78 imageConstE <= epsilon by norm_num; the source\'s route through the '
           'spectral-norm bound 20E and 60E < epsilon is not reproduced and main\'s covariance_relative_bound is neither imported, cited '
           'nor relied on; none of these matrices is shown to be the covariance of any random vector, and the passage to marginal and '
           'conditional covariances is not specified)')
ne = replace_once(ne, 'and the value 29/6 - sqrt 6 (ConeMomentD2Algebra) is not used). None of them is the Gaussian conditioning, ImageBound, '
                  'the covariance ordering or any enclosure of the coefficient',
                  'and the value 29/6 - sqrt 6 (ConeMomentD2Algebra) is not used)' + IMG_DESC + EC_DESC + CO_DESC + '. None of them is the '
                  'Gaussian conditioning or any enclosure of the coefficient', 'C.round6_descriptions_and_none_of_them')
ne = replace_once(ne, remaining_sentence(without_old), remaining_sentence(without_new), 'C.remaining_sentence')
r['not_established'] = ne

# ----------------------------------------------------------------------------- D. next_step of side24-coefficient
ns = r['next_step']
ns = replace_once(ns, targets_clause(without_old, 'Math- Mathlib lane'), targets_clause(without_new, 'Math- Mathlib lane'), 'D.targets_clause')
ns = replace_once(ns, f'; every theorem of this row has a hosted receipt: the 2 added after PR head {C9} (',
                  f'; the {len(T6)} theorems of this row added in round 6 after PR head {C10} ({short(T6)}) await the next hosted '
                  f'lean-specification-proofs.yml run, their first receipt; the other theorems of this row have a hosted receipt: the 2 added '
                  f'after PR head {C9} (', 'D.receipt_clause_pending')
ns = replace_once(ns, f'; every theorem of this row receipted on PR head {C10} by run {RUN10}',
                  f'; each of those {len(row_mine_old)} theorems receipted on PR head {C10} by run {RUN10}', 'D.receipt_clause_narrowed')
IB_ITEM = ('ImageBound BLOCKED_MISSING_MATHLIB (weeks; missing at Mathlib d13f23b7: no local (open-set) iteratedFDeriv_tsum; the global '
           'version is inapplicable to lattice translates; present but insufficient: iteratedFDeriv_tsum (SmoothSeries.lean), '
           'iteratedFDeriv_tsum_apply (SmoothSeries.lean))')
ns = replace_once(ns, IB_ITEM + '; ',
                  IB_ITEM + ': proved in round 6 (2026-10-12; see entries, no hosted receipt yet; the missing local iteratedFDeriv_tsum is '
                  'supplied inside the proof module, not by Mathlib, as the helper Aux.ib_iteratedFDeriv_tsum_of_isOpen, by the induction of '
                  'the global version from hasFDerivAt_tsum_of_isPreconnected); ', 'D.triage_item_image')
CO_ITEM_END = ('present but insufficient: iteratedFDeriv_tsum (SmoothSeries.lean), hasFDerivAt_tsum_of_isPreconnected (SmoothSeries.lean))'
               '; EntrywiseCovarianceBound BLOCKED_MISSING_MATHLIB')
ns = replace_once(ns, CO_ITEM_END,
                  'present but insufficient: iteratedFDeriv_tsum (SmoothSeries.lean), hasFDerivAt_tsum_of_isPreconnected (SmoothSeries.lean))'
                  ': proved in round 6 (2026-10-12; see entries, no hosted receipt yet; the local term-wise differentiation is the same helper '
                  'as for ImageBound; the higher-order symmetry used is Mathlib\'s ContDiffAt.iteratedFDeriv_comp_perm '
                  '(Analysis/Analytic/IteratedFDeriv.lean), present at d13f23b7 and applicable because Kinf d is analytic: the analytic case of the '
                  'listed "higher-order symmetry of iteratedFDeriv", which is all the proof needs, was present at the pin, while the general '
                  'version for finitely smooth functions beyond order two is still absent at d13f23b7; the bound used is 26E, 2E times at most 13 coordinates, with '
                  '78E <= epsilon); EntrywiseCovarianceBound BLOCKED_MISSING_MATHLIB', 'D.triage_item_ordering')
EC_ITEM_END = ('present but insufficient: norm_iteratedFDeriv_comp_le (Bounds.lean), iteratedFDeriv_tsum (SmoothSeries.lean)); '
               'CoefficientEnclosure BLOCKED_INTERFACE')
ns = replace_once(ns, EC_ITEM_END,
                  'present but insufficient: norm_iteratedFDeriv_comp_le (Bounds.lean), iteratedFDeriv_tsum (SmoothSeries.lean)): proved in '
                  'round 6 (2026-10-12; see entries, no hosted receipt yet; from the ImageBound theorem, whose proof supplies the local '
                  'term-wise differentiation); CoefficientEnclosure BLOCKED_INTERFACE', 'D.triage_item_entrywise')
ns = replace_once(ns, 'the reasons for ImageBound, CovarianceOrdering, EntrywiseCovarianceBound, CoefficientEnclosure and ReferenceComparison '
                  'stand (for the last two the triage confirms the README\'s reason as the precise missing ingredient)',
                  'for ImageBound, CovarianceOrdering and EntrywiseCovarianceBound, proved in round 6 (no hosted receipt yet), the README\'s '
                  'BLOCKED_MISSING_MATHLIB (weeks) reasons are superseded by their theorems (the local term-wise differentiation the README '
                  'named as missing from Mathlib is supplied inside the proof module, and of the higher-order symmetry it named as missing the '
                  'analytic case the proof needs is present at the pin as ContDiffAt.iteratedFDeriv_comp_perm, the general version for '
                  'finitely smooth functions being still absent); the reasons for CoefficientEnclosure '
                  'and ReferenceComparison stand (for these two the triage confirms the README\'s reason as the precise missing ingredient)',
                  'D.readme_reasons')
ns = replace_once(ns, 'in particular of the 19 Props of this row that have a theorem of exactly their type',
                  f'in particular of the {len(with_new)} Props of this row that have a theorem of exactly their type', 'D.reviewer_count')
ns = replace_once(ns, 'ReferenceCovarianceLowerBound with the Section 3 display C_ref >= I/3, with Mathlib\'s Matrix.PosSemidef, IsHermitian '
                  'included, read as the source\'s order).',
                  'ReferenceCovarianceLowerBound with the Section 3 display C_ref >= I/3, with Mathlib\'s Matrix.PosSemidef, IsHermitian '
                  'included, read as the source\'s order; ImageBound with display (2); EntrywiseCovarianceBound with the Section 3 entrywise '
                  'sentence; CovarianceOrdering with display (3), with the same reading of the order).', 'D.reviewer_anchors')
r['next_step'] = ns
assert ns.count(TRIAGE_MARK) == 1

# ----------------------------------------------------------------------------- E. not_established of math.side24-coefficient
n = rows[NODE]
_, _, w_node, wo_node = row_props(n, ALL_NAMES)
assert not w_node and [prop_short(p) for p in wo_node] == ['CoefficientEnclosure', 'ReferenceComparison']
nn = n['not_established']
nn = replace_once(nn, f'the 19 proof-packet theorems of the landing row ({short(row_mine_old)}; none of them the enclosure;',
                  f'the {len(row_mine)} proof-packet theorems of the landing row ({short(row_mine)}; none of them the enclosure;', 'E.count_listing')
nn = replace_once(nn, 'the base of PR #439, and the merge has the same tree as its head) are listed on the landing row',
                  f'the base of PR #439, and the merge has the same tree as its head; {len(T6)} of them ({short(T6)}) {PENDING6_PHRASE}) '
                  'are listed on the landing row', 'E.pending_tier')
n['not_established'] = nn

# ----------------------------------------------------------------------------- F. invariants: nothing else changed
for k in reg:
    if k != 'claims':
        assert reg[k] == old[k], k
assert [c['claim_id'] for c in reg['claims']] == [c['claim_id'] for c in old['claims']]
ALLOWED = {ROW: {'statements', 'kernel_evidence', 'not_established', 'next_step'}, NODE: {'not_established'}}
for cid in pp_rows:
    ALLOWED.setdefault(cid, set()).add('kernel_evidence')
changed_fields = {}
for cid, row in rows.items():
    o = old_rows[cid]
    assert list(row) == list(o), cid
    assert row['formal_progress'] == o['formal_progress'] and row['formal_progress'] in ('none', 'specified'), cid
    for k in row:
        if row[k] != o[k]:
            assert k in ALLOWED.get(cid, set()), (cid, k)
            changed_fields.setdefault(cid, []).append(k)
    if cid == ROW:
        assert row['statements'][:-3] == o['statements'] and len(row['statements']) == len(o['statements']) + 3
    elif 'statements' in row:
        assert row['statements'] == o['statements'], cid
    for e_new, e_old in zip(row['kernel_evidence'], o['kernel_evidence']):
        assert len(row['kernel_evidence']) == len(o['kernel_evidence']) and list(e_new) == list(e_old) and e_new['package'] == e_old['package']
        if e_new['package'] != PROOF_PACKAGE:
            assert e_new == e_old, (cid, e_new['package'])
        elif cid != ROW:
            assert e_new['receipt_or_workflow'] == e_old['receipt_or_workflow'], cid
            assert e_new['note'] == e_old['note'].replace(OLD_INV, NEW_INV), cid       # only the inventory parenthetical moved
assert set(changed_fields) == set(pp_rows) | {NODE}, sorted(changed_fields)
assert sorted(changed_fields[ROW]) == ['kernel_evidence', 'next_step', 'not_established', 'statements']
assert changed_fields[NODE] == ['not_established']
for c in reg['claims']:
    for s in c['statements']:
        if s['package'] == STMT_PACKAGE:
            assert s['formal_progress'] == 'specified' and s['kind'] in ('def_prop', 'structure')
        else:
            assert (s['kind'] == 'theorem') == (s['formal_progress'] == 'proved')
        assert s['formal_progress'] != 'kernel-checked'
triaged_after = {c['claim_id'] for c in reg['claims'] if TRIAGE_MARK in c['next_step']}
assert triaged_after == triaged_before and all(rows[c]['next_step'].count(TRIAGE_MARK) == 1 for c in triaged_after)
for cid in triaged_after - {ROW}:
    assert rows[cid]['next_step'].split(TRIAGE_MARK)[1] == old_rows[cid]['next_step'].split(TRIAGE_MARK)[1], cid
# every triage verdict word of the row is kept: the multiset of verdict tokens is unchanged
VERDICTS = ('PROVABLE_NOW', 'PROVABLE_WITH_WORK', 'BLOCKED_MISSING_MATHLIB', 'BLOCKED_INTERFACE', 'PREDICATE_VOCABULARY')
tri_old, tri_new = old_rows[ROW]['next_step'].split(TRIAGE_MARK)[1], ns.split(TRIAGE_MARK)[1]
for v in VERDICTS:   # the README-reasons sentence (after the triage sentence) names BLOCKED_MISSING_MATHLIB once more
    assert tri_new.count(v) == tri_old.count(v) + (1 if v == 'BLOCKED_MISSING_MATHLIB' else 0), v
assert ns.split(TRIAGE_MARK)[0] == old_rows[ROW]['next_step'].split(TRIAGE_MARK)[0].replace(
    targets_clause(without_old, 'Math- Mathlib lane'), targets_clause(without_new, 'Math- Mathlib lane')).replace(
    f'; every theorem of this row has a hosted receipt: the 2 added after PR head {C9} (',
    f'; the {len(T6)} theorems of this row added in round 6 after PR head {C10} ({short(T6)}) await the next hosted '
    f'lean-specification-proofs.yml run, their first receipt; the other theorems of this row have a hosted receipt: the 2 added '
    f'after PR head {C9} (').replace(f'; every theorem of this row receipted on PR head {C10} by run {RUN10}',
                                     f'; each of those {len(row_mine_old)} theorems receipted on PR head {C10} by run {RUN10}')
# the three BLOCKED_MISSING_MATHLIB items of this row each now carry 'proved in round 6', and no other item does
assert tri_new.count('proved in round 6 (2026-10-12; see entries, no hosted receipt yet;') == 3
for p in NEW_PROPS:
    item = tri_new.split(prop_short(p) + ' BLOCKED_MISSING_MATHLIB (weeks;', 1)
    assert len(item) == 2 and 'proved in round 6' in item[1].split(' BLOCKED_')[0], p
for s in strings(reg):
    assert 'PROVABLE_WITH_WORK (weeks); ' not in s and 'PROVABLE_WITH_WORK (weeks).' not in s, s[:120]

# C422-E1: every run id named in any string leaf is bound to its PR head and actual tested merge in that same leaf
for s in strings(reg):
    for run, merge in RUN_MERGE:
        if run in s:
            assert merge in s and RUN_HEAD[run] in s, (run, s[:160])
    for m in re.finditer(r'\b38\d{9}\b', s):
        assert m.group(0) in dict(RUN_MERGE) or m.group(0) == WORKSPACE_LANDING_RUN, (m.group(0), s[:120])
# the round-6 facts appear exactly where planned: 3 PENDING6 phrases (row note, row not_established, node not_established), one WF clause
assert count_all(reg, PENDING6_PHRASE) == 3 and count_all(reg, WF_PENDING6) == 1
assert count_all(reg, 'not evidence about any commit') == count_all(old, 'not evidence about any commit') + 3
assert count_all(reg, 'await the next hosted') == count_all(old, 'await the next hosted') + 1 == 1
# 'no hosted receipt yet': WF_PENDING6, the three triage items and the README-reasons sentence
assert count_all(old, 'no hosted receipt yet') == 0 and count_all(reg, 'no hosted receipt yet') == 5
assert count_all(reg, 'round 6') == 3 + 1 + 3 + 1 + 1, count_all(reg, 'round 6')   # 3 PENDING6, WF, 3 triage items, README, receipt clause
# verifier fix round (2026-10-12): the analytic-only symmetry is never described as the general lemma, every round-6
# description keeps the Props' restriction d = 2 or 3, and every run mention in WF_PENDING6 carries its PR head inline
assert count_all(reg, 'was not missing') == count_all(old, 'was not missing')
assert count_all(reg, 'general version for finitely smooth functions') == 2
assert all(x in ne for x in ('display (2): |D^q K24(0) u - D^q Kinf(0) u| < imageConstE for d = 2 or 3',
                             'the entrywise bound |C24 a b - Cref a b| <= 2 imageConstE for d = 2 or 3',
                             'at u = R 0 in an orthonormal frame for d = 2 or 3 (CovarianceOrdering'))
assert f'run {RUN10} (PR head {C10}; actual tested merge {M10}' in WF_PENDING6
after = census(reg)
assert after == {**before, 'statements': before['statements'] + 3, 'proved': before['proved'] + 3}, (before, after)
kc = json.dumps(reg, ensure_ascii=False)
assert '"formal_progress": "kernel-checked"' not in kc

out = json.dumps(reg, indent=2, ensure_ascii=False) + '\n'
assert json.loads(out) == reg and out != committed
assert json.dumps(old, indent=2, ensure_ascii=False) + '\n' == committed
OUT_REG.write_text(out, encoding='utf-8')
new_blob = hashlib.sha1(b'blob %d\0' % len(out.encode()) + out.encode()).hexdigest()

# ----------------------------------------------------------------------------- log of every changed string leaf
old_leaves, new_leaves = dict(leaves(old)), dict(leaves(reg))
assert set(old_leaves) <= set(new_leaves)
changed = [(p, old_leaves.get(p), v) for p, v in new_leaves.items() if old_leaves.get(p) != v]
log = [f'update_register13: {len(changed)} changed string leaves of {STMT_REL}/REGISTER.json '
       f'(4afcca6a blob {EXPECTED_BLOB} -> worktree blob {new_blob}); census {before} -> {after}', '']
rowname = {f'/claims[{i}]': c['claim_id'] for i, c in enumerate(reg['claims'])}
for p, a, b in changed:
    head_path = re.match(r'/claims\[\d+\]', p).group(0)
    log.append(f'== {p}  (row {rowname[head_path]}){"  [NEW LEAF]" if a is None else ""}')
    if a is None:
        log.append('   + ' + b)
    else:
        aw, bw = a.split(' '), b.split(' ')
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, aw, bw, autojunk=False).get_opcodes():
            if op == 'equal':
                continue
            ctx = ' '.join(aw[max(0, i1 - 6):i1])
            log.append(f'   {op} after "...{ctx}":')
            if i2 > i1:
                log.append('     - ' + ' '.join(aw[i1:i2]))
            if j2 > j1:
                log.append('     + ' + ' '.join(bw[j1:j2]))
    log.append('')
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / 'update_register13.log').write_text('\n'.join(log) + '\n', encoding='utf-8')
(OUT_DIR / 'update_register13_leaves.json').write_text(json.dumps([{'path': p, 'old': a, 'new': b} for p, a, b in changed],
                                                                  indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
summary = {'scientific_effect': 'NONE', 'committed_head': head, 'input_blob': EXPECTED_BLOB, 'output_blob': new_blob, 'output_path': str(OUT_REG),
           'before': before, 'after': after, 'added_entries': added, 'replacements': counts,
           'changed_fields': changed_fields, 'changed_string_leaves': [p for p, _, _ in changed],
           'inventory': {'theorem_count': inv['theorem_count'], 'covered_props': len(inv['covered_props']), 'modules': inv['modules'],
                         'duplicate_proofs': inv['duplicate_proofs'], 'round6_theorems': T6},
           'last_bound_receipt': {'run': RUN10, 'pr': 439, 'pr_head': 'c6ce025652f76be6e580fa24947380253450cd3b', 'actual_tested_merge': M10,
                                  'base': BASE4, 'theorem_count': 75, 'artifact': f'lean-specification-proofs-{RUN10}-1',
                                  'artifact_sha256_github_digest': 'd8abd42be0facd4248647c3c7eb67a264bd19b9b47c6128733b5b5fc93b0a6ff'},
           'claim_level_formal_progress_unchanged': all(rows[c]['formal_progress'] == old_rows[c]['formal_progress'] for c in rows),
           'run_merge_bindings_asserted': dict(RUN_MERGE), 'run_head_bindings_asserted': RUN_HEAD,
           'phrases': {'PENDING6': PENDING6_PHRASE, 'WF_PENDING6': WF_PENDING6, 'IMG_DESC': IMG_DESC, 'EC_DESC': EC_DESC, 'CO_DESC': CO_DESC}}
(OUT_DIR / 'update_summary13.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps({k: v for k, v in summary.items() if k not in ('phrases', 'changed_string_leaves')}, indent=1, ensure_ascii=False))
print(f'{len(changed)} changed string leaves logged to {OUT_DIR / "update_register13.log"}')
