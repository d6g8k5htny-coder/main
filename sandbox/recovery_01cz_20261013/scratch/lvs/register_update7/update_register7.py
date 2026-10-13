"""Round-7 register update: statement entries, four-tier evidence notes and round-4 triage notes for the
round-4 proof-packet theorems (modules SpecificationProofs/{Side24Analysis,Side24Interfaces,RNCluster}.lean).

Derived from update_register5.py (entries, kernel_evidence tiers, row regeneration) and update_register6.py
(tier-3 binding phrases, which are reproduced verbatim here).

Scientific effect: NONE. Adds `proved` statement entries (the label of a committed proof package) for the 13 inventory
theorems not yet listed; regenerates every proof-packet kernel_evidence row with FOUR receipt tiers (run 38116277597,
run 38121713840, run 38149888699 / first 38149524759 as update_register6 wrote them, and "no hosted receipt yet" for
theorems added after PR head 12e072d6, the last head with a hosted receipt: run 38150363063); regenerates the
not_established / next_step texts of rows side24-coefficient, math.side24-coefficient and OA-C6-RARE-CLUSTER-20260929-v1
from the inventory; appends ONE round-4 triage sentence to next_step of every other row of triage_by_row.json. No
claim-level label changes (claim formal_progress stays none/specified). Every mention of a hosted run keeps the C422-E1
binding (PR head AND actual tested merge). Same-account author: organizational-independence credit 0; alignment
PENDING_INDEPENDENT_REVIEW; nonauthor review and acceptance are not claimed.
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

SCRATCH = Path('<SCRATCH>')
STMT = SCRATCH / 'repos/math/specifications/lean_statements_20261010'
PROOF = SCRATCH / 'repos/math/specifications/lean_statement_proofs_20261011'
OUT_DIR = SCRATCH / 'lvs/register_update7'
R5 = SCRATCH / 'lvs/register_update5'
TIERS = json.loads((R5 / 'receipted_52.json').read_text())
T1 = set(TIERS['tier1_run_38116277597_head_00d2ac6e'])
T2 = set(TIERS['tier2_run_38121713840_head_910809d6'])
assert len(T1) == 38 and len(T2) == 14 and not (T1 & T2)
BASE_WF = (R5 / 'base_wf.txt').read_text()
TRIAGE = json.loads((OUT_DIR / 'triage_by_row.json').read_text(encoding='utf-8'))
assert len(TRIAGE) == 30

STMT_PACKAGE = 'specifications/lean_statements_20261010'
PROOF_PACKAGE = 'd6g8k5htny-coder/Math- specifications/lean_statement_proofs_20261011'
BASE = 'b2d70d77876a10c365f5db5d3b6cafcf180af4e2'
BASE2 = '79731b961fac555f33be3cd90e379b246a5a80e3'     # head of PR #422, the base of PR #423 and of every round-3 run
RUN1, C1, M1 = '38116277597', '00d2ac6e', 'c4c66bb8cb26f0cd9bc03d52bb25f1b3925b272c'
RUN2, C2, M2 = '38121713840', '910809d6', '5b198699308e0585904c21d12bf540cf2475d2f0'
RUN3B, C3B, M3B = '38122386459', 'd9b8dd33', '866136cca66b46e6dbd3186b47c9340f7e93b4f6'   # corroboration of the 52 (in BASE_WF)
RUN4, C4, M4 = '38124262242', '79731b96', '318ff194eecc42c39a6b27c1d7bf03f30d6a34d9'   # last receipted head of round 2 (52 theorems)
RUN6, C6, M6, SHA6 = '38149888699', '0dcb0016', '4363b6928c62da242952631c6b0357d096c3b252', 'cdeb5c73aa03124ed0db16fa14d25fb03a8fa842457fce1e0c6565dbc5ee7ca8'
RUNA, C5A, M5A, SHA5A = '38149524759', '88c21058', '3306233bbdeadb15d9b4b07b94cc52a80ed8caeb', 'ddefbdcc63223d05fd8ed34addbeebc199e271525d87dead01109cddc529e836'
RUN7, C7, M7, SHA7 = '38150363063', '12e072d6', '86a6e8c876f7904cadda2d006a56f8792dc5c1af', 'bf0e0515e2035103b07d894e2a73eeb5b542885f6da16b98c228877a5e0c1690'
for r, m in ((RUN1, M1), (RUN2, M2), (RUN3B, M3B), (RUN4, M4), (RUN6, M6), (RUNA, M5A), (RUN7, M7)):
    assert r.isdigit() and re.fullmatch(r'[0-9a-f]{40}', m)
assert all(re.fullmatch(r'[0-9a-f]{64}', s) for s in (SHA6, SHA5A, SHA7))
MIN_ANCHOR = 16
AXIOMS = '{propext, Classical.choice, Quot.sound}'
NEW_MODULES = ('SpecificationProofs/Side24Analysis.lean', 'SpecificationProofs/Side24Interfaces.lean', 'SpecificationProofs/RNCluster.lean')
T3_MODULE = 'SpecificationProofs/P15Interpolation.lean'
TRIAGE_WF = 'wf_a3531fd8-891'

# ---- tier phrases. T1, T2: update_register5 verbatim. T3 (TIER3_PHRASE, WF_TIER3, rc_sub text): update_register6 verbatim.
TIER1_PHRASE = (f'are among the 38 theorems of PR head {C1} kernel-checked by the hosted run {RUN1} (actual tested merge {M1} with base {BASE}; '
                f'job proof-evidence: fresh build, leanchecker PASS, `#print axioms` within {AXIOMS} for each, receipt formalization_status '
                f'kernel-checked, worktree CLEAN; artifact lean-specification-proofs-{RUN1}-1)')
TIER2_PHRASE = (f'were added after commit {C1} and kernel-checked by the hosted run {RUN2} (PR head {C2}; actual tested merge {M2} with base {BASE}) '
                f'(job proof-evidence: fresh build, leanchecker PASS, `#print axioms` within {AXIOMS}, receipt formalization_status kernel-checked, '
                'worktree CLEAN)')
TIER3_PHRASE = (f'were added after PR head {C4} and kernel-checked by the hosted run {RUN6} (PR head {C6}; actual tested merge {M6} with base {BASE2}) '
                f'(job proof-evidence: fresh build, leanchecker PASS, `#print axioms` within {AXIOMS}, receipt formalization_status kernel-checked, '
                f'theorem_count 60, worktree CLEAN; artifact lean-specification-proofs-{RUN6}-1, sha256 {SHA6}); they were first receipted by run {RUNA} '
                f'(PR head {C5A}; actual tested merge {M5A} with base {BASE2}) (same Lean bytes; artifact lean-specification-proofs-{RUNA}-1, sha256 {SHA5A}); '
                f'base {BASE2} is the head of PR #422, and each merge has the same tree as its head')
WF_TIER3 = (f'; the theorems added after PR head {C4} (module P15Interpolation) are receipted by run {RUN6} (PR head {C6}; actual tested merge {M6} with base '
            f'{BASE2}) (artifact lean-specification-proofs-{RUN6}-1, sha256 {SHA6}; receipt formalization_status kernel-checked, leanchecker PASS, '
            f'theorem_count 60, worktree CLEAN) and were first receipted by run {RUNA} (PR head {C5A}; actual tested merge {M5A} with base {BASE2}) '
            f'(artifact lean-specification-proofs-{RUNA}-1, sha256 {SHA5A}); base {BASE2} is the head of PR #422; same trees as the heads')
# the round-3 corroboration on PR head 12e072d6 (FACTS: receipt theorem_count 60, artifact sha256), the last head with a hosted receipt
CORROB_PHRASE = (f'The 60 theorems of PR head {C7} (every theorem of the tiers above) are corroborated by run {RUN7} (actual tested merge {M7} with base '
                 f'{BASE2}; receipt theorem_count 60; artifact lean-specification-proofs-{RUN7}-1, sha256 {SHA7})')
WF_CORROB = (f'; corroborated on PR head {C7} by run {RUN7} (actual tested merge {M7} with base {BASE2}; receipt theorem_count 60; artifact '
             f'lean-specification-proofs-{RUN7}-1, sha256 {SHA7})')
# T4: the task's text, in the style of update_register5's PENDING_PHRASE (backticked commands, full 40-hex base, as the earlier phrases)
PENDING4_PHRASE = (f'were added after PR head {C7} (the last head with a hosted receipt: run {RUN7}, actual tested merge {M7} with base {BASE2}, '
                   f'receipt theorem_count 60): locally `lake env lean -DwarningAsError=true` exit 0 and `#print axioms` within {AXIOMS} on an '
                   'unverified worktree, not evidence about any commit; the next hosted lean-specification-proofs.yml run is the first receipt for them')
WF_PENDING4 = (f'; theorems added after PR head {C7} (run {RUN7}, actual tested merge {M7} with base {BASE2}, receipt theorem_count 60) have no hosted '
               'receipt yet: the next hosted lean-specification-proofs.yml run is their first receipt')
RUN_MERGE = ((RUN1, M1), (RUN2, M2), (RUN3B, M3B), (RUN4, M4), (RUN6, M6), (RUNA, M5A), (RUN7, M7))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pcheck = load_module('proof_check', PROOF / 'check.py')
scheck = load_module('stmt_check', STMT / 'check.py')
inv = pcheck.source_check(PROOF, PROOF.parents[1], allow_dirty=True)
assert inv['formalization_status'] == 'proved'
assert inv['theorem_count'] == 73 and len(inv['covered_props']) == 72, (inv['theorem_count'], len(inv['covered_props']))
assert inv['duplicate_proofs'] == ['UniversalLaw.Spec.P15.TriangleNeedsThree'], inv['duplicate_proofs']
assert all(m in inv['modules'] for m in NEW_MODULES) and T3_MODULE in inv['modules'], inv['modules']
theorems = inv['theorems']
T3 = {t['name'] for t in theorems if t['module'] == T3_MODULE}
T4 = {t['name'] for t in theorems if t['module'] in NEW_MODULES}
assert len(T3) == 8 and len(T4) == 13 and not (T3 & T4) and not ((T1 | T2) & (T3 | T4)), (len(T3), len(T4))
assert {t['name'] for t in theorems} == T1 | T2 | T3 | T4
RECEIPTED = T1 | T2 | T3
LABELS = ('Proves:', 'Method:', 'Does not establish:')


def docstring_fields(module: str, line: int) -> dict:
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
    t['doc_full'] = docstring_fields(t['module'], t['line'])
    assert t['doc_full']['Does not establish'] and t['doc_full']['Proves'].split('`')[1] == t['proves'], t['name']

reg_path = STMT / 'REGISTER.json'
reg = json.loads(reg_path.read_text(encoding='utf-8'))
rows = {c['claim_id']: c for c in reg['claims']}
sources, _ = scheck.load_sources(STMT, None, reg)
before = {'statements': sum(len(c['statements']) for c in reg['claims']),
          'proved': sum(1 for c in reg['claims'] for s in c['statements'] if s['formal_progress'] == 'proved'),
          'kernel_evidence': sum(len(c['kernel_evidence']) for c in reg['claims'])}
listed_before = {s['declaration'] for c in reg['claims'] for s in c['statements'] if s['package'] == PROOF_PACKAGE}
assert listed_before == RECEIPTED, sorted(listed_before ^ RECEIPTED)      # pre-update register: exactly the 60 receipted theorems listed
pp_rows_before = {c['claim_id'] for c in reg['claims'] if any(s['package'] == PROOF_PACKAGE for s in c['statements'])}
for c in reg['claims']:
    assert 'Round-4 triage (' not in c['next_step'], c['claim_id']          # refuse a second run on an already-updated register
    assert c['formal_progress'] in ('none', 'specified')
claim_fp_before = {c['claim_id']: c['formal_progress'] for c in reg['claims']}

# ----------------------------------------------------------------------------- A. new entries
added = []
for t in theorems:
    for cid in t['claim_ids']:
        row = rows[cid]
        if any(s['package'] == PROOF_PACKAGE and s['module'] == t['module'] and s['declaration'] == t['name'] for s in row['statements']):
            continue
        local = [s for s in row['statements'] if s['package'] == STMT_PACKAGE and s['declaration'] == t['proves']]
        assert len(local) == 1, (cid, t['proves'])
        anchor = local[0]['anchor']
        sid = row['layer0']['source_id']
        assert sid and anchor in sources[sid]['text'] and len(anchor) >= MIN_ANCHOR, (cid, t['name'])   # anchor rule: refuse, never weaken
        row['statements'].append({'package': PROOF_PACKAGE, 'module': t['module'], 'declaration': t['name'], 'kind': 'theorem',
                                  'formal_progress': 'proved', 'anchor': anchor, 'does_not_claim': t['doc_full']['Does not establish']})
        added.append((cid, t['name']))
new_names = sorted({n for _, n in added})
assert set(new_names) == T4 and len(added) == 13, (len(added), sorted(T4 ^ set(new_names)))
inv_names = {t['name'] for t in theorems}
for c in reg['claims']:
    for s in c['statements']:
        if s['package'] == PROOF_PACKAGE:
            assert s['declaration'] in inv_names, s['declaration']

# ----------------------------------------------------------------------------- B. kernel_evidence rows regenerated (four tiers)
MODULES_TXT = 'SpecificationProofs/{' + ','.join(Path(m).stem for m in inv['modules']) + '}.lean'
short = lambda ns: ', '.join(n.split('.', 2)[-1] for n in ns)


def tiers(names):
    return ([n for n in names if n in T1], [n for n in names if n in T2], [n for n in names if n in T3], [n for n in names if n in T4])


for cid, row in rows.items():
    mine = [s['declaration'] for s in row['statements'] if s['package'] == PROOF_PACKAGE]
    row['kernel_evidence'] = [e for e in row['kernel_evidence'] if e['package'] != PROOF_PACKAGE]
    if not mine:
        continue
    r1, r2, r3, new = tiers(mine)
    assert len(r1) + len(r2) + len(r3) + len(new) == len(mine)
    note = (f'{len(mine)} theorem(s) of this row ({short(mine)}) are inventoried by the proof packet\'s check.py (source mode '
            f'SOURCE_CHECK_PASS: {inv["theorem_count"]} inventory theorems over {MODULES_TXT}, {len(inv["covered_props"])} distinct registered '
            'Props, each theorem typed exactly as its registered Prop, applied to its own parameters). ')
    if r1:
        note += f'{len(r1)} of them ({short(r1)}) {TIER1_PHRASE}. '
    if r2:
        note += f'{len(r2)} ({short(r2)}) {TIER2_PHRASE}. '
    if r3:
        note += f'{len(r3)} ({short(r3)}) {TIER3_PHRASE}. '
    if r1 or r2 or r3:
        note += CORROB_PHRASE + '. '
    if new:
        note += f'{len(new)} ({short(new)}) {PENDING4_PHRASE}. '
    note += ('`proved` is the label of a committed proof package; `kernel-checked` appears only in a run receipt. Evidence about the '
             'registered Props only: not about the informal claim, its alignment (PENDING_INDEPENDENT_REVIEW), review or acceptance; the '
             'claim-level label is unchanged; organizational-independence credit 0 (Anthropic / Claude).')
    wf = BASE_WF + (WF_TIER3 if r3 else '') + (WF_CORROB if (r1 or r2 or r3) else '') + (WF_PENDING4 if new else '')
    row['kernel_evidence'].append({'package': PROOF_PACKAGE, 'receipt_or_workflow': wf, 'note': note})

# ----------------------------------------------------------------------------- shared helpers for C and D
NO_REVIEW2 = ('Nonauthor review and acceptance are not claimed by any statement or theorem; alignment of every statement with its anchor '
              'is PENDING_INDEPENDENT_REVIEW; organizational-independence credit of this packet and of the proof packet is 0.')
PREDICATES = {'IsRealizedFamily', 'IsGood', 'IsKDecomposable', 'IsGeneratorCover', 'IsPaletteFeasible', 'IsMinimalForbidden',
              'IsTransversal', 'IsProperDownset', 'IsUnitTuple', 'NegDefCone'}
prop_short = lambda n: n.split('.', 3)[-1]
listing = lambda props: ', '.join(prop_short(p) for p in props) if props else 'none'


def row_props(cid):
    row = rows[cid]
    local = [s['declaration'] for s in row['statements'] if s['package'] == STMT_PACKAGE and s['kind'] == 'def_prop']
    proved_props = {}
    for t in theorems:
        if cid in t['claim_ids']:
            proved_props.setdefault(t['proves'], []).append(t['name'])
    return local, proved_props, [p for p in local if p in proved_props], [p for p in local if p not in proved_props]


def evidence_sentence(with_thm, proved_props):
    names = [n for p in with_thm for n in proved_props[p]]
    r1, r2, r3, new = tiers(names)
    s = f'({len(names)} {"entry" if len(names) == 1 else "entries"} below, label `proved`'
    if r1:
        s += f'; {len(r1)} of them ({short(r1)}) {TIER1_PHRASE}'
    if r2:
        s += f'; {len(r2)} of them ({short(r2)}) {TIER2_PHRASE}'
    if r3:
        s += f'; {len(r3)} of them ({short(r3)}) {TIER3_PHRASE}'
    if new:
        s += f'; {len(new)} of them ({short(new)}) {PENDING4_PHRASE}'
    return s + ')'


def split_targets(props):
    return [p for p in props if prop_short(p) not in PREDICATES], [p for p in props if prop_short(p) in PREDICATES]


def remaining_sentence(without):
    targets, preds = split_targets(without)
    s = f'Theorem-shaped Props of this row still `specified` with no theorem: {listing(targets)}'
    if preds:
        s += (f'; the parametrised predicates {listing(preds)} are the vocabulary of the other Props, not closed statements '
              '(a theorem of their type would assert them for every argument, which is false), and are not targets')
    return s + '. '


def targets_clause(without, lane):
    targets, preds = split_targets(without)
    if targets:
        return f'{lane}: {listing(targets)} remain the targets of this row' + (f' (the parametrised predicates {listing(preds)} are vocabulary, not targets)' if preds else '')
    return f'{lane}: every theorem-shaped Prop of this row has a theorem' + (f' (the parametrised predicates {listing(preds)} are vocabulary, not targets)' if preds else '')


def receipt_clause(cid):
    names = [s['declaration'] for s in rows[cid]['statements'] if s['package'] == PROOF_PACKAGE]
    r1, r2, r3, new = tiers(names)
    parts = []
    if new:
        parts.append(f'the {len(new)} theorem(s) of this row added after PR head {C7} ({short(new)}) await the next hosted '
                     'lean-specification-proofs.yml run, their first receipt')
    old = []
    if r1:
        old.append(f'the {len(r1)} of PR head {C1} ({short(r1)}) by run {RUN1} (actual tested merge {M1} with base {BASE})')
    if r2:
        old.append(f'the {len(r2)} added after commit {C1} ({short(r2)}) by run {RUN2} (PR head {C2}; actual tested merge {M2} with base {BASE})')
    if r3:
        old.append(f'the {len(r3)} added after PR head {C4} ({short(r3)}) by run {RUN6} (PR head {C6}; actual tested merge {M6} with base {BASE2}) and first by run {RUNA} (PR head {C5A}; actual tested merge {M5A} with base {BASE2})')
    if old:
        parts.append(('the other theorems of this row have a hosted receipt: ' if new else 'every theorem of this row has a hosted receipt: ')
                     + '; '.join(old) + f'; each of those receipted theorems corroborated on PR head {C7} by run {RUN7} (actual tested merge {M7} with base {BASE2}; receipt theorem_count 60)')
    return '; '.join(parts)


# ---- round-4 triage sentence (D). Verdicts, efforts and the Mathlib entries come from triage_by_row.json; the one-clause
# abbreviations of the BLOCKED_INTERFACE findings are hand-written here from the 'finding' field and checked against it.
INHABITANT = {
    'AnnulusBridgeAllHeight2': 'a Dirac-tilted law at a pinned smooth periodic path with one extra nondegenerate critical point in the annulus violates display (2) for a null Borel E',
    'D5dSummedPinNeighbourhoodC2': 'a Dirac-law inhabitant with one extra nondegenerate critical point at the midpoint gives count >= 1 > C r^3 as r -> 0',
    'D5aPuncturedPinDisks': 'a Dirac-law inhabitant with one extra critical point in the punctured disk and a null E',
    'D5bMicrodisk': 'the same Dirac-law inhabitant with the extra critical point at distance r^2 from M (kappa = 1)',
    'D5cCollarC1': 'a Dirac-law inhabitant with an extra critical point at the midpoint and E = {0} in the collar',
    'D5eIntermediateShellsI3I4': 'a Dirac-law inhabitant with one extra window-height critical point at distance 8r from the midpoint',
    'D5fGlobalWindowFirstMomentI5': 'Dirac-law inhabitants refute both directions (one extra window critical point: probability 1 > C r^3; none: probability 0 < c r^3), and the lower bound is sensitive to the Set.ncard convention (infinitely many window critical points count as 0)',
    'DimensionLiftCompactCollarC': 'a Dirac-law inhabitant with an extra critical point at the midpoint and E = {0}, in every fixed d >= 2',
    'DimensionLiftGlobalWindowG': 'a Dirac-law inhabitant with one extra window-height critical point at the midpoint',
    'DimensionLiftIntermediateShellsI': 'a Dirac-law inhabitant with one extra window-height critical point at distance 8r from the midpoint',
    'DimensionLiftPuncturedPinBallP': 'a Dirac-law inhabitant with one extra critical point in the punctured pin ball and a null E',
    'FixedAnnulusWindowCandidateA2': 'a Dirac inhabitant with an index-j window critical point on a null E0 (a singleton in the annulus); no field of PinnedLawInterface supplies the a.s. absence that the weighted Kac-Rice formula (A21) gives',
    'FixedRemoteCountBoundA3': 'a Dirac inhabitant with one extra critical point on a null remote set fails both displays; the interface transcribes neither the Kac-Rice representation (12) nor any regularity of Q_r^W',
    'FixedRemoteMeanMeasureA2': 'the kernel clause is a conclusion about the interface field I.Lambda, which no field constrains; with Lambda := 0 the clause 0 < cLambda <= I.Lambda fails at any remote point',
    'FourierCutoffPlanarSecondFactorialF4': 'the antecedent G_2 plus the interface fields admit a rare large cluster (two-point law, cluster size >= 1/r with probability r^3/n_r) with E N(N-1) of order r^2 while E N = r^3; Theorem W\'s Gaussian input is not transcribed',
    'PalmRouteFactorialMomentsQ': 'a Dirac inhabitant whose single path has two window critical points off the pins gives E (N)_2 = 2 for all small r',
    'PalmRouteSecondFactorialTheta': 'the lower bounds c r^3 fail for an inhabitant with no window critical points off the pins and for one with infinitely many (windowCountAll uses Set.ncard, 0 on infinite sets)',
    'RemoteCollisionSecondFactorialD': 'a Dirac inhabitant with two prescribed remote critical points falsifies both displays; no field gives a.s. absence of index-j window pairs on null remote sets',
    'WitnessCollisionNearPinPairsTarget': 'a Dirac inhabitant with two window critical points within eta of a pin gives expectation 2 for all small r; consistent with the node\'s OPEN status',
    'ParentUniformSelectionA': 'the falsifying inhabitant p == 0 named in the docstring, confirmed by elaboration (probe Lifetime-B_1)',
    'PlanarElderLowerE2': 'the falsifying inhabitant p == 1 named in the docstring, confirmed by elaboration (probe Lifetime-B_1)',
    'PlanarTwoSidedRateE4': 'p == 1 satisfies the hypothesis and the lower half fails (probe Lifetime-B_1)',
    'CoefficientEnclosure': 'probe-level, not a receipt: (forall I, CoefficientEnclosure I) -> IsEmpty PeriodicCoefficientInterface elaborates, by doubling coneMoment and c24, so no theorem of its type can exist unless the interface is empty',
    'ReferenceComparison': 'probe-level, not a receipt: (forall I, ReferenceComparison I) -> IsEmpty PeriodicCoefficientInterface elaborates likewise',
}
INHABITANT_SHORT = {   # used when a row carries three or more BLOCKED_INTERFACE Props
    'AnnulusBridgeAllHeight2': 'Dirac-tilted law with one extra nondegenerate critical point in the annulus, null E',
    'D5dSummedPinNeighbourhoodC2': 'extra critical point at the midpoint, count >= 1 > C r^3',
    'D5aPuncturedPinDisks': 'extra critical point in the punctured disk, null E',
    'D5bMicrodisk': 'extra critical point at distance r^2 from M',
    'D5cCollarC1': 'extra critical point at the midpoint, E = {0}',
    'D5eIntermediateShellsI3I4': 'extra window-height critical point at distance 8r from the midpoint',
    'D5fGlobalWindowFirstMomentI5': 'one or no extra window critical point refutes either direction; the lower bound is Set.ncard-sensitive',
    'DimensionLiftCompactCollarC': 'extra critical point at the midpoint, E = {0}, every d >= 2',
    'DimensionLiftGlobalWindowG': 'extra window-height critical point at the midpoint',
    'DimensionLiftIntermediateShellsI': 'extra window-height critical point at distance 8r from the midpoint',
    'DimensionLiftPuncturedPinBallP': 'extra critical point in the punctured pin ball, null E',
    'FixedAnnulusWindowCandidateA2': 'index-j window critical point on a null singleton E0; no field gives the a.s. absence of Kac-Rice (A21)',
    'FixedRemoteCountBoundA3': 'extra critical point on a null remote set; Kac-Rice (12) not transcribed',
    'FixedRemoteMeanMeasureA2': 'the free field I.Lambda := 0 fails the kernel clause 0 < cLambda <= I.Lambda',
    'FourierCutoffPlanarSecondFactorialF4': 'rare large cluster compatible with G_2 and the fields; Theorem W not transcribed',
    'PalmRouteFactorialMomentsQ': 'two window critical points off the pins, E (N)_2 = 2 for all small r',
    'PalmRouteSecondFactorialTheta': 'no window critical points off the pins, or infinitely many (Set.ncard gives 0)',
    'RemoteCollisionSecondFactorialD': 'two prescribed remote critical points falsify both displays',
    'WitnessCollisionNearPinPairsTarget': 'two window critical points within eta of a pin, expectation 2',
    'ParentUniformSelectionA': 'p == 0 (docstring inhabitant, probe Lifetime-B_1)',
    'PlanarElderLowerE2': 'p == 1 (docstring inhabitant, probe Lifetime-B_1)',
    'PlanarTwoSidedRateE4': 'p == 1 satisfies the hypothesis, lower half fails',
    'CoefficientEnclosure': 'probe-level: universal closure implies IsEmpty PeriodicCoefficientInterface',
    'ReferenceComparison': 'probe-level: universal closure implies IsEmpty PeriodicCoefficientInterface',
}
# a key phrase of each hand-written clause that must occur in the triage 'finding' it abbreviates (guards against drift)
INHABITANT_KEY = {
    'AnnulusBridgeAllHeight2': 'Dirac tilted law', 'D5dSummedPinNeighbourhoodC2': 'midpoint', 'D5aPuncturedPinDisks': 'punctured disk',
    'D5bMicrodisk': 'r^2 from M', 'D5cCollarC1': 'E = {0}', 'D5eIntermediateShellsI3I4': '8r from the midpoint',
    'D5fGlobalWindowFirstMomentI5': 'both directions', 'DimensionLiftCompactCollarC': 'E = {0}', 'DimensionLiftGlobalWindowG': 'midpoint',
    'DimensionLiftIntermediateShellsI': '8r from the midpoint', 'DimensionLiftPuncturedPinBallP': 'punctured pin ball',
    'FixedAnnulusWindowCandidateA2': '(A21)', 'FixedRemoteCountBoundA3': 'Kac-Rice representation (12)', 'FixedRemoteMeanMeasureA2': 'Λ := 0',
    'FourierCutoffPlanarSecondFactorialF4': 'rare large cluster', 'PalmRouteFactorialMomentsQ': 'E (N)_2 = 2',
    'PalmRouteSecondFactorialTheta': 'infinitely many', 'RemoteCollisionSecondFactorialD': 'two prescribed remote critical points',
    'WitnessCollisionNearPinPairsTarget': 'expectation `2`', 'ParentUniformSelectionA': 'p ≡ 0', 'PlanarElderLowerE2': 'p ≡ 1',
    'PlanarTwoSidedRateE4': 'p ≡ 1', 'CoefficientEnclosure': 'IsEmpty PeriodicCoefficientInterface', 'ReferenceComparison': 'IsEmpty PeriodicCoefficientInterface',
}
EXTRA_NOTE = {   # short clauses from 'finding' for Props whose finding records a shape observation beyond the verdict
    'PlanarDensityGapE11': 'its hypothesis ParentUniformSelectionA is inert (no PairLawInterface field ties p to G or I; p == 1 satisfies it), so a theorem of E11 would be a theorem of its conclusion alone',
    'ImageBound': 'no local (open-set) iteratedFDeriv_tsum at Mathlib d13f23b7; the global version is inapplicable to lattice translates',
    'CovarianceOrdering': 'blockers: local term-wise iterated differentiation of the lattice sum, the d-dimensional Gaussian tail bound and higher-order symmetry of iteratedFDeriv; display (3) needs only the crude norm_iteratedFDeriv_comp_le, with a factor about 8 to spare',
}
PROVED_PROPS = {t['proves'] for t in theorems}
VERDICT_ORDER = ('PROVABLE_NOW', 'PROVABLE_WITH_WORK', 'BLOCKED_MISSING_MATHLIB', 'BLOCKED_INTERFACE', 'PREDICATE_VOCABULARY')


def ml_short(entry: str) -> str:
    """'Namespace.lemma (Mathlib/.../File.lean:line) -- remark' -> 'lemma (File.lean)' (last two dotted components of the name)."""
    name = re.split(r'\s+[(—-]|\s+and\s', entry.strip(), 1)[0].strip()
    parts = name.split('.')
    name = '.'.join(parts[-2:]) if len(parts) >= 2 and len(parts[-1]) <= 4 else parts[-1]   # 'Measure.map', else the bare lemma name
    m = re.search(r'([A-Za-z0-9_]+\.lean)', entry)
    return f'{name} ({m.group(1)})' if m else name


def triage_sentence(cid):
    props = TRIAGE[cid]
    by_prop = {}
    for p in props:
        assert p['verdict'] in VERDICT_ORDER, p
        e = by_prop.setdefault(p['prop'], {'verdict': p['verdict'], 'effort': p['effort'], 'mathlib': p['mathlib'], 'finding': p['finding'], 'modules': []})
        assert e['verdict'] == p['verdict'], (cid, p['prop'])
        e['modules'].append(Path(p['module']).stem)
    items = []
    proved_now = [p for p, e in by_prop.items() if e['verdict'] == 'PROVABLE_NOW' and p in PROVED_PROPS]
    proved_work = [p for p, e in by_prop.items() if e['verdict'] == 'PROVABLE_WITH_WORK' and p in PROVED_PROPS]
    assert all(p in PROVED_PROPS for p, e in by_prop.items() if e['verdict'] == 'PROVABLE_NOW'), cid     # every PROVABLE_NOW Prop was proved in round 4
    if proved_now or proved_work:
        s = []
        if proved_now:
            s.append(f'{listing(proved_now)} PROVABLE_NOW')
        if proved_work:
            s.append(f'{listing(proved_work)} PROVABLE_WITH_WORK')
        items.append(' and '.join(s) + ': proved in round 4 (see entries)')
    for p, e in by_prop.items():
        if e['verdict'] == 'PROVABLE_WITH_WORK' and p not in PROVED_PROPS:
            items.append(f'{prop_short(p)} PROVABLE_WITH_WORK ({e["effort"]})')
    for p, e in by_prop.items():
        if e['verdict'] == 'BLOCKED_MISSING_MATHLIB':
            ml = ', '.join(ml_short(x) for x in e['mathlib'][:2])
            extra = EXTRA_NOTE.get(prop_short(p))
            items.append(f'{prop_short(p)} BLOCKED_MISSING_MATHLIB ({e["effort"]}; Mathlib d13f23b7: {ml}' + (f'; {extra}' if extra else '') + ')')
    blocked = [prop_short(p) for p, e in by_prop.items() if e['verdict'] == 'BLOCKED_INTERFACE']
    for p, e in by_prop.items():
        if e['verdict'] == 'BLOCKED_INTERFACE':
            k = prop_short(p)
            assert k in INHABITANT and k in INHABITANT_SHORT and INHABITANT_KEY[k] in e['finding'], (cid, k)
    if len(blocked) >= 3:   # three or more on one row: one grouped item with the short clauses, to stay near the length budget
        assert all(set(by_prop[p]['modules']) == {'RN'} for p in by_prop if prop_short(p) in blocked), cid   # the Dirac-law header is the RN finding
        items.append('BLOCKED_INTERFACE (interface under-determination, not statement defects; each refuted by a concrete Dirac-law inhabitant): '
                     + ', '.join(f'{k} ({INHABITANT_SHORT[k]})' for k in blocked))
    else:
        clause = INHABITANT if len(by_prop) < 8 else INHABITANT_SHORT          # a crowded row gets the short clauses
        items.extend(f'{k} BLOCKED_INTERFACE ({clause[k]})' for k in blocked)
    preds, pred_modules = [], {}
    for p, e in by_prop.items():
        if e['verdict'] == 'PREDICATE_VOCABULARY':
            k = prop_short(p)
            if k not in preds:
                preds.append(k)
            pred_modules.setdefault(k, set()).update(e['modules'])      # the Lifetime and Field copies of a predicate share a short name
    if preds:
        both = any(len(pred_modules[k]) > 1 for k in preds)
        items.append(', '.join(preds) + (' (Lifetime and Field copies)' if both else '') + ' PREDICATE_VOCABULARY: vocabulary of the other Props, not a target')
    return f'Round-4 triage (2026-10-11, same-account, credit 0, no label moves): ' + '; '.join(items) + '.'


# ----------------------------------------------------------------------------- C. the three rows regenerated
S24 = 'UniversalLaw.Spec.Side24.'
local, pp, with_thm, without = row_props('side24-coefficient')
assert len(local) == 26 and len(with_thm) == 17 and len(without) == 9, (len(local), len(with_thm), len(without))
assert all(S24 + n in pp for n in ('ThetaSummable', 'DerivativePairingBound', 'DerivativeBoundAtLeastOne', 'DerivativeBoundAtZero', 'LatticeImageSumBound',
                                   'ReferenceOddAndPinBlocks', 'ReferencePinDensityProduct', 'IntegrandRatioBounds', 'DensityComparison',
                                   'ConeMomentD1Identity', 'TraceMoments', 'RayleighSquareLaw'))
assert all(S24 + n in without for n in ('ImageBound', 'CovarianceOrdering', 'EntrywiseCovarianceBound', 'ReferenceCovarianceLowerBound',
                                        'ConeMomentD2Identity', 'CoefficientEnclosure', 'ReferenceComparison', 'IsUnitTuple', 'NegDefCone'))
INTERFACE_FINDING = ('The interface does not determine c_{d,24}: CoefficientEnclosure and ReferenceComparison are predicates on the interface object, '
                     'not closed statements; their universal closure over all inhabitants is false whenever the interface is inhabited (scaling the '
                     'given cone moment by a positive constant scales c24 through eq_15_2 and gives another inhabitant), and `specified` refers to the '
                     'statement shape at the source\'s object, which is not constructed. The round-4 triage elaborated this at probe level (a scratch '
                     'file, not a receipt and not in any packet): (forall I, CoefficientEnclosure I) -> IsEmpty PeriodicCoefficientInterface, and '
                     'likewise for ReferenceComparison, by doubling coneMoment and c24; the label stays `specified` and no statement is edited.')
rows['side24-coefficient']['not_established'] = (
    'Nothing here encloses the coefficient. CoefficientEnclosure and ReferenceComparison are statements for any object satisfying '
    'PeriodicCoefficientInterface; the interface takes the cone moment D_u of (15.1) as a GIVEN field, so c_{d,24} is not uniquely determined in '
    'Lean and no inhabitant is constructed. The 27 main theorems listed as `proved` are the integer/rational arithmetic skeleton of PROOF.md '
    'sections 1-4 (coefficient sums, shell constants, Taylor partial sum, block minors, cone-moment algebra), not the Gaussian conditioning, the '
    'Rayleigh cone integration, the Stirling log-gamma step (section 5, not specified), the interval arithmetic of coefficient.py, or the '
    'enclosure itself. Props of this row with a theorem of exactly their type in the proof packet specifications/lean_statement_proofs_20261011: '
    f'{listing(with_thm)} {evidence_sentence(with_thm, pp)}. What those theorems prove, for the formal objects of the statement module only: '
    'elementary real inequalities (ExponentialBounds), the term ratio (SuccessiveTermRatio), the D2 cone-moment algebra (ConeMomentD2Algebra), the '
    'reference-coefficient identity as stated (ReferenceCoefficientIdentity; its left side is not identified with the parent formula (15.2) at the '
    'reference law: rotational invariance of K_infty is not specified) and the elementary cone integral (ElementaryConeIntegral); summability of '
    'the formal theta family n |-> exp(-|z + 24 n|^2/2) over Fin d -> Z for every d (ThetaSummable: the statement entry\'s does_not_claim "that the '
    'family is summable (true, but not shown here)" describes the statement packet, which is not edited; the summability is now shown in the proof '
    'packet, for the formal family, by comparison with a product of one-dimensional Gaussian sums); the Section 2 derivative bounds for q <= 6 as '
    'the Props state them (DerivativePairingBound, DerivativeBoundAtLeastOne, DerivativeBoundAtZero: a Hermite-type recursion on the operator norms '
    'of iteratedFDeriv, no closed form of the Gaussian derivatives, the contraction at a unit tuple bounded by the operator norm; main\'s '
    'pairing_sum_six, pairing_sum_le_six and sixth_moment_double_factorial are neither cited nor re-proved; nothing for q > 6); the lattice image '
    'sum (LatticeImageSumBound: summability and the bound 1458 e^(-288), via 432 e^(-288); the source\'s shell count 729 sum_j j^9 e^(-288 j^2) is '
    'not reproduced and not specified); the five reference block identities and the pin density product for the formal matrices '
    '(ReferenceOddAndPinBlocks, ReferencePinDensityProduct: derivative contractions of Kinf at 0 in an orthonormal frame; none of these matrices is '
    'shown to be the covariance of any random vector or field); pure real arithmetic in epsilon and the exponents a, b for d = 2, 3 '
    '(IntegrandRatioBounds; that the source\'s integrand ratio equals these power quotients is not specified); Loewner determinant and inverse '
    'monotonicity for positive-definite real matrices (DensityComparison; the cone-moment consequence (1 +/- epsilon)^m D0 and the scaling a^m are '
    'not specified); and, over ConeMomentInterface, the m = 1 cone moment 4/3, the trace moments 5/3, 25/3, sqrt(3/8) and the distribution '
    'function P(R^2 <= a) = integral_0^a e^(-z/2)/2 dz for a >= 0 (ConeMomentD1Identity, TraceMoments, RayleighSquareLaw: proved for every '
    'inhabitant from the law fields A_law, sxy_law and isProb alone; no inhabitant is constructed, the laws are not derived as the conditional law '
    'A | V = 0 of the reference field, and the Rayleigh law is proved as a distribution function, not as a density). None of them is the Gaussian '
    'conditioning, ConeMomentD2Identity, ImageBound, the covariance ordering or any enclosure of the coefficient, and the claim-level label stays '
    '`specified` (claim-level vocabulary none/specified; a theorem of a Prop is evidence about that Prop, for the formal objects, not acceptance of '
    'the informal theorem; alignment PENDING_INDEPENDENT_REVIEW). ' + remaining_sentence(without)
    + 'The persistence interpretation through the parent Theorem C (eq. 15.2) is conditional on the parent and untouched. Disposition '
    'REVIEWED_SCOPED is transcribed, not re-derived. ' + NO_REVIEW2 + ' ' + INTERFACE_FINDING)
rows['side24-coefficient']['next_step'] = (
    targets_clause(without, 'Math- Mathlib lane') + '; ' + receipt_clause('side24-coefficient') + '. ' + triage_sentence('side24-coefficient')
    + ' The enclosure needs a proof-producing interval pipeline for Gamma(7/6), pi, exp and the theta sums (the numerical formalizer work offer of '
    'main docs/FORMAL_VERIFICATION.md) and an interface that determines c_{d,24} (the cone moment is a given field; ConeMomentD2Identity for the '
    'reference law under ConeMomentInterface is the open Rayleigh-integration step). Proof packet README: the informal reasons of its groups '
    '"Side24 derivative bounds", "Side24 lattice sums", "Side24 covariance and density" and "Side24 interface statements" were found not '
    'load-bearing by the triage and are superseded by the theorems above (the proof-packet maintainers\' regeneration, not this register\'s). '
    'Independent reviewer: alignment verdicts on the 28 Side24 statements, in particular of the 17 proved Props with their anchors (ThetaSummable '
    'with the source\'s R^d, 24 Z^d and K_24; the derivative Props with Section 2; the interface Props with the source\'s A ~ N(0, 8/3), s ~ N(0, 5/3), '
    'x, y ~ N(0, 1)).')

local, pp, with_thm, without = row_props('math.side24-coefficient')
assert not with_thm and len(without) == 2
land = rows['side24-coefficient']
land_pp = [s['declaration'] for s in land['statements'] if s['package'] == PROOF_PACKAGE]
l1, l2, l3, lnew = tiers(land_pp)
assert len(land_pp) == 17 and len(l1) == 5 and not l2 and not l3 and len(lnew) == 12
rows['math.side24-coefficient']['not_established'] = (
    'See row side24-coefficient: the enclosure is stated for any object satisfying PeriodicCoefficientInterface and is not proved; the '
    '"evaluates_eq_15_2" edge to the parent is represented only by the interface field that takes (15.2) as a hypothesis. No finite-radius band. '
    f'Of the three statements of this node, none has a theorem; stated, not proved: {listing(without)}. The 27 arithmetic theorems of main and the '
    f'{len(land_pp)} proof-packet theorems of the landing row ({short(land_pp)}; none of them the enclosure; {len(l1)} of them ({short(l1)}) '
    f'kernel-checked by the hosted run {RUN1} (PR head {C1}; actual tested merge {M1} with base {BASE}); {len(lnew)} of them ({short(lnew)}) '
    f'{PENDING4_PHRASE}) are listed on the landing row, never here and never as the node; the interface theorems among them (ConeMomentD1Identity, '
    'TraceMoments, RayleighSquareLaw) hold for every inhabitant of ConeMomentInterface from its law fields alone and construct nothing. '
    + NO_REVIEW2 + ' ' + INTERFACE_FINDING)
rows['math.side24-coefficient']['next_step'] = 'As for row side24-coefficient. ' + triage_sentence('math.side24-coefficient')

RNP = 'UniversalLaw.Spec.RN.RareClusterPoissonObstructionO'
local, pp, with_thm, without = row_props('OA-C6-RARE-CLUSTER-20260929-v1')
assert local == [RNP] and with_thm == [RNP] and not without and pp[RNP] == ['UniversalLaw.Proofs.RNCluster.rareClusterPoissonObstructionO']
assert rows['OA-C6-RARE-CLUSTER-20260929-v1']['formal_progress'] == 'none'
rows['OA-C6-RARE-CLUSTER-20260929-v1']['not_established'] = (
    'Only Theorem O (3.1) is stated, for a family of laws with hypotheses (pmf, summable mean, H1, H2) on 0 < r <= r0; Theorems C (compound '
    'Poisson (4.1)), S (6.1), the replica limit (7.1), display (5.2) and the addendum (A1)-(A2) are not stated, so the packet as a whole is not '
    'specified and the claim-level label stays `none` (the register admits only none/specified at claim level, and this row\'s claim is the packet, '
    f'not Theorem O alone). RareClusterPoissonObstructionO has a theorem of exactly its type in the proof packet '
    f'specifications/lean_statement_proofs_20261011 {evidence_sentence(with_thm, pp)}: it proves Theorem O\'s inequality as the Prop states it, for '
    'every family mu and constants a, A satisfying the Prop\'s own antecedents (nonnegativity, summability and total mass one of each mu r, summable '
    'mean, (H1), (H2) on (0, r0]) and no others: the upper half with the witness lam = 0 (the Prop\'s existential form, not the source\'s infimum '
    'verbatim), the lower half through tail events {n >= 2} and {n >= 1}; 2a <= A is derived internally from the antecedents at r0, as in the '
    'source, and is neither a hypothesis nor a conclusion of the Prop. That theorem is evidence about the Prop, for the formal objects pmfTV, '
    'poissonPMF and the family mu : R -> N -> R, not acceptance of Theorem O; alignment of those objects with the source\'s dTV, Pois(lambda) and '
    'Law N_r is PENDING_INDEPENDENT_REVIEW. H1/H2/Hq are not revalidated and are not asserted for the Gaussian window count (the Prop does not tie '
    'mu to any field; the source takes them from [D5] Theorem G_d and [C6] Corollary Theta); the mean-matched statement lambda = mu, Theorems C '
    'and S, the factorial-moment matching, the independent-replica limit (7.1) and the conditional-law liminf remark are neither stated nor proved. '
    'For A < 2a the hypotheses are unsatisfiable (the source\'s own remark), so the Prop is vacuous there by the mathematics; the hypotheses imply '
    '2a <= A, so A > 0 is derivable and the Prop is not vacuous in general (the triage\'s example mu r = (1 - r^3) delta_0 + r^3 delta_2 with a = 1, '
    'A = 2 satisfies the antecedents for r <= 1). Disposition: none transcribed (disposition_transcribed null; the title\'s "PROOF_INDEX reviewed '
    'scoped result" is the source\'s wording, not a status decided here). ' + NO_REVIEW2)
rows['OA-C6-RARE-CLUSTER-20260929-v1']['next_step'] = (
    'State Theorems C and S and (7.1) over pmfs on N (Mathlib PMF or explicit functions N -> R); ' + receipt_clause('OA-C6-RARE-CLUSTER-20260929-v1')
    + '. ' + triage_sentence('OA-C6-RARE-CLUSTER-20260929-v1')
    + ' Independent reviewer: alignment of RareClusterPoissonObstructionO with Theorem O (3.1), in particular the existential form of the upper half '
    'and the (H1), (H2) antecedents on (0, r0].')

# ----------------------------------------------------------------------------- D. the other triage rows: one sentence appended to next_step
C_ROWS = ('side24-coefficient', 'math.side24-coefficient', 'OA-C6-RARE-CLUSTER-20260929-v1')
triage_lengths = {}
for cid in TRIAGE:
    if cid in C_ROWS:
        continue
    sent = triage_sentence(cid)
    assert len(sent) <= (1000 if len(TRIAGE[cid]) > 20 else 740), (cid, len(sent), sent)   # ~700 budget; the 27-Prop lifetime row must name every Prop (reported)
    triage_lengths[cid] = len(sent)
    rows[cid]['next_step'] = rows[cid]['next_step'].rstrip() + ' ' + sent
assert len(triage_lengths) == 27

# ----------------------------------------------------------------------------- E. invariants and write
for c in reg['claims']:
    assert c['formal_progress'] in ('none', 'specified') and c['formal_progress'] == claim_fp_before[c['claim_id']]
    for s in c['statements']:
        if s['package'] == STMT_PACKAGE:
            assert s['formal_progress'] == 'specified' and s['kind'] in ('def_prop', 'structure')
        else:
            assert (s['kind'] == 'theorem') == (s['formal_progress'] == 'proved')


def strings(o):
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from strings(v)


for s in strings(reg):
    for run, merge in RUN_MERGE:
        if run in s:
            assert merge in s, (run, s[:120])
    assert not re.search(r'\b381\d{8}\b', re.sub('|'.join(r for r, _ in RUN_MERGE), '', s)), s[:120]   # no run id outside the binding list
for cid in TRIAGE:
    assert rows[cid]['next_step'].count('Round-4 triage (2026-10-11') == 1, cid
after = {'statements': sum(len(c['statements']) for c in reg['claims']),
         'proved': sum(1 for c in reg['claims'] for s in c['statements'] if s['formal_progress'] == 'proved'),
         'kernel_evidence': sum(len(c['kernel_evidence']) for c in reg['claims'])}
pp_rows_after = {c['claim_id'] for c in reg['claims'] if any(s['package'] == PROOF_PACKAGE for s in c['statements'])}
assert pp_rows_after - pp_rows_before == {'OA-C6-RARE-CLUSTER-20260929-v1'}, sorted(pp_rows_after ^ pp_rows_before)   # the one row that newly has a proof-packet theorem
assert after['statements'] == before['statements'] + 13 and after['proved'] == before['proved'] + 13
assert after['kernel_evidence'] == before['kernel_evidence'] + len(pp_rows_after - pp_rows_before), (before, after)
reg_path.write_text(json.dumps(reg, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
pp_rows = {cid: dict(zip(('T1', 'T2', 'T3', 'T4'), map(len, tiers([s['declaration'] for s in row['statements'] if s['package'] == PROOF_PACKAGE]))))
           for cid, row in rows.items() if any(s['package'] == PROOF_PACKAGE for s in row['statements'])}
summary = {'scientific_effect': 'NONE', 'before': before, 'after': after, 'added_entries': len(added), 'new_theorems': new_names,
           'rows_with_additions': sorted({c for c, _ in added}, key=list(rows).index),
           'tiers': {'T1_run_38116277597': len(T1), 'T2_run_38121713840': len(T2), 'T3_run_38149888699_first_38149524759': len(T3),
                     'T4_no_hosted_receipt_after_head_12e072d6': len(T4)},
           'kernel_evidence_rows_regenerated': pp_rows,
           'rows_regenerated': list(C_ROWS), 'triage_sentence_rows': triage_lengths,
           'claim_level_formal_progress_unchanged': all(rows[c]['formal_progress'] == claim_fp_before[c] for c in rows),
           'run_merge_bindings_asserted': dict(RUN_MERGE),
           'phrases': {'T4': PENDING4_PHRASE, 'WF_T4': WF_PENDING4, 'corroboration': CORROB_PHRASE, 'WF_corroboration': WF_CORROB},
           'inventory': {'theorem_count': inv['theorem_count'], 'covered_props': len(inv['covered_props']),
                         'duplicate_proofs': inv['duplicate_proofs'], 'modules': inv['modules']}}
(OUT_DIR / 'update_summary7.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding='utf-8')
print(json.dumps(summary, indent=1, ensure_ascii=False))
