"""Round-5 register update: statement entries and evidence notes for the round-3 proof-packet theorems
(Theorem F's hazard interpolation and the F3 enclosure, module SpecificationProofs/P15Interpolation.lean).

Scientific effect: NONE. Adds `proved` statement entries (the label of a committed proof package) for inventory
theorems not yet listed; regenerates every proof-packet kernel_evidence row with three receipt tiers (run 38116277597,
run 38121713840, and "no hosted receipt yet" for theorems added after PR head 79731b96); regenerates the two
p15-full-price rows' not_established / next_step texts from the inventory. No claim-level label changes. Every
mention of a hosted run keeps the C422-E1 binding (PR head and actual tested merge).
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

SCRATCH = Path('<SCRATCH>')
STMT = SCRATCH / 'repos/math/specifications/lean_statements_20261010'
PROOF = SCRATCH / 'repos/math/specifications/lean_statement_proofs_20261011'
OUT_DIR = SCRATCH / 'lvs/register_update5'
TIERS = json.loads((OUT_DIR / 'receipted_52.json').read_text())
T1 = set(TIERS['tier1_run_38116277597_head_00d2ac6e'])
T2 = set(TIERS['tier2_run_38121713840_head_910809d6'])
assert len(T1) == 38 and len(T2) == 14 and not (T1 & T2)
BASE_WF = (OUT_DIR / 'base_wf.txt').read_text()

STMT_PACKAGE = 'specifications/lean_statements_20261010'
PROOF_PACKAGE = 'd6g8k5htny-coder/Math- specifications/lean_statement_proofs_20261011'
BASE = 'b2d70d77876a10c365f5db5d3b6cafcf180af4e2'
RUN1, C1, M1 = '38116277597', '00d2ac6e', 'c4c66bb8cb26f0cd9bc03d52bb25f1b3925b272c'
RUN2, C2, M2 = '38121713840', '910809d6', '5b198699308e0585904c21d12bf540cf2475d2f0'
RUN4, C4, M4 = '38124262242', '79731b96', '318ff194eecc42c39a6b27c1d7bf03f30d6a34d9'   # last receipted head (52 theorems)
MIN_ANCHOR = 16
AXIOMS = '{propext, Classical.choice, Quot.sound}'
TIER1_PHRASE = (f'are among the 38 theorems of PR head {C1} kernel-checked by the hosted run {RUN1} (actual tested merge {M1} with base {BASE}; '
                f'job proof-evidence: fresh build, leanchecker PASS, `#print axioms` within {AXIOMS} for each, receipt formalization_status '
                f'kernel-checked, worktree CLEAN; artifact lean-specification-proofs-{RUN1}-1)')
TIER2_PHRASE = (f'were added after commit {C1} and kernel-checked by the hosted run {RUN2} (PR head {C2}; actual tested merge {M2} with base {BASE}) '
                f'(job proof-evidence: fresh build, leanchecker PASS, `#print axioms` within {AXIOMS}, receipt formalization_status kernel-checked, '
                'worktree CLEAN)')
PENDING_PHRASE = (f'were added after PR head {C4} (the last head with a hosted receipt: run {RUN4}, actual tested merge {M4} with base {BASE}, '
                  f'receipt theorem_count 52): locally `lake env lean -DwarningAsError=true` exit 0 and `#print axioms` within {AXIOMS} on an '
                  'unverified worktree, not evidence about any commit; the next hosted lean-specification-proofs.yml run is the first receipt for them')
WF_PENDING = (f'; theorems added after PR head {C4} (run {RUN4}, actual tested merge {M4} with base {BASE}, receipt theorem_count 52) have no hosted '
              'receipt yet: the next hosted lean-specification-proofs.yml run is their first receipt')


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
theorems = inv['theorems']
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
    assert t['doc_full']['Does not establish']

reg_path = STMT / 'REGISTER.json'
reg = json.loads(reg_path.read_text(encoding='utf-8'))
rows = {c['claim_id']: c for c in reg['claims']}
sources, _ = scheck.load_sources(STMT, None, reg)
before = {'statements': sum(len(c['statements']) for c in reg['claims']),
          'proved': sum(1 for c in reg['claims'] for s in c['statements'] if s['formal_progress'] == 'proved'),
          'kernel_evidence': sum(len(c['kernel_evidence']) for c in reg['claims'])}
listed_before = {s['declaration'] for c in reg['claims'] for s in c['statements'] if s['package'] == PROOF_PACKAGE}
assert listed_before == T1 | T2, sorted(listed_before ^ (T1 | T2))

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
        assert sid and anchor in sources[sid]['text'] and len(anchor) >= MIN_ANCHOR, (cid, t['name'])
        row['statements'].append({'package': PROOF_PACKAGE, 'module': t['module'], 'declaration': t['name'], 'kind': 'theorem',
                                  'formal_progress': 'proved', 'anchor': anchor, 'does_not_claim': t['doc_full']['Does not establish']})
        added.append((cid, t['name']))
new_names = sorted({n for _, n in added})
assert all(n not in T1 | T2 for n in new_names)
inv_names = {t['name'] for t in theorems}
for c in reg['claims']:
    for s in c['statements']:
        if s['package'] == PROOF_PACKAGE:
            assert s['declaration'] in inv_names, s['declaration']

# ----------------------------------------------------------------------------- B. kernel_evidence rows regenerated
MODULES_TXT = 'SpecificationProofs/{' + ','.join(Path(m).stem for m in inv['modules']) + '}.lean'
short = lambda ns: ', '.join(n.split('.', 2)[-1] for n in ns)
for cid, row in rows.items():
    mine = [s['declaration'] for s in row['statements'] if s['package'] == PROOF_PACKAGE]
    row['kernel_evidence'] = [e for e in row['kernel_evidence'] if e['package'] != PROOF_PACKAGE]
    if not mine:
        continue
    r1 = [n for n in mine if n in T1]
    r2 = [n for n in mine if n in T2]
    new = [n for n in mine if n not in T1 | T2]
    note = (f'{len(mine)} theorem(s) of this row ({short(mine)}) are inventoried by the proof packet\'s check.py (source mode '
            f'SOURCE_CHECK_PASS: {inv["theorem_count"]} inventory theorems over {MODULES_TXT}, {len(inv["covered_props"])} distinct registered '
            'Props, each theorem typed exactly as its registered Prop, applied to its own parameters). ')
    if r1:
        note += f'{len(r1)} of them ({short(r1)}) {TIER1_PHRASE}. '
    if r2:
        note += f'{len(r2)} ({short(r2)}) {TIER2_PHRASE}. '
    if new:
        note += f'{len(new)} ({short(new)}) {PENDING_PHRASE}. '
    note += ('`proved` is the label of a committed proof package; `kernel-checked` appears only in a run receipt. Evidence about the '
             'registered Props only: not about the informal claim, its alignment (PENDING_INDEPENDENT_REVIEW), review or acceptance; the '
             'claim-level label is unchanged; organizational-independence credit 0 (Anthropic / Claude).')
    wf = BASE_WF + (WF_PENDING if new else '')
    row['kernel_evidence'].append({'package': PROOF_PACKAGE, 'receipt_or_workflow': wf, 'note': note})

# ----------------------------------------------------------------------------- C. the two full-price rows regenerated
NO_REVIEW2 = ('Nonauthor review and acceptance are not claimed by any statement or theorem; alignment of every statement with its anchor '
              'is PENDING_INDEPENDENT_REVIEW; organizational-independence credit of this packet and of the proof packet is 0.')
MAIN_WHERE = 'at commit 91cffbb5 of main PR #358, not on main'
PREDICATES = {'IsRealizedFamily', 'IsGood', 'IsKDecomposable', 'IsGeneratorCover', 'IsPaletteFeasible', 'IsMinimalForbidden',
              'IsTransversal', 'IsProperDownset'}
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
    r1 = [n for n in names if n in T1]
    r2 = [n for n in names if n in T2]
    new = [n for n in names if n not in T1 | T2]
    s = f'({len(names)} entries below, label `proved`'
    if r1:
        s += f'; {len(r1)} of them ({short(r1)}) {TIER1_PHRASE}'
    if r2:
        s += f'; {len(r2)} of them ({short(r2)}) {TIER2_PHRASE}'
    if new:
        s += f'; {len(new)} of them ({short(new)}) {PENDING_PHRASE}'
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
        return f'{lane}: {listing(targets)} remain the targets of this row'
    return f'{lane}: every theorem-shaped Prop of this row has a theorem' + (f' (the parametrised predicates {listing(preds)} are vocabulary, not targets)' if preds else '')


def receipt_clause(cid):
    names = [s['declaration'] for s in rows[cid]['statements'] if s['package'] == PROOF_PACKAGE]
    new = [n for n in names if n not in T1 | T2]
    if new:
        return f'the {len(new)} theorem(s) of this row added after PR head {C4} ({short(new)}) await the next hosted lean-specification-proofs.yml run, their first receipt'
    return f'every theorem of this row has a hosted receipt (run {RUN2} (PR head {C2}; actual tested merge {M2} with base {BASE}))'


local, pp, with_thm, without = row_props('p15-full-price')
F2, F3, SH = ('UniversalLaw.Spec.P15.' + n for n in ('FullTransformedPriceBudget', 'RhoStarEnclosure', 'SharpnessOfRhoStar'))
INTERP = ['UniversalLaw.Spec.P15.' + n for n in ('CoordinatewiseConcavity', 'HazardProductBound', 'HazardPhiBound', 'FullGroundPriceBound',
                                                  'FullGroundBudgetIff', 'LocalPriceBoundFull')]
heads = ['FullTransformedPriceBudget states (F2)' + (' and is proved as stated' if F2 in pp else ', not proved'),
         'RhoStarEnclosure states (F3)' + (' and is proved as stated' if F3 in pp else ', not proved'),
         'SharpnessOfRhoStar states the attainment clause' + (' and is proved as stated' if SH in pp else ', not proved')]
f2_sentence = ('Theorem F is proved as FullTransformedPriceBudget states it: for the formal RealizedData record with a_i >= 1, d_i >= 2, '
               'every p in [0,1]^X, every 0 <= c_v <= phi(p_v) and the same K >= K_H(d), coverCost <= cappedHazard rho_star mu, through '
               'the coordinatewise concavity (CoordinatewiseConcavity, a theorem of exactly that Prop), (F6)-(F8), (F12) and (F13); that is '
               'evidence about those Props, not acceptance of the informal theorem. ' if F2 in pp else '')
f3_sentence = ('RhoStarEnclosure is proved by a rational certificate (Real.exp_bound series enclosures for e and for exp with explicit '
               'remainders, compared by norm_num), not by the source\'s full_price.py Fraction procedure, which remains unspecified. '
               if F3 in pp else '')
rows['p15-full-price']['not_established'] = (
    '; '.join(heads) + '. The claim-level label stays `specified` (claim-level vocabulary none/specified; a theorem of a Prop is evidence '
    'about that Prop, for the formal objects, not acceptance of Theorem F; alignment PENDING_INDEPENDENT_REVIEW). ' + f2_sentence + f3_sentence
    + f'Props of this row with a theorem of exactly their type in the proof packet specifications/lean_statement_proofs_20261011: '
    f'{listing(with_thm)} {evidence_sentence(with_thm, pp)}. ' + remaining_sentence(without)
    + 'The derivative formulas (F4)-(F5) are not specified' + ('; CoordinatewiseConcavity, their consequence, is proved from the one-coordinate split of mu_p(A) (display (F4): affine in exp(-t_i) with '
       'coefficients 0 <= S_in <= S_out for a downset) and the second-derivative criterion for s |-> -log(a + b exp(-s)). ' if 'UniversalLaw.Spec.P15.CoordinatewiseConcavity' in pp else '. ')
    + 'SixSeventhsCertificate uses Mathlib\'s decimal bound on e in place of the source\'s series argument. The 7 main P15 arithmetic theorems '
    'listed as `proved` (finite arithmetic of the Section 6 certificate chain: adjacent 20-digit endpoints as Nat, 16/27 < 6/7, n >= 2a+1 '
    f'instances, the rational orderings and the degree-5 Taylor partial sum) are {MAIN_WHERE}; they do not place rho_star = 1/(3 - log(3e - 2)) '
    'between the endpoints' + ('' if F3 in pp else ' (nothing does: RhoStarEnclosure is specified only)')
    + ', do not prove e < 87/32 or 197/32 < exp(11/6) as real-number facts (the SixSeventhsCertificate theorem does, as that Prop states '
    'them), and are not Theorem F. The full_price.py Fraction procedure is not a kernel proof and is not specified. No arbitrary-downset or '
    'unrestricted P15 prize statement (original_prize_closed stays false). Disposition REVIEWED_SCOPED is transcribed, not re-derived. '
    + NO_REVIEW2)
targets, _ = split_targets(without)
remarks = []
if F3 in without:
    remarks.append('RhoStarEnclosure ((F3), 20 digits) needs proof-producing exp/log enclosures')
if any(p in without for p in INTERP + [F2]):
    remarks.append('the remaining interpolation Props need the coordinatewise concavity route (F4)-(F8)')
rows['p15-full-price']['next_step'] = (
    targets_clause(without, 'Math- Mathlib lane') + (f' ({"; ".join(remarks)})' if remarks else '') + '; ' + receipt_clause('p15-full-price')
    + '. Independent reviewer: alignment of FullTransformedPriceBudget with (F2), in particular the quantification over ALL p in [0,1]^X and '
    'the SAME palette K >= K_H(d)' + (', and of the proved interpolation chain (CoordinatewiseConcavity through LocalPriceBoundFull) with Sections 2-3 of the source' if F2 in pp else '')
    + '. main: PR #358 merge (operator) so the 7 arithmetic theorems are on main.')

local, pp, with_thm, without = row_props('math.p15-full-price')
rows['math.p15-full-price']['not_established'] = (
    f'See row p15-full-price. Of the three Props of this node, with a theorem of exactly their type in the proof packet: {listing(with_thm)} '
    + (evidence_sentence(with_thm, pp) if with_thm else '')
    + (f'; stated, not proved: {listing(without)}' if without else '; none of the three is left unproved')
    + '. A theorem of SharpnessOfRhoStar is about the formal triple block and the formal cover cost'
    + ('; a theorem of FullTransformedPriceBudget is about the formal RealizedData record, downset, coverCost and mu' if F2 in pp else '')
    + '; none of this is Theorem F, and the node-level label stays `specified`. The 7 main P15 arithmetic theorems of source p15-full-price '
    f'({MAIN_WHERE}) are listed here as the arithmetic skeleton, never as the node. The price-boundary refutation is a scope boundary '
    '(GRAPH edge boundary_counterexample_not_required_premise), not a hypothesis. ' + NO_REVIEW2)
rows['math.p15-full-price']['next_step'] = 'As for row p15-full-price.'

# ----------------------------------------------------------------------------- D. invariants and write
for c in reg['claims']:
    assert c['formal_progress'] in ('none', 'specified')


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
    for run, merge in ((RUN1, M1), (RUN2, M2), ('38122386459', '866136cca66b46e6dbd3186b47c9340f7e93b4f6'), (RUN4, M4)):
        if run in s:
            assert merge in s, (run, s[:120])
reg_path.write_text(json.dumps(reg, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
after = {'statements': sum(len(c['statements']) for c in reg['claims']),
         'proved': sum(1 for c in reg['claims'] for s in c['statements'] if s['formal_progress'] == 'proved'),
         'kernel_evidence': sum(len(c['kernel_evidence']) for c in reg['claims'])}
summary = {'before': before, 'after': after, 'added_entries': len(added), 'new_theorems': new_names,
           'rows_with_additions': sorted({c for c, _ in added}, key=list(rows).index),
           'inventory': {'theorem_count': inv['theorem_count'], 'covered_props': len(inv['covered_props']),
                         'duplicate_proofs': inv['duplicate_proofs'], 'modules': inv['modules']}}
(OUT_DIR / 'update_summary5.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding='utf-8')
print(json.dumps(summary, indent=1, ensure_ascii=False))
