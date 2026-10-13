"""Round-2 update of specifications/lean_statements_20261010/REGISTER.json for the proof packet.

Scientific effect: NONE. Adds statement entries of kind `theorem` (label `proved`, the label of a
committed proof package) for proof-packet theorems not yet listed, rewrites the proof-packet
kernel_evidence rows with the hosted run 38116277597 (commit 00d2ac6e) for the 38 theorems it
receipted and "pending" for theorems added afterwards, and regenerates the P15 rows'
not_established / next_step texts from the inventory. No claim-level label changes.
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

SCRATCH = Path('<SCRATCH>')
STMT = SCRATCH / 'repos/math/specifications/lean_statements_20261010'
PROOF = SCRATCH / 'repos/math/specifications/lean_statement_proofs_20261011'
OUT_DIR = SCRATCH / 'lvs/register_update2'
RECEIPTED = json.loads((OUT_DIR / 'receipted_38.json').read_text())

STMT_PACKAGE = 'specifications/lean_statements_20261010'
PROOF_PACKAGE = 'd6g8k5htny-coder/Math- specifications/lean_statement_proofs_20261011'
MAIN_COMMIT = '91cffbb51d1636f382e9d61a1ceb6f7cd38e1d06'.replace('1ceb6', '1d636')  # kept for reference only
RUN = RECEIPTED['run_id']
COMMIT = RECEIPTED['commit']
ART = RECEIPTED['artifact']
ART_SHA = RECEIPTED['artifact_sha256']
RECEIPTED_NAMES = {name for mod in RECEIPTED['theorems'].values() for name, _ in mod}
assert len(RECEIPTED_NAMES) == 38
MIN_ANCHOR = 16
PROOF_WORKFLOW = (f'.github/workflows/lean-specification-proofs.yml; hosted run {RUN} on commit {COMMIT[:8]} (job proof-evidence, '
                  f'artifact {ART}, sha256 {ART_SHA}): receipt formalization_status kernel-checked, leanchecker PASS, theorem_count 38, '
                  f'worktree CLEAN; theorems committed after {COMMIT[:8]} await the next hosted run')
RECEIPT_PHRASE = (f'the hosted lean-specification-proofs.yml run {RUN} on commit {COMMIT[:8]} kernel-checked them (receipt '
                  'formalization_status kernel-checked, leanchecker PASS, `#print axioms` within {propext, Classical.choice, Quot.sound})')
PENDING_PHRASE = ('added after commit ' + COMMIT[:8] + ': locally `lake env lean -DwarningAsError=true` exit 0 and `#print axioms` within '
                  '{propext, Classical.choice, Quot.sound} on an unverified worktree, not evidence about any commit; the next hosted '
                  'lean-specification-proofs.yml run is the first receipt for them')
OLD_LOCAL_EVIDENCE = ('locally `lake env lean -DwarningAsError=true` exit 0 and `#print axioms` exactly {propext, Classical.choice, Quot.sound} '
                      'on an unverified worktree; the hosted lean-specification-proofs.yml receipt, the only record that may say `kernel-checked`, '
                      'does not yet exist')


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

# ----------------------------------------------------------------------------- A. new proof-packet theorems under every row listing the Prop
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
        entry = {'package': PROOF_PACKAGE, 'module': t['module'], 'declaration': t['name'], 'kind': 'theorem',
                 'formal_progress': 'proved', 'anchor': anchor, 'does_not_claim': t['doc_full']['Does not establish']}
        row['statements'].append(entry)
        added.append((cid, t['name']))
assert all(n not in RECEIPTED_NAMES for _, n in added), 'a receipted theorem was missing from the register'
new_names = sorted({n for _, n in added})

# every listed proof-packet theorem must still exist in the inventory
inv_names = {t['name'] for t in theorems}
for c in reg['claims']:
    for s in c['statements']:
        if s['package'] == PROOF_PACKAGE:
            assert s['declaration'] in inv_names, s['declaration']

# ----------------------------------------------------------------------------- B. kernel_evidence rows of the proof packet (rewritten everywhere)
MODULES_TXT = 'SpecificationProofs/{' + ','.join(Path(m).stem for m in inv['modules']) + '}.lean'
for cid, row in rows.items():
    mine = [s['declaration'] for s in row['statements'] if s['package'] == PROOF_PACKAGE]
    row['kernel_evidence'] = [e for e in row['kernel_evidence'] if e['package'] != PROOF_PACKAGE]
    if not mine:
        continue
    rec = [n for n in mine if n in RECEIPTED_NAMES]
    new = [n for n in mine if n not in RECEIPTED_NAMES]
    short = lambda ns: ', '.join(n.split('.', 2)[-1] for n in ns)
    note = (f'{len(mine)} theorem(s) of this row ({short(mine)}) are inventoried by the proof packet\'s check.py (source mode '
            f'SOURCE_CHECK_PASS: {inv["theorem_count"]} inventory theorems over {MODULES_TXT}, {len(inv["covered_props"])} distinct registered '
            'Props, each theorem typed exactly as its registered Prop, applied to its own parameters). ')
    if rec:
        note += (f'{len(rec)} of them ({short(rec)}) are among the 38 theorems of commit {COMMIT[:8]} kernel-checked by the hosted run {RUN} '
                 '(job proof-evidence: fresh build, leanchecker PASS, `#print axioms` within {propext, Classical.choice, Quot.sound} for each, '
                 f'receipt formalization_status kernel-checked, worktree CLEAN; artifact {ART}). ')
    if new:
        note += f'{len(new)} ({short(new)}) were {PENDING_PHRASE}. '
    note += ('`proved` is the label of a committed proof package; `kernel-checked` appears only in a run receipt. Evidence about the '
             'registered Props only: not about the informal claim, its alignment (PENDING_INDEPENDENT_REVIEW), review or acceptance; the '
             'claim-level label is unchanged; organizational-independence credit 0 (Anthropic / Claude).')
    row['kernel_evidence'].append({'package': PROOF_PACKAGE, 'receipt_or_workflow': PROOF_WORKFLOW, 'note': note})

# ----------------------------------------------------------------------------- C. generic phrase updates (rows whose theorems are all receipted)
REPL = [
    (OLD_LOCAL_EVIDENCE, RECEIPT_PHRASE),
    ('(listed as `proved`; hosted receipt pending)', f'(listed as `proved`; kernel-checked by the hosted run {RUN} on commit {COMMIT[:8]})'),
    ('(5 entries below, `proved`; hosted receipt pending)', f'(5 entries below, `proved`; kernel-checked by the hosted run {RUN} on commit {COMMIT[:8]})'),
    ('none of them the enclosure; hosted receipt pending)', f'none of them the enclosure; kernel-checked by the hosted run {RUN} on commit {COMMIT[:8]})'),
    ('the real-number P15 theorems, hosted receipt pending)', f'the real-number P15 theorems, kernel-checked by the hosted run {RUN} on commit {COMMIT[:8]})'),
    ('specifications/lean_statement_proofs_20261011 (hosted receipt pending), which places rho_star below 6/7',
     f'specifications/lean_statement_proofs_20261011 (kernel-checked by the hosted run {RUN} on commit {COMMIT[:8]}), which places rho_star below 6/7'),
    ('are already proved in specifications/lean_statement_proofs_20261011; hosted receipt pending)',
     f'are already proved in specifications/lean_statement_proofs_20261011 and kernel-checked by the hosted run {RUN} on commit {COMMIT[:8]})'),
    ('(PENDING_INDEPENDENT_REVIEW), no hosted kernel receipt exists yet, and a kernel-checked Prop is',
     f'(PENDING_INDEPENDENT_REVIEW), the hosted run {RUN} on commit {COMMIT[:8]} kernel-checked the Prop and nothing more, and a kernel-checked Prop is'),
    ('Hosted lean-specification-proofs.yml run of the committed proof packet (first receipt; `kernel-checked` appears only there). ',
     f'The hosted lean-specification-proofs.yml run {RUN} on commit {COMMIT[:8]} produced the first receipt (`kernel-checked` appears only there); later commits need their own run. '),
    ('PENDING_INDEPENDENT_REVIEW, no hosted receipt, a Prop-level proof is evidence about the Prop only)',
     f'PENDING_INDEPENDENT_REVIEW, a Prop-level proof is evidence about the Prop only, whatever the hosted run {RUN} receipted)'),
    (' Hosted lean-specification-proofs.yml run for the five proved entries.', f' The hosted run {RUN} on commit {COMMIT[:8]} kernel-checked the five proved entries.'),
    ('; hosted lean-specification-proofs.yml run for the 5 proved entries;', f'; the hosted run {RUN} on commit {COMMIT[:8]} kernel-checked the 5 proved entries;'),
]
counts = {}
for cid, row in rows.items():
    for k in ('not_established', 'next_step', 'title'):
        v = row[k]
        if not isinstance(v, str):
            continue
        for old, new in REPL:
            if old in v:
                v = v.replace(old, new)
                counts[old[:50]] = counts.get(old[:50], 0) + 1
        row[k] = v
print('phrase replacements:', json.dumps(counts, indent=1))

# ----------------------------------------------------------------------------- D. P15 rows regenerated from the inventory
NO_REVIEW2 = ('Nonauthor review and acceptance are not claimed by any statement or theorem; alignment of every statement with its anchor '
              'is PENDING_INDEPENDENT_REVIEW; organizational-independence credit of this packet and of the proof packet is 0.')
MAIN_WHERE = 'at commit 91cffbb5 of main PR #358, not on main'


def prop_short(n):
    return n.split('.', 3)[-1]


def row_props(cid):
    row = rows[cid]
    local = [s['declaration'] for s in row['statements'] if s['package'] == STMT_PACKAGE and s['kind'] == 'def_prop']
    proved_props = {}
    for t in theorems:
        if cid in t['claim_ids']:
            proved_props.setdefault(t['proves'], []).append(t['name'])
    with_thm = [p for p in local if p in proved_props]
    without = [p for p in local if p not in proved_props]
    return local, proved_props, with_thm, without


def evidence_sentence(cid, with_thm, proved_props):
    names = [n for p in with_thm for n in proved_props[p]]
    rec = [n for n in names if n in RECEIPTED_NAMES]
    new = [n for n in names if n not in RECEIPTED_NAMES]
    s = f'({len(names)} entries below, label `proved`'
    if rec:
        s += f'; {len(rec)} of them ({", ".join(n.split(".", 2)[-1] for n in rec)}) kernel-checked by the hosted run {RUN} on commit {COMMIT[:8]}'
    if new:
        s += f'; {len(new)} of them ({", ".join(n.split(".", 2)[-1] for n in new)}) {PENDING_PHRASE}'
    return s + ')'


def listing(props):
    return ', '.join(prop_short(p) for p in props) if props else 'none'


PREDICATES = {'IsRealizedFamily', 'IsGood', 'IsKDecomposable', 'IsGeneratorCover', 'IsPaletteFeasible', 'IsMinimalForbidden',
              'IsTransversal', 'IsProperDownset'}


def split_targets(props):
    preds = [p for p in props if prop_short(p) in PREDICATES]
    targets = [p for p in props if prop_short(p) not in PREDICATES]
    return targets, preds


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


# p15-realized-covers
local, pp, with_thm, without = row_props('p15-realized-covers')
P2_PARTS = ['UniversalLaw.Spec.P15.FullBlockCoverIff', 'UniversalLaw.Spec.P15.CoverCostHazardBound']
p2_whole = all(p in pp for p in P2_PARTS)
rows['p15-realized-covers']['not_established'] = (
    f'Props of this row with a theorem of exactly their type in the proof packet specifications/lean_statement_proofs_20261011: '
    f'{listing(with_thm)} {evidence_sentence("p15-realized-covers", with_thm, pp)}; among them Proposition P1 '
    '(MinimalForbiddenCharacterisation), both halves of Theorem P2 (FullBlockCoverIff, display (P7), and CoverCostHazardBound, display '
    '(P8)), the local statements (P2)-(P5) and the benchmark Props including the union bound (P11)'
    if p2_whole else
    f'Props of this row with a theorem of exactly their type in the proof packet specifications/lean_statement_proofs_20261011: '
    f'{listing(with_thm)} {evidence_sentence("p15-realized-covers", with_thm, pp)}'
) + (
    '. ' + remaining_sentence(without)
    + ('Theorem P2 is proved as the two Props state it (for the formal RealizedData record and the formal downset, prices 0 <= c_v <= p_v), '
       'which is a statement about those Props, not acceptance of the informal theorem: the claim-level label stays `specified` (this '
       'register admits only none/specified at claim level), alignment of RealizedData / IsRealizedFamily / downset / coverCost / mu with '
       'the informal objects is PENDING_INDEPENDENT_REVIEW, and the price range p_v < c_v <= phi(p_v) is a different claim row. '
       if p2_whole else
       'Theorem P2 as a whole is not proved and the claim-level label stays `specified`. ')
    + 'The 12 main P15 arithmetic theorems listed as `proved` (finite arithmetic of Section 6: 409, 2454, 15, 500616, 419743994415, 816, '
      f'815*3 < 2448, 818, 409p = 1/100, the (P11) rational bound, positivity of 6/40900^409, the elementary instances of (P9)) are {MAIN_WHERE}; '
      'they do not count the minimal forbidders, show the palette is 816 or evaluate 6/40900^409, and the proof-packet theorems that do '
      '(Benchmark816MinimalForbiddenCounts, Benchmark816Palette, Benchmark816Price) do so for the formal RealizedData record, whose '
      'identification with the informal six-block construction is an alignment question. Nothing is asserted for prices p_v < c_v <= phi(p_v) '
      '(row p15-price-budget-restricted), for arbitrary P15 inputs or for the prize (original_prize_closed stays false). Disposition '
      'HOLD_WITH_DOMAIN is transcribed, not re-derived. ' + NO_REVIEW2)
rows['p15-realized-covers']['next_step'] = (
    targets_clause(without, 'Proof lane') + '; '
    + f'next hosted lean-specification-proofs.yml run for the theorems added after commit {COMMIT[:8]}. Independent reviewer: alignment of '
    'MinimalForbiddenCharacterisation with Proposition P1 (completeness clause), of FullBlockCoverIff with (P7), of CoverCostHazardBound '
    'with (P8) (the sInf cover cost, the cappedHazard endpoint convention) and of the Benchmark816 data record with the Section 6 '
    'construction. main: PR #358 merge (operator) so the 12 arithmetic theorems are on main.')

# p15-price-budget-restricted
local, pp, with_thm, without = row_props('p15-price-budget-restricted')
T1 = 'UniversalLaw.Spec.P15.RestrictedTransformedPriceBudget'
rows['p15-price-budget-restricted']['not_established'] = (
    ('RestrictedTransformedPriceBudget states (T1) for every realized family with a_i >= 1, d_i >= 2, p_v <= 1/4, c_v <= phi(p_v) and is '
     'proved as stated in the proof packet; that is evidence about the Prop (for the formal RealizedData record, downset, coverCost and mu), '
     'not acceptance of the informal theorem: the claim-level label stays `specified` (claim-level vocabulary none/specified; alignment '
     'PENDING_INDEPENDENT_REVIEW). '
     if T1 in pp else
     'RestrictedTransformedPriceBudget states (T1) for every realized family with a_i >= 1, d_i >= 2, p_v <= 1/4, c_v <= phi(p_v); not '
     'proved, so the claim-level label stays `specified`. ')
    + f'Props of this row with a theorem of exactly their type in the proof packet specifications/lean_statement_proofs_20261011: '
    f'{listing(with_thm)} {evidence_sentence("p15-price-budget-restricted", with_thm, pp)}. ' + remaining_sentence(without) + 'The intermediate ratio steps of (T3) are proved inside the theorems that use them; the integral representation '
    'behind (T2), the admissibility integral and the Section 5 checker program are not specified. The 3 main P15 arithmetic theorems '
    f'listed as `proved` (the a = 1, n = 3 instance 16/27, finite instances of the (T3) bound and the rational identity 81801/3345620000) are '
    f'{MAIN_WHERE}; they are not (T1). The dependency on claim p15-realized-covers is discharged only to the extent that row records. '
    'Disposition HOLD_WITH_DOMAIN ("independent theorem review not yet source-bound in manifest") is transcribed, not re-derived. ' + NO_REVIEW2)
rows['p15-price-budget-restricted']['next_step'] = (
    targets_clause(without, 'Math- Mathlib lane') + '; '
    + f'next hosted lean-specification-proofs.yml run for the theorems added after commit {COMMIT[:8]}. Independent reviewer: alignment of '
    'RestrictedTransformedPriceBudget with (T1) including the per-coordinate price hypothesis, the K >= K_H(d) palette and the cappedHazard '
    'endpoint convention. main: PR #358 merge (operator) so the 3 arithmetic theorems are on main.')

# p15-full-price
local, pp, with_thm, without = row_props('p15-full-price')
F2 = 'UniversalLaw.Spec.P15.FullTransformedPriceBudget'
F3 = 'UniversalLaw.Spec.P15.RhoStarEnclosure'
SH = 'UniversalLaw.Spec.P15.SharpnessOfRhoStar'
heads = []
heads.append('FullTransformedPriceBudget states (F2)' + (' and is proved as stated' if F2 in pp else ', not proved'))
heads.append('RhoStarEnclosure states (F3)' + (' and is proved as stated' if F3 in pp else ', not proved'))
heads.append('SharpnessOfRhoStar states the attainment clause' + (' and is proved as stated' if SH in pp else ', not proved'))
rows['p15-full-price']['not_established'] = (
    '; '.join(heads) + '. The claim-level label stays `specified` (claim-level vocabulary none/specified; a theorem of a Prop is evidence '
    'about that Prop, for the formal objects, not acceptance of Theorem F; alignment PENDING_INDEPENDENT_REVIEW). '
    f'Props of this row with a theorem of exactly their type in the proof packet specifications/lean_statement_proofs_20261011: '
    f'{listing(with_thm)} {evidence_sentence("p15-full-price", with_thm, pp)}. ' + remaining_sentence(without) +
    'The derivative formulas (F4)-(F5) are not specified. SixSeventhsCertificate uses Mathlib\'s decimal bound on e in place of the '
    'source\'s series argument. The 7 main P15 arithmetic theorems listed as `proved` (finite arithmetic of the Section 6 certificate '
    f'chain: adjacent 20-digit endpoints as Nat, 16/27 < 6/7, n >= 2a+1 instances, the rational orderings and the degree-5 Taylor partial '
    f'sum) are {MAIN_WHERE}; they do not place rho_star = 1/(3 - log(3e - 2)) between the endpoints'
    + ('' if F3 in pp else ' (nothing does: RhoStarEnclosure is specified only)')
    + ', do not prove e < 87/32 or 197/32 < exp(11/6) as real-number facts (the SixSeventhsCertificate theorem does, as that Prop states '
    'them), and are not Theorem F. The full_price.py Fraction procedure is not a kernel proof and is not specified. No arbitrary-downset or '
    'unrestricted P15 prize statement (original_prize_closed stays false). Disposition REVIEWED_SCOPED is transcribed, not re-derived. '
    + NO_REVIEW2)
rows['p15-full-price']['next_step'] = (
    targets_clause(without, 'Math- Mathlib lane') + ' (RhoStarEnclosure ((F3), 20 digits) needs proof-producing exp/log enclosures; '
    '(F6)-(F8), (F12) and (F2) need the coordinatewise concavity (F4)-(F5), which is not specified); '
    + f'next hosted lean-specification-proofs.yml run for the theorems added after commit {COMMIT[:8]}. Independent reviewer: alignment of '
    'FullTransformedPriceBudget with (F2), in particular the quantification over ALL p in [0,1]^X and the SAME palette K >= K_H(d). main: '
    'PR #358 merge (operator) so the 7 arithmetic theorems are on main.')

# math.p15-full-price
local, pp, with_thm, without = row_props('math.p15-full-price')
rows['math.p15-full-price']['not_established'] = (
    f'See row p15-full-price. Of the three Props of this node, with a theorem of exactly their type in the proof packet: {listing(with_thm)} '
    + (evidence_sentence('math.p15-full-price', with_thm, pp) if with_thm else '')
    + f'; stated, not proved: {listing(without)}. A theorem of SharpnessOfRhoStar is about the formal triple block and the formal cover cost; '
    'none of this is Theorem F, and the node-level label stays `specified`. The 7 main P15 arithmetic theorems of source p15-full-price '
    f'({MAIN_WHERE}) are listed here as the arithmetic skeleton, never as the node. The price-boundary refutation is a scope boundary '
    '(GRAPH edge boundary_counterexample_not_required_premise), not a hypothesis. ' + NO_REVIEW2)
rows['math.p15-full-price']['next_step'] = 'As for row p15-full-price.'

# ----------------------------------------------------------------------------- E. claim-level labels unchanged; write
for c in reg['claims']:
    assert c['formal_progress'] in ('none', 'specified')
leftover = [(c['claim_id'], k) for c in reg['claims'] for k in ('not_established', 'next_step')
            if 'hosted receipt pending' in c[k] or 'does not yet exist' in c[k] or 'has not yet happened' in c[k]]
print('leftover pending phrases:', leftover)
reg_path.write_text(json.dumps(reg, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
after = {'statements': sum(len(c['statements']) for c in reg['claims']),
         'proved': sum(1 for c in reg['claims'] for s in c['statements'] if s['formal_progress'] == 'proved'),
         'kernel_evidence': sum(len(c['kernel_evidence']) for c in reg['claims'])}
summary = {'before': before, 'after': after, 'added_entries': len(added), 'new_theorems': new_names,
           'rows_with_additions': sorted({c for c, _ in added}, key=list(rows).index),
           'inventory': {'theorem_count': inv['theorem_count'], 'covered_props': len(inv['covered_props']),
                         'duplicate_proofs': inv['duplicate_proofs'], 'modules': inv['modules']}}
(OUT_DIR / 'update_summary2.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding='utf-8')
print(json.dumps(summary, indent=1, ensure_ascii=False))
