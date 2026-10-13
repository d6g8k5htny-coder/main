#!/usr/bin/env python3
"""Round 7, first slice (Side24 successor interface), binder: LOCAL ONLY until the coordinator activates (main#307 6137276515,
author confirmation 6137468192). Stacked on Math-#445 head 9fb0a4a0. Four files of specifications/lean_statements_20261010:

  Specifications/Side24.lean   append the round-7 block (scratch lvs/r7/hold/Side24_append.lean) before `end`; every earlier byte identical
  REGISTER.json                three statement rows under claim side24-coefficient after ReferenceComparison (structure + 2 def_prop,
                               all `specified`), one sentence each in that row's not_established and next_step; nothing else
  README.md, SCOPE.md          Declarations/Claims sections re-rendered from the new register with render_docs14's build() code and the
                               committed receipt phrases (extracted from the committed docs and verified by byte-identical reproduction of
                               9fb0a4a0 from the committed register first); the frontier section of #445 updated (66 = 64 + 2 open targets,
                               counts 141 -> 143, the two Side24 rows corrected per main#307 6137276515 item 5); prose counts 141 -> 143,
                               150 -> 153 where present tense.
Scientific effect NONE. No label moves; every new statement is `specified`; no theorem; the gate, tests, root module, lock, toolchain and the
proof packet are untouched. usage: python3 -B -S bind_round7.py <scratch> [--dry <outdir>]
"""
import hashlib, importlib.util, json, re, subprocess, sys
from collections import Counter, OrderedDict
from pathlib import Path, PurePosixPath

S = Path(sys.argv[1]); DRY = Path(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[2] == '--dry' else None
WT = S / 'lvs/r7/wt7'; PACKAGE = 'specifications/lean_statements_20261010'; PK = WT / PACKAGE
BASE_HEAD = '9fb0a4a06900da5cc3f9126f30e117a747df1587'       # Math-#445 head (frontier docs); its REGISTER.json and modules are 8acc4856's
MAIN_CUT = '8acc48566de3ae613c1ae320903b66c1d5411797'
PROOF_PACKAGE = 'd6g8k5htny-coder/Math- specifications/lean_statement_proofs_20261011'
MAIN_SIDE24 = 'd6g8k5htny-coder/main formal/ (UniversalLaw, side24-arithmetic-skeleton)'
MAIN_P15 = 'd6g8k5htny-coder/main formal/ at 91cffbb51d1636f382e9d61a1ecb6f7cd38e1d06 (PR #358, not yet on main)'
WORKSPACE_LANDING_RUN = '38096827443'
RUN1, C1, M1 = '38116277597', '00d2ac6e', 'c4c66bb8cb26f0cd9bc03d52bb25f1b3925b272c'
RUN2, C2, M2 = '38121713840', '910809d6', '5b198699308e0585904c21d12bf540cf2475d2f0'
RUN3, C3, M3 = '38122386459', 'd9b8dd33', '866136cca66b46e6dbd3186b47c9340f7e93b4f6'
BASE = 'b2d70d77876a10c365f5db5d3b6cafcf180af4e2'
ROW = 'side24-coefficient'
NEW = ['UniversalLaw.Spec.Side24.PeriodicCoefficientInterface2', 'UniversalLaw.Spec.Side24.CoefficientEnclosure2', 'UniversalLaw.Spec.Side24.ReferenceComparison2']

def git(*a, binary=False):
    r = subprocess.run(['git', '-C', str(WT), *a], check=True, capture_output=True)
    return r.stdout if binary else r.stdout.decode('utf-8')
def sha(b): return hashlib.sha256(b).hexdigest()
def once(t, old, new):
    assert t.count(old) == 1, (old[:70], t.count(old)); return t.replace(old, new)
def show(path): return git('show', f'{BASE_HEAD}:{PACKAGE}/{path}', binary=True)

assert git('rev-parse', 'HEAD').strip() == BASE_HEAD, 'worktree is not at the #445 head'
assert git('status', '--porcelain').strip() == '', 'worktree not clean'
for f in ('REGISTER.json', 'check.py', 'test_check.py', 'Specifications.lean', 'Specifications/Side24.lean'):
    assert git('rev-parse', f'{BASE_HEAD}:{PACKAGE}/{f}') == git('rev-parse', f'{MAIN_CUT}:{PACKAGE}/{f}'), f   # #445 changed docs only
spec = importlib.util.spec_from_file_location('chk', PK / 'check.py'); check = importlib.util.module_from_spec(spec); spec.loader.exec_module(check)
MODULES = list(check.MODULES)

# ----------------------------------------------------------------------------- 1. Side24.lean
orig = show('Specifications/Side24.lean'); assert (PK / 'Specifications/Side24.lean').read_bytes() == orig
block = (S / 'lvs/r7/hold/Side24_append.lean').read_text(encoding='utf-8')
END = 'end UniversalLaw.Spec.Side24\n'; assert orig.decode().endswith('\n' + END)
head_txt = orig.decode()[:-len(END)].rstrip('\n')
new_side24 = head_txt + '\n' + block.rstrip('\n') + '\n\n' + END
assert new_side24.startswith(head_txt) and new_side24.encode() == (S / 'lvs/r7/hold/Side24_draft.lean').read_bytes(), 'draft module drifted from the block'

# ----------------------------------------------------------------------------- 2. REGISTER.json
reg_raw = show('REGISTER.json'); assert (PK / 'REGISTER.json').read_bytes() == reg_raw
reg = json.loads(reg_raw.decode('utf-8'))
assert (json.dumps(reg, indent=2, ensure_ascii=False) + '\n').encode() == reg_raw, 'serialization not reproduced'
proposed = json.loads((S / 'lvs/r7/hold/register_rows_proposed.json').read_text(encoding='utf-8'))
rows_new = proposed['rows']; assert [r['declaration'] for r in rows_new] == NEW
row = next(c for c in reg['claims'] if c['claim_id'] == ROW)
sts = row['statements']; i = next(k for k, s in enumerate(sts) if s['declaration'] == 'UniversalLaw.Spec.Side24.ReferenceComparison')
assert all(s['declaration'] not in NEW for c in reg['claims'] for s in c['statements'])
row['statements'] = sts[:i + 1] + rows_new + sts[i + 1:]
NE_AFTER = proposed['claim_row_text_edits_proposed']['not_established_append_after']
NE_INS = (' round 7 (2026-10-13) adds, appended to Side24.lean with every earlier declaration byte-identical, the successor interface PeriodicCoefficientInterface2 '
          '(every field of PeriodicCoefficientInterface plus pinCov_posDef, condHessianCov_posDef and coneMoment_def, which pins coneMoment d (R 0) to the Gaussian cone '
          'expectation coneMomentOf d (K24 d) (R 0) R, the integral of det(A)^2 1{A<0} against multivariateGaussian 0 (condHessianCov d (K24 d) (R 0) R) with condHessianCov '
          'the Schur complement transCov - transPinCov pinCov^(-1) transPinCov^T of the derivative-rule blocks) and the successor Props CoefficientEnclosure2, '
          'ReferenceComparison2 over it, all `specified`; the successor excludes the doubling inhabitant but still leaves sphereMeasure free beyond support and total area '
          '(immaterial to the two successor inequalities by the source\'s uniform bounds, not to the value of c24; not shown) and does not state frame invariance or '
          'integrability; no inhabitant is constructed, no theorem of either successor Prop exists, the original Props and interface are unchanged, and nothing here '
          'encloses the coefficient;')
NS_AFTER = proposed['claim_row_text_edits_proposed']['next_step_append_after']
NS_INS = (' CoefficientEnclosure2, ReferenceComparison2 (round-7 successors over PeriodicCoefficientInterface2, `specified`) are the targets of this row: '
          'ReferenceComparison2 needs the Section 3 passage of (3) to the Schur complement, the Section 4 Gaussian comparison of the cone expectation under ordered '
          'covariances and the frame-uniform integrand ratio over the packet\'s CovarianceOrdering, ReferenceCovarianceLowerBound, DensityComparison and '
          'IntegrandRatioBounds (estimate weeks); CoefficientEnclosure2 needs in addition the proof-producing interval pipeline (months);')
row['not_established'] = once(row['not_established'], NE_AFTER, NE_AFTER + NE_INS)
row['next_step'] = once(row['next_step'], NS_AFTER, NS_AFTER + NS_INS)
new_reg_raw = (json.dumps(reg, indent=2, ensure_ascii=False) + '\n').encode()
old_reg = json.loads(reg_raw.decode('utf-8'))
for o, n in zip(old_reg['claims'], reg['claims']):     # nothing else changed
    if o['claim_id'] != ROW: assert o == n, o['claim_id']
    else:
        assert {k for k in o if o[k] != n[k]} == {'statements', 'not_established', 'next_step'}
        assert n['formal_progress'] == o['formal_progress'] == 'specified' and n['kernel_evidence'] == o['kernel_evidence']
assert sum(len(c['statements']) for c in reg['claims']) == sum(len(c['statements']) for c in old_reg['claims']) + 3

# ----------------------------------------------------------------------------- 3. gate scan of the new module + register (in memory, sources from the packet)
sources, _ = check.load_sources(PK, None, reg)
scans = {m: check.scan_module((new_side24 if m == 'Specifications/Side24.lean' else (PK / m).read_text(encoding='utf-8')), PurePosixPath(m).stem, sources) for m in MODULES}
stats = check.check_register(reg, scans, sources)
scan = [{'module': m, **d} for m in MODULES for d in scans[m]]
n_def = sum(1 for d in scan if d['kind'] == 'def_prop'); n_struct = sum(1 for d in scan if d['kind'] == 'structure'); n_aux = sum(1 for d in scan if d['kind'] is None)
assert (n_def, n_struct, n_aux) == (143, 10, 132), (n_def, n_struct, n_aux)
print('gate check_register on the new register and scan:', stats)

# ----------------------------------------------------------------------------- 4. docs: render_docs14's build() with phrases extracted from the committed docs
COMMITTED = {n: show(n).decode('utf-8') for n in ('README.md', 'SCOPE.md')}
for n in COMMITTED: assert (PK / n).read_text(encoding='utf-8') == COMMITTED[n]
FRONTIER_OLD = '## Frontier: the 64 registered Props without a theorem'
HEADINGS = {'README.md': (('## Declarations', '## Claims'), ('## Claims', FRONTIER_OLD)),
            'SCOPE.md': (('## Declarations', '## Claims and what is not established'), ('## Claims and what is not established', FRONTIER_OLD))}
hdr_line = next(l for l in COMMITTED['README.md'].splitlines() if l.startswith('| declaration | kind | source | registered under claim rows |'))
HEADER_BIND = hdr_line[len('| declaration | kind | source | registered under claim rows | proved by (proof packet, `proved`; '):-len(') |')]
intro_committed = COMMITTED['README.md'].split('## Declarations\n\n', 1)[1].split('\n', 1)[0]
PEND_TXT = intro_committed.split('each merge has the same tree as its head', 1)[1].split('). That label belongs', 1)[0]
assert HEADER_BIND.startswith('52 kernel-checked by hosted run 38121713840') and PEND_TXT.startswith('; the 8 theorems of `P15Interpolation`')

def cell(s): return str(s).replace('|', '\\|').replace('\n', ' ')
def collect(claim_rows, order, src_of):
    out = OrderedDict()
    for r in claim_rows:
        for s in r['statements']:
            if s['package'] != PACKAGE: continue
            key = (s['module'], s['declaration'])
            if key not in out: out[key] = {'kind': s['kind'], 'anchor': s['anchor'], 'dnc': s['does_not_claim'], 'claims': [], 'proofs': []}
            out[key]['claims'].append(r['claim_id'])
    for r in claim_rows:
        for s in r['statements']:
            if s['package'] == PROOF_PACKAGE:
                prop = None
                for loc in r['statements']:
                    if loc['package'] == PACKAGE and loc['anchor'] == s['anchor'] and loc['kind'] == 'def_prop': prop = (loc['module'], loc['declaration'])
                assert prop is not None, s['declaration']
                if s['declaration'] not in out[prop]['proofs']: out[prop]['proofs'].append(s['declaration'])
    return out
def counts(d):
    ks = list(d); return (sum(1 for k in ks if d[k]['kind'] == 'def_prop'), sum(1 for k in ks if d[k]['proofs']), len({t for k in ks for t in d[k]['proofs']}))

def build(rows, scan_rows, n_keys_expected):
    order = {(r['module'], r['name']): (MODULES.index(r['module']), r['line']) for r in scan_rows}
    src_of = {(r['module'], r['name']): r['source_id'] for r in scan_rows if r['kind']}
    decls = collect(rows, order, src_of); keys = sorted(decls, key=lambda k: order[k]); assert len(keys) == n_keys_expected, len(keys)
    n_def, NP, NT = counts(decls); n_struct = sum(1 for k in keys if decls[k]['kind'] == 'structure'); n_aux = sum(1 for r in scan_rows if r['kind'] is None)
    dups = [k for k in keys if len(decls[k]['proofs']) > 1]
    DUP_TXT = '; `TriangleNeedsThree` is proved twice' if [k[1].split('.', 3)[-1] for k in dups] == ['TriangleNeedsThree'] else \
        ('; proved twice: ' + ', '.join('`' + k[1].split('.', 3)[-1] + '`' for k in dups) if dups else '')
    assert NT == NP + sum(len(decls[k]['proofs']) - 1 for k in dups)
    intro = (f'{len(keys)} registered declarations ({n_def} `def … : Prop`, {n_struct} interface `structure`), every one with formal progress '
             f'`specified` in this package; the {n_aux} auxiliary definitions and abbreviations of the modules are inventory only and are not '
             'register statements. Names are given without the module namespace `UniversalLaw.Spec.<Module>.`; the source column is the pinned '
             'source named by the docstring\'s `Source:` line. Anchors and "Not established" cells are copied verbatim from the Lean docstrings '
             '(`Anchor:` and `Does not claim:` lines) through REGISTER.json. '
             f'The last column names the theorems of the sibling proof packet `specifications/lean_statement_proofs_20261011` whose type is exactly '
             f'the declaration (namespace `UniversalLaw.Proofs.<Family>.`): {NP} of the {n_def} Props have one ({NT} '
             f'theorems{DUP_TXT}), each listed in the register under the claim rows as a statement entry of kind '
             '`theorem` with label `proved` (the label of a committed proof package; `kernel-checked` appears only in a run receipt: the hosted '
             f'`lean-specification-proofs.yml` run {RUN2} receipted the 52 theorems of PR head {C2} and run {RUN1} the first 38; a pull_request run checks out the '
             f'synthetic merge of the PR head into the base, so each receipt records as checked_commit that merge and not the head: run {RUN2} PR head {C2}, '
             f'actual tested merge {M2} with base {BASE}; run {RUN1} PR head {C1}, actual tested merge {M1} with the same base; run {RUN3} corroborates on PR head {C3}, '
             f'actual tested merge {M3}; each merge has the same tree as its head{PEND_TXT}). That label belongs to the theorem, not to the declaration here, which stays '
             '`specified`, and not to any claim row; a kernel-checked theorem is evidence about its Prop only, and alignment of every statement is '
             '`PENDING_INDEPENDENT_REVIEW`.')
    def proofs_cell(d): return ', '.join('`' + t.split('.', 2)[-1] + '`' for t in d['proofs']) if d['proofs'] else '—'
    def decl_table(scope):
        out = [intro, '']
        for m in MODULES:
            ks = [k for k in keys if k[0] == m]
            out.append(f'### `{m}` ({len(ks)} registered declarations)'); out.append('')
            if scope:
                out.append('| declaration | kind | source | exact coverage (verbatim anchor) | Not established (verbatim `Does not claim:`) | proved by (proof packet, `proved`; ' + HEADER_BIND + ') |'); out.append('|---|---|---|---|---|---|')
            else:
                out.append('| declaration | kind | source | registered under claim rows | proved by (proof packet, `proved`; ' + HEADER_BIND + ') |'); out.append('|---|---|---|---|---|')
            for k in ks:
                d = decls[k]; short = k[1].split('.', 3)[-1]
                if scope: out.append(f'| `{short}` | {d["kind"]} | `{src_of[k]}` | {cell(d["anchor"])} | {cell(d["dnc"])} | {cell(proofs_cell(d))} |')
                else: out.append(f'| `{short}` | {d["kind"]} | `{src_of[k]}` | {cell(", ".join("`" + c + "`" for c in d["claims"]))} | {cell(proofs_cell(d))} |')
            out.append('')
        return '\n'.join(out).rstrip()
    kinds = Counter(r['kind'] for r in rows); fps = Counter(r['formal_progress'] for r in rows)
    packet_rows = [r for r in rows if r['not_established'].startswith('Statement not extracted in this programme')]; other_rows = [r for r in rows if r not in packet_rows]
    n_local_entries = sum(1 for r in rows for s in r['statements'] if s['package'] == PACKAGE)
    n_side24 = sum(1 for r in rows for s in r['statements'] if s['package'] == MAIN_SIDE24); n_p15 = sum(1 for r in rows for s in r['statements'] if s['package'] == MAIN_P15)
    n_proof = sum(1 for r in rows for s in r['statements'] if s['package'] == PROOF_PACKAGE); n_proved = sum(1 for r in rows for s in r['statements'] if s['formal_progress'] == 'proved')
    assert n_proved == n_side24 + n_p15 + n_proof and fps['proved'] == 0 and set(fps) <= {'none', 'specified'}
    claim_intro = (
        f'{len(rows)} claim rows: ' + ', '.join(f'{v} `{k}`' for k, v in kinds.items()) + f'. Claim-level formal progress: {fps["specified"]} '
        f'`specified` (a `def … : Prop` stating the claim itself is registered), {fps["none"]} `none`; no research claim is `proved` or '
        f'`kernel-checked`. {n_local_entries} statement entries point at the {len(keys)} declarations of this package (all `specified`). '
        f'{n_proved} statement entries carry the label `proved` and are theorems of other packages, listed as the arithmetic skeleton or the '
        f'Prop-level proof of the claim they sit under and never as the claim: {n_side24} are the 31 Side24 arithmetic targets of main\'s `formal/` '
        f'at origin/main `f6deeba7`; {n_p15} are the 28 P15 arithmetic targets of main\'s `formal/` at `91cffbb5` (main PR #358, not yet on main; '
        f'kernel evidence the hosted workspace-landing run {WORKSPACE_LANDING_RUN}), each listed on every P15 row whose layer 0 source it anchors to; '
        f'{n_proof} are theorems of the proof packet `specifications/lean_statement_proofs_20261011` whose type is exactly a registered Prop of '
        f'this package ({NT} theorems, {NP} distinct Props, each listed on every row that registers its Prop; the '
        f'hosted `lean-specification-proofs.yml` run {RUN2} receipted the 52 theorems of PR head {C2} as kernel-checked and run {RUN1} the first 38; a pull_request run checks out the '
        f'synthetic merge of the PR head into the base, which the receipt records as checked_commit: run {RUN2} PR head {C2}, actual tested merge {M2} with base {BASE}; '
        f'run {RUN1} PR head {C1}, actual tested merge {M1} with the same base; run {RUN3} corroborates on PR head {C3}, actual tested merge {M3}; same trees as the heads{PEND_TXT}). Exactly one Prop registered as stating a '
        'claim itself has such a theorem: `DemandOneRefutation` on rows `p15-price-boundary` and `math.p15-price-boundary`; their claim-level label '
        'stays `specified` (the register admits only none/specified at claim level, alignment of the Prop with the source paragraph is '
        'PENDING_INDEPENDENT_REVIEW, and a receipt is about the Prop only), as the rows\' `not established` cells record. `disposition` is the verbatim '
        'LANDING_CLAIMS value or `—` (null). Nothing in these tables is a scientific status, a review verdict or an acceptance; alignment of every '
        'statement is `PENDING_INDEPENDENT_REVIEW`.')
    def statements_cell(r):
        loc = [s for s in r['statements'] if s['package'] == PACKAGE]; s24 = [s for s in r['statements'] if s['package'] == MAIN_SIDE24]
        p15 = [s for s in r['statements'] if s['package'] == MAIN_P15]; prf = [s for s in r['statements'] if s['package'] == PROOF_PACKAGE]
        assert len(loc) + len(s24) + len(p15) + len(prf) == len(r['statements']), r['claim_id']
        parts = []
        if loc: parts.append(f'{len(loc)} specified: ' + ', '.join('`' + s['declaration'].split('.', 3)[-1] + '`' for s in loc))
        if s24: parts.append(f'{len(s24)} proved (main, origin/main f6deeba7): ' + ', '.join('`' + s['declaration'].split('.', 2)[-1] + '`' for s in s24))
        if p15: parts.append(f'{len(p15)} proved (main 91cffbb5, PR #358, not on main): ' + ', '.join('`' + s['declaration'].split('.', 2)[-1] + '`' for s in p15))
        if prf: parts.append(f'{len(prf)} proved (proof packet): ' + ', '.join('`' + s['declaration'].split('.', 2)[-1] + '`' for s in prf))
        return '; '.join(parts) if parts else '—'
    def claim_table_readme():
        out = [claim_intro, '', '| claim_id | kind | disposition (LANDING_CLAIMS, verbatim) | formal_progress | layer 0 source | statements |', '|---|---|---|---|---|---|']
        for r in other_rows:
            src = r['layer0']['source_id'] or '—'
            out.append(f'| `{r["claim_id"]}` | {r["kind"]} | {r["disposition_transcribed"] or "—"} | `{r["formal_progress"]}` | `{src}` | {cell(statements_cell(r))} |')
        out.append(''); out.append(f'Packets listed by PROOF_INDEX by id only ({len(packet_rows)} rows, kind `proof_index_result`, formal progress `none`, no statements, bytes not pinned here; '
                   'each row\'s `not_established` says the statement was not extracted and no verdict is implied):'); out.append('')
        out.append(', '.join('`' + r['claim_id'] + '`' for r in packet_rows) + '.')
        return '\n'.join(out)
    def claim_table_scope():
        out = [claim_intro, '', '| claim_id | kind | disposition | formal_progress | statements | not established | next step |', '|---|---|---|---|---|---|---|']
        for r in other_rows:
            out.append(f'| `{r["claim_id"]}` | {r["kind"]} | {r["disposition_transcribed"] or "—"} | `{r["formal_progress"]}` | {cell(statements_cell(r))} | {cell(r["not_established"])} | {cell(r["next_step"])} |')
        out.append(''); out.append(f'### Packets listed by id only ({len(packet_rows)} rows)'); out.append('')
        out.append('Kind `proof_index_result`, formal progress `none`, no statements, no kernel evidence, layer 0 source null (bytes not pinned here). Shared `not_established`: '
                   + cell(packet_rows[0]['not_established']) + ' Shared `next_step` (per packet path): ' + cell(packet_rows[0]['next_step']).replace(packet_rows[0]['claim_id'], '<id>'))
        out.append(''); out.append('| claim_id | title |'); out.append('|---|---|')
        for r in packet_rows: out.append(f'| `{r["claim_id"]}` | {cell(r["title"])} |')
        return '\n'.join(out)
    return decl_table(False), claim_table_readme(), decl_table(True), claim_table_scope(), {'NP': NP, 'NT': NT, 'n_def': n_def, 'n_struct': n_struct, 'n_aux': n_aux, 'keys': len(keys)}

def splice(text, start_heading, end_heading, body):
    i = text.index('\n' + start_heading + '\n') + 1; j = text.index('\n' + end_heading + '\n', i) + 1
    return text[:i] + start_heading + '\n\n' + body + '\n\n' + text[j:]
def render_all(rows, scan_rows, n_keys):
    rd, rc, sd, sc, st = build(rows, scan_rows, n_keys); bodies = {'README.md': (rd, rc), 'SCOPE.md': (sd, sc)}; out = {}
    for name in ('README.md', 'SCOPE.md'):
        text = COMMITTED[name]
        for (start, end), body in zip(HEADINGS[name], bodies[name]): text = splice(text, start, end, body)
        out[name] = text
    return out, st

# 4a. REPRODUCE the committed 9fb0a4a0 docs from the committed register and the committed modules
old_scans = {m: check.scan_module((PK / m).read_text(encoding='utf-8'), PurePosixPath(m).stem, sources) for m in MODULES}
old_scan = [{'module': m, **d} for m in MODULES for d in old_scans[m]]
repro, st_old = render_all(old_reg['claims'], old_scan, 150)
for n in COMMITTED: assert repro[n] == COMMITTED[n], f'{n}: committed text not reproduced; refusing'
assert (st_old['NP'], st_old['NT'], st_old['n_def'], st_old['n_struct'], st_old['n_aux']) == (77, 78, 141, 9, 123)
print('reproduced 9fb0a4a0 README.md and SCOPE.md byte-identically from the committed register (77/78/141/9/123)')
# 4b. RENDER from the new register and the new scan
rendered, st = render_all(reg['claims'], scan, 153)
assert (st['NP'], st['NT'], st['n_def'], st['n_struct'], st['n_aux']) == (77, 78, 143, 10, 132), st

# 4c. frontier section (#445) -> 66 rows; prose counts
def defline(short):
    mod, name = short.split('.', 1); lines = new_side24.splitlines() if mod == 'Side24' else (PK / f'Specifications/{mod}.lean').read_text(encoding='utf-8').splitlines()
    hits = [i + 1 for i, l in enumerate(lines) if re.match(r'^def\s+' + re.escape(name) + r'\b', l)]; assert len(hits) == 1, (short, hits); return f'{mod}.lean:{hits[0]}'
OPEN_BULLET = ('- **open target** (2): the round-7 successor Props `CoefficientEnclosure2` and `ReferenceComparison2` over the successor\n'
               '  interface `PeriodicCoefficientInterface2` (appended to `Side24.lean` in round 7, the predecessor unchanged), whose fields pin\n'
               '  the cone moment to the Gaussian cone expectation under the conditional covariance of `K_24` given the pin `V_u = H_f u = 0`\n'
               '  and carry the parent\'s nondegeneracy clause as positivity fields. The doubling inhabitant of the predecessor is excluded;\n'
               '  `sphereMeasure` is still a given beyond support and total area (immaterial to these two inequalities by the source\'s\n'
               '  uniform bounds, not to the value of `c24`; not shown). Both stay `specified`; their proofs are open work, named below.\n')
ROW_CE2 = (f'| `Side24.CoefficientEnclosure2` | {defline("Side24.CoefficientEnclosure2")} | open target | Over `PeriodicCoefficientInterface2` (round 7): the headline '
           'enclosures with the cone moment pinned by `coneMoment_def`. A proof needs a proof-producing interval pipeline for `Gamma(7/6)`, `pi`, `exp`, the theta sums '
           'and the Gaussian cone integral (months); nothing here encloses the coefficient. |')
ROW_RC2 = (f'| `Side24.ReferenceComparison2` | {defline("Side24.ReferenceComparison2")} | open target | Over `PeriodicCoefficientInterface2` (round 7): display (4) with '
           'the cone moment pinned. A proof needs the Section 3 passage of (3) to the Schur complement, the Section 4 Gaussian comparison of the cone expectation under '
           'ordered covariances and the frame-uniform integrand ratio, over the proof packet\'s `CovarianceOrdering`, `ReferenceCovarianceLowerBound`, '
           '`DensityComparison` and `IntegrandRatioBounds` (weeks); the sphere-measure choice is immaterial to this inequality by the source\'s uniform bounds, not shown here. |')
LAST_MATHLIB_ROW_START = '| `Lifetime.RemainderBoundedR1` | Lifetime.lean:559 | Mathlib-blocked |'
def frontier_edit(t, xref_old, xref_new):
    t = once(t, FRONTIER_OLD, '## Frontier: the 66 registered Props without a theorem')
    t = once(t, xref_old, xref_new)
    t = once(t, 'the sibling proof packet proves 77 of the 141 registered Props. This section\nclassifies the 64 others,',
                'the sibling proof packet proves 77 of the 141 registered Props, 143 with the two round-7\nsuccessor Props appended to `Side24.lean` on top of that cut. This section classifies the 66 others,')
    t = once(t, 'every one of the 64 stays `specified`,', 'every one of the 66 stays `specified`,')
    t = once(t, 'Three classes, by what a proof would need:', 'Four classes, by what a proof would need:')
    t = once(t, 'it is counted among the 141 because', 'it is counted among the 143 because')
    t = once(t, 'Of the 105 target Props, 77 are proved and 28 remain.', 'Of the 107 target Props, 77 are proved and 30 remain.')
    t = once(t, '  (effort estimated in weeks each).\n', '  (effort estimated in weeks each).\n' + OPEN_BULLET)
    t = once(t, 'Missing field: the conditional law of the Hessian block given the gradient under `K24`, which pins `coneMoment`. |',
                'Missing field: the conditional law of the transverse Hessian block conditioned on the pin `V_u = H_f u = 0` under `K24`, which pins `coneMoment`; '
                'supplied in round 7 by the successor interface `PeriodicCoefficientInterface2` with the successor Prop `CoefficientEnclosure2` (open target below). |')
    t = once(t, 'Missing field: the same conditional law; the arithmetic is trivial. |',
                'Missing field: the same conditional law; the arithmetic is trivial. Successor in round 7: `ReferenceComparison2` over `PeriodicCoefficientInterface2` (open target below). |')
    i = t.index(LAST_MATHLIB_ROW_START); j = t.index('\n', i) + 1
    t = t[:j] + ROW_CE2 + '\n' + ROW_RC2 + '\n' + t[j:]
    t = once(t, 'Counts: 36 vocabulary + 24 interface-blocked + 4 Mathlib-blocked = 64; 64 + 77 = 141.',
                'Counts: 36 vocabulary + 24 interface-blocked + 4 Mathlib-blocked + 2 open targets = 66; 66 + 77 = 143.')
    return t
rendered['README.md'] = frontier_edit(rendered['README.md'], '(77 of the 141 Props, 78 theorems); the 64 Props without a theorem are classified in the Frontier section below',
                                      '(77 of the 143 Props, 78 theorems); the 66 Props without a theorem are classified in the Frontier section below')
rendered['SCOPE.md'] = frontier_edit(rendered['SCOPE.md'], '(78 theorems typed exactly as 77 of the 141 Props; the 64 others are classified in the Frontier section below;',
                                     '(78 theorems typed exactly as 77 of the 143 Props; the 66 others are classified in the Frontier section below;')
rendered['README.md'] = once(rendered['README.md'], 'The 150 declarations here remain `specified`;', 'The 153 declarations here remain `specified`;')
for n, t in rendered.items():   # no stale present-tense count: the only remaining '141'/'150' are the historical local-run sentences of README (2026-10-10 run)
    stale = [l for l in t.splitlines() if not l.startswith('|') and re.search(r'\b(141|150|64)\b', l) and 'proves 77 of the 141 registered Props, 143' not in l]
    for l in stale: assert n == 'README.md' and ('#check` of all 150' in l or '141 statements, 9 structures' in l), (n, l[:120])   # the 2026-10-10 local-run record
    assert not check.PLACEHOLDER.findall(t) and '†' not in t, n     # the gate's own placeholder test (check_docs)
check.check_docs  # exists
# ----------------------------------------------------------------------------- 5. write
targets = {'Specifications/Side24.lean': new_side24.encode(), 'REGISTER.json': new_reg_raw, 'README.md': rendered['README.md'].encode(), 'SCOPE.md': rendered['SCOPE.md'].encode()}
outroot = DRY if DRY else PK
for rel, data in targets.items():
    p = outroot / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
    print(f'{rel}: {len(data)} bytes, sha256 {sha(data)}')
json.dump({'base_head': BASE_HEAD, 'dry': bool(DRY), 'counts': st, 'register_statements': sum(len(c['statements']) for c in reg['claims']),
           'files': {rel: {'bytes': len(d), 'sha256': sha(d)} for rel, d in targets.items()}}, open(S / 'lvs/r7/bind_round7_summary.json', 'w'), indent=1)
print('done', 'DRY' if DRY else 'WORKTREE')
