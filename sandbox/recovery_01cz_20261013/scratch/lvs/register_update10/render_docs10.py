"""Round-10 re-render of the Declarations and Claims sections of README.md and SCOPE.md of the statement packet from
REGISTER.json after update_register10.py: the 2 round-5 proof-packet theorems (SpecificationProofs/{Side24Covariance,
Side24ConeMoment}.lean) are shown with the dagger † = "no hosted receipt yet", in render_docs8.py's convention (round 4's
pending tier), next to the 73 receipted theorems, which keep the receipted forms of render_docs9.py as amended in commit
be014794 (r4amend/apply_r4_fix2.py and the header wording "the merge having the same tree as PR head 48bfcbae").

Method (fail closed):
  1. REPRODUCE: render both docs from the COMMITTED register (read-only `git show be014794:...`) with the committed phrases
     and no pending theorem, and assert the result is byte-identical to the committed README.md / SCOPE.md (read-only
     `git show`). This pins every phrase constant below to the committed text before anything is changed.
  2. RENDER: render from the worktree register (after update_register10.py), with the round-5 theorems pending (†), apply
     the exact-string PROSE edits (counts 72 -> 74 Props, 73 -> 75 theorems, plus one pending clause per receipt passage),
     assert byte-for-byte preservation outside the two regenerated sections modulo exactly those edits, and write.
The worktree docs must equal the committed ones before the render (refused otherwise); no git command that changes state is
run (no checkout).

Scientific effect: NONE. The tables transcribe the register; they decide nothing. No claim-level label changes (claim
formal_progress stays none/specified); `proved` is the label of a committed proof package's theorem entry; `kernel-checked`
appears only in run receipts. A theorem is evidence about its registered Prop for the formal objects only, never about an
informal claim. Every mention of a hosted run binds the PR head AND the actual tested merge commit (C422-E1), asserted per
paragraph and per table cell; the round-5 last-receipted-head facts: run 38187538744, PR #434 head be014794, actual tested
merge 79d3964cf70e2eadad5c9629eef8f4b47f218dd0 with base 12e072d614dbe6e91b66d22a2ab55cf87860d693, receipt theorem_count
73, artifact lean-specification-proofs-38187538744-1 sha256 60c9be0b... (GitHub's digest). Same-account author:
organizational-independence credit 0; alignment of every statement with its anchor PENDING_INDEPENDENT_REVIEW; nonauthor
review and acceptance are not claimed.

Usage: python3 -B -S render_docs10.py        (R10_DOCS_OUT=<dir> writes the two rendered docs there instead of the worktree;
                                             R10_REG=<file> reads the round-10 register from a dry-run file)
"""
import importlib.util
import json
import os
import re
import subprocess
import sys
from collections import Counter, OrderedDict
from pathlib import Path, PurePosixPath

SCRATCH = Path('<SCRATCH>')
REPO = SCRATCH / 'repos/math_r5'
PACKAGE = 'specifications/lean_statements_20261010'
PK = REPO / PACKAGE
OUT = SCRATCH / 'lvs/register_update10/sdocs'
OUT.mkdir(parents=True, exist_ok=True)
DOCS_OUT = Path(os.environ['R10_DOCS_OUT']) if os.environ.get('R10_DOCS_OUT') else PK
PROOF_PACKAGE = 'd6g8k5htny-coder/Math- specifications/lean_statement_proofs_20261011'
MAIN_SIDE24 = 'd6g8k5htny-coder/main formal/ (UniversalLaw, side24-arithmetic-skeleton)'
MAIN_P15 = 'd6g8k5htny-coder/main formal/ at 91cffbb51d1636f382e9d61a1ecb6f7cd38e1d06 (PR #358, not yet on main)'
EXPECTED_HEAD = 'be01479435f98f34b802258ee9ec3c3578f37169'
BASE = 'b2d70d77876a10c365f5db5d3b6cafcf180af4e2'
BASE2 = '79731b961fac555f33be3cd90e379b246a5a80e3'
RUN1, C1, M1 = '38116277597', '00d2ac6e', 'c4c66bb8cb26f0cd9bc03d52bb25f1b3925b272c'
RUN2, C2, M2 = '38121713840', '910809d6', '5b198699308e0585904c21d12bf540cf2475d2f0'
RUN3, C3, M3 = '38122386459', 'd9b8dd33', '866136cca66b46e6dbd3186b47c9340f7e93b4f6'
RUN4, C4, M4 = '38124262242', '79731b96', '318ff194eecc42c39a6b27c1d7bf03f30d6a34d9'
RUN6, C6, M6 = '38149888699', '0dcb0016', '4363b6928c62da242952631c6b0357d096c3b252'
RUNA, C5A, M5A = '38149524759', '88c21058', '3306233bbdeadb15d9b4b07b94cc52a80ed8caeb'
RUN7, C7, M7, SHA7 = ('38150363063', '12e072d6', '86a6e8c876f7904cadda2d006a56f8792dc5c1af',
                      'bf0e0515e2035103b07d894e2a73eeb5b542885f6da16b98c228877a5e0c1690')
RUN8, C8, M8, BASE3, SHA8 = ('38180705952', '48bfcbae', '9833d6d9f9facc335e1a56e243c3144adefe6225', '12e072d614dbe6e91b66d22a2ab55cf87860d693',
                             '87309efad9b804c53c543b60fdb466de1f797971417363cc2e8fa1e10eac26bd')
RUN_STMT, RUN_AGG = '38180705999', '38180706314'   # round-4 companion runs: never cited here
# round 5 (FACTS): the last head with a hosted receipt, PR #434 head be014794
RUN9, C9, M9, SHA9 = ('38187538744', 'be014794', '79d3964cf70e2eadad5c9629eef8f4b47f218dd0',
                      '60c9be0b49d45931d6c82f04d73fb7e6b6e97d539e7c10a376c3e99e302f2617')
assert BASE3.startswith(C7) and EXPECTED_HEAD.startswith(C9)
RUN_MERGE = ((RUN1, M1), (RUN2, M2), (RUN3, M3), (RUN4, M4), (RUN6, M6), (RUNA, M5A), (RUN7, M7), (RUN8, M8), (RUN9, M9))
RUN_HEAD = {RUN1: C1, RUN2: C2, RUN3: C3, RUN4: C4, RUN6: C6, RUNA: C5A, RUN7: C7, RUN8: C8, RUN9: C9}
for r, m in RUN_MERGE:
    assert r.isdigit() and len(r) == 11 and re.fullmatch(r'[0-9a-f]{40}', m), (r, m)
assert all(re.fullmatch(r'[0-9a-f]{64}', s) for s in (SHA7, SHA8, SHA9))
RUNS = {r for r, _ in RUN_MERGE}
WORKSPACE_LANDING_RUN = '38096827443'
T3_MODULE = 'SpecificationProofs/P15Interpolation.lean'
T4_MODULES = ('SpecificationProofs/Side24Analysis.lean', 'SpecificationProofs/Side24Interfaces.lean', 'SpecificationProofs/RNCluster.lean')
T5_MODULES = ('SpecificationProofs/Side24Covariance.lean', 'SpecificationProofs/Side24ConeMoment.lean')   # root import order


def git(*args):
    return subprocess.run(['git', '-C', str(REPO), *args], check=True, capture_output=True, text=True, encoding='utf-8').stdout


head = git('rev-parse', 'HEAD').strip()
if head != EXPECTED_HEAD:
    sys.exit(f'REFUSED: worktree HEAD is {head}, expected {EXPECTED_HEAD}')
COMMITTED = {name: git('show', f'{EXPECTED_HEAD}:{PACKAGE}/{name}') for name in ('README.md', 'SCOPE.md')}
for name, text in COMMITTED.items():
    if (PK / name).read_text(encoding='utf-8') != text:
        sys.exit(f'REFUSED: worktree {name} differs from be014794 (already rendered?)')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


check = load_module('stmt_check_r10', PK / 'check.py')
RECEIPTED52 = set(json.loads((SCRATCH / 'lvs/register_update5/receipted_52.json').read_text())['all_52'])
assert len(RECEIPTED52) == 52
MODULES = list(check.MODULES)
REG_IN = Path(os.environ['R10_REG']) if os.environ.get('R10_REG') else PK / 'REGISTER.json'   # dry runs read a register written by R10_OUT
new_reg = json.loads(REG_IN.read_text(encoding='utf-8'))
old_reg = json.loads(git('show', f'{EXPECTED_HEAD}:{PACKAGE}/REGISTER.json'))
assert RUN9 in json.dumps(new_reg) and RUN9 not in json.dumps(old_reg), 'the worktree register must be the round-10 one (run update_register10.py first)'
assert [c['formal_progress'] for c in old_reg['claims']] == [c['formal_progress'] for c in new_reg['claims']]
sources, _ = check.load_sources(PK, None, new_reg)


def cell(s):
    return str(s).replace('|', '\\|').replace('\n', ' ')


scan = []
for m in MODULES:
    for d in check.scan_module((PK / m).read_text(encoding='utf-8'), PurePosixPath(m).stem, sources):
        scan.append({'module': m, **d})
order = {(r['module'], r['name']): (MODULES.index(r['module']), r['line']) for r in scan}
src_of = {(r['module'], r['name']): r['source_id'] for r in scan if r['kind']}


def collect(claim_rows):          # render_docs8/9 verbatim
    out = OrderedDict()
    for r in claim_rows:
        for s in r['statements']:
            if s['package'] != PACKAGE:
                continue
            key = (s['module'], s['declaration'])
            if key not in out:
                out[key] = {'kind': s['kind'], 'anchor': s['anchor'], 'dnc': s['does_not_claim'], 'claims': [], 'proofs': []}
            out[key]['claims'].append(r['claim_id'])
    for r in claim_rows:
        for s in r['statements']:
            if s['package'] == PROOF_PACKAGE:
                prop = None
                for loc in r['statements']:
                    if loc['package'] == PACKAGE and loc['anchor'] == s['anchor'] and loc['kind'] == 'def_prop':
                        prop = (loc['module'], loc['declaration'])
                assert prop is not None, s['declaration']
                if s['declaration'] not in out[prop]['proofs']:
                    out[prop]['proofs'].append(s['declaration'])
    return out


def counts(d):
    ks = list(d)
    return (sum(1 for k in ks if d[k]['kind'] == 'def_prop'), sum(1 for k in ks if d[k]['proofs']),
            len({t for k in ks for t in d[k]['proofs']}))


OLD_NDEF, OLD_NP, OLD_NT = counts(collect(old_reg['claims']))
assert (OLD_NDEF, OLD_NP, OLD_NT) == (141, 72, 73), (OLD_NDEF, OLD_NP, OLD_NT)
pp_new = [s for r in new_reg['claims'] for s in r['statements'] if s['package'] == PROOF_PACKAGE]
T3 = {s['declaration'] for s in pp_new if s['module'] == T3_MODULE}
T4 = {s['declaration'] for s in pp_new if s['module'] in T4_MODULES}
T5 = {s['declaration'] for s in pp_new if s['module'] in T5_MODULES}
assert len(T3) == 8 and len(T4) == 13 and len(T5) == 2 and not (T3 & T4 or T4 & T5 or T3 & T5) and not ((T3 | T4 | T5) & RECEIPTED52)
RECEIPTED60 = RECEIPTED52 | T3
RECEIPTED73 = RECEIPTED60 | T4
assert {s['declaration'] for r in old_reg['claims'] for s in r['statements'] if s['package'] == PROOF_PACKAGE} == RECEIPTED73
N3, N4, N5 = len(T3), len(T4), len(T5)
N_T5_ENTRIES = sum(1 for s in pp_new if s['declaration'] in T5)
stems = lambda mods: [PurePosixPath(m).stem for m in mods]
backticked = lambda ss: ', '.join('`' + s + '`' for s in ss[:-1]) + ' and `' + ss[-1] + '`'
MODS_TXT, MODS_PLAIN = backticked(stems(T4_MODULES)), ', '.join(stems(T4_MODULES)[:-1]) + ' and ' + stems(T4_MODULES)[-1]
MODS5_TXT = backticked(stems(T5_MODULES))

# ----------------------------------------------------------------------------- receipt phrases
# tiers 1-3 verbatim from render_docs6/8/9; tier 4 as committed at be014794 (render_docs9 + apply_r4_fix2 + the final header wording),
# with the literal 73 of run 38180705952's receipt (never NT, which this round moves to 75)
HEADER_BIND6 = (f'{len(RECEIPTED52)} kernel-checked by hosted run {RUN2}: PR head {C2}, actual tested merge {M2[:8]}… with base {BASE[:8]}…; the {N3} of P15Interpolation by '
                f'run {RUN6}: PR head {C6}, actual tested merge {M6[:8]}… with base {BASE2[:8]}… (the head of PR #422); same trees as the heads')
HEADER_BIND9 = HEADER_BIND6 + (f'; the {N4} of {MODS_PLAIN} by run {RUN8}: PR head {C8}, actual tested merge {M8[:8]}… with base {BASE3[:8]}… (the head of PR #423), '
                               f'the merge having the same tree as PR head {C8}; receipt theorem_count 73, artifact sha256 {SHA8[:8]}…')
HEADER_BIND10 = HEADER_BIND9 + f'; † = added after PR head {C9} (run {RUN9}, actual tested merge {M9[:8]}…), no hosted receipt yet'
PEND3_TXT = (f'; the {N3} theorems of `P15Interpolation`, added after PR head {C4} (run {RUN4}, actual tested merge {M4} with base {BASE}, receipt theorem_count 52), '
             f'are kernel-checked by the hosted run {RUN6} on PR head {C6} (actual tested merge {M6} with base {BASE2}, the head of PR #422; receipt theorem_count 60, '
             f'leanchecker PASS) and were first receipted by run {RUNA} on PR head {C5A} (actual tested merge {M5A} with the same base); same trees as the heads')
REC4_TXT = (f'; the {N4} theorems of {MODS_TXT}, added after PR head {C7} (run {RUN7}, actual tested merge {M7} with base {BASE2}; receipt theorem_count 60, '
            f'corroborating the {len(RECEIPTED60)} theorems above on that head; artifact lean-specification-proofs-{RUN7}-1, sha256 {SHA7}), are kernel-checked by '
            f'the hosted run {RUN8} on PR head {C8} (actual tested merge {M8} with base {BASE3}, the head of PR #423; receipt theorem_count 73, leanchecker PASS; '
            f'artifact lean-specification-proofs-{RUN8}-1, sha256 {SHA8}) and were first receipted by that run, which receipts all 73 theorems of that head; '
            'same tree as the head')
# round 5 (†): render_docs8's PEND4_TXT shape with be014794's facts
PEND5_TXT = (f'; the {N5} theorems marked † ({MODS5_TXT}) were added after PR head {C9}, the last head with a hosted receipt (run {RUN9}, actual tested merge {M9} '
             f'with base {BASE3}; receipt theorem_count 73, corroborating the {len(RECEIPTED73)} theorems above on that head; artifact lean-specification-proofs-{RUN9}-1, '
             f'sha256 {SHA9}), and have no hosted receipt yet: the next hosted `lean-specification-proofs.yml` run is their first receipt')


def build(rows, pending, header_bind, pend_txt):
    """The four regenerated section bodies (README decl, README claims, SCOPE decl, SCOPE claims) of a register, render_docs9's code."""
    decls = collect(rows)
    keys = sorted(decls, key=lambda k: order[k])
    assert len(keys) == 150, len(keys)
    n_def, NP, NT = counts(decls)
    n_struct = sum(1 for k in keys if decls[k]['kind'] == 'structure')
    n_aux = sum(1 for r in scan if r['kind'] is None)
    proof_theorems = {t for k in keys for t in decls[k]['proofs']}
    assert pending <= proof_theorems and proof_theorems - pending <= RECEIPTED73
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
             f'actual tested merge {M3}; each merge has the same tree as its head{pend_txt}). That label belongs to the theorem, not to the declaration here, which stays '
             '`specified`, and not to any claim row; a kernel-checked theorem is evidence about its Prop only, and alignment of every statement is '
             '`PENDING_INDEPENDENT_REVIEW`.')

    def proofs_cell(d):
        return ', '.join('`' + t.split('.', 2)[-1] + ('†' if t in pending else '') + '`' for t in d['proofs']) if d['proofs'] else '—'

    def decl_table(scope):
        out = [intro, '']
        for m in MODULES:
            ks = [k for k in keys if k[0] == m]
            out.append(f'### `{m}` ({len(ks)} registered declarations)')
            out.append('')
            if scope:
                out.append('| declaration | kind | source | exact coverage (verbatim anchor) | Not established (verbatim `Does not claim:`) | proved by (proof packet, `proved`; ' + header_bind + ') |')
                out.append('|---|---|---|---|---|---|')
            else:
                out.append('| declaration | kind | source | registered under claim rows | proved by (proof packet, `proved`; ' + header_bind + ') |')
                out.append('|---|---|---|---|---|')
            for k in ks:
                d = decls[k]
                short = k[1].split('.', 3)[-1]
                if scope:
                    out.append(f'| `{short}` | {d["kind"]} | `{src_of[k]}` | {cell(d["anchor"])} | {cell(d["dnc"])} | {cell(proofs_cell(d))} |')
                else:
                    out.append(f'| `{short}` | {d["kind"]} | `{src_of[k]}` | {cell(", ".join("`" + c + "`" for c in d["claims"]))} | {cell(proofs_cell(d))} |')
            out.append('')
        return '\n'.join(out).rstrip()

    kinds = Counter(r['kind'] for r in rows)
    fps = Counter(r['formal_progress'] for r in rows)
    packet_rows = [r for r in rows if r['not_established'].startswith('Statement not extracted in this programme')]
    other_rows = [r for r in rows if r not in packet_rows]
    n_local_entries = sum(1 for r in rows for s in r['statements'] if s['package'] == PACKAGE)
    n_side24 = sum(1 for r in rows for s in r['statements'] if s['package'] == MAIN_SIDE24)
    n_p15 = sum(1 for r in rows for s in r['statements'] if s['package'] == MAIN_P15)
    n_proof = sum(1 for r in rows for s in r['statements'] if s['package'] == PROOF_PACKAGE)
    n_proved = sum(1 for r in rows for s in r['statements'] if s['formal_progress'] == 'proved')
    assert n_proved == n_side24 + n_p15 + n_proof
    assert fps['proved'] == 0 and fps['kernel-checked'] == 0 and set(fps) <= {'none', 'specified'}, dict(fps)
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
        f'run {RUN1} PR head {C1}, actual tested merge {M1} with the same base; run {RUN3} corroborates on PR head {C3}, actual tested merge {M3}; same trees as the heads{pend_txt}). Exactly one Prop registered as stating a '
        'claim itself has such a theorem: `DemandOneRefutation` on rows `p15-price-boundary` and `math.p15-price-boundary`; their claim-level label '
        'stays `specified` (the register admits only none/specified at claim level, alignment of the Prop with the source paragraph is '
        'PENDING_INDEPENDENT_REVIEW, and a receipt is about the Prop only), as the rows\' `not established` cells record. `disposition` is the verbatim '
        'LANDING_CLAIMS value or `—` (null). Nothing in these tables is a scientific status, a review verdict or an acceptance; alignment of every '
        'statement is `PENDING_INDEPENDENT_REVIEW`.')

    def statements_cell(r):
        loc = [s for s in r['statements'] if s['package'] == PACKAGE]
        s24 = [s for s in r['statements'] if s['package'] == MAIN_SIDE24]
        p15 = [s for s in r['statements'] if s['package'] == MAIN_P15]
        prf = [s for s in r['statements'] if s['package'] == PROOF_PACKAGE]
        assert len(loc) + len(s24) + len(p15) + len(prf) == len(r['statements']), r['claim_id']
        parts = []
        if loc:
            parts.append(f'{len(loc)} specified: ' + ', '.join('`' + s['declaration'].split('.', 3)[-1] + '`' for s in loc))
        if s24:
            parts.append(f'{len(s24)} proved (main, origin/main f6deeba7): ' + ', '.join('`' + s['declaration'].split('.', 2)[-1] + '`' for s in s24))
        if p15:
            parts.append(f'{len(p15)} proved (main 91cffbb5, PR #358, not on main): ' + ', '.join('`' + s['declaration'].split('.', 2)[-1] + '`' for s in p15))
        if prf:
            parts.append(f'{len(prf)} proved (proof packet): ' + ', '.join('`' + s['declaration'].split('.', 2)[-1] + ('†' if s['declaration'] in pending else '') + '`' for s in prf))
        return '; '.join(parts) if parts else '—'

    def claim_table_readme():
        out = [claim_intro, '', '| claim_id | kind | disposition (LANDING_CLAIMS, verbatim) | formal_progress | layer 0 source | statements |', '|---|---|---|---|---|---|']
        for r in other_rows:
            src = r['layer0']['source_id'] or '—'
            out.append(f'| `{r["claim_id"]}` | {r["kind"]} | {r["disposition_transcribed"] or "—"} | `{r["formal_progress"]}` | `{src}` | {cell(statements_cell(r))} |')
        out.append('')
        out.append(f'Packets listed by PROOF_INDEX by id only ({len(packet_rows)} rows, kind `proof_index_result`, formal progress `none`, no statements, bytes not pinned here; '
                   'each row\'s `not_established` says the statement was not extracted and no verdict is implied):')
        out.append('')
        out.append(', '.join('`' + r['claim_id'] + '`' for r in packet_rows) + '.')
        return '\n'.join(out)

    def claim_table_scope():
        out = [claim_intro, '', '| claim_id | kind | disposition | formal_progress | statements | not established | next step |', '|---|---|---|---|---|---|---|']
        for r in other_rows:
            out.append(f'| `{r["claim_id"]}` | {r["kind"]} | {r["disposition_transcribed"] or "—"} | `{r["formal_progress"]}` | {cell(statements_cell(r))} | {cell(r["not_established"])} | {cell(r["next_step"])} |')
        out.append('')
        out.append(f'### Packets listed by id only ({len(packet_rows)} rows)')
        out.append('')
        out.append('Kind `proof_index_result`, formal progress `none`, no statements, no kernel evidence, layer 0 source null (bytes not pinned here). Shared `not_established`: '
                   + cell(packet_rows[0]['not_established']) + ' Shared `next_step` (per packet path): ' + cell(packet_rows[0]['next_step']).replace(packet_rows[0]['claim_id'], '<id>'))
        out.append('')
        out.append('| claim_id | title |')
        out.append('|---|---|')
        for r in packet_rows:
            out.append(f'| `{r["claim_id"]}` | {cell(r["title"])} |')
        return '\n'.join(out)

    stats = {'NP': NP, 'NT': NT, 'n_def': n_def, 'dups': [k[1] for k in dups], 'intro': intro, 'claim_intro': claim_intro,
             'entries': {'local': n_local_entries, 'side24': n_side24, 'p15': n_p15, 'proof': n_proof, 'proved_total': n_proved},
             'claim_formal_progress': dict(fps), 'claim_rows': len(rows)}
    return decl_table(False), claim_table_readme(), decl_table(True), claim_table_scope(), stats


def splice(text, start_heading, end_heading, body):
    i = text.index('\n' + start_heading + '\n') + 1
    j = text.index('\n' + end_heading + '\n', i) + 1
    return text[:i] + start_heading + '\n\n' + body + '\n\n' + text[j:]


def outside(text, headings):
    for start, end in headings:
        i = text.index('\n' + start + '\n') + 1
        j = text.index('\n' + end + '\n', i) + 1
        text = text[:i] + start + '\n\n' + text[j:]
    return text


HEADINGS = {'README.md': (('## Declarations', '## Claims'), ('## Claims', '## What this packet is not')),
            'SCOPE.md': (('## Declarations', '## Claims and what is not established'),
                         ('## Claims and what is not established', '## Not established (by construction)'))}


def render_all(rows, pending, header_bind, pend_txt):
    rd, rc, sd, sc, stats = build(rows, pending, header_bind, pend_txt)
    bodies = {'README.md': (rd, rc), 'SCOPE.md': (sd, sc)}
    out = {}
    for name in ('README.md', 'SCOPE.md'):
        text = COMMITTED[name]
        for (start, end), body in zip(HEADINGS[name], bodies[name]):
            text = splice(text, start, end, body)
        out[name] = text
    return out, bodies, stats


# ----------------------------------------------------------------------------- 1. REPRODUCE the committed docs from the committed register
repro, _, old_stats = render_all(old_reg['claims'], set(), HEADER_BIND9, PEND3_TXT + REC4_TXT)
for name in ('README.md', 'SCOPE.md'):
    assert repro[name] == COMMITTED[name], f'{name}: the renderer does not reproduce the committed be014794 text; refusing to render'
assert (old_stats['NP'], old_stats['NT'], old_stats['n_def']) == (OLD_NP, OLD_NT, OLD_NDEF)
print('reproduction of be014794 README.md and SCOPE.md from the committed register: byte-identical')

# ----------------------------------------------------------------------------- 2. RENDER the round-10 state
rendered, bodies, st = render_all(new_reg['claims'], T5, HEADER_BIND10, PEND3_TXT + REC4_TXT + PEND5_TXT)
NP, NT, n_def = st['NP'], st['NT'], st['n_def']
assert (NP, NT, n_def) == (74, 75, 141), (NP, NT, n_def)       # check.py source mode: theorem_count 75, 74 covered Props

# prose phrases: the committed tier-3 / tier-4 phrases (render_docs6/9 verbatim, as committed) and the round-5 pending (†) successors
PEND3_SHORT = (f'; the {N3} of `P15Interpolation`, added after PR head {C4} (run {RUN4}, actual tested merge {M4[:8]}…), by run {RUN6}:\n'
               f'PR head {C6}, actual tested merge {M6[:8]}… with base {BASE2[:8]}…, the head of PR #422, same tree as the head')
REC4_SHORT = (f'; the {N4} of {MODS_TXT}, added after PR head {C7} (run {RUN7}, actual tested merge {M7[:8]}…), by run {RUN8}:\n'
              f'PR head {C8}, actual tested merge {M8[:8]}… with base {BASE3[:8]}…, the head of PR #423, same tree as the head')
PEND5_SHORT = (f'; the {N5} of {MODS5_TXT}, added after PR head {C9}, the last head with a hosted receipt (run {RUN9}, actual tested merge {M9[:8]}…),\n'
               'have no hosted receipt yet and are marked † in the tables')
PEND3_BULLET = (f'; the {N3} theorems\n'
                f'  of `P15Interpolation`, added after PR head {C4} (run {RUN4}, actual tested merge {M4[:8]}…), are receipted by run {RUN6}: PR head {C6}, actual tested\n'
                f'  merge {M6[:8]}… with base {BASE2[:8]}…, the head of PR #422, and first by run {RUNA}: PR head {C5A}, actual tested merge {M5A[:8]}…, same base')
REC4_BULLET = (f'; the {N4} theorems\n'
               f'  of {MODS_TXT}, added after PR head {C7} (run {RUN7}, actual tested merge {M7[:8]}…, receipt theorem_count 60), are receipted by run {RUN8}: PR head {C8},\n'
               f'  actual tested merge {M8[:8]}… with base {BASE3[:8]}…, the head of PR #423, receipt theorem_count 73, and were first receipted by that run')
PEND5_BULLET = (f'; the {N5} theorems\n'
                f'  of {MODS5_TXT}, added after PR head {C9} (run {RUN9}, actual tested merge {M9[:8]}…, receipt theorem_count 73), have no hosted receipt yet, marked †')
PEND5_SCOPE = (f', where † marks\n'
               f'a theorem added after PR head {C9} (run {RUN9}, actual tested merge {M9[:8]}…) with no hosted receipt yet')
PEND3_SCOPE_END = (f'; the {N3} of `P15Interpolation` by run {RUN6}: PR head {C6}, actual tested merge {M6[:8]}… with base\n'
                   f'{BASE2[:8]}…, the head of PR #422')
REC4_SCOPE_END = (f'; the {N4} of {MODS_TXT} by run {RUN8}: PR head {C8}, actual tested merge {M8[:8]}… with base\n'
                  f'{BASE3[:8]}…, the head of PR #423; receipt theorem_count 73; artifact lean-specification-proofs-{RUN8}-1, sha256 {SHA8[:8]}…')
PEND5_SCOPE_END = (f'; theorems added after PR head {C9} (run {RUN9}, actual tested merge {M9[:8]}… with base {BASE3[:8]}…) await the next hosted\n'
                   '`lean-specification-proofs.yml` run, their first receipt')
PROSE = {
    'README.md': [
        (f'Props ({OLD_NP} of the {OLD_NDEF} Props, {OLD_NT} theorems); the register lists them under the claim rows as\n'
         f'statement entries of kind `theorem` with label `proved` ({len(RECEIPTED52)} of them kernel-checked by the hosted run {RUN2}: PR head {C2},\n'
         f'actual tested merge {M2[:8]}… with base {BASE[:8]}…, same tree as the head{PEND3_SHORT}{REC4_SHORT}),\n',
         f'Props ({NP} of the {n_def} Props, {NT} theorems); the register lists them under the claim rows as\n'
         f'statement entries of kind `theorem` with label `proved` ({len(RECEIPTED52)} of them kernel-checked by the hosted run {RUN2}: PR head {C2},\n'
         f'actual tested merge {M2[:8]}… with base {BASE[:8]}…, same tree as the head{PEND3_SHORT}{REC4_SHORT}{PEND5_SHORT}),\n'),
        (f'  packet by path and holds {OLD_NT} theorems whose type is exactly one of {OLD_NP} registered Props (P15, Side24\n',
         f'  packet by path and holds {NT} theorems whose type is exactly one of {NP} registered Props (P15, Side24\n'),
        (f'  appears only in that packet\'s hosted run receipts (`lean-specification-proofs.yml`; run {RUN2} receipted the 52\n'
         f'  theorems of PR head {C2} and run {RUN1} the first 38; each receipt\'s checked_commit is the synthetic merge of the PR head into the\n'
         f'  base that a pull_request run checks out, not the head itself: run {RUN2} PR head {C2}, actual tested merge {M2[:8]}…; run {RUN1}\n'
         f'  PR head {C1}, actual tested merge {M1[:8]}…; both with base {BASE[:8]}… and each with the same tree as its head{PEND3_BULLET}{REC4_BULLET}).\n',
         f'  appears only in that packet\'s hosted run receipts (`lean-specification-proofs.yml`; run {RUN2} receipted the 52\n'
         f'  theorems of PR head {C2} and run {RUN1} the first 38; each receipt\'s checked_commit is the synthetic merge of the PR head into the\n'
         f'  base that a pull_request run checks out, not the head itself: run {RUN2} PR head {C2}, actual tested merge {M2[:8]}…; run {RUN1}\n'
         f'  PR head {C1}, actual tested merge {M1[:8]}…; both with base {BASE[:8]}… and each with the same tree as its head{PEND3_BULLET}{REC4_BULLET}{PEND5_BULLET}).\n'),
    ],
    'SCOPE.md': [
        (f'`specifications/lean_statement_proofs_20261011` ({OLD_NT} theorems typed exactly as {OLD_NP} of the {OLD_NDEF} Props;\n'
         f'the last column of the declaration tables and the `proved (proof packet …)` entries of the claims table);\n',
         f'`specifications/lean_statement_proofs_20261011` ({NT} theorems typed exactly as {NP} of the {n_def} Props;\n'
         f'the last column of the declaration tables and the `proved (proof packet …)` entries of the claims table{PEND5_SCOPE});\n'),
        (f'identity), never about the claim row it sits under; `kernel-checked` is a run-receipt word (the proof packet\'s\n'
         f'hosted run {RUN2} receipted the 52 theorems of PR head {C2}: actual tested merge {M2[:8]}… with base {BASE[:8]}…, the\n'
         f'synthetic merge a pull_request run checks out, same tree as the head{PEND3_SCOPE_END}{REC4_SCOPE_END}).',
         f'identity), never about the claim row it sits under; `kernel-checked` is a run-receipt word (the proof packet\'s\n'
         f'hosted run {RUN2} receipted the 52 theorems of PR head {C2}: actual tested merge {M2[:8]}… with base {BASE[:8]}…, the\n'
         f'synthetic merge a pull_request run checks out, same tree as the head{PEND3_SCOPE_END}{REC4_SCOPE_END}{PEND5_SCOPE_END}).'),
    ],
}
for name in PROSE:   # every replacement keeps every earlier binding of its anchor
    for old, new in PROSE[name]:
        assert new != old, (name, old[:60])
        for phrase in (PEND3_SHORT, REC4_SHORT, PEND3_BULLET, REC4_BULLET, PEND3_SCOPE_END, REC4_SCOPE_END, RUN2, RUN1, RUN6, RUNA, RUN7, RUN8):
            assert phrase not in old or phrase in new, (name, phrase[:60])

PAIRS = tuple((run, merge[:8]) for run, merge in RUN_MERGE)
summary = {'scientific_effect': 'NONE', 'reproduction_of_committed_docs': 'byte-identical', 'files': {}}
for name in ('README.md', 'SCOPE.md'):
    orig = COMMITTED[name]
    text = rendered[name]
    for old, new in PROSE[name]:
        assert text.count(old) == 1, (name, old[:60])
        text = text.replace(old, new)
    # byte-for-byte preservation outside the regenerated sections, modulo exactly the PROSE edits
    expect = outside(orig, HEADINGS[name])
    for old, new in PROSE[name]:
        assert expect.count(old) == 1, (name, 'prose anchor inside a regenerated section', old[:60])
        expect = expect.replace(old, new)
    assert outside(text, HEADINGS[name]) == expect, name
    assert not re.search(r'<<[^<>\n]*>>', text), name
    for phrase in ('Scientific effect', 'NONE', 'PENDING_INDEPENDENT_REVIEW', PEND3_TXT, REC4_TXT, PEND5_TXT, HEADER_BIND9, HEADER_BIND10, RUN9, M9, SHA9):
        assert phrase in text, (name, phrase[:60])
    # C422-E1: every mention of a hosted run binds its actual tested merge, per paragraph and per table cell; the round-5 run also
    # names its PR head wherever it appears; no unlisted run id; the round-4 companion runs are never cited
    for para in re.split(r'\n\s*\n', text):
        for run, merge in PAIRS:
            if run in para:
                assert merge in para, (name, run, para[:100])
        for run in (RUN8, RUN9):
            if run in para:
                assert RUN_HEAD[run] in para, (name, run, para[:100])
    for line in text.split('\n'):
        if line.startswith('|'):
            for c in line.split(' | '):
                for run, merge in PAIRS:
                    assert run not in c or merge in c, (name, run, c[-160:])
                assert RUN9 not in c or C9 in c, (name, c[-200:])
    for found in set(re.findall(r'\b38\d{9}\b', text)):
        assert found in RUNS or found == WORKSPACE_LANDING_RUN, (name, found)
    assert RUN_STMT not in text and RUN_AGG not in text, name
    # the † marks: one per pending theorem in the declaration tables (+ one header cell per module table and one in the intro),
    # one per pending statement entry in the claims table (+ one in the claims intro), and the prose edits; nothing else
    decl_body, claim_body = bodies[name]
    assert st['intro'].count('†') == 1 and st['claim_intro'].count('†') == 1
    assert decl_body.count('†') == N5 + len(MODULES) + 1, (name, decl_body.count('†'))
    assert claim_body.count('†') == N_T5_ENTRIES + 1, (name, claim_body.count('†'))
    assert text.count('†') == decl_body.count('†') + claim_body.count('†') + sum(new.count('†') for _, new in PROSE[name]), name
    assert orig.count('†') == 0
    for t in T5:
        assert ('`' + t.split('.', 2)[-1] + '†`') in decl_body, (name, t)
    for t in RECEIPTED73:
        assert ('`' + t.split('.', 2)[-1] + '†`') not in text and ('`' + t.split('.', 2)[-1] + '`') in decl_body, (name, t)
    (DOCS_OUT / name).write_text(text, encoding='utf-8')
    summary['files'][name] = {'bytes': len(text.encode()), 'dagger_marks': text.count('†'), 'changed': text != orig, 'written_to': str(DOCS_OUT / name)}
    print(name, len(text.encode()), 'bytes', f'({text.count("†")} † marks)')
summary.update({'props_with_proof': NP, 'proof_theorems': NT, 'def_props': n_def, 'pending_receipt': sorted(T5), 'pending_modules': list(T5_MODULES),
                'pending_statement_entries': N_T5_ENTRIES, 'committed_register_counts': {'props': OLD_NP, 'theorems': OLD_NT, 'def_props': OLD_NDEF},
                'duplicates': st['dups'], 'tiers': {'T1_T2_receipted_52': len(RECEIPTED52), 'T3_P15Interpolation': N3, 'T4_run_38180705952': N4,
                                                    'T5_no_hosted_receipt_after_head_be014794': N5},
                'entries': st['entries'], 'claim_rows': st['claim_rows'], 'claim_formal_progress': st['claim_formal_progress'],
                'run_merge_bindings_asserted': dict(RUN_MERGE), 'run_head_bindings_asserted': RUN_HEAD,
                'phrases': {'HEADER_BIND10': HEADER_BIND10, 'PEND5_TXT': PEND5_TXT, 'PEND5_SHORT': PEND5_SHORT, 'PEND5_BULLET': PEND5_BULLET,
                            'PEND5_SCOPE': PEND5_SCOPE, 'PEND5_SCOPE_END': PEND5_SCOPE_END}})
(OUT / 'render_summary10.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps({k: v for k, v in summary.items() if k != 'phrases'}, indent=1, ensure_ascii=False))
