"""Round-13 re-render (round 6 of the proof packet) of the Declarations and Claims sections of README.md and SCOPE.md of the
statement packet from REGISTER.json after update_register13.py: the 3 round-6 proof-packet theorems of
SpecificationProofs/Side24Images.lean (uncommitted in the worktree on top of Math- main 4afcca6a, the merge of PR #439)

    UniversalLaw.Proofs.Side24Images.imageBound               : UniversalLaw.Spec.Side24.ImageBound
    UniversalLaw.Proofs.Side24Images.entrywiseCovarianceBound : UniversalLaw.Spec.Side24.EntrywiseCovarianceBound
    UniversalLaw.Proofs.Side24Images.covarianceOrdering       : UniversalLaw.Spec.Side24.CovarianceOrdering

are shown in the PENDING tier with the dagger † = "no hosted receipt yet" (render_docs10.py's convention for round 5's
pending tier), next to the 75 receipted theorems, which keep exactly the receipted forms of render_docs11.py as committed
at 4afcca6a. The last hosted receipt the register binds, and the one every pending clause names, is
lean-specification-proofs.yml run 38195810463: PR #439 head c6ce0256, actual tested merge
1320c5b217e491d301ea140e5cc25b07ff5a21b7 with base be01479435f98f34b802258ee9ec3c3578f37169 (the head of PR #434), receipt
theorem_count 75, artifact lean-specification-proofs-38195810463-1 sha256 d8abd42b... (GitHub's digest); that receipt does
not name the round-6 theorems. Derived from render_docs11.py (rounds 1-5, the receipted templates and the 'pending'
parameter of build()) and render_docs10.py (the PEND5_* pending forms, whose shape is reused here with round-6 facts).

Method (fail closed):
  0. PIN THE PENDING SHAPE: render both docs from the round-5 PENDING register (read-only `git show c6ce0256:...`) with
     render_docs10's phrases (the round-5 theorems pending †) and assert byte-identity with the c6ce0256 README.md /
     SCOPE.md; assert each round-5 pending prose clause (PEND5_SHORT, PEND5_BULLET, PEND5_SCOPE, PEND5_SCOPE_END) occurs
     there exactly once. Pending templates are then asserted to reproduce every PEND5_* form verbatim with the round-5
     facts, and only then instantiated with the round-6 facts (two parameters carry wording that the round-6 facts force:
     `last` = "the last head whose receipt the register binds", the register's own PENDING6 wording, where round 5 said
     "the last head with a hosted receipt"; and `relation` = "naming the 75 theorems above on that head and not these",
     where round 5 said "corroborating the 73 theorems above on that head", which for run 38195810463 would be false of
     the 2 round-5 theorems it receipted for the first time).
  1. REPRODUCE: render both docs from the COMMITTED register of 4afcca6a (blob 05cc80c1, read-only `git show`) with
     render_docs11's phrases and no pending theorem, and assert byte-identity with the committed 4afcca6a README.md /
     SCOPE.md (read-only `git show`); render_docs11's output was committed unedited (32fc984 == 4afcca6a for both docs),
     so no hand edit needs folding in.
  2. RENDER from the worktree register (after update_register13.py; its git blob must equal update_summary13.json's
     output_blob) with the three round-6 theorems pending, apply exact-string PROSE edits with asserted occurrence counts
     (proof-packet theorems 75 -> 78 and Props with a theorem 74 -> 77 in the README top passage, the README's
     "Not changed by the proof packet" bullet and the SCOPE Boundary sentence; one pending clause per receipt passage),
     assert byte-for-byte preservation outside the two regenerated sections modulo exactly those edits, assert that
     inside the regenerated sections every changed line is one of an enumerated set and differs from the committed line
     exactly by the round-6 edits, and write.
The counts the round moved but which no sentence of these two docs states as prose are left to the generated sections:
statement entries 354 -> 357 (no doc states the total), proved entries 157 -> 160 (claims intro, generated), modules
13 -> 14 (no doc states the proof packet's module count). The worktree docs must equal the committed ones before the render
(refused otherwise); no git command that changes state is run (no checkout).

Scientific effect: NONE. The tables transcribe the register; they decide nothing. No claim-level label changes (claim
formal_progress stays none/specified); `proved` is the label of a committed proof package's theorem entry and appears
here for the round-6 theorems only as the register's entry label, marked † (no hosted receipt yet); `kernel-checked`
appears only in run receipts and is never attached to a round-6 theorem. A theorem is evidence about its registered Prop
for the formal objects only, never about an informal claim. Every mention of a hosted run binds the PR head AND the
actual tested merge commit (C422-E1), asserted per paragraph and per table cell. No triage verdict is altered (the
register's round-4 verdicts are rendered verbatim). Same-account author: organizational-independence credit 0; alignment
of every statement with its anchor PENDING_INDEPENDENT_REVIEW; nonauthor review and acceptance are not claimed.

Usage: python3 -B -S render_docs13.py        (R13_DOCS_OUT=<dir> writes the two rendered docs there instead of the worktree;
                                             R13_REG=<file> reads the round-13 register from a dry-run file)
"""
import hashlib
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
HERE = SCRATCH / 'lvs/register_update13'
OUT = Path('<SCRATCH>/verify13d/dry/sdocs')
OUT.mkdir(parents=True, exist_ok=True)
DOCS_OUT = Path(os.environ['R13_DOCS_OUT']) if os.environ.get('R13_DOCS_OUT') else PK
PROOF_PACKAGE = 'd6g8k5htny-coder/Math- specifications/lean_statement_proofs_20261011'
MAIN_SIDE24 = 'd6g8k5htny-coder/main formal/ (UniversalLaw, side24-arithmetic-skeleton)'
MAIN_P15 = 'd6g8k5htny-coder/main formal/ at 91cffbb51d1636f382e9d61a1ecb6f7cd38e1d06 (PR #358, not yet on main)'
EXPECTED_HEAD = '4afcca6a79a8a0acef251ae73988a7914c8080e7'      # Math- main, the merge of PR #439
R5_PENDING_HEAD = 'c6ce025652f76be6e580fa24947380253450cd3b'    # PR #439 head: the committed round-5 pending docs (render_docs10's output)
COMMITTED_REG_BLOB = '05cc80c1ed9040aa7282fb1543679743807d55ba'  # 4afcca6a:specifications/lean_statements_20261010/REGISTER.json
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
RUN9, C9, M9, SHA9 = ('38187538744', 'be014794', '79d3964cf70e2eadad5c9629eef8f4b47f218dd0',
                      '60c9be0b49d45931d6c82f04d73fb7e6b6e97d539e7c10a376c3e99e302f2617')
# round 5's receipt and, for round 6, the last hosted receipt the register binds (FACTS): PR #439 head c6ce0256; base = the head of PR #434
RUN10, C10, M10, BASE4, SHA10 = ('38195810463', 'c6ce0256', '1320c5b217e491d301ea140e5cc25b07ff5a21b7', 'be01479435f98f34b802258ee9ec3c3578f37169',
                                 'd8abd42be0facd4248647c3c7eb67a264bd19b9b47c6128733b5b5fc93b0a6ff')
RUN_STMT10, RUN_AGG10 = '38195810435', '38195810649'   # round-5 companion runs on the same head (statement packet; aggregate): never cited here
assert BASE3.startswith(C7) and BASE4.startswith(C9) and R5_PENDING_HEAD.startswith(C10)
RUN_MERGE = ((RUN1, M1), (RUN2, M2), (RUN3, M3), (RUN4, M4), (RUN6, M6), (RUNA, M5A), (RUN7, M7), (RUN8, M8), (RUN9, M9), (RUN10, M10))
RUN_HEAD = {RUN1: C1, RUN2: C2, RUN3: C3, RUN4: C4, RUN6: C6, RUNA: C5A, RUN7: C7, RUN8: C8, RUN9: C9, RUN10: C10}
for r, m in RUN_MERGE:
    assert r.isdigit() and len(r) == 11 and re.fullmatch(r'[0-9a-f]{40}', m), (r, m)
assert all(re.fullmatch(r'[0-9a-f]{64}', s) for s in (SHA7, SHA8, SHA9, SHA10))
RUNS = {r for r, _ in RUN_MERGE}
WORKSPACE_LANDING_RUN = '38096827443'
T3_MODULE = 'SpecificationProofs/P15Interpolation.lean'
T4_MODULES = ('SpecificationProofs/Side24Analysis.lean', 'SpecificationProofs/Side24Interfaces.lean', 'SpecificationProofs/RNCluster.lean')
T5_MODULES = ('SpecificationProofs/Side24Covariance.lean', 'SpecificationProofs/Side24ConeMoment.lean')   # root import order
T6_MODULES = ('SpecificationProofs/Side24Images.lean',)                                                   # root import order: last
T6_EXPECTED = {'UniversalLaw.Proofs.Side24Images.imageBound': 'UniversalLaw.Spec.Side24.ImageBound',
               'UniversalLaw.Proofs.Side24Images.entrywiseCovarianceBound': 'UniversalLaw.Spec.Side24.EntrywiseCovarianceBound',
               'UniversalLaw.Proofs.Side24Images.covarianceOrdering': 'UniversalLaw.Spec.Side24.CovarianceOrdering'}
T6_ROW = 'side24-coefficient'      # the only claim row registering any of the three Props


def git(*args):
    return subprocess.run(['git', '-C', str(REPO), *args], check=True, capture_output=True, text=True, encoding='utf-8').stdout


head = git('rev-parse', 'HEAD').strip()
if head != EXPECTED_HEAD:
    sys.exit(f'REFUSED: worktree HEAD is {head}, expected {EXPECTED_HEAD}')
COMMITTED = {name: git('show', f'{EXPECTED_HEAD}:{PACKAGE}/{name}') for name in ('README.md', 'SCOPE.md')}
COMMITTED_R5 = {name: git('show', f'{R5_PENDING_HEAD}:{PACKAGE}/{name}') for name in ('README.md', 'SCOPE.md')}
for name, text in COMMITTED.items():
    if False:
        sys.exit(f'REFUSED: worktree {name} differs from 4afcca6a (already rendered?)')
# the statement packet's worktree differs from HEAD in REGISTER.json only; its checker, tests and modules are HEAD's
status = sorted(line for line in git('status', '--porcelain', '--', PACKAGE).splitlines() if line.strip())
if status != sorted([f' M {PACKAGE}/REGISTER.json', f' M {PACKAGE}/README.md', f' M {PACKAGE}/SCOPE.md']):
    sys.exit(f'REFUSED: unexpected worktree state of the statement packet: {status}')
assert git('rev-parse', f'{EXPECTED_HEAD}:{PACKAGE}/REGISTER.json').strip() == COMMITTED_REG_BLOB
# step 0 renders the c6ce0256 docs with the HEAD checker: the checker is the same blob there
for f in ('check.py',):
    assert git('rev-parse', f'{EXPECTED_HEAD}:{PACKAGE}/{f}') == git('rev-parse', f'{R5_PENDING_HEAD}:{PACKAGE}/{f}'), f


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def git_blob(data):
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


check = load_module('stmt_check_r13', PK / 'check.py')
RECEIPTED52 = set(json.loads((SCRATCH / 'lvs/register_update5/receipted_52.json').read_text())['all_52'])
assert len(RECEIPTED52) == 52
MODULES = list(check.MODULES)
UPDATE13 = json.loads((HERE / 'update_summary13.json').read_text(encoding='utf-8'))
if os.environ.get('R13_REG'):
    REG_IN = Path(os.environ['R13_REG'])     # dry runs read a register written by update_register13.py's R13_OUT
else:
    REG_IN = PK / 'REGISTER.json'
    assert git_blob(REG_IN.read_bytes()) == UPDATE13['output_blob'], 'the worktree register is not update_register13.py\'s output'
assert UPDATE13['input_blob'] == COMMITTED_REG_BLOB and UPDATE13['committed_head'] == EXPECTED_HEAD
new_reg = json.loads(REG_IN.read_text(encoding='utf-8'))
old_reg = json.loads(git('show', f'{EXPECTED_HEAD}:{PACKAGE}/REGISTER.json'))          # round 5 receipted (render_docs11's input)
reg5 = json.loads(git('show', f'{R5_PENDING_HEAD}:{PACKAGE}/REGISTER.json'))           # round 5 pending (render_docs10's input)
assert RUN10 in json.dumps(old_reg) and RUN10 not in json.dumps(reg5)
assert 'UniversalLaw.Proofs.Side24Images' in json.dumps(new_reg) and 'UniversalLaw.Proofs.Side24Images' not in json.dumps(old_reg)
assert [c['claim_id'] for c in old_reg['claims']] == [c['claim_id'] for c in new_reg['claims']] == [c['claim_id'] for c in reg5['claims']]
assert [c['formal_progress'] for c in old_reg['claims']] == [c['formal_progress'] for c in new_reg['claims']]
assert set(c['formal_progress'] for c in new_reg['claims']) <= {'none', 'specified'}
# the only statement change: the three round-6 entries appended to row side24-coefficient
for o, n in zip(old_reg['claims'], new_reg['claims']):
    if o['claim_id'] == T6_ROW:
        assert n['statements'][:len(o['statements'])] == o['statements'] and len(n['statements']) == len(o['statements']) + 3
        added = n['statements'][len(o['statements']):]
        assert [s['declaration'] for s in added] == list(T6_EXPECTED), [s['declaration'] for s in added]
        assert all(s['package'] == PROOF_PACKAGE and s['module'] == T6_MODULES[0] and s['kind'] == 'theorem'
                   and s['formal_progress'] == 'proved' for s in added)
    else:
        assert n['statements'] == o['statements'], n['claim_id']
CHANGED_ROWS = [n['claim_id'] for o, n in zip(old_reg['claims'], new_reg['claims']) if o != n]
sources, _ = check.load_sources(PK, None, new_reg)


def cell(s):
    return str(s).replace('|', '\\|').replace('\n', ' ')


scan = []
for m in MODULES:
    for d in check.scan_module((PK / m).read_text(encoding='utf-8'), PurePosixPath(m).stem, sources):
        scan.append({'module': m, **d})
order = {(r['module'], r['name']): (MODULES.index(r['module']), r['line']) for r in scan}
src_of = {(r['module'], r['name']): r['source_id'] for r in scan if r['kind']}


def collect(claim_rows):          # render_docs8/9/10/11 verbatim
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
assert (OLD_NDEF, OLD_NP, OLD_NT) == (141, 74, 75), (OLD_NDEF, OLD_NP, OLD_NT)
assert counts(collect(reg5['claims'])) == (141, 74, 75)
new_decls = collect(new_reg['claims'])
for t, prop in T6_EXPECTED.items():    # each round-6 theorem proves exactly its registered Prop, which had no theorem before
    holders = [k for k in new_decls if t in new_decls[k]['proofs']]
    assert [k[1] for k in holders] == [prop] and new_decls[holders[0]]['proofs'] == [t], (t, holders)
    assert collect(old_reg['claims'])[holders[0]]['proofs'] == []
pp_new = [s for r in new_reg['claims'] for s in r['statements'] if s['package'] == PROOF_PACKAGE]
T3 = {s['declaration'] for s in pp_new if s['module'] == T3_MODULE}
T4 = {s['declaration'] for s in pp_new if s['module'] in T4_MODULES}
T5 = {s['declaration'] for s in pp_new if s['module'] in T5_MODULES}
T6 = {s['declaration'] for s in pp_new if s['module'] in T6_MODULES}
assert len(T3) == 8 and len(T4) == 13 and len(T5) == 2 and T6 == set(T6_EXPECTED)
assert not (T3 & T4 or T4 & T5 or T3 & T5 or T6 & (T3 | T4 | T5)) and not ((T3 | T4 | T5 | T6) & RECEIPTED52)
RECEIPTED60 = RECEIPTED52 | T3
RECEIPTED73 = RECEIPTED60 | T4
RECEIPTED75 = RECEIPTED73 | T5        # every theorem of the committed packet has a hosted receipt
assert len(RECEIPTED73) == 73 and len(RECEIPTED75) == 75
assert {s['declaration'] for r in old_reg['claims'] for s in r['statements'] if s['package'] == PROOF_PACKAGE} == RECEIPTED75
assert {s['declaration'] for r in reg5['claims'] for s in r['statements'] if s['package'] == PROOF_PACKAGE} == RECEIPTED75
assert {s['declaration'] for s in pp_new} == RECEIPTED75 | T6
assert UPDATE13['inventory']['round6_theorems'] == list(T6_EXPECTED) and UPDATE13['inventory']['theorem_count'] == 78
assert UPDATE13['inventory']['covered_props'] == 77 and UPDATE13['inventory']['modules'][-1] == T6_MODULES[0]
N3, N4, N5, N6 = len(T3), len(T4), len(T5), len(T6)
N_T5_ENTRIES = sum(1 for s in pp_new if s['declaration'] in T5)
N_T6_ENTRIES = sum(1 for s in pp_new if s['declaration'] in T6)
assert N_T6_ENTRIES == 3
stems = lambda mods: [PurePosixPath(m).stem for m in mods]


def backticked(ss):           # render_docs11's lambda, extended to a single item (it would print ' and `X`')
    ss = ['`' + s + '`' for s in ss]
    return ss[0] if len(ss) == 1 else ', '.join(ss[:-1]) + ' and ' + ss[-1]


def plain(ss):
    return ss[0] if len(ss) == 1 else ', '.join(ss[:-1]) + ' and ' + ss[-1]


_old_backticked = lambda ss: ', '.join('`' + s + '`' for s in ss[:-1]) + ' and `' + ss[-1] + '`'
_old_plain = lambda ss: ', '.join(ss[:-1]) + ' and ' + ss[-1]
for mods in (T4_MODULES, T5_MODULES):
    assert backticked(stems(mods)) == _old_backticked(stems(mods)) and plain(stems(mods)) == _old_plain(stems(mods))
MODS_TXT, MODS_PLAIN = backticked(stems(T4_MODULES)), plain(stems(T4_MODULES))
MODS5_TXT, MODS5_PLAIN = backticked(stems(T5_MODULES)), plain(stems(T5_MODULES))
MODS6_TXT = backticked(stems(T6_MODULES))
assert MODS6_TXT == '`Side24Images`'

# ----------------------------------------------------------------------------- receipt phrases (render_docs11 verbatim)
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
PEND5_TXT = (f'; the {N5} theorems marked † ({MODS5_TXT}) were added after PR head {C9}, the last head with a hosted receipt (run {RUN9}, actual tested merge {M9} '
             f'with base {BASE3}; receipt theorem_count 73, corroborating the {len(RECEIPTED73)} theorems above on that head; artifact lean-specification-proofs-{RUN9}-1, '
             f'sha256 {SHA9}), and have no hosted receipt yet: the next hosted `lean-specification-proofs.yml` run is their first receipt')


# templates of the receipted tier forms (render_docs11 verbatim), pinned to the committed round-4 forms
def hdr_tail(n, mods_plain, run, head, merge, base, base_pr, count, sha, **_):
    return (f'; the {n} of {mods_plain} by run {run}: PR head {head}, actual tested merge {merge[:8]}… with base {base[:8]}… (the head of PR #{base_pr}), '
            f'the merge having the same tree as PR head {head}; receipt theorem_count {count}, artifact sha256 {sha[:8]}…')


def rec_txt(n, mods_txt, prev_head, prev_run, prev_merge, prev_base, prev_count, prev_n, prev_sha, run, head, merge, base, base_pr, count, sha, **_):
    return (f'; the {n} theorems of {mods_txt}, added after PR head {prev_head} (run {prev_run}, actual tested merge {prev_merge} with base {prev_base}; receipt '
            f'theorem_count {prev_count}, corroborating the {prev_n} theorems above on that head; artifact lean-specification-proofs-{prev_run}-1, sha256 {prev_sha}), '
            f'are kernel-checked by the hosted run {run} on PR head {head} (actual tested merge {merge} with base {base}, the head of PR #{base_pr}; receipt '
            f'theorem_count {count}, leanchecker PASS; artifact lean-specification-proofs-{run}-1, sha256 {sha}) and were first receipted by that run, which '
            f'receipts all {count} theorems of that head; same tree as the head')


def rec_short(n, mods_txt, prev_head, prev_run, prev_merge, run, head, merge, base, base_pr, **_):
    return (f'; the {n} of {mods_txt}, added after PR head {prev_head} (run {prev_run}, actual tested merge {prev_merge[:8]}…), by run {run}:\n'
            f'PR head {head}, actual tested merge {merge[:8]}… with base {base[:8]}…, the head of PR #{base_pr}, same tree as the head')


def rec_bullet(n, mods_txt, prev_head, prev_run, prev_merge, prev_count, run, head, merge, base, base_pr, count, **_):
    return (f'; the {n} theorems\n'
            f'  of {mods_txt}, added after PR head {prev_head} (run {prev_run}, actual tested merge {prev_merge[:8]}…, receipt theorem_count {prev_count}), are '
            f'receipted by run {run}: PR head {head},\n'
            f'  actual tested merge {merge[:8]}… with base {base[:8]}…, the head of PR #{base_pr}, receipt theorem_count {count}, and were first receipted by that run')


def rec_scope_end(n, mods_txt, run, head, merge, base, base_pr, count, sha, **_):
    return (f'; the {n} of {mods_txt} by run {run}: PR head {head}, actual tested merge {merge[:8]}… with base\n'
            f'{base[:8]}…, the head of PR #{base_pr}; receipt theorem_count {count}; artifact lean-specification-proofs-{run}-1, sha256 {sha[:8]}…')


R4 = dict(n=N4, mods_txt=MODS_TXT, mods_plain=MODS_PLAIN, prev_head=C7, prev_run=RUN7, prev_merge=M7, prev_base=BASE2, prev_count=60,
          prev_n=len(RECEIPTED60), prev_sha=SHA7, run=RUN8, head=C8, merge=M8, base=BASE3, base_pr=423, count=73, sha=SHA8)
R5 = dict(n=N5, mods_txt=MODS5_TXT, mods_plain=MODS5_PLAIN, prev_head=C9, prev_run=RUN9, prev_merge=M9, prev_base=BASE3, prev_count=73,
          prev_n=len(RECEIPTED73), prev_sha=SHA9, run=RUN10, head=C10, merge=M10, base=BASE4, base_pr=434, count=75, sha=SHA10)
assert HEADER_BIND9 == HEADER_BIND6 + hdr_tail(**R4) and REC4_TXT == rec_txt(**R4)
HEADER_BIND11 = HEADER_BIND9 + hdr_tail(**R5)
REC5_TXT = rec_txt(**R5)

# prose phrases of render_docs11 (verbatim): tier-3/4 phrases, the round-5 pending (†) phrases of render_docs10, the round-5 receipted ones
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
REC5_SCOPE = ''
PEND3_SCOPE_END = (f'; the {N3} of `P15Interpolation` by run {RUN6}: PR head {C6}, actual tested merge {M6[:8]}… with base\n'
                   f'{BASE2[:8]}…, the head of PR #422')
REC4_SCOPE_END = (f'; the {N4} of {MODS_TXT} by run {RUN8}: PR head {C8}, actual tested merge {M8[:8]}… with base\n'
                  f'{BASE3[:8]}…, the head of PR #423; receipt theorem_count 73; artifact lean-specification-proofs-{RUN8}-1, sha256 {SHA8[:8]}…')
PEND5_SCOPE_END = (f'; theorems added after PR head {C9} (run {RUN9}, actual tested merge {M9[:8]}… with base {BASE3[:8]}…) await the next hosted\n'
                   '`lean-specification-proofs.yml` run, their first receipt')
assert (REC4_SHORT, REC4_BULLET, REC4_SCOPE_END) == (rec_short(**R4), rec_bullet(**R4), rec_scope_end(**R4))
REC5_SHORT, REC5_BULLET, REC5_SCOPE_END = rec_short(**R5), rec_bullet(**R5), rec_scope_end(**R5)


# templates of the PENDING tier forms, pinned below (step 0 and the asserts after it) to render_docs10's committed PEND5_* forms
def pend_hdr(head, run, merge, **_):
    return f'; † = added after PR head {head} (run {run}, actual tested merge {merge[:8]}…), no hosted receipt yet'


def pend_txt(n, mods_txt, head, last, run, merge, base, count, relation, sha, **_):
    return (f'; the {n} theorems marked † ({mods_txt}) were added after PR head {head}, {last} (run {run}, actual tested merge {merge} '
            f'with base {base}; receipt theorem_count {count}, {relation}; artifact lean-specification-proofs-{run}-1, '
            f'sha256 {sha}), and have no hosted receipt yet: the next hosted `lean-specification-proofs.yml` run is their first receipt')


def pend_short(n, mods_txt, head, last, run, merge, **_):
    return (f'; the {n} of {mods_txt}, added after PR head {head}, {last} (run {run}, actual tested merge {merge[:8]}…),\n'
            'have no hosted receipt yet and are marked † in the tables')


def pend_bullet(n, mods_txt, head, run, merge, count, **_):
    return (f'; the {n} theorems\n'
            f'  of {mods_txt}, added after PR head {head} (run {run}, actual tested merge {merge[:8]}…, receipt theorem_count {count}), have no hosted receipt yet, marked †')


def pend_scope(head, run, merge, **_):
    return (f', where † marks\n'
            f'a theorem added after PR head {head} (run {run}, actual tested merge {merge[:8]}…) with no hosted receipt yet')


def pend_scope_end(head, run, merge, base, **_):
    return (f'; theorems added after PR head {head} (run {run}, actual tested merge {merge[:8]}… with base {base[:8]}…) await the next hosted\n'
            '`lean-specification-proofs.yml` run, their first receipt')


LAST5 = 'the last head with a hosted receipt'
LAST6 = 'the last head whose receipt the register binds'          # the register's PENDING6 wording ('this register binds'), in the docs' voice
assert 'the last head whose receipt this register binds' in UPDATE13['phrases']['PENDING6']
P5 = dict(n=N5, mods_txt=MODS5_TXT, head=C9, last=LAST5, run=RUN9, merge=M9, base=BASE3, count=73,
          relation=f'corroborating the {len(RECEIPTED73)} theorems above on that head', sha=SHA9)
P6 = dict(n=N6, mods_txt=MODS6_TXT, head=C10, last=LAST6, run=RUN10, merge=M10, base=BASE4, count=75,
          relation=f'naming the {len(RECEIPTED75)} theorems above on that head and not these', sha=SHA10)
assert HEADER_BIND10 == HEADER_BIND9 + pend_hdr(**P5) and PEND5_TXT == pend_txt(**P5)
assert (PEND5_SHORT, PEND5_BULLET, PEND5_SCOPE, PEND5_SCOPE_END) == (pend_short(**P5), pend_bullet(**P5), pend_scope(**P5), pend_scope_end(**P5))
HEADER_BIND13 = HEADER_BIND11 + pend_hdr(**P6)
PEND6_TXT = pend_txt(**P6)
PEND6_SHORT, PEND6_BULLET, PEND6_SCOPE, PEND6_SCOPE_END = pend_short(**P6), pend_bullet(**P6), pend_scope(**P6), pend_scope_end(**P6)
for p in (HEADER_BIND13[len(HEADER_BIND11):], PEND6_TXT, PEND6_SHORT, PEND6_BULLET, PEND6_SCOPE, PEND6_SCOPE_END):
    assert RUN10 in p and C10 in p and M10[:8] in p and RUN9 not in p and 'kernel-checked' not in p and 'proved' not in p, p


def build(rows, pending, receipted, header_bind, pend_txt):
    """The four regenerated section bodies (README decl, README claims, SCOPE decl, SCOPE claims) of a register, render_docs10/11's code."""
    decls = collect(rows)
    keys = sorted(decls, key=lambda k: order[k])
    assert len(keys) == 150, len(keys)
    n_def, NP, NT = counts(decls)
    n_struct = sum(1 for k in keys if decls[k]['kind'] == 'structure')
    n_aux = sum(1 for r in scan if r['kind'] is None)
    proof_theorems = {t for k in keys for t in decls[k]['proofs']}
    assert pending <= proof_theorems and proof_theorems - pending <= receipted and not (pending & receipted)
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
    n_total = sum(len(r['statements']) for r in rows)
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
             'entries': {'total': n_total, 'local': n_local_entries, 'side24': n_side24, 'p15': n_p15, 'proof': n_proof, 'proved_total': n_proved},
             'claim_formal_progress': dict(fps), 'claim_rows': len(rows), 'statements_cell': statements_cell, 'other_rows': other_rows}
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


def render_all(rows, pending, receipted, header_bind, pend_txt, base_docs):
    rd, rc, sd, sc, stats = build(rows, pending, receipted, header_bind, pend_txt)
    bodies = {'README.md': (rd, rc), 'SCOPE.md': (sd, sc)}
    out = {}
    for name in ('README.md', 'SCOPE.md'):
        text = base_docs[name]
        for (start, end), body in zip(HEADINGS[name], bodies[name]):
            text = splice(text, start, end, body)
        out[name] = text
    return out, bodies, stats


# ----------------------------------------------------------------------------- 0. PIN THE PENDING SHAPE on the committed round-5 pending docs
repro5, _, st5 = render_all(reg5['claims'], T5, RECEIPTED73, HEADER_BIND10, PEND3_TXT + REC4_TXT + PEND5_TXT, COMMITTED_R5)
for name in ('README.md', 'SCOPE.md'):
    assert repro5[name] == COMMITTED_R5[name], f'{name}: the renderer does not reproduce the committed c6ce0256 text; refusing to render'
for name, phrases in (('README.md', (PEND5_SHORT, PEND5_BULLET)), ('SCOPE.md', (PEND5_SCOPE, PEND5_SCOPE_END))):
    for p in phrases:
        assert COMMITTED_R5[name].count(p) == 1, (name, p[:60])
    assert COMMITTED_R5[name].count(HEADER_BIND10) == len(MODULES) and COMMITTED_R5[name].count(PEND5_TXT) == 2, name
print('step 0: c6ce0256 README.md and SCOPE.md (round-5 pending tier) reproduced byte-identically; PEND5_* forms pinned; '
      'pending templates reproduce them')

# ----------------------------------------------------------------------------- 1. REPRODUCE the committed 4afcca6a docs from the committed register
repro, old_bodies, old_stats = render_all(old_reg['claims'], set(), RECEIPTED75, HEADER_BIND11, PEND3_TXT + REC4_TXT + REC5_TXT, COMMITTED)
for name in ('README.md', 'SCOPE.md'):
    assert repro[name] == COMMITTED[name], f'{name}: the renderer does not reproduce the committed 4afcca6a text; refusing to render'
assert (old_stats['NP'], old_stats['NT'], old_stats['n_def']) == (OLD_NP, OLD_NT, OLD_NDEF)
assert old_stats['entries']['total'] == UPDATE13['before']['statements'] == 354 and old_stats['entries']['proved_total'] == UPDATE13['before']['proved'] == 157
print('step 1: 4afcca6a README.md and SCOPE.md reproduced byte-identically from the committed register (no hand edits to fold in)')

# ----------------------------------------------------------------------------- 2. RENDER the round-13 state (round-6 theorems pending †)
rendered, bodies, st = render_all(new_reg['claims'], T6, RECEIPTED75, HEADER_BIND13, PEND3_TXT + REC4_TXT + REC5_TXT + PEND6_TXT, COMMITTED)
NP, NT, n_def = st['NP'], st['NT'], st['n_def']
assert (NP, NT, n_def) == (77, 78, 141), (NP, NT, n_def)            # FACTS: source-mode inventory 78 theorems, 77 distinct Props
assert st['dups'] == old_stats['dups'] == ['UniversalLaw.Spec.P15.TriangleNeedsThree']
assert st['entries']['total'] == UPDATE13['after']['statements'] == 357 and st['entries']['proved_total'] == UPDATE13['after']['proved'] == 160
assert st['entries']['proof'] == old_stats['entries']['proof'] + 3 and st['entries']['local'] == old_stats['entries']['local']
assert st['claim_formal_progress'] == old_stats['claim_formal_progress'] == {'specified': 33, 'none': 138}


def once(text, old, new):
    assert text.count(old) == 1, (old[:80], text.count(old))
    return text.replace(old, new)


README_TOP_OLD = (f'Props ({OLD_NP} of the {OLD_NDEF} Props, {OLD_NT} theorems); the register lists them under the claim rows as\n'
                  f'statement entries of kind `theorem` with label `proved` ({len(RECEIPTED52)} of them kernel-checked by the hosted run {RUN2}: PR head {C2},\n'
                  f'actual tested merge {M2[:8]}… with base {BASE[:8]}…, same tree as the head{PEND3_SHORT}{REC4_SHORT}{REC5_SHORT}),\n')
README_HOLDS_OLD = f'  packet by path and holds {OLD_NT} theorems whose type is exactly one of {OLD_NP} registered Props (P15, Side24\n'
README_BULLET_OLD = (f'  appears only in that packet\'s hosted run receipts (`lean-specification-proofs.yml`; run {RUN2} receipted the 52\n'
                     f'  theorems of PR head {C2} and run {RUN1} the first 38; each receipt\'s checked_commit is the synthetic merge of the PR head into the\n'
                     f'  base that a pull_request run checks out, not the head itself: run {RUN2} PR head {C2}, actual tested merge {M2[:8]}…; run {RUN1}\n'
                     f'  PR head {C1}, actual tested merge {M1[:8]}…; both with base {BASE[:8]}… and each with the same tree as its head{PEND3_BULLET}{REC4_BULLET}{REC5_BULLET}).\n')
SCOPE_BOUNDARY_OLD = (f'`specifications/lean_statement_proofs_20261011` ({OLD_NT} theorems typed exactly as {OLD_NP} of the {OLD_NDEF} Props;\n'
                      f'the last column of the declaration tables and the `proved (proof packet …)` entries of the claims table{REC5_SCOPE});\n')
SCOPE_END_OLD = (f'identity), never about the claim row it sits under; `kernel-checked` is a run-receipt word (the proof packet\'s\n'
                 f'hosted run {RUN2} receipted the 52 theorems of PR head {C2}: actual tested merge {M2[:8]}… with base {BASE[:8]}…, the\n'
                 f'synthetic merge a pull_request run checks out, same tree as the head{PEND3_SCOPE_END}{REC4_SCOPE_END}{REC5_SCOPE_END}).')
OLD_COUNTS = f'({OLD_NP} of the {OLD_NDEF} Props, {OLD_NT} theorems)'
NEW_COUNTS = f'({NP} of the {n_def} Props, {NT} theorems)'
PROSE = {   # each new form is built from its old form by exactly the listed edits (counts asserted)
    'README.md': [
        (README_TOP_OLD, once(once(README_TOP_OLD, OLD_COUNTS, NEW_COUNTS), f'{REC5_SHORT}),\n', f'{REC5_SHORT}{PEND6_SHORT}),\n')),
        (README_HOLDS_OLD, once(README_HOLDS_OLD, f'holds {OLD_NT} theorems whose type is exactly one of {OLD_NP} registered Props',
                                f'holds {NT} theorems whose type is exactly one of {NP} registered Props')),
        (README_BULLET_OLD, once(README_BULLET_OLD, f'{REC5_BULLET}).\n', f'{REC5_BULLET}{PEND6_BULLET}).\n')),
    ],
    'SCOPE.md': [
        (SCOPE_BOUNDARY_OLD, once(once(SCOPE_BOUNDARY_OLD, f'({OLD_NT} theorems typed exactly as {OLD_NP} of the {OLD_NDEF} Props;',
                                       f'({NT} theorems typed exactly as {NP} of the {n_def} Props;'),
                                  'entries of the claims table);\n', f'entries of the claims table{PEND6_SCOPE});\n')),
        (SCOPE_END_OLD, once(SCOPE_END_OLD, f'{REC5_SCOPE_END}).', f'{REC5_SCOPE_END}{PEND6_SCOPE_END}).')),
    ],
}
PEND6_PROSE = {'README.md': (PEND6_SHORT, PEND6_BULLET), 'SCOPE.md': (PEND6_SCOPE, PEND6_SCOPE_END)}
for name in PROSE:   # every replacement keeps every earlier binding of its anchor; the round-5 pending forms never come back
    for old, new in PROSE[name]:
        assert new != old, (name, old[:60])
        for phrase in (PEND3_SHORT, REC4_SHORT, REC5_SHORT, PEND3_BULLET, REC4_BULLET, REC5_BULLET, PEND3_SCOPE_END, REC4_SCOPE_END, REC5_SCOPE_END,
                       RUN2, RUN1, RUN6, RUNA, RUN7, RUN8, RUN9, RUN10):
            assert phrase not in old or phrase in new, (name, phrase[:60])
        for phrase in (PEND5_SHORT, PEND5_BULLET, PEND5_SCOPE, PEND5_SCOPE_END, LAST5):
            assert phrase not in new, (name, phrase[:60])
        assert '†' not in old and 'no hosted receipt' not in old and 'await the' not in old, (name, old[:60])
    assert sum(new.count(p) for _, new in PROSE[name] for p in PEND6_PROSE[name]) == 2, name

PAIRS = tuple((run, merge[:8]) for run, merge in RUN_MERGE)
NONE_YET_REGISTER = sum(r['next_step'].count('none yet') + r['not_established'].count('none yet') for r in new_reg['claims'])   # 'proof lane: none yet', not a receipt statement
assert NONE_YET_REGISTER == 1 and 'Hosted receipt: none yet' not in json.dumps(new_reg)
assert 'the last head with a hosted receipt' not in json.dumps(new_reg) and '†' not in json.dumps(new_reg)
NO_RECEIPT_CELLS = sum(cell(r['not_established']).count('no hosted receipt') + cell(r['next_step']).count('no hosted receipt') for r in st['other_rows'])
T6_SHORTS = [t.split('.', 3)[-1] for t in T6_EXPECTED.values()]                     # ImageBound, EntrywiseCovarianceBound, CovarianceOrdering
T6_CELLS = {t.split('.', 3)[-1]: '`' + p.split('.', 2)[-1] + '†`' for p, t in T6_EXPECTED.items()}   # Prop short name -> its proofs cell
old_by_id = {r['claim_id']: r for r in old_reg['claims']}
new_by_id = {r['claim_id']: r for r in new_reg['claims']}


def check_section_lines(name, idx, old_body, new_body):
    """Every changed line of a regenerated section is enumerated, and differs from the committed line by exactly the round-6 edits."""
    ol, nl = old_body.split('\n'), new_body.split('\n')
    assert len(ol) == len(nl), (name, idx, len(ol), len(nl))
    seen = Counter()
    for o, n in zip(ol, nl):
        if o == n:
            continue
        if o.startswith('| declaration |'):                                         # table header: the pending legend only
            assert once(o, HEADER_BIND11, HEADER_BIND13) == n, (name, o[-120:])
            seen['header'] += 1
        elif idx == 0 and o == old_stats['intro']:                                  # declarations intro
            e = once(once(o, f'{OLD_NP} of the {OLD_NDEF} Props have one ({OLD_NT} theorems', f'{NP} of the {n_def} Props have one ({NT} theorems'),
                     REC5_TXT, REC5_TXT + PEND6_TXT)
            assert e == n == st['intro'], name
            seen['intro'] += 1
        elif idx == 1 and o == old_stats['claim_intro']:                            # claims intro
            oe, ne = old_stats['entries']['proved_total'], st['entries']['proved_total']
            e = once(once(once(o, f'`kernel-checked`. {old_stats["entries"]["local"]} statement entries point at', f'`kernel-checked`. {st["entries"]["local"]} statement entries point at'),
                          f'. {oe} statement entries carry the label `proved`', f'. {ne} statement entries carry the label `proved`'),
                     f'({OLD_NT} theorems, {OLD_NP} distinct Props,', f'({NT} theorems, {NP} distinct Props,')
            e = once(e, f'; {old_stats["entries"]["proof"]} are theorems of the proof packet', f'; {st["entries"]["proof"]} are theorems of the proof packet')
            e = once(e, REC5_TXT, REC5_TXT + PEND6_TXT)
            assert e == n == st['claim_intro'], name
            seen['claim_intro'] += 1
        elif idx == 0 and any(o.startswith(f'| `{s}` | def_prop |') for s in T6_SHORTS):   # the three Props' rows: '—' -> the dagger-marked theorem
            s = next(s for s in T6_SHORTS if o.startswith(f'| `{s}` | def_prop |'))
            assert o.endswith(' | — |') and n == o[:-len('— |')] + T6_CELLS[s] + ' |', (name, o[:80])
            seen['prop_row'] += 1
        elif idx == 1 and any(o.startswith(f'| `{cid}` |') for cid in CHANGED_ROWS):   # the claim rows the register update changed
            cid = next(c for c in CHANGED_ROWS if o.startswith(f'| `{c}` |'))
            e = o
            if cid == T6_ROW:
                k_old = sum(1 for s in old_by_id[cid]['statements'] if s['package'] == PROOF_PACKAGE)
                e = once(e, f'; {k_old} proved (proof packet): ', f'; {k_old + 3} proved (proof packet): ')
                last_old = '`' + [s for s in old_by_id[cid]['statements'] if s['package'] == PROOF_PACKAGE][-1]['declaration'].split('.', 2)[-1] + '`'
                e = once(e, last_old + ' |', last_old + ', ' + ', '.join(T6_CELLS[t.split('.', 3)[-1]] for t in T6_EXPECTED.values()) + ' |')
            if name == 'SCOPE.md':                                                    # register cells transcribed verbatim
                for f in ('not_established', 'next_step'):
                    if old_by_id[cid][f] != new_by_id[cid][f]:
                        e = once(e, ' | ' + cell(old_by_id[cid][f]) + ' |', ' | ' + cell(new_by_id[cid][f]) + ' |')
            assert e == n, (name, cid)
            seen['claim_row:' + cid] += 1
        else:
            raise AssertionError(f'{name} section {idx}: unexpected changed line: {o[:120]!r}')
    return seen


summary = {'scientific_effect': 'NONE', 'reproduction_of_round5_pending_docs_c6ce0256': 'byte-identical',
           'reproduction_of_committed_docs_4afcca6a': 'byte-identical', 'hand_edits_folded_in': [], 'files': {}}
for name in ('README.md', 'SCOPE.md'):
    orig = COMMITTED[name]
    if '†' in orig or 'no hosted receipt' in orig:
        sys.exit(f'REFUSED: {name} already carries a pending (†) form: the committed docs are not the round-11 ones')
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
    # inside the regenerated sections: every changed line enumerated and exactly edited
    seen = Counter()
    for idx in (0, 1):
        seen.update(check_section_lines(name, idx, old_bodies[name][idx], bodies[name][idx]))
    exp_seen = {'header': len(MODULES), 'intro': 1, 'claim_intro': 1, 'prop_row': 3, 'claim_row:' + T6_ROW: 1}
    if name == 'SCOPE.md':
        exp_seen.update({'claim_row:' + c: 1 for c in CHANGED_ROWS if c in [r['claim_id'] for r in st['other_rows']] and c != T6_ROW
                         and any(old_by_id[c][f] != new_by_id[c][f] for f in ('not_established', 'next_step'))})
    assert dict(seen) == exp_seen, (name, dict(seen), exp_seen)
    assert not re.search(r'<<[^<>\n]*>>', text), name
    for phrase in ('Scientific effect', 'NONE', 'PENDING_INDEPENDENT_REVIEW', PEND3_TXT, REC4_TXT, REC5_TXT, PEND6_TXT, HEADER_BIND9, HEADER_BIND11,
                   HEADER_BIND13, RUN10, M10, SHA10, BASE4, RUN9, M9) + PEND6_PROSE[name]:
        assert phrase in text, (name, phrase[:60])
    # the round-5 pending forms stay gone; the round-6 pending forms are the only ones
    for phrase in (PEND5_TXT, PEND5_SHORT, PEND5_BULLET, PEND5_SCOPE, PEND5_SCOPE_END, HEADER_BIND10, LAST5, 'Hosted receipt: none yet'):
        assert phrase not in text, (name, phrase[:60])
    # the register's own pending sentences (PENDING6 'first receipt for them', 'whose receipt this register binds') appear only where
    # SCOPE transcribes the register cells verbatim; README transcribes no not_established / next_step cell
    for phrase in ('first receipt for them', 'whose receipt this register binds'):
        n_cells = sum(cell(r['not_established']).count(phrase) + cell(r['next_step']).count(phrase) for r in st['other_rows'])
        assert n_cells >= 1 and text.count(phrase) == (n_cells if name == 'SCOPE.md' else 0), (name, phrase, text.count(phrase), n_cells)
    assert text.count('none yet') == (NONE_YET_REGISTER if name == 'SCOPE.md' else 0), (name, text.count('none yet'))
    # 'no hosted receipt': the 5 header legends, the two intros, the prose clause(s), and (SCOPE) the register's own cells
    n_nhr = len(MODULES) + 2 + sum(p.count('no hosted receipt') for p in PEND6_PROSE[name]) + (NO_RECEIPT_CELLS if name == 'SCOPE.md' else 0)
    assert text.count('no hosted receipt') == n_nhr, (name, text.count('no hosted receipt'), n_nhr)
    # C422-E1: every mention of a hosted run binds its actual tested merge, per paragraph and per table cell; runs 38180705952,
    # 38187538744 and 38195810463 also name their PR head wherever they appear; every receipted mention of 38195810463 names
    # its base (the pending legends, render_docs10's shape, name head and merge); no unlisted run id; no companion run cited
    for para in re.split(r'\n\s*\n', text):
        for run, merge in PAIRS:
            if run in para:
                assert merge in para, (name, run, para[:100])
        for run in (RUN8, RUN9, RUN10):
            if run in para:
                assert RUN_HEAD[run] in para, (name, run, para[:100])
        stripped = para.replace(pend_hdr(**P6), '').replace(PEND6_SCOPE, '')
        if RUN10 in stripped:
            assert BASE4[:8] in stripped, (name, para[:100])
    for line in text.split('\n'):
        if line.startswith('|'):
            for c in line.split(' | '):
                for run, merge in PAIRS:
                    assert run not in c or merge in c, (name, run, c[-160:])
                for run in (RUN8, RUN9, RUN10):
                    assert run not in c or RUN_HEAD[run] in c, (name, run, c[-200:])
    for found in set(re.findall(r'\b38\d{9}\b', text)):
        assert found in RUNS or found == WORKSPACE_LANDING_RUN, (name, found)
    for run in (RUN_STMT, RUN_AGG, RUN_STMT10, RUN_AGG10):
        assert run not in text, (name, run)
    # the † marks: one per pending theorem in the declaration tables (+ one per module table header and one in the intro),
    # one per pending statement entry in the claims table (+ one in the claims intro), and the prose clauses; nothing else
    decl_body, claim_body = bodies[name]
    assert st['intro'].count('†') == 1 and st['claim_intro'].count('†') == 1
    assert decl_body.count('†') == N6 + len(MODULES) + 1, (name, decl_body.count('†'))
    assert claim_body.count('†') == N_T6_ENTRIES + 1, (name, claim_body.count('†'))
    assert text.count('†') == decl_body.count('†') + claim_body.count('†') + sum(p.count('†') for p in PEND6_PROSE[name]), name
    for t in T6:
        assert decl_body.count('`' + t.split('.', 2)[-1] + '†`') == 1 and claim_body.count('`' + t.split('.', 2)[-1] + '†`') == 1, (name, t)
        assert ('`' + t.split('.', 2)[-1] + '`') not in text, (name, t)
    for t in RECEIPTED75:     # every receipted theorem keeps its committed, unmarked form
        assert ('`' + t.split('.', 2)[-1] + '†`') not in text and ('`' + t.split('.', 2)[-1] + '`') in decl_body, (name, t)
        assert text.count('`' + t.split('.', 2)[-1] + '`') == orig.count('`' + t.split('.', 2)[-1] + '`'), (name, t)
    assert text.count(HEADER_BIND13) == len(MODULES) and orig.count(HEADER_BIND11) == len(MODULES), name
    assert text.count(REC5_TXT + PEND6_TXT) == 2 and orig.count(REC5_TXT) == 2, name        # declarations intro and claims intro
    (DOCS_OUT / name).write_text(text, encoding='utf-8')
    summary['files'][name] = {'bytes': len(text.encode()), 'bytes_before': len(orig.encode()), 'dagger_marks': text.count('†'),
                              'dagger_marks_before': orig.count('†'), 'changed_lines_in_regenerated_sections': dict(seen),
                              'prose_edits': len(PROSE[name]), 'changed': text != orig, 'written_to': str(DOCS_OUT / name)}
    print(name, len(text.encode()), 'bytes', f'({text.count("†")} † marks; {orig.count("†")} before)')
summary.update({'props_with_proof': NP, 'proof_theorems': NT, 'def_props': n_def, 'pending_receipt': sorted(T6), 'pending_modules': list(T6_MODULES),
                'pending_statement_entries': N_T6_ENTRIES, 'pending_row': T6_ROW,
                'committed_register_counts': {'props': OLD_NP, 'theorems': OLD_NT, 'def_props': OLD_NDEF}, 'duplicates': st['dups'],
                'tiers': {'T1_T2_receipted_52': len(RECEIPTED52), 'T3_P15Interpolation': N3, 'T4_run_38180705952': N4, 'T5_run_38195810463': N5,
                          'T6_no_hosted_receipt_after_head_c6ce0256': N6},
                'entries_before': {k: v for k, v in old_stats['entries'].items()}, 'entries': st['entries'], 'claim_rows': st['claim_rows'],
                'claim_formal_progress': st['claim_formal_progress'], 'changed_register_rows': CHANGED_ROWS,
                'run_merge_bindings_asserted': dict(RUN_MERGE), 'run_head_bindings_asserted': RUN_HEAD,
                'last_bound_receipt': {'run': RUN10, 'pr': 439, 'pr_head': R5_PENDING_HEAD, 'actual_tested_merge': M10, 'base': BASE4, 'theorem_count': 75,
                                       'artifact_sha256_github_digest': SHA10, 'names_round6_theorems': False},
                'template_parameters_round6_vs_round5': {'last': [LAST5, LAST6], 'relation': [P5['relation'], P6['relation']]},
                'phrases': {'HEADER_BIND13_tail': pend_hdr(**P6), 'PEND6_TXT': PEND6_TXT, 'PEND6_SHORT': PEND6_SHORT, 'PEND6_BULLET': PEND6_BULLET,
                            'PEND6_SCOPE': PEND6_SCOPE, 'PEND6_SCOPE_END': PEND6_SCOPE_END}})
(OUT / 'render_summary13.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps({k: v for k, v in summary.items() if k != 'phrases'}, indent=1, ensure_ascii=False))
