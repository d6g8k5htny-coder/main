"""Round-14 re-render of the Declarations and Claims sections of README.md and SCOPE.md of the statement packet from
REGISTER.json after update_register14.py: the 3 round-6 proof-packet theorems of SpecificationProofs/Side24Images.lean
(committed in 9e305ea5)

    UniversalLaw.Proofs.Side24Images.imageBound               : UniversalLaw.Spec.Side24.ImageBound
    UniversalLaw.Proofs.Side24Images.entrywiseCovarianceBound : UniversalLaw.Spec.Side24.EntrywiseCovarianceBound
    UniversalLaw.Proofs.Side24Images.covarianceOrdering       : UniversalLaw.Spec.Side24.CovarianceOrdering

lose the dagger † ("no hosted receipt yet", render_docs13.py) and are shown RECEIPTED like the round-4 and round-5 theorems,
bound to their first hosted receipt: lean-specification-proofs.yml run 38277458954 on PR #441 head 9e305ea5, actual tested
merge 1376175ce5536a6ac74cb03706d18e65fdb695fa with base a0323b94ae54a78f3d2878dc9c358dc40cdfe1dc (Math- main at the
merge of PR #440), receipt theorem_count 78, leanchecker PASS, artifact lean-specification-proofs-38277458954-1 sha256
1f32d60e... (GitHub's digest; the artifact bytes were not downloaded). Derived from render_docs13.py (which reproduces the
committed 9e305ea5 docs from the committed register with the round-6 pending tier; its PEND6_* phrases are the pending forms
replaced here) and render_docs11.py (round 5's de-daggering: its receipted templates are the shape, its REC5_* forms are
what step 0 pins), in the way render_docs11 turned render_docs10's pending tier into a receipted one.

THE TESTED MERGE IS NOT THE HEAD'S TREE for this run (round 4's and round 5's were): the synthetic merge 1376175c (tree
0ccd93a2) differs from PR head 9e305ea5 (tree 95a33c52) only outside both packets (13 files under
companions/torus_filtration_20261011/ and .github/workflows/torus-filtration.yml, main's changes from PRs #428 and #440);
this packet and the proof packet are byte-identical between head and merge (asserted by update_register14.py from local git
objects). So every receipted round-6 form says that, and none says 'same tree' (asserted); the template parameters that
carry it (and the base description 'Math- main at the merge of PR #440', where rounds 4/5 named the head of the stacked
base PR) are asserted to reproduce the round-4/5 forms verbatim with the round-4/5 values first.

Method (fail closed):
  0. PIN THE RECEIPTED SHAPE: render both docs from the round-5 RECEIPTED register (read-only `git show 4afcca6a:...`) with
     render_docs11's phrases and assert byte-identity with the 4afcca6a README.md / SCOPE.md (render_docs11's output, as
     landed); assert REC5_SHORT, REC5_BULLET and REC5_SCOPE_END occur there exactly once. The receipted templates are
     asserted to reproduce every REC4_* / REC5_* form verbatim with the round-4/5 facts, and only then instantiated with
     the round-6 facts.
  1. REPRODUCE: render both docs from the COMMITTED register of 9e305ea5 (blob 84d0dd07, read-only `git show`) with
     render_docs13's phrases (the round-6 theorems pending †) and assert byte-identity with the committed 9e305ea5
     README.md / SCOPE.md (read-only `git show`); assert each round-6 pending prose clause (PEND6_SHORT, PEND6_BULLET,
     PEND6_SCOPE, PEND6_SCOPE_END) occurs there exactly once.
  2. RENDER from the worktree register (after update_register14.py; its git blob must equal update_summary14.json's
     output_blob) with no pending theorem, apply the exact-string PROSE edits (each round-6 pending clause -> its
     receipted successor; the SCOPE boundary sentence loses its dagger legend, as render_docs11 did for round 5), assert
     byte-for-byte preservation outside the two regenerated sections modulo exactly those edits, assert that inside the
     regenerated sections every changed line is one of an enumerated set and differs from the committed line exactly by
     the round-6 binding, and write.
No count moves (78 theorems, 77 Props, 141 `def … : Prop`, 357 statement entries, 160 `proved`). The worktree docs must equal
the committed ones before the render (refused otherwise); no git command that changes state is run (no checkout).

Scientific effect: NONE. The tables transcribe the register; they decide nothing. No claim-level label changes (claim
formal_progress stays none/specified); `proved` is the label of a committed proof package's theorem entry; `kernel-checked`
appears only in run receipts. A receipt is a record of a computation about the registered Props for the formal objects
only, never a review, an acceptance or an alignment, and never evidence about an informal claim. Every mention of a hosted
run binds the PR head AND the actual tested merge commit (C422-E1), asserted per paragraph and per table cell. No triage
verdict is altered. Same-account author: organizational-independence credit 0; alignment of every statement with its
anchor PENDING_INDEPENDENT_REVIEW; nonauthor review and acceptance are not claimed.

Usage: python3 -B -S render_docs14.py        (R14_DOCS_OUT=<dir> writes the two rendered docs there instead of the worktree;
                                             R14_REG=<file> reads the round-14 register from a dry-run file)
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
HERE = SCRATCH / 'lvs/register_update14'
OUT = HERE / 'sdocs'
OUT.mkdir(parents=True, exist_ok=True)
DOCS_OUT = Path(os.environ['R14_DOCS_OUT']) if os.environ.get('R14_DOCS_OUT') else PK
PROOF_PACKAGE = 'd6g8k5htny-coder/Math- specifications/lean_statement_proofs_20261011'
MAIN_SIDE24 = 'd6g8k5htny-coder/main formal/ (UniversalLaw, side24-arithmetic-skeleton)'
MAIN_P15 = 'd6g8k5htny-coder/main formal/ at 91cffbb51d1636f382e9d61a1ecb6f7cd38e1d06 (PR #358, not yet on main)'
EXPECTED_HEAD = '9e305ea5141a1d1503560efd0e3deaf25441749e'      # PR #441 head: the committed round-6 pending docs (render_docs13's output)
R5_RECEIPTED_HEAD = '4afcca6a79a8a0acef251ae73988a7914c8080e7'  # Math- main, the merge of PR #439: the round-5 receipted docs (render_docs11's output)
COMMITTED_REG_BLOB = '84d0dd07f4c0a11bc9f0c452b1e494bbabb6ceae'  # 9e305ea5:specifications/lean_statements_20261010/REGISTER.json
R5_REG_BLOB = '05cc80c1ed9040aa7282fb1543679743807d55ba'         # 4afcca6a:specifications/lean_statements_20261010/REGISTER.json
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
RUN10, C10, M10, BASE4, SHA10 = ('38195810463', 'c6ce0256', '1320c5b217e491d301ea140e5cc25b07ff5a21b7', 'be01479435f98f34b802258ee9ec3c3578f37169',
                                 'd8abd42be0facd4248647c3c7eb67a264bd19b9b47c6128733b5b5fc93b0a6ff')
RUN_STMT10, RUN_AGG10 = '38195810435', '38195810649'   # round-5 companion runs on the same head (statement packet; aggregate): never cited here
# round 14 (FACTS): first hosted receipt of the 3 round-6 theorems, PR #441 head 9e305ea5; base = Math- main at the merge of PR #440
RUN11, C11, M11, BASE5, SHA11 = ('38277458954', '9e305ea5', '1376175ce5536a6ac74cb03706d18e65fdb695fa', 'a0323b94ae54a78f3d2878dc9c358dc40cdfe1dc',
                                 '1f32d60e08b0ff11b634c0413291686982f0b02e6809b9e96af2848350cbbf79')
RUN_STMT11, RUN_AGG11 = '38277458488', '38277459272'   # round-6 companion runs on the same head and merge: never cited here
assert BASE3.startswith(C7) and BASE4.startswith(C9) and EXPECTED_HEAD.startswith(C11)
RUN_MERGE = ((RUN1, M1), (RUN2, M2), (RUN3, M3), (RUN4, M4), (RUN6, M6), (RUNA, M5A), (RUN7, M7), (RUN8, M8), (RUN9, M9), (RUN10, M10),
             (RUN11, M11))
RUN_HEAD = {RUN1: C1, RUN2: C2, RUN3: C3, RUN4: C4, RUN6: C6, RUNA: C5A, RUN7: C7, RUN8: C8, RUN9: C9, RUN10: C10, RUN11: C11}
for r, m in RUN_MERGE:
    assert r.isdigit() and len(r) == 11 and re.fullmatch(r'[0-9a-f]{40}', m), (r, m)
assert all(re.fullmatch(r'[0-9a-f]{64}', s) for s in (SHA7, SHA8, SHA9, SHA10, SHA11))
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
COMMITTED_R5 = {name: git('show', f'{R5_RECEIPTED_HEAD}:{PACKAGE}/{name}') for name in ('README.md', 'SCOPE.md')}
for name, text in COMMITTED.items():
    if (PK / name).read_text(encoding='utf-8') != text:
        sys.exit(f'REFUSED: worktree {name} differs from 9e305ea5 (already rendered?)')
# the statement packet's worktree differs from HEAD in REGISTER.json only; its checker, tests and modules are HEAD's
status = sorted(line for line in git('status', '--porcelain', '--', PACKAGE).splitlines() if line.strip())
if status != ([] if os.environ.get('R14_REG') else [f' M {PACKAGE}/REGISTER.json']):     # a dry run leaves the worktree register untouched
    sys.exit(f'REFUSED: unexpected worktree state of the statement packet: {status}')
assert git('rev-parse', f'{EXPECTED_HEAD}:{PACKAGE}/REGISTER.json').strip() == COMMITTED_REG_BLOB
assert git('rev-parse', f'{R5_RECEIPTED_HEAD}:{PACKAGE}/REGISTER.json').strip() == R5_REG_BLOB
# step 0 renders the 4afcca6a docs with the HEAD checker and modules: they are the same blobs there
for f in ['check.py'] + [line.split('\t', 1)[1] for line in git('ls-tree', '-r', EXPECTED_HEAD, '--', PACKAGE).splitlines()
                         if line.split('\t', 1)[1].endswith('.lean')]:
    f = f[len(PACKAGE) + 1:] if f.startswith(PACKAGE + '/') else f
    assert git('rev-parse', f'{EXPECTED_HEAD}:{PACKAGE}/{f}') == git('rev-parse', f'{R5_RECEIPTED_HEAD}:{PACKAGE}/{f}'), f


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def git_blob(data):
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


check = load_module('stmt_check_r14', PK / 'check.py')
RECEIPTED52 = set(json.loads((SCRATCH / 'lvs/register_update5/receipted_52.json').read_text())['all_52'])
assert len(RECEIPTED52) == 52
MODULES = list(check.MODULES)
UPDATE14 = json.loads((HERE / 'update_summary14.json').read_text(encoding='utf-8'))
if os.environ.get('R14_REG'):
    REG_IN = Path(os.environ['R14_REG'])     # dry runs read a register written by update_register14.py's R14_OUT
else:
    REG_IN = PK / 'REGISTER.json'
    assert git_blob(REG_IN.read_bytes()) == UPDATE14['output_blob'], 'the worktree register is not update_register14.py\'s output'
assert git_blob(REG_IN.read_bytes()) == UPDATE14['output_blob']
assert UPDATE14['input_blob'] == COMMITTED_REG_BLOB and UPDATE14['committed_head'] == EXPECTED_HEAD
new_reg = json.loads(REG_IN.read_text(encoding='utf-8'))
old_reg = json.loads(git('show', f'{EXPECTED_HEAD}:{PACKAGE}/REGISTER.json'))          # round 6 pending (render_docs13's input)
reg5 = json.loads(git('show', f'{R5_RECEIPTED_HEAD}:{PACKAGE}/REGISTER.json'))         # round 5 receipted (render_docs11's input)
assert RUN11 not in json.dumps(old_reg) and RUN11 in json.dumps(new_reg) and RUN11 not in json.dumps(reg5)
assert 'UniversalLaw.Proofs.Side24Images' in json.dumps(old_reg) and 'UniversalLaw.Proofs.Side24Images' not in json.dumps(reg5)
assert [c['claim_id'] for c in old_reg['claims']] == [c['claim_id'] for c in new_reg['claims']] == [c['claim_id'] for c in reg5['claims']]
assert [c['formal_progress'] for c in old_reg['claims']] == [c['formal_progress'] for c in new_reg['claims']]
assert set(c['formal_progress'] for c in new_reg['claims']) <= {'none', 'specified'}
# the binding changes no statement entry: every statements array is identical to the committed one
for o, n in zip(old_reg['claims'], new_reg['claims']):
    assert n['statements'] == o['statements'], n['claim_id']
CHANGED_ROWS = [n['claim_id'] for o, n in zip(old_reg['claims'], new_reg['claims']) if o != n]
assert sorted(CHANGED_ROWS) == sorted(UPDATE14['changed_fields']), CHANGED_ROWS
sources, _ = check.load_sources(PK, None, new_reg)


