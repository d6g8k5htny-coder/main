"""Round-14 register update: bind the three round-6 proof-packet theorems

    UniversalLaw.Proofs.Side24Images.imageBound               : UniversalLaw.Spec.Side24.ImageBound
    UniversalLaw.Proofs.Side24Images.entrywiseCovarianceBound : UniversalLaw.Spec.Side24.EntrywiseCovarianceBound
    UniversalLaw.Proofs.Side24Images.covarianceOrdering       : UniversalLaw.Spec.Side24.CovarianceOrdering

(module SpecificationProofs/Side24Images.lean, committed in 9e305ea5 on top of Math- main 4afcca6a) to their FIRST hosted
receipt, run 38277458954 on PR #441 head 9e305ea5.

Derived from update_register11.py (round 5's binding, committed as 32fc9845: pending tier -> receipted tier by exact-string
replacement, one corroboration clause per proof-packet kernel_evidence row, the 'await the next hosted ...' receipt clause ->
the receipted binding, the 'no hosted receipt yet' parentheticals removed, and its invariants) and update_register13.py
(whose PENDING6_PHRASE, WF_PENDING6, receipt clause, narrowed 'each of those 19 theorems' sentence, three triage-item
clauses and README-reasons clause are the pending forms replaced here, verbatim). The receipted phrases are NOT retyped:
each is produced by a template first asserted to reproduce, with the round-4 and round-5 facts, the round-4 and round-5
phrases exactly as committed in 9e305ea5, and then instantiated with the round-6 facts. Three template parameters carry
wording the round-6 facts force:
  * `added`: round 6's own provenance wording from PENDING6 / WF_PENDING6 / the receipt clause ('added in round 6 on top of
    Math- main 4afcca6a, the merge of PR #439, after PR head c6ce0256' in the tier phrase, 'added in round 6 after PR head
    c6ce0256' in the WF clause and the receipt item), where rounds 4/5 said 'added after PR head <h>';
  * `base_desc`: 'Math- main at the merge of PR #440', where rounds 4/5 said 'the head of PR #<n>' (PR #441's base branch is
    main; PRs #434 and #439 were stacked);
  * `tree`: THE MERGE IS NOT THE HEAD'S TREE for this run. Rounds 4/5 said 'the merge has the same tree as its head'; here
    the synthetic merge 1376175c (tree 0ccd93a2) differs from PR head 9e305ea5 (tree 95a33c52) only outside both packets: 13
    files under companions/torus_filtration_20261011/ and .github/workflows/torus-filtration.yml, main's changes from PRs
    #428 and #440 (4afcca6a..a0323b94); both packets are byte-identical between head and merge (asserted below from local
    git objects, read-only). The phrase 'same tree' never occurs in a round-6 phrase (asserted).
A fourth parameter, `caveat`, is '' for rounds 4/5 (so the pinned round-4/5 phrases reproduce verbatim) and for round 6
is CAVEAT6 = ', the tested merge differing from the head only outside both packets' (never a bare 'the merge differing',
which would collide with 'Math- main at the merge of PR #440'). It sits right after the base inside the first parenthesis
of the corroboration clauses (corrob, wf_corrob), of the next_step receipt item (rc_item) and of its narrowed tail
(rc_tail_each: 'each of those 22 theorems receipted on PR head 9e305ea5 ...', where round 5 said 'every theorem of this
row', which would overclaim: row side24-coefficient also holds the 27 main-package theorems, which that run did not
receipt). So run 38277458954 is never named in a leaf, nor in any phrase, without either the short caveat or the long
tree clause TREE6 (asserted per phrase and per string leaf).

Input: the committed register 9e305ea5:specifications/lean_statements_20261010/REGISTER.json, blob 84d0dd07..., sha256
67f7fb1e... (= the receipt's register_sha256), asserted equal to the worktree copy.

Scientific effect: NONE. Edits, and nothing else (asserted as exact expected strings per field):
  (a) the 10 proof-packet kernel_evidence rows: in row side24-coefficient the round-6 pending sentence of the note
      (PENDING6_PHRASE) becomes the receipted tier phrase naming run 38277458954, PR head 9e305ea5, actual tested merge
      1376175c... with base a0323b94..., receipt theorem_count 78, leanchecker PASS, artifact sha256 1f32d60e..., the tree
      difference, and its receipt_or_workflow clause WF_PENDING6 becomes the receipted WF clause; every one of the 10 rows
      has theorems of earlier tiers and gains one corroboration clause in the note and one in receipt_or_workflow (all 78
      theorems of PR head 9e305ea5 receipted on that head by that run), in the round-5 shape plus CAVEAT6 after the base;
  (b) not_established of side24-coefficient and of math.side24-coefficient: PENDING6_PHRASE -> the receipted tier phrase;
      next_step of side24-coefficient: the 'the 3 theorems of this row added in round 6 ... await the next hosted ... their
      first receipt; the other theorems of this row have a hosted receipt: ' clause -> 'every theorem of this row has a
      hosted receipt: the 3 added in round 6 after PR head c6ce0256 (...) first by run 38277458954 (PR head 9e305ea5;
      actual tested merge ... with base ..., the tested merge differing from the head only outside both packets) ...'
      (round 5's rc_sub shape plus CAVEAT6), one 'each of those 22 theorems receipted on PR head 9e305ea5 by run
      38277458954 (... with base ..., the tested merge differing ...; receipt theorem_count 78)' tail after the narrowed
      'each of those 19 theorems ...' sentence (round 6's own narrowed shape; 'every theorem of this row receipted on PR
      head 9e305ea5' never occurs, asserted), and the 'no hosted receipt yet' parentheticals that round 6 put in
      the three triage items and the README-reasons sentence are removed (round 5's bound items read 'proved in round 5
      (2026-10-12; see entries)' with no receipt word; the receipt is stated once, in the receipt clause of the same field);
  (c) nothing else: statements arrays, every formal_progress (claim level and entry level), every other field and every
      other kernel_evidence entry byte-identical; no triage verdict or effort altered.
No claim-level label changes; `proved` only on theorem entries; `kernel-checked` only as a receipt word; every mention of
a hosted run names the PR head AND the actual tested merge (C422-E1), asserted per string leaf. A receipt is a record of a
computation about the registered Props for the formal objects only, never a review, an acceptance or an alignment, and
never evidence about an informal claim. Same-account author: organizational-independence credit 0; alignment
PENDING_INDEPENDENT_REVIEW.

Run 38195810463 (PR head c6ce0256) stays named: it is round 5's receipt and the corroboration of the tiers before it; only
the two round-6 pending clauses that named it as 'the last head whose receipt this register binds' / 'does not name them'
go (asserted count).

FACTS (coordinator-verified 2026-10-12 from the job logs, the Actions REST API and local git objects; re-read here with
read-only `gh api` (run head_sha, event, conclusion, attempt, PR #441 head/base, artifact name/id/size/digest) and read-only
git (merge parents and trees, head..merge diff); the artifact bytes were NOT downloaded): workflow
lean-specification-proofs.yml run 38277458954, attempt 1, job 114906310832 (proof-evidence, success), event pull_request
on PR #441 head 9e305ea5141a1d1503560efd0e3deaf25441749e; checked_commit = synthetic merge
1376175ce5536a6ac74cb03706d18e65fdb695fa, parents a0323b94ae54a78f3d2878dc9c358dc40cdfe1dc (Math- main at the merge of PR
#440) and the head; receipt formalization_status kernel-checked, leanchecker PASS, theorem_count 78, covered_props 77,
worktree CLEAN, passed true, negative controls refused for their intended reasons and honest controls accepted,
register_sha256 67f7fb1e... (= this input); Lean 4.34.1; artifact lean-specification-proofs-38277458954-1 (id 11759872846,
29 files, 47041 bytes), GitHub digest sha256 1f32d60e08b0ff11b634c0413291686982f0b02e6809b9e96af2848350cbbf79. Companion
runs on the same head and merge (not cited in the register): statement run 38277458488, aggregate 38277459272.

usage: python3 -B -S update_register14.py   (R14_OUT=<file> writes the result there instead of the worktree register)
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

SCRATCH = Path('<SCRATCH>')
REPO = SCRATCH / 'repos/math_r5'
STMT_REL = 'specifications/lean_statements_20261010'
PROOF_REL = 'specifications/lean_statement_proofs_20261011'
STMT, PROOF = REPO / STMT_REL, REPO / PROOF_REL
REG_PATH = STMT / 'REGISTER.json'
OUT_DIR = SCRATCH / 'lvs/register_update14'
OUT_REG = Path(os.environ['R14_OUT']) if os.environ.get('R14_OUT') else REG_PATH
STMT_PACKAGE = STMT_REL
PROOF_PACKAGE = 'd6g8k5htny-coder/Math- specifications/lean_statement_proofs_20261011'
EXPECTED_HEAD = '9e305ea5141a1d1503560efd0e3deaf25441749e'
EXPECTED_BLOB = '84d0dd07f4c0a11bc9f0c452b1e494bbabb6ceae'      # 9e305ea5:specifications/lean_statements_20261010/REGISTER.json
EXPECTED_SHA256 = '67f7fb1e2d5c29a3e3034df74820a1585a12ffe7110e9182b90d32ac8b8937d4'   # the receipt's register_sha256
AXIOMS = '{propext, Classical.choice, Quot.sound}'

# ---- rounds 1-5 (as bound in the committed register; never re-derived here)
BASE = 'b2d70d77876a10c365f5db5d3b6cafcf180af4e2'
BASE2 = '79731b961fac555f33be3cd90e379b246a5a80e3'
RUN1, C1, M1 = '38116277597', '00d2ac6e', 'c4c66bb8cb26f0cd9bc03d52bb25f1b3925b272c'
RUN2, C2, M2 = '38121713840', '910809d6', '5b198699308e0585904c21d12bf540cf2475d2f0'
RUN3B, C3B, M3B = '38122386459', 'd9b8dd33', '866136cca66b46e6dbd3186b47c9340f7e93b4f6'
RUN4, C4, M4 = '38124262242', '79731b96', '318ff194eecc42c39a6b27c1d7bf03f30d6a34d9'
RUN6, C6, M6 = '38149888699', '0dcb0016', '4363b6928c62da242952631c6b0357d096c3b252'
RUNA, C5A, M5A = '38149524759', '88c21058', '3306233bbdeadb15d9b4b07b94cc52a80ed8caeb'
RUN7, C7, M7 = '38150363063', '12e072d6', '86a6e8c876f7904cadda2d006a56f8792dc5c1af'
RUN8, C8, M8 = '38180705952', '48bfcbae', '9833d6d9f9facc335e1a56e243c3144adefe6225'
SHA8 = '87309efad9b804c53c543b60fdb466de1f797971417363cc2e8fa1e10eac26bd'
BASE3 = '12e072d614dbe6e91b66d22a2ab55cf87860d693'     # head of PR #423 = base of PR #434
RUN9, C9, M9 = '38187538744', 'be014794', '79d3964cf70e2eadad5c9629eef8f4b47f218dd0'
RUN10, C10, M10 = '38195810463', 'c6ce0256', '1320c5b217e491d301ea140e5cc25b07ff5a21b7'
SHA10 = 'd8abd42be0facd4248647c3c7eb67a264bd19b9b47c6128733b5b5fc93b0a6ff'
BASE4 = 'be01479435f98f34b802258ee9ec3c3578f37169'     # head of PR #434 = base of PR #439
# ---- the first receipt of the round-6 theorems (FACTS)
RUN11, C11, M11 = '38277458954', '9e305ea5', '1376175ce5536a6ac74cb03706d18e65fdb695fa'
SHA11 = '1f32d60e08b0ff11b634c0413291686982f0b02e6809b9e96af2848350cbbf79'
BASE5 = 'a0323b94ae54a78f3d2878dc9c358dc40cdfe1dc'     # Math- main at the merge of PR #440 = the tip of PR #441's base branch main
MAIN_R5 = '4afcca6a79a8a0acef251ae73988a7914c8080e7'   # Math- main at the merge of PR #439 = parent of the PR head
HEAD_TREE, MERGE_TREE = '95a33c522f69a25db88d59b9fdccff75ba9b75a8', '0ccd93a213007fdbda2798017677f4572e28678f'
JOB11, ART11_ID, ART11_FILES, ART11_BYTES = '114906310832', '11759872846', 29, 47041
RUN_STMT11, RUN_AGG11 = '38277458488', '38277459272'   # companion runs on the same head and merge: not cited in the register
MERGE_ONLY_PATHS_PREFIX = 'companions/torus_filtration_20261011/'
MERGE_ONLY_WORKFLOW = '.github/workflows/torus-filtration.yml'
assert BASE4.startswith(C9) and BASE3.startswith(C7) and EXPECTED_HEAD.startswith(C11)
RUN_MERGE = ((RUN1, M1), (RUN2, M2), (RUN3B, M3B), (RUN4, M4), (RUN6, M6), (RUNA, M5A), (RUN7, M7), (RUN8, M8), (RUN9, M9),
             (RUN10, M10), (RUN11, M11), (RUN_STMT11, M11), (RUN_AGG11, M11))
RUN_HEAD = {RUN1: C1, RUN2: C2, RUN3B: C3B, RUN4: C4, RUN6: C6, RUNA: C5A, RUN7: C7, RUN8: C8, RUN9: C9, RUN10: C10, RUN11: C11,
            RUN_STMT11: C11, RUN_AGG11: C11}
for r, m in RUN_MERGE:
    assert r.isdigit() and len(r) == 11 and re.fullmatch(r'[0-9a-f]{40}', m), (r, m)
assert all(re.fullmatch(r'[0-9a-f]{64}', s) for s in (SHA8, SHA10, SHA11)) and len({r for r, _ in RUN_MERGE}) == len(RUN_MERGE)
WORKSPACE_LANDING_RUN = '38096827443'   # main's workspace-landing run, pre-existing in the register; not a pull_request run
NEW_MODULES = ('SpecificationProofs/Side24Images.lean',)
ROW, NODE = 'side24-coefficient', 'math.side24-coefficient'

# ----------------------------------------------------------------------------- the pending forms (update_register13, verbatim) ...
PENDING6_PHRASE = (f'were added in round 6 on top of Math- main 4afcca6a, the merge of PR #439, after PR head {C10} (the last head whose '
                   f'receipt this register binds: run {RUN10}, actual tested merge {M10} with base {BASE4}, receipt theorem_count 75): '
                   f'locally `lake env lean -DwarningAsError=true` exit 0 and `#print axioms` within {AXIOMS} on an unverified worktree, '
                   'not evidence about any commit; the next hosted lean-specification-proofs.yml run is the first receipt for them')
WF_PENDING6 = (f'; the theorems added in round 6 after PR head {C10} (module Side24Images; run {RUN10} (PR head {C10}; actual tested merge '
               f'{M10} with base {BASE4}), receipt theorem_count 75, does not name them) have no hosted receipt yet: the next hosted '
               'lean-specification-proofs.yml run is their first receipt')
PROVED_LABEL_SENTENCE = '. `proved` is the label of a committed proof package; `kernel-checked` appears only in a run receipt.'
ADDED6_TIER = f'added in round 6 on top of Math- main 4afcca6a, the merge of PR #439, after PR head {C10}'
ADDED6 = f'added in round 6 after PR head {C10}'
assert PENDING6_PHRASE.startswith('were ' + ADDED6_TIER + ' (') and WF_PENDING6.startswith(f'; the theorems {ADDED6} (')
RC6_OLD_PREFIX = (f'; the 3 theorems of this row {ADDED6} (Side24Images.imageBound, Side24Images.entrywiseCovarianceBound, '
                  'Side24Images.covarianceOrdering) await the next hosted lean-specification-proofs.yml run, their first receipt; the other '
                  'theorems of this row have a hosted receipt: ')
RC_TAIL19 = f'; each of those 19 theorems receipted on PR head {C10} by run {RUN10} (actual tested merge {M10} with base {BASE4}; receipt theorem_count 75)'
NHRY6_TRIAGE = ('proved in round 6 (2026-10-12; see entries, no hosted receipt yet; ', 'proved in round 6 (2026-10-12; see entries; ')
NHRY6_README = ('for ImageBound, CovarianceOrdering and EntrywiseCovarianceBound, proved in round 6 (no hosted receipt yet), the README\'s',
                'for ImageBound, CovarianceOrdering and EntrywiseCovarianceBound, proved in round 6, the README\'s')

# ----------------------------------------------------------------------------- ... and the receipted forms: templates pinned to the committed round-4/5 phrases
SAME_TREE = 'the merge has the same tree as its head'
TREE6 = (f'the merge differs from its head only outside both packets (main\'s changes from PRs #428 and #440: 13 files under '
         f'{MERGE_ONLY_PATHS_PREFIX} and {MERGE_ONLY_WORKFLOW}; {STMT_REL} and {PROOF_REL} are byte-identical between head and merge)')
BASE_DESC6 = 'Math- main at the merge of PR #440'


def tier_phrase(verb, added, run, head, merge, base, count, sha, pr, base_desc, tree, **_):
    they_were = 'it was' if verb == 'was' else 'they were'
    return (f'{verb} {added} and kernel-checked by the hosted run {run} (PR head {head}; actual tested merge {merge} with base {base}) '
            f'(job proof-evidence: fresh build, leanchecker PASS, `#print axioms` within {AXIOMS}, receipt formalization_status kernel-checked, '
            f'theorem_count {count}, worktree CLEAN; artifact lean-specification-proofs-{run}-1, sha256 {sha}); {they_were} first receipted by that run '
            f'(PR #{pr}, event pull_request); base {base} is {base_desc}, the base of PR #{pr}, and {tree}')


def wf_tier(added, mods, run, head, merge, base, count, sha, pr, base_desc, tree, **_):
    return (f'; the theorems {added} ({mods}) are receipted by run {run} (PR head {head}; actual tested merge {merge} '
            f'with base {base}) (artifact lean-specification-proofs-{run}-1, sha256 {sha}; receipt formalization_status kernel-checked, leanchecker PASS, '
            f'theorem_count {count}, worktree CLEAN) and were first receipted by that run; base {base} is {base_desc}, the base of PR #{pr}; '
            f'{tree}')


def corrob(run, head, merge, base, count, sha, caveat='', **_):        # update_register11 verbatim when caveat == ''
    return (f'All {count} theorems of PR head {head}, including every theorem of the tiers above, are receipted on that head by run {run} (actual tested '
            f'merge {merge} with base {base}{caveat}; receipt theorem_count {count}; artifact lean-specification-proofs-{run}-1, sha256 {sha})')


def wf_corrob(run, head, merge, base, count, sha, caveat='', **_):     # update_register11 verbatim when caveat == ''
    return (f'; all {count} theorems of PR head {head} receipted on that head by run {run} (actual tested merge {merge} with base {base}{caveat}; receipt '
            f'theorem_count {count}; artifact lean-specification-proofs-{run}-1, sha256 {sha})')


def rc_item(n, names, added_rc, run, head, merge, base, count, caveat='', **_):   # update_register11 verbatim when caveat == ''
    return (f'the {n} {added_rc} ({names}) first by run {run} (PR head {head}; actual tested merge {merge} with base {base}{caveat}) '
            f'(receipt theorem_count {count}); ')


def rc_tail(run, head, merge, base, count, **_):            # update_register11 verbatim
    return f'; every theorem of this row receipted on PR head {head} by run {run} (actual tested merge {merge} with base {base}; receipt theorem_count {count})'


def rc_tail_each(n, run, head, merge, base, count, caveat='', **_):   # round 6: 'each of those n theorems', not 'every theorem of this row'
    return (f'; each of those {n} theorems receipted on PR head {head} by run {run} (actual tested merge {merge} with base {base}{caveat}; '
            f'receipt theorem_count {count})')


R4 = dict(added=f'added after PR head {C7}', added_rc=f'added after PR head {C7}', mods='modules Side24Analysis, Side24Interfaces, RNCluster',
          run=RUN8, head=C8, merge=M8, base=BASE3, count=73, sha=SHA8, pr=434, base_desc='the head of PR #423', tree=SAME_TREE)
R5 = dict(added=f'added after PR head {C9}', added_rc=f'added after PR head {C9}', mods='modules Side24Covariance, Side24ConeMoment',
          run=RUN10, head=C10, merge=M10, base=BASE4, count=75, sha=SHA10, pr=439, base_desc='the head of PR #434', tree=SAME_TREE)
# the short caveat: the subject is 'the tested merge', never a bare 'the merge' (which would read as 'the merge of PR #440')
CAVEAT6 = ', the tested merge differing from the head only outside both packets'
OUTSIDE = 'only outside both packets'
assert OUTSIDE in CAVEAT6 and OUTSIDE in TREE6 and 'the merge differing' not in CAVEAT6 and 'same tree' not in CAVEAT6
N_ROW_PP = 22          # side24-coefficient's Math- proof-packet theorems: the 19 of rounds 1-5 (RC_TAIL19) and the 3 of round 6 (asserted below)
R6 = dict(added=ADDED6_TIER, added_rc=ADDED6, mods='module Side24Images', run=RUN11, head=C11, merge=M11, base=BASE5, count=78, sha=SHA11,
          pr=441, base_desc=BASE_DESC6, tree=TREE6, caveat=CAVEAT6)
TIER6_PHRASE = tier_phrase('were', **R6)
WF_TIER6 = wf_tier(**{**R6, 'added': ADDED6})        # the WF clause keeps WF_PENDING6's own 'added in round 6 after PR head c6ce0256'
CORROB11_PHRASE = corrob(**R6)
WF_CORROB11 = wf_corrob(**R6)
RC_ITEM6_SHAPE = rc_item(3, 'x', **R6)                     # shape check only; the instance with the theorem names is built in B
RC_TAIL_EACH6 = rc_tail_each(N_ROW_PP, **R6)
RC_TAIL_NEW = RC_TAIL19 + RC_TAIL_EACH6
NO_EVERY_ROW6 = f'every theorem of this row receipted on PR head {C11}'      # round 5's broad tail: must never be written for round 6
assert 'every theorem of this row' not in RC_TAIL_EACH6 and NO_EVERY_ROW6 not in RC_TAIL_NEW
for p in (CORROB11_PHRASE, WF_CORROB11, RC_ITEM6_SHAPE, RC_TAIL_EACH6):       # the caveat sits right after the base, inside the first parenthesis
    assert p.count(f'with base {BASE5}{CAVEAT6}') == 1 and p.count(CAVEAT6) == 1 and TREE6 not in p, p[:120]
for p in (TIER6_PHRASE, WF_TIER6):                                       # the long tree clause, once, and no short caveat
    assert p.count(TREE6) == 1 and CAVEAT6 not in p, p[:120]
for p in (TIER6_PHRASE, WF_TIER6, CORROB11_PHRASE, WF_CORROB11, RC_ITEM6_SHAPE, RC_TAIL_EACH6):
    assert 'same tree' not in p and RUN11 in p and C11 in p and M11 in p and BASE5 in p, p[:120]
    assert RUN_STMT11 not in p and RUN_AGG11 not in p and RUN9 not in p, p[:120]
    assert 'the merge differing' not in p and (CAVEAT6 in p or TREE6 in p), p[:120]      # every round-6 phrase carries a tree statement
assert TREE6 in TIER6_PHRASE and TREE6 in WF_TIER6 and SHA11 in TIER6_PHRASE and SHA11 in WF_TIER6
# with caveat '' the four caveat-bearing templates are round 5's verbatim (pinned to the committed text below)
for f in (corrob, wf_corrob, rc_item):
    _args = (3, 'x') if f is rc_item else ()
    assert f(*_args, **R5) == f(*_args, **{**R5, 'caveat': ''}) and CAVEAT6 not in f(*_args, **R5)
assert rc_tail_each(19, **R5) == RC_TAIL19          # round 6's committed narrowed tail is rc_tail_each with the round-5 facts


# ----------------------------------------------------------------------------- load: HEAD, committed == worktree == blob; the git facts of the run
def git(*args):
    return subprocess.run(['git', '-C', str(REPO), *args], check=True, capture_output=True, text=True, encoding='utf-8').stdout


head = git('rev-parse', 'HEAD').strip()
if head != EXPECTED_HEAD:
    sys.exit(f'REFUSED: worktree HEAD is {head}, expected {EXPECTED_HEAD}')
committed = git('show', f'{EXPECTED_HEAD}:{STMT_REL}/REGISTER.json')
raw = REG_PATH.read_bytes()
blob = hashlib.sha1(b'blob %d\0' % len(raw) + raw).hexdigest()
if raw.decode('utf-8') != committed or blob != EXPECTED_BLOB:
    sys.exit(f'REFUSED: the worktree REGISTER.json (blob {blob}) is not the 9e305ea5 blob {EXPECTED_BLOB} (already transformed?)')
assert hashlib.sha256(committed.encode('utf-8')).hexdigest() == EXPECTED_SHA256          # the receipt's register_sha256
# the tested merge (read-only git objects): parents base + head; its tree differs from the head's only outside both packets
assert git('rev-parse', f'{EXPECTED_HEAD}^{{tree}}').strip() == HEAD_TREE
assert git('rev-parse', f'{M11}^{{tree}}').strip() == MERGE_TREE and HEAD_TREE != MERGE_TREE
assert git('rev-list', '--parents', '-n', '1', M11).split() == [M11, BASE5, EXPECTED_HEAD]
assert git('rev-list', '--parents', '-n', '1', EXPECTED_HEAD).split() == [EXPECTED_HEAD, MAIN_R5]
_diff = git('diff', '--name-only', EXPECTED_HEAD, M11).split('\n')
_diff = [p for p in _diff if p]
assert len(_diff) == 14 and MERGE_ONLY_WORKFLOW in _diff and sum(p.startswith(MERGE_ONLY_PATHS_PREFIX) for p in _diff) == 13, _diff
assert not any(p.startswith(STMT_REL + '/') or p.startswith(PROOF_REL + '/') for p in _diff)
assert git('diff', '--name-only', EXPECTED_HEAD, M11, '--', STMT_REL, PROOF_REL).strip() == ''      # both packets byte-identical
subprocess.run(['git', '-C', str(REPO), 'merge-base', '--is-ancestor', MAIN_R5, BASE5], check=True)
_main_log = git('log', '--first-parent', '--format=%s', f'{MAIN_R5}..{BASE5}')
assert _main_log.split('\n')[0].startswith('Merge pull request #440 ') and 'Merge pull request #428 ' in _main_log, _main_log[:300]
old = json.loads(committed)
reg = json.loads(committed)
rows = {c['claim_id']: c for c in reg['claims']}
old_rows = {c['claim_id']: c for c in old['claims']}
assert len(rows) == 171 == len(old_rows)
assert json.dumps(old, indent=2, ensure_ascii=False) + '\n' == committed     # the serialisation is the committed one


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# the source-mode inventory of the committed tree (read-only parse; no Lean): 78 theorems, the 3 round-6 ones last
pcheck = load_module('proof_check14', PROOF / 'check.py')
inv = pcheck.source_check(PROOF, REPO, allow_dirty=True)
assert inv['formalization_status'] == 'proved' and inv['checked_commit'] == EXPECTED_HEAD
assert inv['theorem_count'] == 78 and len(inv['covered_props']) == 77, (inv['theorem_count'], len(inv['covered_props']))
assert list(inv['modules'][-1:]) == list(NEW_MODULES) and len(inv['modules']) == 14, inv['modules']
T6 = [t['name'] for t in inv['theorems'] if t['module'] in NEW_MODULES]
assert T6 == ['UniversalLaw.Proofs.Side24Images.imageBound', 'UniversalLaw.Proofs.Side24Images.entrywiseCovarianceBound',
              'UniversalLaw.Proofs.Side24Images.covarianceOrdering'], T6
assert all(t['claim_ids'] == [ROW] for t in inv['theorems'] if t['module'] in NEW_MODULES)
ALL_NAMES = {t['name'] for t in inv['theorems']}
assert {s['declaration'] for c in old['claims'] for s in c['statements'] if s['package'] == PROOF_PACKAGE} == ALL_NAMES
short = lambda ns: ', '.join(n.split('.', 2)[-1] for n in ns)
assert RC6_OLD_PREFIX == f'; the 3 theorems of this row {ADDED6} ({short(T6)}) await the next hosted lean-specification-proofs.yml run, their first receipt; the other theorems of this row have a hosted receipt: '
# 'each of those 22 theorems': the row's Math- proof-packet theorems (19 before round 6 + the 3 of round 6), all in run 38277458954's
# inventory; the row's other theorem entries come from the main package, which that run does not receipt
_row_pp = [s['declaration'] for s in old_rows[ROW]['statements'] if s['package'] == PROOF_PACKAGE]
assert len(_row_pp) == N_ROW_PP == 19 + len(T6) and set(_row_pp) <= ALL_NAMES and set(T6) <= set(_row_pp), len(_row_pp)
assert all(s['kind'] == 'theorem' for s in old_rows[ROW]['statements'] if s['package'] == PROOF_PACKAGE)
assert any(s['kind'] == 'theorem' and s['package'] != PROOF_PACKAGE for s in old_rows[ROW]['statements'])   # why not 'every theorem of this row'


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
assert before == {'statements': 357, 'proved': 160, 'kernel_evidence': 28, 'claims': 171, 'claim_fp': {'none': 138, 'specified': 33}}, before
TRIAGE_MARK = 'Round-4 triage (2026-10-11'
triaged = {c['claim_id'] for c in old['claims'] if TRIAGE_MARK in c['next_step']}
assert len(triaged) == 30 and all(old_rows[cid]['next_step'].count(TRIAGE_MARK) == 1 for cid in triaged)

# the round-4 / round-5 templates reproduce the committed round-4 / round-5 phrases (the shape is pinned, not retyped)
R4_ROW_NAMES = [s['declaration'] for s in old_rows[ROW]['statements'] if s['package'] == PROOF_PACKAGE and s['module'] in
                ('SpecificationProofs/Side24Analysis.lean', 'SpecificationProofs/Side24Interfaces.lean')]
T5 = [s['declaration'] for s in old_rows[ROW]['statements'] if s['package'] == PROOF_PACKAGE and s['module'] in
      ('SpecificationProofs/Side24Covariance.lean', 'SpecificationProofs/Side24ConeMoment.lean')]
assert len(R4_ROW_NAMES) == 12 and len(T5) == 2
pinned = {'tier4_were': (tier_phrase('were', **R4), 3), 'tier4_was': (tier_phrase('was', **R4), 2), 'wf_tier4': (wf_tier(**R4), 1),
          'tier5_were': (tier_phrase('were', **R5), 3), 'wf_tier5': (wf_tier(**R5), 1),
          'corrob8': (corrob(**R4), 9), 'wf_corrob8': (wf_corrob(**R4), 9), 'corrob10': (corrob(**R5), 10), 'wf_corrob10': (wf_corrob(**R5), 10),
          'rc_item4': (rc_item(12, short(R4_ROW_NAMES), **R4), 1),
          'rc_item5_after_round6': ('the other theorems of this row have a hosted receipt: ' + rc_item(2, short(T5), **R5), 1),
          'rc_item5_round5_lead_in_gone': ('every theorem of this row has a hosted receipt: ' + rc_item(2, short(T5), **R5), 0),
          'rc_tail5_narrowed_round6': (RC_TAIL19, 1)}
for label, (phrase, n) in pinned.items():
    assert count_all(old, phrase) == n, (label, count_all(old, phrase), n)


def same_tree_from_pinned(s):
    """How many 'the merge has the same tree as its head' of a string sit inside a pinned round-4/5 tier or WF phrase."""
    return sum(s.count(tier_phrase(v, **R)) for R in (R4, R5) for v in ('was', 'were')) + sum(s.count(wf_tier(**R)) for R in (R4, R5))


# every committed 'the merge has the same tree as its head' is one of the pinned round-4/5 phrases (tier4 x5, wf_tier4, tier5 x3, wf_tier5)
assert count_all(old, SAME_TREE) == sum(same_tree_from_pinned(s) for s in strings(old)) == 5 + 1 + 3 + 1, count_all(old, SAME_TREE)
# round 6 narrowed round 5's rc_tail ('every theorem of this row' -> 'each of those 19 theorems'): RC_TAIL19
assert RC_TAIL19 == rc_tail(**R5).replace('every theorem of this row', 'each of those 19 theorems')

pending_before = {'PENDING6': count_all(old, PENDING6_PHRASE), 'WF_PENDING6': count_all(old, WF_PENDING6),
                  'rc': count_all(old, RC6_OLD_PREFIX), 'rc_tail': count_all(old, RC_TAIL19),
                  'nhry_triage': count_all(old, NHRY6_TRIAGE[0]), 'nhry_readme': count_all(old, NHRY6_README[0])}
if pending_before['PENDING6'] == 0:
    sys.exit('REFUSED: the register carries no round-6 pending clause (already bound?)')
assert pending_before == {'PENDING6': 3, 'WF_PENDING6': 1, 'rc': 1, 'rc_tail': 1, 'nhry_triage': 3, 'nhry_readme': 1}, pending_before
assert count_all(old, 'no hosted receipt yet') == 5 and count_all(old, 'await the next hosted') == 1
assert count_all(old, 'not evidence about any commit') == 3 and count_all(old, 'whose receipt this register binds') == 3
assert count_all(old, 'first receipt for them') == 3 and count_all(old, 'their first receipt') == 2
assert count_all(old, 'round 6') == 9 and count_all(old, '4afcca6a') == 3
for needle in (RUN11, M11, C11, SHA11, BASE5, RUN_STMT11, RUN_AGG11, 'PR #441', 'PR #440', 'PR #428'):
    assert needle not in committed, needle
RUN10_BEFORE = count_all(old, RUN10)

# ----------------------------------------------------------------------------- A. kernel_evidence rows
counts = {'PENDING6_note': 0, 'WF_PENDING6': 0, 'CORROB11': 0, 'WF_CORROB11': 0, 'PENDING6_text': 0, 'rc': 0, 'rc_tail': 0, 'nhry_triage': 0,
          'nhry_readme': 0}
pp_rows = []
expected_fields = {}         # (cid, field) -> the exact expected new text, checked in C
for cid, row in rows.items():
    mine = [s['declaration'] for s in row['statements'] if s['package'] == PROOF_PACKAGE]
    pp = [e for e in row['kernel_evidence'] if e['package'] == PROOF_PACKAGE]
    assert len(pp) == (1 if mine else 0), (cid, len(pp), len(mine))
    if not pp:
        continue
    pp_rows.append(cid)
    e = pp[0]
    note, wf = e['note'], e['receipt_or_workflow']
    assert note.startswith(f'{len(mine)} theorem'), (cid, note[:60])
    earlier = [d for d in mine if d not in T6]
    assert earlier, cid                                  # every proof-packet row has theorems of earlier tiers
    # (a) the round-6 pending sentence -> the receipted tier phrase (only where round-6 theorems sit)
    n = note.count(PENDING6_PHRASE)
    assert (n == 1) == any(d in T6 for d in mine) == (cid == ROW) and n <= 1, cid
    if n:
        assert note.count(f'. 3 ({short(T6)}) {PENDING6_PHRASE}' + PROVED_LABEL_SENTENCE) == 1, cid
        note = note.replace(PENDING6_PHRASE, TIER6_PHRASE)
        counts['PENDING6_note'] += 1
    if WF_PENDING6 in wf:
        assert cid == ROW and wf.count(WF_PENDING6) == 1 and wf.endswith(WF_PENDING6), cid
        wf = wf.replace(WF_PENDING6, WF_TIER6)
        counts['WF_PENDING6'] += 1
    # one corroboration clause per row with earlier tiers (all 10): all 78 theorems receipted on PR head 9e305ea5 by run 38277458954
    assert note.count(PROVED_LABEL_SENTENCE) == 1, cid
    note = note.replace(PROVED_LABEL_SENTENCE, '. ' + CORROB11_PHRASE + PROVED_LABEL_SENTENCE)
    wf = wf + WF_CORROB11
    counts['CORROB11'] += 1
    counts['WF_CORROB11'] += 1
    expected_fields[(cid, 'kernel_evidence.note')] = (e['note'].replace(PENDING6_PHRASE, TIER6_PHRASE)
                                                      .replace(PROVED_LABEL_SENTENCE, '. ' + CORROB11_PHRASE + PROVED_LABEL_SENTENCE))
    expected_fields[(cid, 'kernel_evidence.receipt_or_workflow')] = e['receipt_or_workflow'].replace(WF_PENDING6, WF_TIER6) + WF_CORROB11
    e['note'], e['receipt_or_workflow'] = note, wf
assert len(pp_rows) == 10 and counts['PENDING6_note'] == 1 and counts['WF_PENDING6'] == 1, counts
assert counts['CORROB11'] == 10 and counts['WF_CORROB11'] == 10, counts

# ----------------------------------------------------------------------------- B. side24-coefficient / math.side24-coefficient: not_established, next_step
RC6_NEW_PREFIX = '; every theorem of this row has a hosted receipt: ' + rc_item(3, short(T6), **R6)
assert RC6_NEW_PREFIX.count(f'with base {BASE5}{CAVEAT6}) (receipt theorem_count 78); ') == 1      # the caveat, inside the item's first parenthesis
for cid, keys in ((ROW, ('not_established', 'next_step')), (NODE, ('not_established', 'next_step'))):
    row = rows[cid]
    for key in keys:
        text = row[key]
        n = text.count(PENDING6_PHRASE)
        assert n == (1 if key == 'not_established' else 0), (cid, key, n)
        text = text.replace(PENDING6_PHRASE, TIER6_PHRASE)
        counts['PENDING6_text'] += n
        if cid == ROW and key == 'next_step':
            assert text.count(RC6_OLD_PREFIX) == 1 and text.count(RC_TAIL19) == 1
            text = text.replace(RC6_OLD_PREFIX, RC6_NEW_PREFIX)
            counts['rc'] += 1
            text = text.replace(RC_TAIL19, RC_TAIL_NEW)
            counts['rc_tail'] += 1
            assert text.count(NHRY6_TRIAGE[0]) == 3 and text.count(NHRY6_README[0]) == 1
            text = text.replace(NHRY6_TRIAGE[0], NHRY6_TRIAGE[1])
            counts['nhry_triage'] += 3
            text = text.replace(NHRY6_README[0], NHRY6_README[1])
            counts['nhry_readme'] += 1
        else:
            assert RC6_OLD_PREFIX not in text and RC_TAIL19 not in text and NHRY6_TRIAGE[0] not in text and NHRY6_README[0] not in text, (cid, key)
        row[key] = text
assert counts['PENDING6_text'] == 2 and counts['rc'] == 1 and counts['rc_tail'] == 1 and counts['nhry_triage'] == 3 and counts['nhry_readme'] == 1, counts
expected_fields[(ROW, 'not_established')] = old_rows[ROW]['not_established'].replace(PENDING6_PHRASE, TIER6_PHRASE)
expected_fields[(NODE, 'not_established')] = old_rows[NODE]['not_established'].replace(PENDING6_PHRASE, TIER6_PHRASE)
expected_fields[(ROW, 'next_step')] = (old_rows[ROW]['next_step'].replace(RC6_OLD_PREFIX, RC6_NEW_PREFIX).replace(RC_TAIL19, RC_TAIL_NEW)
                                       .replace(NHRY6_TRIAGE[0], NHRY6_TRIAGE[1]).replace(NHRY6_README[0], NHRY6_README[1]))
assert rows[NODE]['next_step'] == old_rows[NODE]['next_step']      # 'As for row side24-coefficient. ' + triage: untouched
# the receipt clause follows round 5's bound form: the lead-in is round 5's committed one, then the tiers newest first
assert rows[ROW]['next_step'].count('; every theorem of this row has a hosted receipt: the 3 added in round 6 after PR head c6ce0256 (') == 1
assert rows[ROW]['next_step'].count(RC6_NEW_PREFIX + 'the 2 added after PR head be014794 (') == 1

# ----------------------------------------------------------------------------- C. nothing else changed
for k in reg:
    if k != 'claims':
        assert reg[k] == old[k], k
assert [c['claim_id'] for c in reg['claims']] == [c['claim_id'] for c in old['claims']]
changed_fields = {}
for cid, row in rows.items():
    o = old_rows[cid]
    assert list(row) == list(o), cid
    for k in row:
        if k == 'kernel_evidence':
            assert len(row[k]) == len(o[k]), cid
            for e_new, e_old in zip(row[k], o[k]):
                assert list(e_new) == list(e_old) and e_new['package'] == e_old['package'], cid
                if e_new['package'] != PROOF_PACKAGE:
                    assert e_new == e_old, (cid, e_new['package'])
                else:
                    for ek in e_new:
                        if ek not in ('note', 'receipt_or_workflow'):
                            assert e_new[ek] == e_old[ek], (cid, ek)
                        else:
                            assert e_new[ek] == expected_fields[(cid, f'kernel_evidence.{ek}')], (cid, ek)
                            if e_new[ek] != e_old[ek]:
                                changed_fields.setdefault(cid, []).append(f'kernel_evidence.{ek}')
        elif (cid, k) in expected_fields:
            assert row[k] == expected_fields[(cid, k)], (cid, k)
            if row[k] != o[k]:
                changed_fields.setdefault(cid, []).append(k)
        else:
            assert json.dumps(row[k], ensure_ascii=False, sort_keys=True) == json.dumps(o[k], ensure_ascii=False, sort_keys=True), (cid, k)
assert NODE not in pp_rows and set(changed_fields) == set(pp_rows) | {NODE}, sorted(set(changed_fields) ^ (set(pp_rows) | {NODE}))
assert all(sorted(v) == ['kernel_evidence.note', 'kernel_evidence.receipt_or_workflow'] for c, v in changed_fields.items() if c not in (ROW, NODE))
assert sorted(changed_fields[ROW]) == ['kernel_evidence.note', 'kernel_evidence.receipt_or_workflow', 'next_step', 'not_established']
assert changed_fields[NODE] == ['not_established']
for c in reg['claims']:
    assert c['formal_progress'] == old_rows[c['claim_id']]['formal_progress'] and c['formal_progress'] in ('none', 'specified')
    assert json.dumps(c['statements'], ensure_ascii=False) == json.dumps(old_rows[c['claim_id']]['statements'], ensure_ascii=False)   # byte-identical
    for s in c['statements']:
        if s['package'] == STMT_PACKAGE:
            assert s['formal_progress'] == 'specified' and s['kind'] in ('def_prop', 'structure')
        else:
            assert (s['kind'] == 'theorem') == (s['formal_progress'] == 'proved')
        assert s['formal_progress'] != 'kernel-checked'
assert {c['claim_id'] for c in reg['claims'] if TRIAGE_MARK in c['next_step']} == triaged
assert sum(c['next_step'].count(TRIAGE_MARK) for c in reg['claims']) == 30
for cid in triaged - {ROW}:          # every triage sentence other than side24-coefficient's is byte-identical
    assert rows[cid]['next_step'].split(TRIAGE_MARK)[1] == old_rows[cid]['next_step'].split(TRIAGE_MARK)[1], cid
# in side24-coefficient's triage sentence only the four 'no hosted receipt yet' parentheticals moved: every verdict, effort and finding kept
_t_old, _t_new = old_rows[ROW]['next_step'].split(TRIAGE_MARK)[1], rows[ROW]['next_step'].split(TRIAGE_MARK)[1]
assert _t_old.replace(NHRY6_TRIAGE[0], NHRY6_TRIAGE[1]).replace(NHRY6_README[0], NHRY6_README[1]) == _t_new
for p in ('ImageBound', 'CovarianceOrdering', 'EntrywiseCovarianceBound'):    # each round-6 item now reads 'proved in round 6 (2026-10-12; see entries; ...'
    item = _t_new.split(p + ' BLOCKED_MISSING_MATHLIB (weeks;', 1)
    assert len(item) == 2 and 'proved in round 6 (2026-10-12; see entries; ' in item[1].split(' BLOCKED_')[0], p
for verdict in ('PROVABLE_NOW', 'PROVABLE_WITH_WORK', 'BLOCKED_MISSING_MATHLIB', 'BLOCKED_INTERFACE', 'PREDICATE_VOCABULARY'):
    assert count_all(reg, verdict) == count_all(old, verdict), verdict

# ----------------------------------------------------------------------------- D. after: no round-6 pending phrase; C422-E1 on every run id; counts
HISTORICAL_FIRST_RECEIPT = (f'run {RUN1} (PR head {C1}; actual tested merge {M1} with base {BASE}) produced the first receipt (`kernel-checked` '
                            'appears only there); later commits need their own run')   # p15-price-boundary, round 1: dated to PR head 00d2ac6e
assert old_rows['p15-price-boundary']['next_step'].count(HISTORICAL_FIRST_RECEIPT) == 1
survivors = []
for s in strings(reg):
    assert 'no hosted receipt yet' not in s and 'await the next hosted' not in s and 'Hosted receipt: none yet' not in s, s[:160]
    assert 'not evidence about any commit' not in s and 'whose receipt this register binds' not in s and 'first receipt for them' not in s, s[:160]
    assert 'their first receipt' not in s and 'no hosted receipt' not in s and 'the last head with a hosted receipt' not in s, s[:160]
    for m in re.finditer(r'first receipt(?!ed)', s):
        assert HISTORICAL_FIRST_RECEIPT in s, s[:160]       # the one survivor: round 1's sentence, dated to PR head 00d2ac6e with its merge
        survivors.append(s[max(0, m.start() - 120):m.end() + 60])
    for run, merge in RUN_MERGE:
        if run in s:
            assert merge in s and RUN_HEAD[run] in s, (run, s[:160])
    for m in re.finditer(r'\b38\d{9}\b', s):
        assert m.group(0) in dict(RUN_MERGE) or m.group(0) == WORKSPACE_LANDING_RUN, (m.group(0), s[:120])
    if RUN11 in s:      # the new run is always named with its PR head, its actual tested merge and that merge's base, in every leaf naming it
        assert M11 in s and C11 in s and BASE5 in s, s[:160]
        # ... and never without the short caveat or the long tree clause: per leaf, and per occurrence (every occurrence sits inside a
        # planned round-6 phrase, each of which carries CAVEAT6 or TREE6, asserted at definition)
        assert CAVEAT6 in s or TREE6 in s, s[:160]
        _rest = s
        for _p in sorted((TIER6_PHRASE, WF_TIER6, CORROB11_PHRASE, WF_CORROB11, RC6_NEW_PREFIX, RC_TAIL_EACH6), key=len, reverse=True):
            _rest = _rest.replace(_p, '')
        assert RUN11 not in _rest, _rest[max(0, _rest.find(RUN11) - 160):_rest.find(RUN11) + 80]
    assert 'the merge differing' not in s and NO_EVERY_ROW6 not in s, s[:160]
    # the round-6 run never sits next to a same-tree claim: every 'same tree' sentence of a leaf naming run 38277458954 is a round-4/5 phrase
    assert s.count(SAME_TREE) == same_tree_from_pinned(s), s[:160]
assert len(survivors) == 1, survivors
assert count_all(reg, SAME_TREE) == count_all(old, SAME_TREE) == 10                       # no new same-tree claim
assert count_all(reg, TREE6) == 4 and count_all(old, TREE6) == 0                          # 3 tier phrases + 1 WF clause
# the short caveat: 10 note corroborations + 10 receipt_or_workflow corroborations + the next_step receipt item + its tail
assert count_all(reg, CAVEAT6) == 10 + 10 + 1 + 1 == 22 and count_all(old, CAVEAT6) == 0, count_all(reg, CAVEAT6)
assert count_all(reg, OUTSIDE) == count_all(reg, TREE6) + count_all(reg, CAVEAT6) == 26 and count_all(old, OUTSIDE) == 0, count_all(reg, OUTSIDE)
assert count_all(reg, 'the merge differing') == 0 and count_all(reg, NO_EVERY_ROW6) == 0
assert count_all(reg, RC_TAIL_EACH6) == 1 and count_all(reg, f'each of those {N_ROW_PP} theorems') == 1
assert count_all(reg, 'none yet') == count_all(old, 'none yet') == 1   # 'proof lane: none yet' (math.rn-region...), not a receipt statement
assert RUN_STMT11 not in json.dumps(reg) and RUN_AGG11 not in json.dumps(reg)
assert count_all(reg, RUN10) == RUN10_BEFORE - 3 * PENDING6_PHRASE.count(RUN10) - WF_PENDING6.count(RUN10), (count_all(reg, RUN10), RUN10_BEFORE)
assert count_all(reg, RUN10) > 0          # round 5's receipt stays bound
expected_run11 = (3 * TIER6_PHRASE.count(RUN11) + counts['WF_PENDING6'] * WF_TIER6.count(RUN11) + counts['CORROB11'] * CORROB11_PHRASE.count(RUN11)
                  + counts['WF_CORROB11'] * WF_CORROB11.count(RUN11) + counts['rc'] * rc_item(3, short(T6), **R6).count(RUN11)
                  + counts['rc_tail'] * RC_TAIL_EACH6.count(RUN11))
assert count_all(reg, RUN11) == expected_run11, (count_all(reg, RUN11), expected_run11)   # every mention of run 38277458954 is a planned phrase
assert count_all(reg, TIER6_PHRASE) == 3 and count_all(reg, WF_TIER6) == 1 and count_all(reg, CORROB11_PHRASE) == 10 and count_all(reg, WF_CORROB11) == 10
assert count_all(reg, RC6_NEW_PREFIX) == 1 and count_all(reg, RC_TAIL_NEW) == 1
for label, (phrase, n) in pinned.items():       # every earlier receipted phrase survives unchanged
    if label == 'rc_item5_after_round6':
        assert count_all(reg, phrase) == 0 and count_all(reg, rc_item(3, short(T6), **R6) + rc_item(2, short(T5), **R5)) == 1
    elif label == 'rc_tail5_narrowed_round6':
        assert count_all(reg, phrase) == 1
    else:
        assert count_all(reg, phrase) == n, (label, count_all(reg, phrase), n)
# `kernel-checked` stays a receipt word: it is added only inside the receipted phrases (2 per tier phrase, 1 per WF tier clause)
assert count_all(reg, 'kernel-checked') - count_all(old, 'kernel-checked') == 3 * TIER6_PHRASE.count('kernel-checked') + WF_TIER6.count('kernel-checked') == 7
assert count_all(reg, 'round 6') == count_all(old, 'round 6') == 9 and count_all(reg, '4afcca6a') == count_all(old, '4afcca6a') == 3
after = census(reg)
assert after == before, (before, after)
kc = json.dumps(reg, ensure_ascii=False)
assert '"formal_progress": "kernel-checked"' not in kc and '"formal_progress": "proved"' in kc

out = json.dumps(reg, indent=2, ensure_ascii=False) + '\n'
assert json.loads(out) == reg and out != committed
OUT_REG.write_text(out, encoding='utf-8')
new_blob = hashlib.sha1(b'blob %d\0' % len(out.encode()) + out.encode()).hexdigest()
new_sha256 = hashlib.sha256(out.encode()).hexdigest()

# ----------------------------------------------------------------------------- log of every changed string leaf
old_leaves, new_leaves = dict(leaves(old)), dict(leaves(reg))
assert set(old_leaves) == set(new_leaves)
changed = [(p, old_leaves[p], v) for p, v in new_leaves.items() if old_leaves[p] != v]
PLANNED = {f'/claims[{i}]/kernel_evidence[{j}]/{ek}' for i, c in enumerate(reg['claims']) if c['claim_id'] in pp_rows
           for j, e in enumerate(c['kernel_evidence']) if e['package'] == PROOF_PACKAGE for ek in ('note', 'receipt_or_workflow')}
_i = {c['claim_id']: i for i, c in enumerate(reg['claims'])}
PLANNED |= {f'/claims[{_i[ROW]}]/not_established', f'/claims[{_i[ROW]}]/next_step', f'/claims[{_i[NODE]}]/not_established'}
assert {p for p, _, _ in changed} == PLANNED and len(PLANNED) == 23, sorted({p for p, _, _ in changed} ^ PLANNED)
log = [f'update_register14: {len(changed)} changed string leaves of {STMT_REL}/REGISTER.json '
       f'(9e305ea5 blob {EXPECTED_BLOB} -> blob {new_blob}, sha256 {new_sha256}); census {before} -> {after} (unchanged)', '']
rowname = {f'/claims[{i}]': c['claim_id'] for i, c in enumerate(reg['claims'])}
for p, a, b in changed:
    head_path = re.match(r'/claims\[\d+\]', p).group(0)
    log.append(f'== {p}  (row {rowname[head_path]})')
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
(OUT_DIR / 'update_register14.log').write_text('\n'.join(log) + '\n', encoding='utf-8')
(OUT_DIR / 'update_register14_leaves.json').write_text(json.dumps([{'path': p, 'row': rowname[re.match(r'/claims\[\d+\]', p).group(0)], 'old': a, 'new': b}
                                                                   for p, a, b in changed], indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
occurrences_run11 = {cid: {k: v.count(RUN11) for k, v in (('not_established', row['not_established']), ('next_step', row['next_step']),
                                                           *((f'kernel_evidence.{ek}', e[ek]) for e in row['kernel_evidence'] if e['package'] == PROOF_PACKAGE
                                                             for ek in ('note', 'receipt_or_workflow'))) if v.count(RUN11)}
                     for cid, row in rows.items() if RUN11 in json.dumps(row, ensure_ascii=False)}
summary = {'scientific_effect': 'NONE', 'committed_head': head, 'input_blob': EXPECTED_BLOB, 'input_sha256': EXPECTED_SHA256,
           'output_blob': new_blob, 'output_sha256': new_sha256, 'output_bytes': len(out.encode()), 'output_path': str(OUT_REG),
           'before': before, 'after': after, 'replacements': counts, 'pending_forms_before': pending_before,
           'round4_round5_templates_pinned_to_committed_text': {k: v[1] for k, v in pinned.items()},
           'changed_fields': changed_fields, 'changed_string_leaves': [p for p, _, _ in changed],
           'run11_mentions_by_field': occurrences_run11,
           'receipt': {'run': RUN11, 'attempt': 1, 'job': JOB11, 'pr': 441, 'pr_head': EXPECTED_HEAD, 'checked_commit_merge': M11, 'base': BASE5,
                       'base_description': BASE_DESC6, 'head_tree': HEAD_TREE, 'merge_tree': MERGE_TREE,
                       'merge_differs_from_head_only_in': {'companions/torus_filtration_20261011/': 13, MERGE_ONLY_WORKFLOW: 1},
                       'both_packets_identical_between_head_and_merge': True,
                       'theorem_count': 78, 'covered_props': 77, 'leanchecker': 'PASS', 'worktree': 'CLEAN',
                       'register_sha256': EXPECTED_SHA256,
                       'artifact': f'lean-specification-proofs-{RUN11}-1', 'artifact_id': ART11_ID, 'artifact_files': ART11_FILES,
                       'artifact_bytes': ART11_BYTES, 'artifact_sha256_github_digest': SHA11, 'artifact_bytes_downloaded': False,
                       'companion_runs_not_cited_in_register': [RUN_STMT11, RUN_AGG11]},
           'run_merge_bindings_asserted': dict(RUN_MERGE), 'run_head_bindings_asserted': RUN_HEAD,
           'first_receipt_survivor_historical': survivors,
           'triage_rows_exactly_one_sentence': len(triaged),
           'claim_level_formal_progress_unchanged': all(rows[c]['formal_progress'] == old_rows[c]['formal_progress'] for c in rows),
           'statements_arrays_byte_identical': True,
           'template_parameters_round6_vs_round5': {'added_tier': [R5['added'], ADDED6_TIER], 'added_wf_and_rc': [R5['added'], ADDED6],
                                                    'base_desc': [R5['base_desc'], BASE_DESC6], 'tree': [SAME_TREE, TREE6],
                                                    'caveat': ['', CAVEAT6]},
           'counts_after': {'only outside both packets': count_all(reg, OUTSIDE), 'TREE6': count_all(reg, TREE6), 'CAVEAT6': count_all(reg, CAVEAT6),
                            'the merge differing': count_all(reg, 'the merge differing'), NO_EVERY_ROW6: count_all(reg, NO_EVERY_ROW6)},
           'phrases': {'TIER6': TIER6_PHRASE, 'WF_TIER6': WF_TIER6, 'CORROB11': CORROB11_PHRASE, 'WF_CORROB11': WF_CORROB11,
                       'RC_ITEM6': RC6_NEW_PREFIX, 'RC_TAIL_NEW': RC_TAIL_NEW, 'RC_TAIL_EACH6': RC_TAIL_EACH6, 'TREE6': TREE6, 'CAVEAT6': CAVEAT6,
                       'NHRY6_removed': [NHRY6_TRIAGE[0], NHRY6_README[0]]}}
(OUT_DIR / 'update_summary14.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps({k: v for k, v in summary.items() if k not in ('phrases', 'changed_string_leaves')}, indent=1, ensure_ascii=False))
print(f'{len(changed)} changed string leaves logged to {OUT_DIR / "update_register14.log"}')
