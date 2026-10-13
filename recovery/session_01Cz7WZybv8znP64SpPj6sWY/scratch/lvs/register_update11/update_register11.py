"""Round-11 register update: bind the two round-5 proof-packet theorems

    UniversalLaw.Proofs.Side24Covariance.referenceCovarianceLowerBound : UniversalLaw.Spec.Side24.ReferenceCovarianceLowerBound
    UniversalLaw.Proofs.Side24ConeMoment.coneMomentD2Identity (I)      : UniversalLaw.Spec.Side24.ConeMomentD2Identity I

(modules SpecificationProofs/{Side24Covariance,Side24ConeMoment}.lean, committed in c6ce0256) to their FIRST hosted
receipt, run 38195810463 on PR #439 head c6ce0256.

Derived from update_register9.py (the round-4 receipt binding: pending tier -> receipted tier by exact-string replacement,
one corroboration clause for the rows with earlier tiers, the 'await the next hosted ...' receipt clause -> the receipted
binding, and its invariants) and update_register10.py (whose PENDING5_PHRASE, WF_PENDING5 and 'await the next hosted ...'
receipt clause are the pending forms replaced here). The receipted phrases are NOT retyped: each is produced by a template
that is first asserted to reproduce, with the round-4 facts, the round-4 phrase exactly as committed in c6ce0256 (that is,
update_register9's output as amended by r4amend/apply_r4_fix2.py: 'the merge has the same tree as its head', 'each of
those ... corroborated'), and is then instantiated with the round-5 facts. Input: the committed register
c6ce0256:specifications/lean_statements_20261010/REGISTER.json, blob adbd0125..., asserted equal to the worktree copy.

Scientific effect: NONE. Edits, and nothing else (asserted as exact expected strings per field):
  (a) the 10 proof-packet kernel_evidence rows: in row side24-coefficient the round-5 pending sentence of the note
      (PENDING5_PHRASE) becomes the receipted tier phrase naming run 38195810463, PR head c6ce0256, actual tested merge
      1320c5b2... with base be014794..., receipt theorem_count 75, leanchecker PASS, artifact sha256 d8abd42b..., and its
      receipt_or_workflow clause WF_PENDING5 becomes the receipted WF clause; every one of the 10 rows has theorems of
      earlier tiers and gains one corroboration clause in the note and one in receipt_or_workflow (all 75 theorems
      receipted on PR head c6ce0256 by that run), in the round-4 shape;
  (b) not_established of side24-coefficient and of math.side24-coefficient: PENDING5_PHRASE -> the receipted tier phrase;
      next_step of side24-coefficient: the 'await the next hosted ... their first receipt' clause -> the receipted
      binding (rc_sub, round-4 shape), one 'every theorem of this row receipted on PR head c6ce0256 ...' tail clause (the
      round-4 RC_TAIL shape), and the three 'no hosted receipt yet' parentheticals that round 10 put in the triage item,
      the Rayleigh-step sentence and the README-reasons sentence are removed (the round-4 items read 'proved in round 4
      (see entries)' with no receipt word; the receipt is stated once, in the receipt clause of the same field);
  (c) nothing else: statements arrays, every formal_progress (claim level and entry level), every other field and every
      other kernel_evidence entry byte-identical; no triage verdict or effort altered.
No claim-level label changes; `proved` only on theorem entries; `kernel-checked` only as a receipt word; every mention of
a hosted run names the PR head AND the actual tested merge (C422-E1), asserted per string leaf. A receipt is evidence about
the registered Props for the formal objects only, never about an informal claim. Same-account author:
organizational-independence credit 0; alignment PENDING_INDEPENDENT_REVIEW.

Run 38187538744 (PR head be014794, the last receipted head before round 5) is named in the committed register only inside
the four pending clauses; the round-4 receipted shapes do not name the previous receipted head's run (update_register9's
tier4_phrase / WF_TIER4 dropped run 38150363063 the same way), so after this update the register no longer names it
(asserted). The statement-packet docs keep it, in the round-4 doc shape (render_docs11.py).

FACTS (coordinator-verified; also verified in review 5486104267; re-read with read-only `gh api` on 2026-10-12: run
head_sha, event, conclusion, PR #439 head/base, artifact name/id/size/digest, job, merge commit parents and tree):
workflow lean-specification-proofs.yml run 38195810463, attempt 1, job 114638065311 (proof-evidence, success), event
pull_request on PR #439 head c6ce025652f76be6e580fa24947380253450cd3b; checked_commit = synthetic merge
1320c5b217e491d301ea140e5cc25b07ff5a21b7 of that head into base be01479435f98f34b802258ee9ec3c3578f37169 (tip of the base
branch = head of PR #434), tree e723d6ab28c680276359196ddd6f5ade333214ed equal to the head's tree; receipt
formalization_status kernel-checked, leanchecker PASS, theorem_count 75, covered_props 74, negative controls refused for
their intended reasons, worktree CLEAN, passed true; artifact lean-specification-proofs-38195810463-1 (id 11723487528, 29
files, 45823 bytes), sha256 d8abd42be0facd4248647c3c7eb67a264bd19b9b47c6128733b5b5fc93b0a6ff (GitHub's digest; the
artifact bytes were not downloaded here: the API redirects to blob storage, which this session's gh refuses). Companion
runs on the same head and merge: statement run 38195810435, aggregate 38195810649 (not cited in the register). The
receipt's '`#print axioms` within {propext, Classical.choice, Quot.sound}' clause is what check.py --execute requires
before it writes a kernel-checked receipt (audit_axioms), as for every earlier tier.

usage: python3 -B -S update_register11.py   (R11_OUT=<file> writes the result there instead of the worktree register)
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
OUT_DIR = SCRATCH / 'lvs/register_update11'
OUT_REG = Path(os.environ['R11_OUT']) if os.environ.get('R11_OUT') else REG_PATH
STMT_PACKAGE = STMT_REL
PROOF_PACKAGE = 'd6g8k5htny-coder/Math- specifications/lean_statement_proofs_20261011'
EXPECTED_HEAD = 'c6ce025652f76be6e580fa24947380253450cd3b'
EXPECTED_BLOB = 'adbd01250717b09e2409db9564a16abb0acafe5a'      # c6ce0256:specifications/lean_statements_20261010/REGISTER.json
AXIOMS = '{propext, Classical.choice, Quot.sound}'

# ---- rounds 1-5 (update_register10 verbatim; never re-derived here)
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
# ---- round 6 of the register = the first receipt of the round-5 theorems (FACTS)
RUN10, C10, M10 = '38195810463', 'c6ce0256', '1320c5b217e491d301ea140e5cc25b07ff5a21b7'
SHA10 = 'd8abd42be0facd4248647c3c7eb67a264bd19b9b47c6128733b5b5fc93b0a6ff'
BASE4 = 'be01479435f98f34b802258ee9ec3c3578f37169'     # tip of the base branch of PR #439 = the head of PR #434
TREE10 = 'e723d6ab28c680276359196ddd6f5ade333214ed'
JOB10, ART10_ID, ART10_FILES, ART10_BYTES = '114638065311', '11723487528', 29, 45823
RUN_STMT10, RUN_AGG10 = '38195810435', '38195810649'   # companion runs on the same head: not cited in the register
assert BASE4.startswith(C9) and EXPECTED_HEAD.startswith(C10) and BASE3.startswith(C7)
RUN_MERGE = ((RUN1, M1), (RUN2, M2), (RUN3B, M3B), (RUN4, M4), (RUN6, M6), (RUNA, M5A), (RUN7, M7), (RUN8, M8), (RUN9, M9),
             (RUN10, M10), (RUN_STMT10, M10), (RUN_AGG10, M10))
RUN_HEAD = {RUN1: C1, RUN2: C2, RUN3B: C3B, RUN4: C4, RUN6: C6, RUNA: C5A, RUN7: C7, RUN8: C8, RUN9: C9, RUN10: C10,
            RUN_STMT10: C10, RUN_AGG10: C10}
for r, m in RUN_MERGE:
    assert r.isdigit() and len(r) == 11 and re.fullmatch(r'[0-9a-f]{40}', m), (r, m)
assert all(re.fullmatch(r'[0-9a-f]{64}', s) for s in (SHA8, SHA10)) and len({r for r, _ in RUN_MERGE}) == len(RUN_MERGE)
WORKSPACE_LANDING_RUN = '38096827443'   # main's workspace-landing run, pre-existing in the register; not a pull_request run
NEW_MODULES = ('SpecificationProofs/Side24Covariance.lean', 'SpecificationProofs/Side24ConeMoment.lean')
ROW, NODE = 'side24-coefficient', 'math.side24-coefficient'

# ----------------------------------------------------------------------------- the pending forms (update_register10, verbatim) ...
PENDING5_PHRASE = (f'were added after PR head {C9} (the last head with a hosted receipt: run {RUN9}, actual tested merge {M9} with base {BASE3}, '
                   f'receipt theorem_count 73): locally `lake env lean -DwarningAsError=true` exit 0 and `#print axioms` within {AXIOMS} on an '
                   'unverified worktree, not evidence about any commit; the next hosted lean-specification-proofs.yml run is the first receipt for them')
WF_PENDING5 = (f'; theorems added after PR head {C9} (run {RUN9}, actual tested merge {M9} with base {BASE3}, receipt theorem_count 73) have no hosted '
               'receipt yet: the next hosted lean-specification-proofs.yml run is their first receipt')
PROVED_LABEL_SENTENCE = '. `proved` is the label of a committed proof package; `kernel-checked` appears only in a run receipt.'
PENDING5_TAIL = PENDING5_PHRASE[len('were '):]
PENDING5_RE = re.compile(r'\b(was|were) ' + re.escape(PENDING5_TAIL))
RC_RE = re.compile(r'the (\d+) (theorem|theorems) of this row added after PR head ' + C9 + r' \(([^()]*)\) await the next hosted '
                   r'lean-specification-proofs\.yml run, their first receipt(; the other theorems of this row have a hosted receipt: )?')
RC_TAIL_OLD = f'; each of those 17 theorems receipted on PR head {C8} by run {RUN8} (actual tested merge {M8} with base {BASE3}; receipt theorem_count 73)'
# the three round-10 'no hosted receipt yet' parentheticals of side24-coefficient's next_step (render: removed, see docstring (b))
NHRY = (
    ('proved in round 5 (2026-10-12; see entries, no hosted receipt yet); ', 'proved in round 5 (2026-10-12; see entries); ', 'triage_item'),
    ('ConeMomentD2Identity now has a theorem for every inhabitant of ConeMomentInterface, with no hosted receipt yet, but no inhabitant is constructed',
     'ConeMomentD2Identity now has a theorem for every inhabitant of ConeMomentInterface, but no inhabitant is constructed', 'rayleigh_step'),
    ('for ReferenceCovarianceLowerBound and ConeMomentD2Identity, proved in round 5 (no hosted receipt yet), the README\'s PROVABLE_WITH_WORK',
     'for ReferenceCovarianceLowerBound and ConeMomentD2Identity, proved in round 5, the README\'s PROVABLE_WITH_WORK', 'readme_reasons'),
)


# ----------------------------------------------------------------------------- ... and the receipted forms: templates pinned to the committed round-4 phrases
def tier_phrase(verb, after_head, run, head, merge, base, count, sha, pr, base_pr, **_):
    they_were = 'it was' if verb == 'was' else 'they were'
    return (f'{verb} added after PR head {after_head} and kernel-checked by the hosted run {run} (PR head {head}; actual tested merge {merge} with base {base}) '
            f'(job proof-evidence: fresh build, leanchecker PASS, `#print axioms` within {AXIOMS}, receipt formalization_status kernel-checked, '
            f'theorem_count {count}, worktree CLEAN; artifact lean-specification-proofs-{run}-1, sha256 {sha}); {they_were} first receipted by that run '
            f'(PR #{pr}, event pull_request); base {base} is the head of PR #{base_pr}, the base of PR #{pr}, and the merge has the same tree as its head')


def wf_tier(after_head, modules, run, head, merge, base, count, sha, pr, base_pr, **_):
    return (f'; the theorems added after PR head {after_head} (modules {modules}) are receipted by run {run} (PR head {head}; actual tested merge {merge} '
            f'with base {base}) (artifact lean-specification-proofs-{run}-1, sha256 {sha}; receipt formalization_status kernel-checked, leanchecker PASS, '
            f'theorem_count {count}, worktree CLEAN) and were first receipted by that run; base {base} is the head of PR #{base_pr}, the base of PR #{pr}; '
            'the merge has the same tree as its head')


def corrob(run, head, merge, base, count, sha, **_):
    return (f'All {count} theorems of PR head {head}, including every theorem of the tiers above, are receipted on that head by run {run} (actual tested '
            f'merge {merge} with base {base}; receipt theorem_count {count}; artifact lean-specification-proofs-{run}-1, sha256 {sha})')


def wf_corrob(run, head, merge, base, count, sha, **_):
    return (f'; all {count} theorems of PR head {head} receipted on that head by run {run} (actual tested merge {merge} with base {base}; receipt '
            f'theorem_count {count}; artifact lean-specification-proofs-{run}-1, sha256 {sha})')


def rc_item(n, names, after_head, run, head, merge, base, count, **_):
    return (f'the {n} added after PR head {after_head} ({names}) first by run {run} (PR head {head}; actual tested merge {merge} with base {base}) '
            f'(receipt theorem_count {count}); ')


def rc_tail(run, head, merge, base, count, **_):
    return f'; every theorem of this row receipted on PR head {head} by run {run} (actual tested merge {merge} with base {base}; receipt theorem_count {count})'


R4 = dict(after_head=C7, modules='Side24Analysis, Side24Interfaces, RNCluster', run=RUN8, head=C8, merge=M8, base=BASE3, count=73, sha=SHA8,
          pr=434, base_pr=423)
R5 = dict(after_head=C9, modules='Side24Covariance, Side24ConeMoment', run=RUN10, head=C10, merge=M10, base=BASE4, count=75, sha=SHA10,
          pr=439, base_pr=434)
TIER5_PHRASE = tier_phrase('were', **R5)
WF_TIER5 = wf_tier(**R5)
CORROB10_PHRASE = corrob(**R5)
WF_CORROB10 = wf_corrob(**R5)
RC_TAIL_NEW = RC_TAIL_OLD + rc_tail(**R5)

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
    sys.exit(f'REFUSED: the worktree REGISTER.json (blob {blob}) is not the c6ce0256 blob {EXPECTED_BLOB} (already transformed?)')
assert git('rev-parse', f'{EXPECTED_HEAD}^{{tree}}').strip() == TREE10      # the receipt's merge has this tree (FACTS)
old = json.loads(committed)
reg = json.loads(committed)
rows = {c['claim_id']: c for c in reg['claims']}
old_rows = {c['claim_id']: c for c in old['claims']}
assert len(rows) == 171 == len(old_rows)
# the serialisation is the committed one: re-serialising the committed register reproduces its bytes
assert json.dumps(old, indent=2, ensure_ascii=False) + '\n' == committed


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# the source-mode inventory of the committed tree (read-only parse; no Lean): 75 theorems, the 2 round-5 ones last
pcheck = load_module('proof_check11', PROOF / 'check.py')
inv = pcheck.source_check(PROOF, REPO, allow_dirty=True)
assert inv['formalization_status'] == 'proved' and inv['checked_commit'] == EXPECTED_HEAD
assert inv['theorem_count'] == 75 and len(inv['covered_props']) == 74, (inv['theorem_count'], len(inv['covered_props']))
assert list(inv['modules'][-2:]) == list(NEW_MODULES) and len(inv['modules']) == 13, inv['modules']
T5 = [t['name'] for t in inv['theorems'] if t['module'] in NEW_MODULES]
assert T5 == ['UniversalLaw.Proofs.Side24Covariance.referenceCovarianceLowerBound', 'UniversalLaw.Proofs.Side24ConeMoment.coneMomentD2Identity'], T5
assert all(t['claim_ids'] == [ROW] for t in inv['theorems'] if t['module'] in NEW_MODULES)
ALL_NAMES = {t['name'] for t in inv['theorems']}
assert {s['declaration'] for c in old['claims'] for s in c['statements'] if s['package'] == PROOF_PACKAGE} == ALL_NAMES
short = lambda ns: ', '.join(n.split('.', 2)[-1] for n in ns)


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
TRIAGE_MARK = 'Round-4 triage (2026-10-11'
triaged = {c['claim_id'] for c in old['claims'] if TRIAGE_MARK in c['next_step']}
assert len(triaged) == 30 and all(old_rows[cid]['next_step'].count(TRIAGE_MARK) == 1 for cid in triaged)

# the round-4 templates reproduce the committed round-4 phrases (the shape is pinned, not retyped)
R4_ROW_NAMES = [s['declaration'] for s in old_rows[ROW]['statements'] if s['package'] == PROOF_PACKAGE and s['module'] in
                ('SpecificationProofs/Side24Analysis.lean', 'SpecificationProofs/Side24Interfaces.lean')]
assert len(R4_ROW_NAMES) == 12
pinned = {'tier4_were': (tier_phrase('were', **R4), 3), 'tier4_was': (tier_phrase('was', **R4), 2), 'wf_tier4': (wf_tier(**R4), 1),
          'corrob8': (corrob(**R4), 9), 'wf_corrob8': (wf_corrob(**R4), 9),
          'rc_item4': ('every theorem of this row has a hosted receipt: ' + rc_item(12, short(R4_ROW_NAMES), **R4), 0),
          'rc_item4_after_round10': ('the other theorems of this row have a hosted receipt: ' + rc_item(12, short(R4_ROW_NAMES), **R4), 1),
          'rc_tail4_narrowed_round10': (RC_TAIL_OLD, 1)}
for label, (phrase, n) in pinned.items():
    assert count_all(old, phrase) == n, (label, count_all(old, phrase), n)
# rc_tail(**R4) as committed was narrowed by round 10 ('every theorem of this row' -> 'each of those 17 theorems'): RC_TAIL_OLD
assert RC_TAIL_OLD == rc_tail(**R4).replace('every theorem of this row', 'each of those 17 theorems')

pending_before = {'PENDING5': sum(len(PENDING5_RE.findall(s)) for s in strings(old)), 'WF_PENDING5': count_all(old, WF_PENDING5),
                  'rc': sum(len(RC_RE.findall(s)) for s in strings(old)), 'rc_tail': count_all(old, RC_TAIL_OLD),
                  'nhry_parentheticals': sum(count_all(old, o) for o, _, _ in NHRY)}
if pending_before['PENDING5'] == 0:
    sys.exit('REFUSED: the register carries no round-5 pending clause (already bound?)')
assert pending_before == {'PENDING5': 3, 'WF_PENDING5': 1, 'rc': 1, 'rc_tail': 1, 'nhry_parentheticals': 3}, pending_before
assert count_all(old, PENDING5_PHRASE) == 3                      # all three are 'were' (two theorems)
assert count_all(old, 'no hosted receipt yet') == 4 and count_all(old, 'await the next hosted') == 1
assert count_all(old, 'not evidence about any commit') == 3 and count_all(old, 'the last head with a hosted receipt') == 3
assert count_all(old, 'first receipt for them') == 3 and count_all(old, 'their first receipt') == 2
assert count_all(old, RUN9) == 4 and count_all(old, M9) == 4
for needle in (RUN10, M10, C10, SHA10, RUN_STMT10, RUN_AGG10, 'PR #439'):
    assert needle not in committed, needle

# ----------------------------------------------------------------------------- A. kernel_evidence rows
counts = {'PENDING5_note': 0, 'WF_PENDING5': 0, 'CORROB10': 0, 'WF_CORROB10': 0, 'PENDING5_text': 0, 'rc': 0, 'rc_tail': 0, 'nhry': 0}
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
    earlier = [d for d in mine if d not in T5]
    assert earlier, cid                                  # every proof-packet row has theorems of earlier tiers
    # (a) the round-5 pending clauses -> the receipted tier phrase (only where round-5 theorems sit)
    note, n = PENDING5_RE.subn(lambda m: tier_phrase(m.group(1), **R5), note)
    assert all(m.group(1) == 'were' for m in PENDING5_RE.finditer(e['note']))
    counts['PENDING5_note'] += n
    assert (n > 0) == any(d in T5 for d in mine) == (cid == ROW), cid
    if WF_PENDING5 in wf:
        assert cid == ROW and wf.count(WF_PENDING5) == 1 and wf.endswith(WF_PENDING5), cid
        wf = wf.replace(WF_PENDING5, WF_TIER5)
        counts['WF_PENDING5'] += 1
    # one corroboration clause per row with earlier tiers (all 10): all 75 theorems receipted on PR head c6ce0256 by run 38195810463
    assert note.count(PROVED_LABEL_SENTENCE) == 1, cid
    note = note.replace(PROVED_LABEL_SENTENCE, '. ' + CORROB10_PHRASE + PROVED_LABEL_SENTENCE)
    wf = wf + WF_CORROB10
    counts['CORROB10'] += 1
    counts['WF_CORROB10'] += 1
    expected_fields[(cid, 'kernel_evidence.note')] = (e['note'].replace(PENDING5_PHRASE, TIER5_PHRASE)
                                                      .replace(PROVED_LABEL_SENTENCE, '. ' + CORROB10_PHRASE + PROVED_LABEL_SENTENCE))
    expected_fields[(cid, 'kernel_evidence.receipt_or_workflow')] = e['receipt_or_workflow'].replace(WF_PENDING5, WF_TIER5) + WF_CORROB10
    e['note'], e['receipt_or_workflow'] = note, wf
assert len(pp_rows) == 10 and counts['PENDING5_note'] == 1 and counts['WF_PENDING5'] == 1, counts
assert counts['CORROB10'] == 10 and counts['WF_CORROB10'] == 10, counts


# ----------------------------------------------------------------------------- B. side24-coefficient / math.side24-coefficient: not_established, next_step
def rc_sub(m):
    n, noun, names, lead_in = m.groups()
    assert lead_in and n == str(len(T5)) and noun == 'theorems' and names == short(T5), m.group(0)
    return 'every theorem of this row has a hosted receipt: ' + rc_item(n, names, **R5)


for cid, keys in ((ROW, ('not_established', 'next_step')), (NODE, ('not_established', 'next_step'))):
    row = rows[cid]
    for key in keys:
        text, n = PENDING5_RE.subn(lambda m: tier_phrase(m.group(1), **R5), row[key])
        counts['PENDING5_text'] += n
        text, n_rc = RC_RE.subn(rc_sub, text)
        counts['rc'] += n_rc
        if RC_TAIL_OLD in text:
            assert text.count(RC_TAIL_OLD) == 1 and n_rc == 1 and cid == ROW and key == 'next_step', (cid, key)
            text = text.replace(RC_TAIL_OLD, RC_TAIL_NEW)
            counts['rc_tail'] += 1
        if cid == ROW and key == 'next_step':
            for o, nw, label in NHRY:
                assert text.count(o) == 1, label
                text = text.replace(o, nw)
                counts['nhry'] += 1
        row[key] = text
assert counts['PENDING5_text'] == 2 and counts['rc'] == 1 and counts['rc_tail'] == 1 and counts['nhry'] == 3, counts
expected_fields[(ROW, 'not_established')] = old_rows[ROW]['not_established'].replace(PENDING5_PHRASE, TIER5_PHRASE)
expected_fields[(NODE, 'not_established')] = old_rows[NODE]['not_established'].replace(PENDING5_PHRASE, TIER5_PHRASE)
_ns = old_rows[ROW]['next_step']
_rc_old = (f'the {len(T5)} theorems of this row added after PR head {C9} ({short(T5)}) await the next hosted lean-specification-proofs.yml run, '
           'their first receipt; the other theorems of this row have a hosted receipt: ')
assert _ns.count(_rc_old) == 1
_ns = _ns.replace(_rc_old, 'every theorem of this row has a hosted receipt: ' + rc_item(len(T5), short(T5), **R5)).replace(RC_TAIL_OLD, RC_TAIL_NEW)
for o, nw, _ in NHRY:
    _ns = _ns.replace(o, nw)
expected_fields[(ROW, 'next_step')] = _ns
assert rows[NODE]['next_step'] == old_rows[NODE]['next_step']      # 'As for row side24-coefficient. ' + triage: untouched

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
    assert c['statements'] == old_rows[c['claim_id']]['statements']
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
# in side24-coefficient's triage sentence only the three parentheticals moved: every verdict and effort is kept
_t_old, _t_new = old_rows[ROW]['next_step'].split(TRIAGE_MARK)[1], rows[ROW]['next_step'].split(TRIAGE_MARK)[1]
for o, nw, _ in NHRY:
    _t_old = _t_old.replace(o, nw)
assert _t_old == _t_new
for verdict in ('PROVABLE_NOW', 'PROVABLE_WITH_WORK', 'BLOCKED_MISSING_MATHLIB', 'BLOCKED_INTERFACE', 'PREDICATE_VOCABULARY'):
    assert count_all(reg, verdict) == count_all(old, verdict), verdict

# ----------------------------------------------------------------------------- D. after: no pending phrase; C422-E1 on every run id; counts
HISTORICAL_FIRST_RECEIPT = (f'run {RUN1} (PR head {C1}; actual tested merge {M1} with base {BASE}) produced the first receipt (`kernel-checked` '
                            'appears only there); later commits need their own run')   # p15-price-boundary, round 1: dated to PR head 00d2ac6e
assert old_rows['p15-price-boundary']['next_step'].count(HISTORICAL_FIRST_RECEIPT) == 1
survivors = []
for s in strings(reg):
    assert 'no hosted receipt yet' not in s and 'await the next hosted' not in s and 'Hosted receipt: none yet' not in s, s[:160]
    assert 'not evidence about any commit' not in s and 'the last head with a hosted receipt' not in s and 'first receipt for them' not in s, s[:160]
    assert 'their first receipt' not in s and 'no hosted receipt' not in s, s[:160]
    for m in re.finditer(r'first receipt(?!ed)', s):
        assert HISTORICAL_FIRST_RECEIPT in s, s[:160]       # the one survivor: round 1's sentence, dated to PR head 00d2ac6e with its merge
        survivors.append(s[max(0, m.start() - 120):m.end() + 60])
    for run, merge in RUN_MERGE:
        if run in s:
            assert merge in s and RUN_HEAD[run] in s, (run, s[:160])
    for m in re.finditer(r'\b38\d{9}\b', s):
        assert m.group(0) in dict(RUN_MERGE) or m.group(0) == WORKSPACE_LANDING_RUN, (m.group(0), s[:120])
    if RUN10 in s:      # the new run is always named with the PR number of its event and its base, in every leaf that names it
        assert M10 in s and C10 in s and BASE4 in s, s[:160]
assert len(survivors) == 1, survivors
assert count_all(reg, 'none yet') == count_all(old, 'none yet') == 1   # 'proof lane: none yet' (math.rn-region...), not a receipt statement
assert count_all(reg, RUN9) == 0 and count_all(reg, M9) == 0          # see docstring: named only in the replaced pending clauses
assert RUN_STMT10 not in json.dumps(reg) and RUN_AGG10 not in json.dumps(reg)
expected_run10 = (3 * TIER5_PHRASE.count(RUN10) + counts['WF_PENDING5'] * WF_TIER5.count(RUN10) + counts['CORROB10'] * CORROB10_PHRASE.count(RUN10)
                  + counts['WF_CORROB10'] * WF_CORROB10.count(RUN10) + counts['rc'] * rc_item(2, short(T5), **R5).count(RUN10)
                  + counts['rc_tail'] * rc_tail(**R5).count(RUN10))
assert count_all(reg, RUN10) == expected_run10, (count_all(reg, RUN10), expected_run10)   # every mention of run 38195810463 is a planned phrase
assert count_all(reg, TIER5_PHRASE) == 3 and count_all(reg, WF_TIER5) == 1 and count_all(reg, CORROB10_PHRASE) == 10 and count_all(reg, WF_CORROB10) == 10
# `kernel-checked` stays a receipt word: it is added only inside the receipted phrases (2 per tier phrase, 1 per WF tier clause)
assert count_all(reg, 'kernel-checked') - count_all(old, 'kernel-checked') == 3 * TIER5_PHRASE.count('kernel-checked') + WF_TIER5.count('kernel-checked') == 7
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
log = [f'update_register11: {len(changed)} changed string leaves of {STMT_REL}/REGISTER.json '
       f'(c6ce0256 blob {EXPECTED_BLOB} -> blob {new_blob}, sha256 {new_sha256}); census {before} -> {after} (unchanged)', '']
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
(OUT_DIR / 'update_register11.log').write_text('\n'.join(log) + '\n', encoding='utf-8')
(OUT_DIR / 'update_register11_leaves.json').write_text(json.dumps([{'path': p, 'row': rowname[re.match(r'/claims\[\d+\]', p).group(0)], 'old': a, 'new': b}
                                                                   for p, a, b in changed], indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
occurrences_run10 = {cid: {k: v.count(RUN10) for k, v in (('not_established', row['not_established']), ('next_step', row['next_step']),
                                                           *((f'kernel_evidence.{ek}', e[ek]) for e in row['kernel_evidence'] if e['package'] == PROOF_PACKAGE
                                                             for ek in ('note', 'receipt_or_workflow'))) if v.count(RUN10)}
                     for cid, row in rows.items() if RUN10 in json.dumps(row, ensure_ascii=False)}
summary = {'scientific_effect': 'NONE', 'committed_head': head, 'input_blob': EXPECTED_BLOB, 'output_blob': new_blob, 'output_sha256': new_sha256,
           'output_bytes': len(out.encode()), 'output_path': str(OUT_REG), 'before': before, 'after': after,
           'replacements': counts, 'pending_forms_before': pending_before, 'round4_templates_pinned_to_committed_text': {k: v[1] for k, v in pinned.items()},
           'changed_fields': changed_fields, 'changed_string_leaves': [p for p, _, _ in changed],
           'run10_mentions_by_field': occurrences_run10,
           'receipt': {'run': RUN10, 'attempt': 1, 'job': JOB10, 'pr': 439, 'pr_head': EXPECTED_HEAD, 'checked_commit_merge': M10, 'base': BASE4,
                       'merge_tree': TREE10, 'theorem_count': 75, 'covered_props': 74, 'leanchecker': 'PASS', 'worktree': 'CLEAN',
                       'artifact': f'lean-specification-proofs-{RUN10}-1', 'artifact_id': ART10_ID, 'artifact_files': ART10_FILES,
                       'artifact_bytes': ART10_BYTES, 'artifact_sha256_github_digest': SHA10,
                       'companion_runs_not_cited_in_register': [RUN_STMT10, RUN_AGG10]},
           'run_no_longer_named_by_register': {'run': RUN9, 'pr_head': C9, 'actual_tested_merge': M9,
                                               'reason': 'named only inside the replaced pending clauses; the round-4 receipted shapes do not name it'},
           'run_merge_bindings_asserted': dict(RUN_MERGE), 'run_head_bindings_asserted': RUN_HEAD,
           'first_receipt_survivor_historical': survivors,
           'triage_rows_exactly_one_sentence': len(triaged),
           'claim_level_formal_progress_unchanged': all(rows[c]['formal_progress'] == old_rows[c]['formal_progress'] for c in rows),
           'phrases': {'TIER5': TIER5_PHRASE, 'WF_TIER5': WF_TIER5, 'CORROB10': CORROB10_PHRASE, 'WF_CORROB10': WF_CORROB10,
                       'RC_ITEM5': 'every theorem of this row has a hosted receipt: ' + rc_item(2, short(T5), **R5), 'RC_TAIL_NEW': RC_TAIL_NEW,
                       'NHRY_removed': [o for o, _, _ in NHRY]}}
(OUT_DIR / 'update_summary11.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps({k: v for k, v in summary.items() if k not in ('phrases', 'changed_string_leaves')}, indent=1, ensure_ascii=False))
print(f'{len(changed)} changed string leaves logged to {OUT_DIR / "update_register11.log"}')
