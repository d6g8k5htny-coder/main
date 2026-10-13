"""Round-9 register update: bind the 13 round-4 proof-packet theorems (modules SpecificationProofs/{Side24Analysis,
Side24Interfaces,RNCluster}.lean) to their FIRST hosted receipt, run 38180705952 on PR #434 head 48bfcbae.

Derived from update_register6.py (the round-3 analogue: pending tier -> receipted tier by exact-string replacement) and
update_register8.py (whose PENDING4_PHRASE, WF_PENDING4, receipt-free-row text and 'await the next hosted ...' receipt
clause are the pending forms replaced here; its tier phrases are reproduced verbatim as the shape of the new ones).

Scientific effect: NONE. Transforms ONLY the receipt bindings of the committed register (asserted equal to the worktree
copy): (a) in every proof-packet kernel_evidence row the round-4 pending clauses become receipted tier-4 phrases naming
run 38180705952, PR head 48bfcbae, actual tested merge 9833d6d9... with base 12e072d6..., receipt theorem_count 73,
leanchecker PASS, artifact sha256 87309efa..., in the shape of the tier-3 phrases, and the rows with earlier tiers gain
one corroboration clause (all 73 theorems receipted on PR head 48bfcbae by that run); (b) the 'await the next hosted ...'
and 'first receipt for them' clauses of next_step / not_established of side24-coefficient, math.side24-coefficient and
OA-C6-RARE-CLUSTER-20260929-v1 become the receipted binding; (c) nothing else: statements arrays, every formal_progress
label and every other field are asserted byte-identical, no triage sentence is re-appended. No claim-level label changes;
`proved` only on theorem entries; `kernel-checked` only as a receipt word; every mention of a hosted run names the PR head
AND the actual tested merge (C422-E1). A receipt is evidence about the registered Props for the formal objects only, never
about an informal claim. Same-account author: organizational-independence credit 0; alignment PENDING_INDEPENDENT_REVIEW.

FACTS (coordinator-verified, cited verbatim): workflow lean-specification-proofs.yml run 38180705952, attempt 1, job
114592458899 (proof-evidence, success), event pull_request on PR #434 head 48bfcbae7834c2d6f4d3123d85c69d0c5afedfa6; the
receipt's checked_commit is the synthetic merge 9833d6d9f9facc335e1a56e243c3144adefe6225 of that head into base
12e072d614dbe6e91b66d22a2ab55cf87860d693 (tip of the base branch, the head of PR #423), tree 1956e6298d8d34ef7b1fca786903d1e0e72cb220
equal to the head's tree; receipt formalization_status kernel-checked, leanchecker PASS, theorem_count 73, covered_props 72,
negative controls refused for their intended reasons, worktree CLEAN; artifact lean-specification-proofs-38180705952-1
(id 11717482416, 29 files, 43937 bytes), GitHub digest sha256 87309efad9b804c53c543b60fdb466de1f797971417363cc2e8fa1e10eac26bd
(the artifact bytes were not downloaded; the digest is GitHub's, equal to the upload step's log line).

usage: python3 -B -S update_register9.py        (run from anywhere; paths are absolute; refuses a register without the pending forms)
"""
import json
import re
import subprocess
import sys
from pathlib import Path

SCRATCH = Path('<SCRATCH>')
REPO = SCRATCH / 'repos/math'
STMT_REL = 'specifications/lean_statements_20261010'
REG_PATH = REPO / STMT_REL / 'REGISTER.json'
OUT_DIR = SCRATCH / 'lvs/register_update9'
PROOF_PACKAGE = 'd6g8k5htny-coder/Math- specifications/lean_statement_proofs_20261011'
STMT_PACKAGE = STMT_REL
EXPECTED_HEAD = '48bfcbae7834c2d6f4d3123d85c69d0c5afedfa6'
AXIOMS = '{propext, Classical.choice, Quot.sound}'

# ---- rounds 1-3 (update_register8 verbatim; never re-derived here)
BASE = 'b2d70d77876a10c365f5db5d3b6cafcf180af4e2'
BASE2 = '79731b961fac555f33be3cd90e379b246a5a80e3'     # head of PR #422, the base of PR #423 and of every round-3 run
RUN1, C1, M1 = '38116277597', '00d2ac6e', 'c4c66bb8cb26f0cd9bc03d52bb25f1b3925b272c'
RUN2, C2, M2 = '38121713840', '910809d6', '5b198699308e0585904c21d12bf540cf2475d2f0'
RUN3B, C3B, M3B = '38122386459', 'd9b8dd33', '866136cca66b46e6dbd3186b47c9340f7e93b4f6'
RUN4, C4, M4 = '38124262242', '79731b96', '318ff194eecc42c39a6b27c1d7bf03f30d6a34d9'
RUN6, C6, M6, SHA6 = '38149888699', '0dcb0016', '4363b6928c62da242952631c6b0357d096c3b252', 'cdeb5c73aa03124ed0db16fa14d25fb03a8fa842457fce1e0c6565dbc5ee7ca8'
RUNA, C5A, M5A, SHA5A = '38149524759', '88c21058', '3306233bbdeadb15d9b4b07b94cc52a80ed8caeb', 'ddefbdcc63223d05fd8ed34addbeebc199e271525d87dead01109cddc529e836'
RUN7, C7, M7, SHA7 = '38150363063', '12e072d6', '86a6e8c876f7904cadda2d006a56f8792dc5c1af', 'bf0e0515e2035103b07d894e2a73eeb5b542885f6da16b98c228877a5e0c1690'
# ---- round 4 (FACTS)
RUN8, C8, M8, SHA8 = '38180705952', '48bfcbae', '9833d6d9f9facc335e1a56e243c3144adefe6225', '87309efad9b804c53c543b60fdb466de1f797971417363cc2e8fa1e10eac26bd'
BASE3 = '12e072d614dbe6e91b66d22a2ab55cf87860d693'     # tip of the base branch of PR #434 = the head of PR #423
TREE8 = '1956e6298d8d34ef7b1fca786903d1e0e72cb220'
RUN_STMT, RUN_AGG = '38180705999', '38180706314'       # companion runs on the same head (statement packet; required aggregate): not cited in the register
assert BASE3.startswith(C7) and EXPECTED_HEAD.startswith(C8)
RUN_MERGE = ((RUN1, M1), (RUN2, M2), (RUN3B, M3B), (RUN4, M4), (RUN6, M6), (RUNA, M5A), (RUN7, M7),
             (RUN8, M8), (RUN_STMT, M8), (RUN_AGG, M8))
RUN_HEAD = {RUN1: C1, RUN2: C2, RUN3B: C3B, RUN4: C4, RUN6: C6, RUNA: C5A, RUN7: C7, RUN8: C8, RUN_STMT: C8, RUN_AGG: C8}
for r, m in RUN_MERGE:
    assert r.isdigit() and len(r) == 11 and re.fullmatch(r'[0-9a-f]{40}', m), (r, m)
assert all(re.fullmatch(r'[0-9a-f]{64}', s) for s in (SHA6, SHA5A, SHA7, SHA8))
assert len({r for r, _ in RUN_MERGE}) == len(RUN_MERGE)

# ----------------------------------------------------------------------------- the pending forms (update_register8, verbatim) ...
PENDING4_PHRASE = (f'were added after PR head {C7} (the last head with a hosted receipt: run {RUN7}, actual tested merge {M7} with base {BASE2}, '
                   f'receipt theorem_count 60): locally `lake env lean -DwarningAsError=true` exit 0 and `#print axioms` within {AXIOMS} on an '
                   'unverified worktree, not evidence about any commit; the next hosted lean-specification-proofs.yml run is the first receipt for them')
WF_PENDING4 = (f'; theorems added after PR head {C7} (run {RUN7}, actual tested merge {M7} with base {BASE2}, receipt theorem_count 60) have no hosted '
               'receipt yet: the next hosted lean-specification-proofs.yml run is their first receipt')
CORROB_PHRASE = (f'The 60 theorems of PR head {C7} (every theorem of the tiers above) are corroborated by run {RUN7} (actual tested merge {M7} with base '
                 f'{BASE2}; receipt theorem_count 60; artifact lean-specification-proofs-{RUN7}-1, sha256 {SHA7})')
WF_CORROB = (f'; corroborated on PR head {C7} by run {RUN7} (actual tested merge {M7} with base {BASE2}; receipt theorem_count 60; artifact '
             f'lean-specification-proofs-{RUN7}-1, sha256 {SHA7})')
PROVED_LABEL_SENTENCE = '. `proved` is the label of a committed proof package; `kernel-checked` appears only in a run receipt.'
# update_register8's singular() turned 'were added' into 'was added' for the one-theorem row, so the tail after the verb is matched
PENDING4_TAIL = PENDING4_PHRASE[len('were '):]
assert PENDING4_TAIL.startswith('added after PR head')
PENDING4_RE = re.compile(r'\b(was|were) ' + re.escape(PENDING4_TAIL))
# the receipt-free row (all its proof-packet theorems round-4): update_register8's text after singular()
FREE_WF_RE = re.compile(
    r'\AHosted receipt: none yet\. The proof-packet (theorem|theorems) of this row \(([^()]*)\) (was|were) added after PR head ' + C7
    + r', the last head with a hosted receipt \(run ' + RUN7 + r', actual tested merge ' + M7 + r' with base ' + BASE2
    + r', receipt theorem_count 60, which does not name (it|them)\); the next hosted lean-specification-proofs\.yml run is the first receipt\.\Z')
# update_register8's receipt_clause (next_step): the pending part, optionally followed by the 'other theorems' lead-in
RC_RE = re.compile(r'the (\d+) (theorem|theorems) of this row added after PR head ' + C7 + r' \(([^()]*)\) await the next hosted '
                   r'lean-specification-proofs\.yml run, their first receipt(; the other theorems of this row have a hosted receipt: )?')
RC_TAIL_OLD = (f'; each of those receipted theorems corroborated on PR head {C7} by run {RUN7} (actual tested merge {M7} with base {BASE2}; '
               'receipt theorem_count 60)')

# ----------------------------------------------------------------------------- ... and the receipted forms (shape: update_register6 / 8 tier 3)
RECEIPT_PARENS = (f'(job proof-evidence: fresh build, leanchecker PASS, `#print axioms` within {AXIOMS}, receipt formalization_status kernel-checked, '
                  f'theorem_count 73, worktree CLEAN; artifact lean-specification-proofs-{RUN8}-1, sha256 {SHA8})')


def tier4_phrase(verb: str) -> str:
    they_were = 'it was' if verb == 'was' else 'they were'
    return (f'{verb} added after PR head {C7} and kernel-checked by the hosted run {RUN8} (PR head {C8}; actual tested merge {M8} with base {BASE3}) '
            f'{RECEIPT_PARENS}; {they_were} first receipted by that run (PR #434, event pull_request); base {BASE3} is the head of PR #423, the base '
            'of PR #434, and the merge has the same tree as its head')


TIER4_PHRASE = tier4_phrase('were')
WF_TIER4 = (f'; the theorems added after PR head {C7} (modules Side24Analysis, Side24Interfaces, RNCluster) are receipted by run {RUN8} (PR head {C8}; '
            f'actual tested merge {M8} with base {BASE3}) (artifact lean-specification-proofs-{RUN8}-1, sha256 {SHA8}; receipt formalization_status '
            'kernel-checked, leanchecker PASS, theorem_count 73, worktree CLEAN) and were first receipted by that run; base '
            f'{BASE3} is the head of PR #423, the base of PR #434; same tree as the head')
CORROB8_PHRASE = (f'All 73 theorems of PR head {C8}, including every theorem of the tiers above, are receipted on that head by run {RUN8} (actual tested '
                  f'merge {M8} with base {BASE3}; receipt theorem_count 73; artifact lean-specification-proofs-{RUN8}-1, sha256 {SHA8})')
WF_CORROB8 = (f'; all 73 theorems of PR head {C8} receipted on that head by run {RUN8} (actual tested merge {M8} with base {BASE3}; receipt '
              f'theorem_count 73; artifact lean-specification-proofs-{RUN8}-1, sha256 {SHA8})')


def free_wf_sub(m):
    noun, names, verb, pron = m.groups()
    was_were = 'was' if verb == 'was' else 'were'
    return (f'.github/workflows/lean-specification-proofs.yml; hosted run {RUN8} (PR head {C8}; actual tested checkout the merge commit {M8} of that head '
            f'into base {BASE3}, the synthetic merge GitHub builds for a pull_request run, whose tree equals the head\'s and which the receipt records as '
            f'checked_commit) (job proof-evidence, artifact lean-specification-proofs-{RUN8}-1, sha256 {SHA8}): receipt formalization_status kernel-checked, '
            f'leanchecker PASS, theorem_count 73, worktree CLEAN; the proof-packet {noun} of this row ({names}) {verb} added after PR head {C7} (run {RUN7}, '
            f'actual tested merge {M7} with base {BASE2}, receipt theorem_count 60, which does not name {pron}) and {was_were} first receipted by that run; '
            f'base {BASE3} is the head of PR #423, the base of PR #434')


def rc_sub(m):
    n, noun, names, lead_in = m.groups()
    if lead_in:     # a row that also has earlier-tier theorems: the lead-in continues with their receipts
        return (f'every theorem of this row has a hosted receipt: the {n} added after PR head {C7} ({names}) first by run {RUN8} (PR head {C8}; actual tested '
                f'merge {M8} with base {BASE3}) (receipt theorem_count 73); ')
    assert n == '1' and noun == 'theorem', m.group(0)
    return (f'the theorem of this row ({names}, added after PR head {C7}) has a hosted receipt: first receipted by run {RUN8} (PR head {C8}; actual tested '
            f'merge {M8} with base {BASE3}) (receipt theorem_count 73)')


RC_TAIL_NEW = (f'; each theorem of the earlier runs corroborated on PR head {C7} by run {RUN7} (actual tested merge {M7} with base {BASE2}; receipt '
               f'theorem_count 60); every theorem of this row receipted on PR head {C8} by run {RUN8} (actual tested merge {M8} with base {BASE3}; receipt '
               'theorem_count 73)')


# ----------------------------------------------------------------------------- load: committed == worktree
head = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], check=True, capture_output=True, text=True).stdout.strip()
if head != EXPECTED_HEAD:
    sys.exit(f'REFUSED: clone HEAD is {head}, expected {EXPECTED_HEAD}')
committed = subprocess.run(['git', '-C', str(REPO), 'show', f'HEAD:{STMT_REL}/REGISTER.json'], check=True, capture_output=True, text=True,
                           encoding='utf-8').stdout
worktree = REG_PATH.read_text(encoding='utf-8')
if committed != worktree:
    sys.exit('REFUSED: the worktree REGISTER.json differs from the committed one (git show HEAD:...)')
old = json.loads(committed)
reg = json.loads(committed)
rows = {c['claim_id']: c for c in reg['claims']}
old_rows = {c['claim_id']: c for c in old['claims']}
assert len(rows) == 171 == len(old_rows)
C_ROWS = ('side24-coefficient', 'math.side24-coefficient', 'OA-C6-RARE-CLUSTER-20260929-v1')


def strings(o):
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from strings(v)


def count_all(o, needle):
    return sum(s.count(needle) for s in strings(o))


def census(r):
    return {'statements': sum(len(c['statements']) for c in r['claims']),
            'proved': sum(1 for c in r['claims'] for s in c['statements'] if s['formal_progress'] == 'proved'),
            'kernel_evidence': sum(len(c['kernel_evidence']) for c in r['claims']),
            'claims': len(r['claims']),
            'claim_fp': {k: sum(1 for c in r['claims'] if c['formal_progress'] == k) for k in ('none', 'specified')}}


before = census(old)
assert before == {'statements': 352, 'proved': 155, 'kernel_evidence': 28, 'claims': 171, 'claim_fp': {'none': 138, 'specified': 33}}, before
triaged = {c['claim_id'] for c in old['claims'] if 'Round-4 triage (2026-10-11' in c['next_step']}
assert len(triaged) == 30 and all(old_rows[cid]['next_step'].count('Round-4 triage (2026-10-11') == 1 for cid in triaged)
pending_before = {'PENDING4': sum(len(PENDING4_RE.findall(s)) for s in strings(old)), 'WF_PENDING4': count_all(old, WF_PENDING4),
                  'free_wf': sum(1 for c in old['claims'] for e in c['kernel_evidence'] if FREE_WF_RE.match(e['receipt_or_workflow'] or '')),
                  'rc': sum(len(RC_RE.findall(s)) for s in strings(old)), 'rc_tail': count_all(old, RC_TAIL_OLD)}
if pending_before['PENDING4'] == 0:
    sys.exit('REFUSED: the register carries no round-4 pending clause (already bound?)')
assert pending_before == {'PENDING4': 5, 'WF_PENDING4': 1, 'free_wf': 1, 'rc': 2, 'rc_tail': 1}, pending_before
assert count_all(old, 'no hosted receipt yet') == 1 and count_all(old, 'await the next hosted') == 2
assert count_all(old, 'not evidence about any commit') == 5 and count_all(old, 'the last head with a hosted receipt') == 6
assert RUN8 not in committed and RUN_STMT not in committed and RUN_AGG not in committed

# ----------------------------------------------------------------------------- A. kernel_evidence rows
counts = {'PENDING4_note': 0, 'WF_PENDING4': 0, 'free_wf': 0, 'CORROB8': 0, 'WF_CORROB8': 0, 'PENDING4_text': 0, 'rc': 0, 'rc_tail': 0}
pp_rows = []
for cid, row in rows.items():
    mine = [s['declaration'] for s in row['statements'] if s['package'] == PROOF_PACKAGE]
    pp = [e for e in row['kernel_evidence'] if e['package'] == PROOF_PACKAGE]
    assert len(pp) == (1 if mine else 0), (cid, len(pp), len(mine))
    if not pp:
        continue
    pp_rows.append(cid)
    e = pp[0]
    note, wf = e['note'], e['receipt_or_workflow']
    assert str(len(mine)) in note.split(' theorem')[0], (cid, note[:60])
    earlier = CORROB_PHRASE + '. ' in note
    assert earlier == (WF_CORROB in wf), cid
    # (a) the round-4 pending clauses -> receipted tier-4 phrases
    note, n = PENDING4_RE.subn(lambda m: tier4_phrase(m.group(1)), note)
    counts['PENDING4_note'] += n
    if WF_PENDING4 in wf:
        assert wf.count(WF_PENDING4) == 1 and wf.endswith(WF_PENDING4), cid
        wf = wf.replace(WF_PENDING4, WF_TIER4)
        counts['WF_PENDING4'] += 1
    m = FREE_WF_RE.match(wf)
    if m:
        assert not earlier and n == 1, cid     # a receipt-free row has only round-4 theorems
        wf = free_wf_sub(m)
        counts['free_wf'] += 1
    # one corroboration clause for the rows with earlier tiers: all 73 theorems receipted on PR head 48bfcbae by run 38180705952
    if earlier:
        assert note.count(PROVED_LABEL_SENTENCE) == 1, cid
        note = note.replace(PROVED_LABEL_SENTENCE, '. ' + CORROB8_PHRASE + PROVED_LABEL_SENTENCE)
        wf = wf + WF_CORROB8
        counts['CORROB8'] += 1
        counts['WF_CORROB8'] += 1
    assert (n > 0) == any(s.split('.')[2] in ('Side24Analysis', 'Side24Interfaces', 'RNCluster') for s in mine), cid   # tier 4 exactly where round-4 theorems sit
    e['note'], e['receipt_or_workflow'] = note, wf
assert len(pp_rows) == 10 and counts['PENDING4_note'] == 2 and counts['WF_PENDING4'] == 1 and counts['free_wf'] == 1, counts
assert counts['CORROB8'] == 9 and counts['WF_CORROB8'] == 9, counts

# ----------------------------------------------------------------------------- B. the three rows' next_step / not_established
for cid in C_ROWS:
    row = rows[cid]
    for key in ('not_established', 'next_step'):
        text, n = PENDING4_RE.subn(lambda m: tier4_phrase(m.group(1)), row[key])
        counts['PENDING4_text'] += n
        text, n = RC_RE.subn(rc_sub, text)
        counts['rc'] += n
        if RC_TAIL_OLD in text:
            assert text.count(RC_TAIL_OLD) == 1 and n == 1 and cid == 'side24-coefficient', cid
            text = text.replace(RC_TAIL_OLD, RC_TAIL_NEW)
            counts['rc_tail'] += 1
        row[key] = text
assert counts['PENDING4_text'] == 3 and counts['rc'] == 2 and counts['rc_tail'] == 1, counts
assert rows['math.side24-coefficient']['next_step'] == old_rows['math.side24-coefficient']['next_step']   # 'As for row side24-coefficient. ' + triage: untouched

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
                        elif e_new[ek] != e_old[ek]:
                            changed_fields.setdefault(cid, []).append(f'kernel_evidence.{ek}')
        elif k in ('not_established', 'next_step') and cid in C_ROWS:
            if row[k] != o[k]:
                changed_fields.setdefault(cid, []).append(k)
        else:
            assert json.dumps(row[k], ensure_ascii=False, sort_keys=True) == json.dumps(o[k], ensure_ascii=False, sort_keys=True), (cid, k)
assert set(changed_fields) == set(pp_rows) | set(C_ROWS), sorted(set(changed_fields) ^ (set(pp_rows) | set(C_ROWS)))
assert all('kernel_evidence.note' in v and 'kernel_evidence.receipt_or_workflow' in v for c, v in changed_fields.items() if c in pp_rows)
assert sorted(changed_fields['math.side24-coefficient']) == ['not_established']
assert sorted(changed_fields['side24-coefficient']) == ['kernel_evidence.note', 'kernel_evidence.receipt_or_workflow', 'next_step', 'not_established']
assert sorted(changed_fields['OA-C6-RARE-CLUSTER-20260929-v1']) == ['kernel_evidence.note', 'kernel_evidence.receipt_or_workflow', 'next_step', 'not_established']
for c in reg['claims']:
    assert c['formal_progress'] == old_rows[c['claim_id']]['formal_progress'] and c['formal_progress'] in ('none', 'specified')
    assert c['statements'] == old_rows[c['claim_id']]['statements']
    for s in c['statements']:
        if s['package'] == STMT_PACKAGE:
            assert s['formal_progress'] == 'specified' and s['kind'] in ('def_prop', 'structure')
        else:
            assert (s['kind'] == 'theorem') == (s['formal_progress'] == 'proved')
        assert s['formal_progress'] != 'kernel-checked'
for cid in triaged:
    assert rows[cid]['next_step'].count('Round-4 triage (2026-10-11') == 1, cid
assert sum(c['next_step'].count('Round-4 triage (2026-10-11') for c in reg['claims']) == 30

# ----------------------------------------------------------------------------- D. after: no pending phrase; C422-E1 on every run id; counts
HISTORICAL_FIRST_RECEIPT = (f'run {RUN1} (PR head {C1}; actual tested merge {M1} with base {BASE}) produced the first receipt (`kernel-checked` '
                            'appears only there); later commits need their own run')   # p15-price-boundary, round 1: dated to PR head 00d2ac6e
assert old_rows['p15-price-boundary']['next_step'].count(HISTORICAL_FIRST_RECEIPT) == 1
survivors = []
for s in strings(reg):
    assert 'no hosted receipt yet' not in s and 'await the next hosted' not in s and 'Hosted receipt: none yet' not in s, s[:160]
    assert 'not evidence about any commit' not in s and 'the last head with a hosted receipt' not in s and 'first receipt for them' not in s, s[:160]
    for m in re.finditer(r'first receipt(?!ed)', s):
        assert HISTORICAL_FIRST_RECEIPT in s, s[:160]       # the one survivor: round 1's sentence, dated to PR head 00d2ac6e with its merge
        survivors.append(s[max(0, m.start() - 120):m.end() + 60])
    for run, merge in RUN_MERGE:
        if run in s:
            assert merge in s and RUN_HEAD[run] in s, (run, s[:160])
    for m in re.finditer(r'\b38\d{9}\b', s):   # every eleven-digit run id is in the binding table (38096827443 is main's workspace-landing run, pre-existing)
        assert m.group(0) in dict(RUN_MERGE) or m.group(0) == '38096827443', (m.group(0), s[:120])
assert len(survivors) == 1, survivors
assert count_all(reg, 'none yet') == count_all(old, 'none yet') - 1 == 1   # the surviving 'proof lane: none yet' is not a receipt statement
expected_run8 = (5 * TIER4_PHRASE.count(RUN8) + counts['WF_PENDING4'] * WF_TIER4.count(RUN8)
                 + counts['free_wf'] * rows['OA-C6-RARE-CLUSTER-20260929-v1']['kernel_evidence'][0]['receipt_or_workflow'].count(RUN8)
                 + counts['CORROB8'] * CORROB8_PHRASE.count(RUN8) + counts['WF_CORROB8'] * WF_CORROB8.count(RUN8)
                 + counts['rc'] * 1 + counts['rc_tail'] * RC_TAIL_NEW.count(RUN8))
assert count_all(reg, RUN8) == expected_run8, (count_all(reg, RUN8), expected_run8)   # every mention of run 38180705952 is one of the planned phrases
after = census(reg)
assert after == before, (before, after)
kc = json.dumps(reg, ensure_ascii=False)
assert '"formal_progress": "kernel-checked"' not in kc and '"formal_progress": "proved"' in kc

out = json.dumps(reg, indent=2, ensure_ascii=False) + '\n'
assert json.loads(out) == reg and out != committed
REG_PATH.write_text(out, encoding='utf-8')
occurrences_run8 = {cid: {k: v.count(RUN8) for k, v in (('not_established', row['not_established']), ('next_step', row['next_step']),
                                                          *((f'kernel_evidence.{ek}', e[ek]) for e in row['kernel_evidence'] if e['package'] == PROOF_PACKAGE for ek in ('note', 'receipt_or_workflow')))
                          if v.count(RUN8)}
                    for cid, row in rows.items() if RUN8 in json.dumps(row, ensure_ascii=False)}
summary = {'scientific_effect': 'NONE', 'committed_head': head, 'before': before, 'after': after,
           'replacements': counts, 'pending_forms_before': pending_before, 'changed_fields': changed_fields,
           'run8_mentions_by_field': occurrences_run8,
           'receipt': {'run': RUN8, 'pr': 434, 'pr_head': EXPECTED_HEAD, 'checked_commit_merge': M8, 'base': BASE3, 'merge_tree': TREE8,
                       'theorem_count': 73, 'covered_props': 72, 'leanchecker': 'PASS', 'worktree': 'CLEAN',
                       'artifact': f'lean-specification-proofs-{RUN8}-1', 'artifact_sha256_github_digest': SHA8,
                       'companion_runs_not_cited_in_register': [RUN_STMT, RUN_AGG]},
           'run_merge_bindings_asserted': dict(RUN_MERGE), 'run_head_bindings_asserted': RUN_HEAD,
           'first_receipt_survivor_historical': survivors,
           'none_yet_survivor': [s[max(0, s.index('none yet') - 80):s.index('none yet') + 70] for s in strings(reg) if 'none yet' in s],
           'triage_rows_exactly_one_sentence': len(triaged),
           'claim_level_formal_progress_unchanged': all(rows[c]['formal_progress'] == old_rows[c]['formal_progress'] for c in rows),
           'phrases': {'TIER4': TIER4_PHRASE, 'TIER4_singular': tier4_phrase('was'), 'WF_TIER4': WF_TIER4, 'CORROB8': CORROB8_PHRASE,
                       'WF_CORROB8': WF_CORROB8, 'RC_TAIL_NEW': RC_TAIL_NEW,
                       'free_wf_new': rows['OA-C6-RARE-CLUSTER-20260929-v1']['kernel_evidence'][0]['receipt_or_workflow'],
                       'side24_next_step_receipt_clause': rows['side24-coefficient']['next_step'].split('. Round-4 triage')[0],
                       'oa_c6_next_step_receipt_clause': rows['OA-C6-RARE-CLUSTER-20260929-v1']['next_step'].split('. Round-4 triage')[0]}}
(OUT_DIR / 'update_summary9.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps({k: v for k, v in summary.items() if k != 'phrases'}, indent=1, ensure_ascii=False))
