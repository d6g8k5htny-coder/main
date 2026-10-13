"""Round-6 register update: bind the hosted receipts of the round-3 theorems (module P15Interpolation).

usage: update_register6.py RUN6 M6 SHA6      (the run on PR head 0dcb0016: run id, checked_commit merge sha, artifact sha256)

Scientific effect: NONE. Replaces the "no hosted receipt yet" phrases written by update_register5 with the receipt
binding (PR head AND actual tested merge, C422-E1) for the run on head 0dcb0016 and the first-receipt run on head
88c21058. No entry is added or removed; no label changes.
"""
import json
import re
import sys
from pathlib import Path

SCRATCH = Path('<SCRATCH>')
REG = Path(__import__('os').environ.get('RU6_REGISTER', str(SCRATCH / 'repos/math/specifications/lean_statements_20261010/REGISTER.json')))
RUN6, M6, SHA6 = sys.argv[1:4]
assert RUN6.isdigit() and re.fullmatch(r'[0-9a-f]{40}', M6) and re.fullmatch(r'[0-9a-f]{64}', SHA6)
C6 = '0dcb0016'
RUNA, C5A, M5A, SHA5A = '38149524759', '88c21058', '3306233bbdeadb15d9b4b07b94cc52a80ed8caeb', 'ddefbdcc63223d05fd8ed34addbeebc199e271525d87dead01109cddc529e836'
BASE2 = '79731b961fac555f33be3cd90e379b246a5a80e3'     # head of PR #422, the base of PR #423
BASE = 'b2d70d77876a10c365f5db5d3b6cafcf180af4e2'
RUN4, C4, M4 = '38124262242', '79731b96', '318ff194eecc42c39a6b27c1d7bf03f30d6a34d9'
RUN2, C2, M2 = '38121713840', '910809d6', '5b198699308e0585904c21d12bf540cf2475d2f0'
AXIOMS = '{propext, Classical.choice, Quot.sound}'

PENDING_PHRASE = (f'were added after PR head {C4} (the last head with a hosted receipt: run {RUN4}, actual tested merge {M4} with base {BASE}, '
                  f'receipt theorem_count 52): locally `lake env lean -DwarningAsError=true` exit 0 and `#print axioms` within {AXIOMS} on an '
                  'unverified worktree, not evidence about any commit; the next hosted lean-specification-proofs.yml run is the first receipt for them')
WF_PENDING = (f'; theorems added after PR head {C4} (run {RUN4}, actual tested merge {M4} with base {BASE}, receipt theorem_count 52) have no hosted '
              'receipt yet: the next hosted lean-specification-proofs.yml run is their first receipt')
TIER3_PHRASE = (f'were added after PR head {C4} and kernel-checked by the hosted run {RUN6} (PR head {C6}; actual tested merge {M6} with base {BASE2}) '
                f'(job proof-evidence: fresh build, leanchecker PASS, `#print axioms` within {AXIOMS}, receipt formalization_status kernel-checked, '
                f'theorem_count 60, worktree CLEAN; artifact lean-specification-proofs-{RUN6}-1, sha256 {SHA6}); they were first receipted by run {RUNA} '
                f'(PR head {C5A}; actual tested merge {M5A} with base {BASE2}) (same Lean bytes; artifact lean-specification-proofs-{RUNA}-1, sha256 {SHA5A}); '
                f'base {BASE2} is the head of PR #422, and each merge has the same tree as its head')
WF_TIER3 = (f'; the theorems added after PR head {C4} (module P15Interpolation) are receipted by run {RUN6} (PR head {C6}; actual tested merge {M6} with base '
            f'{BASE2}) (artifact lean-specification-proofs-{RUN6}-1, sha256 {SHA6}; receipt formalization_status kernel-checked, leanchecker PASS, '
            f'theorem_count 60, worktree CLEAN) and were first receipted by run {RUNA} (PR head {C5A}; actual tested merge {M5A} with base {BASE2}) '
            f'(artifact lean-specification-proofs-{RUNA}-1, sha256 {SHA5A}); base {BASE2} is the head of PR #422; same trees as the heads')
RC_RE = re.compile(r'the (\d+) theorem\(s\) of this row added after PR head 79731b96 \((.*?)\) await the next hosted lean-specification-proofs\.yml run, their first receipt')


def rc_sub(m):
    return (f'every theorem of this row has a hosted receipt: the {m.group(1)} added after PR head {C4} ({m.group(2)}) by run {RUN6} (PR head {C6}; '
            f'actual tested merge {M6} with base {BASE2}) and first by run {RUNA} (PR head {C5A}; actual tested merge {M5A} with base {BASE2}), the others '
            f'by run {RUN2} (PR head {C2}; actual tested merge {M2} with base {BASE})')


def walk(o, counts):
    if isinstance(o, str):
        n1, n2 = o.count(PENDING_PHRASE), o.count(WF_PENDING)
        counts['pending'] += n1; counts['wf'] += n2
        o = o.replace(PENDING_PHRASE, TIER3_PHRASE).replace(WF_PENDING, WF_TIER3)
        o, n3 = RC_RE.subn(rc_sub, o); counts['rc'] += n3
        return o
    if isinstance(o, dict):
        return {k: walk(v, counts) for k, v in o.items()}
    if isinstance(o, list):
        return [walk(v, counts) for v in o]
    return o


def strings(o):
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from strings(v)


reg = json.loads(REG.read_text(encoding='utf-8'))
counts = {'pending': 0, 'wf': 0, 'rc': 0}
new = walk(reg, counts)
assert counts['pending'] >= 3 and counts['wf'] >= 2 and counts['rc'] == 1, counts   # 2 kernel_evidence notes + not_established (+ math row), 2 wf, 1 next_step
for s in strings(new):
    assert 'no hosted receipt yet' not in s and 'await the next hosted' not in s and 'not evidence about any commit' not in s, s[:120]
    for run, merge in ((RUN6, M6), (RUNA, M5A), (RUN2, M2), (RUN4, M4), ('38116277597', 'c4c66bb8cb26f0cd9bc03d52bb25f1b3925b272c'), ('38122386459', '866136cca66b46e6dbd3186b47c9340f7e93b4f6')):
        if run in s:
            assert merge in s, (run, s[:120])
for c in new['claims']:
    assert c['formal_progress'] in ('none', 'specified')
assert sum(len(c['statements']) for c in new['claims']) == sum(len(c['statements']) for c in reg['claims'])
REG.write_text(json.dumps(new, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print('update_register6', counts, REG)
