"""Round-4 register update (reviewer finding C422-E1): bind each hosted run to BOTH the PR head and the
actually tested checkout (GitHub's pull_request runs check out the synthetic merge of the head into the
base; the receipt's checked_commit is that merge commit). Scientific effect: NONE; evidence wording only.

Verified before running (git API, 2026-10-11): each merge commit has parents [BASE, head] and the same tree as
its head: c4c66bb8 -> 908e5efc (= tree of 00d2ac6e), 5b198699 -> 708521e8 (= 910809d6), 866136cc -> c39c3913 (= d9b8dd33).
"""
import json
from pathlib import Path
STMT = Path('<SCRATCH>/repos/math/specifications/lean_statements_20261010')
PROOF_PACKAGE = 'd6g8k5htny-coder/Math- specifications/lean_statement_proofs_20261011'
BASE = 'b2d70d77876a10c365f5db5d3b6cafcf180af4e2'
RUN1, C1, M1 = '38116277597', '00d2ac6e', 'c4c66bb8cb26f0cd9bc03d52bb25f1b3925b272c'
RUN2, C2, M2 = '38121713840', '910809d6', '5b198699308e0585904c21d12bf540cf2475d2f0'
RUN3, C3, M3 = '38122386459', 'd9b8dd33', '866136cca66b46e6dbd3186b47c9340f7e93b4f6'
ART3, SHA3 = 'lean-specification-proofs-38122386459-1', 'edd52190d6af5828e8229c161c3b77e87a540caf440070d66b1b7f415008ebc3'
# compact binding (the reviewer's form): run (PR head; actual tested merge with base)
B1 = f'run {RUN1} (PR head {C1}; actual tested merge {M1} with base {BASE})'
B2 = f'run {RUN2} (PR head {C2}; actual tested merge {M2} with base {BASE})'
# long form, once per receipt_or_workflow, explaining the mechanism
NEW2 = (f'run {RUN2} (PR head {C2}; actual tested checkout the merge commit {M2} of that head into base {BASE}, '
        'the synthetic merge GitHub builds for a pull_request run, whose tree equals the head\'s and which the receipt records as checked_commit)')
CORROB = (f'; corroborated on PR head {C3} by run {RUN3} (actual tested merge {M3} with base {BASE}, same tree '
          f'as the head; receipt formalization_status kernel-checked, theorem_count 52, worktree CLEAN; artifact {ART3}, sha256 {SHA3})')
OLD1, OLD2 = f'run {RUN1} on commit {C1}', f'run {RUN2} on commit {C2}'
OLD3 = f'the 38 theorems of commit {C1} were first receipted by run {RUN1} (artifact'
NEW3 = f'the 38 theorems of PR head {C1} were first receipted by run {RUN1} (actual tested merge {M1} with base {BASE}, same tree as the head; artifact'
OLD4 = f'among the 38 theorems of commit {C1} kernel-checked by the hosted run {RUN1} (job proof-evidence:'
NEW4 = f'among the 38 theorems of PR head {C1} kernel-checked by the hosted run {RUN1} (actual tested merge {M1} with base {BASE}; job proof-evidence:'
reg = json.loads((STMT / 'REGISTER.json').read_text(encoding='utf-8'))
counts = {'wf': 0, 'old1': 0, 'old2': 0, 'old3': 0, 'old4': 0}


def fix(s):
    for k, old, new in (('old3', OLD3, NEW3), ('old4', OLD4, NEW4), ('old2', OLD2, B2), ('old1', OLD1, B1)):
        counts[k] += s.count(old)
        s = s.replace(old, new)
    return s


for row in reg['claims']:
    for ev in row['kernel_evidence']:
        if ev['package'] != PROOF_PACKAGE:
            continue
        w = ev['receipt_or_workflow']
        assert OLD2 in w and OLD3 in w and CORROB not in w, w[:100]
        counts['old2'] += 1; counts['old3'] += 1
        w = w.replace(OLD2, NEW2).replace(OLD3, NEW3)
        assert OLD1 not in w and OLD4 not in w
        ev['receipt_or_workflow'] = w + CORROB; counts['wf'] += 1
        ev['note'] = fix(ev['note'])
    for k in ('not_established', 'next_step'):
        row[k] = fix(row[k])
s = json.dumps(reg, indent=2, ensure_ascii=False) + '\n'
for old in (OLD1, OLD2, OLD3, OLD4):
    assert old not in s, old
for c in reg['claims']:
    assert c['formal_progress'] in ('none', 'specified')


# binding check: every string that names one of the three runs also names its tested merge commit
def strings(o):
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from strings(v)


bound = {'run1': 0, 'run2': 0, 'run3': 0}
for t in strings(reg):
    for key, run, merge in (('run1', RUN1, M1), ('run2', RUN2, M2), ('run3', RUN3, M3)):
        if run in t:
            assert merge in t, (run, t[:120])
            bound[key] += 1
(STMT / 'REGISTER.json').write_text(s, encoding='utf-8')
print('register round 4:', counts, 'bound strings:', bound)
