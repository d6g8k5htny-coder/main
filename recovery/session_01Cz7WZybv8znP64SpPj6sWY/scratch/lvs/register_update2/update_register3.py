"""Round-3 register update: the hosted run on commit 910809d6 receipted all 52 proof-packet theorems.
Scientific effect: NONE. Only evidence notes and the 'awaiting the next run' sentences change; no label moves.
Usage: update_register3.py <artifact-name> <artifact-sha256>
"""
import json, sys
from pathlib import Path
STMT = Path('<SCRATCH>/repos/math/specifications/lean_statements_20261010')
PROOF_PACKAGE = 'd6g8k5htny-coder/Math- specifications/lean_statement_proofs_20261011'
RUN1, C1, RUN2, C2 = '38116277597', '00d2ac6e', '38121713840', '910809d6'
ART2, SHA2 = sys.argv[1], sys.argv[2]
OLD_PENDING = ('added after commit ' + C1 + ': locally `lake env lean -DwarningAsError=true` exit 0 and `#print axioms` within '
               '{propext, Classical.choice, Quot.sound} on an unverified worktree, not evidence about any commit; the next hosted '
               'lean-specification-proofs.yml run is the first receipt for them')
NEW_PENDING = (f'added after commit {C1} and kernel-checked by the hosted run {RUN2} on commit {C2} (job proof-evidence: fresh build, '
               'leanchecker PASS, `#print axioms` within {propext, Classical.choice, Quot.sound}, receipt formalization_status kernel-checked, '
               'worktree CLEAN)')
OLD_WF = (f'.github/workflows/lean-specification-proofs.yml; hosted run {RUN1} on commit {C1} (job proof-evidence, artifact '
          f'lean-specification-proofs-{RUN1}-1, sha256 cd2c16b120cd2ae54f598082463a38167ac2e6ec479e2aaf51f27910b90949b6): receipt '
          f'formalization_status kernel-checked, leanchecker PASS, theorem_count 38, worktree CLEAN; theorems committed after {C1} await the next hosted run')
NEW_WF = (f'.github/workflows/lean-specification-proofs.yml; hosted run {RUN2} on commit {C2} (job proof-evidence, artifact {ART2}, sha256 {SHA2}): '
          f'receipt formalization_status kernel-checked, leanchecker PASS, theorem_count 52, worktree CLEAN; the 38 theorems of commit {C1} were '
          f'first receipted by run {RUN1} (artifact lean-specification-proofs-{RUN1}-1)')
OLD_38 = f'; artifact lean-specification-proofs-{RUN1}-1). '
NEW_38 = f'; artifact lean-specification-proofs-{RUN1}-1), and again by the hosted run {RUN2} on commit {C2}. '
OLD_NEXT = f'next hosted lean-specification-proofs.yml run for the theorems added after commit {C1}'
NEW_NEXT = f'every theorem of this row has a hosted receipt (run {RUN2} on commit {C2})'

reg = json.loads((STMT / 'REGISTER.json').read_text(encoding='utf-8'))
counts = {'pending': 0, 'wf': 0, 'again': 0, 'next': 0}
for row in reg['claims']:
    for ev in row['kernel_evidence']:
        if ev['package'] != PROOF_PACKAGE:
            continue
        assert ev['receipt_or_workflow'] == OLD_WF, ev['receipt_or_workflow'][:80]
        ev['receipt_or_workflow'] = NEW_WF; counts['wf'] += 1
        n = ev['note']
        if OLD_PENDING in n:
            n = n.replace(OLD_PENDING, NEW_PENDING); counts['pending'] += 1
        if OLD_38 in n:
            n = n.replace(OLD_38, NEW_38); counts['again'] += 1
        ev['note'] = n
    for k in ('not_established', 'next_step'):
        v = row[k]
        if OLD_PENDING in v:
            v = v.replace(OLD_PENDING, NEW_PENDING); counts['pending'] += 1
        if OLD_NEXT in v:
            v = v.replace(OLD_NEXT, NEW_NEXT); counts['next'] += 1
        row[k] = v
s = json.dumps(reg, indent=2, ensure_ascii=False) + '\n'
for bad in ('await the next hosted run', 'the next hosted lean-specification-proofs.yml run', 'awaiting the next', 'is the first receipt for them'):
    assert bad not in s, bad
for c in reg['claims']:
    assert c['formal_progress'] in ('none', 'specified')
(STMT / 'REGISTER.json').write_text(s, encoding='utf-8')
print('register round 3:', counts)
