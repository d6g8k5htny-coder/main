"""Independent audit helper: scan the three new modules with check.scan_module, enumerate every
theorem (inventory + Aux) from the comment-stripped code with namespace tracking, and write a
scratch Lean file with `#check @` for each inventory theorem and `#print axioms` for every theorem."""
import json, re, sys
from pathlib import Path
PACKET = Path('<SCRATCH>/repos/math/specifications/lean_statement_proofs_20261011')
sys.path.insert(0, str(PACKET))
import check
stmt = (PACKET / check.STATEMENT_DIR).resolve()
registered, structures, _ = check.load_register(stmt)
print('registered props:', len(registered))
mods = ['Side24Analysis', 'Side24Interfaces', 'RNCluster']
inventory, allthms = [], []
for stem in mods:
    text = (PACKET / 'SpecificationProofs' / f'{stem}.lean').read_text(encoding='utf-8')
    scan = check.scan_module(text, stem, registered)
    print(f'--- {stem}: SCAN_OK imports={scan["imports"]} namespaces={scan["namespaces"]} inventory={len(scan["theorems"])} '
          f'aux_theorems={scan["aux_theorems"]} auxiliary_definitions={scan["auxiliary_definitions"]} set_options={scan["set_options"]}')
    for t in scan['theorems']:
        print('    ', t['name'], ':', t['proves'], 'binders', [(b['kind'], b['names']) for b in t['binders']], 'args', t['arguments'])
        inventory.append((t['name'], t['proves'], t['binders'], t['arguments']))
    # independent enumeration of every theorem from the comment-stripped code
    code, docs = check.lex(text)
    ns = []
    mine_inv, mine_aux = [], []
    for ln in code.split('\n'):
        if not ln or ln[0].isspace():
            continue
        m = re.match(r'^namespace\s+(\S+)\s*$', ln)
        if m:
            ns.append(m.group(1)); continue
        m = re.match(r'^end(?:\s+(\S+))?\s*$', ln)
        if m:
            assert ns and ns[-1] == m.group(1), (stem, ln); ns.pop(); continue
        m = check.DECL_LINE.match(ln)
        if m and m.group(3) in ('theorem', 'lemma'):
            fq = '.'.join(ns + [m.group(4)])
            rel = fq[len(check.NAMESPACE + stem) + 1:].split('.')
            (mine_aux if ('Aux' in rel[:-1] or 'private' in m.group(2)) else mine_inv).append(fq)
        elif m:
            print('    NON-THEOREM DECL:', ln[:100])
    assert not ns, (stem, ns)
    assert mine_inv == [t['name'] for t in scan['theorems']], (mine_inv, [t['name'] for t in scan['theorems']])
    assert len(mine_aux) == scan['aux_theorems'], (len(mine_aux), scan['aux_theorems'])
    print(f'    independent enumeration agrees: {len(mine_inv)} inventory, {len(mine_aux)} Aux')
    allthms += mine_inv + mine_aux
assert len(allthms) == len(set(allthms))
print('TOTAL targets:', len(allthms), '(inventory', len(inventory), ')')
out = []
out.append('import SpecificationProofs\n')
for name, *_ in inventory:
    out.append(f'#check @{name}\n')
for name in allthms:
    out.append(f'#print axioms {name}\n')
Path('_audit_scratch.lean').write_text(''.join(out), encoding='utf-8')
json.dump({'inventory': [{'name': n, 'proves': p, 'binders': b, 'arguments': a} for n, p, b, a in inventory], 'targets': allthms}, open('_audit_targets.json', 'w'), indent=1)
# informational: compare with the integrator's target list
try:
    theirs = json.load(open('<SCRATCH>/lvs/round4/integrate/_int_axioms_targets.json'))
    if isinstance(theirs, dict):
        theirs = theirs.get('targets', list(theirs))
    print('integrator target list: len', len(theirs), 'same set as mine:', set(theirs) == set(allthms))
except Exception as e:
    print('could not compare with integrator list:', e)
