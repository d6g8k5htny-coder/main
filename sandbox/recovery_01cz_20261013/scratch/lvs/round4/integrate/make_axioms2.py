import sys, re, json
from pathlib import Path
P = Path('<SCRATCH>/repos/math/specifications/lean_statement_proofs_20261011')
sys.path.insert(0, str(P))
import check
W = Path('<SCRATCH>/lvs/round4/work')
names = []
for stem in ('Side24Analysis', 'Side24Interfaces', 'RNCluster'):
    ns = 'UniversalLaw.Proofs.' + stem
    code, _ = check.lex((W / (stem + '.lean')).read_text(encoding='utf-8'))
    in_aux = False
    for line in code.splitlines():
        if line == 'namespace Aux':
            in_aux = True
        elif line == 'end Aux':
            in_aux = False
        m = re.match(r"^theorem\s+(\S+)", line)
        if m:
            names.append(ns + ('.Aux.' if in_aux else '.') + m.group(1))
assert len(names) == len(set(names)), 'duplicate target'
Path('_int_axioms.lean').write_text('import SpecificationProofs\n\n' + ''.join(f'#print axioms {n}\n' for n in names), encoding='utf-8')
Path('_int_axioms_targets.json').write_text(json.dumps(names, indent=1), encoding='utf-8')
print(len(names), 'targets')
