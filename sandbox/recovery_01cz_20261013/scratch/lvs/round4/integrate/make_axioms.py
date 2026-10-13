import re
from pathlib import Path
W = Path('<SCRATCH>/lvs/round4/work')
names = []
for stem in ('Side24Analysis', 'Side24Interfaces', 'RNCluster'):
    ns = 'UniversalLaw.Proofs.' + stem
    text = (W / (stem + '.lean')).read_text(encoding='utf-8')
    in_aux = False
    for line in text.splitlines():
        if line == 'namespace Aux':
            in_aux = True
        elif line == 'end Aux':
            in_aux = False
        m = re.match(r'^theorem\s+([A-Za-z0-9_\'₀-₉]+)', line)
        if m:
            names.append(ns + ('.Aux.' if in_aux else '.') + m.group(1))
out = Path(__file__).with_name('_int_axioms.lean')
out.write_text('import SpecificationProofs\n\n' + ''.join(f'#print axioms {n}\n' for n in names), encoding='utf-8')
print(len(names), 'targets written to', out)
print('\n'.join(names))
