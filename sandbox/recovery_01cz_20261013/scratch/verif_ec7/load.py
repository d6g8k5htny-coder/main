import importlib.util, json, sys
P='<SCRATCH>/repos/math_c264ctl/companions/torus_filtration_20261011/check.py'
spec=importlib.util.spec_from_file_location('chk',P); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
c=json.load(open('<SCRATCH>/lvs/c264_controls/gate/controls.json'))
for k,v in c.items():
    print(k, tuple(v)==m.FALSE_CONTROLS.get(k))
print(list(m.FALSE_CONTROLS)[-3:]==list(c))
print('controls', m.CONTROLS)
print('n evidence', len(m.EVIDENCE_FILES))
print(sorted(m.EVIDENCE_FILES))
print('BUILDS', m.BUILDS)
print('EMITTED', m.EMITTED_FILES)
print('LEANCHECKER', m.LEANCHECKER)
print(repr(m.ROOT_SOURCE))
print(m.REQUIRED_FILES)
import os
d='<SCRATCH>/repos/math_c264ctl/.lake/torus-filtration-evidence'
on=set(os.listdir(d))-{'receipt.json'}
print('EVIDENCE_FILES == dir', on==set(m.EVIDENCE_FILES), on^set(m.EVIDENCE_FILES))
# emitted files equal to fresh emit? compare emitted control files to gate/out
g='<SCRATCH>/lvs/c264_controls/gate/out'
if os.path.isdir(g):
    for n in ('RejectMergerKillsPointClass.lean','RejectSeparationWithoutNonzeroCoeff.lean','RejectNonmergerKillsDifference.lean'):
        p=os.path.join(g,n)
        print(n, os.path.exists(p) and open(p,'rb').read()==open(os.path.join(d,n),'rb').read())
