import sys, json
from pathlib import Path
PKT = Path(sys.argv[1]); STMT = Path(sys.argv[2]); FILE = Path(sys.argv[3])
sys.path.insert(0, str(PKT))
import check
reg = check.load_register(STMT)
registered = reg[0] if isinstance(reg, tuple) else reg
if isinstance(registered, dict) and 'registered' in registered:
    registered = registered['registered']
print('REGISTERED_HAS_PROP', 'UniversalLaw.Spec.Side24.RayleighSquareLaw' in registered, registered.get('UniversalLaw.Spec.Side24.RayleighSquareLaw'))
info = check.scan_module(FILE.read_text(encoding='utf-8'), 'Side24Interfaces', registered)
print('IMPORTS', info['imports'])
print('NAMESPACES', info['namespaces'])
print('AUX_THEOREMS', info.get('aux_theorems'), 'AUX_DEFS', info.get('auxiliary_definitions'), 'SET_OPTIONS', info.get('set_options'))
for t in info['theorems']:
    print('THEOREM', json.dumps(t, ensure_ascii=False))
print('SCAN_OK')
# negative control: the scanner must refuse a mutated copy (extra hypothesis on the inventory theorem)
txt = FILE.read_text(encoding='utf-8')
mut = txt.replace('theorem rayleighSquareLaw (I : ConeMomentInterface) :', 'theorem rayleighSquareLaw (I : ConeMomentInterface) (h : 0 < (1:ℝ)) :')
assert mut != txt
try:
    check.scan_module(mut, 'Side24Interfaces', registered); print('NEGATIVE_CONTROL_FAILED: mutated file accepted')
except Exception as e:
    print('NEGATIVE_CONTROL_OK:', str(e)[:160])
mut2 = txt.replace('  linarith\n\nend UniversalLaw', '  sorry\n\nend UniversalLaw')
assert mut2 != txt
try:
    check.scan_module(mut2, 'Side24Interfaces', registered); print('NEGATIVE_CONTROL_FAILED: sorry accepted')
except Exception as e:
    print('NEGATIVE_CONTROL_OK:', str(e)[:160])
