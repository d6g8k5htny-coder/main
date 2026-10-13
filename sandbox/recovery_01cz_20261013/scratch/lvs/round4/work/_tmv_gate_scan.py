import sys, json
from pathlib import Path
PKT = Path(sys.argv[1]); STMT = Path(sys.argv[2]); FILE = Path(sys.argv[3])
sys.path.insert(0, str(PKT))
import check
reg = check.load_register(STMT)
registered = reg[0] if isinstance(reg, tuple) else reg
if isinstance(registered, dict) and 'registered' in registered:
    registered = registered['registered']
print('REGISTERED_TYPE', type(registered).__name__, 'HAS_PROP', 'UniversalLaw.Spec.Side24.TraceMoments' in registered)
try:
    print('REG_ENTRY', json.dumps(registered['UniversalLaw.Spec.Side24.TraceMoments'], ensure_ascii=False)[:600])
except Exception as e:
    print('REG_ENTRY_ERR', e)
text = FILE.read_text(encoding='utf-8')
info = check.scan_module(text, 'Side24Interfaces', registered)
print('IMPORTS', info['imports'])
print('NAMESPACES', info['namespaces'])
print('AUX_THEOREMS', info.get('aux_theorems'), 'AUXILIARY', info.get('auxiliary'))
print('SET_OPTIONS', info.get('set_options'))
for t in info['theorems']:
    print('THEOREM', json.dumps(t, ensure_ascii=False))
code, docs = check.lex(text)
print('REFUSED', check.refused_tokens(code))
print('HASH', check.HASH_COMMAND.search(code))
print('SCAN_OK')
# negative control: weaken the file and confirm scan_module refuses it
bad = text.replace('theorem traceMoments (I : ConeMomentInterface) : UniversalLaw.Spec.Side24.TraceMoments I := by',
                   'theorem traceMoments (I : ConeMomentInterface) (h : True) : UniversalLaw.Spec.Side24.TraceMoments I := by')
try:
    check.scan_module(bad, 'Side24Interfaces', registered); print('CONTROL_EXTRA_HYP: NOT REFUSED (checker weak)')
except Exception as e:
    print('CONTROL_EXTRA_HYP refused:', str(e)[:200])
bad2 = text.replace('  simp\n', '  sorry\n', 1)
try:
    check.scan_module(bad2, 'Side24Interfaces', registered); print('CONTROL_SORRY: NOT REFUSED')
except Exception as e:
    print('CONTROL_SORRY refused:', str(e)[:200])
