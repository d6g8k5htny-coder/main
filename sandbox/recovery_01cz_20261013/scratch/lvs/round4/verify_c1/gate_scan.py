import sys, json
from pathlib import Path
PKT = Path(sys.argv[1]); STMT = Path(sys.argv[2]); FILE = Path(sys.argv[3])
sys.path.insert(0, str(PKT))
import check
reg = check.load_register(STMT)
registered = reg[0] if isinstance(reg, tuple) else reg
if isinstance(registered, dict) and 'registered' in registered:
    registered = registered['registered']
text = FILE.read_text(encoding='utf-8')
code, docs = check.lex(text)
print('REFUSED_TOKENS', check.refused_tokens(code))
print('HASH_CMD', check.HASH_COMMAND.search(code))
print('SET_OPTIONS', check.check_set_options(code, text.split('\n'), 'x'))
info = check.scan_module(text, 'Side24Interfaces', registered)
print('IMPORTS', info['imports'])
print('NAMESPACES', info['namespaces'])
print('AUX_THEOREMS', info.get('aux_theorems'))
print('AUX_DEFS', info.get('auxiliary_definitions'))
for t in info['theorems']:
    print('THEOREM', json.dumps(t, ensure_ascii=False))
print('REGISTERED_PROP_ENTRY', json.dumps(registered.get('UniversalLaw.Spec.Side24.ConeMomentD1Identity'), ensure_ascii=False))
print('SCAN_OK')
