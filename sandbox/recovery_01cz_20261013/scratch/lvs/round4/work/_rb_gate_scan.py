import sys, json
from pathlib import Path
PKT = Path(sys.argv[1]); STMT = Path(sys.argv[2]); FILE = Path(sys.argv[3])
sys.path.insert(0, str(PKT))
import check
reg = check.load_register(STMT)
registered = reg[0] if isinstance(reg, tuple) else reg
if isinstance(registered, dict) and 'registered' in registered:
    registered = registered['registered']
info = check.scan_module(FILE.read_text(encoding='utf-8'), 'Side24Analysis', registered)
print('IMPORTS', info['imports'])
print('NAMESPACES', info['namespaces'])
print('AUX_THEOREMS', info.get('aux_theorems'))
for t in info['theorems']:
    print('THEOREM', json.dumps(t, ensure_ascii=False))
print('SCAN_OK')
