import importlib.util, json, sys
from pathlib import Path
S = Path('<SCRATCH>')
check_path = S / 'repos/math/specifications/lean_statement_proofs_20261011/check.py'
spec = importlib.util.spec_from_file_location('packet_check', check_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
registered = mod.load_register(S / 'repos/math/specifications/lean_statements_20261010')
if isinstance(registered, tuple):
    registered = registered[0]
text = (S / 'lvs/round4/work/DerivativePairingBound.lean').read_text()
info = mod.scan_module(text, 'Side24Analysis', registered)
print(json.dumps({k: v for k, v in info.items() if k != 'theorems'}, indent=1))
for t in info['theorems']:
    print('THEOREM', t['name'], 'proves', t['proves'], 'binders', t['binders'], 'args', t['arguments'])
    print('  Proves:', t['doc']['proves'][:120])
print('SCAN OK')
