import importlib.util, sys, json
from pathlib import Path
S = Path('<SCRATCH>')
pkt = S/'repos/math/specifications/lean_statement_proofs_20261011'
spec = importlib.util.spec_from_file_location('check', pkt/'check.py')
check = importlib.util.module_from_spec(spec); spec.loader.exec_module(check)
registered, structures, regsha = check.load_register(S/'repos/math/specifications/lean_statements_20261010')
name = 'UniversalLaw.Spec.Side24.DerivativePairingBound'
print('REGISTERED', name, registered.get(name))
print('REGISTER sha', regsha)
text = (S/'lvs/round4/work/DerivativePairingBound.lean').read_text(encoding='utf-8')
res = check.scan_module(text, 'Side24Analysis', registered)
print(json.dumps(res, indent=1, ensure_ascii=False))
code, docs = check.lex(text)
print('REFUSED_TOKENS_IN_CODE', check.refused_tokens(code))
print('HASH_IN_CODE', check.HASH_COMMAND.search(code))
print('SCAN OK')
