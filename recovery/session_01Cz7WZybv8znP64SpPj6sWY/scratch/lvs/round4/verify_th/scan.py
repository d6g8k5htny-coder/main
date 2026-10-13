import importlib.util, json, sys
from pathlib import Path
S = Path('<SCRATCH>')
P = S / 'repos/math/specifications/lean_statement_proofs_20261011'
spec = importlib.util.spec_from_file_location('check', P / 'check.py')
check = importlib.util.module_from_spec(spec); spec.loader.exec_module(check)
statement_root = S / 'repos/math/specifications/lean_statements_20261010'
registered, structures, reg_sha = check.load_register(statement_root)
name = 'UniversalLaw.Spec.Side24.ThetaSummable'
print('REGISTERED?', name in registered, registered.get(name))
print('REGISTER.json sha256', reg_sha)
text = (S / 'lvs/round4/work/ThetaSummable.lean').read_text(encoding='utf-8')
code, docs = check.lex(text)
print('refused_tokens:', check.refused_tokens(code))
print('hash command in code:', check.HASH_COMMAND.search(code))
try:
    res = check.scan_module(text, 'Side24Analysis', registered)
    print('scan_module OK')
    print(json.dumps({k: v for k, v in res.items() if k != 'theorems'}, indent=1, ensure_ascii=False))
    for t in res['theorems']:
        print(json.dumps(t, indent=1, ensure_ascii=False))
except Exception as e:
    print('scan_module FAILED:', type(e).__name__, e)
# negative controls on the scanner itself: wrong stem must fail; an extra hypothesis must fail
for stem in ('Side24Elementary', 'ThetaSummable'):
    try:
        check.scan_module(text, stem, registered); print('CONTROL stem', stem, 'UNEXPECTEDLY PASSED')
    except Exception as e:
        print('control stem', stem, '-> refused as expected:', str(e)[:120])
mut = text.replace('theorem thetaSummable (d : ℕ) : UniversalLaw.Spec.Side24.ThetaSummable d := by',
                   'theorem thetaSummable (d : ℕ) (hd : 2 ≤ d) : UniversalLaw.Spec.Side24.ThetaSummable d := by')
assert mut != text
try:
    check.scan_module(mut, 'Side24Analysis', registered); print('CONTROL extra-hyp UNEXPECTEDLY PASSED')
except Exception as e:
    print('control extra hypothesis -> refused as expected:', str(e)[:160])
mut2 = text.replace('/-- Proves: `UniversalLaw.Spec.Side24.ThetaSummable`', '/-- Proves: `UniversalLaw.Spec.Side24.ThetaSummableX`')
try:
    check.scan_module(mut2, 'Side24Analysis', registered); print('CONTROL proves-token UNEXPECTEDLY PASSED')
except Exception as e:
    print('control Proves token -> refused as expected:', str(e)[:160])
mut3 = text.replace('  simp [latticeVec]', '  sorry')
try:
    check.scan_module(mut3, 'Side24Analysis', registered); print('CONTROL sorry UNEXPECTEDLY PASSED')
except Exception as e:
    print('control sorry -> refused as expected:', str(e)[:120])
