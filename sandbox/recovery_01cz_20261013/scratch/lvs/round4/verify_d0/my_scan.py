import importlib.util, json, sys, re
from pathlib import Path
check_path, statement_root, deliverable = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
spec = importlib.util.spec_from_file_location('packet_check', check_path)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
registered = mod.load_register(statement_root)
if isinstance(registered, tuple): registered = registered[0]
print('registered Props:', len(registered), 'DerivativeBoundAtZero registered:', 'UniversalLaw.Spec.Side24.DerivativeBoundAtZero' in registered,
      'DerivativePairingBound registered:', 'UniversalLaw.Spec.Side24.DerivativePairingBound' in registered)
text = deliverable.read_text()
info = mod.scan_module(text, 'Side24Analysis', registered)
print(json.dumps({k: v for k, v in info.items() if k != 'theorems'}, indent=1))
for t in info['theorems']:
    print('THEOREM', t['name'], 'line', t['line'], 'proves', t['proves'], 'binders', t['binders'], 'args', t['arguments'])
    for k, v in t['doc'].items(): print('  ', k, '=>', v[:200].replace('\n',' '))
print('SCAN_MODULE OK')
# independent lexical scan on comment-stripped code
code, docs = mod.lex(text)
print('refused_tokens:', mod.refused_tokens(code))
print('hash commands:', [ (code.count('\n',0,m.start())+1) for m in mod.HASH_COMMAND.finditer(code)])
print('set_option:', [ (code.count('\n',0,m.start())+1, m.group(0)) for m in mod.SET_OPTION.finditer(code)])
for pat, label in [(r'fun\s*⟨', 'fun ⟨'), (r'fun\s*\|', 'fun |'), (r'(?<![A-Za-z0-9_.])match(?![A-Za-z0-9_])', 'match'),
                   (r'(?<![A-Za-z0-9_.])nomatch', 'nomatch'), (r'(?<![A-Za-z0-9_.])(decide|rfl)(?![A-Za-z0-9_])', 'decide/rfl'),
                   (r'(?<![A-Za-z0-9_.])(instance|def|abbrev|structure|inductive|example|lemma)(?![A-Za-z0-9_])', 'non-theorem decl kw'),
                   (r'(?<![A-Za-z0-9_.])(private|protected)(?![A-Za-z0-9_])', 'private/protected'),
                   (r'(?<![A-Za-z0-9_.])(section|variable|universe|attribute|deriving|omit)(?![A-Za-z0-9_])', 'other scope kw'),
                   (r'(?<![A-Za-z0-9_.])(open)(?![A-Za-z0-9_])', 'open'), (r'(?<![A-Za-z0-9_.])in(?![A-Za-z0-9_])', ' in token'),
                   (r'[@]\[', 'attribute'), (r'(?<![A-Za-z0-9_.])(convert|simpa?|nlinarith|positivity|omega|norm_num|ring|interval_cases)(?![A-Za-z0-9_])', 'tactics (info)')]:
    hits = [(code.count('\n',0,m.start())+1, m.group(0)) for m in re.finditer(pat, code)]
    print(f'{label}: {len(hits)} {hits[:12]}')
# column-0 lines (code): list them
print('column-0 code lines:')
for i, ln in enumerate(code.split('\n')):
    if ln and not ln[0].isspace(): print('  ', i+1, ln[:100])
# docstrings count and labels
print('docstring blocks:', len(docs))
