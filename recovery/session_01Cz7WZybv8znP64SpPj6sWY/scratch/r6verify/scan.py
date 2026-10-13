import importlib.util, sys, json, re
from pathlib import Path
W = Path(sys.argv[1])
P = W / 'specifications/lean_statement_proofs_20261011'
spec = importlib.util.spec_from_file_location('chk', P / 'check.py')
chk = importlib.util.module_from_spec(spec); spec.loader.exec_module(chk)
reg, _structs, _sha = chk.load_register(W / "specifications/lean_statements_20261010")
text = (P / 'SpecificationProofs/Side24Images.lean').read_text()
info = chk.scan_module(text, 'Side24Images', reg)
for t in info['theorems']:
    print(t['name'], '->', t['proves'], t['arguments'], [b['kind'] for b in t['binders']])
print('aux_theorems', info['aux_theorems'], 'auxiliary_defs', info['auxiliary_definitions'], 'set_options', info['set_options'])
print('imports', info['imports']); print('namespaces', info['namespaces'])
code, docs = chk.lex(text)
words = set(re.findall(r'[A-Za-z_][A-Za-z0-9_]*', code))
print('refused hits', sorted(set(chk.REFUSED_TOKENS) & words))
# tokens near-miss: identifiers containing refused tokens as substrings
near = sorted(w for w in words if any(r in w for r in chk.REFUSED_TOKENS) and w not in chk.REFUSED_TOKENS)
print('near-miss identifiers', near)
# keyword-after-nonascii check
for m in re.finditer(r"([^\x00-\x7f])(theorem|lemma|def|abbrev|instance|structure|inductive|example|private|protected|noncomputable|namespace|section|end|variable|universe)(?![A-Za-z0-9_'!?])", code):
    print('unicode-adjacent keyword', code.count('\n',0,m.start())+1, m.group(0))
# Unicode letters glued to a keyword with no ascii lookbehind protection
for m in chk.COMMAND_KEYWORD.finditer(chk.depth_zero(code)):
    ln = code.count('\n',0,m.start())+1
    line = code.split('\n')[ln-1]
    if line[:1].isspace():
        print('indented keyword', ln, line.strip()[:60])
for t in info['theorems']:
    print(t['name'], 'claims', reg[t['proves']]['claim_ids'], 'module', reg[t['proves']]['module'])
