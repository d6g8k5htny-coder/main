import importlib.util, sys, json, re
from pathlib import Path
W = Path(sys.argv[1]); P = W / 'specifications/lean_statement_proofs_20261011'
spec = importlib.util.spec_from_file_location('chk', P / 'check.py'); chk = importlib.util.module_from_spec(spec); spec.loader.exec_module(chk)
reg, _, _ = chk.load_register(W / 'specifications/lean_statements_20261010')
text = (P / 'SpecificationProofs/Side24Images.lean').read_text()
info = chk.scan_module(text, 'Side24Images', reg)
code, docs = chk.lex(text)
R = json.load(open(W / 'specifications/lean_statements_20261010/REGISTER.json'))
entries = {}
for row in R['claims']:
    for st in row['statements']:
        if 'Side24Images' in st.get('declaration',''):
            entries[st['declaration']] = st
for t in info['theorems']:
    # find raw docstring
    raw = [d for (s,e,d) in docs if e == t['line']-2][0]
    body = raw[3:-2]
    m = re.search(r'^Does not establish:(.*?)\Z', body, re.M|re.S)
    dne = ' '.join(m.group(1).split())
    e = entries.get(t['name'])
    print(t['name'], 'entry present', e is not None)
    if e:
        print(' keys', sorted(e.keys()))
        print(' dnc == docstring DNE:', e.get('does_not_claim') == dne)
        if e.get('does_not_claim') != dne:
            a, b = e.get('does_not_claim'), dne
            i = next((k for k in range(min(len(a),len(b))) if a[k]!=b[k]), min(len(a),len(b)))
            print('  diff at', i, repr(a[i-40:i+80]), '|||', repr(b[i-40:i+80]))
        print(' anchor', e.get('anchor'), '| formal_progress', e.get('formal_progress'), '| kind', e.get('kind'))
