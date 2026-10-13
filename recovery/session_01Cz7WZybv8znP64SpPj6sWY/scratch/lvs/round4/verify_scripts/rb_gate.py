import sys, json, re
from pathlib import Path
PKT = Path(sys.argv[1]); STMT = Path(sys.argv[2]); FILE = Path(sys.argv[3])
sys.path.insert(0, str(PKT))
import check
reg = check.load_register(STMT)
registered = reg[0] if isinstance(reg, tuple) else reg
if isinstance(registered, dict) and 'registered' in registered:
    registered = registered['registered']
print('REGISTER_TYPE', type(reg).__name__, 'n_registered', len(registered))
print('PROP_REGISTERED', 'UniversalLaw.Spec.Side24.ReferenceOddAndPinBlocks' in registered)
text = FILE.read_text(encoding='utf-8')
try:
    info = check.scan_module(text, 'Side24Analysis', registered)
    print('IMPORTS', info['imports'])
    print('NAMESPACES', info['namespaces'])
    print('AUX_THEOREMS', info['aux_theorems'], 'AUX_DEFS', info['auxiliary_definitions'], 'SET_OPTIONS', info['set_options'])
    for t in info['theorems']:
        print('THEOREM', json.dumps(t, ensure_ascii=False))
    print('SCAN_OK')
except Exception as e:
    print('SCAN_REFUSED', repr(e))
# independent lexical checks on comment-stripped code
code, docs = check.lex(text)
words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", code))
print('REFUSED_TOKENS_FOUND', sorted(set(check.REFUSED_TOKENS) & words))
print('HASH_CMDS', [ (code.count('\n',0,m.start())+1, code[m.start():m.start()+20]) for m in re.finditer(r'#[A-Za-z_]', code)])
print('SET_OPTION_LINES', [i+1 for i,l in enumerate(code.split('\n')) if 'set_option' in l])
print('FUN_ANGLE_LAMBDA', [i+1 for i,l in enumerate(code.split('\n')) if re.search(r'fun\s*⟨', l) or re.search(r'fun\s*\|', l)])
print('MATCH_TOKENS', [i+1 for i,l in enumerate(code.split('\n')) if re.search(r'(?<![A-Za-z0-9_.])match(?![A-Za-z0-9_])', l)])
print('OPEN_IN_LINES', [i+1 for i,l in enumerate(code.split('\n')) if re.search(r'^\s*open\b.*\bin\b', l)])
# column-0 lines and their first token
col0 = [(i+1, l.split()[0]) for i,l in enumerate(code.split('\n')) if l and not l[0].isspace()]
from collections import Counter
print('COL0_TOKENS', Counter(t for _,t in col0))
print('COL0_NOT_ALLOWED', [(n,t) for n,t in col0 if t not in check.ALLOWED_COMMANDS and not t.startswith('|')])
# keyword off column 0 (check.py's own rule)
dz = check.depth_zero(code)
offs = []
for m in check.COMMAND_KEYWORD.finditer(dz):
    head = code[code.rfind('\n',0,m.start())+1:m.start()]
    if not check.legal_head(head):
        offs.append((code.count('\n',0,m.start())+1, m.group(1)))
print('KEYWORDS_OFF_COL0', offs)
# non-ASCII inventory
nonascii = Counter(ch for ch in text if ord(ch) > 127)
print('NON_ASCII', {ch: n for ch, n in sorted(nonascii.items())})
# docstring labels of the inventory theorem
for a,b,raw in docs:
    if 'Proves:' in raw:
        body = raw[3:-2]
        for label in check.LABELS:
            hits = re.findall(r'^[ \t]*' + re.escape(label) + r'(.*)$', body, re.M)
            print('LABEL', label, len(hits))
        print('NAMES_PROP', check.names_prop(re.findall(r'^[ \t]*Proves:(.*)$', body, re.M)[0], 'UniversalLaw.Spec.Side24.ReferenceOddAndPinBlocks'))
