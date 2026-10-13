import sys, re, collections
from pathlib import Path
PKT = Path(sys.argv[1]); FILE = Path(sys.argv[2])
sys.path.insert(0, str(PKT))
import check
text = FILE.read_text(encoding='utf-8')
code, docs = check.lex(text)
print('REFUSED_TOKENS_FOUND', check.refused_tokens(code))
print('HASH_CMDS', [(i+1, l) for i, l in enumerate(code.split('\n')) if check.HASH_COMMAND.search(l)])
print('SET_OPTION_LINES', [(i+1, l) for i, l in enumerate(code.split('\n')) if 'set_option' in l])
print('MAXHEARTBEATS', [(i+1, l) for i, l in enumerate(code.split('\n')) if 'maxHeartbeats' in l])
print('FUN_ANGLE_LAMBDA', [(i+1, l.strip()) for i, l in enumerate(code.split('\n')) if re.search(r'fun\s*(⟨|\|)', l)])
print('MATCH_TOKENS', [(i+1, l.strip()) for i, l in enumerate(code.split('\n')) if re.search(r"(?<![A-Za-z0-9_.'])match(?![A-Za-z0-9_'])", l)])
print('NOMATCH_TOKENS', [(i+1, l.strip()) for i, l in enumerate(code.split('\n')) if re.search(r"(?<![A-Za-z0-9_.'])nomatch(?![A-Za-z0-9_'])", l)])
print('OPEN_IN_LINES', [(i+1, l.strip()) for i, l in enumerate(code.split('\n')) if re.match(r'^\s*open\b', l) and check.IN_TOKEN.search(l)])
print('DECIDE_TOKENS', [(i+1, l.strip()) for i, l in enumerate(code.split('\n')) if re.search(r"(?<![A-Za-z0-9_.'])decide(?![A-Za-z0-9_'])", l)])
col0 = collections.Counter()
bad_col0 = []
for i, l in enumerate(code.split('\n')):
    if l and not l[0].isspace():
        tok = l.split()[0] if l.split() else ''
        tok = tok.split('@[')[0] if tok.startswith('@[') else tok
        col0[tok] += 1
        if tok not in check.ALLOWED_COMMANDS and not tok.startswith('@['):
            bad_col0.append((i+1, l))
print('COL0_TOKENS', col0)
print('COL0_NOT_ALLOWED', bad_col0)
off = []
for m in check.COMMAND_KEYWORD.finditer(code):
    ls = code.rfind('\n', 0, m.start()) + 1
    head = code[ls:m.start()]
    if not check.legal_head(head):
        off.append((code.count('\n', 0, m.start())+1, code[ls:code.find('\n', m.start())].strip()))
print('KEYWORDS_OFF_COL0', off)
# namespace lines
print('NAMESPACE_LINES', [(i+1, l) for i, l in enumerate(text.split('\n')) if re.match(r'^(namespace|end)\b', l)])
print('IMPORT_LINES', [(i+1, l) for i, l in enumerate(text.split('\n')) if l.startswith('import')])
# docstring labels of the inventory theorem
lines = text.split('\n')
for lbl in check.LABELS:
    hits = [i+1 for i, l in enumerate(lines) if l.startswith(lbl) or l.startswith('/-- ' + lbl)]
    print('LABEL', lbl, hits)
# Proves line whole-token
pl = [l for l in lines if l.startswith('/-- Proves:')]
print('PROVES_LINE', pl)
print('PROVES_WHOLE_TOKEN', bool(pl) and re.search(r"(?<![A-Za-z0-9_.'])UniversalLaw\.Spec\.Side24\.ReferencePinDensityProduct(?![A-Za-z0-9_'.])", pl[0]) is not None)
# theorems declared and whether non-Aux
print('THEOREM_DECLS', [(i+1, l) for i, l in enumerate(text.split('\n')) if re.match(r'^theorem\b', l)])
# hypothesis form helper header
hdr = [(i+1, l) for i, l in enumerate(lines) if 'rp_referencePinDensityProduct_of_referenceOddAndPinBlocks' in l and l.startswith('theorem')]
print('HYP_FORM_HEADER', hdr, lines[hdr[0][0]] if hdr else None, lines[hdr[0][0]+1] if hdr else None)
