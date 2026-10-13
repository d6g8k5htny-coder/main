import sys, re
from pathlib import Path
PKT = Path(sys.argv[1]); FILE = Path(sys.argv[2])
sys.path.insert(0, str(PKT))
import check
text = FILE.read_text(encoding='utf-8')
code, docs = check.lex(text)
print("REFUSED_TOKENS:", check.refused_tokens(code))
# extra tokens from the task list not in check.REFUSED_TOKENS
words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_']*", code))
extra = {'sorry','admit','axiom','native_decide','unsafe','opaque','partial','implemented_by','extern','macro','syntax','elab','notation','initialize','export','alias','include','_root_','match','decide'}
print("EXTRA_TOKEN_HITS:", sorted(extra & words))
print("HASH_COMMANDS:", [(code.count('\n',0,m.start())+1, code[m.start():m.start()+20]) for m in re.finditer(r'#[A-Za-z_]', code)])
print("SET_OPTION:", [(code.count('\n',0,m.start())+1, m.group()) for m in re.finditer(r'set_option[^\n]*', code)])
print("FUN_ANON_CTOR:", [(code.count('\n',0,m.start())+1, code[m.start():m.start()+30]) for m in re.finditer(r'fun\s*⟨', code)])
print("FUN_BAR:", [(code.count('\n',0,m.start())+1,) for m in re.finditer(r'fun\s*\|', code)])
print("MATCH:", [(code.count('\n',0,m.start())+1,) for m in re.finditer(r'(?<![A-Za-z0-9_.])match(?![A-Za-z0-9_])', code)])
print("OBTAIN/RCASES (tactic, fine):", len(re.findall(r'\bobtain\b|\brcases\b|\brintro\b', code)))
# column-0 lines
col0 = [(i+1, ln[:60]) for i, ln in enumerate(code.split('\n')) if ln and not ln[0].isspace()]
print("COLUMN0_LINES:")
for l in col0: print("  ", l)
# any declaration keyword not at column 0
dz = check.depth_zero(code)
for m in check.COMMAND_KEYWORD.finditer(dz):
    head = code[code.rfind('\n',0,m.start())+1:m.start()]
    if not check.legal_head(head):
        print("OFFCOL0:", code.count('\n',0,m.start())+1, m.group(1), repr(head))
# docstring check for inventory theorem
for a,b,raw in docs:
    if 'Proves:' in raw:
        print("DOCSTRING lines", a+1, "-", b+1)
        body = raw[3:-2]
        for label in check.LABELS:
            hits = re.findall(r'^[ \t]*' + re.escape(label) + r'(.*)$', body, re.M)
            print("  ", label, len(hits), repr(hits[0][:80]) if hits else None)
        print("  names_prop:", check.names_prop(re.findall(r'^[ \t]*Proves:(.*)$', body, re.M)[0], 'UniversalLaw.Spec.Side24.DensityComparison'))
# imports
print("IMPORTS:", re.findall(r'^import\s+(\S+)', code, re.M))
# namespace / end balance
print("NS:", re.findall(r'^(namespace|end)\s+(\S+)', code, re.M))
# count theorems
print("THEOREM_DECLS:", re.findall(r'^theorem\s+(\S+)', code, re.M))
print("DEF_DECLS:", re.findall(r'^(?:noncomputable\s+)?(?:def|abbrev|instance|structure|inductive)\s+(\S+)', code, re.M))
# non-ascii-identifiers potentially hiding tokens
print("TRAILING_WS_LINES:", [i+1 for i,ln in enumerate(text.split('\n')) if ln != ln.rstrip()])
print("TABS:", [i+1 for i,ln in enumerate(text.split('\n')) if '\t' in ln])
print("ENDS_WITH_NEWLINE:", text.endswith('\n'))
