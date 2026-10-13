import re, sys
from pathlib import Path
src = Path(sys.argv[1]).read_text(encoding='utf-8')
# strip nested block comments (incl. docstrings) and line comments, keeping newlines
out = []; i = 0; depth = 0; n = len(src)
while i < n:
    if depth == 0 and src.startswith('--', i):
        j = src.find('\n', i); i = n if j < 0 else j; continue
    if src.startswith('/-', i):
        depth += 1; i += 2; continue
    if depth > 0 and src.startswith('-/', i):
        depth -= 1; i += 2; continue
    if depth > 0:
        out.append('\n' if src[i] == '\n' else ' '); i += 1; continue
    out.append(src[i]); i += 1
code = ''.join(out)
assert depth == 0, 'unbalanced comment'
refused = ['sorry','sorryAx','admit','axiom','native_decide','native','ofReduceBool','trustCompiler','unsafe','opaque','partial',
 'implemented_by','extern','skipKernelTC','byAsSorry','macro','macro_rules','syntax','elab','elab_rules','notation','infix','infixl',
 'infixr','prefix','postfix','run_cmd','run_tac','run_elab','initialize','builtin_initialize','declare_syntax_cat','export','alias',
 'irreducible_def','include','_root_','instance','attribute','local','scoped','set_option','decide','rfl.sorry','Classical.choice']
words = set(re.findall(r"[^\W\d][\w'!?]*(?:\.[^\W\d][\w'!?]*)*", code))
hits = sorted(w for w in words if w in refused)
print('REFUSED_OR_WATCH_TOKENS_IN_CODE', hits)
print('HASH_COMMANDS', re.findall(r'^.*#[A-Za-z_].*$', code, re.M))
print('FUN_ANON_CONSTRUCTOR_PATTERNS', re.findall(r'fun\s*(?:⟨|\||\().*?=>', code))
print('MATCH_USES', re.findall(r'^.*\bmatch\b.*$', code, re.M))
print('SET_OPTION', re.findall(r'^.*set_option.*$', code, re.M))
print('OPEN_LINES', re.findall(r'^.*\bopen\b.*$', code, re.M))
# declaration keywords anywhere not at column 0
for m in re.finditer(r"(?<![A-Za-z0-9_.'])(theorem|lemma|def|abbrev|instance|structure|inductive|example|namespace|section|end|variable|universe)(?![A-Za-z0-9_'!?])", code):
    ls = code.rfind('\n', 0, m.start()) + 1
    if m.start() != ls:
        print('KEYWORD_OFF_COLUMN0 line', code.count('\n', 0, m.start()) + 1, repr(code[ls:m.end()]))
print('COLUMN0_LINES:')
for k, line in enumerate(code.split('\n'), 1):
    if line and not line[0].isspace():
        print('  ', k, line[:110])
# docstring label lines of the inventory theorem
raw = src.split('\n')
for k, line in enumerate(raw, 1):
    if re.match(r'^[ \t]*(Proves:|Method:|Does not establish:)', line) or line.startswith('/-- Proves:'):
        print('LABEL_LINE', k, line[:100])
print('CLAIM_ID_IN_PROVES', 'side24-coefficient' in [l for l in raw if 'Proves:' in l][0])
