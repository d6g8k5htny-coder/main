import re, sys, unicodedata
path = sys.argv[1]
src = open(path, encoding='utf-8').read()

# --- strip comments: nested block comments /- ... -/ (incl. docstrings), and -- line comments
def strip_comments(s):
    out = []; i = 0; n = len(s); depth = 0
    while i < n:
        if s.startswith('/-', i):
            depth += 1; i += 2; continue
        if depth > 0:
            if s.startswith('-/', i):
                depth -= 1; i += 2; continue
            if s[i] == '\n': out.append('\n')
            i += 1; continue
        if s.startswith('--', i):
            while i < n and s[i] != '\n': i += 1
            continue
        out.append(s[i]); i += 1
    return ''.join(out)
code = strip_comments(src)
assert code.count('\n') == src.count('\n')

refused = ['sorry','sorryAx','admit','axiom','native_decide','native','ofReduceBool','trustCompiler',
  'unsafe','opaque','partial','implemented_by','extern','skipKernelTC','byAsSorry','macro','macro_rules',
  'syntax','elab','elab_rules','notation','infix','infixl','infixr','prefix','postfix','run_cmd','run_tac',
  'run_elab','initialize','declare_syntax_cat','export','alias','irreducible_def','include','_root_',
  'set_option','instance','attribute','decide','local','scoped','noncomputable','def','abbrev','lemma','example','match','variable','section','mutual','deriving']
print("== refused / watched tokens on comment-stripped code (word-boundary) ==")
hits = 0
for tok in refused:
    for m in re.finditer(r'(?<![A-Za-z0-9_.!?\'])' + re.escape(tok) + r'(?![A-Za-z0-9_!?\'])', code):
        ln = code.count('\n', 0, m.start()) + 1
        line = code.split('\n')[ln-1]
        print(f"  {tok!r} at line {ln}: {line.strip()}")
        hits += 1
print(f"  total hits: {hits}")

print("== '#' anywhere in code ==")
for i, line in enumerate(code.split('\n'), 1):
    if '#' in line: print(f"  line {i}: {line.strip()}")

print("== column-0 check: declaration/scope keywords not at column 0 ==")
kw = r'(theorem|lemma|def|abbrev|instance|structure|inductive|class|namespace|end|open|section|variable|universe|noncomputable|attribute|example|set_option|import|@\[)'
for i, line in enumerate(code.split('\n'), 1):
    if re.match(r'^\s+' + kw + r'\b', line):
        print(f"  line {i}: {line}")
print("== declarations at column 0 ==")
decls = []
for i, line in enumerate(code.split('\n'), 1):
    m = re.match(r'^(theorem|lemma|def|abbrev|instance|structure|inductive|class|noncomputable def|example)\s+(\S+)', line)
    if m: decls.append((i, m.group(1), m.group(2)))
for d in decls: print("  ", d)
print(f"  count: {len(decls)}")
print("== namespace/end/open/import lines ==")
for i, line in enumerate(code.split('\n'), 1):
    if re.match(r'^(namespace|end|open|import|section)\b', line): print(f"  line {i}: {line}")

print("== term-mode destructuring lambdas / match ==")
for i, line in enumerate(code.split('\n'), 1):
    if re.search(r'fun\s*(⟨|\|)', line) or re.search(r'\bmatch\b', line) or re.search(r'fun\s*\(\s*⟨', line):
        print(f"  line {i}: {line.strip()}")
print("== 'fun' occurrences (for manual review) ==")
for i, line in enumerate(code.split('\n'), 1):
    if re.search(r'\bfun\b', line): print(f"  line {i}: {line.strip()}")

print("== non-ASCII characters used (whole file incl. comments) ==")
chars = sorted(set(c for c in src if ord(c) > 127))
for c in chars:
    try: nm = unicodedata.name(c)
    except ValueError: nm = '?'
    print(f"  U+{ord(c):04X} {c!r} {nm}  count={src.count(c)}")
print("== non-ASCII characters in CODE (comments stripped) ==")
chars = sorted(set(c for c in code if ord(c) > 127))
for c in chars:
    try: nm = unicodedata.name(c)
    except ValueError: nm = '?'
    print(f"  U+{ord(c):04X} {c!r} {nm}  count={code.count(c)}")
print("== control / zero-width / bidi chars ==")
bad = [(i, c) for i, c in enumerate(src) if unicodedata.category(c) in ('Cf','Cc','Co','Cn') and c not in '\n\t']
print("  ", bad[:20], "count", len(bad))

print("== docstring before inventory theorem ==")
lines = src.split('\n')
for i, line in enumerate(lines, 1):
    if line.startswith('theorem derivativeBoundAtLeastOne'):
        j = i - 2
        while j >= 0 and not lines[j].startswith('/--'): j -= 1
        block = lines[j:i-1]
        print(f"  docstring lines {j+1}..{i-1}")
        for k, l in enumerate(block, start=j+1):
            if re.match(r'^(/-- )?(Proves:|Method:|Does not establish:)', l): print(f"  label line {k}: {l[:100]}")
        txt = '\n'.join(block)
        print("  whole-token Prop in Proves line:", bool(re.search(r'Proves: `UniversalLaw\.Spec\.Side24\.DerivativeBoundAtLeastOne`', txt)))
        print("  claim id side24-coefficient cited:", 'side24-coefficient' in txt)
        print("  inventory header line", i, ":", repr(line))
        print("  next line:", repr(lines[i]))
