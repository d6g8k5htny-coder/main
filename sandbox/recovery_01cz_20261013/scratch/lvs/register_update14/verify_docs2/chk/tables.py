import re, subprocess
R = '<SCRATCH>/repos/math_r5'
DOCS = ['specifications/lean_statements_20261010/README.md', 'specifications/lean_statements_20261010/SCOPE.md',
        'specifications/lean_statement_proofs_20261011/README.md', 'specifications/lean_statement_proofs_20261011/SCOPE.md']
def check(text):
    lines = text.split('\n'); res = []; i = 0
    while i < len(lines):
        if lines[i].startswith('|'):
            start = i; tbl = []
            while i < len(lines) and lines[i].startswith('|'):
                tbl.append(lines[i]); i += 1
            n = len(re.findall(r'(?<!\\)\|', tbl[0]))
            sep_ok = len(tbl) > 1 and re.fullmatch(r'\|(\s*:?-+:?\s*\|)+', tbl[1]) is not None and len(re.findall(r'\|', tbl[1])) == n
            bad = [start + j + 1 for j, l in enumerate(tbl) if len(re.findall(r'(?<!\\)\|', l)) != n or not l.endswith('|')]
            trailing = [start + j + 1 for j, l in enumerate(tbl) if l != l.rstrip()]
            res.append((start + 1, len(tbl), n, sep_ok, bad, trailing))
        else:
            i += 1
    return res
for d in DOCS:
    new = open(f'{R}/{d}', encoding='utf-8').read()
    old = subprocess.run(['git', '-C', R, 'show', f'9e305ea5:{d}'], capture_output=True).stdout.decode()
    rn, ro = check(new), check(old)
    print(d, 'tables', len(rn), 'old', len(ro), 'rows', [t[1] for t in rn] == [t[1] for t in ro],
          'all sep ok', all(t[3] for t in rn), 'bad rows', [t[4] for t in rn if t[4]], 'trailing ws', [t[5] for t in rn if t[5]],
          'ends newline', new.endswith('\n'), 'nonascii new chars', sorted(set(c for c in new if ord(c) > 127) - set(c for c in old if ord(c) > 127)),
          'trailing ws lines', sum(1 for l in new.split('\n') if l != l.rstrip()), 'old', sum(1 for l in old.split('\n') if l != l.rstrip()))
