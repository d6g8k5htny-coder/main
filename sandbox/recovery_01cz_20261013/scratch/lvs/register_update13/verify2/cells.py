import json, re, sys
root = sys.argv[1]
reg = json.load(open(root + '/specifications/lean_statements_20261010/REGISTER.json'))
scope = open(root + '/specifications/lean_statements_20261010/SCOPE.md', encoding='utf-8').read().split('\n')
def split_row(line):
    # split on unescaped pipes
    cells = re.split(r'(?<!\\)\|', line)
    return [c.strip() for c in cells[1:-1]]
for cid in ['side24-coefficient', 'math.side24-coefficient']:
    c = [x for x in reg['claims'] if x['claim_id'] == cid][0]
    line = [l for l in scope if l.startswith('| `%s` |' % cid)]
    assert len(line) == 1, (cid, len(line))
    cells = split_row(line[0])
    print(cid, 'ncells', len(cells))
    ne = c['not_established'].replace('|', '\\|')
    ns = c['next_step'].replace('|', '\\|')
    print('  not_established equal:', cells[5] == ne)
    print('  next_step equal:', cells[6] == ns)
    if cells[5] != ne:
        import difflib
        for d in difflib.ndiff([cells[5]], [ne]): print(d[:3000])
    if cid == 'side24-coefficient':
        i = cells[5].find('the entrywise bound')
        print('  EC clause:', cells[5][i:i+400])
        print('  "for d = 2 or 3" count in NE:', cells[5].count('for d = 2 or 3'))
