import re, sys
for f in sys.argv[1:]:
    lines = open(f, encoding='utf-8').read().split('\n')
    tables = []; cur = None
    in_code = False
    for i, l in enumerate(lines, 1):
        if l.startswith('```'):
            in_code = not in_code
        if in_code:
            continue
        if l.startswith('|'):
            if cur is None:
                cur = []; tables.append(cur)
            cur.append((i, l))
        else:
            cur = None
    bad = []
    nrows = 0
    for t in tables:
        hdr = len(re.findall(r'(?<!\\)\|', t[0][1]))
        assert re.fullmatch(r'\|(\s*:?-+:?\s*\|)+', t[1][1]), (f, t[1])
        for i, l in t:
            nrows += 1
            n = len(re.findall(r'(?<!\\)\|', l))
            if n != hdr or not l.rstrip().endswith('|') or l != l.rstrip():
                bad.append((i, n, hdr))
            # backticks balanced
            if l.count('`') % 2:
                bad.append((i, 'odd backticks'))
    print(f.split('specifications/')[-1], 'tables', len(tables), 'rows', nrows, 'bad', bad[:10])
