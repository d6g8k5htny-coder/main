import json, difflib, re, sys
d = json.load(open('diff.json'))
for kind, path, o, n in d:
    if kind != 'CHG': continue
    print('=' * 100)
    print('/'.join(map(str, path)))
    ow = re.findall(r'\S+|\s+', o); nw = re.findall(r'\S+|\s+', n)
    sm = difflib.SequenceMatcher(None, ow, nw, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal': continue
        ctx_b = ''.join(ow[max(0,i1-12):i1])
        ctx_a = ''.join(ow[i2:i2+12])
        print(f'--- {tag}')
        print('  CTX< ...' + ctx_b)
        print('  OLD: ' + ''.join(ow[i1:i2]))
        print('  NEW: ' + ''.join(nw[j1:j2]))
        print('  CTX> ' + ctx_a + '...')
