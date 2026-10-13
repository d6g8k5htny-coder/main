"""For every changed string leaf, show a token-level diff (old -> new) using difflib."""
import json, sys, difflib, re
out = json.load(open(sys.argv[1]))
def toks(s): return re.findall(r'\S+|\s+', s)
for path, o, n in out['changed']:
    print('=' * 100); print(path, 'old len', len(o), 'new len', len(n))
    a, b = toks(o), toks(n)
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal': continue
        print(f'--- {tag} @tok {i1}')
        ctx_before = ''.join(a[max(0,i1-12):i1])
        print('  CONTEXT: ...' + ctx_before[-200:])
        print('  OLD: ' + ''.join(a[i1:i2]))
        print('  NEW: ' + ''.join(b[j1:j2]))
