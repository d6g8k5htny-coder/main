"""NON-CERTIFYING text tool. Show old/new tail differences for changed leaves via difflib on tokens."""
import json, difflib, sys, re
D='<SCRATCH>/lvs/register_update14/verify_reg/'
ch=json.load(open(D+'changed_leaves.json'))
sel = sys.argv[1:] 
for i,c in enumerate(ch):
    if sel and str(i) not in sel: continue
    print('='*100); print(i, c['path'])
    o=re.split(r'(\s+)',c['old']); n=re.split(r'(\s+)',c['new'])
    sm=difflib.SequenceMatcher(None,o,n,autojunk=False)
    for tag,i1,i2,j1,j2 in sm.get_opcodes():
        if tag=='equal': continue
        ctxb=''.join(o[max(0,i1-12):i1]); ctxa=''.join(o[i2:i2+12])
        print(f'--- {tag} ...{ctxb}')
        print('   OLD: '+''.join(o[i1:i2]))
        print('   NEW: '+''.join(n[j1:j2]))
        print('   ...'+ctxa)
