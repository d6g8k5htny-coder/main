import json,difflib,sys
d=json.load(open('<SCRATCH>/verif_r5d/diff.json'))
sel=sys.argv[1:] 
for k,(a,b) in d['changed'].items():
    if sel and not any(s==k for s in sel): continue
    print('=====',k)
    aw=a.split(' ');bw=b.split(' ')
    sm=difflib.SequenceMatcher(None,aw,bw,autojunk=False)
    for op,i1,i2,j1,j2 in sm.get_opcodes():
        if op=='equal': continue
        ctxa=' '.join(aw[max(0,i1-8):i1])
        print(f'  [{op}] ...{ctxa}')
        print('    -',' '.join(aw[i1:i2]))
        print('    +',' '.join(bw[j1:j2]))
