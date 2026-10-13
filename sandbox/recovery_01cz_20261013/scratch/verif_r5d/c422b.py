import json,re,difflib
d=json.load(open('<SCRATCH>/verif_r5d/diff.json'))
for k,(a,b) in d['changed'].items():
    aw=a.split(' ');bw=b.split(' ')
    sm=difflib.SequenceMatcher(None,aw,bw,autojunk=False)
    for op,i1,i2,j1,j2 in sm.get_opcodes():
        if op in('insert','replace'):
            ins=' '.join(bw[j1:j2])
            for m in re.finditer(r'(?<!proofs-)\b(38\d{9})\b',ins):
                print(k,'| run',m.group(1),'| head be014794' if 'PR head be014794' in ins else '| NO HEAD','| merge' if '79d3964cf70e2eadad5c9629eef8f4b47f218dd0' in ins else '| NO MERGE')
            for kw in ('kernel-checked','receipted','kernel-check','verified'):
                if kw in ins: print(k,'| KEYWORD',kw,':',ins[max(0,ins.index(kw)-150):ins.index(kw)+80])
for k,v in d['added'].items():
    for kw in ('kernel-checked','receipted','verified'):
        if isinstance(v,str) and kw in v: print(k,'| ADDED KEYWORD',kw)
