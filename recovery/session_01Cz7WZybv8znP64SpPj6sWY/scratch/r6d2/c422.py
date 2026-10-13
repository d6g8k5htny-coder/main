import json,re,sys,difflib
a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2]))
heads={"38116277597":"00d2ac6e","38121713840":"910809d6","38122386459":"d9b8dd33","38124262242":"79731b96","38149888699":"0dcb0016","38149524759":"88c21058","38150363063":"12e072d6","38180705952":"48bfcbae","38187538744":"be014794","38195810463":"c6ce0256"}
merges={"38116277597":"c4c66bb8","38121713840":"5b198699","38122386459":"866136cc","38124262242":"318ff194","38149888699":"4363b692","38149524759":"3306233b","38150363063":"86a6e8c8","38180705952":"9833d6d9","38187538744":"79d3964c","38195810463":"1320c5b2"}
def flat(x,p='',out=None):
    if out is None: out={}
    if isinstance(x,dict):
        for k,v in x.items(): flat(v,p+'/'+k,out)
    elif isinstance(x,list):
        for i,v in enumerate(x): flat(v,p+'[%d]'%i,out)
    elif isinstance(x,str): out[p]=x
    return out
fa=flat(a); fb=flat(b)
bad=0;tot=0
for p,y in fb.items():
    x=fa.get(p)
    if x==y: continue
    # inserted spans
    spans=[]
    if x is None: spans=[(0,len(y))]
    else:
        sm=difflib.SequenceMatcher(None,x,y,autojunk=False)
        for op,i1,i2,j1,j2 in sm.get_opcodes():
            if op in('insert','replace'): spans.append((j1,j2))
    for m in re.finditer(r'\b(3\d{10})\b',y):
        if not any(j1-5<=m.start()<=j2+5 for j1,j2 in spans): continue
        tot+=1
        r=m.group(1)
        # clause window: from previous ';' at paren depth 0 ... approximate with 350 chars before/after
        w=y[max(0,m.start()-350):m.end()+300]
        okh=heads[r] in w; okm=merges[r] in w and 'actual tested merge' in w
        status='OK' if okh and okm else 'MISS'
        if status=='MISS': bad+=1
        print(status,p,r,'head' if okh else 'NOHEAD','merge' if okm else 'NOMERGE')
print('total new run mentions',tot,'miss',bad)
# also look for any other run-like ids in new text not in table
