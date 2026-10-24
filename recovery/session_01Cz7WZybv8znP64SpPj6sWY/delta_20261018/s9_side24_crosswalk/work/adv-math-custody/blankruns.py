import sys,re,os,glob,collections
D=sys.argv[1]
for n in sys.argv[2:]:
    t=open(f'{D}/{n}.txt','rb').read().decode('utf-8-sig').replace('\r\n','\n')
    L=t.split('\n')
    b=[i for i,l in enumerate(L) if re.fullmatch(r'BEGIN_[A-Z0-9_]+_FROZEN_BODY',l)][0]
    tag=L[b][6:]; e=L.index('END_'+tag)
    body=L[b+1:e]
    runs=collections.Counter(); k=0
    for l in body:
        if l=='': k+=1
        else:
            if k: runs[k]+=1
            k=0
    ws=sum(1 for l in body if l!='' and l.strip()=='')
    feff=sum(l.count('﻿') for l in body)
    tw=sum(1 for l in body if l!=l.rstrip())
    print(n,'blank-run-length histogram',dict(sorted(runs.items())),'whitespace-only lines',ws,'inner BOM',feff,'trailing-ws lines',tw)
