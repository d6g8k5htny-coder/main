import sys,re,hashlib,itertools
raw=open(sys.argv[1],'rb').read().decode('utf-8-sig').replace('\r\n','\n')
L=raw.split('\n'); b=L.index('BEGIN_LSREQ043_FROZEN_BODY'); e=L.index('END_LSREQ043_FROZEN_BODY')
T='71754d660866721c037743579eaf2f7df282262ad29491faae912e8e587029fe'
for mode in ('strict','collapse'):
    body='\n'.join(L[b+1:e]).rstrip('\n')+'\n'
    if mode=='collapse': body=re.sub(r'\n\n\n+','\n\n',body)
    ls=body.split('\n')
    i0=ls.index('REVIEW OBJECT'); i1=ls.index('MANDATORY CONTEXT')
    idx=list(range(i0+1,i1))
    found=0; near=[]
    for r in range(0,len(idx)+1):
        for sub in itertools.combinations(idx,r):
            s=set(sub)
            v='\n'.join(l for j,l in enumerate(ls) if j not in s)
            bb=v.encode()
            if len(bb)==3056:
                near.append(sub)
                if hashlib.sha256(bb).hexdigest()==T: print(mode,'MATCH removing', sub); found=1
    print(mode,'block lines',len(idx),'subsets hitting 3056 B:',len(near),'match:',found)
