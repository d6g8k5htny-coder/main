import sys, re, hashlib, os
D=sys.argv[1]
def body(n):
    t=open(os.path.join(D,n+'.txt'),'rb').read().decode('utf-8-sig').replace('\r\n','\n').replace('\r','\n')
    L=t.split('\n')
    bi=[i for i,l in enumerate(L) if re.fullmatch(r'BEGIN_[A-Z0-9_]+_FROZEN_BODY',l)][0]
    tag=L[bi][6:]; ei=L.index('END_'+tag)
    return '\n'.join(L[bi+1:ei]).rstrip('\n')+'\n', L, bi, ei
for n in sys.argv[2:]:
    b,L,bi,ei=body(n)
    h=hashlib.sha256(b.encode()).hexdigest()
    runs={}
    for m in re.finditer(r'\n(\n+)', b):
        k=len(m.group(1)); runs[k]=runs.get(k,0)+1
    # whitespace-only lines (not empty) count
    wsonly=sum(1 for l in b.split('\n') if l!='' and l.strip()=='')
    # leading blank lines after BEGIN
    lead=0
    for l in L[bi+1:]:
        if l=='': lead+=1
        else: break
    print(n, 'strict_sha', h, 'blank_run_hist', dict(sorted(runs.items())), 'ws_only_lines', wsonly, 'leading_blank', lead)
